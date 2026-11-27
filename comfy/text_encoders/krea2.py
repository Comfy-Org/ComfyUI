"""Krea 2 (K2) text encoder: Qwen3-VL-4B, 12-layer tap.

K2 conditions on a stack of hidden states from 12 layers of Qwen3-VL-4B
(reference taps ``hidden_states[2,5,8,...,35]``), kept as a ``(B, 12, seq, 2560)`` tensor and
consumed by the DiT's internal ``txtfusion`` adapter. Comfy carries conditioning as a 3D tensor,
so the 12-layer stack is flattened to ``(B, seq, 12*2560)`` here and unpacked inside the model.
"""

import math
import numbers

import torch

import comfy.text_encoders.qwen3vl
from comfy import model_management, sd1_clip

# tap k == hidden_states[k] (no offset).
KREA2_TAP_LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]

# Identical system template to Qwen-Image; Krea2 strips the system+user-opening prefix.
KREA2_TEMPLATE = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


class Krea2Tokenizer(comfy.text_encoders.qwen3vl.Qwen3VLTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, model_type="qwen3vl_4b")
        self.llama_template = KREA2_TEMPLATE  # conditioning template; image text-gen uses qwen3vl's default image template.


class Krea2Qwen3VLClipModel(comfy.text_encoders.qwen3vl.Qwen3VLClipModel):
    def __init__(self, device="cpu", dtype=None, attention_mask=True, model_options={}):
        super().__init__(device=device, layer=KREA2_TAP_LAYERS, layer_idx=None, dtype=dtype,
                         attention_mask=attention_mask, model_options=model_options, model_type="qwen3vl_4b")


class Krea2TEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="qwen3vl_4b", clip_model=Krea2Qwen3VLClipModel, model_options=model_options)

    def memory_estimation_function(self, token_weight_pairs, device=None):
        sections = token_weight_pairs.get(self.clip_name, [])
        if not sections or not any(sections):
            return 0

        lengths = []
        weighted = False
        for section in sections:
            length = 0
            for token, weight, *_ in section:
                weighted |= weight != 1.0
                if torch.is_tensor(token):
                    length += math.prod(token.shape[:-1])
                elif isinstance(token, dict) and token.get("type") == "embedding":
                    length += math.prod(token["data"].shape[:-1])
                elif isinstance(token, dict) and token.get("type") == "image":
                    # Qwen3-VL uses 16px patches merged in 2x2 groups (qwen3vl.preprocess_embed).
                    height, width = token["data"].shape[1:3]
                    h = round(height / 32) * 32
                    w = round(width / 32) * 32
                    if h * w > 12845056:
                        scale = math.sqrt(height * width / 12845056)
                        h = max(32, math.floor(height / scale / 32) * 32)
                        w = max(32, math.floor(width / scale / 32) * 32)
                    elif h * w < 3136:
                        scale = math.sqrt(3136 / (height * width))
                        h = math.ceil(height * scale / 32) * 32
                        w = math.ceil(width * scale / 32) * 32
                    length += (h // 32) * (w // 32)
                else:
                    length += 1
            lengths.append(length)

        batch = len(sections) + int(weighted)  # ClipTokenWeightEncoder adds an empty section for weights.
        seq = max(max(lengths), max(map(len, sections)))
        language_model = getattr(self, self.clip).transformer.model
        config = language_model.config
        hidden = config.hidden_size
        query = config.num_attention_heads * config.head_dim
        kv = config.num_key_value_heads * config.head_dim
        # SDClipModel supplies FP32 embeds; Qwen3VL.forward keeps that dtype even with FP8 weights.
        taps = batch * seq * hidden * len(KREA2_TAP_LAYERS) * 4
        retained = 2 * taps  # Individual tap clones coexist with their concatenation.
        if device is not None and model_management.intermediate_device() == torch.device(device):
            retained += 2 * taps  # Output concatenation and Krea2's flattened conditioning.
        activations = batch * seq * (6 * hidden + max(3 * config.intermediate_size, 4 * query + 4 * kv)) * 4
        masks = max(9 * batch, 4 * (batch + 2)) * seq * seq + 8 * batch * seq
        # Conservative score/softmax workspace, including non-flash attention backends.
        attention = 2 * batch * config.num_attention_heads * seq * seq * 4
        cast_weight = hidden * max(config.intermediate_size, query) * 4
        # Language-model workspace only; image placeholders are expanded without running the visual encoder.
        language_peak = retained + activations + masks + attention + cast_weight
        if language_model.embed_tokens.weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            # Raw FP8 embeddings cast the full table; row-dequantized weights expose their compute dtype.
            embedding_peak = retained + config.vocab_size * hidden * 4
            return max(language_peak, embedding_peak)
        return language_peak

    def encode_token_weights(self, token_weight_pairs, template_end=-1):
        out, pooled, extra = super().encode_token_weights(token_weight_pairs)  # out: (B, 12, seq, 2560)
        tok_pairs = token_weight_pairs["qwen3vl_4b"][0]

        # Strip the system + user-opening prefix
        count_im_start = 0
        if template_end == -1:
            for i, v in enumerate(tok_pairs):
                elem = v[0]
                if not torch.is_tensor(elem) and isinstance(elem, numbers.Integral):
                    if elem == 151644 and count_im_start < 2:
                        template_end = i
                        count_im_start += 1
            if out.shape[2] > (template_end + 3):
                if tok_pairs[template_end + 1][0] == 872:      # "user"
                    if tok_pairs[template_end + 2][0] == 198:   # "\n"
                        template_end += 3

        out = out[:, :, template_end:]

        b, n, seq, h = out.shape
        # Flatten the 12-layer axis into the feature dim: (B, seq, 12*2560). Unpacked in the model.
        out = out.permute(0, 2, 1, 3).reshape(b, seq, n * h)

        if "attention_mask" in extra:
            extra["attention_mask"] = extra["attention_mask"][:, template_end:]
            if extra["attention_mask"].sum() == torch.numel(extra["attention_mask"]):
                extra.pop("attention_mask")

        return out, pooled, extra


def te(dtype_llama=None, llama_quantization_metadata=None):
    class Krea2TEModel_(Krea2TEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            if dtype_llama is not None:
                dtype = dtype_llama
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return Krea2TEModel_
