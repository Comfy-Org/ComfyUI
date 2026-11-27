from transformers import Qwen2Tokenizer
from comfy import model_management, sd1_clip
import comfy.text_encoders.llama
import math
import os
import torch
import numbers

def _estimate_qwen_image_memory(sections, config, vision_config, device=None, embedding_dtype=None):
    if not sections or not any(sections):
        return 0

    lengths = []
    weighted = False
    image_tokens = 0
    vision_peak = 0
    patch = vision_config["patch_size"]
    factor = patch * vision_config["spatial_merge_size"]
    vision_hidden = vision_config["hidden_size"]
    deepstack = len(vision_config.get("deepstack_visual_indexes", []))
    for section in sections:
        length = 0
        for token, weight, *_ in section:
            weighted |= weight != 1.0
            if torch.is_tensor(token):
                length += math.prod(token.shape[:-1])
            elif isinstance(token, dict) and token.get("type") == "embedding":
                length += math.prod(token["data"].shape[:-1])
            elif isinstance(token, dict) and token.get("type") == "image":
                # Match process_qwen2vl_images, which processes only data[0].
                height, width = token["data"].shape[1:3]
                h = round(height / factor) * factor
                w = round(width / factor) * factor
                if h * w > 12845056:
                    scale = math.sqrt(height * width / 12845056)
                    h = max(factor, math.floor(height / scale / factor) * factor)
                    w = max(factor, math.floor(width / scale / factor) * factor)
                elif h * w < 3136:
                    scale = math.sqrt(3136 / (height * width))
                    h = math.ceil(height * scale / factor) * factor
                    w = math.ceil(width * scale / factor) * factor
                merged = (h // factor) * (w // factor)
                patches = (h // patch) * (w // patch)
                length += merged
                image_tokens += merged
                # Global visual attention runs before spatial merging, also in Qwen2.5's full-attention blocks.
                vision = 4 * patches * (16 * vision_hidden + 3 * vision_config["intermediate_size"] + 24 * patch * patch)
                vision += 16 * merged * config.hidden_size
                vision += 8 * vision_config["num_heads"] * patches * patches
                vision += 4 * vision_hidden * max(3 * vision_hidden, vision_config["intermediate_size"], 4 * config.hidden_size)
                vision_peak = max(vision_peak, vision)
            else:
                length += 1
        lengths.append(length)

    batch = len(sections) + int(weighted)
    seq = max(max(lengths), max(map(len, sections)))
    hidden = config.hidden_size
    query = config.num_attention_heads * config.head_dim
    kv = config.num_key_value_heads * config.head_dim
    # SDClipModel supplies FP32 embeddings even with quantized weights.
    retained = 2 * batch * seq * hidden * 4
    if device is not None and model_management.intermediate_device() == torch.device(device):
        retained += 2 * batch * seq * hidden * 4
    image_features = image_tokens * hidden * (1 + 2 * deepstack) * 4
    activations = batch * seq * (6 * hidden + max(3 * config.intermediate_size, 4 * query + 4 * kv)) * 4
    masks = max(9 * batch, 4 * (batch + 2)) * seq * seq + 8 * batch * seq
    # Conservative score/softmax workspace without inspecting the selected attention backend.
    attention = 2 * batch * config.num_attention_heads * seq * seq * 4
    cast_weight = hidden * max(config.intermediate_size, query) * 4
    language_peak = retained + image_features + activations + masks + attention + cast_weight
    # Raw FP8 Embedding casts the entire vocabulary before looking up FP32 rows.
    embedding_peak = 0
    if embedding_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        embedding_peak = retained + image_features + config.vocab_size * hidden * 4
    return max(language_peak, retained + image_features + vision_peak, embedding_peak)


class Qwen25_7BVLITokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "qwen25_tokenizer")
        super().__init__(tokenizer_path, pad_with_end=False, embedding_size=3584, embedding_key='qwen25_7b', tokenizer_class=Qwen2Tokenizer, has_start_token=False, has_end_token=False, pad_to_max_length=False, max_length=99999999, min_length=1, pad_token=151643, tokenizer_data=tokenizer_data)


class QwenImageTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name="qwen25_7b", tokenizer=Qwen25_7BVLITokenizer)
        self.llama_template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.llama_template_images = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"

    def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None, images=[], prevent_empty_text=False, **kwargs):
        skip_template = False
        if text.startswith('<|im_start|>'):
            skip_template = True
        if text.startswith('<|start_header_id|>'):
            skip_template = True
        if prevent_empty_text and text == '':
            text = ' '

        if skip_template:
            llama_text = text
        else:
            if llama_template is None:
                if len(images) > 0:
                    llama_text = self.llama_template_images.format(text)
                else:
                    llama_text = self.llama_template.format(text)
            else:
                llama_text = llama_template.format(text)
        tokens = super().tokenize_with_weights(llama_text, return_word_ids=return_word_ids, disable_weights=True, **kwargs)
        key_name = next(iter(tokens))
        embed_count = 0
        qwen_tokens = tokens[key_name]
        for r in qwen_tokens:
            for i in range(len(r)):
                if r[i][0] == 151655:
                    if len(images) > embed_count:
                        r[i] = ({"type": "image", "data": images[embed_count], "original_type": "image"},) + r[i][1:]
                        embed_count += 1
        return tokens


class Qwen25_7BVLIModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, attention_mask=True, model_options={}):
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config={}, dtype=dtype, special_tokens={"pad": 151643}, layer_norm_hidden_state=False, model_class=comfy.text_encoders.llama.Qwen25_7BVLI, enable_attention_masks=attention_mask, return_attention_masks=attention_mask, model_options=model_options)


class QwenImageTEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="qwen25_7b", clip_model=Qwen25_7BVLIModel, model_options=model_options)

    def encode_token_weights(self, token_weight_pairs, template_end=-1):
        out, pooled, extra = super().encode_token_weights(token_weight_pairs)
        tok_pairs = token_weight_pairs["qwen25_7b"][0]
        count_im_start = 0
        if template_end == -1:
            for i, v in enumerate(tok_pairs):
                elem = v[0]
                if not torch.is_tensor(elem):
                    if isinstance(elem, numbers.Integral):
                        if elem == 151644 and count_im_start < 2:
                            template_end = i
                            count_im_start += 1

            if out.shape[1] > (template_end + 3):
                if tok_pairs[template_end + 1][0] == 872:
                    if tok_pairs[template_end + 2][0] == 198:
                        template_end += 3

        out = out[:, template_end:]

        extra["attention_mask"] = extra["attention_mask"][:, template_end:]
        if extra["attention_mask"].sum() == torch.numel(extra["attention_mask"]):
            extra.pop("attention_mask")  # attention mask is useless if no masked elements

        return out, pooled, extra


def te(dtype_llama=None, llama_quantization_metadata=None):
    class QwenImageTEModel_(QwenImageTEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            if dtype_llama is not None:
                dtype = dtype_llama
            super().__init__(device=device, dtype=dtype, model_options=model_options)
            # CLIP uses this flag to select the non-dynamic patcher and reserve encode workspace.
            self.disable_offload = True

        def memory_estimation_function(self, token_weight_pairs, device=None):
            model = getattr(self, self.clip).transformer.model
            vision_config = dict(patch_size=14, spatial_merge_size=2, hidden_size=1280, intermediate_size=3420, num_heads=16)
            return _estimate_qwen_image_memory(token_weight_pairs.get(self.clip_name, []), model.config, vision_config, device, model.embed_tokens.weight.dtype)

    return QwenImageTEModel_
