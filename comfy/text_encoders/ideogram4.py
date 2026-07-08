"""Ideogram 4 text encoder: Qwen3-VL-8B language model, 13-layer tap.

Ideogram 4 conditions on the concatenation of hidden states from 13 layers of
Qwen3-VL (layers 0,3,...,33,35), giving a 4096*13 = 53248-dim feature per token.
"""

import os

from transformers import Qwen2Tokenizer

import comfy.text_encoders.llama
import comfy.text_encoders.qwen3vl
from comfy import sd1_clip

# Reference taps outputs of layers (0,3,...,35); comfy captures layer inputs, offset by +1.
IDEOGRAM4_TAP_LAYERS = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 36]


class Qwen3VLTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "qwen25_tokenizer")
        super().__init__(tokenizer_path, pad_with_end=False, embedding_directory=embedding_directory,
                         embedding_size=4096, embedding_key='qwen3vl_8b', tokenizer_class=Qwen2Tokenizer,
                         has_start_token=False, has_end_token=False, pad_to_max_length=False,
                         max_length=99999999, min_length=1, pad_token=151643, tokenizer_data=tokenizer_data)


class Ideogram4Tokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data,
                         name="qwen3vl_8b", tokenizer=Qwen3VLTokenizer)

        self.llama_template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

    def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None, **kwargs):
        if text.startswith('<|im_start|>'):
            llama_text = text
        elif llama_template is None:
            llama_text = self.llama_template.format(text)
        else:
            llama_text = llama_template.format(text)
        return super().tokenize_with_weights(llama_text, return_word_ids=return_word_ids, disable_weights=True, **kwargs)


# Qwen3-VL-8B = 5e6 (vs plain Qwen3-8B's 1e6)
# final_norm/lm_head off -> Ideogram only reads raw tapped hidden states
QWEN3VL_8B_CONFIG = {"rope_theta": 5000000.0, "final_norm": False, "lm_head": False}


class Qwen3VL8BModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="hidden", layer_idx=None, dtype=None, attention_mask=True, model_options={}):
        super().__init__(device=device, layer=IDEOGRAM4_TAP_LAYERS, layer_idx=None,
                         textmodel_json_config=dict(QWEN3VL_8B_CONFIG),
                         dtype=dtype, special_tokens={"pad": 151643}, layer_norm_hidden_state=False,
                         model_class=comfy.text_encoders.llama.Qwen3_8B,
                         enable_attention_masks=attention_mask, return_attention_masks=attention_mask,
                         model_options=model_options)


class Ideogram4TEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="qwen3vl_8b", clip_model=Qwen3VL8BModel, model_options=model_options)

    def encode_token_weights(self, token_weight_pairs):
        out, pooled, extra = super().encode_token_weights(token_weight_pairs)
        b, n, seq, h = out.shape # (B, n_taps=13, seq, 4096) stacked in ascending layer order.
        out = out.permute(0, 2, 3, 1).reshape(b, seq, h * n) # (B, seq, 4096*13). permute -> (B, seq, H, taps).
        return out, pooled, extra


def te(dtype_llama=None, llama_quantization_metadata=None):
    class Ideogram4TEModel_(Ideogram4TEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_llama is not None:
                dtype = dtype_llama
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return Ideogram4TEModel_


# Full Qwen3-VL-8B variant with vision

class Ideogram4Qwen3VLClipModel(comfy.text_encoders.qwen3vl.Qwen3VLClipModel):
    def __init__(self, device="cpu", dtype=None, attention_mask=True, model_options={}):
        super().__init__(device=device, layer=IDEOGRAM4_TAP_LAYERS, layer_idx=None, dtype=dtype,
                         attention_mask=attention_mask, model_options=model_options, model_type="qwen3vl_8b")


class Ideogram4Qwen3VLTEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="qwen3vl_8b", clip_model=Ideogram4Qwen3VLClipModel, model_options=model_options)

    def encode_token_weights(self, token_weight_pairs):
        out, pooled, extra = super().encode_token_weights(token_weight_pairs)
        b, n, seq, h = out.shape  # (B, n_taps=13, seq, 4096), ascending layer order.
        out = out.permute(0, 2, 3, 1).reshape(b, seq, h * n)  # (B, seq, 4096*13 = 53248).
        return out, pooled, extra


class Ideogram4Qwen3VLTokenizer(comfy.text_encoders.qwen3vl.Qwen3VLTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, model_type="qwen3vl_8b")

    def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None, images=[], prevent_empty_text=False, thinking=True, **kwargs):
        # Ideogram 4 conditions on the no-think template; default thinking=True drops the empty think block qwen3vl adds.
        return super().tokenize_with_weights(text, return_word_ids=return_word_ids, llama_template=llama_template, images=images, prevent_empty_text=prevent_empty_text, thinking=thinking, **kwargs)


def te_qwen3vl(dtype_llama=None, llama_quantization_metadata=None):
    class Ideogram4Qwen3VLTEModel_(Ideogram4Qwen3VLTEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_llama is not None:
                dtype = dtype_llama
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return Ideogram4Qwen3VLTEModel_
