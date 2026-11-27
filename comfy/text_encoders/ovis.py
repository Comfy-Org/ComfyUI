from transformers import Qwen2Tokenizer
import comfy.text_encoders.llama
from comfy import sd1_clip
import os
import torch
import numbers

class Qwen3Tokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "qwen25_tokenizer")
        super().__init__(tokenizer_path, pad_with_end=False, embedding_size=2048, embedding_key='qwen3_2b', tokenizer_class=Qwen2Tokenizer, has_start_token=False, has_end_token=False, pad_to_max_length=False, max_length=99999999, min_length=284, pad_token=151643, tokenizer_data=tokenizer_data)


class OvisTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name="qwen3_2b", tokenizer=Qwen3Tokenizer)
        self.llama_template = "<|im_start|>user\nDescribe the image by detailing the color, quantity, text, shape, size, texture, spatial relationships of the objects and background: {}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None, thinking=None, system_prompt="", skip_template=False, **kwargs):
        if thinking is not None and llama_template is None:  # TextGenerate (it always passes thinking): plain Qwen3 chat, not the image description prompt
            llama_text = text
            if not skip_template:  # hybrid Qwen3 chat template: the model opens its own think block, an empty one disables reasoning
                llama_text = "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n" + ("" if thinking else "<think>\n\n</think>\n\n")
                if system_prompt:
                    llama_text = "<|im_start|>system\n" + system_prompt + "<|im_end|>\n" + llama_text
        elif llama_template is None:
            llama_text = self.llama_template.format(text)
        else:
            llama_text = llama_template.format(text)

        tokens = super().tokenize_with_weights(llama_text, return_word_ids=return_word_ids, disable_weights=True, **kwargs)
        return tokens

class Ovis25_2BModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, attention_mask=True, model_options={}):
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config={}, dtype=dtype, special_tokens={"pad": 151643}, layer_norm_hidden_state=False, model_class=comfy.text_encoders.llama.Ovis25_2B, enable_attention_masks=attention_mask, return_attention_masks=False, zero_out_masked=True, model_options=model_options)


class OvisTEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="qwen3_2b", clip_model=Ovis25_2BModel, model_options=model_options)

    def encode_token_weights(self, token_weight_pairs, template_end=-1):
        out, pooled = super().encode_token_weights(token_weight_pairs)
        tok_pairs = token_weight_pairs["qwen3_2b"][0]
        count_im_start = 0
        if template_end == -1:
            for i, v in enumerate(tok_pairs):
                elem = v[0]
                if not torch.is_tensor(elem):
                    if isinstance(elem, numbers.Integral):
                        if elem == 4004 and count_im_start < 1:
                            template_end = i
                            count_im_start += 1

            if out.shape[1] > (template_end + 1):
                if tok_pairs[template_end + 1][0] == 25:
                    template_end += 1

        out = out[:, template_end:]
        return out, pooled, {}


def te(dtype_llama=None, llama_quantization_metadata=None):
    class OvisTEModel_(OvisTEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_llama is not None:
                dtype = dtype_llama
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return OvisTEModel_
