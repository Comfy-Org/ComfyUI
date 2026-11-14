import comfy.text_encoders.qwen35
from comfy import sd1_clip


# Qwen3.5-9B chat template with the PAN 2 system turn, the open think block of the default generation prompt
PAN2_TEMPLATE = "<|im_start|>system\nYou are a helpful assistant. Generate the video with the following instructions.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n"


class PAN2Tokenizer(comfy.text_encoders.qwen35.Qwen35ImageTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, model_type="qwen35_9b")

    def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None, thinking=None, **kwargs):
        if llama_template is None and thinking is None and not text.startswith('<|im_start|>'):
            # conditioning: the chat template trims the caption
            text, llama_template = text.strip(), PAN2_TEMPLATE
        return super().tokenize_with_weights(text, return_word_ids=return_word_ids, llama_template=llama_template, thinking=bool(thinking), **kwargs)


class PAN2TEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}, mtp=False):
        # output of layer 30 of 32, without the final norm
        clip_model = lambda **kw: comfy.text_encoders.qwen35.Qwen35ClipModel(**kw, layer="hidden", layer_idx=-3, model_type="qwen35_9b", mtp=mtp)
        super().__init__(device=device, dtype=dtype, name="qwen35_9b", clip_model=clip_model, model_options=model_options)

    def encode_token_weights(self, token_weight_pairs):
        out, pooled, extra = super().encode_token_weights(token_weight_pairs)
        # the DiT sees the tokens after <|im_start|>user\n: the caption and the rest of the template
        tokens = [t[0] for t in token_weight_pairs["qwen35_9b"][0]]
        start = next(i + 3 for i in range(len(tokens) - 2) if tokens[i:i + 3] == [248045, 846, 198])
        extra.pop("attention_mask", None)
        return out[:, start:], pooled, extra


def te(dtype_llama=None, llama_quantization_metadata=None, mtp=False):
    class PAN2TEModel_(PAN2TEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_llama is not None:
                dtype = dtype_llama
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            super().__init__(device=device, dtype=dtype, model_options=model_options, mtp=mtp)
    return PAN2TEModel_
