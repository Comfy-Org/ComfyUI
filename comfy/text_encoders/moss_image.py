import comfy.text_encoders.qwen35
from comfy import sd1_clip


class MOSSImageTokenizer(comfy.text_encoders.qwen35.Qwen35ImageTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, model_type="qwen35_9b")

    def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None, thinking=None, **kwargs):
        if llama_template is None and thinking is None and not text.startswith('<|im_start|>'):
            # conditioning: the chat template with enable_thinking=True trims the user turn and leaves an open think block
            text, llama_template, thinking = text.strip(), "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n", True
        return super().tokenize_with_weights(text, return_word_ids=return_word_ids, llama_template=llama_template, thinking=bool(thinking), **kwargs)


class MOSSImageTEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}, mtp=False):
        clip_model = lambda **kw: comfy.text_encoders.qwen35.Qwen35ClipModel(**kw, layer="last", layer_idx=None, model_type="qwen35_9b", mtp=mtp)
        super().__init__(device=device, dtype=dtype, name="qwen35_9b", clip_model=clip_model, model_options=model_options)


def te(dtype_llama=None, llama_quantization_metadata=None, mtp=False):
    class MOSSImageTEModel_(MOSSImageTEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_llama is not None:
                dtype = dtype_llama
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            super().__init__(device=device, dtype=dtype, model_options=model_options, mtp=mtp)
    return MOSSImageTEModel_
