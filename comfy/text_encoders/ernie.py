from .flux import Mistral3Tokenizer
from comfy import sd1_clip
import comfy.text_encoders.llama

class Ministral3_3BTokenizer(Mistral3Tokenizer):
    def __init__(self, embedding_directory=None, embedding_size=5120, embedding_key='ministral3_3b', tokenizer_data={}):
        return super().__init__(embedding_directory=embedding_directory, embedding_size=embedding_size, embedding_key=embedding_key, tokenizer_data=tokenizer_data)

class ErnieTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name="ministral3_3b", tokenizer=Mistral3Tokenizer)

    def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None, **kwargs):
        tokens = super().tokenize_with_weights(text, return_word_ids=return_word_ids, disable_weights=True, **kwargs)
        return tokens


class Ministral3_3BModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="hidden", layer_idx=-2, dtype=None, attention_mask=True, model_options={}):
        textmodel_json_config = {}
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=textmodel_json_config, dtype=dtype, special_tokens={"start": 1, "pad": 0}, layer_norm_hidden_state=False, model_class=comfy.text_encoders.llama.Ministral3_3B, enable_attention_masks=attention_mask, return_attention_masks=attention_mask, model_options=model_options)


class ErnieTEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}, name="ministral3_3b", clip_model=Ministral3_3BModel):
        super().__init__(device=device, dtype=dtype, name=name, clip_model=clip_model, model_options=model_options)


def te(dtype_llama=None, llama_quantization_metadata=None):
    class ErnieTEModel_(ErnieTEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_llama is not None:
                dtype = dtype_llama
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return ErnieTEModel_
