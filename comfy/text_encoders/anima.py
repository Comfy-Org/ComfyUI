from transformers import Qwen2Tokenizer, T5TokenizerFast
import comfy.text_encoders.llama
from comfy import sd1_clip
import os
import torch


class Qwen3Tokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "qwen25_tokenizer")
        super().__init__(tokenizer_path, pad_with_end=False, embedding_directory=embedding_directory, embedding_size=1024, embedding_key='qwen3_06b', tokenizer_class=Qwen2Tokenizer, has_start_token=False, has_end_token=False, pad_to_max_length=False, max_length=99999999, min_length=1, pad_token=151643, tokenizer_data=tokenizer_data)

class T5XXLTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "t5_tokenizer")
        super().__init__(tokenizer_path, embedding_directory=embedding_directory, pad_with_end=False, embedding_size=4096, embedding_key='t5xxl', tokenizer_class=T5TokenizerFast, has_start_token=False, pad_to_max_length=False, max_length=99999999, min_length=1, tokenizer_data=tokenizer_data)

class AnimaTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.qwen3_06b = Qwen3Tokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        self.t5xxl = T5XXLTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)

    def tokenize_with_weights(self, text:str, return_word_ids=False, **kwargs):
        out = {}
        qwen_ids = self.qwen3_06b.tokenize_with_weights(text, return_word_ids, **kwargs)
        out["qwen3_06b"] = [[(k[0], 1.0, k[2]) if return_word_ids else (k[0], 1.0) for k in inner_list] for inner_list in qwen_ids]  # Set weights to 1.0
        out["t5xxl"] = self.t5xxl.tokenize_with_weights(text, return_word_ids, **kwargs)
        return out

    def untokenize(self, token_weight_pair):
        return self.t5xxl.untokenize(token_weight_pair)

    def state_dict(self):
        return {}

    def decode(self, token_ids, **kwargs):
        return self.qwen3_06b.decode(token_ids, **kwargs)

class Qwen3_06BModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, attention_mask=True, model_options={}):
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config={}, dtype=dtype, special_tokens={"pad": 151643}, layer_norm_hidden_state=False, model_class=comfy.text_encoders.llama.Qwen3_06B, enable_attention_masks=attention_mask, return_attention_masks=attention_mask, model_options=model_options)


class AnimaTEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="qwen3_06b", clip_model=Qwen3_06BModel, model_options=model_options)

    def encode_token_weights(self, token_weight_pairs):
        out = super().encode_token_weights(token_weight_pairs)
        t5_tokens = token_weight_pairs["t5xxl"][0]
        # embedding: entries carry tensors instead of token ids; substitute the t5 pad token
        out[2]["t5xxl_ids"] = torch.tensor(list(map(lambda a: 0 if torch.is_tensor(a[0]) else a[0], t5_tokens)), dtype=torch.int)
        out[2]["t5xxl_weights"] = torch.tensor(list(map(lambda a: a[1], t5_tokens)))
        return out

def te(dtype_llama=None, llama_quantization_metadata=None):
    class AnimaTEModel_(AnimaTEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_llama is not None:
                dtype = dtype_llama
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return AnimaTEModel_
