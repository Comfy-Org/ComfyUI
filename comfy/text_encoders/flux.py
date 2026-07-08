from comfy import sd1_clip
import comfy.text_encoders.t5
import comfy.text_encoders.sd3_clip
import comfy.text_encoders.llama
import comfy.model_management
from transformers import T5TokenizerFast, LlamaTokenizerFast, Qwen2Tokenizer
import torch
import os
import json
import base64

class T5XXLTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "t5_tokenizer")
        super().__init__(tokenizer_path, embedding_directory=embedding_directory, pad_with_end=False, embedding_size=4096, embedding_key='t5xxl', tokenizer_class=T5TokenizerFast, has_start_token=False, pad_to_max_length=False, max_length=99999999, min_length=256, tokenizer_data=tokenizer_data)


class FluxTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.clip_l = sd1_clip.SDTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        self.t5xxl = T5XXLTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)

    def tokenize_with_weights(self, text:str, return_word_ids=False, **kwargs):
        out = {}
        out["l"] = self.clip_l.tokenize_with_weights(text, return_word_ids, **kwargs)
        out["t5xxl"] = self.t5xxl.tokenize_with_weights(text, return_word_ids, **kwargs)
        return out

    def untokenize(self, token_weight_pair):
        return self.clip_l.untokenize(token_weight_pair)

    def state_dict(self):
        return {}


class FluxClipModel(torch.nn.Module):
    def __init__(self, dtype_t5=None, device="cpu", dtype=None, model_options={}):
        super().__init__()
        dtype_t5 = comfy.model_management.pick_weight_dtype(dtype_t5, dtype, device)
        self.clip_l = sd1_clip.SDClipModel(device=device, dtype=dtype, return_projected_pooled=False, model_options=model_options)
        self.t5xxl = comfy.text_encoders.sd3_clip.T5XXLModel(device=device, dtype=dtype_t5, model_options=model_options)
        self.dtypes = set([dtype, dtype_t5])

    def set_clip_options(self, options):
        self.clip_l.set_clip_options(options)
        self.t5xxl.set_clip_options(options)

    def reset_clip_options(self):
        self.clip_l.reset_clip_options()
        self.t5xxl.reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs_l = token_weight_pairs["l"]
        token_weight_pairs_t5 = token_weight_pairs["t5xxl"]

        t5_out, t5_pooled = self.t5xxl.encode_token_weights(token_weight_pairs_t5)
        l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l)
        return t5_out, l_pooled

    def load_sd(self, sd):
        if "text_model.encoder.layers.1.mlp.fc1.weight" in sd:
            return self.clip_l.load_sd(sd)
        else:
            return self.t5xxl.load_sd(sd)

def flux_clip(dtype_t5=None, t5_quantization_metadata=None):
    class FluxClipModel_(FluxClipModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if t5_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["t5xxl_quantization_metadata"] = t5_quantization_metadata
            super().__init__(dtype_t5=dtype_t5, device=device, dtype=dtype, model_options=model_options)
    return FluxClipModel_

def load_mistral_tokenizer(data):
    if torch.is_tensor(data):
        data = data.numpy().tobytes()

    try:
        from transformers.integrations.mistral import MistralConverter
    except ModuleNotFoundError:
        from transformers.models.pixtral.convert_pixtral_weights_to_hf import MistralConverter

    mistral_vocab = json.loads(data)

    special_tokens = {}
    vocab = {}

    max_vocab = mistral_vocab["config"]["default_vocab_size"]
    max_vocab -= len(mistral_vocab["special_tokens"])

    for w in mistral_vocab["vocab"]:
        r = w["rank"]
        if r >= max_vocab:
            continue

        vocab[base64.b64decode(w["token_bytes"])] = r

    for w in mistral_vocab["special_tokens"]:
        if "token_bytes" in w:
            special_tokens[base64.b64decode(w["token_bytes"])] = w["rank"]
        else:
            special_tokens[w["token_str"]] = w["rank"]

    all_special = []
    for v in special_tokens:
        all_special.append(v)

    special_tokens.update(vocab)
    vocab = special_tokens
    return {"tokenizer_object": MistralConverter(vocab=vocab, additional_special_tokens=all_special).converted(), "legacy": False}

class MistralTokenizerClass:
    @staticmethod
    def from_pretrained(path, **kwargs):
        return LlamaTokenizerFast(**kwargs)

class Mistral3Tokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, embedding_size=5120, embedding_key='mistral3_24b', tokenizer_data={}):
        self.tekken_data = tokenizer_data.get("tekken_model", None)
        super().__init__("", pad_with_end=False, embedding_directory=embedding_directory, embedding_size=embedding_size, embedding_key=embedding_key, tokenizer_class=MistralTokenizerClass, has_end_token=False, pad_to_max_length=False, pad_token=11, start_token=1, max_length=99999999, min_length=1, pad_left=True, disable_weights=True, tokenizer_args=load_mistral_tokenizer(self.tekken_data), tokenizer_data=tokenizer_data)

    def state_dict(self):
        return {"tekken_model": self.tekken_data}

class Flux2Tokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name="mistral3_24b", tokenizer=Mistral3Tokenizer)
        self.llama_template = '[SYSTEM_PROMPT]You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object\nattribution and actions without speculation.[/SYSTEM_PROMPT][INST]{}[/INST]'

    def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None, **kwargs):
        if llama_template is None:
            llama_text = self.llama_template.format(text)
        else:
            llama_text = llama_template.format(text)

        tokens = super().tokenize_with_weights(llama_text, return_word_ids=return_word_ids, disable_weights=True, **kwargs)
        return tokens

class Mistral3_24BModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer=[10, 20, 30], layer_idx=None, dtype=None, attention_mask=True, model_options={}):
        textmodel_json_config = {}
        num_layers = model_options.get("num_layers", None)
        if num_layers is not None:
            textmodel_json_config["num_hidden_layers"] = num_layers
            if num_layers < 40:
                textmodel_json_config["final_norm"] = False
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=textmodel_json_config, dtype=dtype, special_tokens={"start": 1, "pad": 0}, layer_norm_hidden_state=False, model_class=comfy.text_encoders.llama.Mistral3Small24B, enable_attention_masks=attention_mask, return_attention_masks=attention_mask, model_options=model_options)

class Flux2TEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}, name="mistral3_24b", clip_model=Mistral3_24BModel):
        super().__init__(device=device, dtype=dtype, name=name, clip_model=clip_model, model_options=model_options)

    def encode_token_weights(self, token_weight_pairs):
        out, pooled, extra = super().encode_token_weights(token_weight_pairs)

        out = torch.stack((out[:, 0], out[:, 1], out[:, 2]), dim=1)
        out = out.movedim(1, 2)
        out = out.reshape(out.shape[0], out.shape[1], -1)
        return out, pooled, extra

def flux2_te(dtype_llama=None, llama_quantization_metadata=None, pruned=False):
    class Flux2TEModel_(Flux2TEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_llama is not None:
                dtype = dtype_llama
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            if pruned:
                model_options = model_options.copy()
                model_options["num_layers"] = 30
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return Flux2TEModel_

class Qwen3Tokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "qwen25_tokenizer")
        super().__init__(tokenizer_path, pad_with_end=False, embedding_directory=embedding_directory, embedding_size=2560, embedding_key='qwen3_4b', tokenizer_class=Qwen2Tokenizer, has_start_token=False, has_end_token=False, pad_to_max_length=False, max_length=99999999, min_length=512, pad_token=151643, tokenizer_data=tokenizer_data)

class Qwen3Tokenizer8B(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "qwen25_tokenizer")
        super().__init__(tokenizer_path, pad_with_end=False, embedding_directory=embedding_directory, embedding_size=4096, embedding_key='qwen3_8b', tokenizer_class=Qwen2Tokenizer, has_start_token=False, has_end_token=False, pad_to_max_length=False, max_length=99999999, min_length=512, pad_token=151643, tokenizer_data=tokenizer_data)

class KleinTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}, name="qwen3_4b"):
        if name == "qwen3_4b":
            tokenizer = Qwen3Tokenizer
        elif name == "qwen3_8b":
            tokenizer = Qwen3Tokenizer8B

        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name=name, tokenizer=tokenizer)
        self.llama_template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None, **kwargs):
        if llama_template is None:
            llama_text = self.llama_template.format(text)
        else:
            llama_text = llama_template.format(text)

        tokens = super().tokenize_with_weights(llama_text, return_word_ids=return_word_ids, disable_weights=True, **kwargs)
        return tokens

class KleinTokenizer8B(KleinTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}, name="qwen3_8b"):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name=name)

class Qwen3_4BModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer=[9, 18, 27], layer_idx=None, dtype=None, attention_mask=True, model_options={}):
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config={}, dtype=dtype, special_tokens={"pad": 151643}, layer_norm_hidden_state=False, model_class=comfy.text_encoders.llama.Qwen3_4B, enable_attention_masks=attention_mask, return_attention_masks=attention_mask, model_options=model_options)

class Qwen3_8BModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer=[9, 18, 27], layer_idx=None, dtype=None, attention_mask=True, model_options={}):
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config={}, dtype=dtype, special_tokens={"pad": 151643}, layer_norm_hidden_state=False, model_class=comfy.text_encoders.llama.Qwen3_8B, enable_attention_masks=attention_mask, return_attention_masks=attention_mask, model_options=model_options)

def klein_te(dtype_llama=None, llama_quantization_metadata=None, model_type="qwen3_4b"):
    if model_type == "qwen3_4b":
        model = Qwen3_4BModel
    elif model_type == "qwen3_8b":
        model = Qwen3_8BModel

    class Flux2TEModel_(Flux2TEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            if dtype_llama is not None:
                dtype = dtype_llama
            super().__init__(device=device, dtype=dtype, name=model_type, model_options=model_options, clip_model=model)
    return Flux2TEModel_
