from . import hunyuan_video
from . import sd3_clip
from comfy import sd1_clip
from comfy import sdxl_clip
import comfy.model_management
import torch
import logging
import folder_paths


class HiDreamTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.clip_l = sd1_clip.SDTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        self.clip_g = sdxl_clip.SDXLClipGTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        self.t5xxl = sd3_clip.T5XXLTokenizer(embedding_directory=embedding_directory, min_length=128, max_length=128, tokenizer_data=tokenizer_data)
        self.llama = hunyuan_video.LLAMA3Tokenizer(embedding_directory=embedding_directory, min_length=128, pad_token=128009, tokenizer_data=tokenizer_data)

    def tokenize_with_weights(self, text:str, return_word_ids=False, **kwargs):
        out = {}
        out["g"] = self.clip_g.tokenize_with_weights(text, return_word_ids, **kwargs)
        out["l"] = self.clip_l.tokenize_with_weights(text, return_word_ids, **kwargs)
        t5xxl = self.t5xxl.tokenize_with_weights(text, return_word_ids, **kwargs)
        out["t5xxl"] = [t5xxl[0]]  # Use only first 128 tokens
        out["llama"] = self.llama.tokenize_with_weights(text, return_word_ids, **kwargs)
        return out

    def untokenize(self, token_weight_pair):
        return self.clip_g.untokenize(token_weight_pair)

    def state_dict(self):
        return {}


class HiDreamTEModel(torch.nn.Module):
    def __init__(self, clip_l=True, clip_g=True, t5=True, llama=True, dtype_t5=None, dtype_llama=None, device="cpu", dtype=None, model_options={}):
        super().__init__()
        self.dtypes = set()
        if clip_l:
            self.clip_l = sd1_clip.SDClipModel(device=device, dtype=dtype, return_projected_pooled=True, model_options=model_options)
            self.dtypes.add(dtype)
        else:
            self.clip_l = None

        if clip_g:
            self.clip_g = sdxl_clip.SDXLClipG(device=device, dtype=dtype, model_options=model_options)
            self.dtypes.add(dtype)
        else:
            self.clip_g = None

        if t5:
            dtype_t5 = comfy.model_management.pick_weight_dtype(dtype_t5, dtype, device)
            self.t5xxl = sd3_clip.T5XXLModel(device=device, dtype=dtype_t5, model_options=model_options, attention_mask=True)
            self.dtypes.add(dtype_t5)
        else:
            self.t5xxl = None

        if llama:
            dtype_llama = comfy.model_management.pick_weight_dtype(dtype_llama, dtype, device)
            if "vocab_size" not in model_options:
                model_options["vocab_size"] = 128256
            self.llama = hunyuan_video.LLAMAModel(device=device, dtype=dtype_llama, model_options=model_options, layer="all", layer_idx=None, special_tokens={"start": 128000, "pad": 128009})
            self.dtypes.add(dtype_llama)
        else:
            self.llama = None

        logging.debug("Created HiDream text encoder with: clip_l {}, clip_g {}, t5xxl {}:{}, llama {}:{}".format(clip_l, clip_g, t5, dtype_t5, llama, dtype_llama))

    def set_clip_options(self, options):
        if self.clip_l is not None:
            self.clip_l.set_clip_options(options)
        if self.clip_g is not None:
            self.clip_g.set_clip_options(options)
        if self.t5xxl is not None:
            self.t5xxl.set_clip_options(options)
        if self.llama is not None:
            self.llama.set_clip_options(options)

    def reset_clip_options(self):
        if self.clip_l is not None:
            self.clip_l.reset_clip_options()
        if self.clip_g is not None:
            self.clip_g.reset_clip_options()
        if self.t5xxl is not None:
            self.t5xxl.reset_clip_options()
        if self.llama is not None:
            self.llama.reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs_l = token_weight_pairs["l"]
        token_weight_pairs_g = token_weight_pairs["g"]
        token_weight_pairs_t5 = token_weight_pairs["t5xxl"]
        token_weight_pairs_llama = token_weight_pairs["llama"]
        lg_out = None
        pooled = None
        t5_out = None
        ll_out = None
        extra = {}

        if len(token_weight_pairs_g) > 0 or len(token_weight_pairs_l) > 0:
            if self.clip_l is not None:
                lg_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l)
            else:
                l_pooled = torch.zeros((1, 768), device=comfy.model_management.intermediate_device())

            if self.clip_g is not None:
                g_out, g_pooled = self.clip_g.encode_token_weights(token_weight_pairs_g)
            else:
                g_pooled = torch.zeros((1, 1280), device=comfy.model_management.intermediate_device())

            if self.clip_g is not None and self.clip_l is not None:
                pooled = torch.cat((l_pooled, g_pooled), dim=-1)            
                
        if self.t5xxl is not None:
            t5_output = self.t5xxl.encode_token_weights(token_weight_pairs_t5)
            t5_out, t5_pooled = t5_output[:2]
        else:
            t5_out = None

        if self.llama is not None:
            ll_output = self.llama.encode_token_weights(token_weight_pairs_llama)
            ll_out, ll_pooled = ll_output[:2]
            ll_out = ll_out[:, 1:]
        else:
            ll_out = None

        if t5_out is None:
            t5_path = folder_paths.get_full_path_or_raise("hidream_empty_latents", "t5_blank.pt")
            t5_out = torch.load(t5_path, map_location=comfy.model_management.intermediate_device())

        if ll_out is None:
            ll_out = torch.zeros((1, 32, 1, 4096), device=comfy.model_management.intermediate_device())

        if pooled is None:
            pooled_path = folder_paths.get_full_path_or_raise("hidream_empty_latents", "pooled_blank.pt")
            pooled = torch.load(pooled_path, map_location=comfy.model_management.intermediate_device())

        extra["conditioning_llama3"] = ll_out
        return t5_out, pooled, extra

    def load_sd(self, sd):
        if "text_model.encoder.layers.30.mlp.fc1.weight" in sd:
            return self.clip_g.load_sd(sd)
        elif "text_model.encoder.layers.1.mlp.fc1.weight" in sd:
            return self.clip_l.load_sd(sd)
        elif "encoder.block.23.layer.1.DenseReluDense.wi_1.weight" in sd:
            return self.t5xxl.load_sd(sd)
        else:
            return self.llama.load_sd(sd)


def hidream_clip(clip_l=True, clip_g=True, t5=True, llama=True, dtype_t5=None, dtype_llama=None, t5_quantization_metadata=None, llama_quantization_metadata=None):
    class HiDreamTEModel_(HiDreamTEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if t5_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["t5xxl_quantization_metadata"] = t5_quantization_metadata
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["llama_quantization_metadata"] = llama_quantization_metadata
            super().__init__(clip_l=clip_l, clip_g=clip_g, t5=t5, llama=llama, dtype_t5=dtype_t5, dtype_llama=dtype_llama, device=device, dtype=dtype, model_options=model_options)
    return HiDreamTEModel_
