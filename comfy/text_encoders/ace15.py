from .anima import Qwen3Tokenizer
import comfy.text_encoders.llama
from comfy import sd1_clip
import torch
import math
import yaml
import comfy.ops
import comfy.utils


def _audio_logits(model, x, audio_start, audio_end, eos_token=None):
    input = x[:, -1:]
    module = model.embed_tokens

    offload_stream = None
    if module.comfy_cast_weights:
        weight, _, offload_stream = comfy.ops.cast_bias_weight(module, input, offloadable=True)
    else:
        weight = module.weight.to(x)

    logits = torch.nn.functional.linear(input, weight[audio_start:audio_end], None)[:, -1]
    eos_logits = None
    if eos_token is not None:
        eos_logits = torch.nn.functional.linear(input, weight[eos_token:eos_token + 1], None)[:, -1]

    comfy.ops.uncast_bias_weight(module, weight, None, offload_stream)
    return logits, eos_logits


def sample_manual_loop_no_classes(
    model,
    ids=None,
    execution_dtype=None,
    cfg_scale: float = 2.0,
    temperature: float = 0.85,
    top_p: float = 0.9,
    top_k: int = None,
    min_p: float = 0.000,
    seed: int = 1,
    min_tokens: int = 1,
    max_new_tokens: int = 2048,
    audio_start_id: int = 151669,  # The cutoff ID for audio codes
    audio_end_id: int = 215669,
    eos_token_id: int = 151645,
):
    if ids is None:
        return []
    device = model.execution_device

    if execution_dtype is None:
        if comfy.model_management.should_use_bf16(device):
            execution_dtype = torch.bfloat16
        else:
            execution_dtype = torch.float32

    embeds, attention_mask, num_tokens, embeds_info = model.process_tokens(ids, device)
    embeds = embeds.to(execution_dtype)
    embeds_batch = embeds.shape[0]

    output_audio_codes = torch.empty((max_new_tokens,), device=device, dtype=torch.long)
    generated_tokens = 0
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    past_key_values = model.transformer.model.init_kv_cache(embeds_batch, embeds.shape[1] + max_new_tokens, device, execution_dtype)
    fixed_kv = isinstance(past_key_values[0], comfy.text_encoders.llama.FixedKV)

    progress_bar = comfy.utils.ProgressBar(max_new_tokens)
    sampling_logits = None

    for step in comfy.utils.model_trange(max_new_tokens, desc="LM sampling"):
        outputs = model.transformer(None, attention_mask, embeds=embeds, num_tokens=num_tokens, intermediate_output=None, dtype=execution_dtype, embeds_info=embeds_info, past_key_values=past_key_values)
        past_key_values = outputs[2]

        use_eos_score = eos_token_id is not None and eos_token_id < audio_start_id and min_tokens < step
        audio_logits, eos_logits = _audio_logits(model.transformer.model, outputs[0], audio_start_id, audio_end_id, eos_token_id if use_eos_score else None)
        if cfg_scale != 1.0:
            cfg_logits = audio_logits[1:2] + cfg_scale * (audio_logits[0:1] - audio_logits[1:2])
            if use_eos_score:
                cond_eos = eos_logits[0:1, 0]
                uncond_eos = eos_logits[1:2, 0]
                eos_score = uncond_eos + cfg_scale * (cond_eos - uncond_eos)
        else:
            cfg_logits = audio_logits[0:1]
            if use_eos_score:
                eos_score = eos_logits[0:1, 0]

        remove_logit_value = torch.finfo(cfg_logits.dtype).min
        if use_eos_score:
            cfg_logits = torch.cat((eos_score.unsqueeze(1), cfg_logits), dim=1)

        if top_k is not None and top_k > 0:
            top_k_values = torch.topk(cfg_logits, min(top_k, cfg_logits.shape[-1])).values
            cfg_logits[cfg_logits < top_k_values[..., -1, None]] = remove_logit_value

        if min_p is not None and min_p > 0:
            probs = torch.softmax(cfg_logits, dim=-1)
            p_max = probs.max(dim=-1, keepdim=True).values
            indices_to_remove = probs < (min_p * p_max)
            cfg_logits[indices_to_remove] = remove_logit_value

        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(cfg_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = torch.zeros_like(cfg_logits, dtype=torch.bool)
            indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
            cfg_logits[indices_to_remove] = remove_logit_value

        if temperature > 0:
            cfg_logits = cfg_logits / temperature
            if sampling_logits is None:
                sampling_logits = cfg_logits.new_empty((cfg_logits.shape[0], model.transformer.model.vocab_size))
            sampling_logits.fill_(remove_logit_value)
            if use_eos_score:
                sampling_logits[:, eos_token_id] = cfg_logits[:, 0]
                cfg_logits = cfg_logits[:, 1:]
            sampling_logits[:, audio_start_id:audio_end_id] = cfg_logits
            next_token = torch.multinomial(torch.softmax(sampling_logits, dim=-1), num_samples=1, generator=generator).squeeze(1)
        else:
            next_token = torch.argmax(cfg_logits, dim=-1)
            if use_eos_score:
                next_token = torch.where(next_token == 0, eos_token_id, next_token + audio_start_id - 1)
            else:
                next_token += audio_start_id

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

        input_ids = next_token.repeat(embeds_batch).unsqueeze(1)
        embeds = model.transformer.get_input_embeddings()(input_ids, out_dtype=execution_dtype)
        if not fixed_kv:
            attention_mask = torch.cat([attention_mask, torch.ones((embeds_batch, 1), device=device, dtype=attention_mask.dtype)], dim=1)

        output_audio_codes[generated_tokens] = next_token[0] - audio_start_id
        generated_tokens += 1
        progress_bar.update_absolute(step)

    return output_audio_codes[:generated_tokens].tolist()


def generate_audio_codes(model, positive, negative, min_tokens=1, max_tokens=1024, seed=0, cfg_scale=2.0, temperature=0.85, top_p=0.9, top_k=0, min_p=0.000):
    positive = [[token for token, _ in inner_list] for inner_list in positive]
    positive = positive[0]

    if cfg_scale != 1.0:
        negative = [[token for token, _ in inner_list] for inner_list in negative]
        negative = negative[0]

        neg_pad = 0
        if len(negative) < len(positive):
            neg_pad = (len(positive) - len(negative))
            negative = [model.special_tokens["pad"]] * neg_pad + negative

        pos_pad = 0
        if len(negative) > len(positive):
            pos_pad = (len(negative) - len(positive))
            positive = [model.special_tokens["pad"]] * pos_pad + positive

        ids = [positive, negative]
    else:
        ids = [positive]

    with comfy.ops.use_quantized_matmul(model, model.execution_device):
        return sample_manual_loop_no_classes(model, ids, cfg_scale=cfg_scale, temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p, seed=seed, min_tokens=min_tokens, max_new_tokens=max_tokens)


class ACE15Tokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name="qwen3_06b", tokenizer=Qwen3Tokenizer)

    def _metas_to_cot(self, *, return_yaml: bool = False, **kwargs) -> str:
        user_metas = {
            k: kwargs.pop(k)
            for k in ("bpm", "duration", "keyscale", "timesignature")
            if k in kwargs
        }
        timesignature = user_metas.get("timesignature")
        if isinstance(timesignature, str) and timesignature.endswith("/4"):
            user_metas["timesignature"] = timesignature[:-2]
        user_metas = {
            k: v if not isinstance(v, str) or not v.isdigit() else int(v)
            for k, v in user_metas.items()
            if v not in {"unspecified", None}
        }
        if len(user_metas):
            meta_yaml = yaml.dump(user_metas, allow_unicode=True, sort_keys=True).strip()
        else:
            meta_yaml = ""
        return f"<think>\n{meta_yaml}\n</think>" if not return_yaml else meta_yaml

    def _metas_to_cap(self, **kwargs) -> str:
        use_keys = ("bpm", "timesignature", "keyscale", "duration")
        user_metas = { k: kwargs.pop(k, "N/A") for k in use_keys }
        timesignature = user_metas.get("timesignature")
        if isinstance(timesignature, str) and timesignature.endswith("/4"):
            user_metas["timesignature"] = timesignature[:-2]
        duration = user_metas["duration"]
        if duration == "N/A":
            user_metas["duration"] = "30 seconds"
        elif isinstance(duration, (str, int, float)):
            user_metas["duration"] = f"{math.ceil(float(duration))} seconds"
        else:
            raise TypeError("Unexpected type for duration key, must be str, int or float")
        return "\n".join(f"- {k}: {user_metas[k]}" for k in use_keys)

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        text = text.strip()
        text_negative = kwargs.get("caption_negative", text).strip()
        lyrics = kwargs.get("lyrics", "")
        lyrics_negative = kwargs.get("lyrics_negative", lyrics)
        duration = kwargs.get("duration", 120)
        if isinstance(duration, str):
            duration = float(duration.split(None, 1)[0])
        language = kwargs.get("language")
        seed = kwargs.get("seed", 0)

        generate_audio_codes = kwargs.get("generate_audio_codes", True)
        cfg_scale = kwargs.get("cfg_scale", 2.0)
        temperature = kwargs.get("temperature", 0.85)
        top_p = kwargs.get("top_p", 0.9)
        top_k = kwargs.get("top_k", 0.0)
        min_p = kwargs.get("min_p", 0.000)

        duration = math.ceil(duration)
        kwargs["duration"] = duration
        tokens_duration = duration * 5
        min_tokens = int(kwargs.get("min_tokens", tokens_duration))
        max_tokens = int(kwargs.get("max_tokens", tokens_duration))

        metas_negative = {
            k.rsplit("_", 1)[0]: kwargs.pop(k)
            for k in ("bpm_negative", "duration_negative", "keyscale_negative", "timesignature_negative", "language_negative", "caption_negative")
            if k in kwargs
        }
        if not kwargs.get("use_negative_caption"):
            _ = metas_negative.pop("caption", None)

        cot_text = self._metas_to_cot(caption=text, **kwargs)
        cot_text_negative = "<think>\n\n</think>" if not metas_negative else self._metas_to_cot(**metas_negative)
        meta_cap = self._metas_to_cap(**kwargs)

        lm_template = "<|im_start|>system\n# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n<|im_end|>\n<|im_start|>user\n# Caption\n{}\n\n# Lyric\n{}\n<|im_end|>\n<|im_start|>assistant\n{}\n\n<|im_end|>\n"
        lyrics_template = "# Languages\n{}\n\n# Lyric\n{}<|endoftext|><|endoftext|>"
        qwen3_06b_template = "# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n# Caption\n{}\n\n# Metas\n{}\n<|endoftext|>\n<|endoftext|>"

        llm_prompts = {
            "lm_prompt": lm_template.format(text, lyrics.strip(), cot_text),
            "lm_prompt_negative": lm_template.format(text_negative, lyrics_negative.strip(), cot_text_negative),
            "lyrics": lyrics_template.format(language if language is not None else "", lyrics),
            "qwen3_06b": qwen3_06b_template.format(text, meta_cap),
        }

        out = {
            prompt_key: self.qwen3_06b.tokenize_with_weights(
                prompt,
                prompt_key == "qwen3_06b" and return_word_ids,
                disable_weights = True,
                **kwargs,
            )
            for prompt_key, prompt in llm_prompts.items()
        }
        out["lm_metadata"] = {"min_tokens": min_tokens,
                              "max_tokens": max_tokens,
                              "seed": seed,
                              "generate_audio_codes": generate_audio_codes,
                              "cfg_scale": cfg_scale,
                              "temperature": temperature,
                              "top_p": top_p,
                              "top_k": top_k,
                              "min_p": min_p,
                              }
        return out


class Qwen3_06BModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, attention_mask=True, model_options={}):
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config={}, dtype=dtype, special_tokens={"pad": 151643}, layer_norm_hidden_state=False, model_class=comfy.text_encoders.llama.Qwen3_06B_ACE15, enable_attention_masks=attention_mask, return_attention_masks=attention_mask, model_options=model_options)

class Qwen3_2B_ACE15(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, attention_mask=True, model_options={}):
        llama_quantization_metadata = model_options.get("llama_quantization_metadata", None)
        if llama_quantization_metadata is not None:
            model_options = model_options.copy()
            model_options["quantization_metadata"] = llama_quantization_metadata

        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config={}, dtype=dtype, special_tokens={"pad": 151643}, layer_norm_hidden_state=False, model_class=comfy.text_encoders.llama.Qwen3_2B_ACE15_lm, enable_attention_masks=attention_mask, return_attention_masks=attention_mask, model_options=model_options)

class Qwen3_4B_ACE15(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, attention_mask=True, model_options={}):
        llama_quantization_metadata = model_options.get("llama_quantization_metadata", None)
        if llama_quantization_metadata is not None:
            model_options = model_options.copy()
            model_options["quantization_metadata"] = llama_quantization_metadata

        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config={}, dtype=dtype, special_tokens={"pad": 151643}, layer_norm_hidden_state=False, model_class=comfy.text_encoders.llama.Qwen3_4B_ACE15_lm, enable_attention_masks=attention_mask, return_attention_masks=attention_mask, model_options=model_options)

class ACE15TEModel(torch.nn.Module):
    def __init__(self, device="cpu", dtype=None, dtype_llama=None, lm_model=None, model_options={}):
        super().__init__()
        if dtype_llama is None:
            dtype_llama = dtype

        model = None
        self.constant = 0.4375
        if lm_model == "qwen3_4b":
            model = Qwen3_4B_ACE15
            self.constant = 0.5625
        elif lm_model == "qwen3_2b":
            model = Qwen3_2B_ACE15

        self.lm_model = lm_model
        self.qwen3_06b = Qwen3_06BModel(device=device, dtype=dtype, model_options=model_options)
        if model is not None:
            setattr(self, self.lm_model, model(device=device, dtype=dtype_llama, model_options=model_options))
            ar_model = getattr(self, self.lm_model)
            ar_model.transformer.model.fixed_kv = True
            ar_model.transformer.model.prefetch_dynamic_vbars = True
            ar_model.transformer.model.graph_dynamic_vbar_blocks = True
        self.dtypes = set([dtype, dtype_llama])

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs_base = token_weight_pairs["qwen3_06b"]
        token_weight_pairs_lyrics = token_weight_pairs["lyrics"]

        self.qwen3_06b.set_clip_options({"layer": None})
        base_out, _, extra = self.qwen3_06b.encode_token_weights(token_weight_pairs_base)
        self.qwen3_06b.set_clip_options({"layer": [0]})
        lyrics_embeds, _, extra_l = self.qwen3_06b.encode_token_weights(token_weight_pairs_lyrics)

        out = {"conditioning_lyrics": lyrics_embeds[:, 0]}

        lm_metadata = token_weight_pairs["lm_metadata"]
        if lm_metadata["generate_audio_codes"]:
            audio_codes = generate_audio_codes(getattr(self, self.lm_model, self.qwen3_06b), token_weight_pairs["lm_prompt"], token_weight_pairs["lm_prompt_negative"], min_tokens=lm_metadata["min_tokens"], max_tokens=lm_metadata["min_tokens"], seed=lm_metadata["seed"], cfg_scale=lm_metadata["cfg_scale"], temperature=lm_metadata["temperature"], top_p=lm_metadata["top_p"], top_k=lm_metadata["top_k"], min_p=lm_metadata["min_p"])
            out["audio_codes"] = [audio_codes]

        return base_out, None, out

    def set_clip_options(self, options):
        self.qwen3_06b.set_clip_options(options)
        lm_model = getattr(self, self.lm_model, None)
        if lm_model is not None:
            lm_model.set_clip_options(options)

    def reset_clip_options(self):
        self.qwen3_06b.reset_clip_options()
        lm_model = getattr(self, self.lm_model, None)
        if lm_model is not None:
            lm_model.reset_clip_options()

    def get_dynamic_vram__units(self):
        if self.lm_model is None:
            return ([], [])
        model = getattr(self, self.lm_model)
        return model.transformer.model.get_dynamic_vram__units()

    def load_sd(self, sd):
        if "model.layers.0.post_attention_layernorm.weight" in sd:
            shape = sd["model.layers.0.post_attention_layernorm.weight"].shape
            if shape[0] == 1024:
                return self.qwen3_06b.load_sd(sd)
            else:
                return getattr(self, self.lm_model).load_sd(sd)

    def memory_estimation_function(self, token_weight_pairs, device=None):
        lm_metadata = token_weight_pairs.get("lm_metadata", {})
        constant = self.constant
        if comfy.model_management.should_use_bf16(device):
            constant *= 0.5

        token_weight_pairs = token_weight_pairs.get("lm_prompt", [])
        num_tokens = sum(map(lambda a: len(a), token_weight_pairs))
        num_tokens += lm_metadata.get("min_tokens", 0)
        return num_tokens * constant * 1024 * 1024

def te(dtype_llama=None, llama_quantization_metadata=None, lm_model="qwen3_2b"):
    class ACE15TEModel_(ACE15TEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["llama_quantization_metadata"] = llama_quantization_metadata
            super().__init__(device=device, dtype_llama=dtype_llama, lm_model=lm_model, dtype=dtype, model_options=model_options)
    return ACE15TEModel_
