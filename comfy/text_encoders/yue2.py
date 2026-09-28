"""YuE2 score/semantic generation and acoustic prefix conditioning."""

import logging

import torch
from tokenizers import Tokenizer

import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy.utils
from comfy.ldm.yue2.model import model_config
from comfy.text_encoders.llama import FixedKV, Llama2_, rope_matrix


EOD = 151643
ABC_START, ABC_END = 151847, 151848
MUSIC_START, MUSIC_END = 151851, 151852
CODEC_OFFSET, CODEC_SIZE = 151853, 32768
CONTEXT = 24576
FRAMES_PER_SECOND = 25
INSTRUCTIONS = {
    "off": "Generate music with codec tokens from the given conditions.",
    "melody": "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions.",
    "full": "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.",
}


def distribution(logits, history, step, phase, temperature, top_p, top_k, repetition_penalty, penalty_window, min_tokens, legacy_off=False):
    scores = logits.float() if not legacy_off else logits.clone()
    end = ABC_END if phase == "abc" else MUSIC_END

    if step < min_tokens:
        scores[..., end] = -torch.inf

    if repetition_penalty != 1.0 and history:
        recent = history[-penalty_window:]
        counts = {}
        for t in recent:
            counts[t] = counts.get(t, 0) + 1
        tok_indices = torch.tensor(list(counts.keys()), dtype=torch.long, device=scores.device)
        tok_counts = torch.tensor(list(counts.values()), dtype=scores.dtype, device=scores.device)
        penalties = repetition_penalty ** tok_counts
        orig_vals = scores[..., tok_indices]
        scores[..., tok_indices] = torch.where(orig_vals < 0, orig_vals * penalties, orig_vals / penalties)

    if phase == "abc":
        scores[..., EOD:] = -torch.inf
        scores[..., end] = logits[..., end]
    else:
        scores[..., :CODEC_OFFSET] = -torch.inf
        scores[..., CODEC_OFFSET + CODEC_SIZE:] = -torch.inf
        scores[..., end] = logits[..., end]

    if temperature == 0:
        return scores, None
    scores = scores / temperature

    k_val = min(top_k, scores.shape[-1])
    top_values, top_indices = scores.topk(k_val, dim=-1)

    if top_p < 1.0:
        sorted_values, sort_idx = top_values.sort(descending=True)
        sorted_indices = top_indices.gather(-1, sort_idx)

        probs = sorted_values.softmax(-1)
        cum_probs = probs.cumsum(-1) - probs
        mask = cum_probs > top_p
        mask[..., :3 if legacy_off else 1] = False
        sorted_values[mask] = -torch.inf
        return sorted_values, sorted_indices
    else:
        return top_values, top_indices


def chunk_ranges(frames, prefix_tokens, context=CONTEXT):
    size = (context - prefix_tokens - 3) // 2
    if frames < 1 or size < 1:
        raise ValueError("YuE2 needs music tokens and enough context for at least one acoustic frame.")
    return [(start, min(start + size, frames)) for start in range(0, frames, size)]


class YuE2Tokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        data = tokenizer_data["yue2_tokenizer_json"]
        if torch.is_tensor(data):
            data = data.numpy().tobytes()
        self.tokenizer_json = data
        self.tokenizer = Tokenizer.from_str(data.decode("utf-8"))

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        cot = kwargs.get("cot", "full")
        prompt = f"{INSTRUCTIONS[cot]}\n[Tags]\n{text}\n[Lyrics]\n{kwargs.get('lyrics', '')}\n"
        return {
            "prefix": [EOD] + self.tokenizer.encode(prompt).ids + [ABC_START],
            "negative": [EOD] + self.tokenizer.encode(INSTRUCTIONS[cot]).ids,
            "abc_ids": self.tokenizer.encode(kwargs.get("abc", "")).ids,
            "cot": cot,
            "seed": kwargs.get("seed", 0),
            "max_tokens": kwargs.get("max_tokens", 9000),
            "temperature": kwargs.get("temperature", 1.0),
            "top_p": kwargs.get("top_p", 0.95),
            "top_k": kwargs.get("top_k", 100),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.2),
            "penalty_window": kwargs.get("penalty_window", 100),
            "cfg_scale": kwargs.get("cfg_scale", 1.01 if cot == "off" else 1.0),
        }

    def state_dict(self):
        return {"yue2_tokenizer_json": torch.frombuffer(bytearray(self.tokenizer_json), dtype=torch.uint8)}

    def decode(self, ids, skip_special_tokens=True):
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


class YuE2TEModel(torch.nn.Module):
    def __init__(self, device="cpu", dtype=None, model_options={}, config=None):
        super().__init__()
        self.config = model_config(**{"fixed_kv": True, **(config or {})})
        operations = model_options.get("custom_operations", comfy.ops.manual_cast)
        quant = model_options.get("quantization_metadata")
        if quant is not None and "custom_operations" not in model_options:
            operations = comfy.ops.mixed_precision_ops(quant, dtype)
        self.model = Llama2_(self.config, device=device, dtype=dtype, ops=operations)
        self.model.prefetch_dynamic_vbars = True
        self.model.graph_dynamic_vbar_blocks = True
        self.dtypes = {dtype}
        self.execution_device = device

    def get_dynamic_vram__units(self):
        return self.model.get_dynamic_vram__units()

    def set_clip_options(self, options):
        self.execution_device = options.get("execution_device", self.execution_device)

    def reset_clip_options(self):
        pass

    def load_sd(self, state_dict):
        return self.load_state_dict(state_dict, strict=False, assign=getattr(self, "can_assign_sd", False))

    def memory_estimation_function(self, tokens, device=None):
        config = self.config
        abc_length = 0 if tokens.get("cot") == "off" else len(tokens.get("abc_ids", []))
        prefix_length = len(tokens.get("prefix", [])) + abc_length
        branches = 1 if tokens.get("cfg_scale", 1.0) == 1.0 else 2
        dtype = torch.bfloat16 if comfy.model_management.should_use_bf16(device) else torch.float32
        dtype_size = comfy.model_management.dtype_size(dtype)
        req_tokens = prefix_length + tokens.get("max_tokens", 1024) + 2
        is_abc = (tokens.get("cot") != "off" and len(tokens.get("abc_ids", [])) == 0)
        max_cap = 4096 if is_abc else 8192
        total_length = min(config.max_position_embeddings, req_tokens, max_cap)
        kv_cache = branches * 2 * config.num_hidden_layers * config.num_key_value_heads * config.head_dim * total_length * dtype_size
        prefill_act = branches * (prefix_length * prefix_length + prefix_length * (config.intermediate_size * 3 + config.hidden_size * 8)) * dtype_size
        decode_act = branches * (config.intermediate_size * 3 + config.hidden_size * 8) * dtype_size
        return kv_cache + max(prefill_act, decode_act)

    def _prefill(self, prefixes, capacity, dtype):
        length = max(map(len, prefixes))
        ids = torch.tensor([[0] * (length - len(prefix)) + prefix for prefix in prefixes], device=self.execution_device, dtype=torch.long)
        mask = positions = None
        if any(len(prefix) != length for prefix in prefixes):
            mask = torch.ones((len(prefixes), capacity), device=self.execution_device, dtype=torch.long)
            for index, prefix in enumerate(prefixes):
                mask[index, :length - len(prefix)] = 0
            positions = mask[:, :length].cumsum(-1).sub_(1).clamp_min_(0)
        cache = self.model.init_kv_cache(len(prefixes), capacity, self.execution_device, dtype)
        output = self.model(ids, attention_mask=mask[:, :length] if mask is not None else None,
                            position_ids=positions, past_key_values=cache, dtype=dtype)
        return self.model.lm_head(output[0][:, -1]), output[2], mask

    def _generate(self, prefix, seed, max_tokens, phase, dtype, negative=None, cfg_scale=1.0, legacy_off=False, **sampling):
        if max(len(prefix), len(negative or [])) + max_tokens > self.config.max_position_embeddings:
            raise ValueError("YuE2 prompt plus generation budget exceeds the model context; reduce the token budget or prompt length.")
        device = self.execution_device
        rng_device = device if torch.device(device).type != "mps" else "cpu"
        generator = torch.Generator(device=rng_device).manual_seed(seed)
        prefixes = [prefix] if cfg_scale == 1.0 else [prefix, negative]
        prefix_length = max(map(len, prefixes))
        budget_tokens = prefix_length + max_tokens
        if phase == "abc":
            capacity = min(budget_tokens, 4096)
        else:
            total_vram_mb = comfy.model_management.get_total_memory(device) / (1024**2)
            # On 4GB GPUs, total context (prefix + generation) must stay <= 6000 tokens to prevent PCIe shared memory paging
            vram_hard_cap = 6000 if total_vram_mb <= 4500 else self.config.max_position_embeddings
            free_vram_mb = comfy.model_management.get_free_memory(device) / (1024**2)
            kv_byte_per_tok = (1 if cfg_scale == 1.0 else 2) * 2 * self.config.num_hidden_layers * self.config.num_key_value_heads * self.config.head_dim * comfy.model_management.dtype_size(dtype)
            safe_cache_tokens = int(max(0, (free_vram_mb - 200) * (1024**2)) / kv_byte_per_tok)
            max_allowed_cap = min(vram_hard_cap, max(3072, safe_cache_tokens))
            capacity = min(budget_tokens, max_allowed_cap)
        actual_max_tokens = min(max_tokens, capacity - prefix_length)
        logits, cache, mask = self._prefill(prefixes, capacity, dtype)
        fixed_kv = isinstance(cache[0], FixedKV)
        decode_tokens = torch.empty((len(prefixes), 1), device=device, dtype=torch.long)
        positions = torch.tensor([[len(p)] for p in prefixes], device=device, dtype=torch.long)
        # Decoder inputs and rotary tensors must keep their addresses across graph replays.
        decode_buffers = None
        if fixed_kv:
            decode_buffers = (torch.empty((len(prefixes), 1, self.config.hidden_size), device=device, dtype=dtype),
                              rope_matrix(self.model.compute_freqs_cis(positions, device)))
        history = []
        end = ABC_END if phase == "abc" else MUSIC_END
        progress = comfy.utils.ProgressBar(actual_max_tokens)
        try:
            for step in comfy.utils.model_trange(actual_max_tokens, desc="YuE2 ABC sampling" if phase == "abc" else "YuE2 music sampling", unit="token"):
                comfy.model_management.throw_exception_if_processing_interrupted()
                guided = logits if cfg_scale == 1.0 else logits[1:] + cfg_scale * (logits[:1] - logits[1:])
                top_vals, top_idxs = distribution(guided, history, step, phase, legacy_off=legacy_off, **sampling)
                if sampling["temperature"] == 0:
                    if top_idxs is not None:
                        next_id = top_idxs[..., 0:1]
                    else:
                        next_id = top_vals.argmax(-1, keepdim=True)
                else:
                    probabilities = top_vals.softmax(-1)
                    sample_idx = torch.multinomial(probabilities, 1, generator=generator)
                    next_id = top_idxs.gather(-1, sample_idx)
                decode_tokens.copy_(next_id)
                token = next_id.item()
                progress.update_absolute(step + 1)
                if token == end:
                    return history, False
                history.append(token)
                if step + 1 < actual_max_tokens:
                    # Keep decode allocations stable; sampling has a changing history window.
                    if fixed_kv:
                        comfy.model_prefetch.malloc_graph_begin(device)
                    output = self.model(decode_tokens, past_key_values=cache, dtype=dtype, position_ids=positions,
                                        attention_mask=mask[:, :prefix_length + step + 1] if mask is not None and not fixed_kv else None,
                                        decode_buffers=decode_buffers)
                    logits.copy_(self.model.lm_head(output[0][:, -1]))
                    cache = output[2]
                    del output
                    if fixed_kv:
                        comfy.model_prefetch.malloc_graph_end()
                    positions.add_(1)
        finally:
            # Each phase has different KV buffers and may change the CFG batch size.
            comfy.model_prefetch.cleanup_prefetch_queues()
        logging.warning("YuE2 %s reached its token budget before the end token.", phase)
        return history, True

    def _acoustic_conditioning(self, prefix, tokens, dtype):
        config = self.config
        ranges = chunk_ranges(len(tokens), len(prefix), config.max_position_embeddings)
        total = sum(len(prefix) + end - start + 1 for start, end in ranges)
        # A normal [batch, tokens, features] conditioning tensor, with each layer's KV in features.
        output = torch.empty((1, total, config.num_hidden_layers, 2, config.num_key_value_heads, config.head_dim),
                             device=comfy.model_management.intermediate_device(), dtype=dtype)
        chunks = []
        offset = 0
        for start, end in ranges:
            comfy.model_management.throw_exception_if_processing_interrupted()
            ids = prefix + tokens[start:end] + [MUSIC_END]
            _, cache, _ = self._prefill([ids], len(ids), dtype)
            for index, kv in enumerate(cache):
                if isinstance(kv, FixedKV):
                    key, value = kv.key, kv.value
                else:
                    key, value, _ = kv
                    key, value = key.transpose(1, 2), value.transpose(1, 2)
                output[:, offset:offset + len(ids), index, 0].copy_(key)
                output[:, offset:offset + len(ids), index, 1].copy_(value)
            chunks.append((start, end, offset, offset + len(ids)))
            offset += len(ids)
            del cache
        return output.flatten(2), tuple(chunks)

    def generate(self, tokens, do_sample=True, max_length=256, temperature=1.0, top_k=50, top_p=0.95, repetition_penalty=1.0, seed=None, **kwargs):
        dtype = torch.bfloat16 if comfy.model_management.should_use_bf16(self.execution_device) else torch.float32
        ids, _ = self._generate(
            tokens["prefix"], tokens["seed"] if seed is None else seed, max_length, "abc", dtype,
            temperature=temperature if do_sample else 0, top_p=top_p, top_k=top_k,
            repetition_penalty=repetition_penalty, penalty_window=tokens.get("penalty_window", 100), min_tokens=min(32, max_length),
        )
        return ids

    def encode_token_weights(self, tokens):
        device = self.execution_device
        dtype = torch.bfloat16 if comfy.model_management.should_use_bf16(device) else torch.float32
        prefix = tokens["prefix"]
        abc_ids = tokens["abc_ids"]
        cot = tokens["cot"]
        if cot == "off":
            abc_ids = []
        prefix = prefix + abc_ids + [ABC_END, MUSIC_START]
        negative = tokens["negative"] + ([MUSIC_START] if cot == "off" else [ABC_START] + abc_ids + [ABC_END, MUSIC_START])
        context = self.config.max_position_embeddings
        max_tokens = min(tokens["max_tokens"], context - max(len(prefix), len(negative)))
        # One acoustic frame needs two positions plus three boundary tokens.
        if max_tokens < 1 or len(prefix) + 5 > context:
            raise ValueError("YuE2 prompt leaves no room for music; shorten the style, lyrics, or ABC.")
        if max_tokens < tokens["max_tokens"]:
            logging.info("YuE2 music budget reduced to %d tokens (%.2f seconds) to fit the prompt.", max_tokens, max_tokens / FRAMES_PER_SECOND)
        semantic, semantic_truncated = self._generate(
            prefix, tokens["seed"], max_tokens, "semantic", dtype,
            negative=negative, cfg_scale=tokens["cfg_scale"], legacy_off=cot == "off",
            temperature=tokens["temperature"], top_p=tokens["top_p"], top_k=tokens["top_k"],
            repetition_penalty=tokens["repetition_penalty"], penalty_window=50,
            min_tokens=min(200, max_tokens),
        )
        conditioning, chunks = self._acoustic_conditioning(prefix, semantic, dtype)
        return conditioning, None, {
            "yue2_chunks": chunks, "yue2_abc_ids": abc_ids, "yue2_frames": len(semantic),
            "yue2_truncated": semantic_truncated,
        }


def te(dtype_llama=None, llama_quantization_metadata=None):
    class YuE2TEModel_(YuE2TEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            dtype = comfy.model_management.pick_weight_dtype(dtype_llama, dtype, device)
            if llama_quantization_metadata is not None:
                model_options = {**model_options, "quantization_metadata": llama_quantization_metadata}
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return YuE2TEModel_
