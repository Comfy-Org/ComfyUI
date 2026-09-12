"""YuE2 score/semantic generation and acoustic prefix conditioning."""

import logging

import torch
from tokenizers import Tokenizer

import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy.utils
from comfy.ldm.yue2.model import model_config
from comfy.text_encoders.llama import FixedKV, Llama2_


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
    scores = logits.clone() if legacy_off else logits.float().clone()
    end = ABC_END if phase == "abc" else MUSIC_END
    allowed = torch.full_like(scores, -torch.inf)
    if phase == "abc":
        allowed[..., :EOD] = 0
    else:
        allowed[..., CODEC_OFFSET:CODEC_OFFSET + CODEC_SIZE] = 0
    allowed[..., end] = 0
    scores += allowed
    if step < min_tokens:
        scores[..., end] = -torch.inf
    if repetition_penalty != 1.0 and history:
        recent = torch.tensor([history[-penalty_window:]], dtype=torch.long, device=scores.device)
        counts = torch.zeros_like(scores)
        counts.scatter_add_(-1, recent, torch.ones_like(recent, dtype=scores.dtype))
        penalty = repetition_penalty ** counts
        scores = torch.where(scores < 0, scores * penalty, scores / penalty)
    if temperature == 0:
        return scores
    scores /= temperature
    threshold = scores.topk(min(top_k, scores.shape[-1])).values[..., -1, None]
    scores.masked_fill_(scores < threshold, -torch.inf)
    if top_p < 1:
        values, indices = scores.sort(descending=True)
        probabilities = values.softmax(-1)
        removed = probabilities.cumsum(-1) - probabilities > top_p
        removed[..., :3 if legacy_off else 1] = False
        values.masked_fill_(removed, -torch.inf)
        scores = values.scatter(-1, indices, values)
    return scores


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
        abc_length = 0 if tokens["cot"] == "off" else len(tokens["abc_ids"])
        length = min(config.max_position_embeddings, len(tokens["prefix"]) + abc_length + tokens["max_tokens"] + 2)
        branches = 1 if tokens["cfg_scale"] == 1.0 else 2
        dtype = torch.bfloat16 if comfy.model_management.should_use_bf16(device) else torch.float32
        cache = branches * 2 * config.num_hidden_layers * config.num_key_value_heads * config.head_dim * length
        prefill = branches * (length * length + length * (config.intermediate_size * 3 + config.hidden_size * 8))
        return (cache + prefill) * comfy.model_management.dtype_size(dtype)

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
        logits, cache, mask = self._prefill(prefixes, prefix_length + max_tokens, dtype)
        fixed_kv = isinstance(cache[0], FixedKV)
        decode_tokens = torch.empty((len(prefixes), 1), device=device, dtype=torch.long)
        positions = torch.tensor([[len(p)] for p in prefixes], device=device, dtype=torch.long)
        # Decoder inputs and rotary tensors must keep their addresses across graph replays.
        decode_buffers = None
        if fixed_kv:
            decode_buffers = (torch.empty((len(prefixes), 1, self.config.hidden_size), device=device, dtype=dtype),
                              self.model.compute_freqs_cis(positions, device))
        history = []
        end = ABC_END if phase == "abc" else MUSIC_END
        progress = comfy.utils.ProgressBar(max_tokens)
        try:
            for step in comfy.utils.model_trange(max_tokens, desc="YuE2 ABC sampling" if phase == "abc" else "YuE2 music sampling", unit="token"):
                comfy.model_management.throw_exception_if_processing_interrupted()
                guided = logits if cfg_scale == 1.0 else logits[1:] + cfg_scale * (logits[:1] - logits[1:])
                scores = distribution(guided, history, step, phase, legacy_off=legacy_off, **sampling)
                if sampling["temperature"] == 0:
                    next_id = scores.argmax(-1, keepdim=True)
                else:
                    probabilities = scores.softmax(-1).to(rng_device)
                    next_id = torch.multinomial(probabilities, 1, generator=generator).to(device)
                decode_tokens.copy_(next_id)
                token = next_id.item()
                progress.update_absolute(step + 1)
                if token == end:
                    return history, False
                history.append(token)
                if step + 1 < max_tokens:
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
