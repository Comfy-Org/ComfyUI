import dataclasses
import hashlib

import torch
from torch import nn

import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy.utils
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.text_encoders.llama import Llama2_, Qwen3_8BConfig

from .prompt import AUDIO_CODE_OFFSET, SPECIAL_TOKEN_IDS


CFG_SCALE = 1.5
CFG_TOP_K = 50
C0_VOCAB_SIZE = 16384
MAX_PROMPT_TOKENS = 5000
MAX_AUDIO_FRAMES = 9000
AUDIO_FRAMES_PER_SECOND = 25


def derive_seed(seed, *parts):
    digest = hashlib.blake2b(digest_size=8, person=b"minimax-ttm")
    digest.update(int(seed).to_bytes(8, "little", signed=False))
    for part in parts:
        value = str(part).encode("utf-8")
        digest.update(len(value).to_bytes(4, "little"))
        digest.update(value)
    return int.from_bytes(digest.digest(), "little") & ((1 << 63) - 1)


def sample_topk(logits, top_k, generator):
    values = torch.nan_to_num(logits.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
    top_k = min(top_k, values.shape[-1])
    threshold = torch.topk(values, top_k, dim=-1).values[..., -1, None]
    values = values.masked_fill(values < threshold, -float("inf"))
    probabilities = torch.nan_to_num(torch.softmax(values, dim=-1), nan=0.0)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)


class RVQAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, merged_qkv, dtype, device, operations):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.merged_qkv = merged_qkv
        if merged_qkv:
            self.qkv_proj = operations.Linear(hidden_size, hidden_size * 3, bias=False, dtype=dtype, device=device)
        else:
            self.q_proj = operations.Linear(hidden_size, hidden_size, bias=False, dtype=dtype, device=device)
            self.k_proj = operations.Linear(hidden_size, hidden_size, bias=False, dtype=dtype, device=device)
            self.v_proj = operations.Linear(hidden_size, hidden_size, bias=False, dtype=dtype, device=device)
        self.o_proj = operations.Linear(hidden_size, hidden_size, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        batch, length, hidden_size = x.shape
        if self.merged_qkv:
            q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
        q = q.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        mask = torch.full((length, length), torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype).triu_(1)
        attention = optimized_attention_for_device(q.device, mask=True, small_input=True)
        out = attention(q, k, v, self.num_heads, mask=mask, skip_reshape=True)
        return self.o_proj(out)


class RVQRMSNorm(nn.Module):
    def __init__(self, hidden_size, dtype, device):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size, dtype=dtype, device=device))

    def forward(self, x):
        return torch.nn.functional.rms_norm(x, (x.shape[-1],), comfy.ops.cast_to_input(self.weight, x), 1e-6)


class RVQMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, merged_mlp, dtype, device, operations):
        super().__init__()
        self.merged_mlp = merged_mlp
        if merged_mlp:
            self.gate_up_proj = operations.Linear(hidden_size, intermediate_size * 2, bias=False, dtype=dtype, device=device)
        else:
            self.gate_proj = operations.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=device)
            self.up_proj = operations.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=device)
        self.down_proj = operations.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        if self.merged_mlp:
            return comfy.ops.linear_input_act(self.down_proj, self.gate_up_proj(x), "swiglu")
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class RVQDecoderBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, merged_qkv, merged_mlp, dtype, device, operations):
        super().__init__()
        self.input_layernorm = RVQRMSNorm(hidden_size, dtype, device)
        self.self_attn = RVQAttention(hidden_size, num_heads, merged_qkv, dtype, device, operations)
        self.post_attention_layernorm = RVQRMSNorm(hidden_size, dtype, device)
        self.mlp = RVQMLP(hidden_size, intermediate_size, merged_mlp, dtype, device, operations)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class RVQDepthDecoder(nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()
        hidden_size = int(config["hidden_size"])
        audio_vocab_size = int(config["audio_vocab_size"])
        merged_qkv = config.get("decoder_merged_qkv", False)
        merged_mlp = config.get("decoder_merged_mlp", False)
        num_codebooks = int(config["audio_num_codebooks"])
        self.projection = operations.Linear(hidden_size, hidden_size, bias=False, dtype=dtype, device=device)
        self.pos_embedding = operations.Embedding(16, hidden_size, dtype=dtype, device=device)
        self.audio_heads = nn.ModuleList([
            operations.Linear(hidden_size, audio_vocab_size, bias=False, dtype=dtype, device=device)
            for _ in range(num_codebooks - 1)
        ])
        self.layers = nn.ModuleList([
            RVQDecoderBlock(
                hidden_size,
                int(config["decoder_num_heads"]),
                int(config["decoder_intermediate_size"]),
                merged_qkv,
                merged_mlp,
                dtype,
                device,
                operations,
            )
            for _ in range(int(config["decoder_num_layers"]))
        ])
        self.norm = RVQRMSNorm(hidden_size, dtype, device)

    def forward(self, sequence):
        positions = torch.arange(sequence.shape[1], device=sequence.device)
        x = sequence + self.pos_embedding(positions, out_dtype=sequence.dtype).unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class MiniMaxMusic3AR(nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()
        config_fields = {field.name for field in dataclasses.fields(Qwen3_8BConfig)}
        qwen_config = Qwen3_8BConfig(**{key: value for key, value in config.items() if key in config_fields})
        qwen_config.lm_head = False
        qwen_config.fixed_kv = True
        self.model = Llama2_(qwen_config, device=device, dtype=dtype, ops=operations)
        self.model.prefetch_dynamic_vbars = True
        self.model.graph_dynamic_vbar_blocks = True
        self.model.lm_head = operations.Linear(qwen_config.hidden_size, qwen_config.vocab_size, bias=False, dtype=dtype, device=device)
        self.model.lm_head_pruned = operations.Linear(qwen_config.hidden_size, C0_VOCAB_SIZE + 1, bias=False, dtype=dtype, device=device)
        self.model.embed_tokens_prefill = operations.Embedding(AUDIO_CODE_OFFSET, qwen_config.hidden_size, dtype=dtype, device=device)
        self.model.embed_tokens_audio = operations.Embedding(C0_VOCAB_SIZE, qwen_config.hidden_size, dtype=dtype, device=device)
        self.model.pruned_lm_head = None
        self.model.pruned_embedding = None
        self.model.audio_extra_embedding = operations.Embedding(
            int(config["audio_vocab_size"]) * (int(config["audio_num_codebooks"]) - 1),
            qwen_config.hidden_size,
            dtype=dtype,
            device=device,
        )
        self.model.audio_decoder = RVQDepthDecoder(config, dtype, device, operations)
        self.audio_vocab_size = int(config["audio_vocab_size"])
        self.num_codebooks = int(config["audio_num_codebooks"])
        self.embedding_scale = self.num_codebooks ** -0.5

    def _guided_c0(self, logits, cfg_scale, top_k):
        conditioned = logits[0:1].float()
        unconditioned = logits[1:2].float()
        guided = unconditioned + (conditioned - unconditioned) * cfg_scale
        threshold = torch.topk(conditioned, top_k, dim=-1).values[..., -1, None]
        return guided.masked_fill(conditioned < threshold, -float("inf"))

    def _depth_codes(self, hidden, c0, c0_embed, generator, execution_dtype, cfg_scale, top_k):
        decoder = self.model.audio_decoder
        sequence = [decoder.projection(hidden).unsqueeze(1)]
        sequence.append(decoder.projection(c0_embed).unsqueeze(1))
        codes = [c0]
        hidden_parts = []
        for index in range(1, self.num_codebooks):
            out = decoder(torch.cat(sequence, dim=1))[:, -1]
            hidden_parts.append(out[:1].detach())
            logits = decoder.audio_heads[index - 1](out)
            conditioned = logits[:1].float()
            unconditioned = logits[1:2].float()
            code = sample_topk(unconditioned + (conditioned - unconditioned) * cfg_scale, top_k, generator).repeat(2)
            codes.append(code)
            if index < self.num_codebooks - 1:
                embedding = self.model.audio_extra_embedding(
                    code + (index - 1) * self.audio_vocab_size,
                    out_dtype=execution_dtype,
                )
                sequence.append(decoder.projection(embedding).unsqueeze(1))
        return torch.stack(codes, dim=1), torch.cat(hidden_parts, dim=-1)

    def _embed_c0(self, codes, execution_dtype):
        if self.model.pruned_embedding:
            return self.model.embed_tokens_audio(codes, out_dtype=execution_dtype)
        return self.model.embed_tokens(codes + AUDIO_CODE_OFFSET, out_dtype=execution_dtype)

    def _embed_audio_frame(self, codes, execution_dtype):
        c0 = self._embed_c0(codes[:, 0], execution_dtype)
        offsets = torch.arange(self.num_codebooks - 1, device=codes.device) * self.audio_vocab_size
        extra = self.model.audio_extra_embedding(codes[:, 1:] + offsets.unsqueeze(0), out_dtype=execution_dtype).sum(dim=1)
        return ((c0 + extra) * self.embedding_scale).unsqueeze(1)

    def _sample_c0(self, hidden, cfg_scale, top_k, generator, vocab_mask):
        if self.model.pruned_lm_head:
            guided = self._guided_c0(self.model.lm_head_pruned(hidden).float(), cfg_scale, top_k)
            code = sample_topk(guided, top_k, generator)
            stop_token = 0
            offset = 1
        else:
            logits = self.model.lm_head(hidden).float()
            stop_token = SPECIAL_TOKEN_IDS["<|audio_end|>"]
            logits = logits.masked_fill(vocab_mask, -float("inf"))
            guided = self._guided_c0(logits, cfg_scale, top_k).masked_fill(vocab_mask, -float("inf"))
            code = sample_topk(guided, top_k, generator)
            offset = AUDIO_CODE_OFFSET
        return torch.where(code == stop_token, 0, code - offset), code, stop_token

    def generate(self, input_ids, seed, max_audio_frames, device, cfg_scale=CFG_SCALE, top_k=CFG_TOP_K):
        prompt_tokens = int(input_ids.shape[1])
        if prompt_tokens > MAX_PROMPT_TOKENS:
            raise ValueError(f"MiniMax Music3 prompt has {prompt_tokens} tokens; maximum is {MAX_PROMPT_TOKENS}")

        input_ids = input_ids.to(device)
        if comfy.model_management.should_use_bf16(device):
            execution_dtype = torch.bfloat16
        else:
            execution_dtype = torch.float32
        unconditioned = input_ids.clone()
        unconditioned[:, 1:-2] = SPECIAL_TOKEN_IDS["<|audio_cfg|>"]
        text_ids = torch.cat((input_ids, unconditioned), dim=0)
        if self.model.pruned_embedding:
            text_embeds = self.model.embed_tokens_prefill(text_ids, out_dtype=execution_dtype)
        else:
            text_embeds = self.model.embed_tokens(text_ids, out_dtype=execution_dtype)
        decode_limit = min(int(max_audio_frames), MAX_AUDIO_FRAMES)
        past = self.model.init_kv_cache(2, prompt_tokens + decode_limit + 1, device, execution_dtype)
        output = self.model(None, embeds=text_embeds, past_key_values=past, dtype=execution_dtype)
        last_hidden = output[0][:, -1].clone()
        past = output[2]
        del output
        vbar = getattr(self, "dynamic_vbars", {}).get(device)
        if vbar is not None:
            comfy.model_management.reset_cast_buffers()
            vbar.set_watermark(vbar.max_size)

        generator = torch.Generator(device=device).manual_seed(derive_seed(seed, "ar"))
        decoder = self.model.audio_decoder
        depth_io = {
            "hidden": torch.empty_like(last_hidden),
            "c0": torch.empty((last_hidden.shape[0],), dtype=torch.long, device=device),
            "c0_embed": torch.empty_like(last_hidden),
            "codes": torch.empty((last_hidden.shape[0], self.num_codebooks), dtype=torch.long, device=device),
            "depth_hidden": torch.empty((1, last_hidden.shape[-1] * (self.num_codebooks - 1)), dtype=execution_dtype, device=device),
        }
        hidden_frames = []
        pending_code = None
        stop_token = None
        pending_event = None
        pending_hidden = torch.empty(last_hidden.shape[-1] * self.num_codebooks, dtype=execution_dtype, device=device)
        pending_hidden_valid = False
        progress = comfy.utils.ProgressBar(decode_limit)
        cuda_device = torch.device(device).type == "cuda"
        vocab_mask = None
        if not self.model.pruned_lm_head:
            vocab_mask = torch.ones(self.model.vocab_size, dtype=torch.bool, device=device)
            vocab_mask[AUDIO_CODE_OFFSET:AUDIO_CODE_OFFSET + C0_VOCAB_SIZE] = False
            vocab_mask[SPECIAL_TOKEN_IDS["<|audio_end|>"]] = False

        for frame_index in comfy.utils.model_trange(decode_limit + 1, desc="AR sampling"):
            comfy.model_management.throw_exception_if_processing_interrupted()
            if pending_code is not None:
                if pending_event is not None:
                    pending_event.synchronize()
                if int(pending_code.item()) == stop_token:
                    pending_hidden_valid = False
                    break
                if pending_hidden_valid:
                    hidden_frames.append(pending_hidden.clone())
                    progress.update_absolute(len(hidden_frames))
                    if len(hidden_frames) >= decode_limit:
                        break

            if frame_index:
                comfy.model_prefetch.malloc_graph_begin(device)
            c0, code_or_stop, stop_token = self._sample_c0(last_hidden, cfg_scale, top_k, generator, vocab_mask)
            if pending_code is None:
                pending_code = torch.empty_like(code_or_stop, device="cpu", pin_memory=cuda_device)
                if cuda_device:
                    pending_event = torch.cuda.Event()
            pending_code.copy_(code_or_stop, non_blocking=cuda_device)
            if pending_event is not None:
                pending_event.record()

            c0 = c0.repeat(2)
            c0_embed = self._embed_c0(c0, execution_dtype)
            depth_io["hidden"].copy_(last_hidden)
            depth_io["c0"].copy_(c0)
            depth_io["c0_embed"].copy_(c0_embed)

            def depth_core():
                codes, depth_hidden = self._depth_codes(
                    depth_io["hidden"], depth_io["c0"], depth_io["c0_embed"], generator, execution_dtype, cfg_scale, top_k
                )
                depth_io["codes"].copy_(codes)
                depth_io["depth_hidden"].copy_(depth_hidden)

            depth_queue = comfy.model_prefetch.make_prefetch_queue(
                [[decoder, self.model.audio_extra_embedding]], device, {"prefetch_dynamic_vbars": True}
            )
            comfy.model_prefetch.prefetch_queue_pop(
                depth_queue, device, decoder, execution_dtype, core=depth_core, enable_graph=True,
                generator=generator, malloc_scope="depth"
            )
            comfy.model_prefetch.prefetch_queue_pop(
                depth_queue, device, None, malloc_scope="depth"
            )
            feedback_codes = depth_io["codes"]
            depth_hidden = depth_io["depth_hidden"]
            frame_hidden = torch.cat((last_hidden[:1].detach(), depth_hidden), dim=-1)
            if frame_index > 0:
                pending_hidden.copy_(frame_hidden[0])
                pending_hidden_valid = True

            feedback = self._embed_audio_frame(feedback_codes, execution_dtype)
            output = self.model(None, embeds=feedback, past_key_values=past, dtype=execution_dtype)
            last_hidden.copy_(output[0][:, -1])
            past = output[2]
            del output, feedback, frame_hidden, depth_hidden, feedback_codes, c0_embed, c0, code_or_stop
            comfy.model_prefetch.malloc_graph_end()

        if pending_hidden_valid and len(hidden_frames) < decode_limit:
            if pending_event is not None:
                pending_event.synchronize()
            if int(pending_code.item()) != stop_token:
                hidden_frames.append(pending_hidden.clone())

        if not hidden_frames:
            raise ValueError("MiniMax Music3 generated zero audio frames")
        return torch.stack(hidden_frames).to(device="cpu")
