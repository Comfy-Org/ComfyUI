# LongCat-Video-Avatar: https://github.com/meituan-longcat/LongCat-Video
"""Wan-shaped DiT with per-head qk norm, SwiGLU and adaLN driven by a small time embedding.

Clean condition frames (an image, or the overlap of the previous segment, plus an optional
identity reference frame in front) run as their own stream at t = 0: they only attend to
each other and skip the text and audio cross-attention, so their per-block inputs can be
cached for the whole sampling run while the noisy frames attend to everything.
"""

import contextlib

import torch
from torch import nn

import comfy.ldm.common_dit
import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy.patcher_extension
import comfy.quant_ops
from comfy.ldm.flux.math import apply_rope1, rope
from comfy.ldm.modules.attention import ComfyAttention
from comfy.ldm.pan2.model import attention, gated_residual, modulated_norm, rms_rope
from comfy.ldm.qwen_image21.model import PrefixCache

from .model import sinusoidal_embedding_1d
from .model_animate2 import PoseBranchCache
from .model_multitalk import MultiTalkAudioProjModel, normalize_and_scale, project_audio_frames

FFN_CHUNK = 8192


def rope_1d(pos, dim):
    return rope(pos.unsqueeze(0), dim, 10000).unsqueeze(2)


def modulations(modulation, e, chunks, dtype):
    # fp32 like upstream: shifts and scales feed the fused adaln in the compute dtype, gates stay fp32
    return [m if i % 3 == 2 else m.to(dtype) for i, m in enumerate(modulation(e).chunk(chunks, dim=-1))]


def speaker_attn_map(q, k_ref, masks, heads, comfy_attention, transformer_options={}):
    # per mask, the head-averaged attention mass the queries put on its first-frame tokens: one attention call with the normalized masks as values
    b, s, h, d = q.shape
    v = masks.new_zeros(b, k_ref.shape[1], h, d)
    v[..., :masks.shape[0]] = (masks / (masks.sum(dim=1, keepdim=True) + 1e-8)).t().unsqueeze(1)
    out = attention(q, k_ref, v.flatten(2), heads, comfy_attention, transformer_options)
    return out.view(b, s, h, d)[0, :, :, :masks.shape[0]].float().mean(dim=1).t()


def speaker_positions(attn_map, speakers, audio_tokens, class_range=24, class_interval=4):
    # speakers 1 and 2 get the ends of a 1D rope range; the background and any further person share one slot, the middle, or 100 once other people are masked
    back_pos = class_range // 2 if attn_map.shape[0] <= 3 else 100
    h1 = normalize_and_scale(attn_map[0], (attn_map[0].min(), attn_map[0].max()), (0, class_interval))
    h2 = normalize_and_scale(attn_map[1], (attn_map[1].min(), attn_map[1].max()), (class_range - class_interval, class_range))
    back = torch.full_like(h1, back_pos)
    q_pos = torch.stack([h1, h2, back], dim=1).gather(1, attn_map.argmax(dim=0).clamp(max=2).unsqueeze(1)).squeeze(1)
    per_speaker = audio_tokens // speakers
    k_pos = torch.full((audio_tokens,), float(back_pos), device=attn_map.device)
    k_pos[:per_speaker] = class_interval / 2
    k_pos[per_speaker:2 * per_speaker] = class_range - class_interval / 2
    return q_pos, k_pos


class LongCatSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, device=None, dtype=None, operations=None):
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.k = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.v = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.o = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.norm_q = operations.RMSNorm(self.head_dim, eps=1e-6, device=device, dtype=dtype)
        self.norm_k = operations.RMSNorm(self.head_dim, eps=1e-6, device=device, dtype=dtype)

    def qkv(self, x, pe):
        b, s = x.shape[:2]
        q, k = rms_rope(self.q(x).view(b, s, self.num_heads, self.head_dim), self.k(x).view(b, s, self.num_heads, self.head_dim), self.norm_q, self.norm_k, pe)
        return q, k, self.v(x)

    def kv(self, x, pe):
        # the single-tensor form of qkv's fused rms_rope, so cached condition frames give the same K as computing them
        b, s = x.shape[:2]
        k = self.k(x).view(b, s, self.num_heads, self.head_dim)
        if comfy.model_management.in_training:
            return apply_rope1(self.norm_k(k), pe), self.v(x)
        k_scale, _, k_stream = comfy.ops.cast_bias_weight(self.norm_k, k, offloadable=True)
        k = comfy.quant_ops.ck.rms_rope1(k, pe, k_scale, self.norm_k.eps)
        comfy.ops.uncast_bias_weight(self.norm_k, k_scale, None, k_stream)
        return k, self.v(x)

    def forward_cond(self, x, pe, ref_len, transformer_options={}):
        # the reference frame and the condition frames each attend only within themselves
        q, k, v = self.qkv(x, pe)
        out = torch.cat([attention(q[:, a:b], k[:, a:b], v[:, a:b], self.num_heads, self.comfy_attention, transformer_options)
                         for a, b in ((0, ref_len), (ref_len, x.shape[1])) if b > a], dim=1)
        return self.o(out), k, v

    def forward_noise(self, x, pe, k_cond, v_cond, ref_len, ref_mask, speaker_masks, hw, transformer_options={}):
        q, k, v = self.qkv(x, pe)
        if k_cond is not None:
            k = torch.cat([k_cond, k], dim=1)
            v = torch.cat([v_cond, v], dim=1)
        attn_map = None
        if speaker_masks is not None:
            attn_map = speaker_attn_map(q[:1], k[:1, :hw], speaker_masks, self.num_heads, self.comfy_attention, transformer_options)
        if ref_mask is None:
            out = attention(q, k, v, self.num_heads, self.comfy_attention, transformer_options)
        else:
            # noisy frames near the reference frame's temporal slot do not see it, against repeated motion
            a, b = ref_mask
            parts = ((q[:, :a], k, v), (q[:, a:b], k[:, ref_len:], v[:, ref_len:]), (q[:, b:], k, v))
            out = torch.cat([attention(qq, kk, vv, self.num_heads, self.comfy_attention, transformer_options) for qq, kk, vv in parts if qq.shape[1] > 0], dim=1)
        return self.o(out), attn_map


class LongCatCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, device=None, dtype=None, operations=None):
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.num_heads = num_heads
        self.q = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.k = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.v = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.o = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.norm_q = operations.RMSNorm(dim // num_heads, eps=1e-6, device=device, dtype=dtype)
        self.norm_k = operations.RMSNorm(dim // num_heads, eps=1e-6, device=device, dtype=dtype)

    def forward(self, x, context, transformer_options={}):
        b = x.shape[0]
        q = self.norm_q(self.q(x).view(b, x.shape[1], self.num_heads, -1))
        k = self.norm_k(self.k(context).view(b, context.shape[1], self.num_heads, -1))
        return self.o(attention(q, k, self.v(context), self.num_heads, self.comfy_attention, transformer_options))


class LongCatAudioAttention(nn.Module):
    def __init__(self, dim, audio_dim, num_heads, device=None, dtype=None, operations=None):
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_linear = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.kv_linear = operations.Linear(audio_dim, 2 * dim, device=device, dtype=dtype)
        self.proj = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.q_norm = operations.RMSNorm(self.head_dim, eps=1e-6, device=device, dtype=dtype)
        self.k_norm = operations.RMSNorm(self.head_dim, eps=1e-6, device=device, dtype=dtype)

    def forward(self, x, audio, frames, speakers, attn_map=None, transformer_options={}):
        # each frame's tokens attend to that frame's audio tokens, one output per audio token set, the sets share the queries
        b, n, c = x.shape
        q = self.q_norm(self.q_linear(x).view(b, n, self.num_heads, self.head_dim))
        k_pe = None
        if attn_map is not None:
            q_pos, k_pos = speaker_positions(attn_map, speakers, audio[0].shape[1])
            q = apply_rope1(q, rope_1d(q_pos, self.head_dim))
            k_pe = rope_1d(k_pos, self.head_dim)
        q = q.reshape(b * frames, n // frames, self.num_heads, self.head_dim)
        out = []
        for a in audio:
            k, v = self.kv_linear(a).view(a.shape[0], a.shape[1], 2, self.num_heads, self.head_dim).unbind(2)
            k = self.k_norm(k)
            if k_pe is not None:
                k = apply_rope1(k, k_pe)
            out.append(self.proj(attention(q, k, v.flatten(2), self.num_heads, self.comfy_attention, transformer_options).view(b, n, c)))
        return out


class LongCatFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, device=None, dtype=None, operations=None):
        super().__init__()
        self.w1 = operations.Linear(dim, hidden_dim, bias=False, device=device, dtype=dtype)
        self.w2 = operations.Linear(hidden_dim, dim, bias=False, device=device, dtype=dtype)
        self.w3 = operations.Linear(dim, hidden_dim, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)).mul_(self.w3(x)))


class LongCatBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, adaln_dim, audio_dim, device=None, dtype=None, operations=None):
        super().__init__()
        self.norm1 = operations.LayerNorm(dim, eps=1e-6, elementwise_affine=False, device=device, dtype=dtype)
        self.self_attn = LongCatSelfAttention(dim, num_heads, device=device, dtype=dtype, operations=operations)
        self.norm3 = operations.LayerNorm(dim, eps=1e-6, device=device, dtype=dtype)
        self.cross_attn = LongCatCrossAttention(dim, num_heads, device=device, dtype=dtype, operations=operations)
        self.norm2 = operations.LayerNorm(dim, eps=1e-6, elementwise_affine=False, device=device, dtype=dtype)
        self.ffn = LongCatFeedForward(dim, ffn_dim, device=device, dtype=dtype, operations=operations)
        self.modulation = nn.Sequential(nn.SiLU(), operations.Linear(adaln_dim, 6 * dim, device=device, dtype=dtype))
        self.norm_x = operations.LayerNorm(dim, eps=1e-6, device=device, dtype=dtype)
        self.audio_cross_attn = LongCatAudioAttention(dim, audio_dim, num_heads, device=device, dtype=dtype, operations=operations)
        self.audio_modulation = nn.Sequential(nn.SiLU(), operations.Linear(adaln_dim, 3 * dim, device=device, dtype=dtype))

    def feed_forward(self, x, shift, scale, gate):
        if comfy.model_management.in_training:
            return gated_residual(x, self.ffn(modulated_norm(x, self.norm2, scale, shift)), gate)
        # token chunks bound the SwiGLU intermediate
        for chunk in x.split(FFN_CHUNK, dim=1):
            gated_residual(chunk, self.ffn(modulated_norm(chunk, self.norm2, scale, shift)), gate)
        return x

    def cond_kv(self, x, mod, pe):
        return self.self_attn.kv(modulated_norm(x, self.norm1, mod[1], mod[0]), pe)

    def forward_cond(self, x, mod, pe, ref_len, transformer_options={}):
        shift, scale, gate, shift_mlp, scale_mlp, gate_mlp = mod
        y, k, v = self.self_attn.forward_cond(modulated_norm(x, self.norm1, scale, shift), pe, ref_len, transformer_options)
        x = gated_residual(x, y, gate)
        del y
        return self.feed_forward(x, shift_mlp, scale_mlp, gate_mlp), k, v

    def forward_noise(self, x, mod, audio_mod, pe, k_cond, v_cond, context, audio, audio_strength, frames, speakers, ref_len, ref_mask, speaker_masks, hw, transformer_options={}):
        shift, scale, gate, shift_mlp, scale_mlp, gate_mlp = mod
        y, attn_map = self.self_attn.forward_noise(modulated_norm(x, self.norm1, scale, shift), pe, k_cond, v_cond, ref_len, ref_mask, speaker_masks, hw, transformer_options)
        x = gated_residual(x, y, gate)
        del y
        y = self.cross_attn(self.norm3(x), context, transformer_options)
        x = x + y if comfy.model_management.in_training else x.add_(y)
        del y
        if audio is not None:
            audio_shift, audio_scale, audio_gate = audio_mod
            y = [modulated_norm(a, self.norm1, audio_scale, audio_shift) for a in self.audio_cross_attn(self.norm_x(x), audio, frames, speakers, attn_map, transformer_options)]
            # with the null audio: null + strength * (audio - null)
            y = y[0] if len(y) == 1 else y[1].lerp_(y[0], audio_strength)
            x = gated_residual(x, y, audio_gate)
            del y
        return self.feed_forward(x, shift_mlp, scale_mlp, gate_mlp)


class TimeEmbedding(nn.Module):
    def __init__(self, freq_dim, dim, device=None, dtype=None, operations=None):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(operations.Linear(freq_dim, dim, device=device, dtype=dtype), nn.SiLU(), operations.Linear(dim, dim, device=device, dtype=dtype))

    def forward(self, t):
        return self.mlp(sinusoidal_embedding_1d(self.freq_dim, t))


class Head(nn.Module):
    def __init__(self, dim, out_dim, adaln_dim, device=None, dtype=None, operations=None):
        super().__init__()
        self.norm = operations.LayerNorm(dim, eps=1e-6, elementwise_affine=False, device=device, dtype=dtype)
        self.head = operations.Linear(dim, out_dim, device=device, dtype=dtype)
        self.modulation = nn.Sequential(nn.SiLU(), operations.Linear(adaln_dim, 2 * dim, device=device, dtype=dtype))

    def forward(self, x, e):
        shift, scale = self.modulation(e).to(x.dtype).chunk(2, dim=-1)
        # the output projection runs in fp32 like upstream
        return self.head(modulated_norm(x, self.norm, scale, shift).float())


class LongCatVideoModel(nn.Module):
    def __init__(self, patch_size=(1, 2, 2), in_dim=16, dim=4096, ffn_dim=11008, freq_dim=256, adaln_dim=512, text_dim=4096, out_dim=16, num_heads=32, num_layers=48,
                 image_model=None, device=None, dtype=None, operations=None):
        super().__init__()
        self.dtype = dtype
        self.patch_size = patch_size
        self.dim = dim
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.patch_embedding = operations.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size, device=device, dtype=dtype)
        self.text_embedding = nn.Sequential(operations.Linear(text_dim, dim, device=device, dtype=dtype), nn.GELU(approximate="tanh"), operations.Linear(dim, dim, device=device, dtype=dtype))
        self.time_embedding = TimeEmbedding(freq_dim, adaln_dim, device=device, dtype=dtype, operations=operations)
        audio_dim = 768
        self.blocks = nn.ModuleList([LongCatBlock(dim, ffn_dim, num_heads, adaln_dim, audio_dim, device=device, dtype=dtype, operations=operations) for _ in range(num_layers)])
        self.head = Head(dim, out_dim * patch_size[0] * patch_size[1] * patch_size[2], adaln_dim, device=device, dtype=dtype, operations=operations)
        # 5 frame windows of 5 Whisper large-v3 layer groups
        self.multitalk_audio_proj = MultiTalkAudioProjModel(seq_len=5, seq_len_vf=8, blocks=5, channels=1280, intermediate_dim=512, out_dim=audio_dim,
                                                            context_tokens=32, device=device, dtype=dtype, operations=operations)
        self.cond_cache = None
        self.cache_patcher = None

    def reset_cond_cache(self, patcher):
        # set per sampling run by the model base, None disables the cache
        if self.cond_cache is not None:
            self.cond_cache.free()
        self.cond_cache = None
        self.cache_patcher = patcher

    def select_cond_cache(self, key, cache_bytes, device, options):
        # (cache with the slot for these condition frames selected, whether that slot is filled), or (None, False) to recompute
        patcher = self.cache_patcher
        store = options.get("device", "auto")
        if store == "off":
            return None, False
        cache = self.cond_cache
        if cache is not None and cache.select(key, create=False):
            return cache, cache.filled(len(self.blocks))
        dtype = options.get("dtype", "default")
        cache_bytes //= {"int8": 2, "int4": 4}.get(dtype, 1)
        if cache is None:
            if store == "auto":
                # spare VRAM, then RAM; with dynamic VRAM auto mode gets writable slots from the model base's get_writable_model instead
                if patcher.get_free_memory(device) > 4 * cache_bytes:
                    store = device
                elif comfy.model_management.ensure_pin_budget(cache_bytes, evict_active=False):
                    store = torch.device("cpu")
                else:
                    return None, False
            else:
                store = device if store == "gpu" else torch.device("cpu")
            cache = self.cond_cache = PoseBranchCache(store_device=store, dtype=dtype)
        if not (patcher.get_free_memory(device) > 2 * cache_bytes if cache.store_device == device else comfy.model_management.ensure_pin_budget(cache_bytes, evict_active=False)):
            return None, False
        cache.select(key)
        return cache, False

    def rope(self, t_ids, h, w, device):
        ids = torch.stack(torch.meshgrid(t_ids, torch.arange(h, device=device, dtype=torch.float32), torch.arange(w, device=device, dtype=torch.float32), indexing="ij"), dim=-1).reshape(1, -1, 3)
        d = self.dim // self.num_heads
        axes = (d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6))
        # (1, N, 1, ...) for (B, N, H, D) queries and keys
        return torch.cat([rope(ids[..., i], a, 10000) for i, a in enumerate(axes)], dim=-3).unsqueeze(2)

    def forward(self, x, timestep, context, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, timestep, context, transformer_options, **kwargs)

    def _forward(self, x, timestep, context, transformer_options={}, reference_latent=None, cond_latents=None, audio_embed=None, ref_frame_index=10, ref_mask_frame_range=0,
                 speaker_masks=None, audio_strength=1.0, cond_cache_slots=None, **kwargs):
        B, _, T, H, W = x.shape
        dtype, device = x.dtype, x.device
        transformer_options = transformer_options.copy()
        x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size)
        num_cond_latents = 0
        if cond_latents is not None:
            # the clean condition frames replace the leading frames of the sampled latent
            num_cond_latents = cond_latents.shape[2]
            x = torch.cat([comfy.ldm.common_dit.pad_to_patch_size(cond_latents, self.patch_size), x[:, :, num_cond_latents:]], dim=2)
        n_ref = 0
        if reference_latent is not None:
            n_ref = 1
            x = torch.cat([comfy.ldm.common_dit.pad_to_patch_size(reference_latent.unsqueeze(2), self.patch_size), x], dim=2)
        n_cond = n_ref + num_cond_latents
        t_len, gh, gw = (s // p for s, p in zip(x.shape[2:], self.patch_size))
        hw = gh * gw
        n_noise = t_len - n_cond
        cond_len, ref_len = n_cond * hw, n_ref * hw

        pe = self.rope(torch.tensor([ref_frame_index] * n_ref + list(range(t_len - n_ref)), device=device, dtype=torch.float32), gh, gw, device)
        pe_cond, pe = pe[:, :cond_len], pe[:, cond_len:]

        # allocated outside the memory compiler region, the blocks update both streams in place
        tokens = self.patch_embedding(x).flatten(2).transpose(1, 2)
        x_cond = tokens[:, :cond_len].contiguous()
        x_noise = tokens[:, cond_len:].contiguous()
        del tokens

        # upstream rounds the timestep to the model dtype; condition frames are clean, t = 0
        t = timestep.to(dtype).float()
        if n_cond:
            t = torch.cat([t, t.new_zeros(1)])
        e = self.time_embedding(t).unsqueeze(1)
        e_noise = e[:B]

        # the umT5 encoder zeroes padded tokens, they stay zero after the embedding and are still attended
        context = self.text_embedding(context).masked_fill_((context == 0).all(dim=-1, keepdim=True), 0)

        audio = None
        speakers = 0
        if audio_embed is not None:
            # [B, speakers, video frames, window, layers, C] -> per noisy latent frame, the speakers' tokens side by side
            speakers = audio_embed.shape[1]
            audio = [audio_embed]
            if audio_strength != 1.0:
                # zeroed features, upstream's audio guidance null: the strength scales each block's audio residual relative to it
                audio.append(torch.zeros_like(audio_embed))
            audio = [project_audio_frames(self.multitalk_audio_proj, a.flatten(0, 1))[:, -n_noise:].unflatten(0, (B, speakers)).transpose(1, 2).flatten(2, 3).flatten(0, 1) for a in audio]

        ref_mask = None
        if n_ref and ref_mask_frame_range > 0:
            start = ref_frame_index - ref_mask_frame_range - num_cond_latents
            end = start + 2 * ref_mask_frame_range + 1
            if start >= 0 and end <= n_noise:
                ref_mask = (start * hw, end * hw)

        cache, cached = None, False
        if n_cond and self.cache_patcher is not None:
            key = x[:1, :, :n_cond]
            options = transformer_options.get("longcat_cache", {})
            if cond_cache_slots is not None and device in cond_cache_slots:
                # dynamic VRAM's writable slot; a batch mixing different condition frames recomputes them
                slots = list(dict.fromkeys(cond_cache_slots[device]))
                if len(slots) == 1:
                    cache = PrefixCache(slots)
                    cached = cache.filled(key)
                    for layer in slots[0]:
                        layer.read_before_write = cached
            elif options.get("device", "auto") in ("cpu", "gpu") or not self.cache_patcher.is_dynamic():
                cache_bytes = len(self.blocks) * cond_len * self.dim * x.element_size()
                cache, cached = self.select_cond_cache(key, cache_bytes, device, options)
        writable = isinstance(cache, PrefixCache)

        transformer_options["total_blocks"] = len(self.blocks)
        transformer_options["block_type"] = "double"
        units = [(block, cache.slots[0][i]) for i, block in enumerate(self.blocks)] if writable else list(self.blocks)
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(units, device, transformer_options)
        comfy.model_prefetch.malloc_graph_begin(device)
        for i, block in enumerate(self.blocks):
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, dtype, malloc_scope="block")
            transformer_options["block_index"] = i
            with cache.layer(i, device, write=not cached) if writable else contextlib.nullcontext():
                mod = modulations(block.modulation, e, 6, dtype)
                audio_mod = None
                if audio is not None:
                    audio_mod = modulations(block.audio_modulation, e_noise, 3, dtype)
                k_cond = v_cond = None
                if cached:
                    x_in = cache.take(i, device, dtype, 1)
                    if not writable:
                        cache.prefetch(i + 1, device, dtype)  # queue the next block before the compute it should overlap
                    # one row of condition frames serves the whole batch
                    k_cond, v_cond = (t.expand(B, *t.shape[1:]) for t in block.cond_kv(x_in, [m[B:] for m in mod], pe_cond))
                    del x_in
                elif n_cond:
                    if cache is not None:
                        with comfy.model_prefetch.pause_malloc_graph():
                            cache.put(i, x_cond[:1])
                    x_cond, k_cond, v_cond = block.forward_cond(x_cond, [m[B:] for m in mod], pe_cond, ref_len, transformer_options)
                x_noise = block.forward_noise(x_noise, [m[:B] for m in mod], audio_mod, pe, k_cond, v_cond, context, audio, audio_strength, n_noise, speakers, ref_len, ref_mask,
                                              speaker_masks, hw, transformer_options)
                del mod, audio_mod, k_cond, v_cond
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None, malloc_scope="block")
        comfy.model_prefetch.malloc_graph_end()
        if writable and not cached:
            cache.finish(key)

        # LongCat predicts x0 - noise, the reverse of the Wan flow target
        x_noise = -self.head(x_noise, e_noise)
        pt, ph, pw = self.patch_size
        x_noise = x_noise.view(B, n_noise, gh, gw, pt, ph, pw, self.out_dim).permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, self.out_dim, n_noise * pt, gh * ph, gw * pw)
        return nn.functional.pad(x_noise, (0, 0, 0, 0, num_cond_latents * pt, 0))[:, :, :T, :H, :W]
