# https://github.com/huggingface/diffusers (Apache 2.0) Qwen-Image 2.1
import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy.patcher_extension
import comfy.quant_ops
import comfy.rmsnorm
from comfy.ldm.flux.layers import EmbedND, timestep_embedding
from comfy.ldm.flux.math import apply_rope1
from comfy.ldm.lightricks.model import TimestepEmbedding
from comfy.ldm.modules.attention import ComfyAttention, optimized_attention
from comfy.ldm.wan.model_animate2 import PoseBranchCache


class ZeroCenteredRMSNorm(nn.Module):
    # stored weight is scale - 1, applied in fp32
    def __init__(self, dim, eps=1e-6, dtype=None, device=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dtype=dtype, device=device))
        self.eps = eps

    def forward(self, x):
        w = comfy.model_management.cast_to(self.weight, dtype=torch.float32, device=x.device) + 1.0
        return comfy.rmsnorm.rms_norm(x.float(), w, self.eps).to(x.dtype)


class TextProjection(nn.Module):
    def __init__(self, in_dim, hidden_size, eps=1e-6, dtype=None, device=None, operations=None):
        super().__init__()
        self.text_norm = ZeroCenteredRMSNorm(in_dim, eps=eps, dtype=dtype, device=device)
        self.in_layer = operations.Linear(in_dim, hidden_size, bias=False, dtype=dtype, device=device)
        self.out_layer = operations.Linear(hidden_size, hidden_size, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        return self.out_layer(F.gelu(self.in_layer(self.text_norm(x)), approximate="tanh"))


class TimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim, sample_proj_bias=False, dtype=dtype, device=device, operations=operations)

    def forward(self, timestep, dtype):
        return self.timestep_embedder(timestep_embedding(timestep.float(), 256).to(dtype))


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, fused=True, dtype=None, device=None, operations=None):
        super().__init__()
        self.fused = fused
        if fused:
            # [gate; up] in one GEMM, the SiLU-gate folded into the down projection's input quantizer
            self.gate_up = operations.Linear(dim, 2 * hidden_dim, bias=False, dtype=dtype, device=device)
        else:
            self.proj = operations.Linear(dim, hidden_dim, bias=False, dtype=dtype, device=device)
            self.gate_layer = operations.Linear(dim, hidden_dim, bias=False, dtype=dtype, device=device)
        self.out = operations.Linear(hidden_dim, dim, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        if self.fused:
            return comfy.ops.linear_input_act(self.out, self.gate_up(x), "swiglu")
        return self.out(F.silu(self.gate_layer(x)) * self.proj(x))


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, eps=1e-6, dtype=None, device=None, operations=None):
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.heads = heads
        inner_dim = heads * dim_head
        self.to_q = operations.Linear(dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_k = operations.Linear(dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_v = operations.Linear(dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_out = nn.ModuleList([operations.Linear(inner_dim, dim, bias=False, dtype=dtype, device=device)])
        self.norm_q = operations.RMSNorm(dim_head, eps=eps, dtype=dtype, device=device)
        self.norm_k = operations.RMSNorm(dim_head, eps=eps, dtype=dtype, device=device)

    def forward(self, x, pe, attn_fn, prefix_len, transformer_options={}):
        # (B, N, H, D) throughout: no transposes, the rope table is laid out to match
        B, N, _ = x.shape
        q = self.to_q(x).view(B, N, self.heads, -1)
        k = self.to_k(x).view(B, N, self.heads, -1)
        v = self.to_v(x).view(B, N, self.heads, -1)
        patches = transformer_options.get("patches", {}).get("attn1_patch", [])
        if comfy.model_management.in_training or patches:
            q, k = self.norm_q(q), self.norm_k(k)
            if patches:
                # patches see the Qwen-Image convention: (B, H, N, D) before rope, rope table (1, 1, N, ...), target image rows in img_slice
                q, k, v, pe = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), pe.transpose(1, 2)
                extra_options = {**transformer_options, "img_slice": [prefix_len, N]}
                for p in patches:
                    out = p(q, k, v, pe=pe, attn_mask=None, extra_options=extra_options)
                    q, k, v, pe = out.get("q", q), out.get("k", k), out.get("v", v), out.get("pe", pe)
                q, k, v, pe = q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous(), pe.transpose(1, 2).contiguous()
            q = apply_rope1(q, pe)
            k = apply_rope1(k, pe)
        else:
            q_scale, _, q_stream = comfy.ops.cast_bias_weight(self.norm_q, q, offloadable=True)
            k_scale, _, k_stream = comfy.ops.cast_bias_weight(self.norm_k, k, offloadable=True)
            q, k = comfy.quant_ops.ck.rms_rope(q, k, pe, q_scale, k_scale, self.norm_q.eps)
            comfy.ops.uncast_bias_weight(self.norm_q, q_scale, None, q_stream)
            comfy.ops.uncast_bias_weight(self.norm_k, k_scale, None, k_stream)
        return self.to_out[0](attn_fn(q, k, v, self.heads, preferred_attention=self.comfy_attention))


def _split_rows(p):
    # shared modulation rows: (t = 0 row for text and references, sampled-t rows for the target)
    return p[-1:].unsqueeze(1), p[:-1].unsqueeze(1)


def _modulated_norm(norm, x, scale, prefix_len, zero):
    # LayerNorm * (1 + scale), fused over every row with the target scale; the prefix rows are then redone with the t = 0 scale
    s_prefix, s_target = scale
    if comfy.model_management.in_training:
        out = norm(x)
        return torch.cat([out[:, :prefix_len] * (1 + s_prefix), out[:, prefix_len:] * (1 + s_target)], dim=1)
    out = comfy.quant_ops.ck.adaln(x, s_target, zero, norm.eps)
    if prefix_len:
        out[:, :prefix_len] = comfy.quant_ops.ck.adaln(x[:, :prefix_len], s_prefix, zero, norm.eps)
    return out


def _gated_residual(x, y, gate, prefix_len):
    g_prefix, g_target = gate
    x[:, prefix_len:].addcmul_(y[:, prefix_len:], g_target)
    if prefix_len:
        x[:, :prefix_len].addcmul_(y[:, :prefix_len], g_prefix)
    return x


class QwenImage21TransformerBlock(nn.Module):
    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=3, eps=1e-6, fused_mlp=True, dtype=None, device=None, operations=None):
        super().__init__()
        self.img_norm1 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.attn = Attention(dim, num_attention_heads, attention_head_dim, eps=eps, dtype=dtype, device=device, operations=operations)
        self.img_norm2 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.img_mlp = SwiGLUFeedForward(dim, dim * mlp_ratio, fused=fused_mlp, dtype=dtype, device=device, operations=operations)

    def forward(self, x, mod, pe, attn_fn, prefix_len, transformer_options={}):
        scale1, gate1, scale2, gate2, zero = mod
        x = _gated_residual(x, self.attn(_modulated_norm(self.img_norm1, x, scale1, prefix_len, zero), pe, attn_fn, prefix_len, transformer_options), gate1, prefix_len)
        x = _gated_residual(x, self.img_mlp(_modulated_norm(self.img_norm2, x, scale2, prefix_len, zero)), gate2, prefix_len)
        if x.dtype == torch.float16:
            x = x.clip(-65504, 65504)
        return x


class LastLayer(nn.Module):
    # scale only, no shift
    def __init__(self, dim, eps=1e-6, dtype=None, device=None, operations=None):
        super().__init__()
        self.linear = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.norm = operations.LayerNorm(dim, eps, elementwise_affine=False, dtype=dtype, device=device)

    def forward(self, x, temb):
        scale = self.linear(F.silu(temb)).unsqueeze(1)
        if comfy.model_management.in_training:
            return self.norm(x) * (1 + scale)
        return comfy.quant_ops.ck.adaln(x, scale, torch.zeros_like(scale[:1]), self.norm.eps)


def block_causal_attention(segments, transformer_options={}, cache=None, block_index=0, prefix_len=0):
    # segments: (start, end, mask); text segments get a causal mask, image blocks attend to everything before their end
    def attn(q, k, v, heads, preferred_attention=None):
        if cache is not None:
            # K and V stacked on dim 1 so batch stays first and quantized rows are per token and head
            cache.put(block_index, torch.stack([k[:, :prefix_len], v[:, :prefix_len]], dim=1))
        outs = [optimized_attention(q[:, start:end].flatten(2), k[:, :end].flatten(2), v[:, :end].flatten(2), heads, mask=mask, transformer_options=transformer_options, preferred_attention=preferred_attention)
                for start, end, mask in segments]
        return torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
    return attn


def prefix_cached_attention(prefix_k, prefix_v, transformer_options={}):
    # target-only queries: block-causal reduces to full attention over [cached prefix, target]
    def attn(q, k, v, heads, preferred_attention=None):
        return optimized_attention(q.flatten(2), torch.cat([prefix_k, k], dim=1).flatten(2), torch.cat([prefix_v, v], dim=1).flatten(2), heads, transformer_options=transformer_options, preferred_attention=preferred_attention)
    return attn


def prefix_cache_key(x, context, refs, slots):
    # one fp32 tensor per batch row: lengths and slots, then the prompt embedding and reference latents
    # the target shape is part of it because reference rope ids are centred on the target
    header = [context.shape[1], len(refs)] + list(x.shape[-2:]) + list(slots) + [s for r in refs for s in r.shape[-2:]]
    header = torch.tensor(header, dtype=torch.float32, device=context.device).expand(context.shape[0], -1)
    return torch.cat([header, context.float().flatten(1)] + [r.float().flatten(1) for r in refs], dim=1)


class QwenImage21Transformer2DModel(nn.Module):
    def __init__(
        self,
        in_channels=64,
        out_channels=64,
        num_layers=32,
        attention_head_dim=128,
        num_attention_heads=32,
        context_in_dim=4096,
        mlp_ratio=3,
        axes_dims_rope=(16, 56, 56),
        eps=1e-6,
        fused_mlp=True,
        image_model=None,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.dtype = dtype
        self.out_channels = out_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pe_embedder = EmbedND(dim=attention_head_dim, theta=10000, axes_dim=list(axes_dims_rope))
        self.time_text_embed = TimestepProjEmbeddings(self.inner_dim, dtype=dtype, device=device, operations=operations)
        self.txt_in = TextProjection(context_in_dim, self.inner_dim, eps=eps, dtype=dtype, device=device, operations=operations)
        self.img_in = operations.Linear(in_channels, self.inner_dim, bias=False, dtype=dtype, device=device)

        # one modulation shared by every block
        self.modulation = nn.Sequential(nn.SiLU(), operations.Linear(self.inner_dim, 4 * self.inner_dim, bias=False, dtype=dtype, device=device))

        self.transformer_blocks = nn.ModuleList([
            QwenImage21TransformerBlock(self.inner_dim, num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, eps=eps, fused_mlp=fused_mlp, dtype=dtype, device=device, operations=operations)
            for _ in range(num_layers)
        ])

        self.norm_out = LastLayer(self.inner_dim, eps=eps, dtype=dtype, device=device, operations=operations)
        self.proj_out = operations.Linear(self.inner_dim, out_channels, bias=False, dtype=dtype, device=device)

        # text + reference K/V are step-independent (t = 0 modulation, causal prefix), cached for one sampling run
        self.prefix_cache = None
        self.prefix_cache_enabled = False

    def reset_prefix_cache(self, enabled):
        if self.prefix_cache is not None:
            self.prefix_cache.free()
        self.prefix_cache = None
        self.prefix_cache_enabled = enabled

    def select_prefix_cache(self, key, cache_bytes, device, options):
        # returns (cache with the slot to read or fill selected, whether the slot is filled), or (None, False) to recompute
        if options.get("device") == "off":
            return None, False
        cache = self.prefix_cache
        if cache is not None and cache.select(key, create=False):
            return cache, cache.filled(len(self.transformer_blocks))
        dtype = options.get("dtype", "default")
        cache_bytes //= {"int8": 2, "int4": 4}.get(dtype, 1)
        if cache is None:
            store = options.get("device", "auto")
            if store == "auto":
                # spare VRAM first, then reclaim inactive model RAM for a host cache
                if self.current_patcher.get_free_memory(device) > 4 * cache_bytes:
                    store = device
                elif comfy.model_management.ensure_pin_budget(cache_bytes, evict_active=False):
                    store = torch.device("cpu")
                else:
                    return None, False
            else:
                store = device if store == "gpu" else torch.device("cpu")
            cache = self.prefix_cache = PoseBranchCache(store_device=store, dtype=dtype)
        if not (self.current_patcher.get_free_memory(device) > 2 * cache_bytes if cache.store_device == device else comfy.model_management.ensure_pin_budget(cache_bytes, evict_active=False)):
            # no room for this slot: recompute rather than evict the other cond's slot every step
            return None, False
        cache.select(key)
        return cache, False

    def build_sequence(self, x, context, ref_latents, image_slots):
        # text with each reference image spliced in at its slot, target image last
        txt = self.txt_in(context)
        slots = (image_slots + [txt.shape[1]] * len(ref_latents))[:len(ref_latents)]
        bounds = [0] + slots + [txt.shape[1]]

        parts, ids, segments = [], [], []
        pos, length = 0, 0
        for (start, end), img in zip(zip(bounds[:-1], bounds[1:]), ref_latents + [x]):
            n = end - start
            if n > 0:
                parts.append(txt[:, start:end])
                ids.append(torch.arange(pos, pos + n, device=x.device, dtype=torch.float32).unsqueeze(1).expand(n, 3))
                segments.append((length, length + n, torch.ones((n, length + n), dtype=torch.bool, device=x.device).tril(length)))
                pos += n
                length += n
            h, w = img.shape[-2:]
            parts.append(self.img_in(img.flatten(2).transpose(1, 2)))
            # half a token where a reference grid has the other parity, so it centres on the target
            hh = torch.arange(h, device=x.device, dtype=torch.float32) - (h - h // 2) + 0.5 * (h % 2 - x.shape[-2] % 2)
            ww = torch.arange(w, device=x.device, dtype=torch.float32) - (w - w // 2) + 0.5 * (w % 2 - x.shape[-1] % 2)
            ids.append(torch.stack([torch.full((h, w), pos, device=x.device, dtype=torch.float32), hh[:, None].expand(h, w), ww[None, :].expand(h, w)], dim=-1).flatten(0, 1))
            segments.append((length, length + h * w, None))
            pos += max(h, w)
            length += h * w

        # (1, N, 1, ...): the layout the fused rms_rope wants for (B, N, H, D) queries
        pe = self.pe_embedder(torch.cat(ids, dim=0).unsqueeze(0)).transpose(1, 2).contiguous()
        return torch.cat(parts, dim=1), pe, segments

    def forward(self, x, timestep, context, ref_latents=None, image_slots=None, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, timestep, context, ref_latents, image_slots, transformer_options, **kwargs)

    def _forward(self, x, timesteps, context, ref_latents=None, image_slots=None, transformer_options={}, **kwargs):
        B, C, H, W = x.shape
        dtype = x.dtype
        ref_latents = list(ref_latents or [])
        image_slots = list(image_slots or [])

        hidden_states, pe, segments = self.build_sequence(x, context, ref_latents, image_slots)
        prefix_len = hidden_states.shape[1] - H * W
        patches = transformer_options.get("patches", {})
        for p in patches.get("post_input", []):
            out = p({"img": hidden_states, "pe": pe, "transformer_options": transformer_options})
            hidden_states, pe = out["img"], out.get("pe", pe)

        # pipeline rounds t*1000 and t to the compute dtype; text and reference tokens modulate from t = 0
        t = ((timesteps * 1000).to(dtype) / 1000).to(dtype)
        temb = self.time_text_embed(torch.cat([t, t.new_zeros(1)]), dtype)
        scale1, gate1, scale2, gate2 = self.modulation(temb).chunk(4, dim=-1)
        mod = (_split_rows(scale1), _split_rows(gate1.tanh()), _split_rows(scale2), _split_rows(gate2.tanh()), torch.zeros_like(scale1[:1, None]))

        blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
        cache, cached = None, False
        # a cached step runs target rows only, so anything hooked into a block would see a different sequence from step 2
        hooked = blocks_replace or patches.get("post_input") or patches.get("single_block") or patches.get("attn1_patch")
        if self.prefix_cache_enabled and prefix_len > 0 and not hooked:
            key = prefix_cache_key(x, context, ref_latents, image_slots)
            cache_bytes = 2 * len(self.transformer_blocks) * B * prefix_len * self.inner_dim * hidden_states.element_size()
            cache, cached = self.select_prefix_cache(key, cache_bytes, x.device, transformer_options.get("qwen_image21_cache", {}))
        if cached:
            hidden_states, pe, prefix_len = hidden_states[:, prefix_len:], pe[:, prefix_len:], 0
        elif cache is not None:
            prefix_states, hidden_states = hidden_states[:, :prefix_len], hidden_states[:, prefix_len:]
            prefix_pe, pe = pe[:, :prefix_len], pe[:, prefix_len:]
            prefix_len = 0

        transformer_options["total_blocks"] = len(self.transformer_blocks)
        transformer_options["block_type"] = "single"
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.transformer_blocks), x.device, transformer_options)
        comfy.model_prefetch.malloc_graph_begin(x.device)
        for i, block in enumerate(self.transformer_blocks):
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, x.device, block, dtype, malloc_scope="block")
            transformer_options["block_index"] = i
            if cache is not None:
                if not cached:
                    with comfy.model_prefetch.pause_malloc_graph():
                        prefix_attn = block_causal_attention(segments[:-1], transformer_options, cache, i, prefix_states.shape[1])
                        prefix_states = block(prefix_states, mod, prefix_pe, prefix_attn, prefix_states.shape[1], transformer_options)
                prefix_k, prefix_v = cache.take(i, x.device, dtype, B).unbind(1)
                if cached:
                    cache.prefetch(i + 1, x.device, dtype)  # queue the next block before the compute it should overlap
                attn_fn = prefix_cached_attention(prefix_k, prefix_v, transformer_options)
            else:
                attn_fn = block_causal_attention(segments, transformer_options, cache, i, prefix_len)
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    return {"img": block(args["img"], mod, args["pe"], attn_fn, prefix_len, args["transformer_options"])}
                hidden_states = blocks_replace[("single_block", i)]({"img": hidden_states, "vec": temb, "pe": pe, "transformer_options": transformer_options}, {"original_block": block_wrap})["img"]
            else:
                hidden_states = block(hidden_states, mod, pe, attn_fn, prefix_len, transformer_options)
            for p in patches.get("single_block", []):
                hidden_states = p({"img": hidden_states, "x": x, "block_index": i, "transformer_options": transformer_options})["img"]

        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, x.device, None, malloc_scope="block")
        comfy.model_prefetch.malloc_graph_end()
        hidden_states = self.norm_out(hidden_states[:, prefix_len:], temb[:-1])
        hidden_states = self.proj_out(hidden_states)
        return hidden_states.transpose(1, 2).reshape(B, self.out_channels, H, W)
