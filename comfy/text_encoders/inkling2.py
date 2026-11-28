# Port of Inkling-2 (Thinking Machines Lab) inkling2_loop multimodal model for ComfyUI.
# Reference: transformers models/inkling2_loop + models/inkling2 (vision tower).
# Checkpoint uses the original TML key names (model.llm.*, model.visual.*) with
# NVFP4/MXFP8 quantized MoE experts (comfy_quant metadata), loaded via comfy ops.

import math
import os

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops
import comfy.sd1_clip as sd1_clip
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention_for_device


class Inkling2Config:
    def __init__(self, config_dict={}):
        t = config_dict.get("text_config", {})
        self.hidden_size = t.get("hidden_size", 2304)
        self.num_hidden_layers = t.get("num_hidden_layers", 24)
        self.vocab_size = t.get("vocab_size", 201024)
        self.unpadded_vocab_size = t.get("unpadded_vocab_size", 200058)
        self.num_attention_heads = t.get("num_attention_heads", 24)
        self.num_key_value_heads = t.get("num_key_value_heads", 8)
        self.head_dim = t.get("head_dim", 128)
        self.swa_head_dim = t.get("swa_head_dim", 128)
        self.swa_num_attention_heads = t.get("swa_num_attention_heads", 24)
        self.swa_num_key_value_heads = t.get("swa_num_key_value_heads", 8)
        self.sliding_window_size = t.get("sliding_window_size", 1024)
        self.d_rel = t.get("d_rel", 16)
        self.rel_extent = t.get("rel_extent", 1024)
        self.rms_norm_eps = t.get("rms_norm_eps", 1e-6)
        self.qk_l2_clip_threshold = t.get("qk_l2_clip_threshold", 22.627416997969522)
        self.rel_l2_clip_threshold = t.get("rel_l2_clip_threshold", 8.0)
        self.log_scaling_n_floor = t.get("log_scaling_n_floor", 8192)
        self.log_scaling_alpha = t.get("log_scaling_alpha", 0.1)
        self.local_layer_ids = t.get("local_layer_ids", [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22])
        self.layer_types = ["hybrid_sliding" if i in self.local_layer_ids else "hybrid" for i in range(self.num_hidden_layers)]
        self.conv_kernel_size = t.get("sconv_kernel_size", 4)
        self.n_layers_per_attn_res_block = t.get("n_layers_per_attn_res_block", 6)
        self.intermediate_size = t.get("intermediate_size", 1152)
        self.n_routed_experts = t.get("n_routed_experts", 128)
        self.num_experts_per_tok = t.get("num_experts_per_tok", 8)
        self.route_scale = t.get("route_scale", 8.0)
        self.mlp_gate_softcap = t.get("mlp_gate_softcap", 4.0)
        self.mlp_up_softcap = t.get("mlp_up_softcap", 25.0)
        self.final_logit_softcapping = t.get("final_logit_softcapping", 30.0)
        loop = t.get("loop_config", {})
        self.loop_start = loop.get("core_start", 6)
        self.loop_end = loop.get("core_end", 18)
        self.loop_num_passes = loop.get("num_passes", 3)
        self.loop_gate_rank = loop.get("gate_rank", 64)
        v = config_dict.get("vision_config", {})
        self.vision_hidden_size = v.get("hidden_size", 1024)
        self.vision_num_layers = v.get("num_hidden_layers", 23)
        self.vision_num_heads = v.get("num_attention_heads", 16)
        self.vision_patch_size = v.get("native_patch_size", 20)
        self.vision_merge_size = v.get("adapter_merge_size", 2)
        self.vision_posemb_grid_size = v.get("posemb_grid_size", 32)
        self.vision_intermediate_size = int(v.get("mlp_ratio", 4.0) * self.vision_hidden_size)
        self.vision_projection_hidden_dim = v.get("projection_hidden_dim", 4096)
        self.vision_text_hidden_size = v.get("text_hidden_size", self.hidden_size)
        self.vision_layer_norm_eps = v.get("layer_norm_eps", 1e-5)
        self.vision_layer_scale_init_value = v.get("layer_scale_init_value", 0.1)
        self.vision_rope_theta = v.get("rope_theta", 10000.0)
        self.image_token_id = config_dict.get("image_token_id", 200054)
        self.stop_tokens = config_dict.get("stop_tokens", [199999, 200006])
        self.eos_token_id = config_dict.get("eos_token_id", 200006)


# ---------------------------------------------------------------- norms / clips

def rms_norm_weighted(x, weight, eps):
    x = x.to(torch.float32)
    out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (weight.to(torch.float32) * out).to(torch.bfloat16)


class _CachedCast:
    """Cache a detached dtype cast of a static parameter (norm/clip weights).

    Recomputed when the parameter is replaced, moved, or reloaded. Inference tensors
    (torch.inference_mode, used by ComfyUI execution) have no version counter, so the
    load hook and data_ptr cover invalidation there.
    """

    def _cast_cache(self):
        cache = self.__dict__.get("_casts")
        if cache is None:
            cache = self.__dict__["_casts"] = {}
        return cache

    def _cached_cast(self, name, dtype):
        src = getattr(self, name)
        try:
            ver = src._version
        except RuntimeError:  # inference tensors do not track a version counter
            ver = None
        key = (id(src), src.data_ptr(), ver)
        cache = self._cast_cache()
        entry = cache.get(name)
        if entry is None or entry[0] != key:
            entry = (key, src, src.detach().to(dtype))
            cache[name] = entry
        return entry[2]

    def _load_from_state_dict(self, *args, **kwargs):
        self._cast_cache().clear()
        return super()._load_from_state_dict(*args, **kwargs)

    def _apply(self, fn, recurse=True):
        self._cast_cache().clear()
        return super()._apply(fn, recurse)


class RMSNorm(_CachedCast, nn.Module):
    def __init__(self, hidden_size, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, x):
        w = self._cached_cast("weight", torch.float32)
        x = x.to(torch.float32)
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return (w * out).to(torch.bfloat16)


class UnweightedRMSNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        x = x.to(torch.float32)
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)).to(torch.bfloat16)


class L2Clip(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        x = x.to(torch.float32)
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        return (x * (self.threshold / (norm + 1e-6)).clamp(max=1.0)).to(torch.bfloat16)


class ScaledL2Clip(_CachedCast, nn.Module):
    def __init__(self, num_heads, head_dim, threshold):
        super().__init__()
        self.l2_clip = L2Clip(threshold)
        self.scale = nn.Parameter(torch.ones(num_heads, head_dim))

    def forward(self, x, position_ids=None):
        return self.l2_clip(x.to(torch.float32) * self._cached_cast("scale", torch.float32))


class LogScale(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.alpha = config.log_scaling_alpha
        self.n_floor = config.log_scaling_n_floor

    def forward(self, x, position_ids):
        scale = 1.0 + self.alpha * torch.log(((position_ids + 1).to(torch.float32) / self.n_floor).clamp(min=1.0))
        return (x.to(torch.float32) * scale[..., None, None]).to(torch.bfloat16)


class LogScaledL2Clip(_CachedCast, nn.Module):
    def __init__(self, config, num_heads, head_dim, threshold):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_heads, head_dim))
        self.l2_clip = L2Clip(threshold)
        self.log_scale = LogScale(config)

    def forward(self, x, position_ids):
        x = self.l2_clip(x.to(torch.float32) * self._cached_cast("scale", torch.float32))
        return self.log_scale(x, position_ids)


def softcap(x, value):
    return torch.tanh(x / value) * value


# ------------------------------------------------------- attention components

class RelativePositionBias(_CachedCast, nn.Module):
    def __init__(self, config, num_heads, rel_extent, log_scaling, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.rel_extent = rel_extent
        self.linear = nn.Linear(config.hidden_size, num_heads * config.d_rel, bias=False, device=device, dtype=dtype)
        self.norm = (LogScaledL2Clip(config, num_heads, config.d_rel, config.rel_l2_clip_threshold) if log_scaling
                     else ScaledL2Clip(num_heads, config.d_rel, config.rel_l2_clip_threshold))
        self.proj = nn.Parameter(torch.empty(config.d_rel, rel_extent))

    def precompute(self, hidden_states, position_ids):
        """Query-side relative projection states [B, S, H, d_rel] for blockwise bias."""
        relative_states = self.linear(hidden_states).view(*hidden_states.shape[:-1], self.num_heads, -1)
        return self.norm(relative_states, position_ids)

    def bias(self, relative_states, q_start, q_end, kv_start, kv_end):
        """Relative bias [B, H, q_end - q_start, kv_end - kv_start] for one query block."""
        q_pos = torch.arange(q_start, q_end, device=relative_states.device)
        kv_pos = torch.arange(kv_start, kv_end, device=relative_states.device)
        relative_distance = q_pos[:, None] - kv_pos[None, :]
        rel_logits = (relative_states[:, q_start:q_end].to(torch.bfloat16) @
                      self._cached_cast("proj", torch.bfloat16)).to(torch.float32).transpose(1, 2)
        gather_index = relative_distance.clamp(0, self.rel_extent - 1).expand(*rel_logits.shape[:2], -1, -1)
        position_bias = rel_logits.gather(-1, gather_index)
        return position_bias.masked_fill((relative_distance < 0) | (relative_distance >= self.rel_extent), 0.0)

    def forward(self, hidden_states, relative_distance, position_ids):
        relative_states = self.precompute(hidden_states, position_ids)
        rel_logits = (relative_states.to(torch.bfloat16) @ self._cached_cast("proj", torch.bfloat16)).to(torch.float32).transpose(1, 2)
        gather_index = relative_distance.clamp(0, self.rel_extent - 1).expand(*rel_logits.shape[:2], -1, -1)
        position_bias = rel_logits.gather(-1, gather_index)
        return position_bias.masked_fill((relative_distance < 0) | (relative_distance >= self.rel_extent), 0.0)


class Inkling2CacheLayer:
    """Decode state for one layer application: KV cache plus short-convolution ring buffers."""

    def __init__(self, batch, num_kv_heads, head_dim, cap, device, dtype):
        self.k = torch.zeros(batch, num_kv_heads, cap, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros(batch, num_kv_heads, cap, head_dim, device=device, dtype=dtype)
        self.pos = torch.zeros((), device=device, dtype=torch.long)  # absolute position of the next token
        self.conv = {}  # name -> [batch, K, C] fp32 ring of recent conv input states
        # Decode-time scratch, allocated once per slot and refreshed in place every token.
        self.idx = torch.arange(cap, device=device)  # key slot positions
        self.mask = torch.empty(1, 1, 1, cap, device=device, dtype=torch.float32)


def _conv_tail(x, k):
    """Last k-1 rows of x (zero-padded on the left when x is shorter)."""
    n = k - 1
    if x.shape[1] >= n:
        return x[:, -n:].to(torch.float32).contiguous()
    pad = x.new_zeros(x.shape[0], n - x.shape[1], x.shape[2])
    return torch.cat([pad, x], dim=1).to(torch.float32)


class ShortConvolution(nn.Module):
    """Static depthwise causal short convolution (checkpoint stores a [C, 1, K] weight)."""

    def __init__(self, hidden_size, conv_kernel_size):
        super().__init__()
        self.conv_kernel_size = conv_kernel_size
        self.weight = nn.Parameter(torch.empty(hidden_size, 1, conv_kernel_size))

    def forward(self, x, state=None, name=None):
        input_dtype = x.dtype
        k = self.conv_kernel_size
        x = x.to(torch.float32)
        residual = x
        # Only take the decode path once a ring exists; a single-token prefill on a fresh
        # cache must go through the multi-token path to create it.
        if state is not None and x.shape[1] == 1 and name in state.conv:
            ring = state.conv[name]  # [B, K, C]
            ring.index_copy_(1, (state.pos % k).view(1), x)
            idx = (state.pos - (k - 1) + torch.arange(k, device=x.device)) % k  # ctx[j] = x[t-(K-1)+j]
            ctx = torch.index_select(ring, 1, idx)
            w = self.weight.to(torch.float32).reshape(-1, k)  # [C, K], F.conv1d correlation order
            out = (ctx * w.t().unsqueeze(0)).sum(1)[:, None, :]
            return ((out + residual) / math.sqrt(k + 1)).to(input_dtype)
        out = F.conv1d(x.transpose(1, 2), self.weight.to(torch.float32), padding=k - 1, groups=x.shape[-1])
        out = out.transpose(1, 2)[:, : x.shape[1]]
        if state is not None:
            if name not in state.conv:
                state.conv[name] = torch.zeros(x.shape[0], k, x.shape[2], device=x.device, dtype=torch.float32)
            ring = state.conv[name]
            pos0 = int(state.pos)
            L = x.shape[1]
            for j in range(k):
                p = pos0 + L - 1 - j
                if p >= 0:
                    ring[:, p % k] = x[:, L - 1 - j]
        return ((out + residual) / math.sqrt(k + 1)).to(input_dtype)


class DynamicShortConvolution(nn.Module):
    """Dynamic conv for k/v in full-attention layers: static taps + per-token taps from the layer input."""

    def __init__(self, hidden_size, input_size, num_heads, conv_kernel_size, device=None, dtype=None):
        super().__init__()
        self.conv_kernel_size = conv_kernel_size
        self.num_heads = num_heads
        self.static_w = nn.Parameter(torch.zeros(conv_kernel_size, hidden_size))
        self.weight_proj = nn.Linear(input_size, num_heads * conv_kernel_size, bias=False, device=device, dtype=dtype)

    def forward(self, x, dynamic_input, state=None, name=None):
        input_dtype = x.dtype
        batch_size, seq_len, hidden_size = x.shape
        k = self.conv_kernel_size
        dynamic_weights = self.weight_proj(dynamic_input).to(torch.float32)
        dynamic_weights = dynamic_weights.view(batch_size, seq_len, self.num_heads, 1, k)
        weights = self.static_w.to(torch.float32).T.view(self.num_heads, -1, k) + dynamic_weights

        x = x.to(torch.float32)
        residual = x
        if state is not None and seq_len == 1 and name in state.conv:
            ring = state.conv[name]  # [B, K, C]
            ring.index_copy_(1, (state.pos % k).view(1), x)
            idx = (state.pos - torch.arange(k, device=x.device)) % k  # ctx[j] = x[t-j]
            ctx = torch.index_select(ring, 1, idx)  # [B, K, C]
            win = ctx.reshape(batch_size, k, self.num_heads, -1).permute(0, 2, 3, 1).unsqueeze(1)
            out = (win * weights).sum(-1).reshape(batch_size, 1, hidden_size)
            return ((out + residual) / math.sqrt(2 * k + 1)).to(input_dtype)
        context = F.pad(x.transpose(1, 2), (k - 1, 0))
        windows = context.unfold(-1, k, 1).flip(-1).transpose(1, 2)  # [..., lag] = token lag steps back
        windows = windows.reshape(batch_size, seq_len, self.num_heads, -1, k)
        out = (windows * weights).sum(-1).reshape(batch_size, seq_len, hidden_size)
        if state is not None:
            if name not in state.conv:
                state.conv[name] = torch.zeros(batch_size, k, hidden_size, device=x.device, dtype=torch.float32)
            ring = state.conv[name]
            pos0 = int(state.pos)
            for j in range(k):
                p = pos0 + seq_len - 1 - j
                if p >= 0:
                    ring[:, p % k] = x[:, seq_len - 1 - j]
        return ((out + residual) / math.sqrt(2 * k + 1)).to(input_dtype)


class QueryProjection(nn.Module):
    def __init__(self, config, num_heads, head_dim, log_scaling, device=None, dtype=None):
        super().__init__()
        self.head_dim = head_dim
        self.linear = nn.Linear(config.hidden_size, num_heads * head_dim, bias=False, device=device, dtype=dtype)
        self.norm = (LogScaledL2Clip(config, num_heads, head_dim, config.qk_l2_clip_threshold) if log_scaling
                     else ScaledL2Clip(num_heads, head_dim, config.qk_l2_clip_threshold))

    def forward(self, x, position_ids):
        states = self.linear(x)
        states = states.view(*states.shape[:-1], -1, self.head_dim)
        return self.norm(states, position_ids).transpose(1, 2)


class KeyValueProjection(nn.Module):
    def __init__(self, config, num_heads, head_dim, is_key, dynamic, device=None, dtype=None):
        super().__init__()
        self.head_dim = head_dim
        self.linear = nn.Linear(config.hidden_size, num_heads * head_dim, bias=False, device=device, dtype=dtype)
        if dynamic:
            self.sconv = DynamicShortConvolution(num_heads * head_dim, config.hidden_size, num_heads,
                                                 config.conv_kernel_size, device=device, dtype=dtype)
        else:
            self.sconv = ShortConvolution(num_heads * head_dim, config.conv_kernel_size)
        self.norm = L2Clip(config.qk_l2_clip_threshold) if is_key else UnweightedRMSNorm(config.rms_norm_eps)

    def forward(self, x, dynamic_input=None, state=None, name=None):
        states = self.linear(x)
        if isinstance(self.sconv, DynamicShortConvolution):
            states = self.sconv(states, x, state=state, name=name)
        else:
            states = self.sconv(states, state=state, name=name)
        states = states.view(*states.shape[:-1], -1, self.head_dim)
        return self.norm(states).transpose(1, 2)


class GatedOutputProjection(nn.Module):
    def __init__(self, config, num_heads, head_dim, device=None, dtype=None):
        super().__init__()
        self.gate_attn = nn.Linear(config.hidden_size, num_heads * head_dim, bias=False, device=device, dtype=dtype)
        self.linear = nn.Linear(num_heads * head_dim, config.hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, attn_output, x):
        gate = torch.sigmoid(self.gate_attn(x).to(torch.float32))
        attn_output = (attn_output.to(torch.float32) * gate).to(attn_output.dtype)
        return self.linear(attn_output)


def inkling_attention(query_states, key_states, value_states, attention_mask, scaling, sinks,
                      position_bias=None, sliding_window=None, position_bias_fn=None,
                      max_score_elements=64 * 1024 * 1024):
    """Eager attention with optional sinks / position bias / sliding window, softmax in fp32.

    Queries are processed in blocks sized to max_score_elements, and each block only keeps
    its relevant keys, so a long prefill never materializes a full query-by-key score
    tensor. position_bias_fn(q_start, q_end, kv_start, kv_end) computes the relative
    position bias for just the current block.
    """
    batch, num_heads, q_len, head_dim = query_states.shape
    kv_len = key_states.shape[2]
    num_key_value_groups = num_heads // key_states.shape[1]

    if sliding_window is not None:
        block = min(q_len, 1024)
    else:
        block = max(1, min(q_len, max_score_elements // max(1, batch * num_heads * kv_len)))

    outputs = []
    for q_start in range(0, q_len, block):
        q_end = min(q_start + block, q_len)
        kv_start, kv_end = 0, kv_len
        if sliding_window is not None:
            # Keys outside the window of every query in this block are fully masked.
            kv_start = max(0, q_start - (sliding_window - 1))
            kv_end = min(kv_len, q_end + sliding_window - 1)
        q = query_states[:, :, q_start:q_end]
        k = key_states[:, :, kv_start:kv_end]
        v = value_states[:, :, kv_start:kv_end]
        if num_key_value_groups > 1:
            k = k[:, :, None].expand(batch, -1, num_key_value_groups, -1, head_dim).reshape(batch, -1, k.shape[2], head_dim)
            v = v[:, :, None].expand(batch, -1, num_key_value_groups, -1, head_dim).reshape(batch, -1, v.shape[2], head_dim)
        attn_weights = torch.matmul(q.to(torch.float32), k.to(torch.float32).transpose(2, 3)) * scaling
        if position_bias_fn is not None:
            attn_weights = attn_weights + position_bias_fn(q_start, q_end, kv_start, kv_end).to(attn_weights.dtype)
        elif position_bias is not None:
            attn_weights = attn_weights + position_bias[:, :, q_start:q_end, kv_start:kv_end].to(attn_weights.dtype)
        if callable(attention_mask):
            # Built for this block only; no full sequence-by-sequence mask is materialized.
            attn_weights = attn_weights + attention_mask(q_start, q_end, kv_start, kv_end)
        elif attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, q_start:q_end, kv_start:kv_end]
        if sliding_window is not None:
            q_offset = kv_len - q_len
            q_pos = torch.arange(q_offset + q_start, q_offset + q_end, device=attn_weights.device)
            kv_pos = torch.arange(kv_start, kv_end, device=attn_weights.device)
            dist = kv_pos[None, :] - q_pos[:, None]
            attn_weights = attn_weights.masked_fill((dist > sliding_window - 1) | (dist < -(sliding_window - 1)), float("-inf"))
        if sinks is not None:
            sink_block = sinks[:, :, q_start:q_end] if sinks.shape[2] == q_len else sinks
            combined = torch.cat([attn_weights, sink_block.to(attn_weights.dtype)], dim=-1)
            attn_weights = F.softmax(combined, dim=-1)[..., :-1]
        else:
            attn_weights = F.softmax(attn_weights, dim=-1)
        outputs.append(torch.matmul(attn_weights.to(v.dtype), v))
    attn_output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=2)
    return attn_output.transpose(1, 2).contiguous()


# --------------------------------------------------------------- MoE block

def softcapped_swiglu(gate, up, gate_softcap, up_softcap):
    gate = gate.to(torch.float32)
    up = up.to(torch.float32)
    return (softcap(gate, gate_softcap) * torch.sigmoid(gate) * softcap(up, up_softcap)).to(torch.bfloat16)


def _batched_bank_weights(module, ids):
    """Dequantize the selected experts of a 3D quantized bank in one call.

    The per-expert block scales are stored in the cuBLAS tiled layout, whose 128-row
    tiles are independent, so concatenating the selected experts' scale tiles along
    the row dim yields exactly the tiled layout of the flattened expert bank (valid
    when out_features is a multiple of 128). Per-expert tensor scales cannot ride
    along in the kernel, so they are returned separately for the caller to apply.

    Returns (weights [num_sel, out, in] in compute dtype, expert_scale [num_sel] fp32 or None),
    or (None, None) when the bank layout is not supported by this path.
    """
    import dataclasses
    from comfy.ops import QuantizedTensor
    wt = module.weight
    if not isinstance(wt, QuantizedTensor):
        return None, None
    q = wt._qdata
    if getattr(q, "ndim", 0) != 3:
        return None, None
    out_f = module.out_features
    packed = q.dtype == torch.uint8
    if out_f % 128 or q.shape[1] != out_f or q.shape[2] * (2 if packed else 1) != module.in_features:
        return None, None
    params = wt._params
    num_sel = ids.shape[0]
    flat = {}
    expert_scale = None
    for f in dataclasses.fields(params):
        v = getattr(params, f.name)
        if torch.is_tensor(v) and v.ndim == 3 and v.shape[0] == q.shape[0]:
            flat[f.name] = torch.index_select(v, 0, ids).reshape(-1, v.shape[-1])
        elif torch.is_tensor(v) and v.ndim == 1 and v.shape[0] == q.shape[0]:
            expert_scale = torch.index_select(v, 0, ids).to(torch.float32)
        else:
            flat[f.name] = v
    if expert_scale is not None:
        flat["scale"] = torch.ones((), device=q.device, dtype=torch.float32)
    flat["orig_shape"] = (num_sel * out_f, module.in_features)
    qflat = torch.index_select(q, 0, ids).reshape(num_sel * out_f, q.shape[2])
    w = QuantizedTensor(qflat, wt._layout_cls, type(params)(**flat)).dequantize()
    return w.view(num_sel, out_f, module.in_features), expert_scale


def _per_expert_bank_weights(module, ids):
    """Fallback: dequantize selected experts one at a time (any 3D block-scaled bank)."""
    import dataclasses
    from comfy.ops import QuantizedTensor
    wt = module.weight
    if not isinstance(wt, QuantizedTensor) or getattr(wt._qdata, "ndim", 0) != 3:
        return None
    params = wt._params
    per = {}
    for f in dataclasses.fields(params):
        v = getattr(params, f.name)
        if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == wt._qdata.shape[0]:
            per[f.name] = torch.index_select(v, 0, ids)
        else:
            per[f.name] = v
    qds = torch.index_select(wt._qdata, 0, ids)
    outs = []
    for j in range(qds.shape[0]):
        kw = {}
        for f in dataclasses.fields(params):
            v = per[f.name]
            kw[f.name] = v[j] if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == qds.shape[0] else v
        kw["orig_shape"] = (module.out_features, module.in_features)
        outs.append(QuantizedTensor(qds[j], wt._layout_cls, type(params)(**kw)).dequantize())
    return torch.stack(outs)


class Inkling2Experts(nn.Module):
    """MoE experts bank. With comfy mixed ops this is two quantized MoEExperts containers
    (w13 / w2) backed by the checkpoint's NVFP4/MXFP8 comfy_quant tensors."""

    def __init__(self, num_experts, hidden_dim, inter_dim, device=None, dtype=None, operations=None):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.inter_dim = inter_dim
        moe_cls = getattr(operations, "MoEExperts", None)
        if moe_cls is not None:
            self.w13 = moe_cls(num_experts, hidden_dim, 2 * inter_dim, bias=False, device=device, dtype=dtype)
            self.w2 = moe_cls(num_experts, inter_dim, hidden_dim, bias=False, device=device, dtype=dtype)
            self._quantized = True
        else:
            self.w13 = nn.Parameter(torch.empty(num_experts, 2 * inter_dim, hidden_dim))
            self.w2 = nn.Parameter(torch.empty(num_experts, hidden_dim, inter_dim))
            self._quantized = False

    def forward(self, hidden_states, top_k_index, top_k_weights, gate_softcap, up_softcap):
        if hidden_states.shape[0] == 1:
            return self._forward_single(hidden_states, top_k_index, top_k_weights, gate_softcap, up_softcap)
        final_hidden_states = torch.zeros_like(hidden_states, dtype=torch.float32)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        import contextlib
        if self._quantized:
            ctx13 = self.w13.bank_resident(hidden_states)
            ctx2 = self.w2.bank_resident(hidden_states)
        else:
            ctx13, ctx2 = contextlib.nullcontext(), contextlib.nullcontext()
        with ctx13, ctx2:
            for expert_idx in expert_hit:
                expert_idx = expert_idx[0]
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                if self._quantized:
                    gate_up = self.w13.expert_linear(current_state, expert_idx)
                else:
                    gate_up = F.linear(current_state, self.w13[expert_idx])
                gate = gate_up[..., : self.inter_dim]
                up = gate_up[..., self.inter_dim:]
                current_hidden_states = softcapped_swiglu(gate, up, gate_softcap, up_softcap)
                if self._quantized:
                    current_hidden_states = self.w2.expert_linear(current_hidden_states, expert_idx)
                else:
                    current_hidden_states = F.linear(current_hidden_states, self.w2[expert_idx])
                current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(torch.float32))
        return final_hidden_states.to(hidden_states.dtype)
    def _forward_single(self, hidden_states, top_k_index, top_k_weights, gate_softcap, up_softcap):
        if not self._quantized:
            return self._forward_single_banked(hidden_states, top_k_index, top_k_weights, gate_softcap, up_softcap)
        # Match the multi-token path: both banks must be resident on the compute device
        # before their weights and scales are gathered for decode.
        with self.w13.bank_resident(hidden_states), self.w2.bank_resident(hidden_states):
            return self._forward_single_banked(hidden_states, top_k_index, top_k_weights, gate_softcap, up_softcap)

    def _forward_single_banked(self, hidden_states, top_k_index, top_k_weights, gate_softcap, up_softcap):
        # Sync-free single-token path: gather the routed experts with tensor indexing, dequantize
        # them to compute dtype, then one batched matmul pair. No host syncs -> CUDA-graph safe.
        ids = top_k_index[0]
        weights = top_k_weights[0]
        x = hidden_states[0]
        num_sel = ids.shape[0]

        scale13 = scale2 = None
        if self._quantized:
            w13, scale13 = _batched_bank_weights(self.w13, ids)
            w2, scale2 = _batched_bank_weights(self.w2, ids)
            if w13 is None or w2 is None:  # layout without concatenable scale tiles: per-expert dequant
                w13 = _per_expert_bank_weights(self.w13, ids)
                w2 = _per_expert_bank_weights(self.w2, ids)
                scale13 = scale2 = None
        else:
            w13 = self.w13.index_select(0, ids)
            w2 = self.w2.index_select(0, ids)
        if w13 is None or w2 is None:
            expert_ids = ids.tolist()  # fallback: per-expert loop with one sync
            out = torch.zeros(self.hidden_dim, device=x.device, dtype=torch.float32)
            for j, e in enumerate(expert_ids):
                e = int(e)
                if self._quantized:
                    gate_up = self.w13.expert_linear(x.unsqueeze(0), e)
                else:
                    gate_up = F.linear(x, self.w13[e])
                h = softcapped_swiglu(gate_up[..., : self.inter_dim], gate_up[..., self.inter_dim:],
                                      gate_softcap, up_softcap)
                if self._quantized:
                    h = self.w2.expert_linear(h, e)
                else:
                    h = F.linear(h, self.w2[e])
                out += h[0].to(torch.float32) * weights[j]
            return out.to(hidden_states.dtype).unsqueeze(0)

        xb = x.to(w13.dtype).unsqueeze(0).expand(num_sel, 1, -1)
        gate_up = torch.bmm(xb, w13.transpose(1, 2))  # [num_sel, 1, 2*inter]
        if scale13 is not None:
            gate_up = gate_up.to(torch.float32) * scale13[:, None, None]
        h = softcapped_swiglu(gate_up[..., : self.inter_dim], gate_up[..., self.inter_dim:],
                              gate_softcap, up_softcap)
        h = torch.bmm(h, w2.transpose(1, 2))  # [num_sel, 1, hidden]
        wsel = weights.to(torch.float32)
        if scale2 is not None:
            wsel = wsel * scale2
        out = (h[:, 0, :].to(torch.float32) * wsel[:, None]).sum(0)
        return out.to(hidden_states.dtype).unsqueeze(0)


class TopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(self.num_experts, dtype=torch.float32), requires_grad=False)
        self.routed_scaling_factor = config.route_scale

    def forward(self, hidden_states):
        hidden_states = hidden_states.to(torch.float32)
        logits = F.linear(hidden_states, self.weight)
        scores = torch.sigmoid(logits)
        indices = torch.topk(scores + self.bias, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices


class SparseMoeBlock(nn.Module):
    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        self.experts = Inkling2Experts(config.n_routed_experts, config.hidden_size, config.intermediate_size,
                                       device=device, dtype=dtype, operations=operations)
        self.gate = TopKRouter(config)
        self.gate_softcap = config.mlp_gate_softcap
        self.up_softcap = config.mlp_up_softcap

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat = hidden_states.view(-1, hidden_dim)
        _, routing_weights, selected_experts = self.gate(flat)
        out = self.experts(flat, selected_experts, routing_weights, self.gate_softcap, self.up_softcap)
        return out.reshape(batch_size, sequence_length, hidden_dim)


# --------------------------------------------------- loop message machinery

def _unweighted_norm(x, eps=1e-6):
    x = x.to(torch.float32)
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def _query_kv_attention(query_states, stacked_states, hidden_size):
    """query_states: [B, S, Nq, D]; stacked_states: [B, S, K, D] (fp32 in).
    The sequence dim acts as the head dim: each position's queries attend over that
    position's block states. Returns [B, Nq, S, D]."""
    stacked16 = stacked_states.to(torch.bfloat16)
    attn = (query_states.to(torch.bfloat16) @ stacked16.transpose(-1, -2)).to(torch.float32)
    attn = F.softmax(attn * hidden_size ** -0.5, dim=-1).to(torch.bfloat16)
    out = (attn @ stacked16).to(torch.float32)
    return out  # [B, S, Nq, D]; callers unbind(-2) into Nq x [B, S, D]


class MessageGate(nn.Module):
    """Mixes hidden states and message as (1 - forget) * hidden + message_gate * message."""

    def __init__(self, config, device=None, dtype=None):
        super().__init__()
        r = config.loop_gate_rank
        self.down_weight = nn.Parameter(torch.empty(r, 2 * config.hidden_size, device=device, dtype=dtype))
        self.forget_weight = nn.Parameter(torch.empty(config.hidden_size, r, dtype=torch.float32))
        self.gain_weight = nn.Parameter(torch.empty(config.hidden_size, r, dtype=torch.float32))

    def forward(self, hidden_states, message):
        hidden_states = hidden_states.to(torch.float32)
        message = message.to(torch.float32)
        gate_hidden_states = torch.cat([_unweighted_norm(hidden_states), _unweighted_norm(message)], dim=-1)
        gate_hidden_states = F.linear(gate_hidden_states.to(self.down_weight.dtype), self.down_weight).to(torch.float32)
        forget_gate = torch.tanh(F.linear(gate_hidden_states, self.forget_weight))
        message_gate = torch.tanh(F.linear(gate_hidden_states, self.gain_weight))
        return (1 - forget_gate) * hidden_states + message_gate * message


class BlockAttentionResidual(nn.Module):
    def __init__(self, config, num_queries=1):
        super().__init__()
        shape = (config.hidden_size,) if num_queries == 1 else (num_queries, config.hidden_size)
        self.query = nn.Parameter(torch.zeros(*shape, dtype=torch.float32))
        self.num_queries = num_queries

    def forward(self, block_states, partial_block_states):
        stacked_states = _unweighted_norm(torch.cat([block_states, partial_block_states.unsqueeze(-2)], dim=-2))
        hidden_size = stacked_states.shape[-1]
        query_states = self.query.to(torch.float32).reshape(-1, hidden_size).expand(*stacked_states.shape[:-2], -1, -1)
        out = _query_kv_attention(query_states, stacked_states, hidden_size)
        return out.unbind(-2)


class MessageAttentionResidual(nn.Module):
    """Retrieves the (q, k, v, mlp) messages of each loop layer from the previous pass."""

    def __init__(self, config):
        super().__init__()
        num_loop_layers = config.loop_end - config.loop_start
        self.query = nn.Parameter(torch.zeros(num_loop_layers, 4, config.hidden_size, dtype=torch.float32))

    def forward(self, block_states, partial_block_states, loop_layer_idx):
        stacked_states = _unweighted_norm(torch.cat([block_states, partial_block_states.unsqueeze(-2)], dim=-2))
        hidden_size = stacked_states.shape[-1]
        query_states = self.query[loop_layer_idx].to(torch.float32).expand(*stacked_states.shape[:-2], -1, -1)
        return _query_kv_attention(query_states, stacked_states, hidden_size)


# ------------------------------------------------------------ decoder layers

class Inkling2Attention(nn.Module):
    def __init__(self, config, layer_idx, device=None, dtype=None):
        super().__init__()
        is_sliding = config.layer_types[layer_idx] == "hybrid_sliding"
        self.head_dim = config.swa_head_dim if is_sliding else config.head_dim
        self.num_heads = config.swa_num_attention_heads if is_sliding else config.num_attention_heads
        self.num_key_value_heads = config.swa_num_key_value_heads if is_sliding else config.num_key_value_heads
        self.sliding_window = config.sliding_window_size if is_sliding else None
        rel_extent = config.sliding_window_size if is_sliding else config.rel_extent
        log_scaling = not is_sliding
        self.scaling = self.head_dim ** -0.5
        self.q_proj = QueryProjection(config, self.num_heads, self.head_dim, log_scaling, device=device, dtype=dtype)
        self.k_proj = KeyValueProjection(config, self.num_key_value_heads, self.head_dim, is_key=True, dynamic=log_scaling, device=device, dtype=dtype)
        self.v_proj = KeyValueProjection(config, self.num_key_value_heads, self.head_dim, is_key=False, dynamic=log_scaling, device=device, dtype=dtype)
        self.o_proj = GatedOutputProjection(config, self.num_heads, self.head_dim, device=device, dtype=dtype)
        self.attn_sink_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False, device=device, dtype=dtype)
        self.rel_logits_proj = RelativePositionBias(config, self.num_heads, rel_extent, log_scaling, device=device, dtype=dtype)
        if device is not None:
            self.to(device)

    def position_bias_fn(self, hidden_states, position_ids):
        """Blockwise relative bias provider for inkling_attention prefill chunks."""
        relative_states = self.rel_logits_proj.precompute(hidden_states, position_ids)
        return lambda q_start, q_end, kv_start, kv_end: self.rel_logits_proj.bias(
            relative_states, q_start, q_end, kv_start, kv_end)

    def forward(self, hidden_states, key_hidden_states, value_hidden_states, attention_mask,
                relative_distance, position_ids, past_key_value=None):
        query_states = self.q_proj(hidden_states, position_ids)
        key_states = self.k_proj(key_hidden_states, state=past_key_value, name="k")
        value_states = self.v_proj(value_hidden_states, state=past_key_value, name="v")
        sinks = self.attn_sink_proj(hidden_states).transpose(1, 2).unsqueeze(-1)
        if past_key_value is not None:
            if key_states.shape[2] == 1:
                pv = past_key_value
                pv.k.index_copy_(2, pv.pos.view(1), key_states)
                pv.v.index_copy_(2, pv.pos.view(1), value_states)
                cap = pv.k.shape[2]
                key_all = pv.k.to(query_states.dtype)
                value_all = pv.v.to(query_states.dtype)
                idx = pv.idx
                dist = (pv.pos - idx).unsqueeze(0)  # [1, cap], query minus key position
                position_bias = self.rel_logits_proj(hidden_states, dist, position_ids)
                valid = idx <= pv.pos
                if self.sliding_window is not None:
                    valid = valid & (dist[0] <= self.sliding_window - 1) & (dist[0] >= 1 - self.sliding_window)
                dec_mask = pv.mask
                dec_mask.zero_()
                dec_mask.masked_fill_(~valid.view(1, 1, 1, cap), float("-inf"))
                attn_output = inkling_attention(query_states, key_all, value_all, dec_mask, self.scaling,
                                                sinks, position_bias=position_bias, sliding_window=None)
            else:
                past_key_value.k[:, :, :key_states.shape[2]].copy_(key_states)
                past_key_value.v[:, :, :value_states.shape[2]].copy_(value_states)
                attn_output = inkling_attention(query_states, key_states, value_states, attention_mask, self.scaling,
                                                sinks, position_bias_fn=self.position_bias_fn(hidden_states, position_ids),
                                                sliding_window=self.sliding_window)
        else:
            attn_output = inkling_attention(query_states, key_states, value_states, attention_mask, self.scaling,
                                            sinks, position_bias_fn=self.position_bias_fn(hidden_states, position_ids),
                                            sliding_window=self.sliding_window)
        attn_output = attn_output.reshape(hidden_states.shape[0], hidden_states.shape[1], -1)
        return self.o_proj(attn_output, hidden_states)


class Inkling2DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx, is_loop, device=None, dtype=None, operations=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_block_boundary = layer_idx % config.n_layers_per_attn_res_block == 0
        self.is_loop = is_loop
        num_queries = 3 if (not config.layer_types[layer_idx] == "hybrid_sliding") else 1
        if layer_idx > 0:
            self.attention_input = BlockAttentionResidual(config, num_queries)
        else:
            self.attention_input = None
        self.mlp_input = BlockAttentionResidual(config)
        self.self_attn = Inkling2Attention(config, layer_idx, device=device, dtype=dtype)
        self.mlp = SparseMoeBlock(config, device=device, dtype=dtype, operations=operations)
        is_sliding = config.layer_types[layer_idx] == "hybrid_sliding"
        if is_sliding:
            self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        else:
            self.q_input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.k_input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.v_input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn_sconv = ShortConvolution(config.hidden_size, config.conv_kernel_size)
        self.mlp_sconv = ShortConvolution(config.hidden_size, config.conv_kernel_size)
        if is_loop:
            self.controllers = nn.Module()
            for gate_name in ("attn", "k", "v", "mlp"):
                setattr(self.controllers, gate_name, MessageGate(config, device=device, dtype=dtype))

    def forward(self, hidden_states, block_states, messages, attention_mask, relative_distance, position_ids,
                past_key_value=None):
        dtype = hidden_states.dtype
        is_sliding = self.self_attn.sliding_window is not None
        block_residual = hidden_states

        if self.attention_input is not None:
            qkv_in = self.attention_input(block_states, hidden_states)
        else:
            qkv_in = hidden_states

        gated = self.is_loop and messages is not None
        if gated:
            query_message, key_message, value_message, mlp_message = messages.unbind(-2)
            if is_sliding:
                base = qkv_in[0]
                q_in = self.controllers.attn(base, query_message)
                k_in = self.controllers.k(base, key_message)
                v_in = self.controllers.v(base, value_message)
            else:
                q_in, k_in, v_in = qkv_in
                q_in = self.controllers.attn(q_in, query_message)
                k_in = self.controllers.k(k_in, key_message)
                v_in = self.controllers.v(v_in, value_message)
        else:
            if is_sliding:
                base = qkv_in[0] if self.attention_input is not None else qkv_in
                q_in = k_in = v_in = base
            else:
                if self.attention_input is not None:
                    q_in, k_in, v_in = qkv_in
                else:
                    q_in = k_in = v_in = qkv_in

        if is_sliding:
            if gated:
                query_hidden = self.attn_norm(q_in).to(dtype)
                key_hidden = self.attn_norm(k_in).to(dtype)
                value_hidden = self.attn_norm(v_in).to(dtype)
            else:
                query_hidden = key_hidden = value_hidden = self.attn_norm(q_in).to(dtype)
        else:
            query_hidden = self.q_input_norm(q_in).to(dtype)
            key_hidden = self.k_input_norm(k_in).to(dtype)
            value_hidden = self.v_input_norm(v_in).to(dtype)

        hidden_states = self.self_attn(query_hidden, key_hidden, value_hidden, attention_mask,
                                       relative_distance, position_ids, past_key_value=past_key_value)
        hidden_states = self.attn_sconv(hidden_states, state=past_key_value, name="attn")

        residual = hidden_states = hidden_states.to(torch.float32)
        if self.is_block_boundary:
            block_states = torch.cat([block_states, block_residual.unsqueeze(-2).to(torch.float32)], dim=-2)
        else:
            residual = hidden_states = block_residual.to(torch.float32) + hidden_states

        hidden_states = self.mlp_input(block_states, hidden_states)[0]
        if gated:
            hidden_states = self.controllers.mlp(hidden_states, mlp_message)
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states).to(dtype))
        hidden_states = self.mlp_sconv(hidden_states, state=past_key_value, name="mlp")
        return (residual + hidden_states).to(dtype)


# ------------------------------------------------------------- text model

class Inkling2Embedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size, device=device, dtype=dtype))

    def forward(self, input_ids, out_dtype=None):
        x = F.embedding(input_ids, self.weight)
        if out_dtype is not None:
            x = x.to(out_dtype)
        return x


def message_query_attention(previous_block_states, previous_hidden, message_queries, loop_layer_idx):
    stacked_states = _unweighted_norm(torch.cat([previous_block_states, previous_hidden.unsqueeze(-2)], dim=-2))
    hidden_size = stacked_states.shape[-1]
    query_states = message_queries[loop_layer_idx].to(torch.float32).expand(*stacked_states.shape[:-2], -1, -1)
    return _query_kv_attention(query_states, stacked_states, hidden_size)


class Inkling2TextModel(nn.Module):
    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        self.config = config
        self.embed = Inkling2Embedding(config.vocab_size, config.hidden_size, device=device, dtype=dtype)
        self.embed_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)
        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            is_loop = config.loop_start <= layer_idx < config.loop_end
            self.layers.append(Inkling2DecoderLayer(config, layer_idx, is_loop, device=device, dtype=dtype,
                                                    operations=operations))
        self.attn_res = nn.Module()
        self.attn_res.output_query = nn.Parameter(torch.zeros(config.hidden_size, dtype=torch.float32))
        num_loop = config.loop_end - config.loop_start
        self.middle_cycle_cross_pass_queries = nn.Parameter(
            torch.zeros(2 * num_loop, config.hidden_size, dtype=torch.float32))
        self.middle_cycle_cross_pass_kv_queries = nn.Parameter(
            torch.zeros(2 * num_loop, config.hidden_size, dtype=torch.float32))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if device is not None:
            self.to(device)

    def init_kv_cache(self, batch, cap, device, dtype):
        cfg = self.config

        def make(layer_idx):
            is_sw = cfg.layer_types[layer_idx] == "hybrid_sliding"
            hd = cfg.swa_head_dim if is_sw else cfg.head_dim
            nh = cfg.swa_num_key_value_heads if is_sw else cfg.num_key_value_heads
            return Inkling2CacheLayer(batch, nh, hd, cap, device, dtype)

        slots = [make(i) for i in range(cfg.num_hidden_layers)]
        for _ in range(1, cfg.loop_num_passes):
            slots += [make(i) for i in range(cfg.loop_start, cfg.loop_end)]
        return slots

    def cache_slot(self, layer_idx, pass_idx):
        cfg = self.config
        if pass_idx == 0 or not (cfg.loop_start <= layer_idx < cfg.loop_end):
            return layer_idx
        return cfg.num_hidden_layers + (pass_idx - 1) * (cfg.loop_end - cfg.loop_start) + (layer_idx - cfg.loop_start)

    def forward(self, embeds, attention_mask=None, past_key_values=None, intermediate_output=None,
                final_layer_norm_intermediate=False):
        cfg = self.config
        batch, seq_len, _ = embeds.shape
        device = embeds.device
        scalar_tap = None
        list_taps = None
        capture_all = intermediate_output == "all"
        if isinstance(intermediate_output, int):
            scalar_tap = _normalize_layer_tap(intermediate_output, cfg.num_hidden_layers)
        elif isinstance(intermediate_output, list):
            list_taps = [_normalize_layer_tap(t, cfg.num_hidden_layers) for t in intermediate_output]
        tap_out = {}
        if past_key_values is not None and seq_len == 1:
            position_ids = past_key_values[0].pos.reshape(1, 1)
        else:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
        # Relative bias is computed per query block inside attention; do not build a
        # full query-by-key distance matrix here.
        relative_distance = None

        if attention_mask is None:
            def block_mask(q_start, q_end, kv_start, kv_end):
                q_pos = torch.arange(q_start, q_end, device=device)
                kv_pos = torch.arange(kv_start, kv_end, device=device)
                causal = kv_pos[None, :] <= q_pos[:, None]
                out = torch.zeros(q_end - q_start, kv_end - kv_start, device=device, dtype=torch.float32)
                out.masked_fill_(~causal, float("-inf"))
                return out[None, None]
        else:
            pad = attention_mask.to(device)

            def block_mask(q_start, q_end, kv_start, kv_end):
                q_pos = torch.arange(q_start, q_end, device=device)
                kv_pos = torch.arange(kv_start, kv_end, device=device)
                causal = kv_pos[None, :] <= q_pos[:, None]
                allowed = ((pad[:, q_start:q_end] > 0)[:, :, None] &
                           (pad[:, kv_start:kv_end] > 0)[:, None, :] & causal[None])
                out = torch.zeros(pad.shape[0], 1, q_end - q_start, kv_end - kv_start,
                                  device=device, dtype=torch.float32)
                out.masked_fill_(~allowed[:, None], float("-inf"))
                return out

        hidden_states = embeds
        block_states = embeds.new_zeros(*embeds.shape[:-1], 0, embeds.shape[-1]).to(torch.float32)

        used_caches = []

        def run_layer(layer, hidden_states, block_states, messages, pass_idx=0):
            cache = None
            if past_key_values is not None:
                cache = past_key_values[self.cache_slot(layer.layer_idx, pass_idx)]
            new_hidden = layer(hidden_states, block_states, messages, block_mask, relative_distance, position_ids,
                               past_key_value=cache)
            if (scalar_tap is not None and layer.layer_idx == scalar_tap) or capture_all or \
                    (list_taps is not None and layer.layer_idx in list_taps):
                # Loop layers run several times; keep the final application's output.
                tap_out[layer.layer_idx] = new_hidden.clone()
            if cache is not None and all(cache is not c for c in used_caches):
                used_caches.append(cache)
            if layer.is_block_boundary:
                block_states = torch.cat([block_states, hidden_states.unsqueeze(-2).to(torch.float32)], dim=-2)
            return new_hidden, block_states

        for layer in self.layers[: cfg.loop_start]:
            hidden_states, block_states = run_layer(layer, hidden_states, block_states, None)

        pre_loop_hidden, pre_loop_block = hidden_states, block_states
        num_pre_blocks = pre_loop_block.shape[-2]

        for layer in self.layers[cfg.loop_start: cfg.loop_end]:
            hidden_states, block_states = run_layer(layer, hidden_states, block_states, None)

        num_loop = cfg.loop_end - cfg.loop_start
        loop_layers = self.layers[cfg.loop_start: cfg.loop_end]
        if cfg.loop_num_passes > 1:
            # [num_loop, 4, hidden] = [q, k, v, mlp] message queries, identical across passes
            mc_queries = self.middle_cycle_queries()
        for pass_idx in range(1, cfg.loop_num_passes):
            previous_block_states = block_states[..., num_pre_blocks + 1:, :]
            previous_hidden = hidden_states
            hidden_states, block_states = pre_loop_hidden, pre_loop_block
            for loop_layer_idx, layer in enumerate(loop_layers):
                messages = message_query_attention(previous_block_states, previous_hidden,
                                                   mc_queries, loop_layer_idx)
                hidden_states, block_states = run_layer(layer, hidden_states, block_states, messages,
                                                        pass_idx=pass_idx)

        for layer in self.layers[cfg.loop_end:]:
            hidden_states, block_states = run_layer(layer, hidden_states, block_states, None)

        for c in used_caches:
            c.pos.add_(seq_len)
        final_states = torch.cat([block_states, hidden_states.unsqueeze(-2)], dim=-2).to(torch.float32)
        final_states = _unweighted_norm(final_states)
        query_states = self.attn_res.output_query.to(torch.float32).reshape(-1, final_states.shape[-1]) \
            .expand(*final_states.shape[:-2], -1, -1)
        hidden_states = _query_kv_attention(query_states, final_states, final_states.shape[-1])[:, :, 0]
        if intermediate_output is None:
            return self.norm(hidden_states)
        if capture_all:
            intermediate = torch.cat([tap_out[i].unsqueeze(1) for i in range(cfg.num_hidden_layers)], dim=1)
        elif list_taps is not None:
            intermediate = torch.cat([tap_out[i].unsqueeze(1) for i in list_taps], dim=1)
        else:
            intermediate = tap_out.get(scalar_tap)
        if intermediate is not None and final_layer_norm_intermediate:
            intermediate = self.norm(intermediate)
        return self.norm(hidden_states), intermediate

    def middle_cycle_queries(self):
        queries = self.middle_cycle_cross_pass_queries.unflatten(0, (-1, 2))
        kv = self.middle_cycle_cross_pass_kv_queries.unflatten(0, (-1, 2))
        return torch.cat([queries[:, :1], kv, queries[:, 1:]], dim=1)  # [num_loop, 4, hidden]


# ------------------------------------------------------------ vision tower

class VisionRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.vision_hidden_size // config.vision_num_heads
        spatial_dim = dim // 2
        inv_freq = 1.0 / (config.vision_rope_theta ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids):
        freqs = position_ids[..., None].to(torch.float) * self.inv_freq.to(position_ids.device)
        cos, sin = freqs.cos(), freqs.sin()
        cos_h, cos_w = cos[:, 0], cos[:, 1]
        sin_h, sin_w = sin[:, 0], sin[:, 1]
        cos = torch.cat([cos_w, cos_h], dim=-1).repeat_interleave(2, dim=-1)
        sin = torch.cat([sin_w, sin_h], dim=-1).repeat_interleave(2, dim=-1)
        return cos, sin


def _rotate_pairwise(x):
    x = x.view(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def vision_rope_position_ids(patch_grid_thw, merge_size, device):
    """[num_patches, 2] (h, w) ids in spatial-merge-block order per image."""
    position_ids = []
    for t, h, w in patch_grid_thw.tolist():
        hpos, wpos = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        block_shape = (h // merge_size, merge_size, w // merge_size, merge_size)
        hpos = hpos.reshape(block_shape).transpose(1, 2).flatten()
        wpos = wpos.reshape(block_shape).transpose(1, 2).flatten()
        position_ids.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
    return torch.cat(position_ids, dim=0)


def vision_pos_interp_indices_and_weights(patch_grid_thw, side, merge_size, device):
    """Bilinear resampling of a (side, side) pos-embedding table, per patch, in merge-block order."""
    counts = patch_grid_thw.prod(-1).to(torch.long)
    heights = torch.repeat_interleave(patch_grid_thw[:, 1], counts).to(device)
    widths = torch.repeat_interleave(patch_grid_thw[:, 2], counts).to(device)
    num_patches = int(counts.sum())
    idx = torch.arange(num_patches, device=device)
    within = idx - torch.repeat_interleave(F.pad(counts.cumsum(0)[:-1], (1, 0)), counts).to(device)
    blocks_w = widths // merge_size
    in_col = within % merge_size
    in_row = (within // merge_size) % merge_size
    block_col = (within // (merge_size * merge_size)) % blocks_w
    block_row = within // (merge_size * merge_size * blocks_w)
    rows = block_row * merge_size + in_row
    cols = block_col * merge_size + in_col
    src_y = (rows.to(torch.float32) + 0.5) * (side / heights.to(torch.float32)) - 0.5
    src_x = (cols.to(torch.float32) + 0.5) * (side / widths.to(torch.float32)) - 0.5
    src_y = src_y.clamp(0, side - 1)
    src_x = src_x.clamp(0, side - 1)
    y0 = src_y.floor().long(); y1 = (y0 + 1).clamp(max=side - 1)
    x0 = src_x.floor().long(); x1 = (x0 + 1).clamp(max=side - 1)
    wy = (src_y - y0.to(torch.float32)).clamp(0, 1)
    wx = (src_x - x0.to(torch.float32)).clamp(0, 1)
    indices = torch.stack([y0 * side + x0, y0 * side + x1, y1 * side + x0, y1 * side + x1], dim=-1)
    weights = torch.stack([(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx], dim=-1)
    return indices, weights


class VisionAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, eps, device=None, dtype=None, operations=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim ** -0.5
        ops = operations or comfy.ops
        self.q_proj = ops.Linear(hidden_size, hidden_size, bias=True, device=device, dtype=dtype)
        self.k_proj = ops.Linear(hidden_size, hidden_size, bias=True, device=device, dtype=dtype)
        self.v_proj = ops.Linear(hidden_size, hidden_size, bias=True, device=device, dtype=dtype)
        self.out_proj = ops.Linear(hidden_size, hidden_size, bias=True, device=device, dtype=dtype)
        self.comfy_attention = ComfyAttention()

    def forward(self, x, cu_seqlens, position_embeddings):
        seq_len = x.shape[0]
        q = self.q_proj(x).view(seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(seq_len, self.num_heads, self.head_dim)
        cos, sin = position_embeddings
        cos, sin = cos.unsqueeze(-2).to(torch.float32), sin.unsqueeze(-2).to(torch.float32)
        q = (q.to(torch.float32) * cos + _rotate_pairwise(q.to(torch.float32)) * sin).to(q.dtype)
        k = (k.to(torch.float32) * cos + _rotate_pairwise(k.to(torch.float32)) * sin).to(k.dtype)
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        attn = optimized_attention_for_device(x.device, mask=False)
        outputs = []
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        for qs, ks, vs in zip(torch.split(q, lengths, dim=2), torch.split(k, lengths, dim=2),
                              torch.split(v, lengths, dim=2)):
            qs, ks, vs = AttentionTensorContainer(qs), AttentionTensorContainer(ks), AttentionTensorContainer(vs)
            out = attn(qs, ks, vs, self.num_heads, skip_reshape=True, skip_output_reshape=True,
                       preferred_attention=self.comfy_attention)
            outputs.append(out.transpose(1, 2))
        attn_output = torch.cat(outputs, dim=1)
        return self.out_proj(attn_output.reshape(seq_len, -1))


class VisionLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x.to(torch.float32), (x.shape[-1],), self.weight.to(torch.float32),
                            self.bias.to(torch.float32), self.eps).to(x.dtype)


class VisionMLP(nn.Module):
    def __init__(self, in_f, hidden_f, out_f, device=None, dtype=None, operations=None):
        super().__init__()
        ops = operations or comfy.ops
        self.c_fc = ops.Linear(in_f, hidden_f, bias=True, device=device, dtype=dtype)
        self.c_proj = ops.Linear(hidden_f, out_f, bias=True, device=device, dtype=dtype)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class VisionEncoderLayer(nn.Module):
    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        h = config.vision_hidden_size
        self.self_attn = VisionAttention(h, config.vision_num_heads, config.vision_layer_norm_eps,
                                         device=device, dtype=dtype, operations=operations)
        self.ln_1 = VisionLayerNorm(h, config.vision_layer_norm_eps)
        self.ln_2 = VisionLayerNorm(h, config.vision_layer_norm_eps)
        self.mlp = VisionMLP(h, config.vision_intermediate_size, h, device=device, dtype=dtype, operations=operations)
        self.ls_1 = nn.Module()
        self.ls_1.gamma = nn.Parameter(torch.full((h,), config.vision_layer_scale_init_value))
        self.ls_2 = nn.Module()
        self.ls_2.gamma = nn.Parameter(torch.full((h,), config.vision_layer_scale_init_value))

    def forward(self, x, cu_seqlens, position_embeddings):
        residual = x
        y = self.self_attn(self.ln_1(x), cu_seqlens, position_embeddings)
        x = residual + self.ls_1.gamma.to(y.dtype) * y
        residual = x
        y = self.mlp(self.ln_2(x))
        return residual + self.ls_2.gamma.to(y.dtype) * y


class VisionTower(nn.Module):
    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        h = config.vision_hidden_size
        ops = operations or comfy.ops
        self.backbone = nn.Module()
        emb = nn.Module()
        emb.patch_embedding = ops.Linear(3 * config.vision_patch_size ** 2, h, bias=False, device=device, dtype=dtype)
        emb.register_parameter("position_embedding", nn.Parameter(torch.empty(config.vision_posemb_grid_size ** 2, h)))
        emb.class_embedding = nn.Parameter(torch.zeros(h))
        emb.cls_position_embedding = nn.Parameter(torch.zeros(h))
        self.backbone.embeddings = emb
        self.backbone.ln_pre = VisionLayerNorm(h, config.vision_layer_norm_eps)
        self.backbone.layers = nn.ModuleList(
            [VisionEncoderLayer(config, device=device, dtype=dtype, operations=operations)
             for _ in range(config.vision_num_layers)])
        self.proj_in = ops.Linear(config.vision_merge_size ** 2 * h, config.vision_projection_hidden_dim,
                                  bias=False, device=device, dtype=dtype)
        self.proj_out = ops.Linear(config.vision_projection_hidden_dim, config.vision_text_hidden_size,
                                   bias=False, device=device, dtype=dtype)
        self.rope = VisionRotaryEmbedding(config)
        self.config = config

    def forward(self, pixel_values, image_grid_thw):
        # pixel_values: [N, merge*patch, merge*patch, C] in [-1, 1], one per merged image token
        cfg = self.config
        merge = cfg.vision_merge_size
        patch = cfg.vision_patch_size
        patch_grid_thw = image_grid_thw.cpu() * torch.tensor([1, merge, merge])
        patches_per_image = patch_grid_thw.prod(-1)
        num_patches = pixel_values.shape[0] * merge ** 2
        cu_seqlens = F.pad((patches_per_image + 1).cumsum(0), (1, 0)).to(torch.int32).to(pixel_values.device)
        image_ids = torch.arange(1, patch_grid_thw.shape[0] + 1)
        patch_index = torch.arange(num_patches) + torch.repeat_interleave(image_ids, patches_per_image)
        patch_index = patch_index.to(pixel_values.device)

        side = cfg.vision_posemb_grid_size
        x = pixel_values.to(self.backbone.embeddings.patch_embedding.weight.dtype)
        patches = x.reshape(-1, merge, patch, merge, patch, x.shape[-1]).transpose(2, 3)
        flat = patches.reshape(-1, 3 * patch * patch)
        patch_embeds = self.backbone.embeddings.patch_embedding(flat)
        indices, weights = vision_pos_interp_indices_and_weights(patch_grid_thw, side, merge, pixel_values.device)
        pos = (self.backbone.embeddings.position_embedding[indices] * weights[..., None]).sum(1)
        patch_embeds = patch_embeds + pos.to(patch_embeds.dtype)
        cls_embedding = (self.backbone.embeddings.class_embedding +
                         self.backbone.embeddings.cls_position_embedding).to(patch_embeds.dtype)
        embeddings = cls_embedding.repeat(patch_embeds.shape[0] + patch_grid_thw.shape[0], 1).clone()
        embeddings[patch_index] = patch_embeds

        hidden_states = self.backbone.ln_pre(embeddings)

        position_ids = vision_rope_position_ids(patch_grid_thw, merge, pixel_values.device)
        position_ids = position_ids.new_zeros(hidden_states.shape[0], 2).index_copy(0, patch_index, position_ids + 1)
        position_embeddings = self.rope(position_ids)

        for layer in self.backbone.layers:
            hidden_states = layer(hidden_states, cu_seqlens, position_embeddings)

        patch_states = hidden_states[patch_index]
        merged = self.proj_in(patch_states.reshape(-1, merge ** 2 * hidden_states.shape[-1]))
        return self.proj_out(F.gelu(merged))


def process_inkling2_images(image, token_patch=40):
    """Comfy IMAGE batch [B,H,W,C] in [0,1] -> HF pixel_values ([N,40,40,C] in [-1,1]) and image_grid_thw.

    Mirrors Inkling2ImageProcessor: pad to whole 40px tokens with -1/255, normalize with mean/std 0.5,
    emit one patch per image token in raster order.
    """
    if not torch.is_tensor(image):
        image = torch.as_tensor(image)
    image = image.to(torch.float32)
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.shape[1] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = image.permute(0, 2, 3, 1)  # [B,C,H,W] -> [B,H,W,C]
    image = image[..., :3]
    patches, grids = [], []
    for img in image:
        channels, height, width = img.shape[-1], img.shape[0], img.shape[1]
        rows = (height + token_patch - 1) // token_patch
        columns = (width + token_patch - 1) // token_patch
        padded = torch.full((rows * token_patch, columns * token_patch, channels), -1.0 / 255.0,
                            dtype=img.dtype, device=img.device)
        padded[:height, :width] = img
        padded = (padded - 0.5) / 0.5
        patch = padded.reshape(rows, token_patch, columns, token_patch, channels).permute(0, 2, 1, 3, 4)
        patches.append(patch.reshape(rows * columns, token_patch, token_patch, channels))
        grids.append([1, rows, columns])
    return torch.cat(patches, 0), torch.tensor(grids, dtype=torch.long)


# --------------------------------------------------- top level transformer

def _reset_inference_cache(pkv):
    """Reset token positions and clear the convolution rings, in place.

    A prefill restart must drop warmup values from the conv rings as well as the position
    counter. Zeroing in place keeps the tensor addresses a captured CUDA graph reads.
    """
    for cache in pkv:
        cache.pos.zero_()
        for ring in cache.conv.values():
            ring.zero_()


def _normalize_layer_tap(layer, num_layers):
    if not isinstance(layer, int):
        raise ValueError("Invalid Inkling2 layer tap {!r}".format(layer))
    tap = layer + num_layers if layer < 0 else layer
    if not 0 <= tap < num_layers:
        raise ValueError("Invalid Inkling2 layer tap {}; expected -{} <= layer < {}".format(
            layer, num_layers, num_layers))
    return tap


_THINKING_TOKEN = 200008  # <|content_thinking|>
_TEXT_TOKEN = 200004  # <|content_text|>
_CHANNEL_TOKENS = frozenset({
    199999, 200000, 200001, 200002, 200003, 200005, 200006, 200008, 200010, 200022, 200024, 200049, 200054, 200057,
})


class _AnswerStream:
    """Incrementally tracks the active output channel and visible answer tokens.

    prompt_channel is the channel the prompt left open: "text", "thinking", or None when
    the model chooses. Tokens generated before a channel is known are kept provisionally
    and discarded if a thinking marker turns out to open a reasoning section.
    """

    def __init__(self, prompt_channel=None):
        self.known = prompt_channel is not None
        self.in_text = prompt_channel == "text"
        self.visible = []
        self.provisional = 0

    def add(self, token):
        if not self.known:
            if token == _THINKING_TOKEN:
                if self.provisional:
                    del self.visible[-self.provisional:]
                    self.provisional = 0
                self.known = True
                self.in_text = False
            elif token == _TEXT_TOKEN:
                self.provisional = 0
                self.known = True
                self.in_text = True
            else:
                self.visible.append(token)
                self.provisional += 1
            return
        if token == _TEXT_TOKEN:
            self.in_text = True
        elif token in _CHANNEL_TOKENS:
            self.in_text = False
        elif self.in_text:
            self.visible.append(token)

    def ids(self):
        return self.visible


def _visible_answer_ids(generated_ids, prompt_channel=None):
    """Concatenate the contents of every <|content_text|> block, dropping reasoning."""
    stream = _AnswerStream(prompt_channel)
    for token in generated_ids:
        stream.add(token)
    return stream.ids()


class Inkling2(nn.Module):
    """Comfy TE transformer for Inkling-2. State-dict keys match the checkpoint (model.llm.*, model.visual.*)."""

    def __init__(self, config_dict, dtype=None, device=None, operations=None):
        super().__init__()
        self.model_config = Inkling2Config(config_dict)
        self.config = self.model_config
        self.model = nn.Module()
        self.model.llm = Inkling2TextModel(self.model_config, device=device, dtype=dtype, operations=operations)
        self.model.llm.unembed = nn.Linear(self.model_config.hidden_size, self.model_config.vocab_size, bias=False)
        self.model.visual = VisionTower(self.model_config, device=device, dtype=dtype, operations=operations)
        self.num_layers = self.model_config.num_hidden_layers
        self.unpadded_vocab_size = self.model_config.unpadded_vocab_size
        if device is not None:
            self.to(device)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # The detector in comfy/sd.py historically also recognized the original TML key
        # layout (.attn.wq_du/.attn.wr_du). This encoder only registers the repacked
        # ComfyUI names, and CLIP loads with strict=False, so refuse that layout loudly
        # instead of silently leaving the query projections unloaded.
        if any(k.endswith((".attn.wq_du.weight", ".attn.wr_du.weight")) for k in state_dict):
            raise RuntimeError(
                "Inkling2 checkpoint uses the original TML key layout (.attn.wq_du/.attn.wr_du); "
                "repack it to the ComfyUI layout (.self_attn.q_proj.linear/...) before loading.")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def get_input_embeddings(self):
        llm = self.model.llm

        class _NormedEmbed:
            weight = llm.embed.weight

            def __call__(self, input_ids, out_dtype=None):
                return llm.embed_norm(llm.embed(input_ids, out_dtype=out_dtype))

            def __call__outer(self):  # pragma: no cover
                pass

        return _NormedEmbed()

    def preprocess_embed(self, embed, device):
        if embed.get("type") == "image":
            pixel_values, grid = process_inkling2_images(embed["data"])
            return self.model.visual(pixel_values.to(device=device, dtype=torch.float32), grid), grid
        return None, None

    def encode_hidden(self, embeds, attention_mask=None, intermediate_output=None,
                      final_layer_norm_intermediate=False):
        return self.model.llm(embeds.to(torch.bfloat16), attention_mask=attention_mask,
                              intermediate_output=intermediate_output,
                              final_layer_norm_intermediate=final_layer_norm_intermediate)

    def forward(self, tokens=None, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None,
                final_layer_norm_intermediate=False, dtype=None, embeds_info=None, **kwargs):
        outputs = self.encode_hidden(embeds, attention_mask=attention_mask,
                                     intermediate_output=intermediate_output,
                                     final_layer_norm_intermediate=final_layer_norm_intermediate)
        if isinstance(outputs, tuple):
            return outputs
        return outputs, None

    def generate(self, embeds, do_sample, max_length, temperature, top_k, top_p, min_p, repetition_penalty, seed,
                 presence_penalty=0.0, mtp=False, strip_thinking=True, prompt_channel=None):
        from .token_stream import report_generated_tokens
        from tqdm import tqdm
        import comfy.model_management
        import comfy.utils
        device = embeds.device
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
        llm = self.model.llm
        embed_module = self.get_input_embeddings()
        unembed = self.model.llm.unembed.weight[: self.unpadded_vocab_size].to(torch.float32)
        stop_tokens = set(self.config.stop_tokens)
        current = embeds.to(torch.bfloat16)
        batch, seq_len, _ = current.shape
        cap_true = seq_len + max_length + 1
        cap = max(8, (cap_true + 7) // 8 * 8)  # 8-aligned; attention cost scales with cap in decode

        def project_logits(h):
            last = h[:, -1].to(torch.float32)
            return softcap(F.linear(last, unembed), self.config.final_logit_softcapping)

        def sample_step(logits):
            if (repetition_penalty and repetition_penalty != 1.0) or (presence_penalty and presence_penalty != 0.0):
                if generated_ids:
                    pen_tokens = torch.tensor(sorted(set(generated_ids)), device=device, dtype=torch.long)
                    if repetition_penalty and repetition_penalty != 1.0:
                        vals = logits[:, pen_tokens]
                        logits[:, pen_tokens] = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
                    if presence_penalty and presence_penalty != 0.0:
                        logits[:, pen_tokens] -= presence_penalty
            if not do_sample:
                return torch.argmax(logits, dim=-1)
            logits = logits / max(temperature, 1e-5)
            if top_k and top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1)
                logits = torch.where(logits >= kth.values[:, -1:], logits, torch.tensor(float("-inf"), device=device))
            if min_p and 0.0 < min_p < 1.0:
                probs = F.softmax(logits, dim=-1)
                limit = probs.max(dim=-1, keepdim=True).values * min_p
                logits = torch.where(probs >= limit, logits, torch.tensor(float("-inf"), device=device))
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1)
                cum = probs.cumsum(dim=-1)
                remove = (cum - probs) > top_p
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                logits = sorted_logits.scatter(-1, sorted_idx, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1, generator=generator).squeeze(-1)

        generated_ids = []
        answer_stream = _AnswerStream(prompt_channel) if strip_thinking else None
        graphs_ok = device.type == "cuda" and not getattr(getattr(comfy.model_management, "args", None),
                                                           "disable_cuda_graphs", False)
        graph_used = False

        with torch.no_grad():
            # Graph and KV cache are scoped to this call so both are released when it
            # returns or is cancelled, and a captured graph can never outlive the
            # parameter storage it recorded.
            pkv = llm.init_kv_cache(batch, cap, device, torch.bfloat16)
            tok_buf = torch.zeros(batch, 1, dtype=torch.long, device=device)
            logits_buf = torch.empty(batch, unembed.shape[0], device=device, dtype=torch.float32)
            graph = None
            step_fn = None
            s = None
            try:
                hidden = llm(current, attention_mask=None, past_key_values=pkv)
                logits = project_logits(hidden)
                if graphs_ok:
                    try:
                        def step_fn():
                            e = embed_module(tok_buf).to(torch.bfloat16)
                            h = llm(e, attention_mask=None, past_key_values=pkv)
                            logits_buf.copy_(softcap(F.linear(h[:, 0].to(torch.float32), unembed),
                                                     self.config.final_logit_softcapping))

                        s = torch.cuda.Stream(device=device)
                        s.wait_stream(torch.cuda.current_stream(device))
                        with torch.cuda.stream(s):
                            for _ in range(2):
                                step_fn()
                        torch.cuda.current_stream(device).wait_stream(s)
                        # Warmup advanced positions and filled the conv rings: restart
                        # from the prompt and clear both before capturing.
                        _reset_inference_cache(pkv)
                        _ = llm(current, attention_mask=None, past_key_values=pkv)
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g, stream=s):
                            step_fn()
                        graph = g
                        graph_used = True
                    except Exception as exc:
                        print("inkling2: CUDA graph capture failed, using eager decode:", exc)
                        if s is not None:
                            # Warmup may still have work queued on the side stream; order
                            # it before the reset writes the same cache tensors.
                            torch.cuda.current_stream(device).wait_stream(s)
                        _reset_inference_cache(pkv)
                        hidden = llm(current, attention_mask=None, past_key_values=pkv)
                        logits = project_logits(hidden)

                pbar = comfy.utils.ProgressBar(max_length)
                for step in tqdm(range(max_length), desc="Generating tokens"):
                    # Graph replay runs no python ops, so check the interrupt flag ourselves.
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    tid = int(sample_step(logits).item())
                    if tid in stop_tokens:
                        break
                    generated_ids.append(tid)
                    if answer_stream is not None:
                        # Track the output channel incrementally; preview and final result
                        # share the same visible ID list. Reasoning stays hidden.
                        answer_stream.add(tid)
                        report_generated_tokens(answer_stream.ids())
                    else:
                        report_generated_tokens(generated_ids)
                    pbar.update(1)
                    if step == max_length - 1:
                        break
                    tok_buf.fill_(tid)
                    if graph is not None:
                        graph.replay()
                        logits = logits_buf
                    else:
                        hidden = llm(embed_module(tok_buf).to(torch.bfloat16), attention_mask=None, past_key_values=pkv)
                        logits = project_logits(hidden)
            finally:
                # The captured graph owns graph-private allocations, and the KV cache
                # belongs to this call; drop both on completion and cancellation.
                step_fn = None
                graph = None
                s = None
                pkv = tok_buf = logits_buf = None
        self._graph_used = graph_used
        if answer_stream is not None:
            generated_ids = answer_stream.ids()
        return generated_ids


# --------------------------------------------------- tokenizer + TE wrappers

INKLING2_TOKENIZER_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "inkling2_tokenizer")


class Inkling2Tokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}, embedding_size=2304, embedding_key="inkling2"):
        from transformers import PreTrainedTokenizerFast
        super().__init__(INKLING2_TOKENIZER_PATH, pad_with_end=False, embedding_directory=embedding_directory,
                         embedding_size=embedding_size, embedding_key=embedding_key,
                         tokenizer_class=PreTrainedTokenizerFast, has_start_token=False, has_end_token=False,
                         pad_to_max_length=False, max_length=99999999, min_length=1, tokenizer_data=tokenizer_data)
        vocab = self.tokenizer.get_vocab()
        self.pad_token = vocab["<|endoftext|>"]
        self.special_ids = {name: vocab[t] for name, t in {
            "endoftext": "<|endoftext|>", "message_user": "<|message_user|>", "message_model": "<|message_model|>",
            "message_system": "<|message_system|>", "content_text": "<|content_text|>",
            "content_image": "<|content_image|>", "end_message": "<|end_message|>",
            "content_end_sampling": "<|content_model_end_sampling|>"}.items()}


class Inkling2TokenizerWrapper(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer = lambda *a, **kw: Inkling2Tokenizer(*a, **kw, embedding_size=2304, embedding_key="inkling2")
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name="inkling2",
                         tokenizer=tokenizer)
        self.llama_template = "<|message_user|><|content_text|>{}<|end_message|><|message_model|>"
        # HF chat template: each content part gets its own <|message_user|> block; the vision
        # features are spliced into the <|unused_200054|> placeholder that follows <|content_image|>.
        self.llama_template_images = "<|message_user|><|content_image|><|unused_200054|><|end_message|><|message_user|><|content_text|>{}<|end_message|><|message_model|>"
        self.image_token_id = 200054  # <|unused_200054|>

    def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None, images=[], thinking=False,
                              system_prompt="", reasoning_effort=0.9, skip_template=False, videos=[], audios=[],
                              **kwargs):
        kwargs.setdefault("min_length", 1)
        image = kwargs.get("image", None)
        if image is not None and len(images) == 0:
            images = [image[i:i + 1] for i in range(image.shape[0])]
        if len(videos) > 0 or len(audios) > 0:
            logging.warning("Warning: Inkling2 text encoder does not support video/audio input, ignoring.")
        effort_map = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7,
                      "high": 0.9, "xhigh": 0.99, "max": 0.99}
        if isinstance(reasoning_effort, str):
            effort_num = effort_map.get(reasoning_effort.strip().lower(), reasoning_effort)
        else:
            effort_num = reasoning_effort
        effort = "<|message_system|><|content_text|>Thinking effort level: {}<|end_message|>".format(
            "0" if effort_num == 0.0 else effort_num)
        if skip_template or text.startswith("<|message_"):
            llama_text = text
        else:
            if llama_template is not None:
                llama_text = llama_template.format(text)
            elif len(images) == 0:
                llama_text = self.llama_template.format(text)
            else:
                image_blocks = "<|message_user|><|content_image|><|unused_200054|><|end_message|>" * len(images)
                llama_text = (image_blocks + "<|message_user|><|content_text|>" + text +
                              "<|end_message|><|message_model|>")
            if system_prompt:
                llama_text = "<|message_system|><|content_text|>" + system_prompt + "<|end_message|>" + llama_text
            if effort:
                llama_text = effort + llama_text
            if not thinking and llama_template is None:
                # Continue straight into the answer channel instead of the reasoning section.
                llama_text += "<|content_text|>"
        tokens = super().tokenize_with_weights(llama_text, return_word_ids=return_word_ids, disable_weights=True, **kwargs)
        key_name = next(iter(tokens))
        embed_count = 0
        for r in tokens[key_name]:
            for i in range(len(r)):
                if isinstance(r[i][0], (int, float)) and r[i][0] == self.image_token_id:  # <|unused_200054|>
                    if len(images) > embed_count:
                        r[i] = ({"type": "image", "data": images[embed_count], "original_type": "image"},) + r[i][1:]
                        embed_count += 1
        return tokens


class Inkling2ClipModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=-1, dtype=None, attention_mask=True, model_options={},
                 model_type="inkling2"):
        class Inkling2_(Inkling2):
            pass
        Inkling2_.model_type = model_type
        super().__init__(device=device, layer=layer, layer_idx=layer_idx,
                         textmodel_json_config=model_options.get("inkling2_model_config", {}),
                         dtype=dtype, special_tokens={}, layer_norm_hidden_state=False, model_class=Inkling2_,
                         enable_attention_masks=attention_mask, return_attention_masks=attention_mask,
                         model_options=model_options)
        self.special_tokens = {"pad": 199999, "end": self.transformer.model_config.eos_token_id}

    def generate(self, tokens, do_sample, max_length, temperature, top_k, top_p, min_p, repetition_penalty, seed,
                 presence_penalty=0.0, mtp=False):
        if isinstance(tokens, dict):
            tokens = next(iter(tokens.values()))
        tokens_only = [[t[0] for t in b] for b in tokens]
        # The last channel marker after <|message_model|> is the channel the continuation
        # starts in (for example a skip_template prompt ending in <|content_thinking|>).
        prompt_channel = None
        for batch_tokens in tokens_only:
            last_model = -1
            for i, t in enumerate(batch_tokens):
                if isinstance(t, int) and t == 200001:  # <|message_model|>
                    last_model = i
            for t in batch_tokens[last_model + 1:]:
                if not isinstance(t, int):
                    continue
                if t == 200004:  # <|content_text|>
                    prompt_channel = "text"
                elif t == 200008:  # <|content_thinking|>
                    prompt_channel = "thinking"
        dev = self.execution_device or next(self.transformer.parameters()).device
        embeds, _, _, _ = self.process_tokens(tokens_only, dev)
        return self.transformer.generate(embeds, do_sample, max_length, temperature, top_k, top_p, min_p,
                                         repetition_penalty, seed, presence_penalty=presence_penalty,
                                         prompt_channel=prompt_channel)


class Inkling2TEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}, model_type="inkling2"):
        clip_model = lambda **kw: Inkling2ClipModel(**kw, model_type=model_type)
        super().__init__(device=device, dtype=dtype, name=model_type, clip_model=clip_model, model_options=model_options)


def tokenizer(model_type="inkling2"):
    class Inkling2Tokenizer_(Inkling2TokenizerWrapper):
        def __init__(self, embedding_directory=None, tokenizer_data={}):
            super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
    return Inkling2Tokenizer_


def te(dtype_llama=None, llama_quantization_metadata=None, model_type="inkling2", stop_tokens=None,
       config_dict=None):
    class Inkling2TEModel_(Inkling2TEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_llama is not None:
                dtype = dtype_llama
            model_options = model_options.copy()
            cfg = {"stop_tokens": stop_tokens or [199999, 200006]}
            if config_dict is not None:
                cfg["text_config"] = config_dict.get("text_config", {})
                cfg["vision_config"] = config_dict.get("vision_config", {})
                cfg["image_token_id"] = config_dict.get("image_token_id", 200054)
            config_key = "{}_model_config".format(model_type)
            model_options[config_key] = {**cfg, **model_options.get(config_key, {})}
            if llama_quantization_metadata is not None:
                model_options["quantization_metadata"] = llama_quantization_metadata
            super().__init__(device=device, dtype=dtype, model_options=model_options, model_type=model_type)
    return Inkling2TEModel_
