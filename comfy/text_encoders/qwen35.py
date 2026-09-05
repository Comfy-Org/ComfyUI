import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from tqdm import tqdm
import os
import warnings

import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy_kitchen
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy import sd1_clip
import comfy.text_encoders.qwen_vl

from .llama import BaseLlama, BaseGenerate, FixedKV, FixedKVBias, Llama2_, MLP, RMSNorm, apply_rope, fixed_kv_bias_decode, precompute_freqs_cis


@dataclass
class LinearKV(FixedKV):
    # DeltaNet state on the FixedKV interface: key=conv_state, value=recurrent_state (fp32)
    g_decay: torch.Tensor = None
    dt_bias: torch.Tensor = None
    snapshots: list = None  # [(recurrent, conv)] taken after step 1, 2, ... of the last verify
    last_seq: int = 1

    def prepare(self, num_tokens):
        pass

    def rollback(self, discard=1):
        # discard the rejected tail: restore the snapshot taken after the last kept token
        rec, conv = self.snapshots[self.last_seq - discard - 1]
        self.recurrent_state.copy_(rec)
        self.conv_state.copy_(conv)
        self.index -= discard

    @property
    def conv_state(self):
        return self.key

    @property
    def recurrent_state(self):
        return self.value




def _qwen35_layer_types(n):
    return [("full_attention" if (i + 1) % 4 == 0 else "linear_attention") for i in range(n)]

@dataclass
class Qwen35Config:
    vocab_size: int = 248320
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 24
    # Full attention params
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 256
    partial_rotary_factor: float = 0.25
    # Linear attention (DeltaNet) params
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    conv_kernel_size: int = 4
    # Shared params
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    mrope_section: list = field(default_factory=lambda: [11, 11, 10])
    layer_types: list = field(default_factory=lambda: _qwen35_layer_types(24))
    rms_norm_add: bool = True
    mlp_activation: str = "silu"
    qkv_bias: bool = False
    final_norm: bool = True
    lm_head: bool = False
    stop_tokens: list = field(default_factory=lambda: [248044, 248046])
    # These are needed for BaseLlama/BaseGenerate compatibility but unused directly
    transformer_type: str = "qwen35_2b"
    rope_dims: list = None
    rope_scale: float = None
    mtp: bool = False

QWEN35_VISION_DEFAULTS = dict(hidden_size=1024, num_heads=16, intermediate_size=4096, depth=24, patch_size=16, temporal_patch_size=2, in_channels=3, spatial_merge_size=2, num_position_embeddings=2304)

QWEN35_MODELS = {
    "qwen35_08b": dict(hidden_size=1024, intermediate_size=3584, vision=dict(hidden_size=768, num_heads=12, intermediate_size=3072, depth=12)),
    "qwen35_2b": dict(hidden_size=2048, intermediate_size=6144, num_hidden_layers=24, num_attention_heads=8, num_key_value_heads=2, linear_num_value_heads=16),
    "qwen35_4b": dict(hidden_size=2560, intermediate_size=9216, num_hidden_layers=32, num_attention_heads=16, num_key_value_heads=4, linear_num_value_heads=32),
    "qwen35_9b": dict(hidden_size=4096, intermediate_size=12288, num_hidden_layers=32, num_attention_heads=16, num_key_value_heads=4, linear_num_value_heads=32, lm_head=True, vision=dict(hidden_size=1152, intermediate_size=4304, depth=27)),
    "qwen35_27b": dict(hidden_size=5120, intermediate_size=17408, num_hidden_layers=64, num_attention_heads=24, num_key_value_heads=4, linear_num_value_heads=48, lm_head=True, vision=dict(hidden_size=1152, intermediate_size=4304, depth=27)),
}


def _make_config(model_type, config_dict={}):
    overrides = QWEN35_MODELS.get(model_type, {}).copy()
    overrides.pop("vision", None)
    if "num_hidden_layers" in overrides:
        overrides["layer_types"] = _qwen35_layer_types(overrides["num_hidden_layers"])
    overrides.update(config_dict)
    return Qwen35Config(**overrides)


class RMSNormGated(RMSNorm):
    def forward(self, x, gate):
        return super().forward(x) * F.silu(gate.to(x.dtype))

def torch_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False):
    initial_dtype = query.dtype
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


# GatedDeltaNet - Linear Attention Layer

class GatedDeltaNet(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()

        hidden = config.hidden_size
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.conv_kernel_size = config.conv_kernel_size

        key_dim = self.num_key_heads * self.key_head_dim
        value_dim = self.num_value_heads * self.value_head_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        conv_dim = key_dim * 2 + value_dim

        self.in_proj_qkv = ops.Linear(hidden, conv_dim, bias=False, device=device, dtype=dtype)
        self.in_proj_z = ops.Linear(hidden, value_dim, bias=False, device=device, dtype=dtype)
        self.in_proj_b = ops.Linear(hidden, self.num_value_heads, bias=False, device=device, dtype=dtype)
        self.in_proj_a = ops.Linear(hidden, self.num_value_heads, bias=False, device=device, dtype=dtype)
        self.out_proj = ops.Linear(value_dim, hidden, bias=False, device=device, dtype=dtype)

        self.dt_bias = nn.Parameter(torch.empty(self.num_value_heads, device=device, dtype=dtype))
        self.A_log = nn.Parameter(torch.empty(self.num_value_heads, device=device, dtype=dtype))

        self.conv1d = ops.Conv1d(in_channels=conv_dim, out_channels=conv_dim, bias=False, kernel_size=self.conv_kernel_size,
            groups=conv_dim, padding=self.conv_kernel_size - 1, device=device, dtype=dtype)

        self.norm = RMSNormGated(self.value_head_dim, eps=config.rms_norm_eps, device=device, dtype=dtype)

    def forward(self, x, past_key_value=None, **kwargs):
        batch_size, seq_len, _ = x.shape

        use_recurrent = (
            past_key_value is not None
            and past_key_value.index > 0
            and seq_len <= 6
        )

        # Projections (shared)
        mixed_qkv = self.in_proj_qkv(x).transpose(1, 2)  # [B, conv_dim, seq_len]
        z = self.in_proj_z(x)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)

        # Conv1d
        if use_recurrent:
            # decode: exact-width causal window, weight resolved via the vbar-aware context
            combined = torch.cat([past_key_value.conv_state, mixed_qkv], dim=-1)
            if seq_len > 1:
                past_key_value.last_seq = seq_len
                for s in range(seq_len - 1):
                    past_key_value.snapshots[s][1].copy_(combined[:, :, 1 + s:1 + s + self.conv_kernel_size - 1])
            past_key_value.conv_state.copy_(combined[:, :, seq_len:])
            with comfy.ops.CastBiasWeightContext(self.conv1d, combined, offloadable=True) as (conv_weight, conv_bias):
                mixed_qkv = F.silu(F.conv1d(combined, conv_weight, conv_bias, groups=self.conv1d.groups))
        else:
            if past_key_value is not None:
                conv_state = past_key_value.conv_state
                conv_state_init = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                conv_state.copy_(conv_state_init[:, :, -conv_state.shape[-1]:])
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        # Split QKV and compute beta/g
        mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, seq_len, conv_dim]
        query, key, value = mixed_qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        if use_recurrent:
            g_decay, dt_bias = past_key_value.g_decay, past_key_value.dt_bias
        else:
            g_decay = -comfy.model_management.cast_to_device(self.A_log, x.device, torch.float32).exp()
            dt_bias = comfy.model_management.cast_to_device(self.dt_bias, x.device, torch.float32)
            if past_key_value is not None:
                past_key_value.g_decay = g_decay
                past_key_value.dt_bias = dt_bias

        # Delta rule
        if use_recurrent:
            query = query.reshape(batch_size, seq_len, self.num_key_heads, self.key_head_dim)
            key = key.reshape(batch_size, seq_len, self.num_key_heads, self.key_head_dim)
            value = value.reshape(batch_size, seq_len, self.num_value_heads, self.value_head_dim)

            beta = b.sigmoid()
            g = g_decay * F.softplus(a.float() + dt_bias)
            if self.num_value_heads != self.num_key_heads:
                rep = self.num_value_heads // self.num_key_heads
                query = query.repeat_interleave(rep, dim=2)
                key = key.repeat_interleave(rep, dim=2)

            scale = self.key_head_dim ** -0.5
            q = F.normalize(query.float(), dim=-1) * scale
            k = F.normalize(key.float(), dim=-1)
            v = value.float()
            beta_t = beta.reshape(batch_size, seq_len, -1)
            g_t = g.reshape(batch_size, seq_len, -1).exp()

            # In-place state update: [B, heads, k_dim, v_dim]
            recurrent_state = past_key_value.recurrent_state
            snaps = getattr(past_key_value, "snap_backing", None)
            fused = getattr(comfy_kitchen, "gated_delta_decode", None)
            if fused is not None and x.is_cuda and self.key_head_dim == 128 and (seq_len == 1 or snaps is not None):
                core_attn_out = fused(q.contiguous(), k.contiguous(), v.contiguous(),
                                      beta_t.float().contiguous(), g_t.float().contiguous(), recurrent_state,
                                      snaps[:seq_len - 1] if seq_len > 1 else None).to(x.dtype)
            else:
                outs = []
                for s in range(seq_len):
                    recurrent_state.mul_(g_t[:, s, :, None, None])
                    kv_mem = torch.einsum('bhk,bhkv->bhv', k[:, s], recurrent_state)
                    delta = (v[:, s] - kv_mem) * beta_t[:, s, :, None]
                    # rank-1 update via baddbmm_: no materialized [B, H, D, D] outer-product temp
                    recurrent_state.view(-1, self.key_head_dim, self.value_head_dim).baddbmm_(
                        k[:, s].reshape(-1, self.key_head_dim, 1), delta.reshape(-1, 1, self.value_head_dim))
                    outs.append(torch.einsum('bhk,bhkv->bhv', q[:, s], recurrent_state))
                    if seq_len > 1 and s < seq_len - 1:
                        past_key_value.snapshots[s][0].copy_(recurrent_state)
                core_attn_out = torch.stack(outs, dim=1).to(x.dtype)
            present_key_value = past_key_value
        else:
            beta = b.sigmoid()
            g = g_decay * F.softplus(a.float() + dt_bias)
            query = query.reshape(batch_size, seq_len, -1, self.key_head_dim)
            key = key.reshape(batch_size, seq_len, -1, self.key_head_dim)
            value = value.reshape(batch_size, seq_len, -1, self.value_head_dim)

            if self.num_value_heads != self.num_key_heads:
                rep = self.num_value_heads // self.num_key_heads
                query = query.repeat_interleave(rep, dim=2)
                key = key.repeat_interleave(rep, dim=2)

            core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=None,
                output_final_state=past_key_value is not None,
            )

            present_key_value = None
            if past_key_value is not None:
                if last_recurrent_state is not None:
                    past_key_value.recurrent_state.copy_(last_recurrent_state.to(past_key_value.recurrent_state.dtype))
                present_key_value = past_key_value

        # Gated norm + output projection (shared)
        core_attn_out = self.norm(core_attn_out.reshape(-1, self.value_head_dim), z.reshape(-1, self.value_head_dim))
        output = self.out_proj(core_attn_out.reshape(batch_size, seq_len, -1))
        return output, present_key_value


# GatedAttention - Full Attention with output gating
def apply_partial_rope(xq, xk, freqs_cis, rotary_dim):
    """Apply RoPE to only the first rotary_dim dimensions."""
    xq_rot = xq[..., :rotary_dim]
    xq_pass = xq[..., rotary_dim:]
    xk_rot = xk[..., :rotary_dim]
    xk_pass = xk[..., rotary_dim:]

    xq_rot, xk_rot = apply_rope(xq_rot, xk_rot, freqs_cis)

    xq = torch.cat([xq_rot, xq_pass], dim=-1)
    xk = torch.cat([xk_rot, xk_pass], dim=-1)
    return xq, xk


class GatedAttention(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.inner_size = self.num_heads * self.head_dim
        self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)

        # q_proj outputs 2x: query + gate
        self.q_proj = ops.Linear(config.hidden_size, self.inner_size * 2, bias=config.qkv_bias, device=device, dtype=dtype)
        self.k_proj = ops.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias, device=device, dtype=dtype)
        self.v_proj = ops.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias, device=device, dtype=dtype)
        self.o_proj = ops.Linear(self.inner_size, config.hidden_size, bias=False, device=device, dtype=dtype)

        # QK norms with (1+weight) scaling
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)

    def forward(self, x, attention_mask=None, freqs_cis=None, optimized_attention=None, past_key_value=None, graph_decode=False):
        batch_size, seq_length, _ = x.shape

        # Project Q (with gate), K, V
        qg = self.q_proj(x)
        # Split into query and gate: each is [B, seq, inner_size]
        qg = qg.view(batch_size, seq_length, self.num_heads, self.head_dim * 2)
        xq, gate = qg[..., :self.head_dim], qg[..., self.head_dim:]
        gate = gate.reshape(batch_size, seq_length, -1)  # [B, seq, inner_size]

        xk = self.k_proj(x)
        xv = self.v_proj(x)

        xq = self.q_norm(xq).transpose(1, 2)  # [B, heads, seq, head_dim]
        xk = self.k_norm(xk.view(batch_size, seq_length, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        xv = xv.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply partial RoPE
        xq, xk = apply_partial_rope(xq, xk, freqs_cis, self.rotary_dim)

        # KV cache
        present_key_value = past_key_value
        if past_key_value is not None and seq_length <= 6 and attention_mask is None and graph_decode and past_key_value.index > 0:
            # CUDA-graphable decode: device-side write position, full-capacity biased attention
            cache = past_key_value
            cache.key.index_copy_(2, cache.position[:seq_length], xk)
            cache.value.index_copy_(2, cache.position[:seq_length], xv)
            output = fixed_kv_bias_decode(xq, cache, self.num_heads, self.num_kv_heads, self.head_dim)
        else:
            if past_key_value is not None:
                cache = past_key_value
                cache.key[:, :, cache.index:cache.index + seq_length] = xk
                cache.value[:, :, cache.index:cache.index + seq_length] = xv
                xk = cache.key[:, :, :cache.index + seq_length]
                xv = cache.value[:, :, :cache.index + seq_length]
            gqa_kwargs = {"enable_gqa": True} if self.num_heads != self.num_kv_heads else {}
            output = optimized_attention(xq, xk, xv, self.num_heads, mask=attention_mask, skip_reshape=True, **gqa_kwargs)

        output = output * gate.sigmoid()
        return self.o_proj(output), present_key_value


# Hybrid Transformer Block
class Qwen35TransformerBlock(nn.Module):
    def __init__(self, config, index, device=None, dtype=None, ops=None):
        super().__init__()
        self.layer_type = config.layer_types[index]
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config, device=device, dtype=dtype, ops=ops)
        else:
            self.self_attn = GatedAttention(config, device=device, dtype=dtype, ops=ops)
        self.mlp = MLP(config, device=device, dtype=dtype, ops=ops)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)

    def forward(self, x, attention_mask=None, freqs_cis=None, optimized_attention=None, past_key_value=None, graph_decode=None):
        output = x
        if self.layer_type == "linear_attention":
            h, present_key_value = self.linear_attn(self.input_layernorm(x), attention_mask=attention_mask, past_key_value=past_key_value)
        else:
            if graph_decode is None:
                # mirror the conditions under which prefetch_queue_pop can actually capture, so
                # eager fallbacks keep the sliced decode path instead of the full-capacity one
                graph_decode = (getattr(self, "_force_graph_decode", False)
                                or (getattr(self, "_v_block", None) is not None
                                    and comfy.model_management.NUM_STREAMS > 0
                                    and not comfy.model_management.args.disable_cuda_graphs
                                    and comfy.model_management.is_device_cuda(x.device)))
            h, present_key_value = self.self_attn(self.input_layernorm(x), attention_mask=attention_mask, freqs_cis=freqs_cis, optimized_attention=optimized_attention, past_key_value=past_key_value, graph_decode=graph_decode)

        # in-place into the input buffer so CUDA-graph replays land in the static x
        x = torch.add(x, h, out=output)
        x = torch.add(x, self.mlp(self.post_attention_layernorm(x)), out=output)
        return x, present_key_value


# Qwen35 Transformer Backbone
class Qwen35Transformer(Llama2_):
    def __init__(self, config, device=None, dtype=None, ops=None):
        nn.Module.__init__(self)
        self.config = config
        self.prefetch_dynamic_vbars = True
        self.graph_dynamic_vbar_blocks = True
        self.vocab_size = config.vocab_size
        self.embed_tokens = ops.Embedding(config.vocab_size, config.hidden_size, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            Qwen35TransformerBlock(config, index=i, device=device, dtype=dtype, ops=ops)
            for i in range(config.num_hidden_layers)
        ])

        if config.final_norm:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        else:
            self.norm = None

        if config.lm_head:
            self.lm_head = ops.Linear(config.hidden_size, config.vocab_size, bias=False, device=device, dtype=dtype)

    def compute_freqs_cis(self, position_ids, device):
        rotary_dim = int(self.config.head_dim * self.config.partial_rotary_factor)
        return precompute_freqs_cis(rotary_dim, position_ids, self.config.rope_theta,
                                    rope_dims=self.config.mrope_section, interleaved_mrope=True, device=device)


# Vision Encoder
class Qwen35VisionPatchEmbed(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.patch_size = config["patch_size"]
        self.temporal_patch_size = config["temporal_patch_size"]
        self.in_channels = config["in_channels"]
        self.embed_dim = config["hidden_size"]
        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = ops.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True, device=device, dtype=dtype)

    def forward(self, x):
        x = x.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        return self.proj(x).view(-1, self.embed_dim)


class Qwen35VisionMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, device=None, dtype=None, ops=None):
        super().__init__()

        self.linear_fc1 = ops.Linear(hidden_size, intermediate_size, bias=True, device=device, dtype=dtype)
        self.linear_fc2 = ops.Linear(intermediate_size, hidden_size, bias=True, device=device, dtype=dtype)

    def forward(self, hidden_state):
        return self.linear_fc2(F.gelu(self.linear_fc1(hidden_state), approximate="tanh"))


class Qwen35VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen):
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen35VisionAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, device=None, dtype=None, ops=None):
        super().__init__()

        self.dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.dim // self.num_heads
        self.qkv = ops.Linear(self.dim, self.dim * 3, bias=True, device=device, dtype=dtype)
        self.proj = ops.Linear(self.dim, self.dim, device=device, dtype=dtype)

    def forward(self, x, cu_seqlens, position_embeddings, optimized_attention=None):
        seq_length = x.shape[0]
        query_states, key_states, value_states = (
            self.qkv(x).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        query_states, key_states = apply_rope(query_states, key_states, position_embeddings)

        # Process per-sequence attention
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        q_splits = torch.split(query_states, lengths, dim=0)
        k_splits = torch.split(key_states, lengths, dim=0)
        v_splits = torch.split(value_states, lengths, dim=0)

        attn_outputs = []
        for q, k, v in zip(q_splits, k_splits, v_splits):
            q = q.transpose(0, 1).unsqueeze(0)
            k = k.transpose(0, 1).unsqueeze(0)
            v = v.transpose(0, 1).unsqueeze(0)
            attn_outputs.append(optimized_attention(q, k, v, self.num_heads, skip_reshape=True))

        attn_output = torch.cat(attn_outputs, dim=1)
        attn_output = attn_output.reshape(seq_length, -1)
        return self.proj(attn_output)


class Qwen35VisionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, device=None, dtype=None, ops=None):
        super().__init__()

        self.norm1 = ops.LayerNorm(hidden_size, eps=1e-6, device=device, dtype=dtype)
        self.norm2 = ops.LayerNorm(hidden_size, eps=1e-6, device=device, dtype=dtype)
        self.attn = Qwen35VisionAttention(hidden_size, num_heads, device=device, dtype=dtype, ops=ops)
        self.mlp = Qwen35VisionMLP(hidden_size, intermediate_size, device=device, dtype=dtype, ops=ops)

    def forward(self, x, cu_seqlens, position_embeddings, optimized_attention=None):
        x = x + self.attn(self.norm1(x), cu_seqlens=cu_seqlens, position_embeddings=position_embeddings, optimized_attention=optimized_attention)
        return x + self.mlp(self.norm2(x))


class Qwen35VisionPatchMerger(nn.Module):
    def __init__(self, hidden_size, spatial_merge_size, out_hidden_size, device=None, dtype=None, ops=None):
        super().__init__()

        merge_dim = hidden_size * (spatial_merge_size ** 2)
        self.norm = ops.LayerNorm(hidden_size, eps=1e-6, device=device, dtype=dtype)
        self.linear_fc1 = ops.Linear(merge_dim, merge_dim, device=device, dtype=dtype)
        self.linear_fc2 = ops.Linear(merge_dim, out_hidden_size, device=device, dtype=dtype)
        self.merge_dim = merge_dim

    def forward(self, x):
        x = self.norm(x).view(-1, self.merge_dim)
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


class Qwen35VisionModel(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.spatial_merge_size = config["spatial_merge_size"]
        self.patch_size = config["patch_size"]
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_heads"]
        self.num_position_embeddings = config["num_position_embeddings"]

        self.patch_embed = Qwen35VisionPatchEmbed(config, device=device, dtype=dtype, ops=ops)
        self.pos_embed = ops.Embedding(self.num_position_embeddings, self.hidden_size, device=device, dtype=dtype)
        self.num_grid_per_side = int(self.num_position_embeddings ** 0.5)
        self.rotary_pos_emb = Qwen35VisionRotaryEmbedding(self.hidden_size // self.num_heads // 2)
        self.blocks = nn.ModuleList([
            Qwen35VisionBlock(self.hidden_size, self.num_heads, config["intermediate_size"], device=device, dtype=dtype, ops=ops)
            for _ in range(config["depth"])
        ])
        self.merger = Qwen35VisionPatchMerger(self.hidden_size, self.spatial_merge_size, config["out_hidden_size"], device=device, dtype=dtype, ops=ops)
        self.deepstack_visual_indexes = [] # DeepStack, per-layer visual features (Qwen3-VL)
        self.deepstack_merger_list = None

    def rot_pos_emb(self, grid_thw):
        merge_size = self.spatial_merge_size
        grid_thw_list = grid_thw.tolist()
        max_hw = max(max(h, w) for _, h, w in grid_thw_list)
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device
        total_tokens = sum(int(t * h * w) for t, h, w in grid_thw_list)
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)
        offset = 0
        for num_frames, height, width in grid_thw_list:
            num_frames, height, width = int(num_frames), int(height), int(width)
            merged_h, merged_w = height // merge_size, width // merge_size
            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)
            num_tokens = coords.shape[0]
            pos_ids[offset:offset + num_tokens] = coords
            offset += num_tokens
        embeddings = freq_table[pos_ids]
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_thw_list = grid_thw.tolist()
        grid_ts = [int(row[0]) for row in grid_thw_list]
        grid_hs = [int(row[1]) for row in grid_thw_list]
        grid_ws = [int(row[2]) for row in grid_thw_list]
        device = self.pos_embed.weight.device
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]
        for t, h, w in grid_thw_list:
            h, w = int(h), int(w)
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h, device=device)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w, device=device)
            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor
            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side
            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for j in range(4):
                idx_list[j].extend(indices[j].tolist())
                weight_list[j].extend(weights[j].tolist())
        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])
        patch_pos_embeds_permute = []
        merge_size = self.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        return torch.cat(patch_pos_embeds_permute)

    def forward(self, x, grid_thw):
        x = self.patch_embed(x)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw).to(x.device)
        x = x + pos_embeds
        rotary_pos_emb = self.rot_pos_emb(grid_thw).to(x.device)
        seq_len = x.shape[0]
        x = x.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos().unsqueeze(-2)
        sin = emb.sin().unsqueeze(-2)
        sin_half = sin.shape[-1] // 2
        position_embeddings = (cos, sin[..., :sin_half], -sin[..., sin_half:])
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        optimized_attention = optimized_attention_for_device(x.device, mask=False, small_input=True)
        deepstack_features = []
        for layer_num, blk in enumerate(self.blocks):
            x = blk(x, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings, optimized_attention=optimized_attention)
            if self.deepstack_merger_list is not None and layer_num in self.deepstack_visual_indexes:
                deepstack_features.append(self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](x))
        merged = self.merger(x)
        if self.deepstack_merger_list is not None:
            return merged, deepstack_features
        return merged

class MTPHead(nn.Module):
    # multi-token-prediction draft head: fc(cat[norm(embed), norm(hidden)]) -> one
    # full-attention block (own KV) -> norm; logits via the shared lm_head
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.fc = ops.Linear(config.hidden_size * 2, config.hidden_size, bias=False, device=device, dtype=dtype)
        self.pre_fc_norm_embedding = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        self.pre_fc_norm_hidden = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        self.layers = nn.ModuleList([Qwen35TransformerBlock(config, index=3, device=device, dtype=dtype, ops=ops)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)

    def forward(self, embeds, hidden, freqs_cis, past_key_value):
        x = self.fc(torch.cat([self.pre_fc_norm_embedding(embeds), self.pre_fc_norm_hidden(hidden)], dim=-1))
        attention = optimized_attention_for_device(x.device, mask=False, small_input=True)
        # graph_decode: the bias-decode path is device-indexed, so a captured draft
        # graph stays correct as the cache position advances (the sliced fallback
        # would bake the capture-time python index into the graph)
        x, _ = self.layers[0](x, attention_mask=None, freqs_cis=freqs_cis, optimized_attention=attention, past_key_value=past_key_value, graph_decode=True)
        return self.norm(x), x  # (for logits, pre-norm hidden for recursive drafting)


# Model Wrapper
class Qwen35(BaseLlama, BaseGenerate, torch.nn.Module):
    model_type = "qwen35_2b"

    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = _make_config(self.model_type, config_dict)
        self.num_layers = config.num_hidden_layers
        self.model = Qwen35Transformer(config, device=device, dtype=dtype, ops=operations)
        self.mtp = None
        if config.mtp:
            self.mtp = MTPHead(config, device=device, dtype=dtype, ops=operations)
        vision_overrides = QWEN35_MODELS.get(self.model_type, {}).get("vision", {})
        vision_config = {**QWEN35_VISION_DEFAULTS, **vision_overrides, "out_hidden_size": config.hidden_size}
        self.visual = Qwen35VisionModel(vision_config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

    def preprocess_embed(self, embed, device):
        if embed["type"] == "image":
            # Qwen3.5 normalizes to [-1, 1] (mean/std 0.5), same as Qwen3-VL.
            image, grid = comfy.text_encoders.qwen_vl.process_qwen2vl_images(embed["data"], patch_size=16, image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
            return self.visual(image.to(device, dtype=torch.float32), grid), grid
        return None, None

    def forward(self, x, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None, final_layer_norm_intermediate=True, dtype=None, embeds_info=[], past_key_values=None):
        position_ids = comfy.text_encoders.qwen_vl.qwen2vl_mrope_position_ids(embeds_info, embeds.shape[1], embeds.device)
        return super().forward(x, attention_mask=attention_mask, embeds=embeds, num_tokens=num_tokens, intermediate_output=intermediate_output, final_layer_norm_intermediate=final_layer_norm_intermediate, dtype=dtype, position_ids=position_ids, past_key_values=past_key_values)

    def generate(self, embeds=None, do_sample=True, max_length=256, temperature=1.0, top_k=50, top_p=0.9, min_p=0.0, repetition_penalty=1.0, seed=42, stop_tokens=None, **kwargs):
        greedy = (not do_sample) or temperature == 0.0
        mtp = kwargs.pop("mtp", True)
        common = (self.mtp is not None and mtp
                  and kwargs.get("position_ids") is None
                  and kwargs.get("initial_input_ids") is None)
        spec = common and greedy and repetition_penalty == 1.0 and not kwargs.get("presence_penalty", 0.0)
        sampled = common and not greedy
        if not (spec or sampled):
            return super().generate(embeds=embeds, do_sample=do_sample, max_length=max_length, temperature=temperature,
                                    top_k=top_k, top_p=top_p, min_p=min_p, repetition_penalty=repetition_penalty,
                                    seed=seed, stop_tokens=stop_tokens, **kwargs)
        sampling = None
        if sampled:
            sampling = {"temperature": temperature, "top_k": top_k, "top_p": top_p, "min_p": min_p,
                        "repetition_penalty": repetition_penalty,
                        "presence_penalty": kwargs.get("presence_penalty", 0.0) or 0.0,
                        "seed": seed if seed is not None else 42}
        fixed_depth = None if mtp is True else max(2, min(5, int(mtp)))
        return self._generate_mtp(embeds, max_length, stop_tokens, sampling=sampling, fixed_depth=fixed_depth)

    def _generate_mtp(self, embeds, max_length, stop_tokens, sampling=None, fixed_depth=None):
        device = embeds.device
        cfg = self.model.config
        if stop_tokens is None:
            stop_tokens = cfg.stop_tokens
        dt = torch.bfloat16 if comfy.model_management.should_use_bf16(device) else torch.float32
        embeds = embeds.to(dt)
        if embeds.ndim == 2:
            embeds = embeds.unsqueeze(0)
        # greedy acceptance sustains a third draft; sampled acceptance at typical
        # temperatures does not (measured net-negative), so it stays at two.
        # High-acceptance greedy runs deepen to 5 once, after a probe window.
        depth = fixed_depth if fixed_depth is not None else (3 if sampling is None else 2)
        cap = embeds.shape[1] + max_length + 7
        pkv = self.init_kv_cache(embeds.shape[0], cap, device, dt)
        for kv in pkv:
            if isinstance(kv, LinearKV):
                # recurrent snapshots share one contiguous backing so the fused
                # kernel can write all of them; the tuples hold views into it
                kv.snap_backing = torch.empty((depth,) + tuple(kv.recurrent_state.shape), device=device, dtype=torch.float32)
                kv.snapshots = [(kv.snap_backing[s], torch.empty_like(kv.conv_state)) for s in range(depth)]
        mkey = torch.zeros([embeds.shape[0], cfg.num_key_value_heads, cap, cfg.head_dim], device=device, dtype=dt)
        # 6 rows: the prepare repair window spans [i - rows + 1, i + rows), which must
        # cover slots unmasked while drafting `depth` ahead followed by a near-full
        # rollback — rows >= depth + 1 keeps no stale slot unmasked
        mtp_kv = FixedKVBias(mkey, torch.zeros_like(mkey), 0,
                             torch.empty((6,), device=device, dtype=torch.int64), None,
                             torch.full((1, 1, 6, cap), torch.finfo(dt).min, device=device, dtype=dt), {"step": -1})

        # keep the per-step hot weights resident for the whole generate: lm_head and
        # embed_tokens run outside the captured graphs and are otherwise evicted on
        # every unpin once the vbar budget is tight (47 ms re-upload per logits call)
        hot = [self.model.lm_head, self.model.embed_tokens] + list(self.mtp.modules())
        pinned = [m for m in hot if hasattr(m, "_v")]
        if pinned:
            comfy.ops.cast_modules_with_vbar(pinned, None, device, None, True, return_faulted=True)

        nl = cfg.num_hidden_layers
        generator = None
        if sampling is not None:
            generator = torch.Generator(device=device).manual_seed(sampling["seed"])
        x, inter, pkv = self.model.forward(None, embeds=embeds, attention_mask=None, past_key_values=pkv, intermediate_output=nl - 1)
        if sampling is None:
            nt = self.logits(x)[:, -1].argmax(dim=-1, keepdim=True)
        else:
            nt = self.sample_token(self.logits(x)[:, -1], sampling["temperature"], sampling["top_k"], sampling["top_p"], sampling["min_p"],
                                   sampling["repetition_penalty"], [], generator, presence_penalty=sampling["presence_penalty"])
        h = inter[:, -1:, :].to(dt)
        ids = [nt[0].item()]
        pos = embeds.shape[1]
        progress = comfy.utils.ProgressBar(max_length)
        console = tqdm(total=max_length, desc="Generating tokens", initial=1)

        def update_progress(n):
            progress.update(n)
            console.update(min(n, max(0, max_length - console.n)))

        # rope table once per generate: drafts and verify slice it instead of
        # rebuilding a position tensor and recomputing freqs every step
        ftab = self.model.compute_freqs_cis(torch.arange(cap, device=device, dtype=torch.float).unsqueeze(0), device)

        def freqs_at(p, n=1):
            return tuple(t[:, :, p:p + n] for t in ftab)

        use_graph = (device.type == "cuda"
                     and comfy.model_management.NUM_STREAMS > 0
                     and not comfy.model_management.args.disable_cuda_graphs)
        draft_state = {}

        def drop_draft_graph():
            # the allocator's benign "uncaptured free of a captured allocation"
            # notice is only catchable when the free happens inside a torch API
            # call (set_()/reset()); plain refcount-drop dealloc prints to stderr
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for t in (draft_state.get("d"), draft_state.get("r"), *draft_state.get("keep", ())):
                    if t is not None:
                        t.set_()
                g = draft_state.pop("graph", None)
                if g is not None:
                    g.reset()
                draft_state.clear()

        def draft(token, hidden, p):
            # one drafted token: mtp head + lm_head argmax, graph-replayed on cuda
            mtp_kv.prepare(1)
            f = freqs_at(p)
            if not use_graph:
                n1, r1 = self.mtp(self.model.embed_tokens(token).to(dt), hidden, f, mtp_kv)
                mtp_kv.advance(1)
                return self.logits(n1)[:, -1].argmax(dim=-1, keepdim=True), r1
            if "graph" not in draft_state:
                ds = draft_state
                ds["tok"] = token.clone()
                ds["hid"] = hidden.clone()
                ds["f"] = tuple(t.clone() for t in f)
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(2):
                        n1, r1 = self.mtp(self.model.embed_tokens(ds["tok"]).to(dt), ds["hid"], ds["f"], mtp_kv)
                        self.logits(n1)[:, -1].argmax(dim=-1, keepdim=True)
                torch.cuda.current_stream().wait_stream(side)
                del n1, r1  # free warmup temporaries outside the capture, not shadowed inside it
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    n1, r1 = self.mtp(self.model.embed_tokens(ds["tok"]).to(dt), ds["hid"], ds["f"], mtp_kv)
                    lg1 = self.logits(n1)
                    ds["d"] = lg1[:, -1].argmax(dim=-1, keepdim=True)
                    ds["r"] = r1
                    ds["keep"] = (n1, lg1)  # captured allocations must outlive the graph
                ds["graph"] = g
            ds = draft_state
            ds["tok"].copy_(token)
            ds["hid"].copy_(hidden)
            for buf, val in zip(ds["f"], f):
                buf.copy_(val)
            ds["graph"].replay()
            mtp_kv.advance(1)
            return ds["d"], ds["r"]

        def col_dist(row, extra):
            # per-column target distribution: committed-history penalties host-side,
            # in-window draft tokens overlaid GPU-side (values unknown until the sync)
            s = sampling
            rp, pp = s["repetition_penalty"], s["presence_penalty"]
            if rp != 1.0 or pp != 0.0:
                seen = []
                if ids:
                    hist = torch.tensor(list(set(ids)), device=device)
                    tl = row[:, hist]
                    if rp != 1.0:
                        tl = torch.where(tl < 0, tl * rp, tl / rp)
                    if pp != 0.0:
                        tl = tl - pp
                    row[:, hist] = tl
                    seen.append(hist)
                for tok in extra:
                    dup = torch.zeros((), device=device, dtype=torch.bool)
                    for prev in seen:
                        dup = dup | (prev == tok).any()
                    v = row.gather(1, tok)
                    nv = torch.where(v < 0, v * rp, v / rp) if rp != 1.0 else v
                    if pp != 0.0:
                        nv = nv - pp
                    row.scatter_(1, tok, torch.where(dup, v, nv))
                    seen.append(tok)
            return self.processed_probs(row, s["temperature"], s["top_k"], s["top_p"], s["min_p"], 1.0, [])

        def prob_of(probs, idx, tok):
            if idx is None:
                return probs.gather(1, tok).reshape(())
            return (probs * (idx == tok)).sum()

        def draw(probs, idx, exclude=None):
            # residual (max(0, p - q) for one-hot q = p with the draft zeroed);
            # the rescue term only matters when the result will be discarded anyway
            w = probs
            if exclude is not None:
                if idx is None:
                    w = probs.clone()
                    w.scatter_(1, exclude, 0.0)
                else:
                    w = probs * (idx != exclude)
                w = w + (w.sum() == 0) * probs
            tok = torch.multinomial(w, num_samples=1, generator=generator)
            if idx is not None:
                return idx.gather(1, tok)
            return tok

        probe = None if fixed_depth is not None else [0, 0]  # steps, accepted drafts — settles the draft depth once
        try:
            while len(ids) < max_length and ids[-1] not in stop_tokens:
                # draft `depth` tokens recursively (each consumes the prior draft's
                # hidden); non-final drafts are cloned because later replays overwrite
                # the static graph output
                drafts = []
                tok_in, hid_in = nt, h
                for k in range(depth):
                    dk, rk = draft(tok_in, hid_in, pos + k)
                    if k < depth - 1:
                        dk = dk.clone()
                    drafts.append(dk)
                    tok_in, hid_in = dk, rk
                ev = self.model.embed_tokens(torch.cat([nt] + drafts, dim=1)).to(dt)
                x, inter, pkv = self.model.forward(None, embeds=ev, attention_mask=None, past_key_values=pkv, intermediate_output=nl - 1, freqs_cis=freqs_at(pos, depth + 1))
                # all verify positions in one lm_head GEMV, accept decided GPU-side, one sync
                with comfy.ops.CastBiasWeightContext(self.model.lm_head, x, offloadable=True) as (w, _bias):
                    lg = F.linear(x, w)
                if sampling is None:
                    toks = lg.argmax(dim=-1)
                    vals = torch.cat([toks[0]] + [d[0] for d in drafts]).tolist()
                    t, dr = vals[:depth + 1], vals[depth + 1:]
                    accepts = 0
                    while accepts < depth and t[accepts] == dr[accepts]:
                        accepts += 1
                    next_toks = tuple(toks[:, i:i + 1] for i in range(depth + 1))
                    commit = tuple(dr[:accepts]) + (t[accepts],)
                else:
                    # rejection sampling with a one-hot (greedy) draft: accept draft i with
                    # probability p_i(draft), correct from the residual, bonus from the tail
                    lgf = lg.float()
                    dists = tuple(col_dist(lgf[:, c], drafts[:c]) for c in range(depth + 1))
                    corr = [draw(p, i, exclude=drafts[c]) for c, (p, i) in enumerate(dists[:depth])]
                    corr.append(draw(*dists[depth]))
                    u = torch.rand(depth, device=device, generator=generator)
                    a = torch.zeros((), device=device, dtype=torch.long)
                    live = torch.ones((), device=device, dtype=torch.bool)
                    for c in range(depth):
                        live = live & (u[c] < prob_of(*dists[c], drafts[c]))
                        a = a + live.long()
                    vals = torch.cat([d[0] for d in drafts] + [c[0] for c in corr] + [a.reshape(1)]).tolist()
                    dr, cv, accepts = vals[:depth], vals[depth:2 * depth + 1], vals[-1]
                    next_toks = tuple(corr)
                    commit = tuple(dr[:accepts]) + (cv[accepts],)
                if accepts < depth:
                    for kv in pkv:
                        kv.rollback(depth - accepts)
                if accepts < depth - 1:
                    mtp_kv.rollback(depth - 1 - accepts)  # mtp entries fed by a rejected draft token
                ids.extend(commit)
                nt = next_toks[accepts]
                h = inter[:, accepts:accepts + 1, :].to(dt)
                pos += accepts + 1
                update_progress(accepts + 1)
                if probe is not None:
                    probe[0] += 1
                    probe[1] += accepts
                    if probe[0] == 32:
                        # deepen once when acceptance sustains it and enough tokens
                        # remain to amortize the recapture round the new verify shape
                        # costs; the graph machinery needs full boundary semantics for
                        # a shape change (a bare queue rebuild leaves stale staging)
                        if sampling is None and max_length - len(ids) > 512 and 1 + probe[1] / probe[0] >= 2.2:
                            for kv in pkv:
                                if isinstance(kv, LinearKV):
                                    kv.snap_backing = torch.empty((5,) + tuple(kv.recurrent_state.shape), device=device, dtype=torch.float32)
                                    kv.snapshots = [(kv.snap_backing[s], torch.empty_like(kv.conv_state)) for s in range(5)]
                            comfy.model_management.reset_cast_buffers()
                            comfy.model_prefetch.cleanup_prefetch_queues()
                            drop_draft_graph()  # recapture drafts against the fresh cast state
                            depth = 5
                        probe = None
        finally:
            console.close()
            drop_draft_graph()
            if pinned:
                comfy.model_prefetch.cleanup_prefetched_modules(None, pinned)
        for j, tk in enumerate(ids):
            if tk in stop_tokens:
                return ids[:j + 1]
        return ids[:max_length]

    def init_kv_cache(self, batch, max_cache_len, device, execution_dtype):
        model_config = self.model.config
        past_key_values = []
        # all full-attention layers advance in lockstep, so they share one position/bias/tracker
        position = torch.empty((6,), device=device, dtype=torch.int64)
        bias = torch.full((1, 1, 6, max_cache_len), torch.finfo(execution_dtype).min, device=device, dtype=execution_dtype)
        tracker = {"step": -1}
        for i in range(model_config.num_hidden_layers):
            if model_config.layer_types[i] == "linear_attention":
                recurrent_state = torch.zeros(
                    [batch, model_config.linear_num_value_heads, model_config.linear_key_head_dim, model_config.linear_value_head_dim],
                    device=device, dtype=torch.float32
                )
                conv_dim = model_config.linear_num_key_heads * model_config.linear_key_head_dim * 2 + model_config.linear_num_value_heads * model_config.linear_value_head_dim
                conv_state = torch.zeros(
                    [batch, conv_dim, model_config.conv_kernel_size - 1],
                    device=device, dtype=execution_dtype
                )
                past_key_values.append(LinearKV(conv_state, recurrent_state, 0, None, None))
            else:
                # zero-init: decode attends full capacity with masked tails, 0*0 stays finite
                key = torch.zeros([batch, model_config.num_key_value_heads, max_cache_len, model_config.head_dim], device=device, dtype=execution_dtype)
                past_key_values.append(FixedKVBias(key, torch.zeros_like(key), 0, position, None, bias, tracker))
        return past_key_values

# Tokenizer and Text Encoder Wrappers

class Qwen35Tokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}, embedding_size=2048, embedding_key="qwen35_2b"):
        from transformers import Qwen2Tokenizer
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "qwen35_tokenizer")
        super().__init__(tokenizer_path, pad_with_end=False, embedding_directory=embedding_directory, embedding_size=embedding_size, embedding_key=embedding_key, tokenizer_class=Qwen2Tokenizer,
            has_start_token=False, has_end_token=False, pad_to_max_length=False, max_length=99999999, min_length=1, pad_token=248044, tokenizer_data=tokenizer_data)


class Qwen35ImageTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}, model_type="qwen35_2b"):
        embedding_size = QWEN35_MODELS.get(model_type, {}).get("hidden_size", 2048)
        tokenizer = lambda *a, **kw: Qwen35Tokenizer(*a, **kw, embedding_size=embedding_size, embedding_key=model_type)
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name=model_type, tokenizer=tokenizer)
        self.llama_template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.llama_template_images = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"

    def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None, images=[], prevent_empty_text=False, thinking=False, **kwargs):
        image = kwargs.get("image", None)
        if image is not None and len(images) == 0:
            images = [image[i:i + 1] for i in range(image.shape[0])]

        skip_template = False
        if text.startswith('<|im_start|>'):
            skip_template = True
        if prevent_empty_text and text == '':
            text = ' '

        if skip_template:
            llama_text = text
        else:
            if llama_template is not None:
                template = llama_template
            elif len(images) == 0:
                template = self.llama_template
            else:
                template = self.llama_template_images
                if len(images) > 1:
                    vision_block = "<|vision_start|><|image_pad|><|vision_end|>"
                    template = template.replace(vision_block, vision_block * len(images), 1)
            llama_text = template.format(text)
            if not thinking:
                llama_text += "<think>\n</think>\n"

        tokens = super().tokenize_with_weights(llama_text, return_word_ids=return_word_ids, disable_weights=True, **kwargs)
        key_name = next(iter(tokens))
        embed_count = 0
        qwen_tokens = tokens[key_name]
        for r in qwen_tokens:
            for i in range(len(r)):
                if r[i][0] == 248056:  # <|image_pad|>
                    if len(images) > embed_count:
                        r[i] = ({"type": "image", "data": images[embed_count], "original_type": "image"},) + r[i][1:]
                        embed_count += 1
        return tokens


class Qwen35ClipModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="hidden", layer_idx=-2, dtype=None, attention_mask=True, model_options={}, model_type="qwen35_2b", mtp=False):
        class Qwen35_(Qwen35):
            pass
        Qwen35_.model_type = model_type

        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config={"mtp": True} if mtp else {},
            dtype=dtype, special_tokens={"pad": 248044}, layer_norm_hidden_state=False,
            model_class=Qwen35_, enable_attention_masks=attention_mask, return_attention_masks=attention_mask, model_options=model_options)

    def generate(self, tokens, do_sample, max_length, temperature, top_k, top_p, min_p, repetition_penalty, seed, presence_penalty=0.0, mtp=True):
        if isinstance(tokens, dict):
            tokens = next(iter(tokens.values()))
        tokens_only = [[t[0] for t in b] for b in tokens]
        embeds, _, _, embeds_info = self.process_tokens(tokens_only, self.execution_device)
        position_ids = comfy.text_encoders.qwen_vl.qwen2vl_mrope_position_ids(embeds_info, embeds.shape[1], embeds.device)
        return self.transformer.generate(embeds, do_sample, max_length, temperature, top_k, top_p, min_p, repetition_penalty, seed,
                                         presence_penalty=presence_penalty, position_ids=position_ids, mtp=mtp)


class Qwen35TEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}, model_type="qwen35_2b", mtp=False):
        clip_model = lambda **kw: Qwen35ClipModel(**kw, model_type=model_type, mtp=mtp)
        super().__init__(device=device, dtype=dtype, name=model_type, clip_model=clip_model, model_options=model_options)


def tokenizer(model_type="qwen35_2b"):
    class Qwen35ImageTokenizer_(Qwen35ImageTokenizer):
        def __init__(self, embedding_directory=None, tokenizer_data={}):
            super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, model_type=model_type)
    return Qwen35ImageTokenizer_


def te(dtype_llama=None, llama_quantization_metadata=None, model_type="qwen35_2b", mtp=False):
    class Qwen35TEModel_(Qwen35TEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_llama is not None:
                dtype = dtype_llama
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            super().__init__(device=device, dtype=dtype, model_options=model_options, model_type=model_type, mtp=mtp)
    return Qwen35TEModel_
