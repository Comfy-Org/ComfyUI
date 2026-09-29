import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from tqdm import tqdm
import contextlib
import os
import warnings

import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy_kitchen
from comfy.quant_ops import QuantizedTensor
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy import sd1_clip
import comfy.text_encoders.qwen_vl

from .llama import BaseLlama, BaseGenerate, FixedKV, FixedKVBias, Llama2_, MLP, RMSNorm, apply_penalty, apply_rope, penalty_active, precompute_freqs_cis, rope_matrix


@dataclass
class LinearKV(FixedKV):
    # DeltaNet state on the FixedKV interface: key=conv_state, value=recurrent_state (fp32)
    g_decay: torch.Tensor = None
    dt_bias: torch.Tensor = None
    snapshots: list = None  # [(recurrent, conv)] taken after step 1, 2, ... of the last verify
    last_seq: int = 1
    snap_backing: torch.Tensor = None
    conv_snap_backing: torch.Tensor = None
    norm_weight: torch.Tensor = None

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

        fused_available = getattr(comfy_kitchen, "gated_delta_decode_is_available", None)
        use_fused = (use_recurrent and fused_available is not None and fused_available(x.device, self.key_head_dim, self.value_head_dim)
                     and (seq_len == 1 or past_key_value.snap_backing is not None))

        # Projections (shared)
        proj = self.in_proj_qkv(x)  # [B, seq_len, conv_dim]
        z = self.in_proj_z(x)

        if use_fused:
            # decode: kitchen conv step, then gates + delta rule + gated norm in one kernel
            if seq_len > 1:
                past_key_value.last_seq = seq_len
            with comfy.ops.CastBiasWeightContext(self.conv1d, proj, offloadable=True) as (conv_weight, conv_bias):
                conv_out = comfy_kitchen.deltanet_conv_step(proj, past_key_value.conv_state, conv_weight, conv_bias,
                                                            past_key_value.conv_snap_backing[:seq_len - 1] if seq_len > 1 else None)
            with comfy.ops.CastBiasWeightContext(self.in_proj_a, x, offloadable=True) as (w_a, _), \
                 comfy.ops.CastBiasWeightContext(self.in_proj_b, x, offloadable=True) as (w_b, _):
                if isinstance(w_a, QuantizedTensor):
                    w_a = w_a.dequantize()
                if isinstance(w_b, QuantizedTensor):
                    w_b = w_b.dequantize()
                core_attn_out = comfy_kitchen.gated_delta_decode_fused(
                    conv_out, x, w_a, w_b, past_key_value.dt_bias, past_key_value.g_decay, past_key_value.recurrent_state,
                    self.key_dim, self.num_key_heads, self.key_head_dim ** -0.5, z, past_key_value.norm_weight, self.norm.eps,
                    past_key_value.snap_backing[:seq_len - 1] if seq_len > 1 else None)
            return self.out_proj(core_attn_out.reshape(batch_size, seq_len, -1)), past_key_value

        mixed_qkv = proj.transpose(1, 2)  # [B, conv_dim, seq_len]
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
                past_key_value.norm_weight = comfy.model_management.cast_to(self.norm.weight, dtype=x.dtype, device=x.device)

        beta = b.sigmoid()
        g = g_decay * F.softplus(a.float() + dt_bias)
        query = query.reshape(batch_size, seq_len, self.num_key_heads, self.key_head_dim)
        key = key.reshape(batch_size, seq_len, self.num_key_heads, self.key_head_dim)
        value = value.reshape(batch_size, seq_len, self.num_value_heads, self.value_head_dim)
        if self.num_value_heads != self.num_key_heads:
            rep = self.num_value_heads // self.num_key_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

        # Delta rule
        if use_recurrent:
            scale = self.key_head_dim ** -0.5
            q = F.normalize(query.float(), dim=-1) * scale
            k = F.normalize(key.float(), dim=-1)
            v = value.float()
            beta_t = beta.reshape(batch_size, seq_len, -1)
            g_t = g.reshape(batch_size, seq_len, -1).exp()

            # In-place state update: [B, heads, k_dim, v_dim]
            recurrent_state = past_key_value.recurrent_state
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

    def forward(self, x, attention_mask=None, freqs_cis=None, optimized_attention=None, past_key_value=None):
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
        if past_key_value is not None and seq_length <= 6 and attention_mask is None:
            output = past_key_value.decode(xq, xk, xv, self.num_kv_heads)
        else:
            if past_key_value is not None:
                xk, xv = past_key_value.append(xk, xv)
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

    def forward(self, x, attention_mask=None, freqs_cis=None, optimized_attention=None, past_key_value=None):
        output = x
        if self.layer_type == "linear_attention":
            h, present_key_value = self.linear_attn(self.input_layernorm(x), attention_mask=attention_mask, past_key_value=past_key_value)
        else:
            h, present_key_value = self.self_attn(self.input_layernorm(x), attention_mask=attention_mask, freqs_cis=freqs_cis, optimized_attention=optimized_attention, past_key_value=past_key_value)

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
    # MTP draft head: fc(cat[norm(embed), norm(hidden)]) -> attention block (own KV) -> norm
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
        x, _ = self.layers[0](x, attention_mask=None, freqs_cis=freqs_cis, optimized_attention=attention, past_key_value=past_key_value)
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
        mtp = kwargs.pop("mtp", True)
        if self.mtp is None or not mtp or kwargs.get("position_ids") is not None or kwargs.get("initial_input_ids") is not None:
            return super().generate(embeds=embeds, do_sample=do_sample, max_length=max_length, temperature=temperature,
                                    top_k=top_k, top_p=top_p, min_p=min_p, repetition_penalty=repetition_penalty,
                                    seed=seed, stop_tokens=stop_tokens, **kwargs)
        sampling = None
        if do_sample and temperature != 0.0:
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
        # greedy drafts 3 deep (5 after the probe); sampled stays at 2
        depth = fixed_depth if fixed_depth is not None else (3 if sampling is None else 2)
        cap = embeds.shape[1] + max_length + 7
        pkv = self.init_kv_cache(embeds.shape[0], cap, device, dt)
        # repair window: drafting ahead plus a near-full rollback
        mtp_kv = FixedKVBias.zeros(embeds.shape[0], cfg.num_key_value_heads, cap, cfg.head_dim, device, dt,
                                   FixedKVBias.shared(cap, device, dt))
        head = self.model.lm_head if hasattr(self.model, "lm_head") else self.model.embed_tokens

        def verify_logits(x):
            if not head.comfy_cast_weights:
                return F.linear(x, head.weight.to(x), None)
            with comfy.ops.CastBiasWeightContext(head, x, offloadable=True) as (w, _bias):
                return F.linear(x, w)

        # the draft graph bakes these weights' addresses: keep them resident for the generate
        hot = list({id(m): m for m in [head, self.model.embed_tokens, *self.mtp.modules()]}.values())
        pinned = [m for m in hot if hasattr(m, "_v")]

        generator = None
        if sampling is not None:
            generator = torch.Generator(device=device).manual_seed(sampling["seed"])
        penalized = sampling is not None and penalty_active(sampling["repetition_penalty"], sampling["presence_penalty"])

        # rope table once per generate, sliced per draft
        ftab = rope_matrix(self.model.compute_freqs_cis(torch.arange(cap, device=device, dtype=torch.float).unsqueeze(0), device))

        def freqs_at(p, n=1):
            return ftab[:, :, p:p + n]

        # cross-step state lives in static carriers: nothing allocated inside a step may outlive it
        nt_buf = torch.empty((embeds.shape[0], 1), device=device, dtype=torch.long)
        h_buf = torch.empty((embeds.shape[0], 1, cfg.hidden_size), device=device, dtype=dt)
        x, _, _ = self.model.forward(None, embeds=embeds, attention_mask=None, past_key_values=pkv)
        lg0 = self.logits(x)[:, -1]
        if sampling is None:
            nt_buf.copy_(lg0.argmax(dim=-1, keepdim=True))
        else:
            nt_buf.copy_(self.sample_token(lg0, sampling["temperature"], sampling["top_k"], sampling["top_p"], sampling["min_p"],
                                           sampling["repetition_penalty"], [], generator, presence_penalty=sampling["presence_penalty"]))
        pen_mask = torch.zeros((lg0.shape[-1],), device=device, dtype=torch.bool) if penalized else None
        h_buf.copy_(x[:, -1:, :])
        del x, lg0
        ids = [nt_buf[0].item()]
        if penalized:
            pen_mask.index_fill_(0, nt_buf.reshape(-1), True)
        pos = embeds.shape[1]
        progress = comfy.utils.ProgressBar(max_length)
        console = tqdm(total=max_length, desc="Generating tokens", initial=1)

        def update_progress(n):
            progress.update(n)
            console.update(n)

        verify_buffers = None
        snapshot_bytes = sum(kv.recurrent_state.numel() * 4 + kv.conv_state.numel() * kv.conv_state.element_size()
                             for kv in pkv if isinstance(kv, LinearKV))

        def set_depth(d):
            nonlocal depth, verify_buffers
            depth = d
            for kv in pkv:
                if isinstance(kv, LinearKV):
                    # snapshot views share one backing slab so the fused kernel can write them
                    kv.snap_backing = torch.empty((d,) + tuple(kv.recurrent_state.shape), device=device, dtype=torch.float32)
                    kv.conv_snap_backing = torch.empty((d,) + tuple(kv.conv_state.shape), device=device, dtype=kv.conv_state.dtype)
                    kv.snapshots = [(kv.snap_backing[s], kv.conv_snap_backing[s]) for s in range(d)]
            # static hidden/rope buffers: graphed layers bake their input addresses
            verify_buffers = (torch.empty((embeds.shape[0], d + 1, cfg.hidden_size), device=device, dtype=dt), freqs_at(pos, d + 1).clone())

        set_depth(depth)
        use_graph = (device.type == "cuda"
                     and comfy.model_management.NUM_STREAMS > 0
                     and not comfy.model_management.args.disable_cuda_graphs)
        compile_allocations = use_graph and self.model.graph_dynamic_vbar_blocks and comfy.model_prefetch.malloc_graph_enabled(device)
        draft_state = {}

        def drop_draft_graph():
            # free inside torch API calls so the allocator's benign notices stay catchable
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for t in (draft_state.get("d"), draft_state.get("r"), *draft_state.get("keep", ())):
                    if t is not None:
                        t.set_()
                g = draft_state.pop("graph", None)
                if g is not None:
                    g.reset()
                draft_state.clear()

        def draft_capture():
            # captured outside the compiler bracket; its static buffers live for the generate
            ds = draft_state
            mtp_kv.prepare(1)
            ds["tok"] = nt_buf.clone()
            ds["hid"] = h_buf.clone()
            ds["f"] = freqs_at(pos).clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    n1, r1 = self.mtp(self.model.embed_tokens(ds["tok"]).to(dt), ds["hid"], ds["f"], mtp_kv)
                    self.logits(n1)[:, -1].argmax(dim=-1, keepdim=True)
            torch.cuda.current_stream().wait_stream(side)
            del n1, r1  # freed before the capture, not shadowed inside it
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, capture_error_mode="thread_local"):
                n1, r1 = self.mtp(self.model.embed_tokens(ds["tok"]).to(dt), ds["hid"], ds["f"], mtp_kv)
                lg1 = self.logits(n1)
                ds["d"] = lg1[:, -1].argmax(dim=-1, keepdim=True)
                ds["r"] = r1
                ds["keep"] = (n1, lg1)  # captured allocations must outlive the graph
            ds["graph"] = g

        def draft(token, hidden, p):
            # one drafted token: mtp head + lm_head argmax, graph-replayed on cuda
            mtp_kv.prepare(1)
            f = freqs_at(p)
            if not use_graph:
                n1, r1 = self.mtp(self.model.embed_tokens(token).to(dt), hidden, f, mtp_kv)
                mtp_kv.advance(1)
                return self.logits(n1)[:, -1].argmax(dim=-1, keepdim=True), r1
            ds = draft_state
            ds["tok"].copy_(token)
            ds["hid"].copy_(hidden)
            ds["f"].copy_(f)
            ds["graph"].replay()
            mtp_kv.advance(1)
            return ds["d"], ds["r"]

        def verify_sample(lg, drafts):
            # accept draft i w.p. p_i(draft), else sample the residual; all depth+1 columns as one batch
            s = sampling
            rows = lg[0].float()
            if penalized:
                # committed tokens penalize every column, each draft the columns after it
                mask = pen_mask.unsqueeze(0).repeat(depth + 1, 1)
                for c, d in enumerate(drafts):
                    mask[c + 1:].index_fill_(1, d.reshape(-1), True)
                rows = torch.where(mask, apply_penalty(rows, s["repetition_penalty"], s["presence_penalty"]), rows)
            probs, idx = self.processed_probs(rows, s["temperature"], s["top_k"], s["top_p"], s["min_p"], 1.0, [])
            dr = torch.cat(drafts, dim=1).reshape(-1, 1)
            if idx is None:
                p_draft = probs[:depth].gather(1, dr).reshape(-1)
                w = probs.clone()
                w[:depth].scatter_(1, dr, 0.0)
            else:
                hit = idx[:depth] == dr
                p_draft = (probs[:depth] * hit).sum(1)
                w = probs.clone()
                w[:depth].masked_fill_(hit, 0.0)
            # residual: p with the draft zeroed (the rescue term only matters when discarded anyway)
            w = w + (w.sum(1, keepdim=True) == 0) * probs
            tok = torch.multinomial(w, num_samples=1, generator=generator)
            corr = tok if idx is None else idx.gather(1, tok)
            u = torch.rand(depth, device=device, generator=generator)
            accepted = (u < p_draft).long().cumprod(0).sum()
            return dr, corr, accepted

        def step():
            # scoped so every temporary dies before the compiler bracket closes
            nonlocal pos
            drafts = []
            tok_in, hid_in = nt_buf, h_buf
            for k in range(depth):
                dk, rk = draft(tok_in, hid_in, pos + k)
                if k < depth - 1:
                    dk = dk.clone()  # later replays overwrite the static output
                drafts.append(dk)
                tok_in, hid_in = dk, rk
            ev = self.model.embed_tokens(torch.cat([nt_buf] + drafts, dim=1)).to(dt)
            x, _, _ = self.model.forward(None, embeds=ev, attention_mask=None, past_key_values=pkv, decode_buffers=verify_buffers)
            # all verify positions in one lm_head GEMV, accept decided GPU-side, one sync
            lg = verify_logits(x)
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
                dr, corr, accepted = verify_sample(lg, drafts)
                vals = torch.cat([dr[:, 0], corr[:, 0], accepted.reshape(1)]).tolist()
                dr, cv, accepts = vals[:depth], vals[depth:2 * depth + 1], vals[-1]
                next_toks = tuple(corr[i:i + 1] for i in range(depth + 1))
                commit = tuple(dr[:accepts]) + (cv[accepts],)
            if accepts < depth:
                for kv in pkv:
                    kv.rollback(depth - accepts)
            if accepts < depth - 1:
                mtp_kv.rollback(depth - 1 - accepts)  # mtp entries fed by a rejected draft token
            nt_buf.copy_(next_toks[accepts])
            h_buf.copy_(x[:, accepts:accepts + 1, :])
            if penalized:
                for d in drafts[:accepts]:
                    pen_mask.index_fill_(0, d.reshape(-1), True)
                pen_mask.index_fill_(0, nt_buf.reshape(-1), True)
            pos += accepts + 1
            return accepts, commit

        probe = None if fixed_depth is not None else [0, 0]  # steps, accepted drafts
        try:
            if pinned:
                comfy.model_prefetch.pin_modules(pinned, device, dt)
            if use_graph and len(ids) < max_length and ids[-1] not in stop_tokens:
                draft_capture()
            while len(ids) < max_length and ids[-1] not in stop_tokens:
                with (comfy.model_prefetch.malloc_graph_scope(device) if compile_allocations else contextlib.nullcontext()):
                    accepts, commit = step()
                commit = list(commit[:max_length - len(ids)])
                stop = next((i for i, t in enumerate(commit) if t in stop_tokens), None)
                if stop is not None:
                    del commit[stop + 1:]
                ids.extend(commit)
                update_progress(len(commit))
                if probe is not None:
                    probe[0] += 1
                    probe[1] += accepts
                    if probe[0] == 32:
                        # deepen once acceptance sustains it and a recapture round can amortize
                        if (sampling is None and max_length - len(ids) > 512 and 1 + probe[1] / probe[0] >= 2.2
                                and 5 * snapshot_bytes < comfy.model_management.get_free_memory(device)):
                            comfy.model_prefetch.cleanup_prefetch_queues()
                            drop_draft_graph()
                            set_depth(5)
                            if use_graph:
                                draft_capture()
                        probe = None
        finally:
            console.close()
            drop_draft_graph()
            if pinned:
                comfy.model_prefetch.cleanup_prefetched_modules(None, pinned)
        return ids

    def init_kv_cache(self, batch, max_cache_len, device, execution_dtype):
        model_config = self.model.config
        past_key_values = []
        shared = FixedKVBias.shared(max_cache_len, device, execution_dtype)
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
                past_key_values.append(FixedKVBias.zeros(batch, model_config.num_key_value_heads, max_cache_len, model_config.head_dim, device, execution_dtype, shared))
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

    def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None, images=[], prevent_empty_text=False, thinking=False, system_prompt="", **kwargs):
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
            if system_prompt:
                llama_text = "<|im_start|>system\n" + system_prompt + "<|im_end|>\n" + llama_text
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
