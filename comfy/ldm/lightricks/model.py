from abc import ABC, abstractmethod
from enum import Enum
import functools
import logging
import math
from typing import Dict, Optional, Tuple

from einops import rearrange
import numpy as np
import torch
from torch import nn
import comfy.patcher_extension
import comfy.ldm.modules.attention
import comfy.ldm.common_dit

from .symmetric_patchifier import SymmetricPatchifier, latent_to_pixel_coords

logger = logging.getLogger(__name__)

def _log_base(x, base):
    return np.log(x) / np.log(base)

class LTXRopeType(str, Enum):
    INTERLEAVED = "interleaved"
    SPLIT = "split"

    KEY = "rope_type"

    @classmethod
    def from_dict(cls, kwargs, default=None):
        if default is None:
            default = cls.INTERLEAVED
        return cls(kwargs.get(cls.KEY, default))


class LTXFrequenciesPrecision(str, Enum):
    FLOAT32 = "float32"
    FLOAT64 = "float64"

    KEY = "frequencies_precision"

    @classmethod
    def from_dict(cls, kwargs, default=None):
        if default is None:
            default = cls.FLOAT32
        return cls(kwargs.get(cls.KEY, default))


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()

        self.linear_1 = operations.Linear(in_channels, time_embed_dim, sample_proj_bias, dtype=dtype, device=device)

        if cond_proj_dim is not None:
            self.cond_proj = operations.Linear(cond_proj_dim, in_channels, bias=False, dtype=dtype, device=device)
        else:
            self.cond_proj = None

        self.act = nn.SiLU()

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = operations.Linear(
            time_embed_dim, time_embed_dim_out, sample_proj_bias, dtype=dtype, device=device
        )

        if post_act_fn is None:
            self.post_act = None
        # else:
        #     self.post_act = get_activation(post_act_fn)

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb


class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    """
    For PixArt-Alpha.

    Reference:
    https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L164C9-L168C29
    """

    def __init__(
        self,
        embedding_dim,
        size_emb_dim,
        use_additional_conditions: bool = False,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()

        self.outdim = size_emb_dim
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim, dtype=dtype, device=device, operations=operations
        )

    def forward(self, timestep, resolution, aspect_ratio, batch_size, hidden_dtype):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)
        return timesteps_emb


class AdaLayerNormSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm single (adaLN-single).

    As proposed in PixArt-Alpha (see: https://arxiv.org/abs/2310.00426; Section 2.3).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        use_additional_conditions (`bool`): To use additional conditions for normalization or not.
    """

    def __init__(
        self, embedding_dim: int, embedding_coefficient: int = 6, use_additional_conditions: bool = False, dtype=None, device=None, operations=None
    ):
        super().__init__()

        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim,
            size_emb_dim=embedding_dim // 3,
            use_additional_conditions=use_additional_conditions,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        self.silu = nn.SiLU()
        self.linear = operations.Linear(embedding_dim, embedding_coefficient * embedding_dim, bias=True, dtype=dtype, device=device)

    def forward(
        self,
        timestep: torch.Tensor,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        batch_size: Optional[int] = None,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # No modulation happening here.
        added_cond_kwargs = added_cond_kwargs or {"resolution": None, "aspect_ratio": None}
        embedded_timestep = self.emb(timestep, **added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep


class PixArtAlphaTextProjection(nn.Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(
        self, in_features, hidden_size, out_features=None, act_fn="gelu_tanh", dtype=None, device=None, operations=None
    ):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = operations.Linear(
            in_features=in_features, out_features=hidden_size, bias=True, dtype=dtype, device=device
        )
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = operations.Linear(
            in_features=hidden_size, out_features=out_features, bias=True, dtype=dtype, device=device
        )

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class NormSingleLinearTextProjection(nn.Module):
    """Text projection for 20B models - single linear with RMSNorm (no activation)."""

    def __init__(
        self, in_features, hidden_size, dtype=None, device=None, operations=None
    ):
        super().__init__()
        if operations is None:
            operations = comfy.ops.disable_weight_init
        self.in_norm = operations.RMSNorm(
            in_features, eps=1e-6, elementwise_affine=False
        )
        self.linear_1 = operations.Linear(
            in_features, hidden_size, bias=True, dtype=dtype, device=device
        )
        self.hidden_size = hidden_size
        self.in_features = in_features

    def forward(self, caption):
        caption = self.in_norm(caption)
        caption = caption * (self.hidden_size / self.in_features) ** 0.5
        return self.linear_1(caption)


class GELU_approx(nn.Module):
    def __init__(self, dim_in, dim_out, dtype=None, device=None, operations=None):
        super().__init__()
        self.proj = operations.Linear(dim_in, dim_out, dtype=dtype, device=device)

    def forward(self, x):
        return torch.nn.functional.gelu(self.proj(x), approximate="tanh")


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out, mult=4, glu=False, dropout=0.0, dtype=None, device=None, operations=None):
        super().__init__()
        inner_dim = int(dim * mult)
        project_in = GELU_approx(dim, inner_dim, dtype=dtype, device=device, operations=operations)

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), operations.Linear(inner_dim, dim_out, dtype=dtype, device=device)
        )

    def forward(self, x):
        return self.net(x)

def apply_rotary_emb(input_tensor, freqs_cis):
    cos_freqs, sin_freqs = freqs_cis[0], freqs_cis[1]
    split_pe = freqs_cis[2] if len(freqs_cis) > 2 else False
    return (
        apply_split_rotary_emb(input_tensor, cos_freqs, sin_freqs)
        if split_pe else
        apply_interleaved_rotary_emb(input_tensor, cos_freqs, sin_freqs)
    )

def apply_interleaved_rotary_emb(input_tensor, cos_freqs, sin_freqs):  # TODO: remove duplicate funcs and pick the best/fastest one
    t_dup = rearrange(input_tensor, "... (d r) -> ... d r", r=2)
    t1, t2 = t_dup.unbind(dim=-1)
    t_dup = torch.stack((-t2, t1), dim=-1)
    input_tensor_rot = rearrange(t_dup, "... d r -> ... (d r)")

    out = input_tensor * cos_freqs + input_tensor_rot * sin_freqs

    return out

def apply_split_rotary_emb(input_tensor, cos, sin):
    needs_reshape = False
    if input_tensor.ndim != 4 and cos.ndim == 4:
        B, H, T, _ = cos.shape
        input_tensor = input_tensor.reshape(B, T, H, -1).swapaxes(1, 2)
        needs_reshape = True
    split_input = rearrange(input_tensor, "... (d r) -> ... d r", d=2)
    first_half_input = split_input[..., :1, :]
    second_half_input = split_input[..., 1:, :]
    output = split_input * cos.unsqueeze(-2)
    first_half_output = output[..., :1, :]
    second_half_output = output[..., 1:, :]
    first_half_output.addcmul_(-sin.unsqueeze(-2), second_half_input)
    second_half_output.addcmul_(sin.unsqueeze(-2), first_half_input)
    output = rearrange(output, "... d r -> ... (d r)")
    return output.swapaxes(1, 2).reshape(B, T, -1) if needs_reshape else output


class GuideAttentionMask:
    """Holds the two per-group masks for LTXV guide self-attention.
    _attention_with_guide_mask splits queries into noisy and tracked-guide
    groups, so the largest mask is (1, 1, tracked_count, T).
    """
    __slots__ = ("guide_start", "tracked_count", "noisy_mask", "tracked_mask")

    def __init__(self, total_tokens, guide_start, tracked_count, tracked_weights):
        device = tracked_weights.device
        dtype = tracked_weights.dtype
        finfo = torch.finfo(dtype)

        pos = tracked_weights > 0
        log_w = torch.full_like(tracked_weights, finfo.min)
        log_w[pos] = torch.log(tracked_weights[pos].clamp(min=finfo.tiny))

        self.guide_start = guide_start
        self.tracked_count = tracked_count

        self.noisy_mask = torch.zeros((1, 1, 1, total_tokens), device=device, dtype=dtype)
        self.noisy_mask[:, :, :, guide_start:guide_start + tracked_count] = log_w.view(1, 1, 1, -1)

        self.tracked_mask = torch.zeros((1, 1, tracked_count, total_tokens), device=device, dtype=dtype)
        self.tracked_mask[:, :, :, :guide_start] = log_w.view(1, 1, -1, 1)


def _attention_with_guide_mask(q, k, v, heads, guide_mask, attn_precision, transformer_options):
    """Apply the guide mask by partitioning Q into noisy and tracked-guide
    groups, so each group needs only its own sub-mask. Avoids materializing
    the (1,1,T,T) dense mask.
    """
    guide_start = guide_mask.guide_start
    tracked_end = guide_start + guide_mask.tracked_count

    out = torch.empty_like(q)

    if guide_start > 0: # In practice currently guides are always after noise, guard for safety if this changes.
        out[:, :guide_start, :] = comfy.ldm.modules.attention.optimized_attention(
            q[:, :guide_start, :], k, v, heads, mask=guide_mask.noisy_mask,
            attn_precision=attn_precision, transformer_options=transformer_options,
            low_precision_attention=False, # sageattn mask support is unreliable
        )
    out[:, guide_start:tracked_end, :] = comfy.ldm.modules.attention.optimized_attention(
        q[:, guide_start:tracked_end, :], k, v, heads, mask=guide_mask.tracked_mask,
        attn_precision=attn_precision, transformer_options=transformer_options,
        low_precision_attention=False,
    )
    if tracked_end < q.shape[1]: # Every guide token is tracked, and nothing comes after them, guard for safety if this changes.
        out[:, tracked_end:, :] = comfy.ldm.modules.attention.optimized_attention(
            q[:, tracked_end:, :], k, v, heads,
            attn_precision=attn_precision, transformer_options=transformer_options,
        )
    return out


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        attn_precision=None,
        apply_gated_attention=False,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim
        self.attn_precision = attn_precision

        self.heads = heads
        self.dim_head = dim_head

        self.q_norm = operations.RMSNorm(inner_dim, eps=1e-5, dtype=dtype, device=device)
        self.k_norm = operations.RMSNorm(inner_dim, eps=1e-5, dtype=dtype, device=device)

        self.to_q = operations.Linear(query_dim, inner_dim, bias=True, dtype=dtype, device=device)
        self.to_k = operations.Linear(context_dim, inner_dim, bias=True, dtype=dtype, device=device)
        self.to_v = operations.Linear(context_dim, inner_dim, bias=True, dtype=dtype, device=device)

        # Optional per-head gating
        if apply_gated_attention:
            self.to_gate_logits = operations.Linear(query_dim, heads, bias=True, dtype=dtype, device=device)
        else:
            self.to_gate_logits = None

        self.to_out = nn.Sequential(
            operations.Linear(inner_dim, query_dim, dtype=dtype, device=device), nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None, pe=None, k_pe=None, transformer_options={}):
        q = self.to_q(x)
        context = x if context is None else context
        k = self.to_k(context)
        v = self.to_v(context)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if pe is not None:
            q = apply_rotary_emb(q, pe)
            k = apply_rotary_emb(k, pe if k_pe is None else k_pe)

        if mask is None:
            out = comfy.ldm.modules.attention.optimized_attention(q, k, v, self.heads, attn_precision=self.attn_precision, transformer_options=transformer_options)
        elif isinstance(mask, GuideAttentionMask):
            out = _attention_with_guide_mask(q, k, v, self.heads, mask, attn_precision=self.attn_precision, transformer_options=transformer_options)
        else:
            out = comfy.ldm.modules.attention.optimized_attention(q, k, v, self.heads, mask=mask, attn_precision=self.attn_precision, transformer_options=transformer_options)

        # Apply per-head gating if enabled
        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)  # (B, T, H)
            b, t, _ = out.shape
            out = out.view(b, t, self.heads, self.dim_head)
            gates = 2.0 * torch.sigmoid(gate_logits)  # zero-init -> identity
            out = out * gates.unsqueeze(-1)
            out = out.view(b, t, self.heads * self.dim_head)

        return self.to_out(out)

# 6 base ADaLN params (shift/scale/gate for MSA + MLP), +3 for cross-attention Q (shift/scale/gate)
ADALN_BASE_PARAMS_COUNT = 6
ADALN_CROSS_ATTN_PARAMS_COUNT = 9

class BasicTransformerBlock(nn.Module):
    def __init__(
        self, dim, n_heads, d_head, context_dim=None, attn_precision=None, cross_attention_adaln=False, dtype=None, device=None, operations=None
    ):
        super().__init__()

        self.attn_precision = attn_precision
        self.cross_attention_adaln = cross_attention_adaln
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            context_dim=None,
            attn_precision=self.attn_precision,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.ff = FeedForward(dim, dim_out=dim, glu=True, dtype=dtype, device=device, operations=operations)

        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            attn_precision=self.attn_precision,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        num_ada_params = ADALN_CROSS_ATTN_PARAMS_COUNT if cross_attention_adaln else ADALN_BASE_PARAMS_COUNT
        self.scale_shift_table = nn.Parameter(torch.empty(num_ada_params, dim, device=device, dtype=dtype))

        if cross_attention_adaln:
            self.prompt_scale_shift_table = nn.Parameter(torch.empty(2, dim, device=device, dtype=dtype))

    def forward(self, x, context=None, attention_mask=None, timestep=None, pe=None, transformer_options={}, self_attention_mask=None, prompt_timestep=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None, None, :6].to(device=x.device, dtype=x.dtype) + timestep.reshape(x.shape[0], timestep.shape[1], self.scale_shift_table.shape[0], -1)[:, :, :6, :]).unbind(dim=2)

        x += self.attn1(comfy.ldm.common_dit.rms_norm(x) * (1 + scale_msa) + shift_msa, pe=pe, mask=self_attention_mask, transformer_options=transformer_options) * gate_msa

        if self.cross_attention_adaln:
            shift_q_mca, scale_q_mca, gate_mca = (self.scale_shift_table[None, None, 6:9].to(device=x.device, dtype=x.dtype) + timestep.reshape(x.shape[0], timestep.shape[1], self.scale_shift_table.shape[0], -1)[:, :, 6:9, :]).unbind(dim=2)
            x += apply_cross_attention_adaln(
                x, context, self.attn2, shift_q_mca, scale_q_mca, gate_mca,
                self.prompt_scale_shift_table, prompt_timestep, attention_mask, transformer_options,
            )
        else:
            x += self.attn2(x, context=context, mask=attention_mask, transformer_options=transformer_options)

        y = comfy.ldm.common_dit.rms_norm(x)
        y = torch.addcmul(y, y, scale_mlp).add_(shift_mlp)
        x.addcmul_(self.ff(y), gate_mlp)

        return x

def compute_prompt_timestep(adaln_module, timestep_scaled, batch_size, hidden_dtype):
    """Compute a single global prompt timestep for cross-attention ADaLN.

    Uses the max across tokens (matching JAX max_per_segment) and broadcasts
    over text tokens.  Returns None when *adaln_module* is None.
    """
    if adaln_module is None:
        return None
    ts_input = (
        timestep_scaled.max(dim=1, keepdim=True).values.flatten()
        if timestep_scaled.dim() > 1
        else timestep_scaled.flatten()
    )
    prompt_ts, _ = adaln_module(
        ts_input,
        {"resolution": None, "aspect_ratio": None},
        batch_size=batch_size,
        hidden_dtype=hidden_dtype,
    )
    return prompt_ts.view(batch_size, 1, prompt_ts.shape[-1])


def apply_cross_attention_adaln(
    x, context, attn, q_shift, q_scale, q_gate,
    prompt_scale_shift_table, prompt_timestep,
    attention_mask=None, transformer_options={},
):
    """Apply cross-attention with ADaLN modulation (shift/scale/gate on Q and KV).

    Q params (q_shift, q_scale, q_gate) are pre-extracted by the caller so
    that both regular tensors and CompressedTimestep are supported.
    """
    batch_size = x.shape[0]
    shift_kv, scale_kv = (
        prompt_scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
        + prompt_timestep.reshape(batch_size, prompt_timestep.shape[1], 2, -1)
    ).unbind(dim=2)
    attn_input = comfy.ldm.common_dit.rms_norm(x) * (1 + q_scale) + q_shift
    encoder_hidden_states = context * (1 + scale_kv) + shift_kv
    return attn(attn_input, context=encoder_hidden_states, mask=attention_mask, transformer_options=transformer_options) * q_gate

def get_fractional_positions(indices_grid, max_pos):
    n_pos_dims = indices_grid.shape[1]
    assert n_pos_dims == len(max_pos), f'Number of position dimensions ({n_pos_dims}) must match max_pos length ({len(max_pos)})'
    fractional_positions = torch.stack(
        [indices_grid[:, i] / max_pos[i] for i in range(n_pos_dims)],
        axis=-1,
    )
    return fractional_positions


@functools.lru_cache(maxsize=5)
def generate_freq_grid_np(positional_embedding_theta, positional_embedding_max_pos_count, inner_dim, _ = None):
    theta = positional_embedding_theta
    start = 1
    end = theta

    n_elem = 2 * positional_embedding_max_pos_count
    pow_indices = np.power(
        theta,
        np.linspace(
            _log_base(start, theta),
            _log_base(end, theta),
            inner_dim // n_elem,
            dtype=np.float64,
        ),
    )
    return torch.tensor(pow_indices * math.pi / 2, dtype=torch.float32)

def generate_freq_grid_pytorch(positional_embedding_theta, positional_embedding_max_pos_count, inner_dim, device):
    theta = positional_embedding_theta
    start = 1
    end = theta
    n_elem = 2 * positional_embedding_max_pos_count

    indices = theta ** (
        torch.linspace(
            math.log(start, theta),
            math.log(end, theta),
            inner_dim // n_elem,
            device=device,
            dtype=torch.float32,
        )
    )
    indices = indices.to(dtype=torch.float32)

    indices = indices * math.pi / 2

    return indices

def generate_freqs(indices, indices_grid, max_pos, use_middle_indices_grid):
    if use_middle_indices_grid:
        assert(len(indices_grid.shape) == 4 and indices_grid.shape[-1] ==2)
        indices_grid_start, indices_grid_end = indices_grid[..., 0], indices_grid[..., 1]
        indices_grid = (indices_grid_start + indices_grid_end) / 2.0
    elif len(indices_grid.shape) == 4:
        indices_grid = indices_grid[..., 0]

    # Get fractional positions and compute frequency indices
    fractional_positions = get_fractional_positions(indices_grid, max_pos)
    indices = indices.to(device=fractional_positions.device)

    freqs = (
        (indices * (fractional_positions.unsqueeze(-1) * 2 - 1))
        .transpose(-1, -2)
        .flatten(2)
    )
    return freqs

def interleaved_freqs_cis(freqs, pad_size):
    cos_freq = freqs.cos().repeat_interleave(2, dim=-1)
    sin_freq = freqs.sin().repeat_interleave(2, dim=-1)
    if pad_size != 0:
        cos_padding = torch.ones_like(cos_freq[:, :, : pad_size])
        sin_padding = torch.zeros_like(cos_freq[:, :, : pad_size])
        cos_freq = torch.cat([cos_padding, cos_freq], dim=-1)
        sin_freq = torch.cat([sin_padding, sin_freq], dim=-1)
    return cos_freq, sin_freq

def split_freqs_cis(freqs, pad_size, num_attention_heads):
    cos_freq = freqs.cos()
    sin_freq = freqs.sin()

    if pad_size != 0:
        cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = torch.zeros_like(sin_freq[:, :, :pad_size])

        cos_freq = torch.concatenate([cos_padding, cos_freq], axis=-1)
        sin_freq = torch.concatenate([sin_padding, sin_freq], axis=-1)

    # Reshape freqs to be compatible with multi-head attention
    B , T, half_HD = cos_freq.shape

    cos_freq = cos_freq.reshape(B, T, num_attention_heads, half_HD // num_attention_heads)
    sin_freq = sin_freq.reshape(B, T, num_attention_heads, half_HD // num_attention_heads)

    cos_freq = torch.swapaxes(cos_freq, 1, 2)  # (B,H,T,D//2)
    sin_freq = torch.swapaxes(sin_freq, 1, 2)  # (B,H,T,D//2)
    return cos_freq, sin_freq

class LTXBaseModel(torch.nn.Module, ABC):
    """
    Abstract base class for LTX models (Lightricks Transformer models).

    This class defines the common interface and shared functionality for all LTX models,
    including LTXV (video) and LTXAV (audio-video) variants.
    """

    def __init__(
        self,
        in_channels: int,
        cross_attention_dim: int,
        attention_head_dim: int,
        num_attention_heads: int,
        caption_channels: int,
        num_layers: int,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list = [20, 2048, 2048],
        causal_temporal_positioning: bool = False,
        vae_scale_factors: tuple = (8, 32, 32),
        use_middle_indices_grid=False,
        timestep_scale_multiplier = 1000.0,
        caption_proj_before_connector=False,
        cross_attention_adaln=False,
        caption_projection_first_linear=True,
        dtype=None,
        device=None,
        operations=None,
        **kwargs,
    ):
        super().__init__()
        self.generator = None
        self.vae_scale_factors = vae_scale_factors
        self.use_middle_indices_grid = use_middle_indices_grid
        self.dtype = dtype
        self.in_channels = in_channels
        self.cross_attention_dim = cross_attention_dim
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.caption_channels = caption_channels
        self.num_layers = num_layers
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = positional_embedding_max_pos
        self.split_positional_embedding = LTXRopeType.from_dict(kwargs)
        self.freq_grid_generator = (
            generate_freq_grid_np if LTXFrequenciesPrecision.from_dict(kwargs) == LTXFrequenciesPrecision.FLOAT64
            else generate_freq_grid_pytorch
        )
        self.causal_temporal_positioning = causal_temporal_positioning
        self.operations = operations
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.caption_proj_before_connector = caption_proj_before_connector
        self.cross_attention_adaln = cross_attention_adaln
        self.caption_projection_first_linear = caption_projection_first_linear

        # Common dimensions
        self.inner_dim = num_attention_heads * attention_head_dim
        self.out_channels = in_channels

        # Initialize common components
        self._init_common_components(device, dtype)

        # Initialize model-specific components
        self._init_model_components(device, dtype, **kwargs)

        # Initialize transformer blocks
        self._init_transformer_blocks(device, dtype, **kwargs)

        # Initialize output components
        self._init_output_components(device, dtype)

    def _init_common_components(self, device, dtype):
        """Initialize components common to all LTX models
        - patchify_proj: Linear projection for patchifying input
        - adaln_single: AdaLN layer for timestep embedding
        - caption_projection: Linear projection for caption embedding
        """
        self.patchify_proj = self.operations.Linear(
            self.in_channels, self.inner_dim, bias=True, dtype=dtype, device=device
        )

        embedding_coefficient = ADALN_CROSS_ATTN_PARAMS_COUNT if self.cross_attention_adaln else ADALN_BASE_PARAMS_COUNT
        self.adaln_single = AdaLayerNormSingle(
            self.inner_dim, embedding_coefficient=embedding_coefficient, use_additional_conditions=False, dtype=dtype, device=device, operations=self.operations
        )

        if self.cross_attention_adaln:
            self.prompt_adaln_single = AdaLayerNormSingle(
                self.inner_dim, embedding_coefficient=2, use_additional_conditions=False, dtype=dtype, device=device, operations=self.operations
            )
        else:
            self.prompt_adaln_single = None

        if self.caption_proj_before_connector:
            if self.caption_projection_first_linear:
                self.caption_projection = NormSingleLinearTextProjection(
                    in_features=self.caption_channels,
                    hidden_size=self.inner_dim,
                    dtype=dtype,
                    device=device,
                    operations=self.operations,
                )
            else:
                self.caption_projection = lambda a: a
        else:
            self.caption_projection = PixArtAlphaTextProjection(
                in_features=self.caption_channels,
                hidden_size=self.inner_dim,
                dtype=dtype,
                device=device,
                operations=self.operations,
            )

    @abstractmethod
    def _init_model_components(self, device, dtype, **kwargs):
        """Initialize model-specific components. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _init_transformer_blocks(self, device, dtype, **kwargs):
        """Initialize transformer blocks. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _init_output_components(self, device, dtype):
        """Initialize output components. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _process_input(self, x, keyframe_idxs, denoise_mask, **kwargs):
        """Process input data. Must be implemented by subclasses."""
        pass

    def _build_guide_self_attention_mask(self, x, transformer_options, merged_args):
        """Build self-attention mask for per-guide attention attenuation.

        Base implementation returns None (no attenuation). Subclasses that
        support guide-based attention control should override this.
        """
        return None

    @abstractmethod
    def _process_transformer_blocks(self, x, context, attention_mask, timestep, pe, self_attention_mask=None, **kwargs):
        """Process transformer blocks. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _process_output(self, x, embedded_timestep, keyframe_idxs, **kwargs):
        """Process output data. Must be implemented by subclasses."""
        pass

    def _prepare_timestep(self, timestep, batch_size, hidden_dtype, **kwargs):
        """Prepare timestep embeddings."""
        grid_mask = kwargs.get("grid_mask", None)
        if grid_mask is not None:
            timestep = timestep[:, grid_mask]

        timestep_scaled = timestep * self.timestep_scale_multiplier
        timestep, embedded_timestep = self.adaln_single(
            timestep_scaled.flatten(),
            {"resolution": None, "aspect_ratio": None},
            batch_size=batch_size,
            hidden_dtype=hidden_dtype,
        )

        # Second dimension is 1 or number of tokens (if timestep_per_token)
        timestep = timestep.view(batch_size, -1, timestep.shape[-1])
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.shape[-1])

        prompt_timestep = compute_prompt_timestep(
            self.prompt_adaln_single, timestep_scaled, batch_size, hidden_dtype
        )

        return timestep, embedded_timestep, prompt_timestep

    def _prepare_context(self, context, batch_size, x, attention_mask=None):
        """Prepare context for transformer blocks."""
        if self.caption_proj_before_connector is False:
            context = self.caption_projection(context)

        context = context.view(batch_size, -1, x.shape[-1])
        return context, attention_mask

    def _precompute_freqs_cis(
        self,
        indices_grid,
        dim,
        out_dtype,
        theta=10000.0,
        max_pos=[20, 2048, 2048],
        use_middle_indices_grid=False,
        num_attention_heads=32,
    ):
        split_mode = self.split_positional_embedding == LTXRopeType.SPLIT
        indices = self.freq_grid_generator(theta, indices_grid.shape[1], dim, indices_grid.device)
        freqs = generate_freqs(indices, indices_grid, max_pos, use_middle_indices_grid)

        if split_mode:
            expected_freqs = dim // 2
            current_freqs = freqs.shape[-1]
            pad_size = expected_freqs - current_freqs
            cos_freq, sin_freq = split_freqs_cis(freqs, pad_size, num_attention_heads)
        else:
            # 2 because of cos and sin by 3 for (t, x, y), 1 for temporal only
            n_elem = 2 * indices_grid.shape[1]
            cos_freq, sin_freq = interleaved_freqs_cis(freqs, dim % n_elem)
        return cos_freq.to(out_dtype), sin_freq.to(out_dtype), split_mode

    def _prepare_positional_embeddings(self, pixel_coords, frame_rate, x_dtype):
        """Prepare positional embeddings."""
        fractional_coords = pixel_coords.to(torch.float32)
        fractional_coords[:, 0] = fractional_coords[:, 0] * (1.0 / frame_rate)
        pe = self._precompute_freqs_cis(
            fractional_coords,
            dim=self.inner_dim,
            out_dtype=x_dtype,
            max_pos=self.positional_embedding_max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            num_attention_heads=self.num_attention_heads,
        )
        return pe

    def _prepare_attention_mask(self, attention_mask, x_dtype):
        """Prepare attention mask."""
        if attention_mask is not None and not torch.is_floating_point(attention_mask):
            attention_mask = (attention_mask - 1).to(x_dtype).reshape(
                (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
            ) * torch.finfo(x_dtype).max
        return attention_mask

    def forward(
        self, x, timestep, context, attention_mask, frame_rate=25, transformer_options={}, keyframe_idxs=None, denoise_mask=None, **kwargs
    ):
        """
        Forward pass for LTX models.

        Args:
            x: Input tensor
            timestep: Timestep tensor
            context: Context tensor (e.g., text embeddings)
            attention_mask: Attention mask tensor
            frame_rate: Frame rate for temporal processing
            transformer_options: Additional options for transformer blocks
            keyframe_idxs: Keyframe indices for temporal processing
            **kwargs: Additional keyword arguments

        Returns:
            Processed output tensor
        """
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options
            ),
        ).execute(x, timestep, context, attention_mask, frame_rate, transformer_options, keyframe_idxs, denoise_mask=denoise_mask, **kwargs)

    def _forward(
        self, x, timestep, context, attention_mask, frame_rate=25, transformer_options={}, keyframe_idxs=None, denoise_mask=None, **kwargs
    ):
        """
        Internal forward pass for LTX models.

        Args:
            x: Input tensor
            timestep: Timestep tensor
            context: Context tensor (e.g., text embeddings)
            attention_mask: Attention mask tensor
            frame_rate: Frame rate for temporal processing
            transformer_options: Additional options for transformer blocks
            keyframe_idxs: Keyframe indices for temporal processing
            **kwargs: Additional keyword arguments

        Returns:
            Processed output tensor
        """
        if isinstance(x, list):
            input_dtype = x[0].dtype
            batch_size = x[0].shape[0]
        else:
            input_dtype = x.dtype
            batch_size = x.shape[0]
        # Process input
        merged_args = {**transformer_options, **kwargs}
        x, pixel_coords, additional_args = self._process_input(x, keyframe_idxs, denoise_mask, **merged_args)
        merged_args.update(additional_args)

        # Prepare timestep and context
        timestep, embedded_timestep, prompt_timestep = self._prepare_timestep(timestep, batch_size, input_dtype, **merged_args)
        merged_args["prompt_timestep"] = prompt_timestep
        context, attention_mask = self._prepare_context(context, batch_size, x, attention_mask)

        # Prepare attention mask and positional embeddings
        attention_mask = self._prepare_attention_mask(attention_mask, input_dtype)
        pe = self._prepare_positional_embeddings(pixel_coords, frame_rate, input_dtype)

        # Build self-attention mask for per-guide attenuation
        self_attention_mask = self._build_guide_self_attention_mask(
            x, transformer_options, merged_args
        )

        # Process transformer blocks
        x = self._process_transformer_blocks(
            x, context, attention_mask, timestep, pe,
            transformer_options=transformer_options,
            self_attention_mask=self_attention_mask,
            **merged_args,
        )

        # Process output
        x = self._process_output(x, embedded_timestep, keyframe_idxs, **merged_args)
        return x


class LTXVModel(LTXBaseModel):
    """LTXV model for video generation."""

    def __init__(
        self,
        in_channels=128,
        cross_attention_dim=2048,
        attention_head_dim=64,
        num_attention_heads=32,
        caption_channels=4096,
        num_layers=28,
        positional_embedding_theta=10000.0,
        positional_embedding_max_pos=[20, 2048, 2048],
        causal_temporal_positioning=False,
        vae_scale_factors=(8, 32, 32),
        use_middle_indices_grid=False,
        timestep_scale_multiplier=1000.0,
        caption_proj_before_connector=False,
        cross_attention_adaln=False,
        dtype=None,
        device=None,
        operations=None,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            caption_channels=caption_channels,
            num_layers=num_layers,
            positional_embedding_theta=positional_embedding_theta,
            positional_embedding_max_pos=positional_embedding_max_pos,
            causal_temporal_positioning=causal_temporal_positioning,
            vae_scale_factors=vae_scale_factors,
            use_middle_indices_grid=use_middle_indices_grid,
            timestep_scale_multiplier=timestep_scale_multiplier,
            caption_proj_before_connector=caption_proj_before_connector,
            cross_attention_adaln=cross_attention_adaln,
            dtype=dtype,
            device=device,
            operations=operations,
            **kwargs,
        )

    def _init_model_components(self, device, dtype, **kwargs):
        """Initialize LTXV-specific components."""
        pass

    def _init_transformer_blocks(self, device, dtype, **kwargs):
        """Initialize transformer blocks for LTXV."""
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.num_attention_heads,
                    self.attention_head_dim,
                    context_dim=self.cross_attention_dim,
                    cross_attention_adaln=self.cross_attention_adaln,
                    dtype=dtype,
                    device=device,
                    operations=self.operations,
                )
                for _ in range(self.num_layers)
            ]
        )

    def _init_output_components(self, device, dtype):
        """Initialize output components for LTXV."""
        self.scale_shift_table = nn.Parameter(torch.empty(2, self.inner_dim, dtype=dtype, device=device))
        self.norm_out = self.operations.LayerNorm(
            self.inner_dim, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
        )
        self.proj_out = self.operations.Linear(self.inner_dim, self.out_channels, dtype=dtype, device=device)
        self.patchifier = SymmetricPatchifier(1, start_end=True)

    def _process_input(self, x, keyframe_idxs, denoise_mask, **kwargs):
        """Process input for LTXV."""
        additional_args = {"orig_shape": list(x.shape)}
        x, latent_coords = self.patchifier.patchify(x)
        pixel_coords = latent_to_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=self.vae_scale_factors,
            causal_fix=self.causal_temporal_positioning,
        )

        grid_mask = None
        if keyframe_idxs is not None and keyframe_idxs.shape[2] > 0:
            additional_args.update({ "orig_patchified_shape": list(x.shape)})
            denoise_mask = self.patchifier.patchify(denoise_mask)[0]
            grid_mask = ~torch.any(denoise_mask < 0, dim=-1)[0]
            additional_args.update({"grid_mask": grid_mask})
            x = x[:, grid_mask, :]
            pixel_coords = pixel_coords[:, :, grid_mask, ...]

            kf_grid_mask = grid_mask[-keyframe_idxs.shape[2]:]

            # Compute per-guide surviving token counts from guide_attention_entries.
            # Each entry tracks one guide reference; they are appended in order and
            # their pre_filter_counts partition the kf_grid_mask.
            guide_entries = kwargs.get("guide_attention_entries", None)
            if guide_entries:
                total_pfc = sum(e["pre_filter_count"] for e in guide_entries)
                if total_pfc != len(kf_grid_mask):
                    raise ValueError(
                        f"guide pre_filter_counts ({total_pfc}) != "
                        f"keyframe grid mask length ({len(kf_grid_mask)})"
                    )
                resolved_entries = []
                offset = 0
                for entry in guide_entries:
                    pfc = entry["pre_filter_count"]
                    entry_mask = kf_grid_mask[offset:offset + pfc]
                    surviving = int(entry_mask.sum().item())
                    resolved_entries.append({
                        **entry,
                        "surviving_count": surviving,
                    })
                    offset += pfc
                additional_args["resolved_guide_entries"] = resolved_entries

            keyframe_idxs = keyframe_idxs[..., kf_grid_mask, :]

            if keyframe_idxs.shape[2] > 0: # Guard for the case of no keyframes surviving
                pixel_coords[:, :, -keyframe_idxs.shape[2]:, :] = keyframe_idxs

            # Total surviving guide tokens (all guides)
            additional_args["num_guide_tokens"] = keyframe_idxs.shape[2]

        x = self.patchify_proj(x)
        return x, pixel_coords, additional_args

    def _build_guide_self_attention_mask(self, x, transformer_options, merged_args):
        """Build self-attention mask for per-guide attention attenuation.

        Reads resolved_guide_entries from merged_args (computed in _process_input)
        to build a log-space additive bias mask that attenuates noisy ↔ guide
        attention for each guide reference independently.

        Returns None if no attenuation is needed (all strengths == 1.0 and no
        spatial masks, or no guide tokens).
        """
        if isinstance(x, list):
            # AV model: x = [vx, ax]; use vx for token count and device
            total_tokens = x[0].shape[1]
            device = x[0].device
            dtype = x[0].dtype
        else:
            total_tokens = x.shape[1]
            device = x.device
            dtype = x.dtype

        num_guide_tokens = merged_args.get("num_guide_tokens", 0)
        if num_guide_tokens == 0:
            return None

        resolved_entries = merged_args.get("resolved_guide_entries", None)
        if not resolved_entries:
            return None

        # strength != 1.0 means we want to either attenuate (< 1) or amplify (> 1) guide attention.
        needs_mask = any(
            e["strength"] != 1.0 or e.get("pixel_mask") is not None
            for e in resolved_entries
        )
        if not needs_mask:
            return None

        # Build per-guide-token weights for all tracked guide tokens.
        # Guides are appended in order at the end of the sequence.
        guide_start = total_tokens - num_guide_tokens
        all_weights = []
        total_tracked = 0

        for entry in resolved_entries:
            surviving = entry["surviving_count"]
            if surviving == 0:
                continue

            strength = entry["strength"]
            pixel_mask = entry.get("pixel_mask")
            latent_shape = entry.get("latent_shape")

            if pixel_mask is not None and latent_shape is not None:
                f_lat, h_lat, w_lat = latent_shape
                per_token = self._downsample_mask_to_latent(
                    pixel_mask.to(device=device, dtype=dtype),
                    f_lat, h_lat, w_lat,
                )
                # per_token shape: (B, f_lat*h_lat*w_lat).
                # Collapse batch dim — the mask is assumed identical across the
                # batch; validate and take the first element to get (1, tokens).
                if per_token.shape[0] > 1:
                    ref = per_token[0]
                    for bi in range(1, per_token.shape[0]):
                        if not torch.equal(ref, per_token[bi]):
                            logger.warning(
                                "pixel_mask differs across batch elements; "
                                "using first element only."
                            )
                            break
                    per_token = per_token[:1]
                # `surviving` is the post-grid_mask token count.
                # Clamp to surviving to handle any mismatch safely.
                n_weights = min(per_token.shape[1], surviving)
                weights = per_token[:, :n_weights] * strength  # (1, n_weights)
            else:
                weights = torch.full(
                    (1, surviving), strength, device=device, dtype=dtype
                )

            all_weights.append(weights)
            total_tracked += weights.shape[1]

        if not all_weights:
            return None

        # Concatenate per-token weights for all tracked guides
        tracked_weights = torch.cat(all_weights, dim=1)  # (1, total_tracked)

        # Skip when every weight is exactly 1.0 (additive bias would be 0).
        if (tracked_weights == 1.0).all():
            return None

        return GuideAttentionMask(total_tokens, guide_start, total_tracked, tracked_weights)

    @staticmethod
    def _downsample_mask_to_latent(mask, f_lat, h_lat, w_lat):
        """Downsample a pixel-space mask to per-token latent weights.

        Args:
            mask: (B, 1, F_pix, H_pix, W_pix) pixel-space mask with values in [0, 1].
            f_lat: Number of latent frames (pre-dilation original count).
            h_lat: Latent height (pre-dilation original height).
            w_lat: Latent width (pre-dilation original width).

        Returns:
            (B, F_lat * H_lat * W_lat) flattened per-token weights.
        """
        b = mask.shape[0]
        f_pix = mask.shape[2]

        # Spatial downsampling: area interpolation per frame
        spatial_down = torch.nn.functional.interpolate(
            rearrange(mask, "b 1 f h w -> (b f) 1 h w"),
            size=(h_lat, w_lat),
            mode="area",
        )
        spatial_down = rearrange(spatial_down, "(b f) 1 h w -> b 1 f h w", b=b)

        # Temporal downsampling: first pixel frame maps to first latent frame,
        # remaining pixel frames are averaged in groups for causal temporal structure.
        first_frame = spatial_down[:, :, :1, :, :]
        if f_pix > 1 and f_lat > 1:
            remaining_pix = f_pix - 1
            remaining_lat = f_lat - 1
            t = remaining_pix // remaining_lat
            if t < 1:
                # Fewer pixel frames than latent frames — upsample by repeating
                # the available pixel frames via nearest interpolation.
                rest_flat = rearrange(
                    spatial_down[:, :, 1:, :, :],
                    "b 1 f h w -> (b h w) 1 f",
                )
                rest_up = torch.nn.functional.interpolate(
                    rest_flat, size=remaining_lat, mode="nearest",
                )
                rest = rearrange(
                    rest_up, "(b h w) 1 f -> b 1 f h w",
                    b=b, h=h_lat, w=w_lat,
                )
            else:
                # Trim trailing pixel frames that don't fill a complete group
                usable = remaining_lat * t
                rest = rearrange(
                    spatial_down[:, :, 1:1 + usable, :, :],
                    "b 1 (f t) h w -> b 1 f t h w",
                    t=t,
                )
                rest = rest.mean(dim=3)
            latent_mask = torch.cat([first_frame, rest], dim=2)
        elif f_lat > 1:
            # Single pixel frame but multiple latent frames — repeat the
            # single frame across all latent frames.
            latent_mask = first_frame.expand(-1, -1, f_lat, -1, -1)
        else:
            latent_mask = first_frame

        return rearrange(latent_mask, "b 1 f h w -> b (f h w)")

    def _process_transformer_blocks(self, x, context, attention_mask, timestep, pe, transformer_options={}, self_attention_mask=None, **kwargs):
        """Process transformer blocks for LTXV."""
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        prompt_timestep = kwargs.get("prompt_timestep", None)

        for i, block in enumerate(self.transformer_blocks):
            if ("double_block", i) in blocks_replace:

                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], context=args["txt"], attention_mask=args["attention_mask"], timestep=args["vec"], pe=args["pe"], transformer_options=args["transformer_options"], self_attention_mask=args.get("self_attention_mask"), prompt_timestep=args.get("prompt_timestep"))
                    return out

                out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "attention_mask": attention_mask, "vec": timestep, "pe": pe, "transformer_options": transformer_options, "self_attention_mask": self_attention_mask, "prompt_timestep": prompt_timestep}, {"original_block": block_wrap})
                x = out["img"]
            else:
                x = block(
                    x,
                    context=context,
                    attention_mask=attention_mask,
                    timestep=timestep,
                    pe=pe,
                    transformer_options=transformer_options,
                    self_attention_mask=self_attention_mask,
                    prompt_timestep=prompt_timestep,
                )

        return x

    def _process_output(self, x, embedded_timestep, keyframe_idxs, **kwargs):
        """Process output for LTXV."""
        # Apply scale-shift modulation
        scale_shift_values = (
            self.scale_shift_table[None, None].to(device=x.device, dtype=x.dtype) + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        x = self.norm_out(x)
        x = x * (1 + scale) + shift
        x = self.proj_out(x)

        if keyframe_idxs is not None and keyframe_idxs.shape[2] > 0:
            grid_mask = kwargs["grid_mask"]
            orig_patchified_shape = kwargs["orig_patchified_shape"]
            full_x = torch.zeros(orig_patchified_shape, dtype=x.dtype, device=x.device)
            full_x[:, grid_mask, :] = x
            x = full_x
        # Unpatchify to restore original dimensions
        orig_shape = kwargs["orig_shape"]
        x = self.patchifier.unpatchify(
            latents=x,
            output_height=orig_shape[3],
            output_width=orig_shape[4],
            output_num_frames=orig_shape[2],
            out_channels=orig_shape[1] // math.prod(self.patchifier.patch_size),
        )

        return x
