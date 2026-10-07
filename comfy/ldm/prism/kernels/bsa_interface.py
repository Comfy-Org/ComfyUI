"""Prism inference-only block-sparse attention, adapted from upstream BSA.

Original attribution and license terms are retained in LICENSE.
"""
import math
import os

import torch
import torch.nn.functional as F
import triton

from .flash_attn_bsa_varlen_mask import (
    _attn_fwd_bsa_varlen, _attn_fwd_bsa_varlen_align,
    configs_fwd_bsa_varlen_preset, configs_fwd_bsa_varlen_align_preset,
)


def mean_pooling_compression(
    x: torch.Tensor,
    block_size: int
) -> torch.Tensor:
    """Pool consecutive token blocks, zero-padding an incomplete block."""
    B, H, S = x.shape[:3]
    num_block = math.ceil(S / block_size)
    if S % block_size != 0:
        x = F.pad(x, (0, 0, 0, num_block * block_size - S))
    x_cmp = x.view(B, H, num_block, block_size, -1).mean(dim=3)
    return x_cmp


def masked_mean_pooling_compression(
    x: torch.Tensor,
    block_size: int,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean pooling that ignores padded positions marked False in valid_mask.

    Handles both cases:
      - S divisible by block_size (after 3D padding + block rearrangement)
      - S not divisible by block_size (pads the tail and extends valid_mask)
    """
    B, H, S, D = x.shape
    num_block = math.ceil(S / block_size)
    if S % block_size != 0:
        pad_len = num_block * block_size - S
        x = F.pad(x, (0, 0, 0, pad_len))
        valid_mask = F.pad(valid_mask, (0, pad_len), value=False)

    x_blocks = x.view(B, H, num_block, block_size, D)
    mask_blocks = valid_mask.view(num_block, block_size)
    mask_f = mask_blocks.float().unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    counts = mask_f.sum(dim=3, keepdim=True).clamp(min=1.0)
    x_cmp = (x_blocks * mask_f).sum(dim=3) / counts.squeeze(3)
    return x_cmp


def cal_score(q, k):
    """Compute block-level Q/K scores for sparse block selection."""
    k_transposed = k.transpose(-1, -2)  # [b, h, d, s_k]
    score = torch.matmul(q, k_transposed)  # [b, h, s_q, s_k]
    return score


def get_select_indices_topk_from_score(score, sparsity):
    """Return sorted top-k block indices and per-query selection counts."""
    num_selected = int((1 - sparsity) * score.shape[-1])
    block_indices = torch.topk(score, num_selected)[1]
    block_indices, _ = torch.sort(block_indices, dim=-1)

    block_indices_lens = torch.full(
        (block_indices.shape[0], block_indices.shape[1], block_indices.shape[2]),
        num_selected,
        dtype=torch.int32,
        device=block_indices.device
    )

    return block_indices, block_indices_lens


def get_select_indices_cdf_from_score(score, cdf_threshold, sm_scale):
    """Select the smallest top-p prefix, including its threshold-crossing block."""
    weights = torch.softmax(score * sm_scale, dim=-1)

    B, H, Sq, Sk = weights.shape
    upper_bound = min(Sk, int(cdf_threshold * Sk) + 1)
    topk_vals, topk_idx = torch.topk(weights, k=upper_bound, dim=-1, largest=True, sorted=True)
    cdf = torch.cumsum(topk_vals, dim=-1)
    # Standard nucleus (top-p): pick the smallest set whose cumulative prob >= threshold,
    # i.e. INCLUDE the block that crosses the threshold. (cdf < p).sum() counts the prefix
    # blocks still strictly below p; +1 adds the crossing block.
    num_selected = (cdf < cdf_threshold).to(torch.int32).sum(dim=-1, keepdim=True) + 1
    num_selected = num_selected.clamp(min=1, max=Sk)

    block_indices = topk_idx.contiguous()
    pos = torch.arange(upper_bound, device=block_indices.device)
    block_indices = block_indices.masked_fill(pos >= num_selected, Sk)
    block_indices, _ = torch.sort(block_indices, dim=-1)
    return block_indices, num_selected.squeeze(-1).to(torch.int32)


def get_select_indices_cdf_topk_from_score(score, sparsity, cdf_threshold, sm_scale):
    """Combine a top-k lower bound with threshold-crossing top-p selection."""
    weights = torch.softmax(score * sm_scale, dim=-1)

    B, H, Sq, Sk = weights.shape
    num_selected_topk = max(1, int((1 - sparsity) * Sk))
    upper_bound = min(Sk, max(num_selected_topk, int(cdf_threshold * Sk) + 1))

    topk_vals, topk_idx = torch.topk(weights, k=upper_bound, dim=-1, largest=True, sorted=True)
    cdf = torch.cumsum(topk_vals, dim=-1)
    # Standard nucleus (top-p): pick the smallest set whose cumulative prob >= threshold,
    # i.e. INCLUDE the block that crosses the threshold. (cdf < p).sum() counts the prefix
    # blocks still strictly below p; +1 adds the crossing block.
    num_selected = (cdf < cdf_threshold).to(torch.int32).sum(dim=-1, keepdim=True) + 1
    num_selected = num_selected.clamp(min=num_selected_topk, max=Sk)

    block_indices = topk_idx.contiguous()
    pos = torch.arange(upper_bound, device=block_indices.device)
    block_indices = block_indices.masked_fill(pos >= num_selected, Sk)
    block_indices, _ = torch.sort(block_indices, dim=-1)
    return block_indices, num_selected.squeeze(-1).to(torch.int32)


def attn_fwd_bsa_varlen_triton(
    q,
    k,
    v,
    sm_scale,
    block_indices,
    block_indices_lens,
    chunk_size_q,
    chunk_size_k,
    sparsity,
    kv_valid_mask=None,
):

    """Launch the forward-only variable-length BSA kernel with an optional key mask."""
    B, H, Seq, D = q.shape

    o = torch.empty_like(q)
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

    grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

    config_key = 'BLOCK_N_LG=64' if chunk_size_k == 64 else 'default'
    if chunk_size_k > 128:
        fwd_func = _attn_fwd_bsa_varlen
        kernel_config = {} if os.environ.get('TRITON_AUTOTUNE_ENBALE', '0') == '1' else configs_fwd_bsa_varlen_preset[config_key]
    else:
        fwd_func = _attn_fwd_bsa_varlen_align
        kernel_config = {} if os.environ.get('TRITON_AUTOTUNE_ENBALE', '0') == '1' else configs_fwd_bsa_varlen_align_preset[config_key]

    block_indices = block_indices.contiguous()
    block_indices_lens = block_indices_lens.contiguous()

    has_kv_mask = 1 if kv_valid_mask is not None else 0
    _kv_mask = kv_valid_mask if kv_valid_mask is not None else torch.empty(0, dtype=torch.bool, device=q.device)

    fwd_func[grid](
        q, k, v, sm_scale, M, o,
        block_indices,
        block_indices_lens,
        _kv_mask,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        block_indices.stride(0), block_indices.stride(1), block_indices.stride(2), block_indices.stride(3),
        block_indices_lens.stride(0), block_indices_lens.stride(1), block_indices_lens.stride(2),
        H, Seq,
        D,
        BLOCK_M=chunk_size_q,
        BLOCK_N_LG=chunk_size_k,
        SPARSITY=sparsity,
        HAS_KV_MASK=has_kv_mask,
        **kernel_config
    )

    return o


def flash_attn_bsa(q, k, v, chunk_size_q, chunk_size_k, sparsity, cdf_threshold, sm_scale,
            kv_valid_mask=None):
    """Pool tokens, select key blocks, and evaluate sparse attention for inference."""
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}

    # ---------------------- gating ----------------------
    if kv_valid_mask is not None and q.shape[2] == k.shape[2]:
        q_cmp = masked_mean_pooling_compression(q, chunk_size_q, kv_valid_mask)
    else:
        q_cmp = mean_pooling_compression(q, chunk_size_q)

    if kv_valid_mask is not None:
        k_cmp = masked_mean_pooling_compression(k, chunk_size_k, kv_valid_mask)
    else:
        k_cmp = mean_pooling_compression(k, chunk_size_k)

    score = cal_score(q_cmp, k_cmp)

    if sparsity is not None and cdf_threshold is None:
        block_indices, block_indices_lens = get_select_indices_topk_from_score(score, sparsity)
    elif sparsity is None and cdf_threshold is not None:
        block_indices, block_indices_lens = get_select_indices_cdf_from_score(score, cdf_threshold, sm_scale)
    elif sparsity is not None and cdf_threshold is not None:
        block_indices, block_indices_lens = get_select_indices_cdf_topk_from_score(score, sparsity, cdf_threshold, sm_scale)
    else:
        raise ValueError("Either sparsity or cdf_threshold must be provided")

    # ---------------------- bsa ----------------------

    o = attn_fwd_bsa_varlen_triton(
        q, k, v,
        sm_scale, block_indices, block_indices_lens,
        chunk_size_q, chunk_size_k,
        sparsity,
        kv_valid_mask=kv_valid_mask,
    )

    return o


def rearrange_THW_to_3d_block(x, Nt, Nh, Nw, t, h, w, D):
    """Make each spatial-temporal block contiguous in the token sequence."""
    B, H, _, D = x.shape
    x = x.view(B, H, Nt, t, Nh, h, Nw, w, D)
    x = x.permute(0, 1, 2, 4, 6, 3, 5, 7, 8)  # B H Nt Nh Nw t h w D
    return x.contiguous().view(B, H, Nt * Nh * Nw * t * h * w, D)


def rearrange_3d_block_to_THW(x, Nt, Nh, Nw, t, h, w, D):
    """Restore block-ordered tokens to their original temporal/spatial order."""
    B, H, _, D = x.shape
    x = x.view(B, H, Nt, Nh, Nw, t, h, w, D)
    x = x.permute(0, 1, 2, 5, 3, 6, 4, 7, 8)  # B H Nt t Nh h Nw w D
    return x.contiguous().view(B, H, Nt * t * Nh * h * Nw * w, D)


def rearrange_THW_to_3d_block_1d(mask, Nt, Nh, Nw, t, h, w):
    """Rearrange a 1D bool mask from THW order to 3D block order (same permutation as tokens)."""
    mask = mask.view(Nt, t, Nh, h, Nw, w)
    mask = mask.permute(0, 2, 4, 1, 3, 5)  # Nt Nh Nw t h w
    return mask.contiguous().view(-1)


def flash_attn_bsa_3d(
    q: torch.Tensor, # [B, H, Sq, D]
    k: torch.Tensor, # [B, H, Skv, D]
    v: torch.Tensor, # [B, H, Skv, D]
    latent_shape_q,
    latent_shape_k,
    # bsa_params
    sparsity=0.875,
    cdf_threshold=None,
    chunk_3d_shape_q=[4, 4, 8],
    chunk_3d_shape_k=[4, 4, 8],
    valid_mask=None,  # [Sq] bool, True=valid token, False=padded
) -> torch.Tensor:
    """Evaluate 3D block-sparse attention on padded grids, excluding masked keys."""
    _, _, Sq, head_dim_q = q.shape
    _, _, Sk, head_dim_k = k.shape

    assert head_dim_q == head_dim_k
    head_dim = head_dim_q

    Tq, Hq, Wq = latent_shape_q
    Tk, Hk, Wk = latent_shape_k

    assert Tq * Hq * Wq == Sq
    assert Tk * Hk * Wk == Sk

    tq, hq, wq = chunk_3d_shape_q
    tk, hk, wk = chunk_3d_shape_k

    assert Tq % tq == 0 and Hq % hq == 0 and Wq % wq == 0
    assert Tk % tk == 0 and Hk % hk == 0 and Wk % wk == 0

    Ntq = Tq // tq
    Nhq = Hq // hq
    Nwq = Wq // wq

    Ntk = Tk // tk
    Nhk = Hk // hk
    Nwk = Wk // wk

    q = rearrange_THW_to_3d_block(q, Ntq, Nhq, Nwq, tq, hq, wq, q.shape[-1])
    k = rearrange_THW_to_3d_block(k, Ntk, Nhk, Nwk, tk, hk, wk, k.shape[-1])
    v = rearrange_THW_to_3d_block(v, Ntk, Nhk, Nwk, tk, hk, wk, v.shape[-1])

    kv_valid_mask = None
    if valid_mask is not None:
        kv_valid_mask = rearrange_THW_to_3d_block_1d(valid_mask, Ntk, Nhk, Nwk, tk, hk, wk)

    chunk_size_q = tq * hq * wq
    chunk_size_k = tk * hk * wk

    output = flash_attn_bsa(q, k, v, chunk_size_q, chunk_size_k, sparsity, cdf_threshold, 1 / head_dim**0.5, kv_valid_mask)

    output = rearrange_3d_block_to_THW(output, Ntq, Nhq, Nwq, tq, hq, wq, output.shape[-1])
    return output
