import os
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import math

from .common import _attn_fwd_gating, _attn_bwd_preprocess, configs_gating_preset
from .flash_attn_bsa_varlen_mask import (
    _attn_fwd_bsa_varlen, _attn_fwd_bsa_varlen_align, _attn_bwd_dkdv_bsa_varlen_wrapper, _attn_bwd_dq_bsa_varlen_wrapper, _attn_bwd_dq_bsa_varlen_align_wrapper,
    configs_fwd_bsa_varlen_preset, configs_fwd_bsa_varlen_align_preset, configs_bwd_dkdv_bsa_varlen_preset, configs_bwd_dq_bsa_varlen_preset, configs_bwd_dq_bsa_varlen_align_preset
)


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9

HAS_TMA_DESC = "nv_tma_desc_type" in dir(tl)

if HAS_TMA_DESC:
    print("TMA benchmarks will be running with experimental grid constant TMA descriptor.", )
else:
    print("TMA benchmarks will be running without grid constant TMA descriptor.", )


# TmaAutoTuneHelper used in htyu's PR #5622
class TmaAutoTuneHelper:

    # duck typing wrapper to implement the same interface as TmaDescKernelParam in Triton PR #4498
    class KernelParamWrapper:

        def __init__(self, desc):
            self.desc = desc

        def tma_desc_cpu_ptr(self):
            return self.desc.data_ptr()

    TMA_SIZE = 128

    def __init__(self):
        self.fill_1d_tma_descriptor_inner = (triton.runtime.driver.active.utils.fill_1d_tma_descriptor)
        self.fill_2d_tma_descriptor_inner = (triton.runtime.driver.active.utils.fill_2d_tma_descriptor)
        if HAS_TMA_DESC:
            self.descriptors = {}
        else:
            self.cuda_descriptors = {}

    # Call this method outside of the lambda function for grid size
    def init_tma_descriptor(self, name):
        if HAS_TMA_DESC:
            self.descriptors[name] = torch.empty(TmaAutoTuneHelper.TMA_SIZE, device="cpu", dtype=torch.int8)
        else:
            self.cuda_descriptors[name] = torch.empty(TmaAutoTuneHelper.TMA_SIZE, device="cuda", dtype=torch.int8)

    # Call this method inside the lambda function for grid size
    def fill_1d_tma_descriptor(self, name, ptr, dim, block_dim, element_size):
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0
            self.fill_1d_tma_descriptor_inner(ptr, dim, block_dim, element_size, desc_x.data_ptr())
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_1d_tma_descriptor_inner(ptr, dim, block_dim, element_size, buf_x.data_ptr())
            desc_x.copy_(buf_x, non_blocking=True)

    # Call this method inside the lambda function for grid size
    def fill_2d_tma_descriptor(self, name, ptr, dim1, dim0, block_dim1, block_dim0, element_size):
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0
            self.fill_2d_tma_descriptor_inner(ptr, dim1, dim0, block_dim1, block_dim0, element_size, desc_x.data_ptr())
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_2d_tma_descriptor_inner(ptr, dim1, dim0, block_dim1, block_dim0, element_size, buf_x.data_ptr())
            desc_x.copy_(buf_x, non_blocking=True)

    def get_tma_descriptor_kernel_param(self, name):
        if HAS_TMA_DESC:
            assert self.descriptors[name] is not None
            return self.KernelParamWrapper(self.descriptors[name])
        else:
            assert self.cuda_descriptors[name] is not None
            return self.cuda_descriptors[name]


@triton.jit
def create_mask_from_indices_kernel(
    block_indices,
    block_mask,
    stride_bz, stride_bh, stride_bm, stride_bs,
    stride_mz, stride_mh, stride_mm, stride_mn,
    H,
):
    i_zh, i_m, i_s = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_z, i_h = i_zh // H, i_zh % H

    off_b = i_z.to(tl.int64) * stride_bz + i_h.to(tl.int64) * stride_bh + i_m.to(tl.int64) * stride_bm + i_s.to(tl.int64) * stride_bs

    b_i = tl.load(block_indices + off_b)

    off_m = i_z.to(tl.int64) * stride_mz + i_h.to(tl.int64) * stride_mh + i_m.to(tl.int64) * stride_mm + b_i.to(tl.int64) * stride_mn

    b_m = 1
    tl.store(block_mask + off_m, b_m.to(block_mask.dtype.element_ty))

def create_mask_from_indices_triton(
    block_indices,
    N_cols
):
    B, H, N_rows, S = block_indices.shape
    block_mask = torch.zeros((B, H, N_rows, N_cols), dtype=torch.bool, device=block_indices.device)
    create_mask_from_indices_kernel[(B * H, N_rows, S)](
        block_indices,
        block_mask,
        block_indices.stride(0), block_indices.stride(1), block_indices.stride(2), block_indices.stride(3),
        block_mask.stride(0), block_mask.stride(1), block_mask.stride(2), block_mask.stride(3),
        H,
    )
    return block_mask

def create_mask_from_indices_varlen(block_indices, N_cols_mask):

    B, H, M, _ = block_indices.shape
    device = block_indices.device

    mask = torch.zeros((B, H, M, N_cols_mask), dtype=torch.bool, device=device)

    valid = block_indices < N_cols_mask

    b_idx = torch.arange(B, device=device)[:, None, None, None].expand_as(block_indices)
    h_idx = torch.arange(H, device=device)[None, :, None, None].expand_as(block_indices)
    m_idx = torch.arange(M, device=device)[None, None, :, None].expand_as(block_indices)

    valid_coords = (b_idx[valid], h_idx[valid], m_idx[valid], block_indices[valid])

    mask[valid_coords] = True

    return mask

def create_indices_k_from_indices_q_varlen(
    block_indices,
    N_cols_mask # indicate the number of the last dimension of the bool mask, since this information cannot be determined by block_indices, which may contain invalid elements
):
    block_mask_qk = create_mask_from_indices_varlen(block_indices, N_cols_mask)
    B, H, M, N = block_mask_qk.shape
    block_mask_kq = block_mask_qk.permute(0, 1, 3, 2)
    indices = torch.arange(M, device=block_indices.device).view(1, 1, 1, -1).expand_as(block_mask_kq)
    block_indices_k = torch.where(block_mask_kq, indices, M)
    block_indices_k, _ = torch.sort(block_indices_k, dim=-1)

    block_indices_k_lens = (block_indices_k < M).sum(dim=-1)

    return block_indices_k, block_indices_k_lens


def mean_pooling_compression(
    x: torch.Tensor,
    block_size: int
) -> torch.Tensor:
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
    k_transposed = k.transpose(-1, -2)  # [b, h, d, s_k]
    score = torch.matmul(q, k_transposed)  # [b, h, s_q, s_k]
    return score

def cal_score_triton(q, k):
    B, H, s_q, D = q.shape
    s_k = k.shape[2]

    score = torch.empty(B, H, s_q, s_k, device=q.device, dtype=q.dtype)

    kernel_config = {} if os.environ.get('TRITON_AUTOTUNE_ENBALE', '0') == '1' else configs_gating_preset['default']

    grid = lambda args: (triton.cdiv(s_q, args["BLOCK_M"]), B * H, 1)
    _attn_fwd_gating[grid](
        q, k, score,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        score.stride(0), score.stride(1), score.stride(2), score.stride(3),
        H, s_q, s_k,
        HEAD_DIM=D,
        **kernel_config
    )
    return score

def get_select_indices_topk(q, k, sparsity):
    score = cal_score(q, k)
    block_indices, block_indices_lens = get_select_indices_topk_from_score(score, sparsity)
    return block_indices, block_indices_lens

def get_select_indices_topk_from_score(score, sparsity):
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

def get_select_indices_cdf(q, k, cdf_threshold):
    score = cal_score(q, k)
    head_dim = q.shape[-1]
    block_indices, block_indices_lens = get_select_indices_cdf_from_score(score, cdf_threshold, 1 / head_dim**0.5)
    return block_indices, block_indices_lens

def get_select_indices_cdf_from_score(score, cdf_threshold, sm_scale):
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

def get_select_indices_cdf_topk(q, k, sparsity, cdf_threshold):
    score = cal_score(q, k)
    head_dim = q.shape[-1]
    block_indices, block_indices_lens = get_select_indices_cdf_topk_from_score(score, sparsity, cdf_threshold, 1 / head_dim**0.5)
    return block_indices, block_indices_lens

def get_select_indices_cdf_topk_from_score(score, sparsity, cdf_threshold, sm_scale):
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

def get_select_indices(q, k, sparsity, cdf_threshold):
    if sparsity is not None and cdf_threshold is None:
        block_indices, block_indices_lens = get_select_indices_topk(q, k, sparsity)
    elif sparsity is None and cdf_threshold is not None:
        block_indices, block_indices_lens = get_select_indices_cdf(q, k, cdf_threshold)
    elif sparsity is not None and cdf_threshold is not None:
        block_indices, block_indices_lens = get_select_indices_cdf_topk(q, k, sparsity, cdf_threshold)
    else:
        raise ValueError
    return block_indices, block_indices_lens

def get_select_indices_from_score(score, sparsity, cdf_threshold):
    if sparsity is not None and cdf_threshold is None:
        block_indices, block_indices_lens = get_select_indices_topk_from_score(score, sparsity)
    elif sparsity is None and cdf_threshold is not None:
        block_indices, block_indices_lens = get_select_indices_cdf_from_score(score, cdf_threshold)
    elif sparsity is not None and cdf_threshold is not None:
        block_indices, block_indices_lens = get_select_indices_cdf_topk_from_score(score, sparsity, cdf_threshold)
    else:
        raise ValueError
    return block_indices, block_indices_lens

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

    LN2 = 0.6931471824645996
    lse = M * LN2

    return o, lse

def attn_bwd_bsa_varlen_triton(
    do,
    q,
    k,
    v,
    o,
    dq,
    dk,
    dv,
    sm_scale,
    M,
    block_indices,
    block_indices_lens,
    chunk_size_q,
    chunk_size_k,
    sparsity,
    kv_valid_mask=None,
):
    RCP_LN2 = 1.4426950408889634
    M = M * RCP_LN2 # ln -> log2

    do = do.contiguous()
    # assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()

    BATCH, N_HEAD, N_CTX, HEAD_DIM = q.shape
    N_CTX_KV = k.shape[-2]

    RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2) # reciprocal
    arg_k = k
    arg_k = arg_k * (sm_scale * RCP_LN2)

    if min(chunk_size_q, chunk_size_k) >= 128:
        PRE_BLOCK = 128
    else:
        PRE_BLOCK = min(chunk_size_q, chunk_size_k)

    assert N_CTX % PRE_BLOCK == 0
    pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
    delta = torch.empty_like(M)
    _attn_bwd_preprocess[pre_grid](
        o, do,
        delta,
        N_CTX,
        BLOCK_M=PRE_BLOCK,
        HEAD_DIM=HEAD_DIM
    )

    _N_cols = N_CTX_KV // chunk_size_k
    _Hn = block_indices.shape[1]
    _ki_list, _kl_list = [], []
    for _h in range(_Hn):
        _ki_h, _kl_h = create_indices_k_from_indices_q_varlen(
            block_indices=block_indices[:, _h:_h+1],
            N_cols_mask=_N_cols,
        )
        _ki_list.append(_ki_h)
        _kl_list.append(_kl_h)
        del _ki_h, _kl_h
    block_indices_k = torch.cat(_ki_list, dim=1)
    block_indices_k_lens = torch.cat(_kl_list, dim=1)
    del _ki_list, _kl_list

    block_indices = block_indices.contiguous()
    block_indices_lens = block_indices_lens.contiguous()
    block_indices_k = block_indices_k.contiguous()
    block_indices_k_lens = block_indices_k_lens.contiguous()

    has_kv_mask = 1 if kv_valid_mask is not None else 0
    _kv_mask = kv_valid_mask if kv_valid_mask is not None else torch.empty(0, dtype=torch.bool, device=q.device)

    config_key = 'BLOCK_N_DQ_LG=64' if chunk_size_k == 64 else 'default'
    kernel_config = {} if os.environ.get('TRITON_AUTOTUNE_ENBALE', '0') == '1' else configs_bwd_dkdv_bsa_varlen_preset[config_key]

    grid_dkdv = lambda args: (triton.cdiv(arg_k.shape[2], args["BLOCK_N"]), 1, arg_k.shape[0] * arg_k.shape[1])
    _attn_bwd_dkdv_bsa_varlen_wrapper[grid_dkdv](
        q, arg_k, v, sm_scale,
        do,
        dk, dv,
        M,
        delta,
        block_indices_k,
        block_indices_k_lens,
        _kv_mask,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        M.stride(0), M.stride(1), M.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        block_indices_k.stride(0), block_indices_k.stride(1), block_indices_k.stride(2), block_indices_k.stride(3),
        block_indices_k_lens.stride(0), block_indices_k_lens.stride(1), block_indices_k_lens.stride(2),
        N_HEAD, N_CTX,
        BLOCK_M=chunk_size_q,
        BLOCK_N_DQ_LG=chunk_size_k,
        HEAD_DIM=HEAD_DIM,
        SPARSITY=sparsity,
        HAS_KV_MASK=has_kv_mask,
        **kernel_config
    )

    config_key = 'BLOCK_N_DQ_LG=64' if chunk_size_k == 64 else 'default'
    if chunk_size_k > 128:
        bwd_dq_func = _attn_bwd_dq_bsa_varlen_wrapper
        kernel_config = {} if os.environ.get('TRITON_AUTOTUNE_ENBALE', '0') == '1' else configs_bwd_dq_bsa_varlen_preset[config_key]
    else:
        bwd_dq_func = _attn_bwd_dq_bsa_varlen_align_wrapper
        kernel_config = {} if os.environ.get('TRITON_AUTOTUNE_ENBALE', '0') == '1' else configs_bwd_dq_bsa_varlen_align_preset[config_key]

    grid_dq = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), 1, q.shape[0] * q.shape[1])
    bwd_dq_func[grid_dq](
        q, arg_k, v,
        do,
        dq,
        M,
        delta,
        block_indices,
        block_indices_lens,
        _kv_mask,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        M.stride(0), M.stride(1), M.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        block_indices.stride(0), block_indices.stride(1), block_indices.stride(2), block_indices.stride(3),
        block_indices_lens.stride(0), block_indices_lens.stride(1), block_indices_lens.stride(2),
        N_HEAD, N_CTX,
        BLOCK_M=chunk_size_q,
        BLOCK_N_DQ_LG=chunk_size_k,
        HEAD_DIM=HEAD_DIM,
        SPARSITY=sparsity,
        HAS_KV_MASK=has_kv_mask,
        **kernel_config
    )

def make_block_indices_varlen_cp_list(block_indices, cp_size, num_blocks_k_full):
    """
    Args:
        block_indices: [B, H, num_blocks_q_per_cp_rank, num_blocks_k_full]

    Return:
        a list of [block_indices, block_indices_lens] for k from each cp_rank
            - each block_indices starts from zero
            - block_indices_lens indicates the valid number of elements in the last dimension of block_indices
    """
    res = []
    num_blocks_per_rank = num_blocks_k_full // cp_size
    for i in range(cp_size):
        block_indices_tmp = block_indices.clone()
        min_block_idx = i * num_blocks_per_rank
        block_indices_tmp -= min_block_idx
        block_indices_tmp[block_indices_tmp < 0] = num_blocks_per_rank # block_indices_tmp < 0 indicate invalid indices, set them to num_blocks_per_rank in order to sort them to the tail, so that the first N elements of the block_indices indicated by block_indices_lens are valid

        block_indices_tmp, _ = torch.sort(block_indices_tmp, dim=-1)

        block_indices_tmp_lens = (block_indices_tmp < num_blocks_per_rank).sum(dim=-1)

        res.append([block_indices_tmp, block_indices_tmp_lens])

    return res

def flash_attn_fwd_softmax_lse_correction(
    softmax_lse: torch.Tensor,
    softmax_lse_per_step: torch.Tensor,
):
    """Merge softmax stats of each step in Attention with context parallelism"""
    max_scale = torch.max(softmax_lse, softmax_lse_per_step)
    min_scale = torch.min(softmax_lse, softmax_lse_per_step)
    lse_diff = min_scale - max_scale
    lse_diff = lse_diff.nan_to_num(nan=0.) # handle cases: tensor(-inf) - tensor(-inf) = tensor(nan); In the current cp implementation, it is possible that lses of 2 cp ranks are both -inf, if no block is selected from both cp ranks. In such cases, the finally corrected lse should remain -inf.
    new_scale = max_scale + torch.log1p(torch.exp(lse_diff)) # a + ln(1 + e^(b - a)) = ln(e^a) + ln(1 + e^(b - a)) = ln(e^a + e^b)
    softmax_lse.copy_(new_scale)

def flash_attn_fwd_out_correction_init(
    out_init_step: torch.Tensor, # b h s d
    softmax_lse: torch.Tensor, # b h s
    softmax_lse_init_step: torch.Tensor,
):
    """Merge partial outputs of the first step in Attention with context parallelism"""
    softmax_lse_corrected_exp = torch.exp(softmax_lse_init_step - softmax_lse)
    softmax_lse_corrected_exp = softmax_lse_corrected_exp.unsqueeze(-1)
    out_corrected = out_init_step * softmax_lse_corrected_exp
    return out_corrected.to(out_init_step.dtype)


def flash_attn_fwd_out_correction(
    out: torch.Tensor,
    out_per_step: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_lse_per_step: torch.Tensor,
):
    """Merge partial outputs of each step in Attention with context parallelism"""
    softmax_lse_corrected_exp = torch.exp(softmax_lse_per_step - softmax_lse)
    softmax_lse_corrected_exp = softmax_lse_corrected_exp.unsqueeze(-1)
    out_corrected = out_per_step * softmax_lse_corrected_exp
    out.add_(out_corrected)

def topk_sort(score, num_chunks_selected):
    block_indices = torch.topk(score, num_chunks_selected)[1]
    block_indices, _ = torch.sort(block_indices, dim=-1)
    return block_indices

class _attention_bsa(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, chunk_size_q, chunk_size_k, sparsity, cdf_threshold, sm_scale,
                kv_valid_mask=None, q_valid_mask=None, score_modifier=None):
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        # ---------------------- gating ----------------------
        if q_valid_mask is not None:
            q_cmp = masked_mean_pooling_compression(q, chunk_size_q, q_valid_mask)
        elif kv_valid_mask is not None and q.shape[2] == k.shape[2]:
            q_cmp = masked_mean_pooling_compression(q, chunk_size_q, kv_valid_mask)
        else:
            q_cmp = mean_pooling_compression(q, chunk_size_q)

        if kv_valid_mask is not None:
            k_cmp = masked_mean_pooling_compression(k, chunk_size_k, kv_valid_mask)
        else:
            k_cmp = mean_pooling_compression(k, chunk_size_k)

        score = cal_score(q_cmp, k_cmp)

        if score_modifier is not None:
            score = score * score_modifier

        if sparsity is not None and cdf_threshold is None:
            block_indices, block_indices_lens = get_select_indices_topk_from_score(score, sparsity)
        elif sparsity is None and cdf_threshold is not None:
            block_indices, block_indices_lens = get_select_indices_cdf_from_score(score, cdf_threshold, sm_scale)
        elif sparsity is not None and cdf_threshold is not None:
            block_indices, block_indices_lens = get_select_indices_cdf_topk_from_score(score, sparsity, cdf_threshold, sm_scale)
        else:
            raise ValueError("Either sparsity or cdf_threshold must be provided")

        # ---------------------- bsa ----------------------

        o, lse = attn_fwd_bsa_varlen_triton(
            q, k, v,
            sm_scale, block_indices, block_indices_lens,
            chunk_size_q, chunk_size_k,
            sparsity,
            kv_valid_mask=kv_valid_mask,
        )

        ctx.save_for_backward(q, k, v, o, lse, block_indices, block_indices_lens)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.chunk_size_q = chunk_size_q
        ctx.chunk_size_k = chunk_size_k
        ctx.sparsity = sparsity
        ctx.kv_valid_mask = kv_valid_mask

        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, block_indices, block_indices_lens = ctx.saved_tensors

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        attn_bwd_bsa_varlen_triton(
            do,
            q,
            k,
            v,
            o,
            dq,
            dk,
            dv,
            ctx.sm_scale,
            lse,
            block_indices,
            block_indices_lens,
            ctx.chunk_size_q,
            ctx.chunk_size_k,
            ctx.sparsity,
            kv_valid_mask=ctx.kv_valid_mask,
        )

        return dq, dk, dv, None, None, None, None, None, None, None, None

flash_attn_bsa = _attention_bsa.apply


class _attention_bsa_audio_guided(torch.autograd.Function):
    """BSA with audio-guided score modulation (Gate 2: Audio Spatial Concentration Gate).

    The audio saliency modulates the block-level QK scores BEFORE topk selection,
    but the actual sparse attention Triton kernel remains unchanged. This ensures
    Triton-level efficiency for the compute-heavy attention, while the lightweight
    score modulation uses standard PyTorch ops (torch.compile friendly).
    """

    @staticmethod
    def forward(ctx, q, k, v, chunk_size_q, chunk_size_k, sparsity, cdf_threshold, sm_scale,
                audio_block_saliency, audio_gate_value, audio_boost_gamma,
                kv_valid_mask=None, q_valid_mask=None,
                audio_norms_block_order=None, audio_weighted_lambda=0.0,
                score_modifier=None):
        """
        Args:
            audio_block_saliency: [N_blocks_k] or [B, 1, N_blocks_k] — per-key-block saliency in [0,1]
            audio_gate_value: scalar in [0,1] — Audio Spatial Concentration Gate g_a
            audio_boost_gamma: scalar >= 0 — boost strength γ
            audio_norms_block_order: [S] — per-token norms in block-rearranged order (for Path B)
            audio_weighted_lambda: float — Path B weighting strength (0=standard pooling)
            score_modifier: optional broadcastable tensor — external additive boost term
                (e.g., γ_var * ρ from variance guidance). Combined with audio INSIDE
                one boost bracket: score *= (1 + γ_audio * S̃ + score_modifier).
        """
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        B, H, Sq, D = q.shape
        Sk = k.shape[2]

        # ---------------------- gating (block-level scoring) ----------------------
        if q_valid_mask is not None:
            q_cmp = masked_mean_pooling_compression(q, chunk_size_q, q_valid_mask)
        elif kv_valid_mask is not None and q.shape[2] == k.shape[2]:
            q_cmp = masked_mean_pooling_compression(q, chunk_size_q, kv_valid_mask)
        else:
            q_cmp = mean_pooling_compression(q, chunk_size_q)

        use_path_b = (audio_norms_block_order is not None and audio_weighted_lambda > 0)
        if use_path_b:
            k_cmp = audio_weighted_mean_pooling_compression(
                k, chunk_size_k, audio_norms_block_order,
                valid_mask=kv_valid_mask, lambda_w=audio_weighted_lambda,
            )
        elif kv_valid_mask is not None:
            k_cmp = masked_mean_pooling_compression(k, chunk_size_k, kv_valid_mask)
        else:
            k_cmp = mean_pooling_compression(k, chunk_size_k)

        score = cal_score(q_cmp, k_cmp)  # [B, H, N_blocks_q, N_blocks_k]

        # ---------------------- Unified Additive Boost (md 3.5.1.2) ----------------------
        # I_{i,j} = P^QK * (1 + γ_audio * S̃ + additive_boost_from_external)
        # All boost terms are ADDITIVE inside one bracket, then multiply P^QK.
        N_blocks_k = score.shape[-1]
        boost = torch.zeros(1, device=score.device, dtype=score.dtype)

        has_audio = (audio_block_saliency is not None
                     and audio_block_saliency.numel() > 0
                     and audio_boost_gamma > 0)
        if has_audio:
            if audio_block_saliency.dim() == 1:
                saliency = audio_block_saliency.view(1, 1, 1, -1)
            else:
                saliency = audio_block_saliency.unsqueeze(-2)
            uniform_val = 1.0 / N_blocks_k
            gated_saliency = audio_gate_value * saliency + (1.0 - audio_gate_value) * uniform_val
            boost = boost + audio_boost_gamma * gated_saliency

        if score_modifier is not None:
            boost = boost + score_modifier

        score = score * (1.0 + boost)

        # ---------------------- topk selection from modulated scores ----------------------
        if sparsity is not None and cdf_threshold is None:
            block_indices, block_indices_lens = get_select_indices_topk_from_score(score, sparsity)
        elif sparsity is None and cdf_threshold is not None:
            block_indices, block_indices_lens = get_select_indices_cdf_from_score(score, cdf_threshold, sm_scale)
        elif sparsity is not None and cdf_threshold is not None:
            block_indices, block_indices_lens = get_select_indices_cdf_topk_from_score(score, sparsity, cdf_threshold, sm_scale)
        else:
            raise ValueError("Either sparsity or cdf_threshold must be provided")

        # ---------------------- bsa (Triton kernel unchanged) ----------------------
        o, lse = attn_fwd_bsa_varlen_triton(
            q, k, v,
            sm_scale, block_indices, block_indices_lens,
            chunk_size_q, chunk_size_k,
            sparsity,
            kv_valid_mask=kv_valid_mask,
        )

        ctx.save_for_backward(q, k, v, o, lse, block_indices, block_indices_lens)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.chunk_size_q = chunk_size_q
        ctx.chunk_size_k = chunk_size_k
        ctx.sparsity = sparsity
        ctx.kv_valid_mask = kv_valid_mask

        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, block_indices, block_indices_lens = ctx.saved_tensors

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        attn_bwd_bsa_varlen_triton(
            do,
            q,
            k,
            v,
            o,
            dq,
            dk,
            dv,
            ctx.sm_scale,
            lse,
            block_indices,
            block_indices_lens,
            ctx.chunk_size_q,
            ctx.chunk_size_k,
            ctx.sparsity,
            kv_valid_mask=ctx.kv_valid_mask,
        )

        # Order: q, k, v, chunk_size_q, chunk_size_k, sparsity, cdf_threshold, sm_scale,
        #         audio_block_saliency, audio_gate_value, audio_boost_gamma,
        #         kv_valid_mask, q_valid_mask, audio_norms_block_order, audio_weighted_lambda,
        #         score_modifier
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None


flash_attn_bsa_audio_guided = _attention_bsa_audio_guided.apply


def compute_timestep_reliability_gate(
    timestep_ratio: torch.Tensor,
    mode: str = "power",
    power: float = 1.5,
    scale: float = 10.0,
    threshold: float = 0.7,
) -> torch.Tensor:
    """Compute Gate 1: Timestep Reliability Gate r_t (non-learnable).

    At high noise (t→1): r_t→0 (don't trust audio guidance, bridge residual is noise-driven).
    At low noise (t→0): r_t→1 (trust audio guidance, bridge residual reflects real saliency).

    Two modes:
      - "power": r_t = (1 - t)^power. Physically motivated by flow matching signal decay.
        power=1.5 is recommended (moderate decay, default).
      - "sigmoid": r_t = sigmoid(scale * (threshold - t)). Sharp transition.

    Args:
        timestep_ratio: scalar or [B] — normalized timestep in [0, 1] (0=clean, 1=noise)
        mode: "power" (default) or "sigmoid"
        power: exponent for power-law mode (default 1.5)
        scale: sharpness for sigmoid mode (default 10.0)
        threshold: midpoint for sigmoid mode (default 0.7)

    Returns:
        r_t: scalar in [0, 1]
    """
    if mode == "power":
        r_t = (1.0 - timestep_ratio.clamp(0.0, 1.0)) ** power
    else:
        r_t = torch.sigmoid(scale * (threshold - timestep_ratio))
    return r_t


def compute_audio_spatial_concentration_gate(
    token_norms: torch.Tensor,
    valid_mask: torch.Tensor = None,
    alpha: float = 5.0,
    beta: float = 2.0,
) -> torch.Tensor:
    """Compute Gate 2: Audio Spatial Concentration Gate g_a from per-token L2 norms.

    g_a = sigmoid(alpha * CV - beta)
    where CV = std(norms) / (mean(norms) + eps)

    Args:
        token_norms: [N_tokens] — L2 norms of bridge residual per video token
        valid_mask: [N_tokens] bool — True for real tokens, False for padded
        alpha, beta: fixed hyperparameters (non-learnable)

    Returns:
        g_a: scalar in [0, 1]
    """
    eps = 1e-6
    if valid_mask is not None:
        mask_f = valid_mask.float()
        count = mask_f.sum().clamp(min=1.0)
        mean_a = (token_norms * mask_f).sum() / count
        var_a = ((token_norms - mean_a).pow(2) * mask_f).sum() / count
    else:
        mean_a = token_norms.mean()
        var_a = token_norms.var(unbiased=False)

    std_a = var_a.sqrt()
    cv = std_a / (mean_a + eps)
    g_a = torch.sigmoid(alpha * cv - beta)
    return g_a


def audio_weighted_mean_pooling_compression(
    x: torch.Tensor,
    block_size: int,
    token_norms: torch.Tensor,
    valid_mask: torch.Tensor = None,
    lambda_w: float = 1.0,
) -> torch.Tensor:
    """Audio-weighted mean pooling for K-side block representatives (Path B).

    w_m = 1 + lambda_w * (a_m / max(a_m' in block) + eps)
    K_bar_j = sum(w_m * K_m) / sum(w_m)

    Automatically degrades to standard mean pooling when:
    - silence: all a_m ≈ 0 → w_m ≈ 1
    - high noise: all a_m similar → a_m/max ≈ 1 → uniform weighting
    - lambda_w = 0: w_m = 1 exactly

    Args:
        x: [B, H, S, D] — K or V tokens (already in 3D-block rearranged order)
        block_size: number of tokens per block (e.g., 64)
        token_norms: [S] — per-token L2 norms of bridge residual (block-rearranged order)
        valid_mask: [S] bool — True for real tokens, False for padded
        lambda_w: weighting strength (default 1.0; 0=standard mean pooling)

    Returns:
        x_cmp: [B, H, N_blocks, D] — weighted block representatives
    """
    B, H, S, D = x.shape
    eps = 1e-6
    num_blocks = math.ceil(S / block_size)
    if S % block_size != 0:
        pad_len = num_blocks * block_size - S
        x = F.pad(x, (0, 0, 0, pad_len))
        token_norms = F.pad(token_norms, (0, pad_len), value=0.0)
        if valid_mask is not None:
            valid_mask = F.pad(valid_mask, (0, pad_len), value=False)

    # Reshape to [num_blocks, block_size]
    norms_blocks = token_norms.view(num_blocks, block_size)

    # Intra-block normalized weighting: w_m = 1 + λ * (a_m / max_in_block)
    if valid_mask is not None:
        mask_blocks = valid_mask.view(num_blocks, block_size).float()
        # Mask out padded positions for max computation
        norms_masked = norms_blocks * mask_blocks + (-1e9) * (1 - mask_blocks)
        block_max = norms_masked.max(dim=1, keepdim=True).values.clamp(min=eps)
        # Reset negatives from masking
        norms_for_weight = norms_blocks * mask_blocks
    else:
        block_max = norms_blocks.max(dim=1, keepdim=True).values.clamp(min=eps)
        norms_for_weight = norms_blocks
        mask_blocks = None

    # w_m = 1 + λ * (a_m / max)
    weights = 1.0 + lambda_w * (norms_for_weight / block_max)  # [num_blocks, block_size]

    if mask_blocks is not None:
        weights = weights * mask_blocks  # zero out padded positions

    # Reshape x for block-wise weighted mean: [B, H, num_blocks, block_size, D]
    x_blocks = x.view(B, H, num_blocks, block_size, D)

    # weights: [num_blocks, block_size] → [1, 1, num_blocks, block_size, 1] for broadcasting
    w = weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, num_blocks, block_size, 1]
    w_sum = w.sum(dim=3, keepdim=True).clamp(min=1.0)  # [1, 1, num_blocks, 1, 1]

    x_cmp = (x_blocks * w).sum(dim=3) / w_sum.squeeze(3)  # [B, H, num_blocks, D]
    return x_cmp


def compute_channel_variance_density(
    x: torch.Tensor,
    block_size: int,
    bt: int, bh: int, bw: int,
    valid_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Compute per-block channel-variance information density (non-learnable).

    Averages across all (local) heads for head-agnostic block-level statistics
    (different heads project into different subspaces — single-head would bias
    selection toward that head's variance pattern). Mean pooling stays in native
    dtype (bf16) to avoid large float32 copies; only the tiny [num_blocks, D]
    result converts to float32 for numerically precise variance computation.

    Note (SP): under sequence parallelism each rank owns a head shard, so ρ is
    computed from local heads only. This keeps the per-layer cost minimal (no
    extra collective). ρ is a coarse per-block density signal whose estimate is
    stable across head subsets, and each head's attention is computed on its own
    rank, so the per-rank ρ is correct for that rank's heads.

    Args:
        x: [B, H, S, D] V tensor in 3D-block-rearranged order (V = x @ W_V, no RMSNorm/RoPE).
        block_size: bt * bh * bw tokens per block.
        bt, bh, bw: block shape in temporal, height, width.
        valid_mask: [S] bool — True for real tokens, False for padding.

    Returns:
        rho: [num_blocks] ∈ [0, 1] — normalized variance density per block.
    """
    S, D = x.shape[2], x.shape[3]
    num_blocks = S // block_size
    spatial_size = bh * bw

    # Average across heads in native dtype (bf16) — no large float32 copy.
    # x[0]: [H, S, D] → .mean(dim=0): [S, D]. bf16 mean is numerically stable.
    x_avg = x[0].mean(dim=0)                               # [S, D], bf16, head-agnostic
    x_blocks = x_avg.view(num_blocks, block_size, D)        # [num_blocks, block_size, D]

    if valid_mask is not None:
        mask_blocks = valid_mask.view(num_blocks, block_size)
        mask_f = mask_blocks.unsqueeze(-1).to(x_blocks.dtype)  # [num_blocks, block_size, 1]
        counts = mask_f.sum(dim=1).clamp(min=1.0)              # [num_blocks, 1]
    else:
        mask_f = None
        counts = None

    # --- Spatial Channel Variance σ²_ch-s ---
    # Per-block mean per channel (bf16), then variance across channels (float32)
    if mask_f is not None:
        channel_means = (x_blocks * mask_f).sum(dim=1) / counts  # [num_blocks, D], bf16
    else:
        channel_means = x_blocks.mean(dim=1)                      # [num_blocks, D], bf16

    cm_f = channel_means.float()                                  # [num_blocks, D] — only this is float32 (~2.5MB)
    global_cm = cm_f.mean(dim=-1, keepdim=True)                   # [num_blocks, 1]
    spatial_var = ((cm_f - global_cm) ** 2).mean(dim=-1)          # [num_blocks]

    # --- Temporal Channel Variance σ²_ch-t ---
    if bt > 1:
        x_temporal = x_blocks.view(num_blocks, bt, spatial_size, D)

        if valid_mask is not None:
            mask_t_3d = mask_blocks.view(num_blocks, bt, spatial_size)
            mask_t = mask_t_3d.unsqueeze(-1).to(x_blocks.dtype)   # [num_blocks, bt, spatial_size, 1]
            fc = mask_t.sum(dim=2).clamp(min=1.0)                 # [num_blocks, bt, 1]
            frame_means = (x_temporal * mask_t).sum(dim=2) / fc    # [num_blocks, bt, D], bf16

            # Frame-level validity: only diff between pairs where BOTH frames have ≥1 valid token
            frame_valid = mask_t_3d.any(dim=2)                     # [num_blocks, bt]
            both_valid = frame_valid[:, 1:] & frame_valid[:, :-1]  # [num_blocks, bt-1]
            bv_f = both_valid.float().unsqueeze(-1)                # [num_blocks, bt-1, 1]
            valid_pairs = bv_f.sum(dim=1).clamp(min=1.0)           # [num_blocks, 1]

            fm_f = frame_means.float()
            frame_diffs = fm_f[:, 1:, :] - fm_f[:, :-1, :]        # [num_blocks, bt-1, D]
            temporal_var = ((frame_diffs.pow(2) * bv_f).sum(dim=1) / valid_pairs).mean(dim=-1)
        else:
            frame_means = x_temporal.mean(dim=2)                    # [num_blocks, bt, D], bf16
            fm_f = frame_means.float()
            frame_diffs = fm_f[:, 1:, :] - fm_f[:, :-1, :]         # [num_blocks, bt-1, D]
            temporal_var = frame_diffs.pow(2).mean(dim=1).mean(dim=-1)
    else:
        temporal_var = torch.zeros(num_blocks, device=x.device, dtype=torch.float32)

    # --- Combine and normalize to [0, 1] ---
    combined = spatial_var + temporal_var
    rho = combined / combined.max().clamp(min=1e-6)

    return rho


def compute_block_audio_saliency(
    token_norms: torch.Tensor,
    chunk_size: int,
    valid_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Compute per-block audio saliency from per-token L2 norms.

    For each block j: S_j = mean(norms in block j) / max(mean norms across all blocks)
    Then clamp to [0, 1].

    Args:
        token_norms: [S] — per-token L2 norms (already in 3D-block rearranged order)
        chunk_size: block size (e.g., 64 for 4x4x4)
        valid_mask: [S] bool — True for real tokens, False for padded

    Returns:
        block_saliency: [N_blocks] in [0, 1]
    """
    S = token_norms.shape[0]
    num_blocks = math.ceil(S / chunk_size)
    if S % chunk_size != 0:
        pad_len = num_blocks * chunk_size - S
        token_norms = F.pad(token_norms, (0, pad_len), value=0.0)
        if valid_mask is not None:
            valid_mask = F.pad(valid_mask, (0, pad_len), value=False)

    norms_blocks = token_norms.view(num_blocks, chunk_size)

    if valid_mask is not None:
        mask_blocks = valid_mask.view(num_blocks, chunk_size).float()
        counts = mask_blocks.sum(dim=1).clamp(min=1.0)
        block_means = (norms_blocks * mask_blocks).sum(dim=1) / counts
    else:
        block_means = norms_blocks.mean(dim=1)

    max_val = block_means.max().clamp(min=1e-6)
    block_saliency = block_means / max_val
    return block_saliency


def rearrange_THW_to_3d_block(x, Nt, Nh, Nw, t, h, w, D):
    B, H, _, D = x.shape
    x = x.view(B, H, Nt, t, Nh, h, Nw, w, D)
    x = x.permute(0, 1, 2, 4, 6, 3, 5, 7, 8)  # B H Nt Nh Nw t h w D
    return x.contiguous().view(B, H, Nt * Nh * Nw * t * h * w, D)

def rearrange_3d_block_to_THW(x, Nt, Nh, Nw, t, h, w, D):
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


def flash_attn_bsa_3d_variance_guided(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    latent_shape_q,
    latent_shape_k,
    variance_boost_gamma: float = 1.0,
    sparsity=0.875,
    cdf_threshold=None,
    chunk_3d_shape_q=[4, 4, 4],
    chunk_3d_shape_k=[4, 4, 4],
    valid_mask=None,
) -> torch.Tensor:
    """BSA with channel-variance information density score modulation.

    Completely independent from audio guidance — no shared state or code.

    Computes ρ_j on V blocks (key/value side only), then applies:
      score_{i,j} *= (1 + γ * ρ_j)

    Query-awareness comes from P^QK being per-query (zero-gate effect):
      - Sky query → lip key: P^QK ≈ 0, boost = 1.85, final ≈ 0 → not selected
      - Face query → lip key: P^QK ≈ 0.12, boost = 1.85, final = 0.222 → selected
    Same ρ_j, different P^QK → different top-k per query block.

    Why V (not K or hidden states)?
      V = x @ W_V — raw linear projection, no RMSNorm, no RoPE distortion.
      V channel variance directly predicts information contribution I(V_block; Output):
      low variance → V rows near-parallel → output gain ≈ 0 regardless of attention weight.
    """
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

    with torch.no_grad():
        rho = compute_channel_variance_density(v, chunk_size_k, tk, hk, wk, kv_valid_mask)

    # Multiplicative modifier for variance-only path: score *= (1 + γ * ρ)
    score_modifier = (
        1.0 + variance_boost_gamma * rho.to(q.dtype).view(1, 1, 1, -1)
    ).contiguous()

    output = flash_attn_bsa(
        q, k, v, chunk_size_q, chunk_size_k, sparsity, cdf_threshold,
        1 / head_dim**0.5, kv_valid_mask, None, score_modifier,
    )

    output = rearrange_3d_block_to_THW(output, Ntq, Nhq, Nwq, tq, hq, wq, output.shape[-1])
    return output


def _pad_1d_for_bsa(x, chunk_size):
    """Pad a 1D sequence [B, H, S, D] to be divisible by chunk_size. Returns (padded_x, valid_mask_or_None, orig_len)."""
    B, H, S, D = x.shape
    remainder = S % chunk_size
    if remainder == 0:
        return x, None, S
    pad_len = chunk_size - remainder
    x_padded = F.pad(x, (0, 0, 0, pad_len)).contiguous()
    valid_mask = torch.zeros(S + pad_len, dtype=torch.bool, device=x.device)
    valid_mask[:S] = True
    return x_padded, valid_mask, S


def flash_attn_bsa_cross(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_structure: str,
    k_structure: str,
    q_grid_size=None,
    k_grid_size=None,
    sparsity=0.875,
    cdf_threshold=None,
    chunk_size_q=64,
    chunk_size_k=64,
    chunk_3d_shape_q=None,
    chunk_3d_shape_k=None,
) -> torch.Tensor:
    """Cross-modal BSA supporting asymmetric Q/K structures (3D video vs 1D audio).

    Args:
        q: [B, H, Sq, D]
        k: [B, H, Sk, D]
        v: [B, H, Sk, D]
        q_structure: "3d" (video with grid T,H,W) or "1d" (audio sequential)
        k_structure: "3d" or "1d"
        q_grid_size: (T, H, W) when q_structure="3d"
        k_grid_size: (T, H, W) when k_structure="3d"
        sparsity: fraction of K blocks to skip
        chunk_size_q: block size for 1D Q (must be multiple of 64)
        chunk_size_k: block size for 1D K (must be multiple of 64)
        chunk_3d_shape_q: [t,h,w] for 3D Q blocking
        chunk_3d_shape_k: [t,h,w] for 3D K blocking
    """
    _, _, Sq, head_dim = q.shape
    _, _, Sk, _ = k.shape

    q_valid_mask = None
    k_valid_mask = None
    q_orig_len = Sq
    k_orig_len = Sk

    if q_structure == "3d":
        assert q_grid_size is not None
        Tq, Hq, Wq = q_grid_size
        tq, hq, wq = chunk_3d_shape_q or [4, 4, 4]
        pad_tq = (tq - Tq % tq) % tq
        pad_hq = (hq - Hq % hq) % hq
        pad_wq = (wq - Wq % wq) % wq

        if pad_tq > 0 or pad_hq > 0 or pad_wq > 0:
            B, H, _, D = q.shape
            Tq_p, Hq_p, Wq_p = Tq + pad_tq, Hq + pad_hq, Wq + pad_wq
            idx_t = torch.arange(Tq_p, device=q.device)
            idx_h = torch.arange(Hq_p, device=q.device)
            idx_w = torch.arange(Wq_p, device=q.device)
            q_valid_mask = ((idx_t[:, None, None] < Tq) & (idx_h[None, :, None] < Hq) & (idx_w[None, None, :] < Wq)).reshape(-1).contiguous()
            q_3d = q.view(B, H, Tq, Hq, Wq, D)
            q = F.pad(q_3d, (0, 0, 0, pad_wq, 0, pad_hq, 0, pad_tq)).reshape(B, H, -1, D).contiguous()
            Tq, Hq, Wq = Tq_p, Hq_p, Wq_p

        Ntq, Nhq, Nwq = Tq // tq, Hq // hq, Wq // wq
        q = rearrange_THW_to_3d_block(q, Ntq, Nhq, Nwq, tq, hq, wq, q.shape[-1])
        if q_valid_mask is not None:
            q_valid_mask = rearrange_THW_to_3d_block_1d(q_valid_mask, Ntq, Nhq, Nwq, tq, hq, wq)
        actual_chunk_size_q = tq * hq * wq
    else:
        q, q_valid_mask, q_orig_len = _pad_1d_for_bsa(q, chunk_size_q)
        actual_chunk_size_q = chunk_size_q
        Ntq = Nhq = Nwq = tq = hq = wq = None

    if k_structure == "3d":
        assert k_grid_size is not None
        Tk, Hk, Wk = k_grid_size
        tk, hk, wk = chunk_3d_shape_k or [4, 4, 4]
        pad_tk = (tk - Tk % tk) % tk
        pad_hk = (hk - Hk % hk) % hk
        pad_wk = (wk - Wk % wk) % wk

        if pad_tk > 0 or pad_hk > 0 or pad_wk > 0:
            B, H, _, D = k.shape
            Tk_p, Hk_p, Wk_p = Tk + pad_tk, Hk + pad_hk, Wk + pad_wk
            idx_t = torch.arange(Tk_p, device=k.device)
            idx_h = torch.arange(Hk_p, device=k.device)
            idx_w = torch.arange(Wk_p, device=k.device)
            k_valid_mask = ((idx_t[:, None, None] < Tk) & (idx_h[None, :, None] < Hk) & (idx_w[None, None, :] < Wk)).reshape(-1).contiguous()
            B, H, _, D = k.shape
            k_3d = k.view(B, H, Tk, Hk, Wk, D)
            k = F.pad(k_3d, (0, 0, 0, pad_wk, 0, pad_hk, 0, pad_tk)).reshape(B, H, -1, D).contiguous()
            v_3d = v.view(B, H, Tk, Hk, Wk, D)
            v = F.pad(v_3d, (0, 0, 0, pad_wk, 0, pad_hk, 0, pad_tk)).reshape(B, H, -1, D).contiguous()
            Tk, Hk, Wk = Tk_p, Hk_p, Wk_p

        Ntk, Nhk, Nwk = Tk // tk, Hk // hk, Wk // wk
        k = rearrange_THW_to_3d_block(k, Ntk, Nhk, Nwk, tk, hk, wk, k.shape[-1])
        v = rearrange_THW_to_3d_block(v, Ntk, Nhk, Nwk, tk, hk, wk, v.shape[-1])
        if k_valid_mask is not None:
            k_valid_mask = rearrange_THW_to_3d_block_1d(k_valid_mask, Ntk, Nhk, Nwk, tk, hk, wk)
        actual_chunk_size_k = tk * hk * wk
    else:
        k, k_valid_mask_k, k_orig_len = _pad_1d_for_bsa(k, chunk_size_k)
        v, _, _ = _pad_1d_for_bsa(v, chunk_size_k)
        if k_valid_mask_k is not None:
            k_valid_mask = k_valid_mask_k
        actual_chunk_size_k = chunk_size_k
        Ntk = Nhk = Nwk = tk = hk = wk = None

    output = flash_attn_bsa(q, k, v, actual_chunk_size_q, actual_chunk_size_k, sparsity, cdf_threshold, 1 / head_dim**0.5, k_valid_mask, q_valid_mask)

    if q_structure == "3d":
        output = rearrange_3d_block_to_THW(output, Ntq, Nhq, Nwq, tq, hq, wq, output.shape[-1])
        if pad_tq > 0 or pad_hq > 0 or pad_wq > 0:
            B, H, _, D = output.shape
            Tq_orig, Hq_orig, Wq_orig = q_grid_size
            output = output.view(B, H, Tq, Hq, Wq, D)[:, :, :Tq_orig, :Hq_orig, :Wq_orig, :].contiguous().reshape(B, H, q_orig_len, D)
    else:
        if q_orig_len != output.shape[2]:
            output = output[:, :, :q_orig_len, :].contiguous()

    return output


def flash_attn_bsa_3d_audio_guided(
    q: torch.Tensor,  # [B, H, Sq, D]
    k: torch.Tensor,  # [B, H, Skv, D]
    v: torch.Tensor,  # [B, H, Skv, D]
    latent_shape_q,
    latent_shape_k,
    audio_token_norms: torch.Tensor = None,  # [Sq] per-token L2 norms of A->V bridge residual
    audio_gate_value: torch.Tensor = None,  # scalar gate g_a in [0,1]
    audio_boost_gamma: torch.Tensor = None,  # scalar boost γ >= 0
    sparsity=0.875,
    cdf_threshold=None,
    chunk_3d_shape_q=[4, 4, 4],
    chunk_3d_shape_k=[4, 4, 4],
    valid_mask=None,  # [Sq] bool, True=valid, False=padded
    audio_weighted_lambda: float = 0.0,  # Path B: weighted pooling strength (0=disabled)
    score_modifier: torch.Tensor = None,  # external additive boost (e.g., γ_var * ρ from variance guidance)
    variance_boost_gamma: float = 0.0,  # if > 0, compute ρ INSIDE on rearranged V (avoids double rearrange)
) -> torch.Tensor:
    """BSA with audio saliency score modulation.

    Audio-specific: gated saliency boost + optional audio-weighted K pooling.

    Combine modes for additional boosts (all additive inside one bracket):
      score *= (1 + γ_audio * S̃ + γ_var * ρ + score_modifier)

    For audio + variance combination, prefer passing `variance_boost_gamma > 0`
    instead of `score_modifier`: that lets us compute ρ on the **already
    rearranged** V tensor, avoiding a second `rearrange_THW_to_3d_block(v)`
    on the caller side. Both produce identical results, but the in-function
    path saves one V permute+contiguous (≈30-80 MB memory bandwidth on 2K).
    """
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

    # Optional in-function variance density (avoids the caller re-rearranging V).
    # Folded additively into score_modifier so audio + variance share one boost
    # bracket — no cross-term, exactly matches the standalone variance path.
    if variance_boost_gamma is not None and variance_boost_gamma > 0:
        with torch.no_grad():
            _rho = compute_channel_variance_density(v, chunk_size_k, tk, hk, wk, kv_valid_mask)
        _var_term = (variance_boost_gamma * _rho.to(q.dtype).view(1, 1, 1, -1)).contiguous()
        score_modifier = (score_modifier + _var_term) if score_modifier is not None else _var_term

    # Audio saliency computation (only when audio norms are provided)
    has_audio_norms = (audio_token_norms is not None and audio_token_norms.numel() > 0)
    if has_audio_norms:
        audio_norms_block_order = rearrange_THW_to_3d_block_1d(
            audio_token_norms, Ntk, Nhk, Nwk, tk, hk, wk
        )
        block_saliency = compute_block_audio_saliency(
            audio_norms_block_order, chunk_size_k, valid_mask=kv_valid_mask
        )
    else:
        block_saliency = None
        audio_norms_block_order = None
        if audio_gate_value is None:
            audio_gate_value = torch.ones(1, device=q.device, dtype=q.dtype)
        if audio_boost_gamma is None:
            audio_boost_gamma = torch.zeros(1, device=q.device, dtype=q.dtype)

    output = flash_attn_bsa_audio_guided(
        q, k, v, chunk_size_q, chunk_size_k, sparsity, cdf_threshold,
        1 / head_dim**0.5,
        block_saliency, audio_gate_value, audio_boost_gamma,
        kv_valid_mask,
        None,  # q_valid_mask (same grid in self-attn)
        audio_norms_block_order if audio_weighted_lambda > 0 else None,
        audio_weighted_lambda,
        score_modifier,
    )

    output = rearrange_3d_block_to_THW(output, Ntq, Nhq, Nwq, tq, hq, wq, output.shape[-1])
    return output
