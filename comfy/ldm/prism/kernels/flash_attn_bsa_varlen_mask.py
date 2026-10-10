import triton
import triton.language as tl
import os

from .common import autotune
"""
TRITON_REEVALUATE_KEY=1
- autotune whenever params in reevaluate keys change
- use in benchmark script to fine the best config

TRITON_AUTOTUNE_ENBALE=1
- if set to 0, autotune will not work, and the related params must be passed to the function call.
"""

configs_fwd_bsa_varlen_preset = {
    'default': {
        'BLOCK_N': 64,
        'num_stages': 3,
        'num_warps': 8,
    },
    'BLOCK_N_LG=64': {
        'BLOCK_N': 64,
        'num_stages': 3,
        'num_warps': 4,
    },
}
configs_fwd_bsa_varlen = [
    triton.Config({'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BN in [32, 64, 128] \
    for s in [2, 3, 4, 5] \
    for w in [4, 8] \
]

fwd_bsa_reevaluate_varlen_keys = ['N_CTX', 'BLOCK_M', 'BLOCK_N_LG', 'SPARSITY'] if os.environ.get('TRITON_REEVALUATE_KEY', '0') == '1' else []
@autotune(list(configs_fwd_bsa_varlen), key=fwd_bsa_reevaluate_varlen_keys)
@triton.jit
def _attn_fwd_bsa_varlen(
    Q, K, V, sm_scale, M, Out,
    block_indices, # [B, H, M_COMPRESS, S_MAX]
    block_indices_lens, # [B, H, M_COMPRESS]
    kv_valid_mask, # [N_CTX] bool, True=valid (shared across B,H)
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_bz, stride_bh, stride_bm, stride_bs,
    stride_lz, stride_lh, stride_lm,
    H, N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N_LG: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPARSITY: tl.constexpr,
    HAS_KV_MASK: tl.constexpr,
    ):

    """Compute online-softmax attention over selected key blocks and store log-sums."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
    b_offset = off_z.to(tl.int64) * stride_bz + off_h.to(tl.int64) * stride_bh
    l_offset = off_z.to(tl.int64) * stride_lz + off_h.to(tl.int64) * stride_lh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    KT_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    block_indices += b_offset + start_m * stride_bm
    block_indices_lens += l_offset + start_m * stride_lm
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/ln2
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    S = tl.load(block_indices_lens)
    for i in range(S):
        block_id = tl.load(block_indices + i * stride_bs).to(tl.int32)
        lo, hi = block_id * BLOCK_N_LG, (block_id + 1) * BLOCK_N_LG
        lo = tl.multiple_of(lo, BLOCK_N)
        KT_block_ptr_i = tl.advance(KT_block_ptr, (0, lo))
        V_block_ptr_i = tl.advance(V_block_ptr, (lo, 0))
        mask_offset = lo

        # loop over k, v and update accumulator
        for start_n in range(lo, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            kT = tl.load(KT_block_ptr_i)
            qkT = tl.dot(q, kT)

            if HAS_KV_MASK:
                offs_n = mask_offset + tl.arange(0, BLOCK_N)
                k_valid = tl.load(kv_valid_mask + offs_n)
                qkT = tl.where(k_valid[None, :], qkT, float('-inf'))

            m_ij = tl.maximum(m_i, tl.max(qkT, 1) * qk_scale)
            qkT = qkT * qk_scale - m_ij[:, None]
            p = tl.math.exp2(qkT)

            if HAS_KV_MASK:
                p = tl.where(k_valid[None, :], p, 0.0)

            # -- update m_i and l_i
            # Guard the all-(-inf) block case: if a selected block is fully padded
            # (every key masked) AND it is the first processed block, m_i and m_ij
            # are both -inf, so (m_i - m_ij) = -inf-(-inf) = NaN. Such a block
            # contributes nothing (p is all-zero via the kv-mask above), so alpha
            # must be 1.0 to leave acc / l_i untouched (alpha=0 would zero l_i and
            # produce 0/0 = NaN in the epilogue for an all-padding-selection query).
            # When any selected block has >=1 valid key (the common case) m_ij is
            # finite and this is numerically identical to the original.
            alpha = tl.where(m_ij == float("-inf"), 1.0, tl.math.exp2(m_i - m_ij))
            l_ij = tl.sum(p, 1)
            # -- update output accumulator --
            acc = acc * alpha[:, None]
            # update acc
            v = tl.load(V_block_ptr_i)
            acc = tl.dot(p.to(v.dtype), v, acc)
            # update m_i and l_i
            l_i = l_i * alpha + l_ij
            m_i = m_ij
            V_block_ptr_i = tl.advance(V_block_ptr_i, (BLOCK_N, 0))
            KT_block_ptr_i = tl.advance(KT_block_ptr_i, (0, BLOCK_N))
            mask_offset += BLOCK_N

    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))

configs_fwd_bsa_varlen_align_preset = {
    'default': {
        'num_stages': 3,
        'num_warps': 8,
    },
    'BLOCK_N_LG=64': {
        'num_stages': 3,
        'num_warps': 4,
    },
}
configs_fwd_bsa_varlen_align = [
    triton.Config({}, num_stages=s, num_warps=w) \
    for s in [2, 3, 4, 5] \
    for w in [4, 8] \
]

fwd_bsa_reevaluate_varlen_align_keys = ['N_CTX', 'BLOCK_M', 'BLOCK_N_LG', 'SPARSITY'] if os.environ.get('TRITON_REEVALUATE_KEY', '0') == '1' else []
@autotune(list(configs_fwd_bsa_varlen_align), key=fwd_bsa_reevaluate_varlen_align_keys)
@triton.jit
def _attn_fwd_bsa_varlen_align(
    Q, K, V, sm_scale, M, Out,
    block_indices, # [B, H, M_COMPRESS, S_MAX]
    block_indices_lens, # [B, H, M_COMPRESS]
    kv_valid_mask, # [N_CTX] bool
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_bz, stride_bh, stride_bm, stride_bs,
    stride_lz, stride_lh, stride_lm,
    H, N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N_LG: tl.constexpr,
    SPARSITY: tl.constexpr,
    HAS_KV_MASK: tl.constexpr,
    ):

    """Compute sparse attention with one aligned tile per selected key block."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
    b_offset = off_z.to(tl.int64) * stride_bz + off_h.to(tl.int64) * stride_bh
    l_offset = off_z.to(tl.int64) * stride_lz + off_h.to(tl.int64) * stride_lh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N_LG, HEAD_DIM),
        order=(1, 0),
    )
    KT_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N_LG),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    block_indices += b_offset + start_m * stride_bm
    block_indices_lens += l_offset + start_m * stride_lm
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/ln2
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    S = tl.load(block_indices_lens)
    for i in range(S):
        block_id = tl.load(block_indices + i * stride_bs).to(tl.int32)
        lo = block_id * BLOCK_N_LG
        lo = tl.multiple_of(lo, BLOCK_N_LG)
        KT_block_ptr_i = tl.advance(KT_block_ptr, (0, lo))
        V_block_ptr_i = tl.advance(V_block_ptr, (lo, 0))

        # -- compute qk ----
        kT = tl.load(KT_block_ptr_i)
        qkT = tl.dot(q, kT)

        if HAS_KV_MASK:
            offs_n = lo + tl.arange(0, BLOCK_N_LG)
            k_valid = tl.load(kv_valid_mask + offs_n)
            qkT = tl.where(k_valid[None, :], qkT, float('-inf'))

        m_ij = tl.maximum(m_i, tl.max(qkT, 1) * qk_scale)
        qkT = qkT * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qkT)

        if HAS_KV_MASK:
            p = tl.where(k_valid[None, :], p, 0.0)

        # -- update m_i and l_i
        # Guard the all-(-inf) block case: a fully-padded selected block processed
        # first leaves m_i == m_ij == -inf, so (m_i - m_ij) = NaN. Such a block
        # contributes nothing (p is all-zero via the kv-mask above); alpha must be
        # 1.0 to leave acc / l_i untouched (alpha=0 would zero l_i -> 0/0 = NaN in
        # the epilogue for an all-padding-selection query). Identical to the
        # original whenever any selected block has >=1 valid key (the common case).
        alpha = tl.where(m_ij == float("-inf"), 1.0, tl.math.exp2(m_i - m_ij))
        l_ij = tl.sum(p, 1)
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr_i)
        acc = tl.dot(p.to(v.dtype), v, acc)
        # update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij


    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))
