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


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv_bsa_varlen(
    dk, dv,
    k, v,
    Q, DO,
    M, D,
    block_indices,
    block_indices_lens,
    # shared by Q/K/V/DO.
    stride_qm, stride_qk,
    stride_dom, stride_dok,
    stride_mm,
    stride_dm,
    stride_bm,
    N_CTX,
    kv_valid_mask,
    k_start_pos,
    BLOCK_N_K: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HAS_KV_MASK: tl.constexpr,
    ):

    if HAS_KV_MASK:
        offs_k = k_start_pos + tl.arange(0, BLOCK_N_K)
        k_valid = tl.load(kv_valid_mask + offs_k)

    QT_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_qk, stride_qm),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_M1),
        order=(0, 1),
    )

    DO_block_ptr = tl.make_block_ptr(
        base=DO,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_dom, stride_dok),
        offsets=(0, 0),
        block_shape=(BLOCK_M1, HEAD_DIM),
        order=(1, 0),
    )

    S = tl.load(block_indices_lens)
    for i in range(S):
        block_id = tl.load(block_indices + i * stride_bm).to(tl.int32)
        start_m = block_id * BLOCK_M1
        start_m = tl.multiple_of(start_m, BLOCK_M1)

        QT_block_ptr_i = tl.advance(QT_block_ptr, (0, start_m))
        DO_block_ptr_i = tl.advance(DO_block_ptr, (start_m, 0))

        qT = tl.load(QT_block_ptr_i)
        offs_m = start_m + tl.arange(0, BLOCK_M1) * stride_mm
        m = tl.load(M + offs_m)
        kqT = tl.dot(k, qT)
        if HAS_KV_MASK:
            kqT = tl.where(k_valid[:, None], kqT, float('-inf'))
        pT = tl.math.exp2(kqT - m[None, :])
        if HAS_KV_MASK:
            pT = tl.where(k_valid[:, None], pT, 0.0)

        do = tl.load(DO_block_ptr_i)
        # Compute dV.
        ppT = pT
        ppT = ppT.to(v.dtype)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        offs_d = start_m + tl.arange(0, BLOCK_M1) * stride_dm
        Di = tl.load(D + offs_d)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(v.dtype)
        dk += tl.dot(dsT, tl.trans(qT))


    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq_bsa_varlen(
    dq,
    q, do,
    m, d,
    K, V,
    N_CTX,
    BLOCK_N2: tl.constexpr,
    BLOCK_N_LG: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    block_indices,
    block_indices_lens,
    stride_bn,
    stride_kn, stride_kk,
    stride_vn, stride_vk,
    kv_valid_mask,
    HAS_KV_MASK: tl.constexpr,
    ):

    VT_block_ptr = tl.make_block_ptr(
        base=V,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N2),
        order=(0, 1),
    )
    KT_block_ptr = tl.make_block_ptr(
        base=K,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N2),
        order=(0, 1),
    )

    S = tl.load(block_indices_lens)
    for i in range(S):
        block_id = tl.load(block_indices + i * stride_bn).to(tl.int32)
        lo, hi = block_id * BLOCK_N_LG, (block_id + 1) * BLOCK_N_LG
        lo = tl.multiple_of(lo, BLOCK_N2)

        KT_block_ptr_i = tl.advance(KT_block_ptr, (0, lo))
        VT_block_ptr_i = tl.advance(VT_block_ptr, (0, lo))
        mask_offset = lo

        for start_n in range(lo, hi, BLOCK_N2):
            start_n = tl.multiple_of(start_n, BLOCK_N2)

            kT = tl.load(KT_block_ptr_i)
            vT = tl.load(VT_block_ptr_i)
            qkT = tl.dot(q, kT)
            if HAS_KV_MASK:
                offs_n = mask_offset + tl.arange(0, BLOCK_N2)
                k_valid = tl.load(kv_valid_mask + offs_n)
                qkT = tl.where(k_valid[None, :], qkT, float('-inf'))
            p = tl.math.exp2(qkT - m)
            if HAS_KV_MASK:
                p = tl.where(k_valid[None, :], p, 0.0)
            # Compute dP and dS.
            dp = tl.dot(do, vT).to(tl.float32)
            ds = p * (dp - d)
            ds = ds.to(kT.dtype)
            # Compute dQ.
            dq += tl.dot(ds, tl.trans(kT))
            # Increment pointers.
            KT_block_ptr_i = tl.advance(KT_block_ptr_i, (0, BLOCK_N2))
            VT_block_ptr_i = tl.advance(VT_block_ptr_i, (0, BLOCK_N2))
            mask_offset += BLOCK_N2

    return dq

# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq_bsa_varlen_align(
    dq,
    q, do,
    m, d,
    K, V,
    N_CTX,
    BLOCK_N_LG: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    block_indices,
    block_indices_lens,
    stride_bn,
    stride_kn, stride_kk,
    stride_vn, stride_vk,
    kv_valid_mask,
    HAS_KV_MASK: tl.constexpr,
    ):

    VT_block_ptr = tl.make_block_ptr(
        base=V,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N_LG),
        order=(0, 1),
    )
    KT_block_ptr = tl.make_block_ptr(
        base=K,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N_LG),
        order=(0, 1),
    )

    S = tl.load(block_indices_lens)
    for i in range(S):
        block_id = tl.load(block_indices + i * stride_bn).to(tl.int32)
        lo = block_id * BLOCK_N_LG
        lo = tl.multiple_of(lo, BLOCK_N_LG)

        KT_block_ptr_i = tl.advance(KT_block_ptr, (0, lo))
        VT_block_ptr_i = tl.advance(VT_block_ptr, (0, lo))

        kT = tl.load(KT_block_ptr_i)
        vT = tl.load(VT_block_ptr_i)
        qkT = tl.dot(q, kT)
        if HAS_KV_MASK:
            offs_n = lo + tl.arange(0, BLOCK_N_LG)
            k_valid = tl.load(kv_valid_mask + offs_n)
            qkT = tl.where(k_valid[None, :], qkT, float('-inf'))
        p = tl.math.exp2(qkT - m)
        if HAS_KV_MASK:
            p = tl.where(k_valid[None, :], p, 0.0)
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - d)
        ds = ds.to(kT.dtype)
        # Compute dQ.
        dq += tl.dot(ds, tl.trans(kT))


    return dq

configs_bwd_dkdv_bsa_varlen_preset = {
    'default': {
        'BLOCK_N': 128,
        'num_stages': 2,
        'num_warps': 8,
    },
    'BLOCK_N_DQ_LG=64': {
        'BLOCK_N': 64,
        'num_stages': 2,
        'num_warps': 4,
    }
}
configs_bwd_dkdv_bsa_varlen = [
    triton.Config({'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BN in [32, 64, 128] \
    for s in [2, 3, 4, 5] \
    for w in [4, 8] \
]
bwd_dkdv_bsa_varlen_reevaluate_keys = ['N_CTX', 'BLOCK_M', 'BLOCK_N_DQ_LG', 'SPARSITY'] if os.environ.get('TRITON_REEVALUATE_KEY', '0') == '1' else []
@autotune(list(configs_bwd_dkdv_bsa_varlen), key=bwd_dkdv_bsa_varlen_reevaluate_keys)
@triton.jit
def _attn_bwd_dkdv_bsa_varlen_wrapper(
    Q, K, V, sm_scale,  # softmax scale
    DO,
    DK, DV,
    M, # lse (log2)
    D,
    block_indices,
    block_indices_lens,
    kv_valid_mask,
    # qkv
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    # dk dv do
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvn, stride_dvk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    # m, d
    stride_mz, stride_mh, stride_mm,
    stride_dz, stride_dh, stride_dm,
    #
    stride_bz, stride_bh, stride_bn, stride_bm, # block_indices
    stride_lz, stride_lh, stride_ln, # block_indices_lens
    #
    H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_N_DQ_LG: tl.constexpr, # logical block size
    HEAD_DIM: tl.constexpr,
    SPARSITY: tl.constexpr,
    HAS_KV_MASK: tl.constexpr,
    ):

    tl.static_assert(BLOCK_N_DQ_LG % BLOCK_N == 0)
    start_n = tl.program_id(0)

    off_hz = tl.program_id(2)
    off_z = off_hz // H
    off_h = off_hz % H

    off_q = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    off_k = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    off_v = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh

    off_dk = off_z.to(tl.int64) * stride_dkz + off_h.to(tl.int64) * stride_dkh
    off_dv = off_z.to(tl.int64) * stride_dvz + off_h.to(tl.int64) * stride_dvh
    off_do = off_z.to(tl.int64) * stride_doz + off_h.to(tl.int64) * stride_doh

    off_m = off_z.to(tl.int64) * stride_mz + off_h.to(tl.int64) * stride_mh
    off_d = off_z.to(tl.int64) * stride_dz + off_h.to(tl.int64) * stride_dh

    off_block_incides =      off_z.to(tl.int64) * stride_bz + off_h.to(tl.int64) * stride_bh
    off_block_incides_lens = off_z.to(tl.int64) * stride_lz + off_h.to(tl.int64) * stride_lh

    # offset pointers for batch/head
    Q += off_q
    K += off_k
    V += off_v
    DO += off_do
    DK += off_dk
    DV += off_dv

    M += off_m
    D += off_d
    block_indices += off_block_incides
    block_indices_lens += off_block_incides_lens

    # ---------------------------------------- [DKDV] ----------------------------------------

    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.

    K_block_ptr = tl.make_block_ptr(
        base=K,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_kn, stride_kk),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    DK_block_ptr = tl.make_block_ptr(
        base=DK,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_dkn, stride_dkk),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    DV_block_ptr = tl.make_block_ptr(
        base=DV,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_dvn, stride_dvk),
        offsets=(start_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )

    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)

    k_compress_idx = start_n * BLOCK_N // BLOCK_N_DQ_LG
    block_indices_i = block_indices + k_compress_idx * stride_bn
    block_indices_lens_i = block_indices_lens + k_compress_idx * stride_ln

    dk, dv = _attn_bwd_dkdv_bsa_varlen(
        dk, dv,
        k, v,
        Q, DO,
        M, D,
        block_indices_i,
        block_indices_lens_i,
        stride_qm, stride_qk,
        stride_dom, stride_dok,
        stride_mm,
        stride_dm,
        stride_bm,
        N_CTX,
        kv_valid_mask,
        start_n * BLOCK_N,
        BLOCK_N,
        BLOCK_M,
        HEAD_DIM,
        HAS_KV_MASK,
    )

    # Write back dk
    dk *= sm_scale # S = scale * QKT; dK = scale * QdST
    tl.store(DK_block_ptr, dk.to(k.dtype))

    # Write back dv
    tl.store(DV_block_ptr, dv.to(v.dtype))

configs_bwd_dq_bsa_varlen_preset = {
    'default': {
        'BLOCK_N_DQ': 64,
        'num_stages': 2,
        'num_warps': 8,
    },
    'BLOCK_N_DQ_LG=64': {
        'BLOCK_N_DQ': 64,
        'num_stages': 2,
        'num_warps': 4,
    },
}
configs_bwd_dq_bsa_varlen = [
    triton.Config({'BLOCK_N_DQ': BN}, num_stages=s, num_warps=w) \
    for BN in [32, 64, 128] \
    for s in [2, 3, 4, 5] \
    for w in [4, 8] \
]
bwd_dq_bsa_varlen_reevaluate_keys = ['N_CTX', 'BLOCK_M', 'BLOCK_N_DQ_LG', 'SPARSITY'] if os.environ.get('TRITON_REEVALUATE_KEY', '0') == '1' else []
@autotune(list(configs_bwd_dq_bsa_varlen), key=bwd_dq_bsa_varlen_reevaluate_keys)
@triton.jit
def _attn_bwd_dq_bsa_varlen_wrapper(
    Q, K, V,
    DO,
    DQ,
    M, # lse (log2)
    D,
    block_indices,
    block_indices_lens,
    kv_valid_mask,
    # qkv
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    # dq do
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    # m, d
    stride_mz, stride_mh, stride_mm,
    stride_dz, stride_dh, stride_dm,
    #
    stride_bz, stride_bh, stride_bm, stride_bn, # block_indices
    stride_lz, stride_lh, stride_lm, # block_indices_lens
    #
    H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N_DQ_LG: tl.constexpr, # logical block size
    BLOCK_N_DQ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SPARSITY: tl.constexpr,
    HAS_KV_MASK: tl.constexpr,
    ):

    tl.static_assert(BLOCK_N_DQ_LG % BLOCK_N_DQ == 0)
    tl.static_assert(BLOCK_N_DQ_LG % BLOCK_M == 0)

    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    start_m = tl.program_id(0)

    off_hz = tl.program_id(2)
    off_z = off_hz // H
    off_h = off_hz % H

    off_q = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    off_k = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    off_v = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh

    off_dq = off_z.to(tl.int64) * stride_dqz + off_h.to(tl.int64) * stride_dqh
    off_do = off_z.to(tl.int64) * stride_doz + off_h.to(tl.int64) * stride_doh

    off_m = off_z.to(tl.int64) * stride_mz + off_h.to(tl.int64) * stride_mh
    off_d = off_z.to(tl.int64) * stride_dz + off_h.to(tl.int64) * stride_dh

    off_block_incides =      off_z.to(tl.int64) * stride_bz + off_h.to(tl.int64) * stride_bh
    off_block_incides_lens = off_z.to(tl.int64) * stride_lz + off_h.to(tl.int64) * stride_lh

    # offset pointers for batch/head
    Q += off_q
    K += off_k
    V += off_v
    DO += off_do
    DQ += off_dq

    M += off_m
    D += off_d
    block_indices += off_block_incides
    block_indices_lens += off_block_incides_lens

    # ---------------------------------------- [DQ] ----------------------------------------

    Q_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_dom, stride_dok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    DQ_block_ptr = tl.make_block_ptr(
        base=DQ,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_dqm, stride_dqk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    q = tl.load(Q_block_ptr)
    do = tl.load(DO_block_ptr)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    start_m = start_m * BLOCK_M

    offs_m = start_m + tl.arange(0, BLOCK_M) * stride_mm
    offs_d = start_m + tl.arange(0, BLOCK_M) * stride_dm

    m = tl.load(M + offs_m)
    m = m[:, None]

    d = tl.load(D + offs_d)
    d = d[:, None]

    block_indices_m = block_indices + (start_m // BLOCK_M) * stride_bm
    block_indices_lens_m = block_indices_lens + (start_m // BLOCK_M) * stride_lm

    dq = _attn_bwd_dq_bsa_varlen(
        dq,
        q, do,
        m, d,
        K, V,
        N_CTX,
        BLOCK_N_DQ,
        BLOCK_N_DQ_LG,
        HEAD_DIM,
        block_indices_m,
        block_indices_lens_m,
        stride_bn,
        stride_kn, stride_kk,
        stride_vn, stride_vk,
        kv_valid_mask,
        HAS_KV_MASK,
    )

    # Write back dQ.
    dq *= LN2
    tl.store(DQ_block_ptr, dq.to(q.dtype))

configs_bwd_dq_bsa_varlen_align_preset = {
    'default': {
        'num_stages': 2,
        'num_warps': 8,
    },
    'BLOCK_N_DQ_LG=64': {
        'num_stages': 2,
        'num_warps': 4,
    },
}
configs_bwd_dq_bsa_varlen_align = [
    triton.Config({}, num_stages=s, num_warps=w) \
    for s in [2, 3, 4, 5] \
    for w in [4, 8] \
]
bwd_dq_bsa_varlen_align_reevaluate_keys = ['N_CTX', 'BLOCK_M', 'BLOCK_N_DQ_LG', 'SPARSITY'] if os.environ.get('TRITON_REEVALUATE_KEY', '0') == '1' else []
@autotune(list(configs_bwd_dq_bsa_varlen_align), key=bwd_dq_bsa_varlen_align_reevaluate_keys)
@triton.jit
def _attn_bwd_dq_bsa_varlen_align_wrapper(
    Q, K, V,
    DO,
    DQ,
    M, # lse (log2)
    D,
    block_indices,
    block_indices_lens,
    kv_valid_mask,
    # qkv
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    # dq do
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    # m, d
    stride_mz, stride_mh, stride_mm,
    stride_dz, stride_dh, stride_dm,
    #
    stride_bz, stride_bh, stride_bm, stride_bn, # block_indices
    stride_lz, stride_lh, stride_lm, # block_indices_lens
    #
    H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N_DQ_LG: tl.constexpr, # logical block size
    HEAD_DIM: tl.constexpr,
    SPARSITY: tl.constexpr,
    HAS_KV_MASK: tl.constexpr,
    ):

    tl.static_assert(BLOCK_N_DQ_LG % BLOCK_N_DQ_LG == 0)
    tl.static_assert(BLOCK_N_DQ_LG % BLOCK_M == 0)

    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    start_m = tl.program_id(0)

    off_hz = tl.program_id(2)
    off_z = off_hz // H
    off_h = off_hz % H

    off_q = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    off_k = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    off_v = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh

    off_dq = off_z.to(tl.int64) * stride_dqz + off_h.to(tl.int64) * stride_dqh
    off_do = off_z.to(tl.int64) * stride_doz + off_h.to(tl.int64) * stride_doh

    off_m = off_z.to(tl.int64) * stride_mz + off_h.to(tl.int64) * stride_mh
    off_d = off_z.to(tl.int64) * stride_dz + off_h.to(tl.int64) * stride_dh

    off_block_incides =      off_z.to(tl.int64) * stride_bz + off_h.to(tl.int64) * stride_bh
    off_block_incides_lens = off_z.to(tl.int64) * stride_lz + off_h.to(tl.int64) * stride_lh

    # offset pointers for batch/head
    Q += off_q
    K += off_k
    V += off_v
    DO += off_do
    DQ += off_dq

    M += off_m
    D += off_d
    block_indices += off_block_incides
    block_indices_lens += off_block_incides_lens

    # ---------------------------------------- [DQ] ----------------------------------------

    Q_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_dom, stride_dok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    DQ_block_ptr = tl.make_block_ptr(
        base=DQ,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_dqm, stride_dqk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    q = tl.load(Q_block_ptr)
    do = tl.load(DO_block_ptr)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    start_m = start_m * BLOCK_M

    offs_m = start_m + tl.arange(0, BLOCK_M) * stride_mm
    offs_d = start_m + tl.arange(0, BLOCK_M) * stride_dm

    m = tl.load(M + offs_m)
    m = m[:, None]

    # D (= delta) is pre-divided by ds_scale.
    d = tl.load(D + offs_d)
    d = d[:, None]

    block_indices_m = block_indices + (start_m // BLOCK_M) * stride_bm
    block_indices_lens_m = block_indices_lens + (start_m // BLOCK_M) * stride_lm

    dq = _attn_bwd_dq_bsa_varlen_align(
        dq,
        q, do,
        m, d,
        K, V,
        N_CTX,
        BLOCK_N_DQ_LG,
        HEAD_DIM,
        block_indices_m,
        block_indices_lens_m,
        stride_bn,
        stride_kn, stride_kk,
        stride_vn, stride_vk,
        kv_valid_mask,
        HAS_KV_MASK,
    )

    # Write back dQ.
    dq *= LN2
    tl.store(DQ_block_ptr, dq.to(q.dtype))
