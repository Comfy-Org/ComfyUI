import triton
import triton.language as tl
import os

if os.environ.get('TRITON_AUTOTUNE_ENBALE', '0') == '1':
    autotune = triton.autotune
else:
    def autotune(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

configs_gating_preset = {
    'default': {
        'BLOCK_M': 64,
        'BLOCK_N': 64,
        'num_stages': 3,
        'num_warps': 8,
    }
}

configs_gating = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64, 128] \
    for BN in [32, 64] \
    for s in [2, 3, 4, 5] \
    for w in [4, 8] \
]

gating_reevaluate_keys = ["M", "N"] if os.environ.get('TRITON_REEVALUATE_KEY', '0') == '1' else []
@autotune(configs_gating, key=gating_reevaluate_keys)
@triton.jit
def _attn_fwd_gating(
    Q, K, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_oz, stride_oh, stride_om, stride_on,
    H, M, N,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ):

    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(M, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_DIM, N),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(M, N),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr, boundary_check=(0,))
    for start_n in range(0, N, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr, boundary_check=(1,))
        qk = tl.dot(q, k)

        tl.store(O_block_ptr, qk.to(Out.type.element_ty), boundary_check=(0, 1))

        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        O_block_ptr = tl.advance(O_block_ptr, (0, BLOCK_N))


@triton.jit
def _attn_bwd_preprocess(
    O, DO,
    Delta, # output
    N_CTX,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr
    ):

    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta)
