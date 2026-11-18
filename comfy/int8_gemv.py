"""Emulated int8 linears for devices without int8 matmul (e.g. RDNA2 on ROCm).

The emulated path otherwise dequantizes the whole weight on every call (fp32 copy, ConvRot un-rotation as a
[N, K/256, 256] @ [256, 256] matmul, cast), which dominates LLM token decoding. Here the rotation is applied
to the activation instead (x @ ((q*s) @ H^T)^T = ((x @ H) @ q^T) * s) and a triton kernel reads the int8
weight directly, so a decode step only streams the int8 bytes.
"""
import logging
import torch
from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation

# Above this many rows (prefill, diffusion models) the matmul is compute bound and the dequant path is as fast.
MAX_ROWS = 16

_kernel = None
_disabled = False


def _get_kernel():
    global _kernel, _disabled
    if _kernel is None and not _disabled:
        try:
            import triton
            import triton.language as tl
            from triton.compiler.errors import CompilationError
            from triton.runtime.errors import OutOfResources
        except ImportError:
            _disabled = True
            return None

        @triton.jit
        def int8_gemv_kernel(x_ptr, q_ptr, s_ptr, o_ptr, N, K, stride_xm, stride_qn, stride_om,
                             BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
            pid_n = tl.program_id(0)
            m = tl.program_id(1).to(tl.int64)
            offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)  # offs_n * stride_qn overflows int32 for N * K > 2**31
            offs_k = tl.arange(0, BLOCK_K)
            mask_n = offs_n < N
            acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for k0 in range(0, K, BLOCK_K):
                k = k0 + offs_k
                mask_k = k < K
                q = tl.load(q_ptr + offs_n[:, None] * stride_qn + k[None, :], mask=mask_n[:, None] & mask_k[None, :], other=0).to(tl.float32)
                x = tl.load(x_ptr + m * stride_xm + k, mask=mask_k, other=0.0)
                acc += tl.sum(q * x[None, :], axis=1)
            s = tl.load(s_ptr + offs_n, mask=mask_n, other=0.0)
            tl.store(o_ptr + m * stride_om + offs_n, (acc * s).to(o_ptr.dtype.element_ty), mask=mask_n)

        _kernel = (triton, int8_gemv_kernel, (CompilationError, OutOfResources, RuntimeError))
    return _kernel


def rotate_activation(x, group_size):
    # fp32 in, fp32 out: autocast would run the rotation matmul in fp16
    with torch.autocast(x.device.type, enabled=False):
        return _rotate_activation(x, _build_hadamard(group_size, device=x.device, dtype=torch.float32), group_size)


def int8_linear(x, weight, bias=None):
    """linear(x, weight) for a TensorWiseINT8Layout QuantizedTensor without dequantizing it, or NotImplemented."""
    global _disabled
    if _disabled or not x.is_cuda or weight._layout_cls != "TensorWiseINT8Layout":
        return NotImplemented
    if torch.is_grad_enabled() and (x.requires_grad or (bias is not None and bias.requires_grad)):
        return NotImplemented  # the kernel has no backward
    params = weight._params
    if getattr(params, "transposed", False):
        return NotImplemented
    K = x.shape[-1]
    x2 = x.reshape(-1, K)
    q = weight._qdata
    if x2.shape[0] > MAX_ROWS or q.ndim != 2 or q.shape[1] != K or q.device != x.device:
        return NotImplemented
    kernel = _get_kernel()
    if kernel is None:
        return NotImplemented
    triton, int8_gemv_kernel, kernel_errors = kernel
    N = q.shape[0]
    scale = params.scale.to(device=x.device, dtype=torch.float32).reshape(-1)
    if scale.numel() not in (1, N):
        raise ValueError("int8 weight {}: {} scales, expected 1 or {} (one per output channel)".format(tuple(q.shape), scale.numel(), N))
    x2 = x2.float()
    if getattr(params, "convrot", False):
        x2 = rotate_activation(x2, params.convrot_groupsize)
    x2 = x2.contiguous()
    q = q.contiguous()
    if scale.numel() == 1:
        scale = scale.expand(N)
    scale = scale.contiguous()
    out = torch.empty((x2.shape[0], N), dtype=torch.float32, device=x.device)
    block_n, block_k = 8, 512
    try:
        int8_gemv_kernel[(triton.cdiv(N, block_n), x2.shape[0])](x2, q, scale, out, N, K, x2.stride(0), q.stride(0), out.stride(0),
                                                                   BLOCK_N=block_n, BLOCK_K=block_k, num_warps=4)
    except torch.OutOfMemoryError:
        raise
    except kernel_errors as e:
        # the kernel does not compile or launch on this device: disable it for the process
        logging.warning("int8 gemv kernel failed, falling back to dequantized weights: {}".format(e))
        _disabled = True
        return NotImplemented
    if bias is not None:
        out += bias.to(torch.float32)
    return out.to(x.dtype).view(*x.shape[:-1], N)


def int8_linear_chunked(x, weight, chunk_mb=256):
    """linear(x, weight) in fp32 row chunks of the int8 table: for huge tables (LM heads) when the kernel is not usable."""
    q = weight._qdata.to(x.device)
    params = weight._params
    x2 = x.reshape(-1, x.shape[-1]).float()
    if getattr(params, "convrot", False):
        x2 = rotate_activation(x2, params.convrot_groupsize)
    rows = max(1, chunk_mb * 1024 * 1024 // (q.shape[1] * 4))
    out = torch.empty((x2.shape[0], q.shape[0]), dtype=torch.float32, device=x.device)
    with torch.autocast(x.device.type, enabled=False):  # under fp16 autocast the products overflow before scaling
        for i in range(0, q.shape[0], rows):
            out[:, i:i + rows] = torch.mm(x2, q[i:i + rows].float().T)
    scale = params.scale.to(device=x.device, dtype=torch.float32)
    out *= scale.view(1, -1) if scale.numel() > 1 else scale
    return out.view(*x.shape[:-1], -1).to(x.dtype)
