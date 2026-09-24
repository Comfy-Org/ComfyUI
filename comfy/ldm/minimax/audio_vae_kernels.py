import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.hip import libdevice
except ImportError:
    triton = None


# Architectures with a successful compile for both kernels.
_SUPPORTED_HIP_ARCHES = {
    "gfx1010", "gfx1011", "gfx1012",
    "gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036",
    "gfx1100", "gfx1101", "gfx1102", "gfx1103",
    "gfx1150", "gfx1151", "gfx1152", "gfx1153",
    "gfx1200", "gfx1201",
    "gfx908", "gfx90a", "gfx942", "gfx950",
}


if triton is not None:
    @triton.jit
    def _fir2x(X, FILTER, Y, LENGTH: tl.constexpr, OUTPUT: tl.constexpr,
               CHANNELS: tl.constexpr, SB: tl.constexpr, SC: tl.constexpr, ST: tl.constexpr,
               UP: tl.constexpr, BLOCK: tl.constexpr):
        bc = tl.program_id(0)
        j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        base = (bc // CHANNELS) * SB + (bc % CHANNELS) * SC
        accum = tl.full((BLOCK,), 0.0, tl.float32)
        if UP:
            # Replicate-pad by 5, transpose-convolve, then crop 15 on each side.
            for tap in tl.static_range(6):
                index = tl.minimum(tl.maximum((j + 5) // 2 - tap, 0), LENGTH - 1)
                weight = tl.load(FILTER + ((j + 15) % 2) + 2 * tap)
                value = tl.load(X + base + index * ST, j < OUTPUT, 0)
                accum = accum + value * weight
            accum = accum * 2.0
        else:
            for tap in tl.static_range(12):
                index = tl.minimum(tl.maximum(2 * j + tap - 5, 0), LENGTH - 1)
                weight = tl.load(FILTER + tap)
                value = tl.load(X + base + index * ST, j < OUTPUT, 0)
                accum = accum + value * weight
        tl.store(Y + bc * OUTPUT + j, accum, j < OUTPUT)


    @triton.jit
    def _snake_beta_f32(X, ALPHA, BETA, Y, LENGTH: tl.constexpr, CHANNELS: tl.constexpr,
                       STRIDE_B: tl.constexpr, STRIDE_C: tl.constexpr, STRIDE_T: tl.constexpr,
                       BLOCK: tl.constexpr):
        bc = tl.program_id(0)
        offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        channel = bc % CHANNELS
        batch = bc // CHANNELS
        x = tl.load(X + batch * STRIDE_B + channel * STRIDE_C + offsets * STRIDE_T, offsets < LENGTH, 0)
        alpha = libdevice.exp(tl.load(ALPHA + channel))
        beta = libdevice.exp(tl.load(BETA + channel))
        sine = libdevice.sin(alpha * x)
        square = sine * sine
        reciprocal = tl.div_rn(1.0, beta + 1.0e-9)
        out = square * reciprocal + x
        tl.store(Y + bc * LENGTH + offsets, out, offsets < LENGTH)


def can_use(x, *parameters):
    if triton is None or torch.version.hip is None or not x.is_cuda or x.dtype != torch.float32 or x.ndim != 3 or x.numel() == 0:
        return False
    if torch.is_grad_enabled() and (x.requires_grad or any(parameter.requires_grad for parameter in parameters)):
        return False
    arch = torch.cuda.get_device_properties(x.device).gcnArchName.split(":")[0]
    return arch in _SUPPORTED_HIP_ARCHES


def fir2x(x, filter, up):
    length = x.shape[-1]
    output_length = 2 * length if up else (length + 1) // 2
    output = torch.empty((*x.shape[:2], output_length), dtype=x.dtype, device=x.device)
    _fir2x[(x.shape[0] * x.shape[1], triton.cdiv(output_length, 256))](
        x, filter.contiguous(), output, length, output_length, x.shape[1], *x.stride(), up, BLOCK=256,
        num_warps=4, enable_fp_fusion=False, allow_flush_denorm=False)
    return output


def snake_beta(x, alpha, beta):
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    _snake_beta_f32[(x.shape[0] * x.shape[1], triton.cdiv(x.shape[2], 256))](
        x, alpha.contiguous(), beta.contiguous(), output, x.shape[2], x.shape[1], *x.stride(), BLOCK=256,
        num_warps=4, enable_fp_fusion=False, allow_flush_denorm=False)
    return output
