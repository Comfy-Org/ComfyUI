import logging

import torch

_CK_STOCHASTIC_ROUNDING_AVAILABLE = False
try:
    import comfy_kitchen as ck
    _ck_stochastic_rounding_fp8 = ck.stochastic_rounding_fp8
    _CK_STOCHASTIC_ROUNDING_AVAILABLE = True
except (AttributeError, ImportError):
    logging.warning("comfy_kitchen does not support stochastic FP8 rounding, please update comfy_kitchen.")

if not _CK_STOCHASTIC_ROUNDING_AVAILABLE:
    def _ck_stochastic_rounding_fp8(value, rng, dtype):
        raise NotImplementedError("comfy_kitchen does not support stochastic FP8 rounding")


def calc_mantissa(abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS, generator=None):
    mantissa_scaled = torch.where(
        normal_mask,
        (abs_x / (2.0 ** (exponent - EXPONENT_BIAS)) - 1.0) * (2**MANTISSA_BITS),
        (abs_x / (2.0 ** (-EXPONENT_BIAS + 1 - MANTISSA_BITS)))
    )

    mantissa_scaled += torch.rand(mantissa_scaled.size(), dtype=mantissa_scaled.dtype, layout=mantissa_scaled.layout, device=mantissa_scaled.device, generator=generator)
    return mantissa_scaled.floor() / (2**MANTISSA_BITS)

#Not 100% sure about this
def manual_stochastic_round_to_float8(x, dtype, generator=None):
    if dtype == torch.float8_e4m3fn:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 4, 3, 7
    elif dtype == torch.float8_e5m2:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 5, 2, 15
    else:
        raise ValueError("Unsupported dtype")

    x = x.half()
    sign = torch.sign(x)
    abs_x = x.abs()
    sign = torch.where(abs_x == 0, 0, sign)

    # Combine exponent calculation and clamping
    exponent = torch.clamp(
        torch.floor(torch.log2(abs_x)) + EXPONENT_BIAS,
        0, 2**EXPONENT_BITS - 1
    )

    # Combine mantissa calculation and rounding
    normal_mask = ~(exponent == 0)

    abs_x[:] = calc_mantissa(abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS, generator=generator)

    sign *= torch.where(
        normal_mask,
        (2.0 ** (exponent - EXPONENT_BIAS)) * (1.0 + abs_x),
        (2.0 ** (-EXPONENT_BIAS + 1)) * abs_x
    )

    inf = torch.finfo(dtype)
    torch.clamp(sign, min=inf.min, max=inf.max, out=sign)
    return sign



def stochastic_rounding(value, dtype, seed=0):
    if dtype == torch.float32:
        return value.to(dtype=torch.float32)
    if dtype == torch.float16:
        return value.to(dtype=torch.float16)
    if dtype == torch.bfloat16:
        return value.to(dtype=torch.bfloat16)
    if dtype == torch.float8_e4m3fn or dtype == torch.float8_e5m2:
        generator = torch.Generator(device=value.device)
        generator.manual_seed(seed)
        if _CK_STOCHASTIC_ROUNDING_AVAILABLE:
            rng = torch.randint(0, 256, value.size(), dtype=torch.uint8, layout=value.layout, device=value.device, generator=generator)
            return _ck_stochastic_rounding_fp8(value, rng, dtype)

        output = torch.empty_like(value, dtype=dtype)
        num_slices = max(1, (value.numel() / (4096 * 4096)))
        slice_size = max(1, round(value.shape[0] / num_slices))
        for i in range(0, value.shape[0], slice_size):
            output[i:i+slice_size].copy_(manual_stochastic_round_to_float8(value[i:i+slice_size], dtype, generator=generator))
        return output

    return value.to(dtype=dtype)


# TODO: improve this?
def stochastic_float_to_fp4_e2m1(x, generator):
    orig_shape = x.shape
    sign = torch.signbit(x).to(torch.uint8)

    exp = torch.floor(torch.log2(x.abs()) + 1.0).clamp(0, 3)
    x += (torch.rand(x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=generator) - 0.5) * (2 ** (exp - 2.0)) * 1.25

    x = x.abs()
    exp = torch.floor(torch.log2(x) + 1.1925).clamp(0, 3)

    mantissa = torch.where(
        exp > 0,
        (x / (2.0 ** (exp - 1)) - 1.0) * 2.0,
        (x * 2.0),
        out=x
    ).round().to(torch.uint8)
    del x

    exp = exp.to(torch.uint8)

    fp4 = (sign << 3) | (exp << 1) | mantissa
    del sign, exp, mantissa

    fp4_flat = fp4.view(-1)
    packed = (fp4_flat[0::2] << 4) | fp4_flat[1::2]
    return packed.reshape(list(orig_shape)[:-1] + [-1])


def to_blocked(input_matrix, flatten: bool = True) -> torch.Tensor:
    """
    Rearrange a large matrix by breaking it into blocks and applying the rearrangement pattern.
    See:
        https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout

    Args:
        input_matrix: Input tensor of shape (H, W)
    Returns:
        Rearranged tensor of shape (32*ceil_div(H,128), 16*ceil_div(W,4))
    """

    def ceil_div(a, b):
        return (a + b - 1) // b

    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    # Calculate the padded shape
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = input_matrix
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols),
            device=input_matrix.device,
            dtype=input_matrix.dtype,
        )
        padded[:rows, :cols] = input_matrix

    # Rearrange the blocks
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    if flatten:
        return rearranged.flatten()

    return rearranged.reshape(padded_rows, padded_cols)


def stochastic_round_quantize_nvfp4_block(x, per_tensor_scale, generator):
    F4_E2M1_MAX = 6.0
    F8_E4M3_MAX = 448.0

    orig_shape = x.shape

    block_size = 16

    x = x.reshape(orig_shape[0], -1, block_size)
    scaled_block_scales_fp8 = torch.clamp(((torch.amax(torch.abs(x), dim=-1)) / F4_E2M1_MAX) / per_tensor_scale.to(x.dtype), max=F8_E4M3_MAX).to(torch.float8_e4m3fn)
    x = x / (per_tensor_scale.to(x.dtype) * scaled_block_scales_fp8.to(x.dtype)).unsqueeze(-1)

    x = x.view(orig_shape).nan_to_num()
    data_lp = stochastic_float_to_fp4_e2m1(x, generator=generator)
    return data_lp, scaled_block_scales_fp8


def stochastic_round_quantize_nvfp4(x, per_tensor_scale, pad_16x, seed=0):
    def roundup(x: int, multiple: int) -> int:
        """Round up x to the nearest multiple."""
        return ((x + multiple - 1) // multiple) * multiple

    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)

    # Handle padding
    if pad_16x:
        rows, cols = x.shape
        padded_rows = roundup(rows, 16)
        padded_cols = roundup(cols, 16)
        if padded_rows != rows or padded_cols != cols:
            x = torch.nn.functional.pad(x, (0, padded_cols - cols, 0, padded_rows - rows))

    x, blocked_scaled = stochastic_round_quantize_nvfp4_block(x, per_tensor_scale, generator)
    return x, to_blocked(blocked_scaled, flatten=False)


def stochastic_round_quantize_nvfp4_by_block(x, per_tensor_scale, pad_16x, seed=0, block_size=4096 * 4096):
    def roundup(x: int, multiple: int) -> int:
        """Round up x to the nearest multiple."""
        return ((x + multiple - 1) // multiple) * multiple

    orig_shape = x.shape

    # Handle padding
    if pad_16x:
        rows, cols = x.shape
        padded_rows = roundup(rows, 16)
        padded_cols = roundup(cols, 16)
        if padded_rows != rows or padded_cols != cols:
            x = torch.nn.functional.pad(x, (0, padded_cols - cols, 0, padded_rows - rows))
            # Note: We update orig_shape because the output tensor logic below assumes x.shape matches
            # what we want to produce. If we pad here, we want the padded output.
            orig_shape = x.shape

    orig_shape = list(orig_shape)

    output_fp4 = torch.empty(orig_shape[:-1] + [orig_shape[-1] // 2], dtype=torch.uint8, device=x.device)
    output_block = torch.empty(orig_shape[:-1] + [orig_shape[-1] // 16], dtype=torch.float8_e4m3fn, device=x.device)

    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)

    num_slices = max(1, (x.numel() / block_size))
    slice_size = max(1, (round(x.shape[0] / num_slices)))

    for i in range(0, x.shape[0], slice_size):
        fp4, block = stochastic_round_quantize_nvfp4_block(x[i: i + slice_size], per_tensor_scale, generator=generator)
        output_fp4[i:i + slice_size].copy_(fp4)
        output_block[i:i + slice_size].copy_(block)

    return output_fp4, to_blocked(output_block, flatten=False)


def stochastic_round_quantize_mxfp8_by_block(x, pad_32x, seed=0):
    def roundup(x_val, multiple):
        return ((x_val + multiple - 1) // multiple) * multiple

    if pad_32x:
        rows, cols = x.shape
        padded_rows = roundup(rows, 32)
        padded_cols = roundup(cols, 32)
        if padded_rows != rows or padded_cols != cols:
            x = torch.nn.functional.pad(x, (0, padded_cols - cols, 0, padded_rows - rows))

    F8_E4M3_MAX = 448.0
    E8M0_BIAS = 127
    BLOCK_SIZE = 32

    rows, cols = x.shape
    x_blocked = x.reshape(rows, -1, BLOCK_SIZE)
    max_abs = torch.amax(torch.abs(x_blocked), dim=-1)

    # E8M0 block scales (power-of-2 exponents)
    scale_needed = torch.clamp(max_abs.float() / F8_E4M3_MAX, min=2**(-127))
    exp_biased = torch.clamp(torch.ceil(torch.log2(scale_needed)).to(torch.int32) + E8M0_BIAS, 0, 254)
    block_scales_e8m0 = exp_biased.to(torch.uint8)

    zero_mask = (max_abs == 0)
    block_scales_f32 = (block_scales_e8m0.to(torch.int32) << 23).view(torch.float32)
    block_scales_f32 = torch.where(zero_mask, torch.ones_like(block_scales_f32), block_scales_f32)

    # Scale per-block then stochastic round
    data_scaled = (x_blocked.float() / block_scales_f32.unsqueeze(-1)).reshape(rows, cols)
    output_fp8 = stochastic_rounding(data_scaled, torch.float8_e4m3fn, seed=seed)

    block_scales_e8m0 = torch.where(zero_mask, torch.zeros_like(block_scales_e8m0), block_scales_e8m0)
    return output_fp8, to_blocked(block_scales_e8m0, flatten=False).view(torch.float8_e8m0fnu)
