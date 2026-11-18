import dataclasses

import pytest
import torch
import torch.nn.functional as F

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.int8_gemv  # noqa: E402
from comfy.quant_ops import QuantizedTensor  # noqa: E402

# scalar scale, per-row scales, per-row scales with ConvRot (comfy-kitchen only allows ConvRot per channel)
WEIGHT_KINDS = [
    {"per_channel": False},
    {"per_channel": True},
    {"per_channel": True, "convrot": True, "convrot_groupsize": 256},
]


def _weight(n, k, device="cpu", **kwargs):
    return QuantizedTensor.from_float(torch.randn(n, k), "TensorWiseINT8Layout", **kwargs).to(device)


@pytest.mark.parametrize("kwargs", WEIGHT_KINDS)
@pytest.mark.parametrize("rows", [1, 5])
def test_int8_linear_chunked_matches_dequantized(kwargs, rows):
    torch.manual_seed(0)
    weight = _weight(96, 512, **kwargs)
    x = torch.randn(rows, 3, 512)
    expected = F.linear(x, weight.dequantize().float())
    # chunk_mb=0 forces one row per chunk, so the chunk loop and the scale broadcast are both exercised
    for chunk_mb in (0, 256):
        out = comfy.int8_gemv.int8_linear_chunked(x, weight, chunk_mb=chunk_mb)
        assert out.shape == expected.shape
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"))])
def test_int8_linear_chunked_ignores_fp16_autocast(device):
    torch.manual_seed(0)
    weight = _weight(64, 512, device=device, per_channel=True, convrot=True, convrot_groupsize=256)
    x = torch.randn(1, 512, device=device) * 3000  # fp16 products of this overflow before the scale is applied
    expected = F.linear(x, weight.dequantize().float())
    with torch.autocast(device, dtype=torch.float16):
        out = comfy.int8_gemv.int8_linear_chunked(x, weight)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-1)


def test_int8_linear_declines_cpu_input():
    weight = _weight(32, 256, per_channel=True)
    assert comfy.int8_gemv.int8_linear(torch.randn(1, 256), weight) is NotImplemented


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU for the triton kernel")


@cuda
@pytest.mark.parametrize("kwargs", WEIGHT_KINDS)
def test_int8_linear_matches_dequantized(kwargs):
    pytest.importorskip("triton")
    torch.manual_seed(0)
    weight = _weight(1000, 512, device="cuda", **kwargs)
    x = torch.randn(2, 3, 512, device="cuda", dtype=torch.float16)
    bias = torch.randn(1000, device="cuda", dtype=torch.float16)
    out = comfy.int8_gemv.int8_linear(x, weight, bias)
    if out is NotImplemented:
        pytest.skip("int8 gemv kernel is not available on this device")
    expected = F.linear(x.float(), weight.dequantize().float(), bias.float())
    assert out.dtype == x.dtype and out.shape == expected.shape
    torch.testing.assert_close(out.float(), expected, rtol=2e-3, atol=2e-2)


@cuda
def test_int8_linear_rejects_bad_scale_count():
    pytest.importorskip("triton")
    weight = _weight(32, 256, device="cuda", per_channel=True)
    weight._params = dataclasses.replace(weight._params, scale=weight._params.scale[:7])
    if comfy.int8_gemv._get_kernel() is None or comfy.int8_gemv._disabled:
        pytest.skip("int8 gemv kernel is not available on this device")
    with pytest.raises(ValueError):
        comfy.int8_gemv.int8_linear(torch.randn(1, 256, device="cuda"), weight)
