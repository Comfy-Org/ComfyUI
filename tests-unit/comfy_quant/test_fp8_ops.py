import pytest
import torch
import torch.nn.functional as F

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy import model_management, ops
from comfy.quant_ops import QuantizedTensor


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda":
        if not torch.cuda.is_available() or not model_management.supports_fp8_compute(torch.device("cuda")):
            pytest.skip("FP8 compute is not supported on this device")
    return torch.device(request.param)


def make_linear(device, weight_dtype=torch.float8_e4m3fn, bias=True):
    layer = ops.fp8_ops.Linear(64, 32, bias=bias, device=device, dtype=weight_dtype)
    layer.weight = torch.nn.Parameter(torch.eye(32, 64, device=device).to(weight_dtype), requires_grad=False)
    layer.bias = torch.nn.Parameter(torch.full((32,), 0.25, device=device, dtype=torch.bfloat16), requires_grad=False) if bias else None
    return layer


@pytest.mark.parametrize("layout", ["2d", "3d", "strided_2d", "strided_3d", "slice", "offset_strided"])
def test_fp8_linear_preserves_input_and_shared_storage(device, layout):
    backing = torch.linspace(-2096, 2096, 32 * 128, device=device).to(torch.bfloat16).reshape(32, 128)
    if layout == "2d":
        x = backing.reshape(64, 64)
    elif layout == "3d":
        x = backing.reshape(2, 32, 64)
    elif layout == "strided_2d":
        x = backing[:, ::2]
    elif layout == "strided_3d":
        x = backing.reshape(2, 16, 128)[:, :, ::2]
    elif layout == "slice":
        x = backing.reshape(64, 64)[8:24]
    else:
        x = backing.reshape(-1)[1:2049].reshape(16, 128)[:, ::2]
    before = backing.clone()
    layer = make_linear(device)

    output = ops.fp8_linear(layer, x)

    assert output is not None
    assert output.shape == (*x.shape[:-1], 32)
    assert output.dtype == x.dtype
    assert torch.isfinite(output).all()
    assert torch.equal(backing, before)


@pytest.mark.parametrize("dtype,bias", [
    (torch.float16, True),
    (torch.bfloat16, True),
    (torch.float32, False),  # CUDA scaled matmul does not support bias with FP32 output.
])
@pytest.mark.parametrize("values", [
    [0.0] * 8,
    [0.0, 2**-9, 2**-6, 0.5, 1.5, 7.5, 448.0, -448.0],
], ids=["zero", "small_and_boundary"])
def test_fp8_linear_preserves_representable_values_and_bias(device, dtype, bias, values):
    x = torch.tensor(values, device=device, dtype=dtype).repeat(16, 8)
    layer = make_linear(device, bias=bias)
    expected = F.linear(x, layer.weight.to(dtype), layer.bias.to(dtype) if bias else None)

    output = ops.fp8_linear(layer, x)

    assert output is not None
    assert output.dtype == dtype
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")], ids=["nan", "inf", "negative_inf"])
def test_fp8_linear_nonfinite_input_preserves_finite_rows(device, nonfinite):
    x = torch.ones(16, 64, device=device, dtype=torch.bfloat16)
    x[0, 0] = nonfinite
    before = x.clone()
    layer = make_linear(device)
    expected = F.linear(x[1:], layer.weight.to(x.dtype), layer.bias)

    output = ops.fp8_linear(layer, x)

    assert output is not None
    assert output.shape == (16, 32)
    assert output.dtype == x.dtype
    torch.testing.assert_close(output[1:], expected, rtol=0, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0, equal_nan=True)


def test_fp8_linear_reduces_error_for_large_activations(device):
    x = torch.tensor([2096.0, -2096.0, 1024.0, -1024.0, 448.0, -448.0, 0.5, -0.5],
                     device=device, dtype=torch.bfloat16).repeat(16, 8)
    layer = make_linear(device, bias=False)
    weight = layer.weight.to(x.dtype)
    expected = F.linear(x, weight)
    # Every clipped value is exactly representable in E4M3, isolating saturation error.
    clipped = F.linear(x.clamp(-448, 448), weight)

    output = ops.fp8_linear(layer, x)

    assert output is not None
    assert torch.isfinite(output).all()
    error = (output.float() - expected.float()).square().mean()
    clipping_error = (clipped.float() - expected.float()).square().mean()
    assert error < clipping_error * 0.01


def test_fp8_linear_uses_effective_weight_dtype(device):
    layer = make_linear(device, weight_dtype=torch.bfloat16)
    x = torch.ones(16, 64, device=device, dtype=torch.bfloat16)
    assert ops.fp8_linear(layer, x) is None
    layer.weight._model_dtype = torch.float8_e4m3fn

    output = ops.fp8_linear(layer, x)

    assert output is not None
    assert layer.weight.dtype == torch.bfloat16
    torch.testing.assert_close(output, F.linear(x, layer.weight, layer.bias), rtol=0, atol=0)


def test_fp8_linear_unsupported_rank_falls_back_without_changing_input():
    layer = make_linear(torch.device("cpu"))
    x = torch.arange(2 * 2 * 4 * 64, dtype=torch.bfloat16).reshape(2, 2, 4, 64)
    before = x.clone()
    expected = F.linear(x, layer.weight.to(x.dtype), layer.bias)

    assert ops.fp8_linear(layer, x) is None
    torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)
    assert torch.equal(x, before)


@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float8_e5m2])
def test_fp8_linear_unsupported_dtype_falls_back(weight_dtype):
    layer = make_linear(torch.device("cpu"), weight_dtype=weight_dtype)
    x = torch.full((16, 64), 2096.0, dtype=torch.bfloat16)
    expected = F.linear(x, layer.weight.to(x.dtype), layer.bias)

    assert ops.fp8_linear(layer, x) is None
    torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)
    assert torch.equal(x, torch.full_like(x, 2096.0))


@pytest.mark.skipif(not torch.cuda.is_available() or not model_management.is_nvidia(), reason="Requires NVIDIA CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["misaligned_2d", "misaligned_3d", "partial_vector", "strided_partial_vector"])
def test_fp8_linear_unsafe_cuda_input_falls_back_before_quantization(dtype, layout, monkeypatch):
    if layout.startswith("misaligned"):
        backing = torch.linspace(-2096, 2096, 16 * 64 + 1, device="cuda").to(dtype)
        x = backing[1:].reshape(16, 64)
        if layout == "misaligned_3d":
            x = x.reshape(2, 8, 64)
        assert x.is_contiguous()
        assert x.data_ptr() % 16 != 0
    else:
        columns = 126 if layout == "strided_partial_vector" else 63
        backing = torch.linspace(-2096, 2096, 17 * columns, device="cuda").to(dtype)
        x = backing.reshape(17, columns)
        if layout == "strided_partial_vector":
            x = x[:, ::2]
        assert x.numel() % 16 != 0
    before = backing.clone()
    layer = ops.fp8_ops.Linear(x.shape[-1], 32, bias=False, device=x.device, dtype=torch.float8_e4m3fn)
    layer.weight = torch.nn.Parameter(torch.eye(32, x.shape[-1], device=x.device).to(torch.float8_e4m3fn), requires_grad=False)
    expected = F.linear(x, layer.weight.to(dtype))

    def reject_quantization(*args, **kwargs):
        pytest.fail("Unsafe CUDA input reached the FP8 quantizer")

    monkeypatch.setattr(ops.TensorCoreFP8Layout, "quantize", reject_quantization)

    assert ops.fp8_linear(layer, x) is None
    output = layer(x)

    assert output.dtype == dtype
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert torch.equal(backing, before)


def test_fp8_linear_cpu_partial_vector_still_uses_fp8():
    x = torch.ones(3, 5, dtype=torch.bfloat16)
    layer = ops.fp8_ops.Linear(5, 4, bias=False, device="cpu", dtype=torch.float8_e4m3fn)
    layer.weight = torch.nn.Parameter(torch.eye(4, 5).to(torch.float8_e4m3fn), requires_grad=False)

    output = ops.fp8_linear(layer, x)

    assert output is not None
    torch.testing.assert_close(output, F.linear(x, layer.weight.to(x.dtype)), rtol=0, atol=0)


def test_fp8_failure_falls_back_with_original_input(device, monkeypatch):
    layer = make_linear(device)
    x = torch.full((16, 64), 2096.0, device=device, dtype=torch.bfloat16)
    before = x.clone()
    expected = F.linear(x, layer.weight.to(x.dtype), layer.bias)
    linear = F.linear
    failed_calls = []

    def fail_quantized_linear(input, weight, bias=None):
        if isinstance(input, QuantizedTensor):
            failed_calls.append(True)
            raise RuntimeError("injected FP8 backend failure")
        return linear(input, weight, bias)

    monkeypatch.setattr(F, "linear", fail_quantized_linear)
    output = layer(x)

    assert failed_calls == [True]
    assert torch.equal(x, before)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
