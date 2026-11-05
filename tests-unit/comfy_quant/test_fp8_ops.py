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


@pytest.mark.parametrize("layout", ["2d", "3d", "strided_2d", "strided_3d", "slice"])
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
    else:
        x = backing.reshape(64, 64)[8:24]
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
