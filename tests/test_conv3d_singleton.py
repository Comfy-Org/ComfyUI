import pytest
import torch
import torch.nn.functional as F

from comfy.ops import disable_weight_init, manual_cast


def require_hip_device():
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip("requires a ROCm GPU")


@pytest.mark.parametrize(("groups", "in_channels", "out_channels"), ((1, 2, 3), (2, 4, 6)))
@pytest.mark.skipif(torch.version.hip is None, reason="requires ROCm")
def test_singleton_causal_conv3d_matches_conv2d(groups, in_channels, out_channels, monkeypatch):
    require_hip_device()
    conv = disable_weight_init.Conv3d(
        in_channels, out_channels, (1, 3, 3), stride=(1, 2, 1),
        padding=(0, 2, 1), dilation=(1, 2, 1), groups=groups, device="cuda", dtype=torch.bfloat16,
    )
    torch.nn.init.normal_(conv.weight, std=0.1)
    torch.nn.init.normal_(conv.bias, std=0.1)
    x = torch.randn(1, in_channels, 1, 9, 11, device="cuda", dtype=torch.bfloat16)
    conv2d_calls = []
    conv2d = F.conv2d

    def record_conv2d(*args, **kwargs):
        conv2d_calls.append(True)
        return conv2d(*args, **kwargs)

    monkeypatch.setattr(F, "conv2d", record_conv2d)

    actual = conv(x, autopad="causal_zero")
    assert conv2d_calls
    expected = F.conv3d(x, conv.weight, conv.bias, (1, 2, 1), (0, 2, 1), (1, 2, 1), groups)

    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(torch.version.hip is None, reason="requires ROCm")
def test_singleton_conv2d_path_accepts_cast_and_patched_parameters(monkeypatch):
    require_hip_device()
    conv = manual_cast.Conv3d(2, 3, (1, 3, 3), padding=(0, 1, 1), device="cuda", dtype=torch.float32)
    torch.nn.init.normal_(conv.weight, std=0.1)
    torch.nn.init.normal_(conv.bias, std=0.1)
    conv.weight_function = [lambda weight: weight + 0.25]
    conv.bias_function = [lambda bias: bias - 0.5]
    x = torch.randn(1, 2, 1, 5, 7, device="cuda", dtype=torch.bfloat16)
    conv2d_calls = []
    conv2d = F.conv2d

    def record_conv2d(*args, **kwargs):
        conv2d_calls.append(True)
        return conv2d(*args, **kwargs)

    monkeypatch.setattr(F, "conv2d", record_conv2d)

    actual = conv(x, autopad="causal_zero")
    assert conv2d_calls
    weight = conv.weight.to(dtype=x.dtype) + 0.25
    bias = conv.bias.to(dtype=x.dtype) - 0.5
    expected = F.conv3d(x, weight, bias, padding=(0, 1, 1))

    torch.testing.assert_close(actual, expected)


def test_singleton_causal_conv3d_cpu_falls_back_and_keeps_gradients(monkeypatch):
    conv = disable_weight_init.Conv3d(2, 3, (1, 3, 3), padding=(0, 1, 1))
    conv.weight = torch.nn.Parameter(torch.randn_like(conv.weight))
    conv.bias = torch.nn.Parameter(torch.randn_like(conv.bias))
    x = torch.randn(1, 2, 1, 5, 7, requires_grad=True)

    conv2d_calls = []
    conv2d = F.conv2d

    def unexpected_fastpath(*args, **kwargs):
        conv2d_calls.append(True)
        return conv2d(*args, **kwargs)

    monkeypatch.setattr(F, "conv2d", unexpected_fastpath)
    output = conv(x, autopad="causal_zero")
    assert not conv2d_calls
    expected = F.conv3d(x, conv.weight, conv.bias, padding=(0, 1, 1))
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    output.sum().backward()

    assert x.grad is not None
    assert conv.weight.grad is not None
    assert conv.bias.grad is not None


@pytest.mark.skipif(torch.version.hip is None, reason="requires ROCm")
def test_singleton_causal_conv3d_preserves_gradients():
    require_hip_device()
    conv = disable_weight_init.Conv3d(
        2, 3, (1, 3, 3), padding=(0, 1, 1), device="cuda", dtype=torch.bfloat16,
    )
    torch.nn.init.normal_(conv.weight, std=0.1)
    torch.nn.init.normal_(conv.bias, std=0.1)
    x = torch.randn(1, 2, 1, 5, 7, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    reference_weight = conv.weight.detach().clone().requires_grad_()
    reference_bias = conv.bias.detach().clone().requires_grad_()
    grad = torch.randn(1, 3, 1, 5, 7, device="cuda", dtype=torch.bfloat16)

    actual = conv(x, autopad="causal_zero")
    expected = F.conv3d(reference_x, reference_weight, reference_bias, padding=(0, 1, 1))
    actual.backward(grad)
    expected.backward(grad)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(conv.weight.grad, reference_weight.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(conv.bias.grad, reference_bias.grad, rtol=1e-2, atol=1e-2)
