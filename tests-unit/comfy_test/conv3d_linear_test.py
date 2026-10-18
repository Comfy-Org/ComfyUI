"""ops.Conv3d computes a kernel that covers the whole input as a linear; it must match F.conv3d."""

import pytest
import torch
import torch.nn.functional as F

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.ops  # noqa: E402


def make_conv(ops, cin, cout, kernel_size, **kwargs):
    conv = ops.Conv3d(cin, cout, kernel_size=kernel_size, dtype=torch.float64, **kwargs)
    conv.weight = torch.nn.Parameter(torch.randn(conv.weight.shape, dtype=torch.float64), requires_grad=False)
    conv.bias = torch.nn.Parameter(torch.randn(cout, dtype=torch.float64), requires_grad=False)
    return conv


@pytest.mark.parametrize("ops", [comfy.ops.disable_weight_init, comfy.ops.manual_cast])
@pytest.mark.parametrize("x_shape,kernel_size,kwargs,linear", [
    ((64, 3, 2, 16, 16), (2, 16, 16), {"stride": (2, 16, 16)}, True),  # Qwen VL pre-patched input
    ((64, 3, 2, 14, 14), (2, 14, 14), {"stride": (2, 14, 14)}, True),
    ((5, 4, 2, 3, 3), (2, 3, 3), {}, True),                           # stride 1, still one output position
    ((0, 3, 2, 16, 16), (2, 16, 16), {"stride": (2, 16, 16)}, True),   # empty batch
    ((1, 16, 3, 8, 10), (1, 2, 2), {"stride": (1, 2, 2)}, False),     # latent patch embed
    ((2, 4, 3, 3, 3), (3, 3, 3), {"padding": 1}, False),              # covers the input but padded
    ((2, 4, 4, 4, 4), (2, 2, 2), {"dilation": 2}, False),
    ((2, 4, 2, 3, 3), (2, 3, 3), {"groups": 2}, False),
])
def test_matches_conv3d(monkeypatch, ops, x_shape, kernel_size, kwargs, linear):
    torch.manual_seed(0)
    conv = make_conv(ops, x_shape[1], 8, kernel_size, **kwargs)
    x = torch.randn(x_shape, dtype=torch.float64)
    ref = F.conv3d(x, conv.weight, conv.bias, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups)
    if linear:
        monkeypatch.setattr(F, "conv3d", lambda *a, **k: pytest.fail("conv3d called for a kernel that covers the input"))
    out = conv(x)
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref)


def test_causal_zero_with_short_input(monkeypatch):
    torch.manual_seed(0)
    conv = make_conv(comfy.ops.disable_weight_init, 4, 8, (3, 2, 2))
    x = torch.randn(2, 4, 1, 2, 2, dtype=torch.float64)
    ref = F.conv3d(x, conv.weight[:, :, -1:], conv.bias)
    monkeypatch.setattr(F, "conv3d", lambda *a, **k: pytest.fail("conv3d called for a kernel that covers the input"))
    out = conv(x, autopad="causal_zero")
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref)
