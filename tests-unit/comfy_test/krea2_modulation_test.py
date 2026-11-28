import pytest
import torch
from torch import nn

from comfy.ldm.krea2.model import LastLayer, RMSNorm, SingleStreamBlock


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_dense_modulation_preserves_rounding(dtype):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        block = SingleStreamBlock(32, 2, 1, kvheads=1, dtype=dtype, operations=nn)
        with torch.no_grad():
            block.mod.lin.normal_(std=0.1)
            for module in block.modules():
                if isinstance(module, RMSNorm):
                    module.scale.normal_(std=0.1)
        x = torch.randn(1, 7, 32, dtype=dtype, requires_grad=True)
        vec = torch.randn(1, 1, 192, dtype=dtype, requires_grad=True)

    prescale, preshift, pregate, postscale, postshift, postgate = block.mod(vec)
    pre = (1 + prescale) * block.prenorm(x) + preshift
    residual = x + pregate * block.attn(pre)
    post = (1 + postscale) * block.postnorm(residual) + postshift
    expected = residual + postgate * block.mlp(post)

    actual = block(x, vec, None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    inputs_and_parameters = (x, vec, *block.parameters())
    actual_gradients = torch.autograd.grad(actual.sum(), inputs_and_parameters)
    expected_gradients = torch.autograd.grad(expected.sum(), inputs_and_parameters)
    for grad, reference in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(grad, reference, rtol=0, atol=0)


@pytest.mark.parametrize("shape,dtype", [((1, 1, 4), torch.float32), ((2, 1, 4), torch.float16)])
def test_modulation_hook_can_promote_or_broadcast_shift(shape, dtype):
    layer = LastLayer(4, 1, 4, dtype=torch.float16, operations=nn)
    layer.linear = nn.Identity()
    scale = torch.zeros(1, 1, 4, dtype=torch.float16)
    shift = torch.ones(shape, dtype=dtype)
    layer.modulation.register_forward_hook(lambda module, inputs, output: (scale, shift))

    with torch.no_grad():
        layer.norm.scale.zero_()
        x = torch.ones(1, 3, 4, dtype=torch.float16)
        torch.testing.assert_close(layer(x, scale), (1 + scale) * layer.norm(x) + shift, rtol=0, atol=0)
