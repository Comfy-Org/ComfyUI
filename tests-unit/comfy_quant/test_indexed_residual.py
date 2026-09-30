import pytest
import torch
from unittest import mock
from comfy import ops


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_indexed_residual_full_precision_and_hooks(dtype):
    layer = torch.nn.Linear(256, 128, bias=False, dtype=dtype)
    x = torch.randn(13, 512, dtype=dtype)
    residual = torch.randn(13, 128, dtype=dtype)
    gates = torch.randn(3, 128, dtype=torch.float32)
    rows = torch.arange(13, dtype=torch.int32) % 3
    seen = []
    hook = layer.register_forward_hook(lambda m, a, y: seen.append(y.shape))
    with torch.no_grad():
        expected = torch.addcmul(residual, layer(ops._swiglu_eager(x)), gates[rows.long()].to(dtype))
        actual = ops.linear_input_act(layer, x, "swiglu", residual=residual,
                                      residual_scale=gates, residual_indices=rows)
    hook.remove()
    assert len(seen) == 2
    assert torch.equal(actual, expected)


def test_training_keeps_eager_path():
    layer = torch.nn.Linear(256, 128, bias=False)
    x = torch.randn(3, 512, requires_grad=True)
    residual = torch.randn(3, 128, requires_grad=True)
    gate = torch.randn(2, 128, requires_grad=True)
    rows = torch.tensor([0, 1, 0], dtype=torch.int32)
    with mock.patch("comfy.model_management.in_training", True):
        result = ops.linear_input_act(layer, x, "swiglu", residual=residual,
                                      residual_scale=gate, residual_indices=rows)
    result.sum().backward()
    assert x.grad is not None and gate.grad is not None and residual.grad is not None
