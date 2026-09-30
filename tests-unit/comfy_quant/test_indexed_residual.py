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
    segments = [(0, 4, 0), (4, 9, 1), (9, 13, torch.tensor([2, 0, 2, 1]))]
    rows = torch.tensor([0] * 4 + [1] * 5 + [2, 0, 2, 1])
    seen = []
    hook = layer.register_forward_hook(lambda m, a, y: seen.append(y))
    residual_before = residual.clone()
    with torch.no_grad():
        expected = torch.addcmul(residual, layer(ops._swiglu_eager(x)), gates[rows.long()].to(dtype))
        actual = ops.linear_input_act(layer, x, "swiglu", residual=residual,
                                      residual_scale=gates, residual_segments=segments)
    hook.remove()
    assert len(seen) == 2
    assert torch.equal(actual, expected)
    assert torch.equal(seen[0], seen[1])
    assert torch.equal(residual, residual_before)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("full_precision_mm", [False, True])
def test_segmented_fallback_reuses_linear_output(dtype, full_precision_mm):
    layer = torch.nn.Linear(256, 128, bias=False, dtype=dtype)
    layer._full_precision_mm = full_precision_mm
    x = torch.randn(13, 512, dtype=dtype)
    residual = torch.randn(13, 128, dtype=dtype)
    gates = torch.randn(3, 128)
    segments = [(0, 4, 0), (4, 9, 1), (9, 13, 2)]
    rows = torch.tensor([0] * 4 + [1] * 5 + [2] * 4)
    with torch.no_grad():
        branch = layer(ops._swiglu_eager(x))
        expected = torch.addcmul(residual, branch, gates[rows].to(dtype))
        with mock.patch.object(layer, "forward", return_value=branch):
            actual = ops.linear_input_act(layer, x, "swiglu", residual=residual,
                                          residual_scale=gates, residual_segments=segments)
    assert actual.data_ptr() == branch.data_ptr()
    assert torch.equal(actual, expected)


def test_training_keeps_eager_path():
    layer = torch.nn.Linear(256, 128, bias=False)
    x = torch.randn(3, 512, requires_grad=True)
    residual = torch.randn(3, 128, requires_grad=True)
    gate = torch.randn(2, 128, requires_grad=True)
    with mock.patch("comfy.model_management.in_training", True):
        result = ops.linear_input_act(layer, x, "swiglu", residual=residual,
                                      residual_scale=gate, residual_segments=[(0, 1, 0), (1, 2, 1), (2, 3, 0)])
    result.sum().backward()
    assert x.grad is not None and gate.grad is not None and residual.grad is not None


def test_int8_segments_preserve_row_mapping_and_uncast():
    operations = ops.mixed_precision_ops({}, compute_dtype=torch.bfloat16)
    layer = operations.Linear(256, 128, bias=False, device="cpu", dtype=torch.bfloat16)
    layer.weight = torch.nn.Parameter(ops.QuantizedTensor.from_float(
        torch.randn(128, 256, dtype=torch.bfloat16), "TensorWiseINT8Layout",
        convrot=True, per_channel=True), requires_grad=False)
    layer.quant_format = "int8_tensorwise"
    x = torch.randn(13, 512, dtype=torch.bfloat16)
    residual = torch.randn(13, 128, dtype=torch.bfloat16)
    gates = torch.randn(3, 128)
    segments = [(0, 4, 0), (4, 9, 1), (9, 13, torch.tensor([2, 0, 2, 1]))]
    rows = torch.tensor([0] * 4 + [1] * 5 + [2, 0, 2, 1], dtype=torch.int32)
    with torch.no_grad(), \
            mock.patch.object(ops.quant_ops.ck, "int8_linear_indexed_gate", wraps=ops.quant_ops.ck.int8_linear_indexed_gate) as fused, \
            mock.patch.object(ops, "uncast_bias_weight", wraps=ops.uncast_bias_weight) as uncast:
        actual = ops.linear_input_act(layer, x, "swiglu", residual=residual,
                                      residual_scale=gates, residual_segments=segments)
        fused.assert_called_once()
        uncast.assert_called_once()
        assert torch.equal(fused.call_args.args[4], rows)
        qdata, scale = ops.TensorWiseINT8Layout.get_plain_tensors(layer.weight)
        branch = ops.quant_ops.ck.int8_linear(x, qdata, scale, convrot=True, input_act="swiglu")
        expected = torch.addcmul(residual, branch, gates[rows.long()].to(actual.dtype))
    assert torch.equal(actual, expected)
