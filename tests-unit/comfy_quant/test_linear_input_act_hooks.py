"""The fused linear helper must preserve the module's adapter and hook behavior."""

import functools
import types

import pytest
import torch
import torch.nn.functional as F

import comfy.cli_args

if not torch.cuda.is_available():
    comfy.cli_args.args.cpu = True

from comfy import ops
from comfy.quant_ops import QuantizedTensor
from comfy.weight_adapter import BypassForwardHook, LoRAAdapter


def _layer(dtype=torch.float32, quantized=True):
    generator = torch.Generator().manual_seed(931)
    weight = torch.randn(8, 16, generator=generator, dtype=dtype) * 0.2
    layer = ops.mixed_precision_ops({}).Linear(16, 8, bias=False, device="cpu", dtype=dtype)
    if quantized:
        weight = QuantizedTensor.from_float(weight, "TensorWiseINT8Layout")
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    return layer


def _adapter(dtype=torch.float32, seed=932):
    generator = torch.Generator().manual_seed(seed)
    up = torch.randn(8, 3, generator=generator, dtype=dtype) * 0.3
    down = torch.randn(3, 16, generator=generator, dtype=dtype) * 0.3
    return LoRAAdapter(set(), (up, down, 3.0, None, None, None))


def _input(dtype=torch.float32):
    return torch.randn(2, 3, 32, generator=torch.Generator().manual_seed(933), dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_int8_input_activation_preserves_runtime_lora(dtype):
    layer, x = _layer(dtype), _input(dtype)
    adapter = _adapter(dtype)
    hook = BypassForwardHook(layer, adapter, multiplier=0.65)
    with torch.no_grad():
        base = layer(ops.INPUT_ACT_EAGER["swiglu"](x))
        hook.inject()
        try:
            expected = layer(ops.INPUT_ACT_EAGER["swiglu"](x))
            actual = ops.linear_input_act(layer, x, "swiglu")
            assert not torch.equal(expected, base)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finally:
            hook.eject()


@pytest.mark.parametrize("external_first", [False, True])
def test_runtime_lora_and_post_hook_match_materialized_weight(external_first):
    layer, x = _layer(quantized=False), _input()
    external, other = _adapter(), _adapter(seed=934)
    hook = BypassForwardHook(layer, external, multiplier=0.65)
    weight = layer.weight.detach().clone()

    def post_hook(_module, inputs, output):
        return output + 0.25 * F.linear(F.linear(inputs[0], other.weights[1]), other.weights[0])

    if external_first:
        hook.inject()
        handle = layer.register_forward_hook(post_hook)
    else:
        handle = layer.register_forward_hook(post_hook)
        hook.inject()
    try:
        materialized = (weight + 0.65 * (external.weights[0] @ external.weights[1])
                        + 0.25 * (other.weights[0] @ other.weights[1]))
        expected = F.linear(ops.INPUT_ACT_EAGER["swiglu"](x), materialized)
        actual = ops.linear_input_act(layer, x, "swiglu")
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(layer.weight, weight, rtol=0, atol=0)
    finally:
        handle.remove()
        hook.eject()


@pytest.mark.parametrize("kind", ["pre", "post", "global_pre", "global_post", "compiled"])
def test_int8_input_activation_preserves_module_call(kind):
    layer, x = _layer(), _input()
    handle = None
    if kind == "pre":
        handle = layer.register_forward_pre_hook(lambda _module, inputs: (inputs[0] * 0.5,))
    elif kind == "post":
        handle = layer.register_forward_hook(lambda _module, _inputs, output: output + 0.5)
    elif kind == "global_pre":
        handle = torch.nn.modules.module.register_module_forward_pre_hook(
            lambda _module, inputs: (inputs[0] * 0.5,))
    elif kind == "global_post":
        handle = torch.nn.modules.module.register_module_forward_hook(
            lambda _module, _inputs, output: output + 0.5)
    else:
        layer._compiled_call_impl = lambda inputs: layer.forward(inputs) + 0.5
    try:
        expected = layer(ops.INPUT_ACT_EAGER["swiglu"](x))
        actual = ops.linear_input_act(layer, x, "swiglu")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        if handle is not None:
            handle.remove()
        layer._compiled_call_impl = None


def test_unmodified_and_ejected_modules_keep_int8_fusion(monkeypatch):
    layer, x = _layer(), _input()
    original = ops.quant_ops.ck.int8_linear
    calls = []

    def fused(*args, **kwargs):
        calls.append(kwargs.get("input_act"))
        return original(*args, **kwargs)

    monkeypatch.setattr(ops.quant_ops.ck, "int8_linear", fused)
    with torch.no_grad():
        ops.linear_input_act(layer, x, "swiglu")
        assert calls == ["swiglu"]
        hook = BypassForwardHook(layer, _adapter(), multiplier=0.65)
        hook.inject()
        try:
            ops.linear_input_act(layer, x, "swiglu")
            # The module may still use an INT8 GEMM, but the helper must call it.
            assert calls.count("swiglu") == 1
        finally:
            hook.eject()
        ops.linear_input_act(layer, x, "swiglu")
        assert calls.count("swiglu") == 2


def _override_forward(layer, wrapped):
    base = type(layer)

    def forward(self, inputs, *args, **kwargs):
        return super(subclass, self).forward(inputs, *args, **kwargs) + 0.5

    if wrapped:
        forward = functools.wraps(base.forward)(forward)
    subclass = type("OverriddenLinear", (base,), {"forward": forward})
    layer.__class__ = subclass
    return layer


@pytest.mark.parametrize("wrapped", [False, True])
def test_int8_input_activation_preserves_subclass_forward(monkeypatch, wrapped):
    layer, x = _override_forward(_layer(), wrapped), _input()
    calls = []
    original = ops.quant_ops.ck.int8_linear

    def fused(*args, **kwargs):
        calls.append(kwargs.get("input_act"))
        return original(*args, **kwargs)

    monkeypatch.setattr(ops.quant_ops.ck, "int8_linear", fused)
    with torch.no_grad():
        expected = layer(ops.INPUT_ACT_EAGER["swiglu"](x))
        actual = ops.linear_input_act(layer, x, "swiglu")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # The module itself may still use an INT8 GEMM, but not the fused activation.
    assert "swiglu" not in calls


@pytest.mark.parametrize("wrapped", [False, True])
def test_fp16_shortcut_preserves_subclass_forward(monkeypatch, wrapped):
    layer = _override_forward(_layer(torch.float16, quantized=False), wrapped)
    x = _input(torch.float16)
    monkeypatch.setattr(ops, "_fp16_linear_wanted", lambda _x: True)
    monkeypatch.setattr(ops.quant_ops.ck, "fp16_linear", lambda *args, **kwargs: pytest.fail("fused fp16 call"))
    expected = layer(ops.INPUT_ACT_EAGER["swiglu"](x))
    actual = ops.linear_input_act(layer, x, "swiglu")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_subclass_inheriting_forward_keeps_int8_fusion(monkeypatch):
    layer, x = _layer(), _input()
    layer.__class__ = type("InheritedLinear", (type(layer),), {})
    calls = []
    original = ops.quant_ops.ck.int8_linear

    def fused(*args, **kwargs):
        calls.append(kwargs.get("input_act"))
        return original(*args, **kwargs)

    monkeypatch.setattr(ops.quant_ops.ck, "int8_linear", fused)
    with torch.no_grad():
        ops.linear_input_act(layer, x, "swiglu")
    assert calls == ["swiglu"]


def test_fp16_shortcut_preserves_custom_forward_and_residual(monkeypatch):
    layer, x = _layer(torch.float16, quantized=False), _input(torch.float16)
    original = layer.forward
    layer.forward = types.MethodType(lambda _self, inputs: original(inputs) + 0.5, layer)
    monkeypatch.setattr(ops, "_fp16_linear_wanted", lambda _x: True)

    def fused(inputs, weight, bias, **kwargs):
        output = F.linear(inputs, weight, bias)
        residual = kwargs.get("residual")
        return output if residual is None else torch.addcmul(residual, output, kwargs["residual_scale"])

    monkeypatch.setattr(ops.quant_ops.ck, "fp16_linear", fused)
    residual = torch.ones(2, 3, 8, dtype=x.dtype)
    scale = torch.full_like(residual, 0.25)
    expected = torch.addcmul(residual, layer(ops.INPUT_ACT_EAGER["swiglu"](x)), scale)
    actual = ops.linear_input_act(layer, x, "swiglu", residual=residual, residual_scale=scale)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("activation", [None, "gelu_tanh", "rms_norm"])
def test_hook_fallback_preserves_activation_parameters_and_residual(activation):
    layer = _layer()
    x = _input()[..., :16]
    norm_weight = torch.linspace(0.5, 1.5, 16)
    residual = torch.ones(2, 3, 8)
    scale = torch.full_like(residual, 0.25)
    hook = BypassForwardHook(layer, _adapter(), multiplier=0.65)
    hook.inject()
    try:
        expected = torch.addcmul(
            residual, layer(ops._eager_input_act(x, activation, norm_weight, 1e-3)), scale)
        actual = ops.linear_input_act(
            layer, x, activation, norm_weight, 1e-3, residual=residual, residual_scale=scale)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        hook.eject()



def test_int8_input_activation_preserves_pre_quant_scale():
    # MixedPrecisionOps.Linear applies AWQ pre-scaling in forward; the fused
    # weight-only helper has no equivalent input transform.
    layer, x = _layer(), _input()
    layer.pre_quant_scale = torch.linspace(0.6, 1.4, 16)
    with torch.no_grad():
        expected = layer(ops.INPUT_ACT_EAGER["swiglu"](x))
        actual = ops.linear_input_act(layer, x, "swiglu")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
