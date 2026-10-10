import types
import weakref

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy.weight_adapter.bypass import BypassForwardHook
from comfy.weight_adapter.lora import LoRAAdapter


def make_adapter(dtype=torch.float32, alpha=3.0, mid=None):
    generator = torch.Generator().manual_seed(56)
    up = torch.randn(6, 2, generator=generator, dtype=dtype)
    down = torch.randn(2, 4, generator=generator, dtype=dtype)
    return LoRAAdapter(set(), (up, down, alpha, mid, None, None))


def reference_delta(adapter, x):
    up, down, alpha, mid, _, _ = adapter.weights
    hidden = F.linear(x, down.to(x.dtype))
    if mid is not None:
        hidden = F.linear(hidden, mid.to(x.dtype))
    scale = (1.0 if alpha is None else alpha / len(down)) * adapter.multiplier
    return F.linear(hidden, up.to(x.dtype)) * scale


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("strength", [-0.75, 0.0, 1.5])
def test_hook_matches_native_arithmetic_and_observes_strength(dtype, strength):
    torch.manual_seed(13)
    module = nn.Linear(4, 6, dtype=dtype)
    adapter = make_adapter(dtype)
    original = module.forward
    hook = BypassForwardHook(module, adapter, multiplier=strength)
    hook.inject()
    x = torch.randn(17, 4, dtype=dtype)
    before = x.clone()
    try:
        with torch.no_grad():
            for multiplier in (strength, -0.125):
                adapter.multiplier = multiplier
                expected = original(x) + reference_delta(adapter, x)
                actual = module(x)
                assert torch.equal(actual, expected)
                assert actual.dtype == expected.dtype and actual.stride() == expected.stride()
                assert torch.equal(x, before)
    finally:
        hook.eject()
    assert module.forward == original


class OutputStorage(TorchDispatchMode):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.pointers = set()

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = function(*args, **(kwargs or {}))
        if isinstance(result, torch.Tensor) and result.shape == self.shape:
            self.pointers.add(result.data_ptr())
        return result


def test_inference_uses_only_base_and_adapter_output_storage():
    module = nn.Linear(4, 6, bias=False)
    hook = BypassForwardHook(module, make_adapter())
    hook.inject()
    try:
        with torch.no_grad(), OutputStorage((17, 6)) as storage:
            result = module(torch.ones(17, 4))
        assert len(storage.pointers) == 2
        assert result.data_ptr() in storage.pointers
    finally:
        hook.eject()


def test_base_output_alias_is_preserved_and_released_before_g():
    adapter = make_adapter()
    adapter.multiplier = 0.5
    x = torch.randn(17, 4)
    base = torch.randn(17, 6)
    before = base.clone()
    base_ref = [None]
    calls = []

    def original(value, *, offset):
        assert value is x
        result = base + offset
        base_ref[0] = weakref.ref(result)
        return result

    def transform(value):
        assert base_ref[0]() is None
        calls.append(value)
        return value * 0.75

    adapter.g = transform
    with torch.no_grad():
        expected = (base + 2 + reference_delta(adapter, x)) * 0.75
        result = adapter.bypass_forward(original, x, offset=2)
    assert torch.equal(result, expected) and torch.equal(base, before)
    assert len(calls) == 1

    adapter = make_adapter()
    adapter.multiplier = 0.5
    x = torch.randn(17, 6)
    adapter.weights = (adapter.weights[0], torch.randn(2, 6), 3.0, None, None, None)
    before = x.clone()
    with torch.no_grad():
        expected = x + reference_delta(adapter, x)
        result = adapter.bypass_forward(lambda value: value, x)
    assert torch.equal(result, expected) and torch.equal(x, before)


def test_gradients_keep_out_of_place_path():
    adapter = make_adapter()
    adapter.multiplier = 0.75
    for tensor in adapter.weights[:2]:
        tensor.requires_grad_()
    x = torch.randn(17, 4, requires_grad=True)
    weight = torch.randn(6, 4, requires_grad=True)
    expected = F.linear(x, weight) + reference_delta(adapter, x)
    expected_gradients = torch.autograd.grad(expected.square().sum(), (x, weight, *adapter.weights[:2]))
    actual = adapter.bypass_forward(lambda value: F.linear(value, weight), x)
    actual_gradients = torch.autograd.grad(actual.square().sum(), (x, weight, *adapter.weights[:2]))
    assert torch.equal(actual, expected)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        assert torch.equal(actual_gradient, expected_gradient)


@pytest.mark.parametrize("alpha", [None, torch.tensor(3.0), torch.tensor([3.0]), torch.arange(6.0)])
def test_alpha_promotion_and_broadcast_are_preserved(alpha):
    adapter = make_adapter(torch.bfloat16, alpha)
    adapter.multiplier = 0.5
    x = torch.randn(17, 4, dtype=torch.bfloat16)
    base = torch.randn(17, 6, dtype=torch.bfloat16)
    with torch.no_grad():
        expected_delta = reference_delta(adapter, x)
        actual_delta = adapter.h(x, base)
        expected = base + expected_delta
        actual = adapter.bypass_forward(lambda value: base, x)
    assert torch.equal(actual_delta, expected_delta) and actual_delta.dtype == expected_delta.dtype
    assert torch.equal(actual, expected) and actual.dtype == expected.dtype


def test_custom_h_output_alias_is_not_overwritten():
    adapter = make_adapter()
    base = torch.randn(17, 6)
    custom = torch.randn(17, 6)
    before = custom.clone()
    adapter.h = types.MethodType(lambda self, x, base_out: custom, adapter)
    with torch.no_grad():
        result = adapter.bypass_forward(lambda value: base, torch.ones(17, 4))
    assert torch.equal(result, base + custom) and torch.equal(custom, before)


def test_noncontiguous_base_output_keeps_layout():
    adapter = make_adapter()
    adapter.multiplier = 0.5
    base = torch.randn(6, 17).t()
    x = torch.randn(17, 4)
    with torch.no_grad():
        expected = base + reference_delta(adapter, x)
        actual = adapter.bypass_forward(lambda value: base, x)
    assert torch.equal(actual, expected) and actual.stride() == expected.stride()


def test_tensor_subclass_preserves_dispatch():
    class Activation(torch.Tensor):
        pass

    adapter = make_adapter()
    adapter.multiplier = 0.5
    x = torch.randn(17, 4).as_subclass(Activation)
    base = torch.randn(17, 6).as_subclass(Activation)
    with torch.no_grad():
        expected = base + reference_delta(adapter, x)
        actual = adapter.bypass_forward(lambda value: base, x)
    assert type(actual) is type(expected) is Activation
    assert torch.equal(actual, expected)


def test_nested_hooks_and_mid_weights_preserve_order():
    module = nn.Linear(4, 6, bias=False)
    first = make_adapter(mid=torch.randn(2, 2))
    second = make_adapter()
    original = module.forward
    hooks = [BypassForwardHook(module, first, 0.25), BypassForwardHook(module, second, -0.5)]
    for hook in hooks:
        hook.inject()
    try:
        x = torch.randn(17, 4)
        with torch.no_grad():
            expected = (original(x) + reference_delta(first, x)) + reference_delta(second, x)
            assert torch.equal(module(x), expected)
    finally:
        for hook in reversed(hooks):
            hook.eject()


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_convolution_bypass_matches_reference(dimension):
    convolution = (nn.Conv1d, nn.Conv2d, nn.Conv3d)[dimension - 1]
    operation = (F.conv1d, F.conv2d, F.conv3d)[dimension - 1]
    module = convolution(4, 6, 3, padding=1, bias=False)
    up = torch.randn(6, 2)
    down = torch.randn(2, 4 * 3 ** dimension)
    adapter = LoRAAdapter(set(), (up, down, 3.0, None, None, None))
    original = module.forward
    hook = BypassForwardHook(module, adapter, multiplier=-0.5)
    hook.inject()
    try:
        x = torch.randn(1, 4, *((5,) * dimension))
        with torch.no_grad():
            hidden = operation(x, down.view(2, 4, *((3,) * dimension)), padding=1)
            delta = operation(hidden, up.view(6, 2, *((1,) * dimension))) * -0.75
            assert torch.equal(module(x), original(x) + delta)
    finally:
        hook.eject()
