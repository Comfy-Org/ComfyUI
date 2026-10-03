import contextlib
import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args import args

args.cpu = True

from comfy import ops, weight_cache
W4A4WeightCache = pytest.importorskip(
    "comfy_kitchen.backends.ascend.weight_cache",
    reason="requires the companion Kitchen W4A4 cache API",
).W4A4WeightCache
from comfy_kitchen.tensor import QuantizedTensor
from comfy_kitchen.tensor.convrot_w4a4 import TensorCoreConvRotW4A4Layout


def test_budget_preserves_headroom():
    assert weight_cache.cache_budget(100, 1000, 200, 100) == 100
    assert weight_cache.cache_budget(900, 1000, 200, 100) == 500
    assert weight_cache.cache_budget(100, 200, 200, 100) == 0


@pytest.mark.parametrize("reason", ["disabled", "cpu", "dynamic", "lowvram", "patches", "hooks", "no_memory"])
def test_ineligible_sampling_does_not_bind(monkeypatch, reason):
    model = SimpleNamespace(model_lowvram=reason == "lowvram")
    patcher = SimpleNamespace(model=model, load_device=SimpleNamespace(type="cpu" if reason == "cpu" else "npu"),
                              is_dynamic=lambda: reason == "dynamic", patches={"x": 1} if reason == "patches" else {},
                              hook_patches={"x": 1} if reason == "hooks" else {},
                              get_free_memory=lambda device: 0)
    with weight_cache.sampling_weight_cache(patcher, 100, 0 if reason == "disabled" else 100) as cache:
        assert cache is None


def test_sampling_cleanup_on_error(monkeypatch):
    seen = []

    class Module:
        @contextlib.contextmanager
        def use_weight_cache(self, cache):
            seen.append(cache)
            try:
                yield
            finally:
                seen.append("restored")

    model = SimpleNamespace(model_lowvram=False, modules=lambda: [Module()])
    patcher = SimpleNamespace(model=model, load_device=SimpleNamespace(type="npu"),
                              is_dynamic=lambda: False, patches={}, hook_patches={}, get_free_memory=lambda device: 10000)
    monkeypatch.setattr(weight_cache.comfy.model_management, "minimum_inference_memory", lambda: 0)
    with pytest.raises(RuntimeError, match="interrupted"):
        with weight_cache.sampling_weight_cache(patcher, 100, 1000) as cache:
            cache._unpack(torch.zeros(4, 8, dtype=torch.int8))
            raise RuntimeError("interrupted")
    assert seen[-1] == "restored"
    assert seen[0].bytes_used == 0


@pytest.mark.parametrize("operations", [ops.manual_cast, ops.mixed_precision_ops({})])
def test_linear_scope_restores_dispatch_and_passes_cache(monkeypatch, operations):
    linear = operations.Linear(256, 32)
    qdata = torch.zeros(32, 128, dtype=torch.int8)
    params = TensorCoreConvRotW4A4Layout.Params(scale=torch.ones(32), orig_dtype=torch.float32, orig_shape=(32, 256))
    weight = QuantizedTensor(qdata, "TensorCoreConvRotW4A4Layout", params)
    linear.weight = torch.nn.Parameter(weight, requires_grad=False)
    input = torch.randn(2, 256)
    seen = []

    def implementation(*args, weight_cache):
        seen.append(weight_cache)
        return torch.zeros(2, 32)

    monkeypatch.setattr(ops, "convrot_w4a4_linear", implementation)
    with W4A4WeightCache(1024) as cache:
        with pytest.raises(RuntimeError):
            with linear.use_weight_cache(cache):
                assert linear._forward(input, weight, None).shape == (2, 32)
                plain = torch.randn(32, 256)
                assert torch.equal(linear._forward(input, plain, None), torch.nn.functional.linear(input, plain))
                raise RuntimeError
    assert seen == [cache]
    assert "_forward" not in linear.__dict__


@pytest.mark.parametrize("cast_weights", [True, False])
def test_manual_cast_forward_reaches_cache(monkeypatch, cast_weights):
    linear = ops.manual_cast.Linear(256, 32, bias=False)
    linear.comfy_cast_weights = cast_weights
    params = TensorCoreConvRotW4A4Layout.Params(scale=torch.ones(32), orig_dtype=torch.float32, orig_shape=(32, 256))
    weight = QuantizedTensor(torch.zeros(32, 128, dtype=torch.int8), "TensorCoreConvRotW4A4Layout", params)
    linear.weight = torch.nn.Parameter(weight, requires_grad=False)
    seen = []

    def implementation(*args, weight_cache):
        seen.append(weight_cache)
        return torch.zeros(2, 32)

    monkeypatch.setattr(ops, "convrot_w4a4_linear", implementation)
    with W4A4WeightCache(1024) as cache, linear.use_weight_cache(cache):
        assert linear(torch.randn(2, 256)).shape == (2, 32)
    assert seen == [cache]
    assert "_weight_cache_active" not in linear.__dict__


def test_compilation_preserves_original_linear(monkeypatch):
    linear = ops.manual_cast.Linear(256, 32, bias=False)
    params = TensorCoreConvRotW4A4Layout.Params(scale=torch.ones(32), orig_dtype=torch.float32, orig_shape=(32, 256))
    weight = QuantizedTensor(torch.zeros(32, 128, dtype=torch.int8), "TensorCoreConvRotW4A4Layout", params)
    linear.weight = torch.nn.Parameter(weight, requires_grad=False)
    input = torch.randn(2, 256)
    expected = linear._forward(input, weight, None)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    with W4A4WeightCache(1024) as cache, linear.use_weight_cache(cache):
        assert torch.equal(linear._forward(input, weight, None), expected)
        assert cache.hits == cache.misses == 0


def test_non_quantized_linear_is_not_wrapped():
    linear = ops.mixed_precision_ops({}).Linear(16, 16)
    linear.weight = torch.nn.Parameter(torch.randn(16, 16))
    with W4A4WeightCache(1024) as cache, linear.use_weight_cache(cache):
        assert "_forward" not in linear.__dict__


def test_scope_does_not_retain_cache_and_preserves_existing_override():
    linear = ops.mixed_precision_ops({}).Linear(256, 32)
    params = TensorCoreConvRotW4A4Layout.Params(scale=torch.ones(32), orig_dtype=torch.float32, orig_shape=(32, 256))
    linear.weight = QuantizedTensor(torch.zeros(32, 128, dtype=torch.int8), "TensorCoreConvRotW4A4Layout", params)
    original = lambda input, weight, bias: input
    linear._forward = original
    cache = W4A4WeightCache(1024)
    reference = weakref.ref(cache)
    with cache, linear.use_weight_cache(cache):
        assert linear._forward is original
        with linear.use_weight_cache(cache):
            assert linear._forward is original
    assert linear._forward is original
    del cache
    gc.collect()
    assert reference() is None
