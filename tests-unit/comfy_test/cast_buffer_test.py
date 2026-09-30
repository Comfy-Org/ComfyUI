from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import comfy.model_management as mm
import comfy.ops as ops
from comfy.quant_ops import QuantizedTensor, TensorCoreFP8Layout


class Stream(nullcontext):
    def wait_stream(self, stream):
        pass


@pytest.fixture
def buffers(monkeypatch):
    monkeypatch.setattr(mm, "STREAM_CAST_BUFFERS", {})
    monkeypatch.setattr(mm, "LARGEST_CASTED_WEIGHT", (None, 0))
    return Stream()


def test_live_cast_is_not_overwritten(buffers):
    first = mm.get_cast_buffer(buffers, "cpu", 128, object())
    first.fill_(1)
    second = mm.get_cast_buffer(buffers, "cpu", 128, object())
    second.fill_(2)
    assert torch.all(first == 1)
    assert first.data_ptr() != second.data_ptr()


@pytest.fixture
def simulated_cast(monkeypatch, buffers):
    get_buffer = mm.get_cast_buffer
    monkeypatch.setattr(mm, "get_offload_stream", lambda device: buffers)
    monkeypatch.setattr(mm, "current_stream", lambda device: buffers)
    monkeypatch.setattr(mm, "get_cast_buffer", lambda stream, device, size, ref: get_buffer(stream, "cpu", size, ref))
    monkeypatch.setattr(mm, "device_supports_non_blocking", lambda device: False)
    monkeypatch.setattr(ops.args, "cuda_malloc", False)

    def copy(weight, dtype=None, device=None, non_blocking=False, copy=False, stream=None, r=None):
        if r is None:
            r = torch.empty_like(weight, dtype=dtype, device="cpu")
        return r.copy_(weight)

    monkeypatch.setattr(mm, "cast_to", copy)
    return SimpleNamespace(weight=torch.arange(1, 9, dtype=torch.float32), bias=torch.ones(8),
                           weight_function=[], bias_function=[])


def test_training_preserves_saved_weight_after_forward(monkeypatch, simulated_cast):
    monkeypatch.setattr(mm, "in_training", True)
    x = torch.ones(8, requires_grad=True)
    weights = []
    outputs = []
    for offset in (0, 10):
        layer = SimpleNamespace(weight=simulated_cast.weight + offset, bias=None,
                                weight_function=[], bias_function=[])
        with ops.CastBiasWeightContext(layer, dtype=torch.float32, device=torch.device("cuda:0"), offloadable=True) as (weight, _):
            weights.append(weight)
            outputs.append(x * weight)
    sum(output.sum() for output in outputs).backward()
    torch.testing.assert_close(x.grad, simulated_cast.weight * 2 + 10, rtol=0, atol=0)
    assert weights[0].data_ptr() != weights[1].data_ptr()
    assert not mm.STREAM_CAST_BUFFERS


def test_training_does_not_consume_existing_inference_buffer(monkeypatch, buffers):
    first = mm.get_cast_buffer(buffers, "cpu", 128, object())
    mm.release_cast_buffer(buffers, first)
    with monkeypatch.context() as patch:
        patch.setattr(mm, "in_training", True)
        temporary = mm.get_cast_buffer(buffers, "cpu", 128, object())
        assert temporary.data_ptr() != first.data_ptr()
        mm.release_cast_buffer(buffers, temporary)
    assert mm.get_cast_buffer(buffers, "cpu", 128, object()).data_ptr() == first.data_ptr()


@pytest.mark.parametrize("stage", ["interpret", "weight_copy", "bias_copy", "weight_function", "bias_function"])
def test_failed_cast_releases_buffer(monkeypatch, simulated_cast, buffers, stage):
    def fail(*args, **kwargs):
        raise RuntimeError("injected cast failure")

    copy = mm.cast_to
    with monkeypatch.context() as patch:
        if stage == "interpret":
            patch.setattr(ops.comfy.memory_management, "interpret_gathered_like", fail)
        elif stage.endswith("copy"):
            def failing_copy(weight, *args, **kwargs):
                if weight is (simulated_cast.weight if stage == "weight_copy" else simulated_cast.bias):
                    fail()
                return copy(weight, *args, **kwargs)
            patch.setattr(mm, "cast_to", failing_copy)
        else:
            patch.setattr(simulated_cast, stage, [fail])
        with pytest.raises(RuntimeError, match="injected cast failure"):
            with ops.CastBiasWeightContext(simulated_cast, dtype=torch.float32, device=torch.device("cuda:0"), offloadable=True):
                pytest.fail("cast should not return")
    cached = mm.STREAM_CAST_BUFFERS[buffers]
    assert not cached.in_use
    with ops.CastBiasWeightContext(simulated_cast, dtype=torch.float32, device=torch.device("cuda:0"), offloadable=True) as (weight, bias):
        assert weight.untyped_storage().data_ptr() == cached.tensor.data_ptr()
        assert torch.equal(weight, simulated_cast.weight)
        assert torch.equal(bias, simulated_cast.bias)


@pytest.mark.parametrize("second_cast_fails", [False, True])
def test_overlapping_contexts_release_on_error(monkeypatch, simulated_cast, buffers, second_cast_fails):
    def fail(weight):
        raise RuntimeError("injected operation failure")

    other = SimpleNamespace(weight=simulated_cast.weight + 10, bias=None,
                            weight_function=[fail] if second_cast_fails else [], bias_function=[])
    with pytest.raises(RuntimeError, match="injected operation failure"):
        with ops.CastBiasWeightContext(simulated_cast, dtype=torch.float32, device=torch.device("cuda:0"), offloadable=True):
            with ops.CastBiasWeightContext(other, dtype=torch.float32, device=torch.device("cuda:0"), offloadable=True):
                fail(None)
    assert not mm.STREAM_CAST_BUFFERS[buffers].in_use


def test_temporary_release_does_not_release_cached_buffer(buffers):
    first = mm.get_cast_buffer(buffers, "cpu", 128, object())
    second = mm.get_cast_buffer(buffers, "cpu", 128, object())
    mm.release_cast_buffer(buffers, second)
    third = mm.get_cast_buffer(buffers, "cpu", 128, object())
    assert third.data_ptr() != first.data_ptr()
    mm.release_cast_buffer(buffers, first[16:32])
    reused = mm.get_cast_buffer(buffers, "cpu", 64, object())
    assert reused.data_ptr() == first.data_ptr()


def test_live_buffer_is_not_resized(buffers):
    first = mm.get_cast_buffer(buffers, "cpu", 64, object())
    larger = mm.get_cast_buffer(buffers, "cpu", 256, object())
    mm.release_cast_buffer(buffers, larger)
    mm.release_cast_buffer(buffers, first)
    assert mm.get_cast_buffer(buffers, "cpu", 64, object()).data_ptr() == first.data_ptr()


def test_streams_have_independent_buffers(buffers):
    first = mm.get_cast_buffer(buffers, "cpu", 128, object())
    second = mm.get_cast_buffer(Stream(), "cpu", 128, object())
    assert first.data_ptr() != second.data_ptr()


def test_unrelated_release_does_not_release_buffer(buffers):
    first = mm.get_cast_buffer(buffers, "cpu", 128, object())
    mm.release_cast_buffer(buffers, torch.empty(128, dtype=torch.int8))
    second = mm.get_cast_buffer(buffers, "cpu", 128, object())
    assert first.data_ptr() != second.data_ptr()


def test_released_buffer_can_grow(buffers):
    first = mm.get_cast_buffer(buffers, "cpu", 64, object())
    mm.release_cast_buffer(buffers, first)
    larger = mm.get_cast_buffer(buffers, "cpu", 256, object())
    assert larger.numel() >= 256
    mm.release_cast_buffer(buffers, larger)
    assert mm.get_cast_buffer(buffers, "cpu", 256, object()).data_ptr() == larger.data_ptr()


@pytest.mark.parametrize("bias_only", [False, True])
def test_uncast_orders_reuse_after_compute(monkeypatch, buffers, bias_only):
    first = mm.get_cast_buffer(buffers, "cpu", 128, object())
    compute_stream = object()
    waited = []

    def wait_stream(stream):
        # The cached buffer must remain occupied until the dependency is queued.
        other = mm.get_cast_buffer(buffers, "cpu", 128, object())
        assert other.data_ptr() != first.data_ptr()
        waited.append(stream)

    monkeypatch.setattr(buffers, "wait_stream", wait_stream)
    monkeypatch.setattr(mm, "current_stream", lambda device: compute_stream)
    weight, bias = (None, first) if bias_only else (first, None)
    ops.uncast_bias_weight(None, weight, bias, (buffers, weight, bias))
    assert waited == [compute_stream]
    assert mm.get_cast_buffer(buffers, "cpu", 128, object()).data_ptr() == first.data_ptr()


def test_quantized_cast_releases_buffer(monkeypatch, buffers):
    monkeypatch.setattr(mm, "current_stream", lambda device: buffers)
    params = TensorCoreFP8Layout.Params(scale=torch.tensor(1.0), orig_dtype=torch.float32, orig_shape=(128,))
    template = QuantizedTensor(torch.ones(128).to(torch.float8_e4m3fn), "TensorCoreFP8Layout", params)
    size = ops.comfy.memory_management.vram_aligned_size([template])
    buffer = mm.get_cast_buffer(buffers, "cpu", size, object())
    weight = ops.comfy.memory_management.interpret_gathered_like([template], buffer)[0]
    ops.uncast_bias_weight(None, weight, None, (buffers, weight, None))
    assert mm.get_cast_buffer(buffers, "cpu", size, object()).data_ptr() == buffer.data_ptr()


def test_resident_weight_with_offloaded_bias_releases_buffer(monkeypatch, buffers):
    monkeypatch.setattr(mm, "get_offload_stream", lambda device: buffers)
    monkeypatch.setattr(mm, "current_stream", lambda device: buffers)
    monkeypatch.setattr(ops.args, "cuda_malloc", False)
    cast_to = mm.cast_to

    def copy(weight, dtype=None, device=None, non_blocking=False, copy=False, stream=None, r=None):
        if weight.device.type == "meta":
            return r.fill_(3)
        return cast_to(weight, dtype, device, non_blocking, copy, stream, r)

    monkeypatch.setattr(mm, "cast_to", copy)
    layer = SimpleNamespace(weight=torch.ones(128), bias=torch.empty(128, device="meta"),
                            weight_function=[], bias_function=[])
    # Only the remote bias copy is simulated; the resident-weight fast path is real.
    first = ops.cast_bias_weight(layer, dtype=torch.float32, device=torch.device("cpu"), offloadable=True)
    assert first[0] is layer.weight
    assert torch.all(first[1] == 3)
    ops.uncast_bias_weight(layer, *first)
    second = ops.cast_bias_weight(layer, dtype=torch.float32, device=torch.device("cpu"), offloadable=True)
    assert second[0] is layer.weight
    assert second[1].data_ptr() == first[1].data_ptr()
    ops.uncast_bias_weight(layer, *second)


@pytest.mark.parametrize("bias", [False, True])
def test_cast_uncast_releases_buffer(monkeypatch, buffers, bias):
    get_buffer = mm.get_cast_buffer
    monkeypatch.setattr(mm, "get_offload_stream", lambda device: buffers)
    monkeypatch.setattr(mm, "current_stream", lambda device: buffers)
    monkeypatch.setattr(mm, "get_cast_buffer", lambda stream, device, size, ref: get_buffer(stream, "cpu", size, ref))
    monkeypatch.setattr(mm, "device_supports_non_blocking", lambda device: False)
    monkeypatch.setattr(ops.args, "cuda_malloc", False)

    def copy(weight, dtype=None, device=None, non_blocking=False, copy=False, stream=None, r=None):
        if r is None:
            r = torch.empty_like(weight, dtype=dtype, device="cpu")
        r.copy_(weight)
        return r

    monkeypatch.setattr(mm, "cast_to", copy)
    q = SimpleNamespace(weight=torch.ones(128), bias=torch.ones(128) if bias else None,
                        weight_function=[], bias_function=[])
    k = SimpleNamespace(weight=torch.full((128,), 2.0), bias=torch.full((128,), 3.0) if bias else None,
                        weight_function=[], bias_function=[])
    # Simulate device transfers with CPU tensors so this regression runs without an accelerator.
    first = ops.cast_bias_weight(q, dtype=torch.float32, device=torch.device("cuda:0"), offloadable=True)
    second = ops.cast_bias_weight(k, dtype=torch.float32, device=torch.device("cuda:0"), offloadable=True)
    assert torch.equal(first[0], q.weight)
    assert torch.equal(second[0], k.weight)
    if bias:
        assert torch.equal(first[1], q.bias)
        assert torch.equal(second[1], k.bias)
    ops.uncast_bias_weight(k, *second)
    ops.uncast_bias_weight(q, *first)
    third = ops.cast_bias_weight(q, dtype=torch.float32, device=torch.device("cuda:0"), offloadable=True)
    assert third[0].data_ptr() == first[0].data_ptr()
    ops.uncast_bias_weight(q, *third)
