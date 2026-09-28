from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

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
