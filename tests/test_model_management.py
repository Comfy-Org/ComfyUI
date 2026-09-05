from unittest.mock import Mock, call

import comfy.model_management as model_management


class NPUDevice:
    type = "npu"


class FakeStream:
    def __init__(self):
        self.waited_for = None

    def wait_stream(self, stream):
        self.waited_for = stream


def test_npu_current_stream(monkeypatch):
    device = NPUDevice()
    current_stream = object()
    npu = Mock()
    npu.current_stream.return_value = current_stream
    monkeypatch.setattr(model_management.torch, "npu", npu, raising=False)

    assert model_management.is_device_npu(device)
    assert model_management.current_stream(device) is current_stream
    npu.current_stream.assert_called_once_with(device)


def test_npu_offload_streams(monkeypatch):
    device = NPUDevice()
    current_stream = object()
    first_stream = FakeStream()
    second_stream = FakeStream()
    stream_context = object()
    npu = Mock()
    npu.current_stream.return_value = current_stream
    npu.Stream.side_effect = [first_stream, second_stream]
    npu.stream = stream_context

    monkeypatch.setattr(model_management.torch, "npu", npu, raising=False)
    monkeypatch.setattr(model_management.torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(model_management, "NUM_STREAMS", 2)
    monkeypatch.setattr(model_management, "STREAMS", {})
    monkeypatch.setattr(model_management, "stream_counters", {})

    assert model_management.get_offload_stream(device) is first_stream
    assert model_management.get_offload_stream(device) is second_stream
    assert model_management.get_offload_stream(device) is first_stream
    assert model_management.STREAMS[device] == [first_stream, second_stream]
    assert first_stream.as_context is stream_context
    assert second_stream.as_context is stream_context
    assert first_stream.waited_for is current_stream
    assert second_stream.waited_for is current_stream
    assert model_management.stream_counters[device] == 0
    assert npu.Stream.call_count == 2
    assert npu.Stream.call_args_list == [
        call(device=device, priority=0),
        call(device=device, priority=0),
    ]


def test_npu_offload_streams_disabled(monkeypatch):
    npu = Mock()
    monkeypatch.setattr(model_management.torch, "npu", npu, raising=False)
    monkeypatch.setattr(model_management, "NUM_STREAMS", 0)

    assert model_management.get_offload_stream(NPUDevice()) is None
    npu.Stream.assert_not_called()


def test_npu_synchronize(monkeypatch):
    npu = Mock()
    monkeypatch.setattr(model_management.torch, "npu", npu, raising=False)
    monkeypatch.setattr(model_management, "cpu_mode", lambda: False)
    monkeypatch.setattr(model_management, "is_intel_xpu", lambda: False)
    monkeypatch.setattr(model_management, "is_ascend_npu", lambda: True)

    model_management.synchronize()

    npu.synchronize.assert_called_once_with()
