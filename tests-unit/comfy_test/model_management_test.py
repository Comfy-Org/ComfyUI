from unittest.mock import Mock, call

import pytest

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


@pytest.fixture
def xpu_memory(monkeypatch):
    xpu = Mock()
    xpu.memory_stats.return_value = {
        "active_bytes.all.current": 0,
        "reserved_bytes.all.current": 0,
    }
    xpu.get_device_properties.return_value.total_memory = 34242297856
    xpu.mem_get_info.return_value = (6965571584, 34242297856)
    xpu.current_device.return_value = 0
    monkeypatch.setattr(model_management.torch, "xpu", xpu)
    monkeypatch.setattr(model_management, "directml_enabled", False)
    monkeypatch.setattr(model_management, "is_intel_xpu", lambda: True)
    monkeypatch.setattr(model_management, "xpu_memory_query_warnings", set())
    monkeypatch.setattr(model_management, "get_torch_device", lambda: model_management.torch.device("xpu:0"))
    return xpu


@pytest.mark.parametrize("driver_free,active,reserved", [
    (6965571584, 0, 0),
    (2 * 1024**3, 4 * 1024**3, 6 * 1024**3),
    (0, 4 * 1024**3, 6 * 1024**3),
])
def test_xpu_free_memory(xpu_memory, driver_free, active, reserved):
    device = model_management.torch.device("xpu:1")
    xpu_memory.mem_get_info.return_value = (driver_free, 34242297856)
    xpu_memory.memory_stats.return_value = {
        "active_bytes.all.current": active,
        "reserved_bytes.all.current": reserved,
    }
    reusable = reserved - active
    assert model_management.get_free_memory(device) == driver_free + reusable
    assert model_management.get_free_memory(device, torch_free_too=True) == (driver_free + reusable, reusable)
    assert xpu_memory.mem_get_info.call_args_list == [call(device), call(device)]
    xpu_memory.get_device_properties.assert_not_called()
    xpu_memory.synchronize.assert_not_called()
    xpu_memory.empty_cache.assert_not_called()


def test_xpu_free_memory_refreshes_default_device(xpu_memory):
    xpu_memory.mem_get_info.side_effect = [(100, 1000), (50, 1000)]
    assert model_management.get_free_memory() == 100
    assert model_management.get_free_memory() == 50
    assert xpu_memory.mem_get_info.call_args_list == [call(model_management.torch.device("xpu:0"))] * 2


def test_xpu_unsupported_memory_query(xpu_memory, caplog):
    xpu_memory.mem_get_info.side_effect = RuntimeError("The device (Intel GPU) doesn't support querying the available free memory.")
    xpu_memory.memory_stats.return_value = {"active_bytes.all.current": 100, "reserved_bytes.all.current": 300}
    for device in ("xpu", "xpu:0", 0, "xpu:1", 1):
        assert model_management.get_free_memory(device, torch_free_too=True) == (34242297756, 200)
    assert len(caplog.records) == 2
    assert all("--reserve-vram" in record.message for record in caplog.records)
    xpu_memory.mem_get_info.side_effect = None
    assert model_management.get_free_memory() == 6965571784


@pytest.mark.parametrize("error", [RuntimeError("device lost"), model_management.torch.OutOfMemoryError("out of memory")])
def test_xpu_memory_query_errors_propagate(xpu_memory, error, caplog):
    xpu_memory.mem_get_info.side_effect = error
    with pytest.raises(type(error), match=str(error)):
        model_management.get_free_memory()
    xpu_memory.get_device_properties.assert_not_called()
    assert not caplog.records


def test_xpu_external_memory_triggers_model_unload(xpu_memory, monkeypatch):
    device = model_management.torch.device("xpu:0")
    model = Mock(device=device)
    model.is_dead.return_value = False
    model.model_offloaded_memory.return_value = 0
    model.model_memory.return_value = 8 * 1024**3
    model.model_unload.return_value = True
    monkeypatch.setattr(model_management, "current_loaded_models", [model])
    monkeypatch.setattr(model_management, "DISABLE_SMART_MEMORY", False)
    monkeypatch.setattr(model_management, "cleanup_models_gc", Mock())
    monkeypatch.setattr(model_management, "soft_empty_cache", Mock())
    required = 8 * 1024**3
    assert model_management.free_memory(required, device) == [model]
    model.model_unload.assert_called_once_with(required - 6965571584)
    assert model_management.current_loaded_models == []
