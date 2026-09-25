import sys
import types
from unittest.mock import Mock, call

if "comfy_aimdo.storage" not in sys.modules:
    sys.modules["comfy_aimdo.storage"] = types.ModuleType("comfy_aimdo.storage")
import torch
if not torch.cuda.is_available():
    import comfy.cli_args
    comfy.cli_args.args.cpu = True

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


def test_free_memory_dynamic_model_partial_unload(monkeypatch):
    import weakref
    import torch
    device = torch.device("cuda:0")
    mock_model = Mock()
    mock_model.is_dynamic.return_value = True
    mock_model.loaded_size.return_value = 10_000_000_000
    mock_model.model_size.return_value = 10_000_000_000
    mock_model.offload_device = torch.device("cpu")
    mock_model.load_device = device
    mock_model.parent = None
    mock_model.current_loaded_device.return_value = device
    mock_model.partially_unload.return_value = 1_000_000_000
    real_inner = Mock()
    mock_model.model = real_inner

    loaded_m = model_management.LoadedModel(mock_model)
    loaded_m.device = device
    loaded_m.real_model = weakref.ref(real_inner)
    loaded_m.model_finalizer = Mock()

    monkeypatch.setattr(model_management, "current_loaded_models", [loaded_m])
    monkeypatch.setattr(model_management, "get_free_memory", lambda dev=None, torch_free_too=False: 500_000_000 if not torch_free_too else (500_000_000, 0))
    soft_empty_called = False
    def mock_soft_empty(*args, **kwargs):
        nonlocal soft_empty_called
        soft_empty_called = True
    monkeypatch.setattr(model_management, "soft_empty_cache", mock_soft_empty)

    unloaded = model_management.free_memory(1_500_000_000, device, for_dynamic=True)
    assert len(unloaded) == 0
    assert len(model_management.current_loaded_models) == 1
    mock_model.partially_unload.assert_called_once_with(torch.device("cpu"), 1_000_000_000)
    assert soft_empty_called


def test_load_models_gpu_dynamic_reentry_headroom(monkeypatch):
    import weakref
    import torch
    device = torch.device("cuda:0")
    mock_model = Mock()
    mock_model.is_dynamic.return_value = True
    mock_model.loaded_size.return_value = 15_000_000_000
    mock_model.model_size.return_value = 15_000_000_000
    mock_model.load_device = device
    mock_model.offload_device = torch.device("cpu")
    mock_model.current_loaded_device.return_value = device
    mock_model.model_patches_models.return_value = []
    mock_model.model_dtype.return_value = torch.float32
    mock_model.partially_load.return_value = None
    mock_model.partially_unload.return_value = 1_500_000_000
    real_inner = Mock()
    mock_model.model = real_inner
    real_inner.dynamic_pins = {device: {"weights": (Mock(size=0),), "weights-loaded": (Mock(size=0),), "weights-fast": (Mock(size=0),)}}
    mock_model.loaded_ram_size.return_value = 0
    mock_model.parent = None
    mock_model.is_clone.return_value = True

    loaded_m = model_management.LoadedModel(mock_model)
    loaded_m.device = device
    loaded_m.real_model = weakref.ref(real_inner)
    loaded_m.model_finalizer = Mock()

    monkeypatch.setattr(model_management, "current_loaded_models", [loaded_m])

    free_mem_state = 500_000_000
    def mock_get_free(dev=None, torch_free_too=False):
        nonlocal free_mem_state
        return free_mem_state if not torch_free_too else (free_mem_state, 0)

    soft_empty_called = False
    def mock_soft_empty(*args, **kwargs):
        nonlocal soft_empty_called, free_mem_state
        soft_empty_called = True
        free_mem_state = 2_000_000_000

    monkeypatch.setattr(model_management, "get_free_memory", mock_get_free)
    monkeypatch.setattr(model_management, "soft_empty_cache", mock_soft_empty)

    model_management.load_models_gpu([mock_model], minimum_memory_required=1_500_000_000)
    expected_free = 1_500_000_000 + model_management.extra_reserved_memory() - 500_000_000
    mock_model.partially_unload.assert_called_once_with(torch.device("cpu"), expected_free)
    assert soft_empty_called

