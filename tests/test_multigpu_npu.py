import threading
from typing import NamedTuple

import pytest

import comfy.model_management as model_management
import comfy.samplers as samplers
from comfy.multigpu import MultiGPUThreadPool


class FakeAccelerator:
    def __init__(self, current=0):
        self.current = current
        self.set_calls = []

    def current_device(self):
        return self.current

    def set_device(self, device):
        self.set_calls.append(device)
        self.current = device.index if hasattr(device, "index") else device


class FakeDevice(NamedTuple):
    type: str
    index: int


def fake_device(device_type, index=None):
    return FakeDevice(device_type, index)


@pytest.mark.parametrize("device_type", ["cuda", "npu"])
def test_device_context_switches_and_restores(monkeypatch, device_type):
    accelerator = FakeAccelerator(current=0)
    monkeypatch.setattr(model_management.torch, device_type, accelerator, raising=False)
    device = fake_device(device_type, 2)

    with model_management.cuda_device_context(device):
        assert accelerator.current_device() == 2

    assert accelerator.current_device() == 0
    assert accelerator.set_calls == [device, 0]


@pytest.mark.parametrize("device_type", ["cuda", "npu"])
def test_device_context_is_noop_for_current_device(monkeypatch, device_type):
    accelerator = FakeAccelerator(current=1)
    monkeypatch.setattr(model_management.torch, device_type, accelerator, raising=False)

    with model_management.cuda_device_context(fake_device(device_type, 1)):
        assert accelerator.current_device() == 1

    assert accelerator.set_calls == []


def test_npu_device_context_restores_after_exception(monkeypatch):
    accelerator = FakeAccelerator(current=0)
    monkeypatch.setattr(model_management.torch, "npu", accelerator, raising=False)
    device = fake_device("npu", 2)

    with pytest.raises(RuntimeError, match="expected failure"):
        with model_management.cuda_device_context(device):
            assert accelerator.current_device() == 2
            raise RuntimeError("expected failure")

    assert accelerator.current_device() == 0
    assert accelerator.set_calls == [device, 0]


def test_set_torch_device_supports_npu(monkeypatch):
    accelerator = FakeAccelerator()
    monkeypatch.setattr(model_management.torch, "npu", accelerator, raising=False)
    device = fake_device("npu", 1)

    model_management.set_torch_device(device)

    assert accelerator.set_calls == [device]


def test_multigpu_aggregation_waits_for_copy_events(monkeypatch):
    operations = []
    stream_devices = []

    class FakeEvent:
        def __init__(self, name):
            self.name = name

    class FakeStream:
        def wait_event(self, event):
            operations.append(("wait", event.name))

    class FakeNPU:
        def current_stream(self, device):
            stream_devices.append(device)
            return FakeStream()

    class FakeOutput:
        def __init__(self, event_name):
            self.event_name = event_name

        def __mul__(self, other):
            assert operations[-1] == ("wait", self.event_name)
            operations.append(("consume", self.event_name))
            return model_management.torch.ones_like(other)

    class FakeThreadPool:
        def __init__(self, responses):
            self.responses = responses

        def submit(self, device, fn, *args):
            pass

        def get_result(self, device):
            return self.responses[device], None

    class FakePatcher:
        def prepare_state(self, timestep, model_options):
            pass

    class FakeModel:
        current_patcher = FakePatcher()

        def memory_required(self, input_shape, cond_shapes):
            return 1

    devices = [fake_device("npu", 0), fake_device("npu", 1)]
    x_in = model_management.torch.zeros((1, 1, 1, 1))
    multiplier = model_management.torch.ones_like(x_in)
    events = [FakeEvent("event-0"), FakeEvent("event-1")]
    responses = {
        device: [([FakeOutput(event.name)], [multiplier], [None], 1, [index], event, None)]
        for index, (device, event) in enumerate(zip(devices, events))
    }
    model_options = {
        "multigpu_clones": {device: object() for device in devices},
        "multigpu_thread_pool": FakeThreadPool(responses),
    }
    conds = [[{"model_conds": {}, "uuid": f"cond-{index}"}] for index in range(2)]

    monkeypatch.setattr(samplers.comfy.model_management, "get_free_memory", lambda device: 1024)
    monkeypatch.setattr(samplers.torch, "npu", FakeNPU(), raising=False)

    samplers._calc_cond_batch_multigpu(
        FakeModel(), conds, x_in, model_management.torch.ones(1), model_options
    )

    assert operations == [
        ("wait", "event-0"),
        ("consume", "event-0"),
        ("wait", "event-1"),
        ("consume", "event-1"),
    ]
    assert stream_devices == [x_in.device, x_in.device]


def test_multigpu_thread_pool_binds_each_worker(monkeypatch):
    worker_state = threading.local()
    devices = ["npu:0", "npu:1", "npu:2"]
    monkeypatch.setattr(
        model_management,
        "set_torch_device",
        lambda device: setattr(worker_state, "device", device),
    )
    pool = MultiGPUThreadPool(devices)
    try:
        for device in devices:
            pool.submit(device, lambda: worker_state.device)
        for device in devices:
            result, error = pool.get_result(device)
            assert error is None
            assert result == device
    finally:
        pool.shutdown()


def test_multigpu_thread_pool_reports_error_and_keeps_worker_alive(monkeypatch):
    monkeypatch.setattr(model_management, "set_torch_device", lambda device: None)
    pool = MultiGPUThreadPool(["npu:0"])
    try:
        def fail():
            raise RuntimeError("expected failure")

        pool.submit("npu:0", fail)
        result, error = pool.get_result("npu:0")
        assert result is None
        assert isinstance(error, RuntimeError)
        assert str(error) == "expected failure"

        pool.submit("npu:0", lambda: "recovered")
        result, error = pool.get_result("npu:0")
        assert error is None
        assert result == "recovered"
    finally:
        pool.shutdown()
