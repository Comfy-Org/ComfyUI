import pytest
import torch

from comfy.cli_args import args
import comfy.model_management as model_management


@pytest.mark.parametrize("cuda_device", [None, "0", "GPU-example"])
def test_core_devices_default_to_current_without_enumerating(monkeypatch, cuda_device):
    current = torch.device("cuda", 0)
    monkeypatch.setattr(args, "cuda_device", cuda_device)
    monkeypatch.setattr(args, "default_device", 1)
    monkeypatch.setattr(model_management, "get_torch_device", lambda: current)
    monkeypatch.setattr(
        model_management,
        "get_all_torch_devices",
        lambda: pytest.fail("secondary devices must not be enumerated"),
    )

    assert model_management.get_core_torch_devices() == [current]


@pytest.mark.parametrize("cuda_device", ["0,1", "GPU-primary,GPU-secondary"])
def test_core_devices_use_explicit_multi_device_selection(monkeypatch, cuda_device):
    current = torch.device("cuda", 0)
    devices = [current, torch.device("cuda", 1)]
    monkeypatch.setattr(args, "cuda_device", cuda_device)
    monkeypatch.setattr(model_management, "get_torch_device", lambda: current)
    monkeypatch.setattr(model_management, "get_all_torch_devices", lambda: devices.copy())

    assert model_management.get_core_torch_devices() == devices
    assert model_management.get_core_torch_devices(exclude_current=True) == devices[1:]


def test_all_devices_remain_discoverable(monkeypatch):
    monkeypatch.setattr(model_management, "cpu_state", model_management.CPUState.GPU)
    monkeypatch.setattr(model_management, "is_nvidia", lambda: True)
    monkeypatch.setattr(model_management.torch.cuda, "device_count", lambda: 2)

    assert model_management.get_all_torch_devices() == [
        torch.device("cuda", 0),
        torch.device("cuda", 1),
    ]


def test_device_options_only_offer_core_devices(monkeypatch):
    current = torch.device("cuda", 0)
    secondary = torch.device("cuda", 1)
    monkeypatch.setattr(args, "cuda_device", None)
    monkeypatch.setattr(model_management, "get_torch_device", lambda: current)
    monkeypatch.setattr(model_management, "get_all_torch_devices", lambda: [current, secondary])

    assert model_management.get_gpu_device_options() == ["default", "cpu"]
    assert model_management.resolve_gpu_device_option("gpu:1") is None

    monkeypatch.setattr(args, "cuda_device", "0,1")
    assert model_management.get_gpu_device_options() == ["default", "cpu", "gpu:0", "gpu:1"]
    assert model_management.resolve_gpu_device_option("gpu:1") == secondary


def test_unload_all_models_only_uses_core_devices(monkeypatch):
    current = torch.device("cuda", 0)
    freed = []
    monkeypatch.setattr(model_management, "get_core_torch_devices", lambda: [current])
    monkeypatch.setattr(model_management, "free_memory", lambda amount, device: freed.append((amount, device)))

    model_management.unload_all_models()

    assert freed == [(1e30, current)]
