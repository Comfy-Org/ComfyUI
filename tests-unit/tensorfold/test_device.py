import os

import pytest
import torch

from comfy.tensorfold.device import require_cuda, subprocess_environment


@pytest.fixture
def cuda(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (12, 0))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


@pytest.mark.parametrize("capability", [(12, 0), (12, 1)])
def test_only_compiled_architectures_are_admitted(cuda, monkeypatch, capability):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: capability)
    assert require_cuda(torch.device("cuda:0")) == 0


@pytest.mark.parametrize("capability", [(8, 0), (8, 9), (9, 0), (10, 0)])
def test_other_gpu_architectures_fail_before_launch(cuda, monkeypatch, capability):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: capability)
    with pytest.raises(RuntimeError, match="SM 120 and 121"):
        require_cuda(torch.device("cuda:0"))


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_other_device_types_fail(cuda, device):
    with pytest.raises(RuntimeError, match="NVIDIA CUDA"):
        require_cuda(torch.device(device))


def test_hip_cannot_enter_the_cuda_driver(cuda, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.0")
    with pytest.raises(RuntimeError, match="NVIDIA CUDA"):
        require_cuda(torch.device("cuda:0"))


def test_unavailable_cuda_fails(cuda, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="NVIDIA CUDA"):
        require_cuda(torch.device("cuda:0"))


def test_unmasked_device_uses_comfy_logical_index(cuda):
    assert subprocess_environment(torch.device("cuda:2")) == {"CUDA_VISIBLE_DEVICES": "2"}
    assert subprocess_environment(torch.device("cuda")) == {"CUDA_VISIBLE_DEVICES": "1"}
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


@pytest.mark.parametrize("visible,expected", [("2,5", "5"), ("GPU-first,GPU-second", "GPU-second"),
                                            (" MIG-first , MIG-second ", "MIG-second")])
def test_logical_device_is_mapped_through_existing_mask(cuda, monkeypatch, visible, expected):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    assert subprocess_environment(torch.device("cuda:1")) == {"CUDA_VISIBLE_DEVICES": expected}
    assert os.environ["CUDA_VISIBLE_DEVICES"] == visible


@pytest.mark.parametrize("visible", ["", "-1", "0"])
def test_invalid_visibility_fails_before_launch(cuda, monkeypatch, visible):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    with pytest.raises(RuntimeError, match="outside CUDA_VISIBLE_DEVICES"):
        subprocess_environment(torch.device("cuda:1"))
