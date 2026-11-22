import os

import pytest
import safetensors.torch
import torch

import comfy.memory_management
import comfy.utils
from comfy.memory_management import _read_file_slice, move_modules_reading_files


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)
    monkeypatch.setattr(comfy.utils, "DISABLE_MMAP", False)
    torch.manual_seed(0)
    tensors = {
        "outer.weight": torch.randn(16, 8),
        "outer.inner.weight": torch.randn(4, 8).to(torch.bfloat16),
        "outer.inner.bias": torch.arange(4, dtype=torch.int32),
        "empty": torch.empty(0),
    }
    path = tmp_path / "model.safetensors"
    safetensors.torch.save_file(tensors, str(path), metadata={"note": "x"})
    return path, tensors


def test_legacy_load_records_file_slices(checkpoint):
    path, tensors = checkpoint
    sd = comfy.utils.load_torch_file(str(path))
    for name, expected in tensors.items():
        info = getattr(sd[name].untyped_storage(), "_comfy_mmap_file_slice", None)
        if expected.numel() == 0:
            assert info is None
            continue
        assert info.size == expected.nbytes
        restored = torch.empty_like(expected)
        _read_file_slice(info, restored.data_ptr())
        assert torch.equal(restored, expected)


def test_disable_mmap_records_no_slices(checkpoint, monkeypatch):
    path, _ = checkpoint
    monkeypatch.setattr(comfy.utils, "DISABLE_MMAP", True)
    sd = comfy.utils.load_torch_file(str(path))
    assert all(not hasattr(t.untyped_storage(), "_comfy_mmap_file_slice") for t in sd.values())


@pytest.mark.parametrize("preadv", [True, False])
def test_move_reads_each_tensor_once(checkpoint, monkeypatch, preadv):
    """A parent and its child can both be listed; each file-backed tensor is read once and keeps its value."""
    path, tensors = checkpoint
    if not preadv:
        monkeypatch.delattr(os, "preadv", raising=False)
    sd = comfy.utils.load_torch_file(str(path))

    outer = torch.nn.Module()
    outer.weight = torch.nn.Parameter(sd["outer.weight"], requires_grad=False)
    outer.inner = torch.nn.Module()
    outer.inner.weight = torch.nn.Parameter(sd["outer.inner.weight"], requires_grad=False)
    outer.inner.register_buffer("bias", sd["outer.inner.bias"])

    reads = []
    real_read = comfy.memory_management._read_file_slice
    monkeypatch.setattr(comfy.memory_management, "_read_file_slice",
                        lambda info, address: reads.append(info.offset) or real_read(info, address))
    # Pinned memory needs a GPU; the CPU runner exercises the same staging with pageable memory.
    real_empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, pin_memory=False, **kw: real_empty(*a, **kw))

    move_modules_reading_files([outer.inner, outer], torch.device("cpu"), workers=2, prefetch=1)

    assert len(reads) == len(set(reads)) == 3
    assert torch.equal(outer.weight, tensors["outer.weight"])
    assert torch.equal(outer.inner.weight, tensors["outer.inner.weight"])
    assert torch.equal(outer.inner.bias, tensors["outer.inner.bias"])
