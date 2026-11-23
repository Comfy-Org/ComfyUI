import os

import pytest
import safetensors.torch
import torch

import comfy.memory_management
import comfy.model_management as mm
import comfy.utils
from comfy.memory_management import _read_file_slice, move_modules_reading_files


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)
    monkeypatch.setattr(comfy.utils, "DISABLE_MMAP", False)
    # Pinned memory needs a GPU; the CPU runner stages through pageable memory instead.
    monkeypatch.setattr(comfy.memory_management, "_staging_arena", None)
    real_empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, pin_memory=False, **kw: real_empty(*a, **kw))
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

    mmap_ptrs = [outer.weight.data_ptr(), outer.inner.weight.data_ptr(), outer.inner.bias.data_ptr()]
    move_modules_reading_files([outer.inner, outer], torch.device("cpu"), max_staged_bytes=1 << 20, workers=2)

    assert len(reads) == len(set(reads)) == 3
    # .to("cpu") moved nothing, so every holder is pointed back at its mmap, not left on the staging copy.
    assert [outer.weight.data_ptr(), outer.inner.weight.data_ptr(), outer.inner.bias.data_ptr()] == mmap_ptrs
    assert torch.equal(outer.weight, tensors["outer.weight"])
    assert torch.equal(outer.inner.weight, tensors["outer.inner.weight"])
    assert torch.equal(outer.inner.bias, tensors["outer.inner.bias"])


def test_module_over_the_staging_budget_moves_without_reading(checkpoint, monkeypatch):
    path, tensors = checkpoint
    sd = comfy.utils.load_torch_file(str(path))
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(sd["outer.weight"], requires_grad=False)
    reads = []
    monkeypatch.setattr(comfy.memory_management, "_read_file_slice", lambda info, address: reads.append(info))
    move_modules_reading_files([module], torch.device("cpu"), max_staged_bytes=tensors["outer.weight"].nbytes - 1)
    assert not reads
    assert torch.equal(module.weight, tensors["outer.weight"])


def test_in_place_update_drops_the_file_slice(checkpoint):
    """A patched weight no longer matches the file, so it must not be re-read from it."""
    path, _ = checkpoint
    sd = comfy.utils.load_torch_file(str(path))
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(sd["outer.weight"].clone(), requires_grad=False)
    module.weight.untyped_storage()._comfy_mmap_file_slice = sd["outer.weight"].untyped_storage()._comfy_mmap_file_slice
    comfy.utils.copy_to_param(module, "weight", torch.zeros(16, 8))
    assert not hasattr(module.weight.untyped_storage(), "_comfy_mmap_file_slice")
    assert comfy.memory_management._mmap_file_slice_tensors(module, set()) == []


def test_staging_follows_the_pinned_memory_policy(monkeypatch):
    gpu = torch.device("cuda", 0)
    monkeypatch.setattr(mm, "is_amd", lambda: True)
    monkeypatch.setattr(mm.args, "high_ram", False)
    monkeypatch.setattr(comfy.memory_management, "RAM_CACHE_HEADROOM", 0)
    monkeypatch.setattr(mm.comfy.system_memory, "virtual_memory_available", lambda: 64 << 30)
    monkeypatch.setattr(mm, "TOTAL_PINNED_MEMORY", 0)
    monkeypatch.setattr(mm, "MAX_PINNED_MEMORY", 64 << 30)
    assert mm.weight_staging_bytes(gpu) == 512 << 20
    assert mm.weight_staging_bytes(torch.device("cpu")) == 0
    monkeypatch.setattr(mm, "TOTAL_PINNED_MEMORY", (64 << 30) - (100 << 20))
    assert mm.weight_staging_bytes(gpu) == 100 << 20
    monkeypatch.setattr(mm, "MAX_PINNED_MEMORY", -1)  # --disable-pinned-memory, integrated GPU
    assert mm.weight_staging_bytes(gpu) == 0
    monkeypatch.setattr(mm, "MAX_PINNED_MEMORY", 64 << 30)
    monkeypatch.setattr(mm, "is_amd", lambda: False)
    assert mm.weight_staging_bytes(gpu) == 0
    monkeypatch.setattr(mm, "is_amd", lambda: True)
    # Low RAM: only what is free beyond ensure_pin_budget's 2 GiB headroom may be staged.
    monkeypatch.setattr(mm.comfy.system_memory, "virtual_memory_available", lambda: (2 << 30) + (64 << 20))
    assert mm.weight_staging_bytes(gpu) == 64 << 20
    monkeypatch.setattr(mm.comfy.system_memory, "virtual_memory_available", lambda: 1 << 30)
    assert mm.weight_staging_bytes(gpu) == 0


def test_failed_move_releases_the_staging_copy(checkpoint, monkeypatch):
    """If .to() raises, the module's holders go back to the mmap instead of keeping pinned memory."""
    path, _ = checkpoint
    sd = comfy.utils.load_torch_file(str(path))
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(sd["outer.weight"], requires_grad=False)
    mmap_ptr = module.weight.data_ptr()
    def fail(*args, **kwargs):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(module, "to", fail)
    with pytest.raises(RuntimeError, match="out of memory"):
        move_modules_reading_files([module], torch.device("cpu"), max_staged_bytes=1 << 20)
    assert module.weight.data_ptr() == mmap_ptr


def test_ring_reuses_the_staging_buffer(checkpoint, monkeypatch):
    """Modules larger in total than the buffer stream through it, each read once, values intact."""
    path, tensors = checkpoint
    sd = comfy.utils.load_torch_file(str(path))
    modules = []
    for name in ("outer.weight", "outer.inner.weight", "outer.inner.bias"):
        m = torch.nn.Module()
        m.register_buffer("t", sd[name])
        modules.append((name, m))
    reads = []
    real_read = comfy.memory_management._read_file_slice
    monkeypatch.setattr(comfy.memory_management, "_read_file_slice",
                        lambda info, address: reads.append(info.offset) or real_read(info, address))
    # 768 bytes: room for the 512-byte weight, or the two smaller tensors, never all three
    move_modules_reading_files([m for _, m in modules], torch.device("cpu"), max_staged_bytes=768, workers=3)
    assert len(reads) == len(set(reads)) == 3
    for name, m in modules:
        assert torch.equal(m.t, tensors[name])
