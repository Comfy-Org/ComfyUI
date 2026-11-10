from unittest.mock import MagicMock

import pytest
import torch
from aiohttp.test_utils import TestClient, TestServer

import comfy.model_management
import server

CUDA0 = torch.device("cuda", 0)
CUDA1 = torch.device("cuda", 1)


@pytest.fixture
def cuda_memory(monkeypatch):
    memory = {str(CUDA0): {"reserved": 6, "active": 2, "free": 10, "total": 100},
              str(CUDA1): {"reserved": 60, "active": 20, "free": 50, "total": 200}}

    def memory_stats(dev):
        m = memory[str(dev)]
        return {"reserved_bytes.all.current": m["reserved"], "active_bytes.all.current": m["active"]}

    for name in ("is_intel_xpu", "is_ascend_npu", "is_mlu"):
        monkeypatch.setattr(comfy.model_management, name, lambda: False)
    set_memory_stats(monkeypatch, memory_stats)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev: (memory[str(dev)]["free"], memory[str(dev)]["total"]))
    monkeypatch.setattr(comfy.model_management, "last_torch_cuda_memory", {})
    return memory


def set_memory_stats(monkeypatch, memory_stats):
    # torch.cuda.memory_reserved and friends call it through torch.cuda.memory
    monkeypatch.setattr(torch.cuda, "memory_stats", memory_stats)
    monkeypatch.setattr(torch.cuda.memory, "memory_stats", memory_stats)


def allocator_locked(dev):
    raise AssertionError("torch.cuda.memory_stats waits on the allocator lock")


def test_cuda_memory_without_allocator_lock_reuses_the_last_torch_figures(monkeypatch, cuda_memory):
    assert comfy.model_management.get_memory_without_allocator_lock(CUDA0) == (100, 0, 10, 0), \
        "no torch figures before anything read them"

    assert comfy.model_management.get_free_memory(CUDA0, torch_free_too=True) == (14, 4)
    cuda_memory[str(CUDA0)].update(reserved=50, active=50, free=1)
    set_memory_stats(monkeypatch, allocator_locked)
    assert comfy.model_management.get_memory_without_allocator_lock(CUDA0) == (100, 6, 5, 4), \
        "driver free and total are current, torch reserved and free are from the last read"
    cuda_memory[str(CUDA0)].update(free=99)
    assert comfy.model_management.get_memory_without_allocator_lock(CUDA0)[2] == 100, "free never passes total"


def test_soft_empty_cache_refreshes_the_torch_figures_of_devices_already_read(monkeypatch, cuda_memory):
    comfy.model_management.get_free_memory(CUDA0)
    comfy.model_management.get_free_memory(CUDA1)
    cuda_memory["0"] = cuda_memory[str(CUDA0)]
    comfy.model_management.get_free_memory(0)  # custom nodes may pass a bare index
    cuda_memory[str(CUDA0)].update(reserved=2, free=11)
    cuda_memory[str(CUDA1)].update(reserved=20, free=51)
    cuda_memory["cuda:2"] = {}  # never read: refreshing it would create a CUDA context on that GPU
    monkeypatch.setattr(comfy.model_management, "cpu_mode", lambda: False)
    monkeypatch.setattr(comfy.model_management, "cpu_state", comfy.model_management.CPUState.GPU)
    for name in ("is_available", "synchronize", "empty_cache", "ipc_collect"):
        monkeypatch.setattr(torch.cuda, name, lambda: True)

    comfy.model_management.soft_empty_cache()

    set_memory_stats(monkeypatch, allocator_locked)
    assert comfy.model_management.get_memory_without_allocator_lock(CUDA0) == (100, 2, 11, 0)
    assert comfy.model_management.get_memory_without_allocator_lock(CUDA1) == (200, 20, 51, 0)
    assert torch.device("cuda", 2) not in comfy.model_management.last_torch_cuda_memory


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("mps"), torch.device("xpu", 0), torch.device("privateuseone", 0)])
def test_non_cuda_memory_without_allocator_lock_reads_as_before(monkeypatch, device):
    monkeypatch.setattr(comfy.model_management, "get_total_memory", lambda dev, torch_total_too: (dev, "total"))
    monkeypatch.setattr(comfy.model_management, "get_free_memory", lambda dev, torch_free_too: (dev, "free"))
    monkeypatch.setattr(torch.cuda, "mem_get_info", allocator_locked)
    assert comfy.model_management.get_memory_without_allocator_lock(device) == (device, "total", device, "free")


@pytest.mark.asyncio
async def test_system_stats_never_asks_torch_allocator_for_a_cuda_device(monkeypatch, cuda_memory):
    comfy.model_management.get_free_memory(CUDA0)
    comfy.model_management.get_free_memory(CUDA1)
    cuda_memory[str(CUDA0)].update(free=30)
    set_memory_stats(monkeypatch, allocator_locked)
    monkeypatch.setattr(comfy.model_management, "get_torch_device", lambda: CUDA0)
    monkeypatch.setattr(comfy.model_management, "get_all_torch_devices", lambda: [CUDA0, CUDA1])
    monkeypatch.setattr(comfy.model_management, "get_torch_device_name", lambda d: "fake")
    prompt_server = server.PromptServer(None, MagicMock(enabled=False))
    handler = next(r.handler for r in prompt_server.routes if getattr(r, "path", None) == "/system_stats")
    prompt_server.app.router.add_get("/system_stats", handler)

    async with TestClient(TestServer(prompt_server.app)) as client:
        response = await client.get("/system_stats")
        assert response.status == 200
        devices = (await response.json())["devices"]

    figures = [(d["index"], d["vram_total"], d["torch_vram_total"], d["vram_free"], d["torch_vram_free"]) for d in devices]
    assert figures == [(0, 100, 6, 34, 4), (1, 200, 60, 90, 40)]
