from unittest.mock import MagicMock

import pytest
import torch
from aiohttp.test_utils import TestClient, TestServer

import comfy.model_management
import server

CUDA = torch.device("cuda", 0)


@pytest.fixture
def cuda_memory(monkeypatch):
    memory = {"reserved": 6, "active": 2, "free": 10, "total": 100}

    def memory_stats(dev):
        return {"reserved_bytes.all.current": memory["reserved"], "active_bytes.all.current": memory["active"]}

    monkeypatch.setattr(torch.cuda, "memory_stats", memory_stats)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev: (memory["free"], memory["total"]))
    monkeypatch.setattr(comfy.model_management, "last_torch_cuda_memory", {})
    return memory


def test_cuda_memory_without_allocator_lock_reuses_the_last_torch_figures(monkeypatch, cuda_memory):
    assert comfy.model_management.get_memory_without_allocator_lock(CUDA) == (100, 0, 10, 0), \
        "no torch figures before anything read them"

    assert comfy.model_management.get_free_memory(CUDA, torch_free_too=True) == (14, 4)
    cuda_memory.update(reserved=50, active=50, free=1)

    def allocator_locked(dev):
        raise AssertionError("torch.cuda.memory_stats waits on the allocator lock")

    monkeypatch.setattr(torch.cuda, "memory_stats", allocator_locked)
    assert comfy.model_management.get_memory_without_allocator_lock(CUDA) == (100, 6, 5, 4), \
        "driver free and total are current, torch reserved and free are from the last read"


def test_non_cuda_memory_without_allocator_lock_reads_as_before():
    cpu = torch.device("cpu")
    total, torch_total, free, torch_free = comfy.model_management.get_memory_without_allocator_lock(cpu)
    assert total == torch_total > 0
    assert 0 < free == torch_free <= total


@pytest.mark.asyncio
async def test_system_stats_never_asks_torch_allocator_for_a_cuda_device(monkeypatch, cuda_memory):
    comfy.model_management.get_free_memory(CUDA)
    cuda_memory.update(free=30)
    asked = []
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda dev: asked.append(dev))
    monkeypatch.setattr(comfy.model_management, "get_torch_device", lambda: CUDA)
    monkeypatch.setattr(comfy.model_management, "get_all_torch_devices", lambda: [CUDA])
    monkeypatch.setattr(comfy.model_management, "get_torch_device_name", lambda d: "fake")
    prompt_server = server.PromptServer(None, MagicMock(enabled=False))
    handler = next(r.handler for r in prompt_server.routes if getattr(r, "path", None) == "/system_stats")
    prompt_server.app.router.add_get("/system_stats", handler)

    async with TestClient(TestServer(prompt_server.app)) as client:
        device = (await (await client.get("/system_stats")).json())["devices"][0]

    assert asked == []
    assert (device["vram_total"], device["torch_vram_total"], device["vram_free"], device["torch_vram_free"]) == (100, 6, 34, 4)
