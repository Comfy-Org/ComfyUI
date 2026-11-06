from unittest.mock import MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

import comfy.model_management
import server


@pytest.mark.asyncio
async def test_system_stats_reports_last_values_while_a_prompt_runs(monkeypatch):
    memory = {"total": (100, 6), "free": (14, 4)}
    queried = []

    def get_total_memory(dev=None, torch_total_too=False):
        queried.append("total")
        return memory["total"] if torch_total_too else memory["total"][0]

    def get_free_memory(dev=None, torch_free_too=False):
        queried.append("free")
        return memory["free"] if torch_free_too else memory["free"][0]

    monkeypatch.setattr(comfy.model_management, "get_total_memory", get_total_memory)
    monkeypatch.setattr(comfy.model_management, "get_free_memory", get_free_memory)
    prompt_server = server.PromptServer(None, MagicMock(enabled=False))
    handler = next(r.handler for r in prompt_server.routes if getattr(r, "path", None) == "/system_stats")
    prompt_server.app.router.add_get("/system_stats", handler)

    def device(stats):
        d = stats["devices"][0]
        return d["vram_total"], d["torch_vram_total"], d["vram_free"], d["torch_vram_free"]

    async with TestClient(TestServer(prompt_server.app)) as client:
        with comfy.model_management.model_execution_lock:
            queried.clear()
            assert device(await (await client.get("/system_stats")).json()) == (100, 6, 14, 4), \
                "a first request during a prompt gets the values read at startup"
            assert queried == ["total", "free"], "only the cpu's RAM is read while a prompt runs"

        memory.update(total=(200, 7), free=(30, 5))
        assert device(await (await client.get("/system_stats")).json()) == (200, 7, 30, 5)

        memory.update(total=(300, 8), free=(40, 6))
        with comfy.model_management.model_execution_lock:
            assert device(await (await client.get("/system_stats")).json()) == (200, 7, 30, 5)
