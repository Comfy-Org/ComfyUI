import asyncio

import pytest

from comfy_execution import caching
from comfy_execution.caching import CacheKeySetID, RAMPressureCache
from comfy_execution.graph import DynamicPrompt
from comfy_execution.utils import current_prompt_id
from execution import CacheEntry


def make_prompt(node_id):
    return DynamicPrompt({node_id: {"class_type": "Test", "inputs": {}}})


async def bind(cache, prompt_id, node_id):
    current_prompt_id.set(prompt_id)
    await cache.set_prompt(make_prompt(node_id), [node_id], None)
    cache.clean_unused()


@pytest.fixture
def cache(monkeypatch):
    monkeypatch.setattr(caching, "virtual_memory_available", lambda: 0)
    return RAMPressureCache(CacheKeySetID)


def test_inactive_release_keeps_entries_of_every_in_flight_prompt(cache):
    async def scenario():
        await bind(cache, "A", "1")
        await cache.set("1", CacheEntry(ui={}, outputs=[object()]))

        await bind(cache, "B", "2")
        await cache.set("2", CacheEntry(ui={}, outputs=[object()]))

        cache.ram_release(1)
        assert set(cache.cache) == {("1", "Test"), ("2", "Test")}

        current_prompt_id.set("A")
        cache.end_prompt()
        cache.ram_release(1)
        assert set(cache.cache) == {("2", "Test")}

    asyncio.run(scenario())
