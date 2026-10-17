import asyncio

from comfy_execution.caching import CacheKeySetID, RAMPressureCache
from comfy_execution.graph import DynamicPrompt
from execution import CacheEntry


async def run(cache, static_id, child_id):
    prompt = DynamicPrompt({
        "loop": {"class_type": "Loop", "inputs": {}},
        static_id: {"class_type": "Static", "inputs": {}},
    })
    await cache.set_prompt(prompt, ["loop", static_id], None)
    prompt.add_ephemeral_node(child_id, {"class_type": "Body", "inputs": {}}, "loop", child_id)
    await cache.ensure_subcache_for("loop", {child_id})
    for node_id in (static_id, child_id):
        await cache.set(node_id, CacheEntry(ui=None, outputs=[[node_id]]))


def test_inactive_sweep_keeps_previous_run_expanded_nodes_until_reexpansion():
    cache = RAMPressureCache(CacheKeySetID)

    async def scenario():
        await run(cache, "static_stale", "child_stale")
        await run(cache, "static_prev", "child_prev")
        prompt = DynamicPrompt({"loop": {"class_type": "Loop", "inputs": {}}})
        await cache.set_prompt(prompt, ["loop"], None)
        cache.ram_release(1 << 62)

    asyncio.run(scenario())

    assert set(cache.cache) == {("child_prev", "Body")}
    assert cache.expanded == {("child_prev", "Body")}


def test_active_release_evicts_previous_run_expanded_nodes():
    cache = RAMPressureCache(CacheKeySetID)

    async def scenario():
        await run(cache, "static_prev", "child_prev")
        prompt = DynamicPrompt({"loop": {"class_type": "Loop", "inputs": {}}})
        await cache.set_prompt(prompt, ["loop"], None)
        cache.ram_release(1 << 62, free_active=True)

    asyncio.run(scenario())

    assert cache.cache == {}
    assert cache.expanded == set()
