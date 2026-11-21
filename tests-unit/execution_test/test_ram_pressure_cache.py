import asyncio

import pytest

import nodes
from comfy_execution.caching import CacheKeySetID, RAMPressureCache
from comfy_execution.graph import DynamicPrompt
from comfy_execution.graph_utils import GraphBuilder
from comfy_extras.nodes_toolkit import CreateList
from execution import CacheEntry, CacheType, PromptExecutor


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


class Server:
    client_id = None

    def send_sync(self, *args, **kwargs):
        pass


class Source:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("STRING",)}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"

    def execute(self, value):
        return (value,)


class SideEffect:
    calls = []

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING",)}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"

    def execute(self, text):
        self.calls.append(text)
        return (text,)


class Expand:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"template": ("STRING",)}}
    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "execute"

    def execute(self, template):
        g = GraphBuilder()
        listnode = g.node("CreateList")
        for i, text in enumerate(x.strip() for x in template.split(",")):
            node = g.node("TestSideEffect")
            node.set_input("text", text)
            listnode.set_input(f"inputs.input{i}", node.out(0))
        return {"result": (listnode.out(0),), "expand": g.finalize()}


class Sink:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("STRING",)}}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    INPUT_IS_LIST = True
    FUNCTION = "execute"

    def execute(self, value):
        return {}


@pytest.fixture
def executor(monkeypatch):
    for name, node in {"TestSource": Source, "TestSideEffect": SideEffect, "TestExpand": Expand, "TestSink": Sink, "CreateList": CreateList}.items():
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, name, node)
    SideEffect.calls = []
    # An inactive headroom above system RAM makes the sweep after every node evict all inactive entries.
    return PromptExecutor(Server(), cache_type=CacheType.RAM_PRESSURE, cache_args={"lru": 0, "ram": 0, "ram_inactive": 1024})


def execute(executor, template, prompt_id):
    before = len(SideEffect.calls)
    prompt = {
        "src": {"class_type": "TestSource", "inputs": {"value": template}},
        "expand": {"class_type": "TestExpand", "inputs": {"template": ["src", 0]}},
        "sink": {"class_type": "TestSink", "inputs": {"value": ["expand", 0]}},
    }
    executor.execute(prompt, prompt_id, execute_outputs=["sink"])
    assert executor.success
    return sorted(SideEffect.calls[before:])


def test_cached_parent_keeps_expanded_nodes_through_identical_reruns(executor):
    assert execute(executor, "a,b,c", "1") == ["a", "b", "c"]
    assert execute(executor, "a,b,c", "2") == []
    assert execute(executor, "a,b,c", "3") == []
    # The source changes and executes first, so the sweep runs before the parent expands again.
    assert execute(executor, "a,b,c,a", "4") == []
    assert execute(executor, "a,b,c,d", "5") == ["d"]


def test_expanded_nodes_of_a_replaced_parent_are_evicted(executor):
    execute(executor, "a,b,c", "1")
    execute(executor, "x", "2")
    execute(executor, "x", "3")
    assert not any("'a'" in str(key) for key in executor.caches.outputs.cache)
