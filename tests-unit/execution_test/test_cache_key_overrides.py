import asyncio

import nodes
from comfy_execution.caching import CacheKeySetInputSignature
from comfy_execution.graph import DynamicPrompt


class Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}


class NotChanged:
    async def get(self, node_id):
        return False


def node_key(dynprompt, node_id):
    keys = CacheKeySetInputSignature(dynprompt, [node_id], NotChanged())
    asyncio.run(keys.add_keys([node_id]))
    return keys.get_data_key(node_id)


def test_cache_keys_follow_an_overrides_cache_key_inputs(monkeypatch):
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "TestNode", Node)
    dynprompt = DynamicPrompt({
        "source": {"class_type": "TestNode", "inputs": {"value": 1}},
        "close": {"class_type": "TestNode", "inputs": {"value": ["source", 0], "flag": True}},
        "after": {"class_type": "TestNode", "inputs": {"value": ["close", 0]}},
    })
    before = node_key(dynprompt, "after")

    dynprompt.override_node(
        "close",
        {"class_type": "TestNode", "inputs": {"flag": True}},
        cache_key_inputs={"value": ["source", 0], "flag": True},
    )

    assert node_key(dynprompt, "after") == before

    dynprompt.override_node("close", {"class_type": "TestNode", "inputs": {"flag": True}}, cache_key_inputs={"flag": True})

    assert node_key(dynprompt, "after") != before
