import asyncio

import pytest

import nodes
from comfy_execution.caching import CacheKeySetInputSignature
from comfy_execution.graph import DynamicPrompt


class Passthrough:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT",)}, "optional": {"link": ("INT",)}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, value, link=None):
        return (value,)


class NoChange:
    async def get(self, node_id):
        return False


@pytest.fixture(autouse=True)
def register_passthrough(monkeypatch):
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Passthrough", Passthrough)


def keys(prompt):
    key_set = CacheKeySetInputSignature(DynamicPrompt(prompt), [], NoChange())
    asyncio.run(key_set.add_keys(list(prompt)))
    return key_set.keys


def chain(ids, values):
    prompt, previous = {}, None
    for node_id, value in zip(ids, values):
        inputs = {"value": value}
        if previous is not None:
            inputs["link"] = [previous, 0]
        prompt[node_id] = {"class_type": "Passthrough", "inputs": inputs}
        previous = node_id
    return prompt


def test_equal_graphs_share_keys_whatever_the_node_ids():
    a = keys(chain(["1", "2", "3"], [1, 2, 3]))
    b = keys(chain(["x", "y", "z"], [1, 2, 3]))
    assert a["3"] == b["z"]
    assert hash(a["3"]) == hash(b["z"])


def test_upstream_change_alters_downstream_key():
    a = keys(chain(["1", "2", "3"], [1, 2, 3]))
    b = keys(chain(["1", "2", "3"], [9, 2, 3]))
    assert a["1"] != b["1"]
    assert a["3"] != b["3"]


def test_extra_link_alters_key():
    prompt = chain(["1", "2", "3"], [1, 2, 3])
    plain = keys(prompt)["3"]
    prompt["3"]["inputs"]["extra"] = ["1", 0]
    assert keys(prompt)["3"] != plain


def test_link_to_missing_node_never_matches():
    prompt = {"1": {"class_type": "Passthrough", "inputs": {"value": 1, "link": ["gone", 0]}}}
    assert keys(prompt)["1"] != keys(prompt)["1"]


def test_long_chain_builds_without_recursion():
    n = 3000
    built = keys(chain([str(i) for i in range(n)], range(n)))
    assert len(built) == n
    assert built[str(n - 1)] != built[str(n - 2)]
