import asyncio
import random

import pytest

import nodes
from comfy_execution.cache_provider import _contains_self_unequal
import comfy_execution.caching as caching
from comfy_execution.caching import CacheKeySetInputSignature, to_hashable
from comfy_execution.graph import DynamicPrompt
from comfy_execution.graph_utils import is_link


class PlainNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}


def recursive_ordered_ancestry(dynprompt, node_id):
    """The recursive walk the cache used before, kept to check that cache keys don't change."""
    ancestors = []
    order_mapping = {}

    def walk(node_id):
        if not dynprompt.has_node(node_id):
            return
        inputs = dynprompt.get_node(node_id)["inputs"]
        for key in sorted(inputs.keys()):
            if is_link(inputs[key]):
                ancestor_id = inputs[key][0]
                if ancestor_id not in order_mapping:
                    ancestors.append(ancestor_id)
                    order_mapping[ancestor_id] = len(ancestors) - 1
                    walk(ancestor_id)

    walk(node_id)
    return ancestors, order_mapping


def ordered_ancestry(prompt, node_id):
    dynprompt = DynamicPrompt(prompt)
    return CacheKeySetInputSignature(dynprompt, [], None).get_ordered_ancestry(dynprompt, node_id)


def node(**inputs):
    return {"class_type": "Test", "inputs": inputs}


def random_prompt(seed):
    rng = random.Random(seed)
    prompt = {}
    for index in range(40):
        inputs = {}
        for _ in range(rng.randint(0, 4)):
            key = rng.choice("abcdefgh")
            if index and rng.random() < 0.8:
                inputs[key] = [str(rng.randrange(index)), rng.randrange(3)]
            elif rng.random() < 0.5:
                inputs[key] = ["missing", 0]
            else:
                inputs[key] = rng.random()
        prompt[str(index)] = node(**inputs)
    return prompt


GRAPHS = {
    "branching": {
        "a": node(value=1),
        "b": node(x=["a", 0]),
        "c": node(y=["a", 0]),
        "d": node(z=["c", 0], a=["b", 0]),
    },
    "shared_ancestors": {
        "root": node(),
        "left": node(value=["root", 0]),
        "right": node(value=["root", 1]),
        "mid": node(b=["right", 0], a=["left", 0], c=["root", 0]),
        "out": node(second=["mid", 0], first=["left", 0], third=["right", 0]),
    },
    "missing_nodes": {
        "a": node(gone=["missing", 0]),
        "b": node(x=["a", 0], y=["also_missing", 1], z=["a", 1]),
        "out": node(m=["b", 0], n=["missing", 0], value="text"),
    },
    "cycle": {
        "a": node(x=["b", 0]),
        "b": node(x=["a", 0]),
        "out": node(x=["a", 0]),
    },
    **{f"random_{seed}": random_prompt(seed) for seed in range(20)},
}


@pytest.mark.parametrize("name", GRAPHS)
def test_ordered_ancestry_matches_recursive_walk(name):
    prompt = GRAPHS[name]
    for node_id in [*prompt, "missing"]:
        expected = recursive_ordered_ancestry(DynamicPrompt(prompt), node_id)
        assert ordered_ancestry(prompt, node_id) == expected


def test_ordered_ancestry_walks_chain_longer_than_recursion_limit():
    prompt = {"0": node()}
    for index in range(1, 5000):
        prompt[str(index)] = node(value=[str(index - 1), 0])

    ancestors, order_mapping = ordered_ancestry(prompt, "4999")

    assert ancestors == [str(index) for index in range(4998, -1, -1)]
    assert order_mapping == {node_id: position for position, node_id in enumerate(ancestors)}


class Changed:
    """An is-changed cache with fixed per-node values."""

    def __init__(self, values):
        self.values = values

    async def get(self, node_id):
        return self.values.get(node_id, False)


class FullKeys(CacheKeySetInputSignature):
    """Builds every key in full, as the cache did before small keys for uncacheable nodes."""

    async def is_uncacheable(self, dynprompt, node_id):
        return False


def keyed(key_class, prompt, changed):
    dynprompt = DynamicPrompt(prompt)
    keys = key_class(dynprompt, [], Changed(changed))
    asyncio.run(keys.add_keys(list(prompt)))
    return keys


def prompt_with_uncacheable_values(seed):
    rng = random.Random(seed)
    prompt = random_prompt(seed)
    changed = {}
    for node_id, value in prompt.items():
        roll = rng.random()
        if roll < 0.05:
            changed[node_id] = float("NaN")
        elif roll < 0.1:
            value["inputs"]["object"] = object()
    return prompt, changed


@pytest.mark.parametrize("seed", range(20))
def test_only_keys_that_could_never_match_change(monkeypatch, seed):
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Test", PlainNode)
    prompt, changed = prompt_with_uncacheable_values(seed)

    keys = keyed(CacheKeySetInputSignature, prompt, changed)
    full = keyed(FullKeys, prompt, changed)

    for node_id in prompt:
        full_key = full.get_data_key(node_id)
        assert keys.uncacheable[node_id] == _contains_self_unequal(full_key)
        if keys.uncacheable[node_id]:
            assert _contains_self_unequal(keys.get_data_key(node_id))
        else:
            assert keys.get_data_key(node_id) == full_key


def test_identical_uncacheable_nodes_still_share_a_key(monkeypatch):
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Test", PlainNode)
    prompt = {
        "source": node(value=1),
        "a": node(x=["source", 0]),
        "b": node(x=["source", 0]),
        "after_a": node(x=["a", 0]),
        "after_b": node(x=["b", 0]),
        "other": node(x=["source", 0], extra=2),
    }
    changed = {"source": float("NaN")}

    keys = keyed(CacheKeySetInputSignature, prompt, changed)
    full = keyed(FullKeys, prompt, changed)

    assert full.get_data_key("a") == full.get_data_key("b")
    assert keys.get_data_key("a") == keys.get_data_key("b")
    assert keys.get_data_key("after_a") == keys.get_data_key("after_b")
    assert keys.get_data_key("other") != keys.get_data_key("a")


def test_identical_nodes_with_identical_cacheable_inputs_share_a_key_in_any_order(monkeypatch):
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Test", PlainNode)
    prompt = {
        "source": node(value=1),
        "c1": node(value=2),
        "c2": node(value=2),
        "a": node(x=["source", 0], y=["c1", 0]),
        "b": node(x=["source", 0], y=["c2", 0]),
    }
    dynprompt = DynamicPrompt(prompt)
    keys = CacheKeySetInputSignature(dynprompt, [], Changed({"source": float("NaN")}))
    asyncio.run(keys.add_keys(["b", "a", "c2", "c1", "source"]))

    assert keys.get_data_key("a") == keys.get_data_key("b")


@pytest.mark.parametrize("never_equal", ["unhashable", "missing"])
def test_nodes_whose_full_keys_never_matched_get_unique_small_keys(monkeypatch, never_equal):
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Test", PlainNode)
    source = "missing" if never_equal == "missing" else "source"
    prompt = {"a": node(x=[source, 0]), "b": node(x=[source, 0])}
    if never_equal == "unhashable":
        prompt["source"] = node(value=object())

    keys = keyed(CacheKeySetInputSignature, prompt, {})
    full = keyed(FullKeys, prompt, {})

    assert full.get_data_key("a") != full.get_data_key("b")
    assert keys.get_data_key("a") != keys.get_data_key("b")


def test_a_cycle_through_an_uncacheable_node_still_keys(monkeypatch):
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Test", PlainNode)
    changed = {"x": float("NaN"), "y": float("NaN")}
    keys = keyed(CacheKeySetInputSignature, {"x": node(y=["y", 0]), "y": node(x=["x", 0])}, changed)

    assert _contains_self_unequal(keys.get_data_key("x")) and _contains_self_unequal(keys.get_data_key("y"))


def test_a_node_keyed_after_an_override_is_judged_on_the_current_graph(monkeypatch):
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Test", PlainNode)
    dynprompt = DynamicPrompt({"source": node(value=1), "close": node(x=["source", 0]), "after": node(x=["close", 0])})
    keys = CacheKeySetInputSignature(dynprompt, [], Changed({"source": float("NaN")}))
    asyncio.run(keys.add_keys(["close"]))

    dynprompt.override_node("close", node())
    asyncio.run(keys.add_keys(["after"]))

    assert keys.uncacheable == {"after": False, "close": False}


def test_full_keys_keep_their_format(monkeypatch):
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Test", PlainNode)
    keys = keyed(CacheKeySetInputSignature, {"a": node(value=1), "b": node(x=["a", 0])}, {})

    assert keys.get_data_key("b") == to_hashable([["Test", False, ("x", ("ANCESTOR", 0, 0))], ["Test", False, ("value", 1)]])


def test_key_budget_counts_ancestors_outside_the_new_nodes(monkeypatch):
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Test", PlainNode)
    monkeypatch.setattr(caching, "FULL_KEY_BUDGET", 60)
    prompt = {"0": node()}
    for index in range(1, 50):
        prompt[str(index)] = node(value=[str(index - 1), 0])
    prompt.update({"new_a": node(x=["49", 0]), "new_b": node(x=["49", 0])})
    cache = caching.HierarchicalCache(CacheKeySetInputSignature)
    asyncio.run(cache.set_prompt(DynamicPrompt(prompt), [], Changed({})))

    with pytest.raises(ValueError, match="turn off cache iterations"):
        asyncio.run(caching.check_full_key_budget(cache, ["new_a", "new_b"]))


def test_key_budget_accepts_a_wide_expansion_of_independent_nodes(monkeypatch):
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Test", PlainNode)
    prompt = {f"new_{index}": node(value=index) for index in range(2000)}
    cache = caching.HierarchicalCache(CacheKeySetInputSignature)
    asyncio.run(cache.set_prompt(DynamicPrompt(prompt), [], Changed({})))

    asyncio.run(caching.check_full_key_budget(cache, list(prompt)))


def test_small_and_full_keys_can_be_ordered_against_each_other(monkeypatch):
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Test", PlainNode)
    keys = keyed(CacheKeySetInputSignature, {"cached": node(value=1), "changed": node(value=2)}, {"changed": float("NaN")})

    # The RAM pressure cache sorts eviction candidates on (score, time, key, size), so tied entries compare keys.
    sorted([(0, 0, keys.get_data_key("cached"), 0), (0, 0, keys.get_data_key("changed"), 0)])
