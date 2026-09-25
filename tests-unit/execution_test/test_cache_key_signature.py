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


class Counted(str):
    # a literal that fails when two keys compare it more often than allowed
    calls = 0
    limit = 0
    __hash__ = str.__hash__

    def __eq__(self, other):
        Counted.calls += 1
        assert Counted.calls <= Counted.limit, "compared once per path through the ancestry, not once per node"
        return str.__eq__(self, other)


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


def fan_in(depth):
    # every node links twice to the one before it, as an expanded loop
    # iteration links several times into the previous iteration
    prompt = {"0": {"class_type": "Passthrough", "inputs": {"value": Counted("root")}}}
    for i in range(1, depth + 1):
        prompt[str(i)] = {"class_type": "Passthrough",
                          "inputs": {"value": i, "a": [str(i - 1), 0], "b": [str(i - 1), 0]}}
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


class Fingerprint:
    def __init__(self, value):
        self.value = value

    async def get(self, node_id):
        return self.value if node_id == "1" else False


def keys_with_fingerprint(value):
    prompt = chain(["1", "2", "3"], [1, 2, 3])
    key_set = CacheKeySetInputSignature(DynamicPrompt(prompt), [], Fingerprint(value))
    asyncio.run(key_set.add_keys(list(prompt)))
    return key_set.keys


def test_nan_fingerprint_never_matches_across_prompts():
    assert keys_with_fingerprint(float("NaN"))["3"] != keys_with_fingerprint(float("NaN"))["3"]


def test_fingerprint_that_does_not_serialize_still_reaches_descendants():
    # json refuses ints past 4300 digits, so the digest fails for both
    assert keys_with_fingerprint(10 ** 5000)["3"] != keys_with_fingerprint(10 ** 5000 + 1)["3"]


def test_cycle_terminates():
    prompt = chain(["1", "2"], [1, 2])
    prompt["1"]["inputs"]["link"] = ["2", 0]
    assert len(keys(prompt)) == 2


def test_separately_built_keys_compare_once_per_node():
    depth = 40
    Counted.limit = 10 ** 9
    a = keys(fan_in(depth))[str(depth)]
    b = keys(fan_in(depth))[str(depth)]
    Counted.calls, Counted.limit = 0, depth + 1
    assert a == b


def test_long_chain_builds_and_compares_without_recursion():
    n = 3000
    built = keys(chain([str(i) for i in range(n)], range(n)))
    assert len(built) == n
    assert built[str(n - 1)] != built[str(n - 2)]
    assert built[str(n - 1)] == keys(chain([str(i) for i in range(n)], range(n)))[str(n - 1)]
