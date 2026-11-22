import random

import pytest

from comfy_execution.caching import CacheKeySetInputSignature
from comfy_execution.graph import DynamicPrompt
from comfy_execution.graph_utils import is_link


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
