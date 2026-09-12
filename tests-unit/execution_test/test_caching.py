"""Tests for cache-signature canonicalization hardening."""

import asyncio
import importlib
import sys
import types

import pytest


class _DummyNode:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {}}


class _FakeDynPrompt:
    def __init__(self, nodes_by_id):
        self._nodes_by_id = nodes_by_id

    def has_node(self, node_id):
        return node_id in self._nodes_by_id

    def get_node(self, node_id):
        return self._nodes_by_id[node_id]


class _FakeIsChangedCache:
    def __init__(self, values):
        self._values = values

    async def get(self, node_id):
        return self._values[node_id]


class _OpaqueValue:
    pass


@pytest.fixture
def caching_module(monkeypatch):
    torch_module = types.ModuleType("torch")
    nodes_module = types.ModuleType("nodes")
    nodes_module.NODE_CLASS_MAPPINGS = {}

    graph_module = types.ModuleType("comfy_execution.graph")
    graph_module.DynamicPrompt = type("DynamicPrompt", (), {})

    model_patcher_module = types.ModuleType("comfy.model_patcher")
    model_patcher_module.is_model_patcher_output = lambda output: False

    system_memory_module = types.ModuleType("comfy.system_memory")
    system_memory_module.virtual_memory_available = lambda: 0

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "nodes", nodes_module)
    monkeypatch.setitem(sys.modules, "comfy_execution.graph", graph_module)
    monkeypatch.setitem(sys.modules, "comfy.model_patcher", model_patcher_module)
    monkeypatch.setitem(sys.modules, "comfy.system_memory", system_memory_module)
    monkeypatch.delitem(sys.modules, "comfy_execution.caching", raising=False)

    module = importlib.import_module("comfy_execution.caching")
    yield module, nodes_module
    sys.modules.pop("comfy_execution.caching", None)


def _primitive(value):
    value_type = type(value)
    return ("primitive", value_type.__module__, value_type.__qualname__, value)


def test_unhashable_keeps_external_cache_self_unequal_marker(caching_module):
    caching, _ = caching_module
    sentinel = caching.Unhashable()

    assert sentinel.value != sentinel.value


def test_to_hashable_preserves_container_types(caching_module):
    caching, _ = caching_module

    assert caching.to_hashable({"a": 1}) == (
        "dict",
        ((_primitive("a"), _primitive(1)),),
    )
    assert caching.to_hashable(["a", 1]) == (
        "list",
        (_primitive("a"), _primitive(1)),
    )
    assert caching.to_hashable(("a", 1)) == (
        "tuple",
        (_primitive("a"), _primitive(1)),
    )


def test_to_hashable_canonicalizes_dict_insertion_order(caching_module):
    caching, _ = caching_module

    first = {"b": 2, "a": 1}
    second = {"a": 1, "b": 2}

    assert caching.to_hashable(first) == (
        "dict",
        (
            (_primitive("a"), _primitive(1)),
            (_primitive("b"), _primitive(2)),
        ),
    )
    assert caching.to_hashable(first) == caching.to_hashable(second)


def test_to_hashable_handles_shared_builtin_substructures(caching_module):
    caching, _ = caching_module
    shared = [{"value": 1}, {"value": 2}]

    result = caching.to_hashable([shared, shared])

    assert result[0] == "list"
    assert result[1][0] == result[1][1]


def test_to_hashable_preserves_primitive_types(caching_module):
    caching, _ = caching_module

    assert caching.to_hashable(1) != caching.to_hashable(True)
    assert caching.to_hashable(1) != caching.to_hashable(1.0)

    factories = [
        lambda value: [value],
        lambda value: (value,),
        lambda value: {"value": value},
        lambda value: {value},
        lambda value: frozenset({value}),
        lambda value: {value: "value"},
    ]
    for factory in factories:
        assert caching.to_hashable(factory(1)) != caching.to_hashable(factory(True))
        assert caching.to_hashable(factory(1)) != caching.to_hashable(factory(1.0))


def test_shallow_is_changed_signature_preserves_primitive_types(caching_module):
    caching, _ = caching_module

    assert caching._shallow_is_changed_signature(1) != caching._shallow_is_changed_signature(True)
    assert caching._shallow_is_changed_signature(1) != caching._shallow_is_changed_signature(1.0)


def test_to_hashable_fails_closed_for_opaque_value(caching_module):
    caching, _ = caching_module

    assert isinstance(caching.to_hashable([object()]), caching.Unhashable)


def test_to_hashable_fails_closed_for_opaque_dict_key(caching_module):
    caching, _ = caching_module

    assert isinstance(caching.to_hashable({_OpaqueValue(): 1}), caching.Unhashable)


def test_to_hashable_fails_closed_for_cycle(caching_module):
    caching, _ = caching_module
    value = []
    value.append(value)

    assert isinstance(caching.to_hashable(value), caching.Unhashable)


def test_to_hashable_fails_closed_at_depth_limit(caching_module):
    caching, _ = caching_module
    value = 1
    for _ in range(caching._MAX_SIGNATURE_DEPTH):
        value = [value]

    assert isinstance(caching.to_hashable(value), caching.Unhashable)


def test_to_hashable_stops_after_container_budget(caching_module):
    caching, _ = caching_module

    assert isinstance(caching.to_hashable([[1], [2]], max_nodes=2), caching.Unhashable)


def test_to_hashable_snapshots_list_before_recursive_descent(caching_module, monkeypatch):
    caching, _ = caching_module
    original = caching._canonicalize_signature_impl
    marker = ("marker",)
    values = [marker, 2]

    def mutating_canonicalize(obj, *args, **kwargs):
        if obj is marker:
            values[1] = 3
        return original(obj, *args, **kwargs)

    monkeypatch.setattr(caching, "_canonicalize_signature_impl", mutating_canonicalize)

    assert caching.to_hashable(values) == (
        "list",
        (("tuple", (_primitive("marker"),)), _primitive(2)),
    )
    assert values[1] == 3


def test_to_hashable_snapshots_dict_before_recursive_descent(caching_module, monkeypatch):
    caching, _ = caching_module
    original = caching._canonicalize_signature_impl
    marker = ("marker",)
    values = {"first": marker, "second": 2}

    def mutating_canonicalize(obj, *args, **kwargs):
        if obj is marker:
            values["second"] = 3
        return original(obj, *args, **kwargs)

    monkeypatch.setattr(caching, "_canonicalize_signature_impl", mutating_canonicalize)

    assert caching.to_hashable(values) == (
        "dict",
        (
            (_primitive("first"), ("tuple", (_primitive("marker"),))),
            (_primitive("second"), _primitive(2)),
        ),
    )
    assert values["second"] == 3


@pytest.mark.parametrize(
    "container_factory",
    [
        lambda marker: [marker],
        lambda marker: (marker,),
        lambda marker: {"key": marker},
        lambda marker: {marker},
        lambda marker: frozenset({marker}),
    ],
)
def test_to_hashable_fails_closed_on_runtimeerror(
    caching_module,
    monkeypatch,
    container_factory,
):
    caching, _ = caching_module
    original = caching._canonicalize_signature_impl
    marker = object()

    def raising_canonicalize(obj, *args, **kwargs):
        if obj is marker:
            raise RuntimeError("container changed during traversal")
        return original(obj, *args, **kwargs)

    monkeypatch.setattr(caching, "_canonicalize_signature_impl", raising_canonicalize)

    assert isinstance(caching.to_hashable(container_factory(marker)), caching.Unhashable)


def test_to_hashable_fails_closed_for_ambiguous_dict_ordering(caching_module, monkeypatch):
    caching, _ = caching_module
    original = caching._primitive_signature_sort_key

    def colliding_sort_key(obj):
        if obj in ("a", "b"):
            return ("COLLIDE",)
        return original(obj)

    monkeypatch.setattr(caching, "_primitive_signature_sort_key", colliding_sort_key)

    assert isinstance(caching.to_hashable({"a": 1, "b": 2}), caching.Unhashable)


@pytest.mark.parametrize("container_factory", [set, frozenset])
def test_to_hashable_fails_closed_for_ambiguous_unordered_values(
    caching_module,
    monkeypatch,
    container_factory,
):
    caching, _ = caching_module
    original = caching._primitive_signature_sort_key

    def colliding_sort_key(obj):
        if obj in ("a", "b"):
            return ("COLLIDE",)
        return original(obj)

    monkeypatch.setattr(caching, "_primitive_signature_sort_key", colliding_sort_key)

    assert isinstance(caching.to_hashable(container_factory({"a", "b"})), caching.Unhashable)


def test_shallow_is_changed_signature_accepts_structured_builtins(caching_module):
    caching, _ = caching_module

    assert caching._shallow_is_changed_signature([("seed", 42), {"cfg": 8}]) == (
        "is_changed_list",
        (
            ("tuple", (_primitive("seed"), _primitive(42))),
            ("dict", ((_primitive("cfg"), _primitive(8)),)),
        ),
    )


def test_shallow_is_changed_signature_fails_closed_for_opaque_payload(caching_module):
    caching, _ = caching_module

    assert isinstance(
        caching._shallow_is_changed_signature([_OpaqueValue()]),
        caching.Unhashable,
    )


def test_get_immediate_node_signature_fails_closed_for_opaque_input(caching_module, monkeypatch):
    caching, nodes_module = caching_module
    monkeypatch.setitem(nodes_module.NODE_CLASS_MAPPINGS, "UnitTestNode", _DummyNode)

    dynprompt = _FakeDynPrompt(
        {
            "node": {
                "class_type": "UnitTestNode",
                "inputs": {"value": _OpaqueValue()},
            }
        }
    )
    key_set = caching.CacheKeySetInputSignature(
        dynprompt,
        ["node"],
        _FakeIsChangedCache({"node": None}),
    )

    signature = asyncio.run(key_set.get_immediate_node_signature(dynprompt, "node", {}))

    assert isinstance(signature, caching.Unhashable)


def test_get_node_signature_propagates_unhashable_fragment(caching_module, monkeypatch):
    caching, nodes_module = caching_module
    monkeypatch.setitem(nodes_module.NODE_CLASS_MAPPINGS, "UnitTestNode", _DummyNode)

    changed = []
    changed.append(changed)
    dynprompt = _FakeDynPrompt(
        {
            "node": {
                "class_type": "UnitTestNode",
                "inputs": {"value": 5},
            }
        }
    )
    key_set = caching.CacheKeySetInputSignature(
        dynprompt,
        ["node"],
        _FakeIsChangedCache({"node": changed}),
    )

    signature = asyncio.run(key_set.get_node_signature(dynprompt, "node"))

    assert isinstance(signature, caching.Unhashable)


def test_get_immediate_node_signature_fails_closed_for_missing_node(caching_module):
    caching, _ = caching_module
    dynprompt = _FakeDynPrompt({})
    key_set = caching.CacheKeySetInputSignature(
        dynprompt,
        [],
        _FakeIsChangedCache({}),
    )

    signature = asyncio.run(key_set.get_immediate_node_signature(dynprompt, "missing", {}))

    assert isinstance(signature, caching.Unhashable)


def test_get_ordered_ancestry_keeps_two_value_contract(caching_module, monkeypatch):
    caching, nodes_module = caching_module
    monkeypatch.setitem(nodes_module.NODE_CLASS_MAPPINGS, "UnitTestNode", _DummyNode)

    dynprompt = _FakeDynPrompt(
        {
            "node": {
                "class_type": "UnitTestNode",
                "inputs": {"link": ["ancestor", 0]},
            },
            "ancestor": {
                "class_type": "UnitTestNode",
                "inputs": {},
            },
        }
    )
    key_set = caching.CacheKeySetInputSignature(
        dynprompt,
        ["node"],
        _FakeIsChangedCache({"node": None, "ancestor": None}),
    )

    ancestors, order_mapping = key_set.get_ordered_ancestry(dynprompt, "node")

    assert ancestors == ["ancestor"]
    assert order_mapping == {"ancestor": 0}


def test_get_node_signature_fails_closed_when_ancestry_snapshot_fails_once(
    caching_module,
    monkeypatch,
):
    caching, nodes_module = caching_module
    monkeypatch.setitem(nodes_module.NODE_CLASS_MAPPINGS, "UnitTestNode", _DummyNode)

    class RuntimeErrorOnceInputs(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._failed = False

        def items(self):
            if not self._failed:
                self._failed = True
                raise RuntimeError("container changed during traversal")
            return super().items()

    dynprompt = _FakeDynPrompt(
        {
            "node": {
                "class_type": "UnitTestNode",
                "inputs": RuntimeErrorOnceInputs({"link": ["ancestor", 0]}),
            }
        }
    )
    key_set = caching.CacheKeySetInputSignature(
        dynprompt,
        ["node"],
        _FakeIsChangedCache({"node": None}),
    )

    signature = asyncio.run(key_set.get_node_signature(dynprompt, "node"))

    assert isinstance(signature, caching.Unhashable)


@pytest.mark.parametrize(
    "mutation",
    ["replace", "mutate_in_place", "remove", "become_link"],
)
def test_get_node_signature_fails_closed_when_snapshotted_non_link_input_changes(
    caching_module,
    monkeypatch,
    mutation,
):
    caching, nodes_module = caching_module
    monkeypatch.setitem(nodes_module.NODE_CLASS_MAPPINGS, "UnitTestNode", _DummyNode)

    node_inputs = {"value": {"nested": [1]}}
    dynprompt = _FakeDynPrompt(
        {
            "node": {
                "class_type": "UnitTestNode",
                "inputs": node_inputs,
            },
            "ancestor": {
                "class_type": "UnitTestNode",
                "inputs": {},
            },
        }
    )
    key_set = caching.CacheKeySetInputSignature(
        dynprompt,
        ["node"],
        _FakeIsChangedCache({"node": None, "ancestor": None}),
    )
    original = key_set._get_immediate_node_signature

    async def mutate_before_immediate(
        current_dynprompt,
        current_node_id,
        ancestor_order_mapping,
        input_items,
    ):
        if current_node_id == "node":
            if mutation == "replace":
                node_inputs["value"] = {"nested": [2]}
            elif mutation == "mutate_in_place":
                node_inputs["value"]["nested"].append(2)
            elif mutation == "remove":
                del node_inputs["value"]
            else:
                node_inputs["value"] = ["ancestor", 0]
        return await original(
            current_dynprompt,
            current_node_id,
            ancestor_order_mapping,
            input_items,
        )

    monkeypatch.setattr(
        key_set,
        "_get_immediate_node_signature",
        mutate_before_immediate,
    )

    signature = asyncio.run(key_set.get_node_signature(dynprompt, "node"))

    assert isinstance(signature, caching.Unhashable)


def test_get_node_signature_fails_closed_when_snapshotted_link_mutates_in_place(
    caching_module,
    monkeypatch,
):
    caching, nodes_module = caching_module
    monkeypatch.setitem(nodes_module.NODE_CLASS_MAPPINGS, "UnitTestNode", _DummyNode)

    node_inputs = {"link": ["ancestor", 0]}
    dynprompt = _FakeDynPrompt(
        {
            "node": {
                "class_type": "UnitTestNode",
                "inputs": node_inputs,
            },
            "ancestor": {
                "class_type": "UnitTestNode",
                "inputs": {},
            },
        }
    )
    key_set = caching.CacheKeySetInputSignature(
        dynprompt,
        ["node"],
        _FakeIsChangedCache({"node": None, "ancestor": None}),
    )
    original = key_set._get_immediate_node_signature

    async def mutate_before_immediate(
        current_dynprompt,
        current_node_id,
        ancestor_order_mapping,
        input_items,
    ):
        if current_node_id == "node":
            node_inputs["link"][0] = "replacement"
        return await original(
            current_dynprompt,
            current_node_id,
            ancestor_order_mapping,
            input_items,
        )

    monkeypatch.setattr(
        key_set,
        "_get_immediate_node_signature",
        mutate_before_immediate,
    )

    signature = asyncio.run(key_set.get_node_signature(dynprompt, "node"))

    assert isinstance(signature, caching.Unhashable)


def test_get_node_signature_uses_stable_link_snapshot_when_live_link_is_unchanged(
    caching_module,
    monkeypatch,
):
    caching, nodes_module = caching_module
    monkeypatch.setitem(nodes_module.NODE_CLASS_MAPPINGS, "UnitTestNode", _DummyNode)

    dynprompt = _FakeDynPrompt(
        {
            "node": {
                "class_type": "UnitTestNode",
                "inputs": {"link": ["ancestor", 0]},
            },
            "ancestor": {
                "class_type": "UnitTestNode",
                "inputs": {},
            },
        }
    )
    key_set = caching.CacheKeySetInputSignature(
        dynprompt,
        ["node"],
        _FakeIsChangedCache({"node": None, "ancestor": None}),
    )

    signature = asyncio.run(key_set.get_node_signature(dynprompt, "node"))

    assert not isinstance(signature, caching.Unhashable)
    assert signature[0][-1] == ("link", ("ANCESTOR", 0, 0))
