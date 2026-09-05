"""Unit tests for :mod:`comfy_execution.type_resolver`.

These tests stand up a small in-memory ``NODE_CLASS_MAPPINGS`` for the test
node classes (V1 and V3) and a fake DynamicPrompt-like dict, then verify the
resolver's behaviour for:

  * Static V1 ``RETURN_TYPES`` resolution.
  * V1 wildcard outputs (must yield ``AnyType`` and warn once).
  * V3 ``MatchType`` chains resolved via the downstream node's bound inputs.
  * ``MatchType`` with no upstream bound (fall back to ``AnyType`` + warn).
  * ``MatchType`` cycles (termination at ``AnyType`` + warn, no recursion blow-up).
  * Deep chains capped by ``MAX_RESOLVE_DEPTH``.
  * Input-type resolution for both literal values and links.
  * Effective slot io_type peeling for ``Autogrow`` (returns the wrapped type).
  * ``compute_live_input_types`` produces the right shape.
  * Cache invalidation.

The tests deliberately patch ``nodes.NODE_CLASS_MAPPINGS`` so they don't need
the whole ComfyUI bootstrap.
"""

from __future__ import annotations

import logging
import sys
import types as _pytypes

import pytest


# ---------------------------------------------------------------------------
# Lightweight V1 test node factory
# ---------------------------------------------------------------------------

def _v1_node(return_types: tuple[str, ...], input_types_dict: dict | None = None,
             output_is_list: tuple[bool, ...] | None = None):
    """Build a V1 node class with the given RETURN_TYPES / INPUT_TYPES()."""
    if input_types_dict is None:
        input_types_dict = {"required": {}}

    class _V1:
        RETURN_TYPES = return_types
        if output_is_list is not None:
            OUTPUT_IS_LIST = output_is_list

        @classmethod
        def INPUT_TYPES(cls):
            return input_types_dict

    return _V1


# ---------------------------------------------------------------------------
# Fixture: install fake nodes module before importing the resolver
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_nodes_module():
    """Install a synthetic ``nodes`` module with an empty mappings dict.

    Yields the mappings dict so tests can populate it per case. Cleans up
    afterwards. We also have to make sure comfy_execution.type_resolver picks
    up our fake module on its local import.
    """
    real_nodes = sys.modules.get("nodes")
    fake = _pytypes.ModuleType("nodes")
    fake.NODE_CLASS_MAPPINGS = {}
    sys.modules["nodes"] = fake
    try:
        yield fake.NODE_CLASS_MAPPINGS
    finally:
        if real_nodes is not None:
            sys.modules["nodes"] = real_nodes
        else:
            del sys.modules["nodes"]


@pytest.fixture
def TypeResolver(fake_nodes_module):
    # Late import so it picks up our fake `nodes` module.
    from comfy_execution.type_resolver import TypeResolver as TR
    return TR


# ---------------------------------------------------------------------------
# V1 resolution
# ---------------------------------------------------------------------------

def test_v1_static_return_types_resolves(fake_nodes_module, TypeResolver):
    fake_nodes_module["AddNode"] = _v1_node(("INT",))
    prompt = {"n1": {"class_type": "AddNode", "inputs": {}}}
    r = TypeResolver(prompt)
    assert r.resolve_output_type("n1", 0) == "INT"


def test_v1_wildcard_warns_once_and_returns_any(fake_nodes_module, TypeResolver, caplog):
    fake_nodes_module["WildNode"] = _v1_node(("*",))
    prompt = {"n1": {"class_type": "WildNode", "inputs": {}}}
    r = TypeResolver(prompt)
    with caplog.at_level(logging.WARNING, logger="root"):
        assert r.resolve_output_type("n1", 0) == "*"
        # second call should still return * but not produce a second warning
        assert r.resolve_output_type("n1", 0) == "*"
    warnings = [rec for rec in caplog.records if "TypeResolver" in rec.message]
    assert len(warnings) == 1, f"expected exactly one warning, got {warnings}"


def test_unknown_node_returns_any(fake_nodes_module, TypeResolver):
    prompt = {"n1": {"class_type": "NopeNode", "inputs": {}}}
    r = TypeResolver(prompt)
    assert r.resolve_output_type("n1", 0) == "*"


def test_out_of_range_slot_returns_any(fake_nodes_module, TypeResolver):
    fake_nodes_module["AddNode"] = _v1_node(("INT",))
    prompt = {"n1": {"class_type": "AddNode", "inputs": {}}}
    r = TypeResolver(prompt)
    assert r.resolve_output_type("n1", 5) == "*"


def test_missing_node_returns_any(fake_nodes_module, TypeResolver):
    fake_nodes_module["AddNode"] = _v1_node(("INT",))
    prompt = {"n1": {"class_type": "AddNode", "inputs": {}}}
    r = TypeResolver(prompt)
    assert r.resolve_output_type("nonexistent", 0) == "*"


# ---------------------------------------------------------------------------
# is_output_list / is_input_list
# ---------------------------------------------------------------------------

def test_is_output_list(fake_nodes_module, TypeResolver):
    fake_nodes_module["ListNode"] = _v1_node(("IMAGE", "MASK"), output_is_list=(True, False))
    prompt = {"n1": {"class_type": "ListNode", "inputs": {}}}
    r = TypeResolver(prompt)
    assert r.is_output_list("n1", 0) is True
    assert r.is_output_list("n1", 1) is False


def test_is_input_list_follows_link(fake_nodes_module, TypeResolver):
    fake_nodes_module["ListNode"] = _v1_node(("IMAGE",), output_is_list=(True,))
    fake_nodes_module["Consumer"] = _v1_node(
        ("INT",),
        {"required": {"img": ("IMAGE",)}},
    )
    prompt = {
        "src": {"class_type": "ListNode", "inputs": {}},
        "dst": {"class_type": "Consumer", "inputs": {"img": ["src", 0]}},
    }
    r = TypeResolver(prompt)
    assert r.is_input_list("dst", "img") is True


# ---------------------------------------------------------------------------
# V3 MatchType resolution
# ---------------------------------------------------------------------------

def _make_switch_node_class():
    """Build a V3 Switch-like node with MatchType inputs/outputs."""
    from comfy_api.latest import io

    class Switch(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            template = io.MatchType.Template("switch")
            return io.Schema(
                node_id="TestSwitch",
                inputs=[
                    io.Boolean.Input("switch"),
                    io.MatchType.Input("on_false", template=template, optional=True),
                    io.MatchType.Input("on_true", template=template, optional=True),
                ],
                outputs=[io.MatchType.Output(template=template)],
            )

        @classmethod
        def execute(cls, switch, on_false=None, on_true=None):
            return io.NodeOutput(on_true if switch else on_false)

    # Force schema computation so SCHEMA / RETURN_TYPES are populated.
    Switch.GET_SCHEMA()
    return Switch


def test_matchtype_resolves_to_upstream_concrete(fake_nodes_module, TypeResolver):
    fake_nodes_module["TestSwitch"] = _make_switch_node_class()
    fake_nodes_module["ImageSrc"] = _v1_node(("IMAGE",))
    prompt = {
        "img": {"class_type": "ImageSrc", "inputs": {}},
        "sw": {
            "class_type": "TestSwitch",
            "inputs": {"switch": True, "on_true": ["img", 0]},
        },
    }
    r = TypeResolver(prompt)
    assert r.resolve_output_type("sw", 0) == "IMAGE"


def test_matchtype_first_concrete_wins(fake_nodes_module, TypeResolver):
    fake_nodes_module["TestSwitch"] = _make_switch_node_class()
    fake_nodes_module["ImageSrc"] = _v1_node(("IMAGE",))
    fake_nodes_module["LatentSrc"] = _v1_node(("LATENT",))
    prompt = {
        "img": {"class_type": "ImageSrc", "inputs": {}},
        "lat": {"class_type": "LatentSrc", "inputs": {}},
        "sw": {
            "class_type": "TestSwitch",
            "inputs": {
                "switch": False,
                "on_false": ["img", 0],   # listed first in schema → wins
                "on_true":  ["lat", 0],
            },
        },
    }
    r = TypeResolver(prompt)
    assert r.resolve_output_type("sw", 0) == "IMAGE"


def test_matchtype_no_bound_input_returns_any(fake_nodes_module, TypeResolver, caplog):
    fake_nodes_module["TestSwitch"] = _make_switch_node_class()
    prompt = {"sw": {"class_type": "TestSwitch", "inputs": {"switch": True}}}
    r = TypeResolver(prompt)
    with caplog.at_level(logging.WARNING, logger="root"):
        assert r.resolve_output_type("sw", 0) == "*"
    assert any("MatchType" in rec.message for rec in caplog.records)


def test_matchtype_skips_wildcard_input(fake_nodes_module, TypeResolver):
    """If the first matched input resolves to AnyType, the resolver tries the next."""
    fake_nodes_module["TestSwitch"] = _make_switch_node_class()
    fake_nodes_module["WildNode"] = _v1_node(("*",))
    fake_nodes_module["ImageSrc"] = _v1_node(("IMAGE",))
    prompt = {
        "wild": {"class_type": "WildNode", "inputs": {}},
        "img": {"class_type": "ImageSrc", "inputs": {}},
        "sw": {
            "class_type": "TestSwitch",
            "inputs": {
                "switch": True,
                "on_false": ["wild", 0],
                "on_true":  ["img", 0],
            },
        },
    }
    r = TypeResolver(prompt)
    assert r.resolve_output_type("sw", 0) == "IMAGE"


def test_matchtype_cycle_terminates_at_any(fake_nodes_module, TypeResolver):
    """Two switches that feed each other must not recurse forever."""
    fake_nodes_module["TestSwitch"] = _make_switch_node_class()
    prompt = {
        "a": {"class_type": "TestSwitch", "inputs": {"switch": True, "on_true": ["b", 0]}},
        "b": {"class_type": "TestSwitch", "inputs": {"switch": True, "on_true": ["a", 0]}},
    }
    r = TypeResolver(prompt)
    # Must not raise / recurse forever; both resolve to AnyType.
    assert r.resolve_output_type("a", 0) == "*"
    assert r.resolve_output_type("b", 0) == "*"


def test_matchtype_chain_resolves_through(fake_nodes_module, TypeResolver):
    """A → B → C → IMAGE: chain must walk all the way."""
    fake_nodes_module["TestSwitch"] = _make_switch_node_class()
    fake_nodes_module["ImageSrc"] = _v1_node(("IMAGE",))
    prompt = {
        "src": {"class_type": "ImageSrc", "inputs": {}},
        "a": {"class_type": "TestSwitch", "inputs": {"switch": True, "on_true": ["src", 0]}},
        "b": {"class_type": "TestSwitch", "inputs": {"switch": True, "on_true": ["a", 0]}},
        "c": {"class_type": "TestSwitch", "inputs": {"switch": True, "on_true": ["b", 0]}},
    }
    r = TypeResolver(prompt)
    assert r.resolve_output_type("c", 0) == "IMAGE"


# ---------------------------------------------------------------------------
# Input resolution and effective io_type peeling
# ---------------------------------------------------------------------------

def test_resolve_input_type_literal_uses_declared(fake_nodes_module, TypeResolver):
    fake_nodes_module["Sink"] = _v1_node(("INT",), {"required": {"steps": ("INT",)}})
    prompt = {"n1": {"class_type": "Sink", "inputs": {"steps": 20}}}
    r = TypeResolver(prompt)
    assert r.resolve_input_type("n1", "steps") == "INT"


def test_resolve_input_type_link(fake_nodes_module, TypeResolver):
    fake_nodes_module["Src"] = _v1_node(("LATENT",))
    fake_nodes_module["Sink"] = _v1_node(("INT",), {"required": {"x": ("*",)}})
    prompt = {
        "src": {"class_type": "Src", "inputs": {}},
        "sink": {"class_type": "Sink", "inputs": {"x": ["src", 0]}},
    }
    r = TypeResolver(prompt)
    assert r.resolve_input_type("sink", "x") == "LATENT"


def test_effective_slot_type_peels_autogrow(fake_nodes_module, TypeResolver):
    from comfy_api.latest import io

    class AutogrowImg(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            template = io.Autogrow.TemplatePrefix(
                input=io.Image.Input("img"),
                prefix="img",
                min=1,
            )
            return io.Schema(
                node_id="AutogrowImg",
                inputs=[io.Autogrow.Input("imgs", template=template)],
                outputs=[io.Image.Output()],
            )

        @classmethod
        def execute(cls, imgs):
            return io.NodeOutput(None)

    AutogrowImg.GET_SCHEMA()
    fake_nodes_module["AutogrowImg"] = AutogrowImg
    prompt = {"n1": {"class_type": "AutogrowImg", "inputs": {}}}
    r = TypeResolver(prompt)
    # The user-facing element type, not the autogrow wrapper.
    assert r.get_declared_slot_io_type("n1", "imgs") == "IMAGE"


# ---------------------------------------------------------------------------
# compute_live_input_types
# ---------------------------------------------------------------------------

def test_effective_slot_type_on_v3_plain_input(fake_nodes_module, TypeResolver):
    """V3 input that is neither Autogrow/DynamicSlot/DynamicCombo must still resolve.

    Regression test: importing ``io`` from the public re-export skipped
    ``DynamicSlot``, so an ``isinstance`` chain in ``_effective_io_type`` raised
    ``AttributeError`` the first time it ran against a plain V3 input.
    """
    from comfy_api.latest import io

    class BoolSink(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="BoolSink",
                inputs=[io.Boolean.Input("flag")],
                outputs=[io.Boolean.Output()],
            )

        @classmethod
        def execute(cls, flag):
            return io.NodeOutput(flag)

    BoolSink.GET_SCHEMA()
    fake_nodes_module["BoolSink"] = BoolSink
    prompt = {"n": {"class_type": "BoolSink", "inputs": {"flag": True}}}
    r = TypeResolver(prompt)
    assert r.get_declared_slot_io_type("n", "flag") == "BOOLEAN"
    assert r.compute_live_input_types("n") == {"flag": "BOOLEAN"}


def test_effective_slot_type_peels_dynamic_slot(fake_nodes_module, TypeResolver):
    """A DynamicSlot input reports its auto-derived slotType (union of `when` types)."""
    from comfy_api.latest import _io as io

    class DSNode(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="DSNode",
                inputs=[
                    io.DynamicSlot.Input("slot", options=[
                        io.DynamicSlot.Option(when=io.Image, inputs=[]),
                        io.DynamicSlot.Option(when=io.Latent, inputs=[]),
                    ]),
                ],
                outputs=[io.String.Output()],
            )

        @classmethod
        def execute(cls, **kwargs):
            return io.NodeOutput("")

    DSNode.GET_SCHEMA()
    fake_nodes_module["DSNode"] = DSNode
    prompt = {"n": {"class_type": "DSNode", "inputs": {}}}
    r = TypeResolver(prompt)
    assert r.get_declared_slot_io_type("n", "slot") == "IMAGE,LATENT"


def test_compute_live_input_types_mixes_links_and_literals(fake_nodes_module, TypeResolver):
    fake_nodes_module["Src"] = _v1_node(("MODEL",))
    fake_nodes_module["Sink"] = _v1_node(
        ("INT",),
        {"required": {"model": ("MODEL",), "steps": ("INT",)}},
    )
    prompt = {
        "src": {"class_type": "Src", "inputs": {}},
        "sink": {
            "class_type": "Sink",
            "inputs": {"model": ["src", 0], "steps": 20},
        },
    }
    r = TypeResolver(prompt)
    assert r.compute_live_input_types("sink") == {"model": "MODEL", "steps": "INT"}


# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------

def test_invalidate_clears_cache(fake_nodes_module, TypeResolver):
    fake_nodes_module["Src"] = _v1_node(("IMAGE",))
    prompt = {"n1": {"class_type": "Src", "inputs": {}}}
    r = TypeResolver(prompt)
    assert r.resolve_output_type("n1", 0) == "IMAGE"
    # Mutate the underlying class and invalidate; the resolver must re-read.
    fake_nodes_module["Src"] = _v1_node(("LATENT",))
    r.invalidate()
    assert r.resolve_output_type("n1", 0) == "LATENT"


# ---------------------------------------------------------------------------
# Malformed input robustness
# ---------------------------------------------------------------------------

def test_malformed_link_does_not_crash(fake_nodes_module, TypeResolver):
    """A link with a non-int slot index must not raise; resolver returns AnyType."""
    fake_nodes_module["Src"] = _v1_node(("IMAGE",))
    fake_nodes_module["Sink"] = _v1_node(("INT",), {"required": {"x": ("*",)}})
    prompt = {
        "src": {"class_type": "Src", "inputs": {}},
        # slot index sent as a string (common API JSON mistake)
        "sink": {"class_type": "Sink", "inputs": {"x": ["src", "0"]}},
    }
    r = TypeResolver(prompt)
    # Falls back to declared slot type (still "*"), no exception.
    assert r.resolve_input_type("sink", "x") == "*"


def test_malformed_link_wrong_arity_does_not_crash(fake_nodes_module, TypeResolver):
    fake_nodes_module["Src"] = _v1_node(("IMAGE",))
    fake_nodes_module["Sink"] = _v1_node(("INT",), {"required": {"x": ("*",)}})
    prompt = {
        "src": {"class_type": "Src", "inputs": {}},
        "sink": {"class_type": "Sink", "inputs": {"x": ["src"]}},  # arity 1
    }
    r = TypeResolver(prompt)
    assert r.resolve_input_type("sink", "x") == "*"


def test_direct_resolve_output_type_with_bad_slot_idx_returns_any(fake_nodes_module, TypeResolver):
    fake_nodes_module["Src"] = _v1_node(("IMAGE",))
    prompt = {"src": {"class_type": "Src", "inputs": {}}}
    r = TypeResolver(prompt)
    # type-wise these should be unreachable through normal validation but the
    # resolver must still degrade gracefully.
    assert r.resolve_output_type("src", "0") == "*"
    assert r.resolve_output_type("src", True) == "*"  # bool is a subclass of int
    assert r.is_output_list("src", "0") is False


def test_non_string_class_type_returns_any(fake_nodes_module, TypeResolver):
    prompt = {"n1": {"class_type": 42, "inputs": {}}}
    r = TypeResolver(prompt)
    assert r.resolve_output_type("n1", 0) == "*"


def test_invalidate_node_only_clears_that_node(fake_nodes_module, TypeResolver):
    fake_nodes_module["SrcA"] = _v1_node(("IMAGE",))
    fake_nodes_module["SrcB"] = _v1_node(("LATENT",))
    prompt = {
        "a": {"class_type": "SrcA", "inputs": {}},
        "b": {"class_type": "SrcB", "inputs": {}},
    }
    r = TypeResolver(prompt)
    r.resolve_output_type("a", 0)
    r.resolve_output_type("b", 0)
    fake_nodes_module["SrcA"] = _v1_node(("MASK",))
    r.invalidate_node("a")
    assert r.resolve_output_type("a", 0) == "MASK"
    # b's cached result survives even though SrcB was unchanged
    assert ("b", 0) in r._output_cache
