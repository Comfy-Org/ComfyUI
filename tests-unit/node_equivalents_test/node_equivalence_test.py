"""Admission rule for "equivalent" node replacements, checked against recorded pack schemas.

An equivalent replacement must be lossless: every saved widget value and every output link
of the old node survives the swap unchanged. The check compares /object_info (V1) schemas,
so a pack node is checked from a snapshot under snapshots/<cnr_id>/, captured at the commit
recorded in each file's pack_commit. It cannot see a node's execute() body or its frontend
extension; those stay with review.
"""
import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from comfy_api.latest._io import NodeReplace
from comfy_extras import nodes_primitive, nodes_replacements

SNAPSHOTS = Path(__file__).parent / "snapshots"
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
CONTROL_SLOT = "control_after_generate"


def spec_options(spec: list) -> dict:
    return spec[1] if len(spec) > 1 else {}


def combo_options(spec: list) -> list | None:
    if isinstance(spec[0], list):
        return spec[0]
    if spec[0] == "COMBO":
        return spec[1]["options"]
    return None


def inputs(info: dict) -> dict:
    return {**info["input"].get("required", {}), **info["input"].get("optional", {})}


def widget_slots(info: dict) -> list[str]:
    """Positional order of a node's saved widgets_values, including the frontend's control widgets."""
    specs = inputs(info)
    slots = []
    for name in [*info["input_order"].get("required", []), *info["input_order"].get("optional", [])]:
        spec = specs[name]
        options = spec_options(spec)
        if not (isinstance(spec[0], list) or spec[0] in WIDGET_TYPES) or options.get("forceInput"):
            continue
        slots.append(name)
        control = options.get(CONTROL_SLOT)
        if control or (control is None and spec[0] == "INT" and name in ("seed", "noise_seed")):
            slots.append(CONTROL_SLOT)
            if combo_options(spec) is not None:
                slots.append("control_filter_list")
    return slots


def float_rounding(options: dict) -> float:
    """Granularity the frontend rounds an edited FLOAT widget to; 0 when it does not round."""
    step = options.get("step", 0.5)
    return options.get("round") or (10 ** math.floor(math.log10(step)) if step < 1 else 0)


def input_problems(old_id: str, new_id: str, old_spec: list, new_spec: list) -> list[str]:
    old_combo, new_combo = combo_options(old_spec), combo_options(new_spec)
    if old_combo is not None and new_combo is not None:
        return [] if set(old_combo) <= set(new_combo) else [f"combo input {old_id!r} has options missing from {new_id!r}"]
    if old_spec[0] != new_spec[0]:
        return [f"input {old_id!r} is {old_spec[0]} but {new_id!r} is {new_spec[0]}"]
    old, new = spec_options(old_spec), spec_options(new_spec)
    problems = []
    if old_spec[0] in ("INT", "FLOAT"):
        if "min" in new and old.get("min", -math.inf) < new["min"]:
            problems.append(f"input {old_id!r} allows values below {new_id!r} min {new['min']}")
        if "max" in new and old.get("max", math.inf) > new["max"]:
            problems.append(f"input {old_id!r} allows values above {new_id!r} max {new['max']}")
    if old_spec[0] == "FLOAT" and float_rounding(new) > float_rounding(old):
        problems.append(f"input {old_id!r} rounds to {float_rounding(old)} but {new_id!r} rounds to {float_rounding(new)}")
    if bool(old.get("dynamicPrompts")) != bool(new.get("dynamicPrompts")):
        problems.append(f"input {old_id!r} and {new_id!r} disagree on dynamicPrompts")
    return problems


def find_equivalence_problems(replacement: NodeReplace, old_info: dict, new_info: dict) -> list[str]:
    """Return every reason the replacement is not a lossless equivalent; empty when admitted."""
    problems = []
    old_slots, new_slots = widget_slots(old_info), widget_slots(new_info)
    if replacement.old_widget_ids != old_slots:
        problems.append(f"old_widget_ids {replacement.old_widget_ids} must be {old_slots}")

    old_inputs, new_inputs = inputs(old_info), inputs(new_info)
    mapping = replacement.input_mapping or []
    mapped_from = {m["old_id"]: m["new_id"] for m in mapping if "old_id" in m}
    set_targets = {m["new_id"] for m in mapping if "set_value" in m}

    for old_id, old_spec in old_inputs.items():
        new_id = mapped_from.get(old_id)
        if new_id is None:
            problems.append(f"input {old_id!r} is not mapped, so its value would be lost")
        elif new_id not in new_inputs:
            problems.append(f"input {old_id!r} maps to {new_id!r}, which {replacement.new_node_id} does not have")
        else:
            problems += input_problems(old_id, new_id, old_spec, new_inputs[new_id])

    if CONTROL_SLOT in old_slots and CONTROL_SLOT in new_slots and mapped_from.get(CONTROL_SLOT) != CONTROL_SLOT:
        problems.append("control_after_generate is not mapped, so the saved control mode would reset")

    for new_id in set_targets:
        if new_id in mapped_from.values():
            problems.append(f"set_value on {new_id!r} overrides a mapped value")
        elif new_id not in new_inputs and new_id not in new_slots:
            problems.append(f"set_value targets {new_id!r}, which {replacement.new_node_id} does not have")

    # The server builds the new node's inputs from the mapping alone, so every required input
    # needs a value the old node always sends: a required input or a widget.
    always_sent = set(old_info["input"].get("required", {})) | set(old_slots)
    sources = {new_id: old_id for old_id, new_id in mapped_from.items()}
    for new_id in new_info["input"].get("required", {}):
        if new_id not in set_targets and sources.get(new_id) not in always_sent:
            problems.append(f"required input {new_id!r} of {replacement.new_node_id} would be missing")

    output_map = {m["old_idx"]: m["new_idx"] for m in replacement.output_mapping or []}
    new_outputs = new_info["output"]
    for old_idx, old_type in enumerate(old_info["output"]):
        new_idx = output_map.get(old_idx, old_idx if replacement.output_mapping is None else None)
        if new_idx is None or new_idx >= len(new_outputs):
            problems.append(f"output {old_idx} ({old_type}) is not mapped, so its links would drop")
        elif new_outputs[new_idx] != old_type:
            problems.append(f"output {old_idx} is {old_type} but new output {new_idx} is {new_outputs[new_idx]}")
    return problems


@pytest.fixture(scope="module")
def core_schemas() -> dict:
    classes = [nodes_primitive.String, nodes_primitive.StringMultiline, nodes_primitive.Int,
               nodes_primitive.Float, nodes_primitive.Boolean]
    # Round-trip through JSON so the schemas have the shape /object_info serves.
    return {info["name"]: info for info in json.loads(json.dumps([cls.GET_NODE_INFO_V1() for cls in classes]))}


def problems_for(replacement: NodeReplace, core_schemas: dict) -> list[str]:
    old = json.loads((SNAPSHOTS / replacement.old_pack / f"{replacement.old_node_id}.json").read_text())
    return find_equivalence_problems(replacement, old, core_schemas[replacement.new_node_id])


def test_registered_equivalents_are_lossless(core_schemas, monkeypatch):
    registered = []

    async def register(entry):
        registered.append(entry)

    monkeypatch.setattr(nodes_replacements, "api", SimpleNamespace(node_replacement=SimpleNamespace(register=register)))
    asyncio.run(nodes_replacements.register_replacements())
    problems = {r.old_node_id: problems_for(r, core_schemas) for r in registered if r.kind == "equivalent"}
    assert problems and not any(problems.values()), problems


def kjnodes(old_node_id: str, new_node_id: str, old_widget_ids: list[str], renames: dict[str, str]) -> NodeReplace:
    return NodeReplace(
        new_node_id=new_node_id,
        old_node_id=old_node_id,
        old_pack="comfyui-kjnodes",
        kind="equivalent",
        old_widget_ids=old_widget_ids,
        input_mapping=[{"new_id": new, "old_id": old} for old, new in renames.items()],
        output_mapping=[{"new_idx": 0, "old_idx": 0}],
    )


@pytest.mark.parametrize("replacement, expected", [
    pytest.param(
        kjnodes("StringConstantMultiline", "PrimitiveStringMultiline", ["string", "strip_newlines"], {"string": "value"}),
        ["input 'strip_newlines' is not mapped"],
        id="unmapped-widget",
    ),
    pytest.param(
        kjnodes("INTConstant", "PrimitiveInt", ["value"], {"value": "value"}),
        ["allows values below 'value' min", "allows values above 'value' max"],
        id="int-range-wider-than-core",
    ),
    pytest.param(
        kjnodes("FloatConstant", "PrimitiveFloat", ["value"], {"value": "value"}),
        ["rounds to 1e-05 but 'value' rounds to 0.1"],
        id="float-rounding-coarser-than-pack",
    ),
    pytest.param(
        kjnodes("StringConstant", "PrimitiveString", [], {"string": "value"}),
        ["old_widget_ids [] must be ['string']"],
        id="widget-ids-misaligned",
    ),
    pytest.param(
        kjnodes("StringConstant", "PrimitiveInt", ["string"], {"string": "value"}),
        ["input 'string' is STRING but 'value' is INT", "output 0 is STRING but new output 0 is INT"],
        id="type-mismatch",
    ),
    pytest.param(
        kjnodes("StringConstant", "PrimitiveString", ["string"], {"string": "missing"}),
        ["which PrimitiveString does not have", "required input 'value' of PrimitiveString would be missing"],
        id="unknown-target-input",
    ),
    pytest.param(
        NodeReplace("PrimitiveString", "StringConstant", old_pack="comfyui-kjnodes", kind="equivalent",
                    old_widget_ids=["string"], input_mapping=[{"new_id": "value", "old_id": "string"},
                                                              {"new_id": "value", "set_value": ""}],
                    output_mapping=[{"new_idx": 0, "old_idx": 1}]),
        ["set_value on 'value' overrides a mapped value", "output 0 (STRING) is not mapped"],
        id="set-value-over-mapped-and-unmapped-output",
    ),
])
def test_rejects_lossy_kjnodes_entries(core_schemas, replacement, expected):
    problems = problems_for(replacement, core_schemas)
    for message in expected:
        assert any(message in p for p in problems), problems


def node(required: dict, optional: dict | None = None) -> dict:
    optional = optional or {}
    return {"input": {"required": required, "optional": optional},
            "input_order": {"required": list(required), "optional": list(optional)}, "output": ["INT"]}


@pytest.mark.parametrize("old_info, new_info, expected", [
    pytest.param(node({"mode": [["a", "b"], {}]}), node({"mode": ["COMBO", {"options": ["a", "b", "c"]}]}),
                 [], id="legacy-combo-into-wider-v3-combo"),
    pytest.param(node({"mode": ["COMBO", {"options": ["a", "b"]}]}), node({"mode": [["a"], {}]}),
                 ["combo input 'mode' has options missing from 'mode'"], id="v3-combo-into-narrower-combo"),
    pytest.param(node({}, {"mode": ["IMAGE", {}]}), node({"mode": ["IMAGE", {}]}),
                 ["required input 'mode' of New would be missing"], id="optional-link-into-required"),
])
def test_input_rules(old_info, new_info, expected):
    replacement = NodeReplace("New", "Old", old_pack="pack", kind="equivalent", old_widget_ids=widget_slots(old_info),
                              input_mapping=[{"new_id": "mode", "old_id": "mode"}])
    assert find_equivalence_problems(replacement, old_info, new_info) == expected


@pytest.mark.parametrize("info, expected", [
    (node({"seed": ["INT", {}], "text": ["STRING", {}]}), ["seed", CONTROL_SLOT, "text"]),
    (node({"n": ["INT", {"control_after_generate": False}], "image": ["IMAGE", {}]},
          {"x": ["FLOAT", {"forceInput": True}], "mode": [["a", "b"], {}]}), ["n", "mode"]),
    (node({"c": ["COMBO", {"options": ["a"], "control_after_generate": True}],
           "f": ["FLOAT", {"control_after_generate": True}]}),
     ["c", CONTROL_SLOT, "control_filter_list", "f", CONTROL_SLOT]),
])
def test_widget_slots_follow_saved_widget_order(info, expected):
    assert widget_slots(info) == expected


def test_equivalent_requires_old_pack():
    with pytest.raises(ValueError, match="must set old_pack"):
        NodeReplace(new_node_id="PrimitiveString", old_node_id="StringConstant", kind="equivalent")
