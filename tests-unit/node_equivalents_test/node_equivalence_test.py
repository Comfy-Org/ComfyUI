"""Checks equivalent node replacements against recorded pack schemas and the live core schemas.

Snapshots under snapshots/<cnr_id>/ are /object_info entries captured from the pack at the
commit recorded in each file's pack_commit.
"""
import json
from pathlib import Path

import pytest

from app.node_equivalence import find_equivalence_problems, widget_slots
from comfy_api.latest._io import NodeReplace

SNAPSHOTS = Path(__file__).parent / "snapshots"


def pack_schema(pack: str, node: str) -> dict:
    return json.loads((SNAPSHOTS / pack / f"{node}.json").read_text())


@pytest.fixture(scope="module")
def core_schemas() -> dict:
    from comfy_extras import nodes_primitive
    classes = [nodes_primitive.String, nodes_primitive.StringMultiline, nodes_primitive.Int,
               nodes_primitive.Float, nodes_primitive.Boolean]
    return {info["name"]: info for info in (cls.GET_NODE_INFO_V1() for cls in classes)}


def equivalent(old_node_id: str, new_node_id: str, old_widget_ids: list[str], renames: dict[str, str]) -> NodeReplace:
    return NodeReplace(
        new_node_id=new_node_id,
        old_node_id=old_node_id,
        old_pack="comfyui-kjnodes",
        kind="equivalent",
        old_widget_ids=old_widget_ids,
        input_mapping=[{"new_id": new, "old_id": old} for old, new in renames.items()],
        output_mapping=[{"new_idx": 0, "old_idx": 0}],
    )


def problems_for(replacement: NodeReplace, core_schemas: dict) -> list[str]:
    old = pack_schema(replacement.old_pack, replacement.old_node_id)
    return find_equivalence_problems(replacement, old, core_schemas[replacement.new_node_id])


def test_admits_kjnodes_string_constant(core_schemas):
    replacement = equivalent("StringConstant", "PrimitiveString", ["string"], {"string": "value"})
    assert problems_for(replacement, core_schemas) == []


@pytest.mark.parametrize("replacement, expected", [
    pytest.param(
        equivalent("StringConstantMultiline", "PrimitiveStringMultiline", ["string", "strip_newlines"],
                   {"string": "value"}),
        "input 'strip_newlines' is not mapped",
        id="unmapped-widget",
    ),
    pytest.param(
        equivalent("INTConstant", "PrimitiveInt", ["value"], {"value": "value"}),
        "allows values above 'value' max",
        id="int-range-wider-than-core",
    ),
    pytest.param(
        equivalent("FloatConstant", "PrimitiveFloat", ["value"], {"value": "value"}),
        "rounds to 0.1",
        id="float-precision-coarser-than-pack",
    ),
    pytest.param(
        equivalent("StringConstant", "PrimitiveString", [], {"string": "value"}),
        "old_widget_ids [] must be ['string']",
        id="widget-ids-misaligned",
    ),
    pytest.param(
        equivalent("StringConstant", "PrimitiveInt", ["string"], {"string": "value"}),
        "input 'string' is STRING but 'value' is INT",
        id="type-mismatch",
    ),
    pytest.param(
        equivalent("StringConstant", "PrimitiveInt", ["string"], {"string": "value"}),
        "output 0 is STRING but new output 0 is INT",
        id="output-type-mismatch",
    ),
    pytest.param(
        equivalent("StringConstant", "PrimitiveString", ["string"], {"string": "missing"}),
        "which PrimitiveString does not have",
        id="unknown-target-input",
    ),
])
def test_rejects_lossy_entries(core_schemas, replacement, expected):
    problems = problems_for(replacement, core_schemas)
    assert any(expected in p for p in problems), problems


def test_rejects_unmapped_output(core_schemas):
    replacement = equivalent("StringConstant", "PrimitiveString", ["string"], {"string": "value"})
    replacement.output_mapping = [{"new_idx": 0, "old_idx": 1}]
    assert any("output 0 (STRING) is not mapped" in p for p in problems_for(replacement, core_schemas))


def test_rejects_set_value_over_mapped_input(core_schemas):
    replacement = equivalent("StringConstant", "PrimitiveString", ["string"], {"string": "value"})
    replacement.input_mapping.append({"new_id": "value", "set_value": ""})
    assert any("overrides a mapped value" in p for p in problems_for(replacement, core_schemas))


def test_rejects_deprecated_kind(core_schemas):
    replacement = NodeReplace(new_node_id="PrimitiveString", old_node_id="StringConstant",
                              old_widget_ids=["string"], input_mapping=[{"new_id": "value", "old_id": "string"}])
    old = pack_schema("comfyui-kjnodes", "StringConstant")
    assert find_equivalence_problems(replacement, old, core_schemas["PrimitiveString"]) == [
        "kind is 'deprecated', not 'equivalent'"
    ]


@pytest.mark.parametrize("info, expected", [
    ({"input": {"required": {"seed": ["INT", {}], "text": ["STRING", {}]}},
      "input_order": {"required": ["seed", "text"]}}, ["seed", "control_after_generate", "text"]),
    ({"input": {"required": {"n": ["INT", {"control_after_generate": False}], "image": ["IMAGE", {}]},
                "optional": {"x": ["FLOAT", {"forceInput": True}], "mode": [["a", "b"], {}]}},
      "input_order": {"required": ["n", "image"], "optional": ["x", "mode"]}}, ["n", "mode"]),
])
def test_widget_slots_follow_saved_widget_order(info, expected):
    assert widget_slots(info) == expected


def test_equivalent_requires_old_pack():
    with pytest.raises(ValueError, match="must set old_pack"):
        NodeReplace(new_node_id="PrimitiveString", old_node_id="StringConstant", kind="equivalent")
