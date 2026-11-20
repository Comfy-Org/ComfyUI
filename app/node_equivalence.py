"""Admission rule for "equivalent" node replacements.

An equivalent replacement must be lossless: every saved widget value and every output
link of the old node survives the swap unchanged. The check compares the replacement
against the old and new node schemas in the /object_info (V1) format, so a pack node can
be checked from a recorded snapshot without installing the pack.

The check cannot see a node's execute() body or its frontend extension; those stay with
review.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from comfy_api.latest._io_public import NodeReplace

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
CONTROL_SLOT = "control_after_generate"
SEED_INPUT_NAMES = {"seed", "noise_seed"}


def _inputs(info: dict[str, Any]) -> dict[str, list]:
    groups = info.get("input", {})
    return {**groups.get("required", {}), **groups.get("optional", {})}


def _ordered_input_names(info: dict[str, Any]) -> list[str]:
    order = info.get("input_order", {})
    return [*order.get("required", []), *order.get("optional", [])]


def _spec_type(spec: list) -> str | list:
    return spec[0]


def _spec_options(spec: list) -> dict[str, Any]:
    return spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}


def _is_widget(spec: list) -> bool:
    spec_type = _spec_type(spec)
    is_widget_type = isinstance(spec_type, list) or spec_type in WIDGET_TYPES
    return is_widget_type and not _spec_options(spec).get("forceInput", False)


def _has_control_slot(name: str, spec: list) -> bool:
    if _spec_type(spec) != "INT":
        return False
    declared = _spec_options(spec).get(CONTROL_SLOT)
    return bool(declared) or (declared is None and name in SEED_INPUT_NAMES)


def widget_slots(info: dict[str, Any]) -> list[str]:
    """Positional order of a node's saved widgets_values.

    The frontend stores one value per widget input in input order, and inserts a
    control_after_generate value after INT inputs that declare it or are named seed/noise_seed.
    """
    inputs = _inputs(info)
    slots: list[str] = []
    for name in _ordered_input_names(info):
        spec = inputs[name]
        if not _is_widget(spec):
            continue
        slots.append(name)
        if _has_control_slot(name, spec):
            slots.append(CONTROL_SLOT)
    return slots


def _range_problems(old_id: str, new_id: str, old_options: dict, new_options: dict) -> list[str]:
    problems = []
    new_min, new_max = new_options.get("min"), new_options.get("max")
    old_min, old_max = old_options.get("min"), old_options.get("max")
    if new_min is not None and (old_min is None or old_min < new_min):
        problems.append(f"input {old_id!r} allows values below {new_id!r} min {new_min} (old min {old_min})")
    if new_max is not None and (old_max is None or old_max > new_max):
        problems.append(f"input {old_id!r} allows values above {new_id!r} max {new_max} (old max {old_max})")
    return problems


def _float_precision(options: dict) -> float | None:
    """Granularity the frontend rounds a FLOAT widget to: round, else step; 0 means unrounded."""
    rounding = options.get("round")
    if rounding is False:
        return 0
    if isinstance(rounding, (int, float)) and not isinstance(rounding, bool):
        return rounding
    return options.get("step")


def _precision_problems(old_id: str, new_id: str, old_options: dict, new_options: dict) -> list[str]:
    old_precision, new_precision = _float_precision(old_options), _float_precision(new_options)
    if not new_precision:
        return []
    if old_precision is None or old_precision == 0 or new_precision > old_precision:
        return [f"input {old_id!r} keeps precision {old_precision} but {new_id!r} rounds to {new_precision}"]
    return []


def _input_problems(old_id: str, new_id: str, old_spec: list, new_spec: list) -> list[str]:
    old_type, new_type = _spec_type(old_spec), _spec_type(new_spec)
    if isinstance(old_type, list) or isinstance(new_type, list):
        if not (isinstance(old_type, list) and isinstance(new_type, list) and set(old_type) <= set(new_type)):
            return [f"combo input {old_id!r} has options missing from {new_id!r}"]
        return []
    if old_type != new_type:
        return [f"input {old_id!r} is {old_type} but {new_id!r} is {new_type}"]
    old_options, new_options = _spec_options(old_spec), _spec_options(new_spec)
    problems = []
    if old_type in ("INT", "FLOAT"):
        problems += _range_problems(old_id, new_id, old_options, new_options)
    if old_type == "FLOAT":
        problems += _precision_problems(old_id, new_id, old_options, new_options)
    if bool(old_options.get("dynamicPrompts")) != bool(new_options.get("dynamicPrompts")):
        problems.append(f"input {old_id!r} and {new_id!r} disagree on dynamicPrompts")
    return problems


def _output_problems(replacement: NodeReplace, old_info: dict, new_info: dict) -> list[str]:
    old_outputs, new_outputs = old_info.get("output", []), new_info.get("output", [])
    output_map = {m["old_idx"]: m["new_idx"] for m in replacement.output_mapping or []}
    problems = []
    for old_idx, old_type in enumerate(old_outputs):
        new_idx = output_map.get(old_idx, old_idx if replacement.output_mapping is None else None)
        if new_idx is None or new_idx >= len(new_outputs):
            problems.append(f"output {old_idx} ({old_type}) is not mapped, so its links would drop")
        elif new_outputs[new_idx] != old_type:
            problems.append(f"output {old_idx} is {old_type} but new output {new_idx} is {new_outputs[new_idx]}")
    return problems


def find_equivalence_problems(replacement: NodeReplace, old_info: dict[str, Any], new_info: dict[str, Any]) -> list[str]:
    """Return every reason the replacement is not a lossless equivalent; empty when admitted."""
    problems: list[str] = []
    if replacement.kind != "equivalent":
        problems.append(f"kind is {replacement.kind!r}, not 'equivalent'")

    expected_slots = widget_slots(old_info)
    if replacement.old_widget_ids != expected_slots:
        problems.append(f"old_widget_ids {replacement.old_widget_ids} must be {expected_slots}")

    old_inputs, new_inputs = _inputs(old_info), _inputs(new_info)
    new_slots = set(widget_slots(new_info))
    mapping = replacement.input_mapping or []
    mapped_from = {m["old_id"]: m["new_id"] for m in mapping if "old_id" in m}
    mapped_targets = set(mapped_from.values())

    for old_id, old_spec in old_inputs.items():
        new_id = mapped_from.get(old_id)
        if new_id is None:
            problems.append(f"input {old_id!r} is not mapped, so its value would be lost")
        elif new_id not in new_inputs:
            problems.append(f"input {old_id!r} maps to {new_id!r}, which {replacement.new_node_id} does not have")
        else:
            problems += _input_problems(old_id, new_id, old_spec, new_inputs[new_id])

    for slot in expected_slots:
        if slot == CONTROL_SLOT and CONTROL_SLOT in new_slots and mapped_from.get(CONTROL_SLOT) != CONTROL_SLOT:
            problems.append("control_after_generate is not mapped, so the saved control mode would reset")

    for m in mapping:
        if "set_value" not in m:
            continue
        if m["new_id"] in mapped_targets:
            problems.append(f"set_value on {m['new_id']!r} overrides a mapped value")
        elif m["new_id"] not in new_inputs and m["new_id"] not in new_slots:
            problems.append(f"set_value targets {m['new_id']!r}, which {replacement.new_node_id} does not have")

    problems += _output_problems(replacement, old_info, new_info)
    return problems
