"""Check node input schemas for API-format workflow compatibility.

API-format prompts are frozen at the time they were saved, so a node's input
schema is a public contract. Two ways we break it today:

- Adding a required input to an existing node. Every stored prompt for that node
  is rejected by validation with `required_input_missing` (execution.py).
- Adding an optional input that the node cannot actually run without. Validation
  passes, then `get_input_data` never passes the key and the call raises
  TypeError at execution time.

An optional input has to be safe to leave out one of two ways. With a default in
its schema it just does not have to be provided, and execution fills it in. With
no default in its schema it is a nullable value, the not-connected case, and the
entry function has to carry the default itself.

`snapshot` dumps the input surface of every registered node, `diff` compares two
snapshots for breaking changes, and `unsafe-optionals` checks a single snapshot
for optional inputs that cannot actually be omitted.
"""

import argparse
import asyncio
import inspect
import json
import sys

from comfy.cli_args import args as comfy_args

comfy_args.cpu = True  # reading schemas never touches a device, and CI torch has no CUDA

import nodes
from comfy_api.internal import _ComfyNodeInternal


def report(line):
    sys.stdout.write(f"{line}\n")


def input_spec(node_class):
    if issubclass(node_class, _ComfyNodeInternal):
        return node_class.GET_NODE_INFO_V1()["input"]
    return node_class.INPUT_TYPES()


def entry_function(node_class):
    """The function that actually receives the node's inputs.

    V3 nodes expose FUNCTION == "EXECUTE_NORMALIZED", a (*args, **kwargs)
    trampoline, so inspecting FUNCTION would say every input is omittable.
    """
    if issubclass(node_class, _ComfyNodeInternal):
        return getattr(node_class, "execute", None)
    function_name = getattr(node_class, "FUNCTION", None)
    if function_name is None:
        return None
    return getattr(node_class, function_name, None)


def omittable_inputs(node_class):
    """Input names the entry function can be called without, or None if all can."""
    function = entry_function(node_class)
    if function is None:
        return set()
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return None
    if any(p.kind is p.VAR_KEYWORD for p in signature.parameters.values()):
        return None
    omittable = set()
    for name, parameter in signature.parameters.items():
        if parameter.default is not inspect.Parameter.empty:
            omittable.add(name)
    return omittable


def jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return repr(value)


def node_snapshot(node_class):
    spec = input_spec(node_class)
    omittable = omittable_inputs(node_class)
    inputs = {}
    for category in ("required", "optional"):
        for name, definition in (spec.get(category) or {}).items():
            if not isinstance(definition, (list, tuple)) or len(definition) == 0:
                continue
            input_type = definition[0]
            options = definition[1] if len(definition) > 1 and isinstance(definition[1], dict) else {}
            entry = {
                "category": category,
                "type": "COMBO" if isinstance(input_type, list) else input_type,
                "omittable": omittable is None or name in omittable,
            }
            if isinstance(input_type, list):
                entry["combo_options"] = [jsonable(o) for o in input_type]
            if "default" in options:
                entry["default"] = jsonable(options["default"])
            inputs[name] = entry
    return {
        "module": getattr(node_class, "RELATIVE_PYTHON_MODULE", "nodes"),
        "inputs": inputs,
    }


def collect_snapshot():
    # A node that cannot be read is left out of the snapshot, which would hide any
    # breaking change to it, so let it fail here instead.
    asyncio.run(nodes.init_extra_nodes(init_api_nodes=True))
    return {name: node_snapshot(node_class) for name, node_class in nodes.NODE_CLASS_MAPPINGS.items()}


def accepted_types(input_type):
    """The set of type names an input accepts.

    Multi-type inputs are stored as a comma separated string, and Combo and
    DynamicCombo are the same contract as far as a stored prompt is concerned.
    """
    if input_type == "COMFY_DYNAMICCOMBO_V3":
        input_type = "COMBO"
    return set(input_type.split(","))


def diff_snapshots(base, head):
    """Breaking changes between two snapshots, as human-readable lines."""
    breaks = []
    for node, head_node in sorted(head.items()):
        base_node = base.get(node)
        if base_node is None:
            continue
        base_inputs = base_node["inputs"]
        for name, head_input in sorted(head_node["inputs"].items()):
            base_input = base_inputs.get(name)
            if head_input["category"] == "required" and base_input is None:
                breaks.append(f"{node}.{name}: new required input on an existing node")
                continue
            if base_input is None:
                continue
            if head_input["category"] == "required" and base_input["category"] == "optional":
                breaks.append(f"{node}.{name}: optional input became required")
            dropped_types = accepted_types(base_input["type"]) - accepted_types(head_input["type"])
            if dropped_types:
                breaks.append(
                    f"{node}.{name}: no longer accepts {', '.join(sorted(dropped_types))}"
                )
            removed = [
                option
                for option in base_input.get("combo_options", [])
                if option not in head_input.get("combo_options", [])
            ]
            if removed:
                breaks.append(f"{node}.{name}: combo options removed: {', '.join(map(str, removed))}")
    return breaks


def unsafe_optionals(snapshot):
    """Optional inputs a stored prompt cannot omit without an error."""
    unsafe = []
    for node, node_entry in sorted(snapshot.items()):
        for name, entry in sorted(node_entry["inputs"].items()):
            if entry["category"] != "optional":
                continue
            if not entry["omittable"] and "default" not in entry:
                unsafe.append(
                    f"{node}.{name}: nullable input with no default in the {node_entry['module']} entry function"
                )
    return unsafe


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_snapshot(args):
    snapshot = collect_snapshot()
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=1, sort_keys=True)
    report(f"Wrote {len(snapshot)} node schemas to {args.output}")
    return 0


def run_diff(args):
    breaks = diff_snapshots(load(args.base), load(args.head))
    if not breaks:
        report("No breaking node input changes.")
        return 0
    report(f"{len(breaks)} breaking node input change(s):")
    for line in breaks:
        report(f"  {line}")
    report("")
    report("Stored API-format prompts do not have these inputs and will be rejected.")
    report("Add the input to 'optional' with a Python default instead, or register a new node.")
    return 1


def run_unsafe_optionals(args):
    unsafe = unsafe_optionals(load(args.snapshot))
    known = set(load(args.allowlist)) if args.allowlist else set()
    new = [line for line in unsafe if line.split(":")[0] not in known]
    fixed = sorted(known - {line.split(":")[0] for line in unsafe})
    if new:
        report(f"{len(new)} optional input(s) that a stored prompt cannot omit:")
        for line in new:
            report(f"  {line}")
        report("")
        report("An optional input with no default in its schema is a nullable value.")
        report("Give its parameter a default in the entry function for the not-connected case.")
    if fixed:
        report(f"{len(fixed)} allowlisted entr(ies) are fixed and must be removed from {args.allowlist}:")
        for name in fixed:
            report(f"  {name}")
    return 1 if new or fixed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot_parser = subparsers.add_parser("snapshot", help="dump every node's input surface")
    snapshot_parser.add_argument("--output", required=True)
    snapshot_parser.set_defaults(func=run_snapshot)

    diff_parser = subparsers.add_parser("diff", help="compare two snapshots for breaking changes")
    diff_parser.add_argument("base")
    diff_parser.add_argument("head")
    diff_parser.set_defaults(func=run_diff)

    optionals_parser = subparsers.add_parser("unsafe-optionals", help="find optional inputs that cannot be omitted")
    optionals_parser.add_argument("snapshot")
    optionals_parser.add_argument("--allowlist")
    optionals_parser.set_defaults(func=run_unsafe_optionals)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
