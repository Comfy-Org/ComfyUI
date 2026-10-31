"""Structural checks on the workflow templates this build ships.

A template that stops working is a release blocker, and nothing else in CI
notices: the templates come from a pinned
`comfyui-workflow-templates` release, so a node renamed, removed or retyped in
this repo breaks them silently until someone opens the template by hand.

These checks read every shipped template graph and hold it against this build's
own node registry. They are structural on purpose: no models, no GPU, no
server, so they can gate a release branch in seconds.

What they cannot see, so that the next person does not assume otherwise:

* widget *values* are not checked. Mapping a node's `widgets_values` array onto
  named inputs needs the frontend's widget ordering rules, so a combo option
  removed from, say, `sampler_name` is not caught here.
* a few hundred of the ~13,000 links cannot be typed on either end -- both ends
  are subgraph instances, or a node a custom-node pack defines -- and are
  skipped rather than guessed at.
* a new *required* input is not caught. Several core nodes declare a required
  input that only the frontend fills (`IMAGECOMPARE`, `LOAD_3D`), and there is
  no reliable marker separating those from an input a template must wire, so
  the check would be mostly false positives.
* nodes from custom-node packs are skipped: this repo does not define them.
"""
import asyncio
import json
from typing import Any

import comfyui_workflow_templates as templates
import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import nodes  # noqa: E402

# Node types the frontend renders itself; they never reach the backend.
FRONTEND_ONLY = {"MarkdownNote", "Note", "Reroute", "PrimitiveNode"}

# Widget names the frontend adds for its own controls, with no input behind them.
UI_ONLY_WIDGETS = {"upload", "audioUI", "imageUI", "videoUI", "model3DUI", "previewUI"}

# Input and return types that match anything: the V3 dynamic kinds, whose
# concrete type is decided at runtime, and the legacy wildcard. COMBO is
# deliberately not here: ComfyUI's own validation does not treat a combo as an
# any-type socket, so neither does this.
WILDCARD_TYPES = {
    "COMFY_MATCHTYPE_V3",
    "COMFY_AUTOGROW_V3",
    "COMFY_DYNAMICCOMBO_V3",
    "COMFY_MULTITYPE_V3",
    "*",
}

# A node saved by the frontend records which pack defines it.
CORE_PACK = "comfy-core"

# The templates package ships its index, its schema and its lookup tables as
# assets alongside the workflows. They are named here rather than detected by
# "has no nodes list", so a workflow that is actually malformed fails the
# structural checks instead of quietly dropping out of them.
NON_WORKFLOW_ASSETS = {"fuse_options", "index_logo", "index.schema"}


def is_workflow_asset(template_id: str) -> bool:
    return template_id not in NON_WORKFLOW_ASSETS and not template_id.startswith(
        "index"
    )

# Muted and bypassed nodes are not executed, so a stale one is not a break.
SKIPPED_MODES = {2, 4}


@pytest.fixture(scope="session")
def node_registry() -> dict[str, Any]:
    """This build's node classes, without custom nodes."""
    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False))
    return nodes.NODE_CLASS_MAPPINGS


@pytest.fixture(scope="session")
def shipped_templates() -> list[tuple[str, dict[str, Any]]]:
    """Every template graph in the pinned templates release, by template id."""
    graphs = []
    malformed = []
    for entry in templates.iter_templates():
        template_id = entry.template_id
        if not is_workflow_asset(template_id):
            continue
        path = templates.get_asset_path(template_id, f"{template_id}.json")
        with open(path, encoding="utf-8") as handle:
            graph = json.load(handle)
        if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list):
            malformed.append(template_id)
            continue
        graphs.append((template_id, graph))
    assert not malformed, f"templates with no node list: {sorted(malformed)}"
    assert graphs, "no template graphs found in the installed templates package"
    return graphs


def iter_nodes(graph: dict[str, Any]):
    """Every node in a template, in its own graph, with that graph's links.

    A template's top-level nodes and each subgraph definition's nodes live in
    separate graphs with separate id spaces, and both have to be checked.
    """
    subgraphs = (graph.get("definitions") or {}).get("subgraphs") or []
    for scope in [graph, *subgraphs]:
        yield scope


SUBGRAPH_INPUT_ID = -10
SUBGRAPH_OUTPUT_ID = -20


def boundary_slot_type(scope: dict[str, Any], side: str, slot: int) -> Any | None:
    """The type a subgraph declares for its own input or output slot `slot`.

    A link inside a subgraph can start at the subgraph's input boundary (-10) or
    end at its output boundary (-20). Those pseudo-ids are not nodes, so without
    this the link is skipped -- and an inner node's output changing type would
    go unnoticed, because its only link is the one to the boundary.
    """
    declared = scope.get("inputs" if side == "inputs" else "outputs") or []
    if not isinstance(slot, int) or slot >= len(declared):
        return None
    return declared[slot].get("type")


def slot_declared_type(node: dict[str, Any], side: str, slot: int) -> Any | None:
    """The type a node's own slot list declares, used for subgraph instances.

    An instance has no node class, so its boundary types are whatever the
    template saved on it -- which is the thing to compare the other end
    against.
    """
    slots = node.get(side) or []
    if not isinstance(slot, int) or slot >= len(slots):
        return None
    return slots[slot].get("type")


def subgraph_ids(graph: dict[str, Any]) -> set[str]:
    """Node types that name a subgraph in this template rather than a node class."""
    return {
        subgraph["id"]
        for subgraph in (graph.get("definitions") or {}).get("subgraphs") or []
        if "id" in subgraph
    }


def backend_nodes(graph: dict[str, Any], registry: dict[str, Any]):
    """(scope, node, node class) for each node this build is expected to define."""
    skip = subgraph_ids(graph) | FRONTEND_ONLY
    for scope in iter_nodes(graph):
        for node in scope.get("nodes") or []:
            node_type = node.get("type")
            if node_type in skip or node.get("mode") in SKIPPED_MODES:
                continue
            node_class = registry.get(node_type)
            if node_class is not None:
                yield scope, node, node_class


def is_core(node: dict[str, Any]) -> bool:
    return (node.get("properties") or {}).get("cnr_id") == CORE_PACK


def input_specs(node_class: Any) -> dict[str, Any]:
    """Every declared input of a node class, required, optional and hidden."""
    spec = node_class.INPUT_TYPES()
    declared = {}
    for section in ("required", "optional", "hidden"):
        declared.update(spec.get(section) or {})
    return declared


def slot_name(slot: dict[str, Any]) -> str | None:
    return (slot.get("widget") or {}).get("name") or slot.get("name")


def declares(declared: dict[str, Any], name: str) -> bool:
    """Whether `name` is one of the node's inputs.

    A V3 dynamic input contributes `<name>.<sub>` slots -- `values.a` on an
    autogrow, `format.bit_depth` once a dynamic combo picks png -- which the
    class declares only under the parent name.
    """
    return name in declared or name.split(".")[0] in declared


def type_names(declared_type: Any) -> set[str]:
    """The type strings a slot accepts. A V3 union writes them comma-separated."""
    if isinstance(declared_type, list):
        return {"COMBO"}
    return {part.strip() for part in str(declared_type).split(",") if part.strip()}


def types_connect(source: Any, target: Any) -> bool:
    source_types, target_types = type_names(source), type_names(target)
    if source_types & WILDCARD_TYPES or target_types & WILDCARD_TYPES:
        return True
    return bool(source_types & target_types)


def input_type(declared: dict[str, Any], name: str) -> Any | None:
    entry = declared.get(name) or declared.get(name.split(".")[0])
    if isinstance(entry, (list, tuple)) and entry:
        return entry[0]
    return None


def return_types(node_class: Any) -> list[Any]:
    return list(getattr(node_class, "RETURN_TYPES", ()) or ())


def links_of(scope: dict[str, Any]) -> list[tuple[Any, int, Any, int]]:
    """(origin node, origin slot, target node, target slot) for one graph's links.

    Top-level links are arrays, subgraph links are objects, and both use the
    pseudo-ids -10 and -20 for the subgraph's own input and output boundary.
    """
    links = []
    for link in scope.get("links") or []:
        if isinstance(link, dict):
            links.append(
                (
                    link.get("origin_id"),
                    link.get("origin_slot"),
                    link.get("target_id"),
                    link.get("target_slot"),
                )
            )
        elif isinstance(link, (list, tuple)) and len(link) >= 5:
            links.append((link[1], link[2], link[3], link[4]))
    return links


def test_templates_are_shipped(shipped_templates):
    """A packaging change that stops shipping templates would make the rest of
    this file pass by checking nothing."""
    assert len(shipped_templates) > 100


def test_every_core_node_a_template_uses_is_registered(
    shipped_templates, node_registry
):
    """A node removed or renamed here leaves every template using it broken."""
    missing = set()
    for template_id, graph in shipped_templates:
        skip = subgraph_ids(graph) | FRONTEND_ONLY
        for scope in iter_nodes(graph):
            for node in scope.get("nodes") or []:
                node_type = node.get("type")
                if node_type in skip or not is_core(node):
                    continue
                if node.get("mode") in SKIPPED_MODES:
                    # Not executed, so a stale one is not worth a release.
                    continue
                if node_type not in node_registry:
                    missing.add((template_id, node_type))
    assert not missing, f"templates use core nodes this build does not define: {sorted(missing)}"


def test_every_connected_input_a_template_uses_still_exists(
    shipped_templates, node_registry
):
    """An input renamed here turns the link feeding it into a missing input.

    A slot with nothing connected is ignored: the frontend leaves one behind
    when an input is renamed, and an unused slot changes nothing at run time.
    """
    unknown = set()
    for template_id, graph in shipped_templates:
        for _scope, node, node_class in backend_nodes(graph, node_registry):
            declared = input_specs(node_class)
            for slot in node.get("inputs") or []:
                name = slot_name(slot)
                if name is None or name in UI_ONLY_WIDGETS:
                    continue
                if slot.get("link") is None or declares(declared, name):
                    continue
                unknown.add((template_id, node["type"], name))
    assert not unknown, f"templates wire inputs that no longer exist: {sorted(unknown)}"


def test_no_template_link_leaves_an_output_a_node_no_longer_has(
    shipped_templates, node_registry
):
    """Dropping or reordering a node's returns strands the links past the end."""
    stranded = set()
    for template_id, graph in shipped_templates:
        for scope in iter_nodes(graph):
            by_id = {node["id"]: node for node in scope.get("nodes") or []}
            for origin_id, origin_slot, _target_id, _target_slot in links_of(scope):
                origin = by_id.get(origin_id)
                if origin is None or not isinstance(origin_slot, int):
                    continue
                node_class = node_registry.get(origin.get("type"))
                if node_class is None:
                    continue
                outputs = return_types(node_class)
                if origin_slot >= len(outputs):
                    stranded.add(
                        (template_id, origin["type"], origin_slot, len(outputs))
                    )
    assert not stranded, (
        "templates link to an output index the node no longer has "
        f"(template, node, slot, outputs): {sorted(stranded)}"
    )


def produced_type(
    scope: dict[str, Any],
    by_id: dict[Any, dict[str, Any]],
    instances: set[str],
    registry: dict[str, Any],
    node_id: Any,
    slot: Any,
) -> Any | None:
    """The type coming out of a link's origin, or None when it cannot be known."""
    if node_id == SUBGRAPH_INPUT_ID:
        return boundary_slot_type(scope, "inputs", slot)
    node = by_id.get(node_id)
    if node is None:
        return None
    if node.get("type") in instances:
        return slot_declared_type(node, "outputs", slot)
    node_class = registry.get(node.get("type"))
    if node_class is None:
        return None
    outputs = return_types(node_class)
    if not isinstance(slot, int) or slot >= len(outputs):
        return None
    return outputs[slot]


def accepted_type(
    scope: dict[str, Any],
    by_id: dict[Any, dict[str, Any]],
    instances: set[str],
    registry: dict[str, Any],
    node_id: Any,
    slot: Any,
) -> tuple[str, Any] | None:
    """(label, type) a link's target accepts, or None when it cannot be known."""
    if node_id == SUBGRAPH_OUTPUT_ID:
        declared = boundary_slot_type(scope, "outputs", slot)
        return (f"subgraph output[{slot}]", declared) if declared else None
    node = by_id.get(node_id)
    if node is None:
        return None
    if node.get("type") in instances:
        declared = slot_declared_type(node, "inputs", slot)
        return (f"subgraph input[{slot}]", declared) if declared else None
    node_class = registry.get(node.get("type"))
    if node_class is None:
        return None
    slots = node.get("inputs") or []
    if not isinstance(slot, int) or slot >= len(slots):
        return None
    name = slot_name(slots[slot])
    if name is None:
        return None
    declared = input_type(input_specs(node_class), name)
    return (f"{node['type']}.{name}", declared) if declared else None


def test_every_template_link_still_connects_types_that_match(
    shipped_templates, node_registry
):
    """Retyping an output or an input breaks the templates wired to it.

    Links across a subgraph boundary are included: inside a subgraph the only
    link an inner node's output has is often the one to the output boundary, so
    skipping those would leave most of a modern template unchecked.
    """
    mismatched = set()
    for template_id, graph in shipped_templates:
        instances = subgraph_ids(graph)
        for scope in iter_nodes(graph):
            by_id = {node["id"]: node for node in scope.get("nodes") or []}
            for origin_id, origin_slot, target_id, target_slot in links_of(scope):
                source = produced_type(
                    scope, by_id, instances, node_registry, origin_id, origin_slot
                )
                sink = accepted_type(
                    scope, by_id, instances, node_registry, target_id, target_slot
                )
                if source is None or sink is None:
                    continue
                label, target_type = sink
                if not types_connect(source, target_type):
                    mismatched.add((template_id, f"{source} -> {label}:{target_type}"))
    assert not mismatched, f"templates wire types that no longer connect: {sorted(mismatched)}"


# --------------------------------------------------------------------------- #
# the comparison rules themselves, so they are pinned without a whole template
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "source, target, connects",
    [
        ("IMAGE", "IMAGE", True),
        ("IMAGE", "LATENT", False),
        # A V3 union writes its alternatives comma-separated.
        ("FILE_3D_GLB", "MESH,FILE_3D_GLB,FILE_3D_OBJ", True),
        ("FILE_3D_PLY", "MESH,FILE_3D_GLB", False),
        # Dynamic kinds take their concrete type at runtime.
        ("COMFY_MATCHTYPE_V3", "IMAGE", True),
        ("IMAGE", "COMFY_MATCHTYPE_V3", True),
        ("IMAGE", "*", True),
        # A combo is not an any-type socket: ComfyUI's validation rejects an
        # IMAGE wired into one, so this has to as well.
        ("IMAGE", ["euler", "ddim"], False),
        (["euler", "ddim"], ["euler", "ddim"], True),
    ],
)
def test_types_connect_follows_comfyuis_own_rules(source, target, connects):
    assert types_connect(source, target) is connects


def test_a_dynamic_inputs_sub_slots_count_as_declared():
    """format.bit_depth and values.a exist on the node, not in the class."""
    declared = {"format": ["COMFY_DYNAMICCOMBO_V3"], "values": ["COMFY_AUTOGROW_V3"]}
    assert declares(declared, "format.bit_depth")
    assert declares(declared, "values.a")
    assert not declares(declared, "quality")


@pytest.mark.parametrize(
    "side, slot, expected",
    [("inputs", 0, "STRING"), ("outputs", 0, "IMAGE"), ("inputs", 9, None)],
)
def test_a_subgraph_boundary_slot_reports_its_declared_type(side, slot, expected):
    scope = {
        "inputs": [{"name": "text", "type": "STRING"}],
        "outputs": [{"name": "IMAGE", "type": "IMAGE"}],
    }
    assert boundary_slot_type(scope, side, slot) == expected
