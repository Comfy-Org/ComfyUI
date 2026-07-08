from __future__ import annotations

import logging

from aiohttp import web

from typing import TYPE_CHECKING, TypedDict
if TYPE_CHECKING:
    from comfy_api.latest._io_public import NodeReplace

from comfy_execution.graph_utils import is_link
import nodes

class NodeStruct(TypedDict):
    inputs: dict[str, str | int | float | bool | tuple[str, int]]
    class_type: str
    _meta: dict[str, str]

def copy_node_struct(node_struct: NodeStruct, empty_inputs: bool = False) -> NodeStruct:
    new_node_struct = node_struct.copy()
    if empty_inputs:
        new_node_struct["inputs"] = {}
    else:
        new_node_struct["inputs"] = node_struct["inputs"].copy()
    new_node_struct["_meta"] = node_struct["_meta"].copy()
    return new_node_struct


class NodeReplaceManager:
    """Manages node replacement registrations."""

    def __init__(self):
        self._replacements: dict[str, list[NodeReplace]] = {}

    def register(self, node_replace: NodeReplace):
        """Register a node replacement mapping.

        Idempotent: if a replacement with the same (old_node_id, new_node_id)
        is already registered, the duplicate is ignored. This prevents stale
        entries from accumulating when custom nodes are reloaded in the same
        process (e.g. via ComfyUI-Manager).
        """
        existing = self._replacements.setdefault(node_replace.old_node_id, [])
        for entry in existing:
            if entry.new_node_id == node_replace.new_node_id:
                logging.debug(
                    "Node replacement %s -> %s already registered, ignoring duplicate.",
                    node_replace.old_node_id, node_replace.new_node_id,
                )
                return
        existing.append(node_replace)

    def get_replacement(self, old_node_id: str) -> list[NodeReplace] | None:
        """Get replacements for an old node ID."""
        return self._replacements.get(old_node_id)

    def has_replacement(self, old_node_id: str) -> bool:
        """Check if a replacement exists for an old node ID."""
        return old_node_id in self._replacements

    def apply_replacements(self, prompt: dict[str, NodeStruct]):
        connections: dict[str, list[tuple[str, str, int]]] = {}
        need_replacement: set[str] = set()
        for node_number, node_struct in prompt.items():
            if "class_type" not in node_struct or "inputs" not in node_struct:
                continue
            class_type = node_struct["class_type"]
            # need replacement if not in NODE_CLASS_MAPPINGS and has replacement
            if class_type not in nodes.NODE_CLASS_MAPPINGS.keys() and self.has_replacement(class_type):
                need_replacement.add(node_number)
            # keep track of connections
            for input_id, input_value in node_struct["inputs"].items():
                if is_link(input_value):
                    conn_number = input_value[0]
                    connections.setdefault(conn_number, []).append((node_number, input_id, input_value[1]))
        for node_number in need_replacement:
            node_struct = prompt[node_number]
            class_type = node_struct["class_type"]
            replacements = self.get_replacement(class_type)
            if replacements is None:
                continue
            # just use the first replacement
            replacement = replacements[0]
            new_node_id = replacement.new_node_id
            # if replacement is not a valid node, skip trying to replace it as will only cause confusion
            if new_node_id not in nodes.NODE_CLASS_MAPPINGS.keys():
                continue
            # first, replace node id (class_type)
            new_node_struct = copy_node_struct(node_struct, empty_inputs=True)
            new_node_struct["class_type"] = new_node_id
            # TODO: consider replacing display_name in _meta as well for error reporting purposes; would need to query node schema
            # second, replace inputs
            if replacement.input_mapping is not None:
                for input_map in replacement.input_mapping:
                    if "set_value" in input_map:
                        new_node_struct["inputs"][input_map["new_id"]] = input_map["set_value"]
                    elif "old_id" in input_map:
                        new_node_struct["inputs"][input_map["new_id"]] = node_struct["inputs"][input_map["old_id"]]
            # finalize input replacement
            prompt[node_number] = new_node_struct
            # third, replace outputs
            if replacement.output_mapping is not None:
                # re-mapping outputs requires changing the input values of nodes that receive connections from this one
                if node_number in connections:
                    for conns in connections[node_number]:
                        conn_node_number, conn_input_id, old_output_idx = conns
                        for output_map in replacement.output_mapping:
                            if output_map["old_idx"] == old_output_idx:
                                new_output_idx = output_map["new_idx"]
                                previous_input = prompt[conn_node_number]["inputs"][conn_input_id]
                                previous_input[1] = new_output_idx

    def as_dict(self):
        """Serialize all replacements to dict."""
        return {
            k: [v.as_dict() for v in v_list]
            for k, v_list in self._replacements.items()
        }

    def add_routes(self, routes):
        @routes.get("/node_replacements")
        async def get_node_replacements(request):
            return web.json_response(self.as_dict())
