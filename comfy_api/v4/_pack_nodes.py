"""The node classes and routes ComfyUI registers for a V4 pack that runs in its own process.

Each class has the pack's schema for its node, as the pack defines it now, and its methods run
the pack's node in the pack's process. Each route forwards its requests to the pack's process,
starting it again if it has stopped, and answers 503 if the process stops while it answers.
"""

import inspect
from typing import TYPE_CHECKING

from comfy_api.latest import io
from comfy_execution.utils import get_executing_context

from . import _channel
from ._extension import ComfyExtension, Route, RouteRequest, RouteResponse

if TYPE_CHECKING:
    from ._pack_host import PackProcess


def extension(process: "PackProcess", module_name: str, loaded: dict) -> ComfyExtension:
    """The extension that lists a node class per schema and a route per route the pack's
    process declared when it loaded the pack."""
    return _RemoteExtension(
        [
            _node_class(process, module_name, schema.node_id, methods)
            for schema, methods in loaded["nodes"]
        ],
        [
            _route(process, index, *route)
            for index, route in enumerate(loaded["routes"])
        ],
    )


class _RemoteExtension(ComfyExtension):
    """What the loader registers for an isolated pack: node classes and routes that use its
    process."""

    def __init__(self, nodes: list[type[io.ComfyNode]], routes: list[Route]):
        self.nodes, self.routes = nodes, routes

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return self.nodes

    async def get_routes(self) -> list[Route]:
        return self.routes


def _method(process: "PackProcess", node_id: str, name: str, signature=None):
    async def method(cls, **inputs):
        hidden = {
            hidden: value
            for hidden in io.Hidden
            if (value := getattr(cls.hidden, hidden.name)) is not None
        }
        lane = "server" if name == "validate_inputs" else "exec"
        running = get_executing_context()
        # The executor names the nodes a node expands into after the node and the list item.
        prefix = running and (running.node_id, running.list_index or 0)
        return await process.run(
            ("node", node_id, name), (inputs, hidden, prefix), lane
        )

    if signature is not None:
        # ComfyUI's executor passes validate_inputs only the inputs its signature names.
        arguments, has_kwargs = signature
        parameters = [
            inspect.Parameter(argument, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for argument in arguments
        ]
        if has_kwargs:
            parameters.append(
                inspect.Parameter("inputs", inspect.Parameter.VAR_KEYWORD)
            )
        method.__signature__ = inspect.Signature(parameters)
    return method


def _fingerprint(process: "PackProcess", node_id: str, signature):
    """The node's fingerprint: the process's epoch, with the pack's own fingerprint if it has
    one."""
    pack_fingerprint = (
        None
        if signature is None
        else _method(process, node_id, "fingerprint_inputs", signature)
    )

    async def fingerprint_inputs(cls, **inputs):
        if pack_fingerprint is None:
            return process.epoch
        value = await pack_fingerprint(cls, **inputs)
        return process.epoch, value

    if pack_fingerprint is not None:
        fingerprint_inputs.__signature__ = pack_fingerprint.__signature__
    return classmethod(fingerprint_inputs)


def _node_class(process: "PackProcess", module_name: str, node_id: str, methods: dict):
    """A node class for ComfyUI with the pack's schema, whose methods run the pack's node."""
    namespace = {
        "__module__": module_name,
        "define_schema": classmethod(lambda cls: process.schema(node_id)),
        "execute": classmethod(_method(process, node_id, "execute")),
        "fingerprint_inputs": _fingerprint(
            process, node_id, methods.get("fingerprint_inputs")
        ),
    }
    for name in ("validate_inputs", "check_lazy_status"):
        if name in methods:
            namespace[name] = classmethod(
                _method(process, node_id, name, methods[name])
            )
    return type(node_id, (io.ComfyNode,), namespace)


def _route(process: "PackProcess", index: int, method: str, path: str) -> Route:
    async def forward(request: RouteRequest) -> RouteResponse:
        try:
            return await process.run(("route", index), (request,), "server")
        except _channel.PackStopped:
            headers = {"content-type": "text/plain"}
            return RouteResponse(
                status=503, headers=headers, body=b"pack process stopped"
            )

    return Route(method=method, path=path, handler=forward)
