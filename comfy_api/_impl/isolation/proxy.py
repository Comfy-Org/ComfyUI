"""Nodes from a pack running in its own process, registered in ComfyUI.

A V2 pack's nodes are registered when ComfyUI starts, from the schemas in its manifest
(``comfy-pack.json``), without importing the pack. A V1 pack's nodes are registered when the pack
first starts, from what its process reports:

A V3 node's ``io.Schema`` crosses as itself (pickled; the classes ``comfytype`` copies are sent as
"the ``Input`` of ``Combo``", ``io.Custom`` types as ``io.Custom(io_type)``), so ComfyUI validates,
builds inputs and serves ``/object_info`` from the pack's real schema, and ``io.ModelFolder``
options are listed by ComfyUI each time. A V1 node's ``INPUT_TYPES`` is asked of the pack each
time. Either way the node's methods run in the pack's process: the proxy is ``ENGINE_API`` so
ComfyUI dispatches it to the execution backend, which sends it to the pack.
"""
from __future__ import annotations

import copy
import functools
import inspect
import io
import pickle
import sys
from typing import Any, Callable

V3_METHODS = ("validate_inputs", "fingerprint_inputs", "check_lazy_status")
V1_METHODS = ("VALIDATE_INPUTS", "IS_CHANGED", "check_lazy_status")
V1_ATTRS = ("RETURN_TYPES", "RETURN_NAMES", "OUTPUT_IS_LIST", "INPUT_IS_LIST", "OUTPUT_NODE", "CATEGORY",
            "DESCRIPTION", "OUTPUT_TOOLTIPS", "DEPRECATED", "EXPERIMENTAL", "NOT_IDEMPOTENT", "SEARCH_ALIASES",
            "API_NODE", "ESSENTIALS_CATEGORY")


@functools.lru_cache(maxsize=None)
def custom_type(io_type: str) -> Any:
    """``io.Custom(io_type)``, the same class every time on this side."""
    from comfy_api._impl import _io
    return _io.Custom(io_type)


def _importable(obj: type) -> bool:
    module = sys.modules.get(obj.__module__)
    target: Any = module
    for part in obj.__qualname__.split("."):
        target = getattr(target, part, None)
    return target is obj


class _SchemaPickler(pickle.Pickler):
    def reducer_override(self, obj: Any) -> Any:
        if not isinstance(obj, type) or _importable(obj):
            return NotImplemented
        parent = getattr(obj, "Parent", None)
        for attr in ("Input", "Output"):
            if parent is not None and getattr(parent, attr, None) is obj:
                return getattr, (parent, attr)
        io_type = getattr(obj, "io_type", None)
        if io_type is not None and "<locals>" in obj.__qualname__:
            return custom_type, (io_type,)
        return NotImplemented


def dump_schema(schema: Any) -> bytes:
    buffer = io.BytesIO()
    _SchemaPickler(buffer, protocol=pickle.HIGHEST_PROTOCOL).dump(schema)
    return buffer.getvalue()


def plain(value: Any) -> Any:
    """V1 type declarations as plain data: ComfyUI doesn't have the pack's classes. A ``str``
    subclass (a pack's "any type") becomes its string; containers keep their shape."""
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, dict):
        return {plain(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(plain(v) for v in value) if type(value) in (list, tuple) else list(map(plain, value))
    return repr(value)


def describe_node(cls: Any) -> dict:
    """What ComfyUI needs to register one of the pack's nodes (runs in the pack's process)."""
    from comfy_api._impl import _io
    from comfy_api._impl.internal import first_real_override

    if isinstance(cls, type) and issubclass(cls, _io.ComfyNode):
        return {"kind": "v3", "schema": dump_schema(cls.define_schema()),
                "is_async": inspect.iscoroutinefunction(cls.execute),
                "methods": [m for m in V3_METHODS if first_real_override(cls, m) is not None]}
    attrs = {name: plain(getattr(cls, name)) for name in V1_ATTRS if hasattr(cls, name)}
    return {"kind": "v1", "attrs": attrs, "methods": [m for m in V1_METHODS if hasattr(cls, m)]}


def _never_local(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("this node runs in its pack's process; ComfyUI dispatches it there")


async def _never_local_async(*args: Any, **kwargs: Any) -> Any:
    _never_local()


def make_proxy(node_id: str, description: dict, pack_key: str, pack: Any,
               input_types: Callable[[str], dict]) -> type:
    """The class ComfyUI registers for ``node_id``. ``input_types(node_id)`` asks the pack (V1)."""
    name = "".join(c if c.isalnum() else "_" for c in node_id)
    namespace: dict[str, Any] = {"__module__": pack_key, "ENGINE_API": True, "ISOLATED_NODE": (pack, node_id)}
    if description["kind"] in ("v3", "manifest"):
        from comfy_api._impl import _io

        if description["kind"] == "manifest":  # from the pack's comfy-pack.json: nothing was imported
            from comfy_api._impl import manifest
            schema = manifest.decode_schema(description["schema"])
        else:
            schema = pickle.loads(description["schema"])
        namespace["define_schema"] = classmethod(lambda cls: copy.deepcopy(schema))
        namespace["execute"] = classmethod(_never_local_async if description["is_async"] else _never_local)
        for method in description["methods"]:
            namespace[method] = classmethod(_never_local)
        return type(name, (_io.ComfyNode,), namespace)
    namespace.update(description["attrs"])
    namespace["INPUT_TYPES"] = classmethod(lambda cls: input_types(node_id))
    namespace["FUNCTION"] = "execute"
    namespace["execute"] = _never_local
    for method in description["methods"]:
        namespace[method] = classmethod(_never_local) if method != "check_lazy_status" else _never_local
    return type(name, (), namespace)
