"""The pack process's side: the worker loop, eng calls back to ComfyUI, module loading."""
from __future__ import annotations

import asyncio
import importlib.util
import inspect
import logging
import os
import socket
import sys
import threading
import types
from typing import Any, Optional

from .boundary import from_worker, keep, restore
from .channel import Channel, WorkerError


class _WorkerOps:
    """``eng`` calls in the worker go to ComfyUI."""

    out_of_process = True  # ``ui`` savers hand their files to the runner (``_ui.collect_writes``)

    def __init__(self, state: "_WorkerState") -> None:
        self.state = state

    def supports(self, op: str) -> bool:
        return op == "engine.call"

    async def apply(self, op: str, subject: Any, params: dict) -> Any:
        if op != "engine.call":
            from comfy_api._impl._runtime import CallNotSupported
            raise CallNotSupported(f"unknown operation {op!r}")
        functions = self.state.functions
        args = from_worker(list(params.get("args", ())), functions, "argument")
        kwargs = from_worker(dict(params.get("kwargs") or {}), functions, "argument")
        return self.state.channel.request({"op": "call", "name": params["name"], "args": args, "kwargs": kwargs})


_current = threading.local()


def current_state() -> Optional["_WorkerState"]:
    """The worker state running a node on this thread, if any (for ``server`` stand-ins)."""
    return getattr(_current, "state", None)


class _WorkerState:
    def __init__(self, pack: Any = None) -> None:
        self.channel: Optional[Channel] = None
        self.functions: dict[int, Any] = {}
        self.pack = pack  # a pack server's loaded pack (server.LoadedPack); None in a plain worker

    def handle(self, message: dict) -> Any:
        op = message.get("op")
        if op == "hello":  # and whether we see ComfyUI's shared memory: then large tensors go through it
            from .channel import can_see
            shared = can_see(message.get("shm_probe"))
            if self.channel is not None:
                self.channel.shared_memory = shared
            from . import torch_state  # ComfyUI's numeric settings (TF32, deterministic...), not PyTorch's defaults
            unset = torch_state.apply(message.get("torch_settings"))
            return {"pid": os.getpid(), "shm": shared, "torch_unset": unset}
        if self.pack is not None and op in ("describe", "nodes", "input_types", "http", "route"):
            return self.pack.handle(message)
        if op in ("invoke", "registered"):
            if op == "registered":  # a sampler or scheduler the pack registered, by name
                table = (self.pack.registrations if self.pack is not None else {}).get(message["kind"] + "s", {})
                fn = table.get(message["name"])
                if fn is None:
                    raise KeyError(f"this pack registered no {message['kind']} {message['name']!r}")
            else:
                fn = self.functions.get(message["closure"])
            if fn is None:
                raise KeyError(f"pack function {message['closure']} is gone")
            previous = getattr(_current, "state", None)
            _current.state = self  # engine functions it was given call ComfyUI on this channel
            from . import torch_state
            try:
                with torch_state.in_mode(message.get("mode")):  # the caller's grad / inference mode
                    result = fn(*message["args"], **message["kwargs"])
            finally:
                _current.state = previous
            return from_worker(result, None, "a sampling function's result")
        if op == "execute":
            _current.state = self
            try:
                return asyncio.run(self.execute(message))
            finally:
                _current.state = None
        raise WorkerError(f"unknown request {op!r}")

    async def execute(self, message: dict) -> dict:
        from comfy_api._impl import _io, _runtime, _ui
        from . import torch_state

        if message["locator"][0] == "node":  # a pack server's node, by id
            cls = self.pack.nodes[message["locator"][1]]
        else:
            cls = _load_class(message["module"], message["file"], message["locator"])
        hidden = message["hidden"]
        target: Any = cls
        method = message["method"]
        if isinstance(cls, type) and issubclass(cls, _io.ComfyNode):
            target = cls.PREPARE_CLASS_CLONE({"hidden_inputs": {
                _io.Hidden.prompt: hidden["prompt"], _io.Hidden.unique_id: hidden["unique_id"],
                _io.Hidden.extra_pnginfo: hidden["extra_pnginfo"]}})
            if method.startswith("EXECUTE_NORMALIZED"):
                method = cls.FUNCTION  # the proxy's execute may be async where the node's isn't
        elif message["locator"][0] == "node":  # a V1 node: ComfyUI names its methods the V3 way
            target = cls()
            method = {"execute": cls.FUNCTION, "validate_inputs": "VALIDATE_INPUTS",
                      "fingerprint_inputs": "IS_CHANGED"}.get(method, method)
        fn = getattr(target, method)
        settings = message.get("settings") or {}
        with _runtime.bind_runtime(_runtime.InProcessRefResolver(), None, _WorkerOps(self), pack=message["pack"]), \
                _ui.collect_writes(settings.get("disable_metadata")) as pending, torch_state.in_mode(message.get("mode")):
            result = fn(**restore(message["inputs"]))
            if inspect.isawaitable(result):
                result = await result
            await pending.flush()  # files ui savers encoded, written by ComfyUI before the ui leaves
        # A node's own output (execute, or ComfyUI's EXECUTE_NORMALIZED wrapper around it), or the
        # plain value of another method such as check_lazy_status.
        is_output = hasattr(result, "result") or (isinstance(result, dict) and ({"result", "ui"} & set(result)))
        if not is_output:
            return {"kind": "plain", "value": keep(result)}  # a V1 node's tuple, or a method's value
        if isinstance(result, dict):
            out = {"result": result.get("result") or (), "ui": result.get("ui"), "expand": result.get("expand"),
                   "block_execution": None}
        elif hasattr(result, "result"):
            out = {"result": result.result or (), "ui": getattr(result, "ui", None),
                   "expand": getattr(result, "expand", None), "block_execution": getattr(result, "block_execution", None)}
        if hasattr(out["ui"], "as_dict"):
            out["ui"] = out["ui"].as_dict()
        out["result"] = tuple(keep(list(out["result"])))  # a pack's own types stay here, as PackValues
        out["ui"] = from_worker(out["ui"], None, "ui")
        out["expand"] = from_worker(out["expand"], None, "expand")
        out["kind"] = "output"
        return out


def _load_class(module_name: str, module_file: str, locator: tuple) -> Any:
    """Import the node's own module (not the pack's ``__init__``) under the name ComfyUI gave it."""
    module = sys.modules.get(module_name)
    if module is None:
        parts = module_name.split(".")
        is_package = os.path.basename(module_file) == "__init__.py"
        directory = os.path.dirname(module_file)
        for i in range(len(parts) - 1, 0, -1):  # ancestors as namespace packages, for relative imports
            name = ".".join(parts[:i])
            if name not in sys.modules:
                parent = types.ModuleType(name)
                parent.__path__ = [directory]
                parent.__package__ = name
                sys.modules[name] = parent
            directory = os.path.dirname(directory)
        spec = importlib.util.spec_from_file_location(
            module_name, module_file, submodule_search_locations=[os.path.dirname(module_file)] if is_package else None)
        module = importlib.util.module_from_spec(spec)
        if not is_package and len(parts) > 1:
            module.__package__ = ".".join(parts[:-1])
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    how, name = locator
    if how == "mapping":
        return module.NODE_CLASS_MAPPINGS[name]
    if how == "global":
        return getattr(module, name)
    target: Any = module
    for part in name.split("."):
        target = getattr(target, part)
    return target


def _worker_main(fd: int) -> None:
    logging.basicConfig(level=logging.INFO, format="[pack worker %(process)d] %(message)s")
    from comfy_api._impl import legacy_imports
    legacy_imports.install()  # packs that import comfy_api.latest
    state = _WorkerState()
    sock = socket.socket(fileno=fd)
    state.channel = Channel(sock, state.handle)
    try:
        state.channel.serve_forever()
    except WorkerError:
        pass  # ComfyUI went away
