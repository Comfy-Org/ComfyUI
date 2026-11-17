"""Pack isolation: ``COMFY_PACK_ISOLATION=true``.

Each custom-node pack gets its own Python worker process. Engine-API nodes
(``ENGINE_API = True``) from that pack run there; everything else (ComfyUI's own
nodes, and custom nodes written against the older API, which use engine objects
directly) runs in ComfyUI's process as usual.

The pack is still imported in ComfyUI's process, so its nodes register, its web
directory is served, and its engine extensions are loaded and initialized there,
next to the engine, as in any other mode. Only the node methods run in the worker.

The boundary follows the same rules as any out-of-process dispatch: engine
objects (MODEL, CLIP, VAE...) stay in ComfyUI and cross as handles the pack passes
back to ``eng`` calls; tensors and plain data cross as copies; anything else is
refused. A function given to a ``model.set_*`` call stays in the worker and is
called from sampling. So a pack that works here works wherever its nodes are
dispatched out of process.

ComfyUI's own pack isolation is for robustness and for keeping packs apart, not a security
boundary: the two processes trust each other. An execution provider (``COMFY_EXECUTION_PROVIDER``)
can run the same packs elsewhere, securely if it wants, over the same seam.
"""
from __future__ import annotations

import asyncio
import atexit
import dataclasses
import importlib.util
import inspect
import itertools
import logging
import os
import pickle
import socket
import struct
import subprocess
import sys
import threading
import traceback
import types
from typing import Any, Optional

logger = logging.getLogger(__name__)
ENV = "COMFY_PACK_ISOLATION"
LEGACY_ENV = "OOP_CUSTOM_NODES"


def enabled() -> bool:
    value = os.environ.get(ENV) or os.environ.get(LEGACY_ENV, "")
    return value.strip().lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------- #
# What crosses
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class Closure:
    """A function that stays in the worker; ComfyUI calls it by id."""
    id: int


@dataclasses.dataclass(frozen=True)
class PackValue:
    """A node output that isn't data (a pack's own type, such as a namedtuple of detections): it stays
    in the pack's process, and the pack's next node gets the object back. ``pid`` is that process."""
    id: int
    type: str
    pid: int

    def __repr__(self) -> str:
        return f"<{self.type} in pack process {self.pid}>"


def _is_tensor(value: Any) -> bool:
    return type(value).__name__ == "Tensor" and hasattr(value, "shape")


def _is_function(value: Any) -> bool:
    """A pack function stays in the pack's process and is called there. An object the pack made to be
    called (a class with ``__call__``, such as a guidance calculator) is one too, as it is in ComfyUI."""
    import functools
    if isinstance(value, (types.FunctionType, types.MethodType, functools.partial)):
        return True
    return (not isinstance(value, type) and type(value).__module__ != "builtins" and not _is_tensor(value)
            and callable(getattr(type(value), "__call__", None)))


_PLAIN = (str, bytes, bool, int, float, type(None))


def to_worker(value: Any, refs: Any, create) -> Any:
    """ComfyUI -> worker: data as it is (pickled, so copied); engine objects as handles."""
    from comfy_api.latest import _runtime

    if isinstance(value, _PLAIN) or _is_tensor(value) or isinstance(value, (_runtime.Ref, Closure, PackValue)):
        return value
    if type(value).__module__ == "numpy":  # numpy arrays cross as data, both ways
        return value
    if isinstance(value, dict):  # dict subclasses (SavedResult...) cross as plain dicts
        return {k: to_worker(v, refs, create) for k, v in value.items()}
    if type(value) in (list, tuple):
        return type(value)(to_worker(v, refs, create) for v in value)
    return create(value)


def from_worker(value: Any, functions: Optional[dict] = None, path: str = "value") -> Any:
    """Worker -> ComfyUI: only data, handles ComfyUI gave out, and (as call arguments) functions."""
    from comfy_api.latest import _runtime

    if isinstance(value, _PLAIN) or _is_tensor(value) or isinstance(value, (_runtime.Ref, PackValue)):
        return value
    if type(value).__module__ == "numpy":
        return value
    if functions is not None and _is_function(value):
        cid = next(_closure_ids)
        functions[cid] = value
        return Closure(cid)
    if isinstance(value, dict):  # dict subclasses (SavedResult...) cross as plain dicts
        return {k: from_worker(v, functions, f"{path}[{k!r}]") for k, v in value.items()}
    if type(value) in (list, tuple):
        return type(value)(from_worker(v, functions, f"{path}[{i}]") for i, v in enumerate(value))
    raise TypeError(f"{path}: a {type(value).__name__} can't leave a pack's process; "
                    "return tensors, plain data, or engine objects it was given")


_closure_ids = itertools.count(1)
_pack_value_ids = itertools.count(1)
_pack_values: dict[int, Any] = {}  # in a pack's process: the objects its PackValues stand for


def keep(value: Any) -> Any:
    """A node output, worker -> ComfyUI: data as ``from_worker`` sends it; anything else stays here
    as a ``PackValue``."""
    from comfy_api.latest import _runtime

    if isinstance(value, _PLAIN) or _is_tensor(value) or isinstance(value, (_runtime.Ref, PackValue)):
        return value
    if type(value).__module__ == "numpy":
        return value
    if type(value).__name__ == "Size" and type(value).__module__.startswith("torch"):
        return tuple(value)  # a tensor's shape is plain data
    if type(value) is dict:
        return {k: keep(v) for k, v in value.items()}
    if type(value) in (list, tuple):
        return type(value)(keep(v) for v in value)
    pid = next(_pack_value_ids)
    _pack_values[pid] = value
    return PackValue(pid, type(value).__name__, os.getpid())


def restore(value: Any) -> Any:
    """Node inputs in a pack's process: PackValues back to the objects they stand for."""
    if isinstance(value, PackValue):
        if value.pid != os.getpid() or value.id not in _pack_values:
            raise ValueError(f"{value!r} was made by a pack process that has stopped, or by another pack; "
                             "run the node that made it again")
        return _pack_values[value.id]
    if type(value) is dict:
        return {k: restore(v) for k, v in value.items()}
    if type(value) in (list, tuple):
        return type(value)(restore(v) for v in value)
    return value


# --------------------------------------------------------------------------- #
# Numeric settings: the worker computes as ComfyUI does
# --------------------------------------------------------------------------- #
def _torch_settings() -> dict:
    """This process's PyTorch switches (TF32, deterministic algorithms, matmul precision), as plain data.
    They are per process: a worker left at PyTorch's defaults computes differently from the same code
    in ComfyUI, so the worker takes ComfyUI's at the handshake."""
    import torch

    settings: dict[str, Any] = {}

    def read(key: str, getter) -> None:
        try:
            settings[key] = getter()
        except Exception:
            pass

    read("cudnn_allow_tf32", lambda: bool(torch.backends.cudnn.allow_tf32))
    read("cudnn_benchmark", lambda: bool(torch.backends.cudnn.benchmark))
    read("cudnn_deterministic", lambda: bool(torch.backends.cudnn.deterministic))
    read("matmul_allow_tf32", lambda: bool(torch.backends.cuda.matmul.allow_tf32))
    read("matmul_fp16_reduced", lambda: bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction))
    read("matmul_bf16_reduced", lambda: bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction))
    read("float32_matmul_precision", lambda: str(torch.get_float32_matmul_precision()))
    read("deterministic", lambda: bool(torch.are_deterministic_algorithms_enabled()))
    read("deterministic_warn_only", lambda: bool(torch.is_deterministic_algorithms_warn_only_enabled()))
    return settings


def _apply_torch_settings(settings: Optional[dict]) -> None:
    import torch

    def write(key: str, setter) -> None:
        if key in (settings or {}):
            try:
                setter(settings[key])
            except Exception:
                logger.warning("couldn't take ComfyUI's %s setting", key)

    write("cudnn_allow_tf32", lambda v: setattr(torch.backends.cudnn, "allow_tf32", bool(v)))
    write("cudnn_benchmark", lambda v: setattr(torch.backends.cudnn, "benchmark", bool(v)))
    write("cudnn_deterministic", lambda v: setattr(torch.backends.cudnn, "deterministic", bool(v)))
    write("matmul_allow_tf32", lambda v: setattr(torch.backends.cuda.matmul, "allow_tf32", bool(v)))
    write("matmul_fp16_reduced",
          lambda v: setattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", bool(v)))
    write("matmul_bf16_reduced",
          lambda v: setattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", bool(v)))
    write("float32_matmul_precision", lambda v: torch.set_float32_matmul_precision(str(v)))
    write("deterministic", lambda v: torch.use_deterministic_algorithms(
        bool(v), warn_only=bool(settings.get("deterministic_warn_only", False))))


def _folder_layout() -> Optional[dict]:
    """ComfyUI's folders (``--output-directory``, extra model paths...), for the worker's ``folder_paths``."""
    folder_paths = sys.modules.get("folder_paths")
    if folder_paths is None:
        return None
    return {"models_dir": folder_paths.models_dir, "output": folder_paths.get_output_directory(),
            "input": folder_paths.get_input_directory(), "temp": folder_paths.get_temp_directory(),
            "user": folder_paths.get_user_directory(),
            "folders": {name: [list(paths), sorted(exts)] for name, (paths, exts) in folder_paths.folder_names_and_paths.items()}}


def _apply_folder_layout(layout: Optional[dict]) -> None:
    """In the worker: ComfyUI's folders, as ComfyUI handed them over (``util.folders``, ``folder_paths``)."""
    if not layout:
        return
    import folder_paths

    folder_paths.models_dir = layout["models_dir"]
    folder_paths.set_output_directory(layout["output"])
    folder_paths.set_input_directory(layout["input"])
    folder_paths.set_temp_directory(layout["temp"])
    folder_paths.set_user_directory(layout["user"])
    for name, (paths, exts) in layout["folders"].items():
        folder_paths.folder_names_and_paths[name] = (list(paths), set(exts))
    folder_paths.filename_list_cache.clear()


def _call_mode() -> dict:
    """The grad / inference mode the calling thread is in, carried into the worker with each call."""
    import torch

    return {"grad": bool(torch.is_grad_enabled()), "inference": bool(torch.is_inference_mode_enabled())}


def _in_mode(mode: Optional[dict]):
    """A context running a call in the caller's grad / inference mode (as ComfyUI ran it)."""
    import contextlib
    import torch

    if not mode:
        return contextlib.nullcontext()
    if mode.get("inference"):
        return torch.inference_mode(True)
    return torch.set_grad_enabled(bool(mode.get("grad", True)))


# --------------------------------------------------------------------------- #
# The channel: length-prefixed pickles, strictly nested requests
# --------------------------------------------------------------------------- #
class WorkerError(RuntimeError):
    """The worker process failed or went away."""


class RemoteError(RuntimeError):
    def __init__(self, kind: str, message: str, trace: str = "") -> None:
        super().__init__(f"{kind}: {message}")
        self.kind, self.message, self.trace = kind, message, trace


_RAISABLE = {"TypeError": TypeError, "ValueError": ValueError, "KeyError": KeyError,
             "PermissionError": PermissionError, "NotImplementedError": NotImplementedError}


def _raise_remote(error: dict) -> None:
    if error["kind"] == "CallNotSupported":
        from comfy_api.latest._runtime import CallNotSupported
        raise CallNotSupported(error["message"])
    import builtins
    cls = _RAISABLE.get(error["kind"]) or getattr(builtins, error["kind"], None)
    if isinstance(cls, type) and issubclass(cls, Exception) and cls not in (SystemExit, KeyboardInterrupt):
        raise cls(error["message"])
    raise RemoteError(error["kind"], error["message"], error.get("trace", ""))


def _describe(exc: BaseException) -> dict:
    return {"kind": type(exc).__name__, "message": str(exc), "trace": traceback.format_exc()}


class Channel:
    """Either side of the socket. A side waiting for a reply serves the other side's nested
    requests in the meantime (an engine call during a node, a sampling function during that
    call), so the conversation can nest to any depth without threads waiting on each other."""

    def __init__(self, sock: socket.socket, handler) -> None:
        self.sock = sock
        self.handler = handler  # handler(message) -> result, for requests from the other side
        self._ids = itertools.count(1)
        self._send_lock = threading.Lock()
        self._recv_lock = threading.Lock()
        if sock.family == socket.AF_INET:  # Windows' socketpair is loopback TCP: send a message's tail at once
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def _send(self, message: dict) -> None:
        data = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
        with self._send_lock:
            try:
                self.sock.sendall(struct.pack("!Q", len(data)) + data)
            except OSError as error:
                raise WorkerError(f"the connection to the other process failed: {error}") from error

    def _recv(self) -> dict:
        with self._recv_lock:
            header = self._read(8)
            (size,) = struct.unpack("!Q", header)
            return pickle.loads(self._read(size))

    def _read(self, size: int) -> bytes:
        chunks, remaining = [], size
        while remaining:
            try:
                chunk = self.sock.recv(min(remaining, 1 << 22))
            except OSError as error:
                raise WorkerError(f"the connection to the other process failed: {error}") from error
            if not chunk:
                raise WorkerError("the other process closed the connection")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def request(self, message: dict) -> Any:
        rid = next(self._ids)
        self._send({**message, "id": rid})
        while True:
            incoming = self._recv()
            if incoming.get("reply") == rid:
                if "error" in incoming:
                    _raise_remote(incoming["error"])
                return incoming["result"]
            if "reply" in incoming:
                raise WorkerError(f"out-of-order reply {incoming['reply']} while waiting for {rid}")
            self.serve(incoming)

    def serve(self, message: dict) -> None:
        try:
            result = self.handler(message)
            reply = {"reply": message["id"], "result": result}
        except BaseException as exc:  # reported to the requester, which raises it
            reply = {"reply": message["id"], "error": _describe(exc)}
        self._send(reply)

    def serve_forever(self) -> None:
        while True:
            message = self._recv()
            if message.get("op") == "shutdown":
                return
            self.serve(message)


# --------------------------------------------------------------------------- #
# ComfyUI's side
# --------------------------------------------------------------------------- #
class Worker:
    """One pack's worker process, as ComfyUI sees it."""

    def __init__(self, pack_root: str) -> None:
        self.pack_root = pack_root
        parent, child = socket.socketpair()
        comfy_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([comfy_root] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
        env.pop(ENV, None)
        env.pop(LEGACY_ENV, None)
        # Imported, not run as __main__, so what it pickles names this module.
        command = [sys.executable, "-c", "from comfy_api.latest.pack_isolation import _worker_main; _worker_main()"]
        if sys.platform == "win32":
            self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, env=env, cwd=comfy_root)
            _share_socket(child, self.proc)
        else:
            self.proc = subprocess.Popen(command + [str(child.fileno())], pass_fds=(child.fileno(),), env=env, cwd=comfy_root)
        child.close()
        self.channel = Channel(parent, self._handle)
        self._calls: Optional[tuple] = None  # (runtime, loop) while one of this pack's nodes runs
        self._call_error: Optional[BaseException] = None
        self._request({"op": "hello", "torch_settings": _torch_settings(), "folders": _folder_layout()})
        logger.info("custom nodes from %s run in worker process %d", pack_root, self.proc.pid)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def _request(self, message: dict) -> Any:
        """A request to the worker. A connection that failed can't be used again, so the worker is
        stopped and the pack's next node gets a new one."""
        try:
            return self.channel.request(message)
        except WorkerError:
            self.proc.kill()
            self.proc.wait()
            raise

    def stop(self) -> None:
        if self.alive():
            try:
                self.channel._send({"op": "shutdown", "id": 0})
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()

    # requests from the worker
    def _handle(self, message: dict) -> Any:
        if message.get("op") != "call":
            raise WorkerError(f"unexpected request {message.get('op')!r} from a pack's process")
        if self._calls is None:
            raise WorkerError("an engine call arrived while none of this pack's nodes is running")
        runtime, loop = self._calls
        try:
            return asyncio.run_coroutine_threadsafe(self._call(message, runtime), loop).result()
        except BaseException as exc:
            self._call_error = exc
            raise

    async def _call(self, message: dict, runtime: Any) -> Any:
        from comfy_api.latest import _runtime

        args = [self._closure(a) if isinstance(a, Closure) else a for a in message["args"]]
        kwargs = {k: self._closure(v) if isinstance(v, Closure) else v for k, v in message["kwargs"].items()}
        with _runtime.bind_runtime(runtime.refs, runtime.ctx, runtime.ops, pack=runtime.pack, plan=runtime.plan):
            result = await runtime.ops.apply("engine.call", None, {"name": message["name"], "args": args, "kwargs": kwargs})
        return self._out(result, runtime)

    def _closure(self, closure: Closure):
        worker = self

        def call_in_pack_process(*args: Any, **kwargs: Any) -> Any:
            if not worker.alive():
                raise WorkerError(f"the worker for {worker.pack_root} is gone")
            return worker._request({"op": "invoke", "closure": closure.id, "args": args, "kwargs": kwargs,
                                    "mode": _call_mode()})
        call_in_pack_process.__qualname__ = f"pack function {closure.id}"
        return call_in_pack_process

    def _out(self, value: Any, runtime: Any) -> Any:
        from comfy_api.latest import _runtime

        def create(obj: Any) -> Any:
            kind = _runtime.handle_kind(obj)
            return _create_sync(runtime.refs, kind, obj)
        return to_worker(value, runtime.refs, create)

    # a node method
    def run(self, plan: Any, runtime: Any, loop: Any, module_file: str, locator: tuple) -> Any:
        self._calls, self._call_error = (runtime, loop), None
        try:
            message = {"op": "execute", "module": plan.node_module, "file": module_file, "locator": locator,
                       "method": plan.method, "inputs": self._out(dict(plan.inputs or {}), runtime),
                       "hidden": self._out({"prompt": plan.prompt, "unique_id": plan.node_id,
                                            "extra_pnginfo": plan.extra_pnginfo}, runtime),
                       "pack": plan.node_module, "settings": _node_settings(), "mode": _call_mode(),
                       "package_file": _package_file(plan.node_module)}
            try:
                return self._request(message)
            except RemoteError:
                if self._call_error is not None:  # e.g. the user's interrupt, raised by an engine call
                    raise self._call_error
                raise
        finally:
            self._calls = None


def _node_settings() -> dict:
    from comfy.cli_args import args
    return {"disable_metadata": bool(args.disable_metadata)}


def _create_sync(refs: Any, kind: str, obj: Any) -> Any:
    """``refs.create`` is async for providers that need it; the in-process table isn't."""
    coro = refs.create(kind, obj)
    try:
        coro.send(None)
    except StopIteration as done:
        return done.value
    coro.close()
    raise RuntimeError("this ref table can't create handles synchronously")


class PackIsolationBackend:
    """The execution backend pack isolation installs."""

    def __init__(self) -> None:
        self.workers: dict[str, Worker] = {}
        self._lock = threading.Lock()
        self._in_process_noted: set[str] = set()
        atexit.register(self.stop)

    def stop(self) -> None:
        for worker in list(self.workers.values()):
            worker.stop()
        self.workers.clear()

    def worker_for(self, pack_root: str) -> Worker:
        with self._lock:
            worker = self.workers.get(pack_root)
            if worker is None or not worker.alive():
                if worker is not None:
                    logger.warning("worker for %s exited (%s); starting a new one", pack_root, worker.proc.returncode)
                worker = self.workers[pack_root] = Worker(pack_root)
            return worker

    async def dispatch(self, plan: Any, local_call: Any, runtime: Any = None) -> Any:
        node_class = _node_class(plan)
        pack_root = _pack_root(node_class) if node_class is not None else None
        if pack_root is None:
            return await local_call()
        if not getattr(node_class, "ENGINE_API", False):
            if pack_root not in self._in_process_noted:
                self._in_process_noted.add(pack_root)
                logger.info("%s: nodes without ENGINE_API run in ComfyUI's process", pack_root)
            return await local_call()
        return await self.run_class(node_class, pack_root, plan, runtime)

    async def run_class(self, node_class: Any, pack_root: str, plan: Any, runtime: Any) -> Any:
        """Run one method of ``node_class`` in ``pack_root``'s worker. Returns a NodeOutput."""
        locator = _locator(node_class)
        module_file = inspect.getsourcefile(sys.modules[node_class.__module__])
        loop = asyncio.get_running_loop()
        worker = self.worker_for(pack_root)
        output = await asyncio.to_thread(worker.run, plan, runtime, loop, module_file, locator)
        if output["kind"] == "plain":
            return output["value"]
        from comfy_api.v4 import io
        return io.NodeOutput(*output["result"], ui=output["ui"], expand=output["expand"],
                             block_execution=output["block_execution"])


def _locator(node_class: Any) -> tuple:
    """How the worker finds the class: the class itself, or for a per-run clone
    (``PREPARE_CLASS_CLONE``) the class it clones. Classes a factory made at import are found
    through the module's ``NODE_CLASS_MAPPINGS`` or its globals. Never a class from another module."""
    module_name = getattr(node_class, "__module__", "")
    module = sys.modules.get(module_name)
    for candidate in getattr(node_class, "__mro__", (node_class,)):
        if getattr(candidate, "__module__", None) != module_name:
            break
        target: Any = module
        for part in candidate.__qualname__.split("."):
            target = getattr(target, part, None)
        if target is candidate:
            return ("qualname", candidate.__qualname__)
        for key, value in (getattr(module, "NODE_CLASS_MAPPINGS", None) or {}).items():
            if value is candidate:
                return ("mapping", key)
        for key, value in vars(module).items():
            if value is candidate:
                return ("global", key)
    raise TypeError(f"{node_class.__qualname__} can't be found by a worker process: it isn't a module "
                    f"attribute or in {module_name}.NODE_CLASS_MAPPINGS")


def _node_class(plan: Any) -> Any:
    """The class a plan is for: ``node_type`` is its name in ``node_module``."""
    module = sys.modules.get(plan.node_module)
    cls = getattr(module, plan.node_type, None) if module is not None else None
    if isinstance(cls, type):
        return cls
    import nodes
    for candidate in nodes.NODE_CLASS_MAPPINGS.values():
        if getattr(candidate, "__module__", None) == plan.node_module and getattr(candidate, "__name__", None) == plan.node_type:
            return candidate
    return None


def _pack_root(node_class: Any) -> Optional[str]:
    """The custom-node directory a class was loaded from, or None for ComfyUI's own nodes."""
    import nodes
    try:
        path = os.path.abspath(inspect.getsourcefile(node_class) or "")
    except TypeError:
        return None
    for root in nodes.LOADED_MODULE_DIRS.values():
        root = os.path.abspath(root)
        if path == root or path.startswith(root + os.sep):
            return root
    return None


def _package_file(module_name: str) -> Optional[str]:
    """The ``__init__.py`` of the pack's package as ComfyUI imported it, so the worker imports it the same way."""
    package = sys.modules.get(module_name.split(".")[0])
    return getattr(package, "__file__", None)


def register(providers: Any) -> None:
    """Entry point for ``_runtime.load_execution_provider``."""
    providers.execution_backend = PackIsolationBackend()


# --------------------------------------------------------------------------- #
# The worker's side
# --------------------------------------------------------------------------- #
class _WorkerOps:
    """``eng`` calls in the worker go to ComfyUI."""

    out_of_process = True

    def __init__(self, state: "_WorkerState") -> None:
        self.state = state

    def supports(self, op: str) -> bool:
        return op == "engine.call"

    async def apply(self, op: str, subject: Any, params: dict) -> Any:
        if op != "engine.call":
            from comfy_api.latest._runtime import CallNotSupported
            raise CallNotSupported(f"unknown operation {op!r}")
        functions = self.state.functions
        args = from_worker(list(params.get("args", ())), functions, "argument")
        kwargs = from_worker(dict(params.get("kwargs") or {}), functions, "argument")
        return self.state.channel.request({"op": "call", "name": params["name"], "args": args, "kwargs": kwargs})


class _WorkerState:
    def __init__(self) -> None:
        self.channel: Optional[Channel] = None
        self.functions: dict[int, Any] = {}

    def handle(self, message: dict) -> Any:
        op = message.get("op")
        if op == "hello":
            _apply_torch_settings(message.get("torch_settings"))
            _apply_folder_layout(message.get("folders"))
            return {"pid": os.getpid()}
        if op == "invoke":
            fn = self.functions.get(message["closure"])
            if fn is None:
                raise KeyError(f"pack function {message['closure']} is gone")
            with _in_mode(message.get("mode")):
                result = fn(*message["args"], **message["kwargs"])
            return from_worker(result, None, "a sampling function's result")
        if op == "execute":
            return asyncio.run(self.execute(message))
        raise WorkerError(f"unknown request {op!r}")

    async def execute(self, message: dict) -> dict:
        from comfy_api.latest import _io, _runtime, _ui

        cls = _load_class(message["module"], message["file"], message["locator"], message.get("package_file"))
        hidden = message["hidden"]
        target: Any = cls
        if isinstance(cls, type) and issubclass(cls, _io.ComfyNode):
            target = cls.PREPARE_CLASS_CLONE({"hidden_inputs": {
                _io.Hidden.prompt: hidden["prompt"], _io.Hidden.unique_id: hidden["unique_id"],
                _io.Hidden.extra_pnginfo: hidden["extra_pnginfo"]}})
        fn = getattr(target, message["method"])
        settings = message.get("settings") or {}
        with _runtime.bind_runtime(_runtime.InProcessRefResolver(), None, _WorkerOps(self), pack=message["pack"]), \
                _ui.collect_writes(settings.get("disable_metadata")) as pending, _in_mode(message.get("mode")):
            result = fn(**restore(message["inputs"]))
            if inspect.isawaitable(result):
                result = await result
            await pending.flush()
        # A node's own output (execute, or ComfyUI's EXECUTE_NORMALIZED wrapper around it), or the
        # plain value of another method such as check_lazy_status.
        is_output = hasattr(result, "result") or (isinstance(result, dict) and ({"result", "ui"} & set(result)))
        if not is_output:
            return {"kind": "plain", "value": keep(result)}  # a method's value, or a V1 node's tuple
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


def _import_package(name: str, init_file: Optional[str]) -> None:
    """Import the pack's package as ComfyUI did, so its ``__init__`` runs and modules that read their
    package's state (``sys.modules[__package__].NODE_CLASS_MAPPINGS``) find it."""
    if not init_file or name in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(name, init_file, submodule_search_locations=[os.path.dirname(init_file)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise


def _load_class(module_name: str, module_file: str, locator: tuple, package_file: Optional[str] = None) -> Any:
    """Import the node's module under the name ComfyUI gave it, after the pack's package."""
    _import_package(module_name.split(".")[0], package_file)
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


def _share_socket(sock: socket.socket, proc: subprocess.Popen) -> None:
    """Windows has no ``pass_fds``: the socket is duplicated for the worker once it has a pid, and the
    worker opens it from what arrives on its stdin."""
    try:
        with proc.stdin:
            proc.stdin.write(sock.share(proc.pid))
    except OSError as error:  # the first request to the worker then fails, and says so
        logger.warning("couldn't hand worker process %d its connection: %s", proc.pid, error)


def _worker_socket() -> socket.socket:
    if sys.platform == "win32":
        return socket.fromshare(sys.stdin.buffer.read())
    return socket.socket(fileno=int(sys.argv[1]))


def _worker_main() -> None:
    logging.basicConfig(level=logging.INFO, format="[pack worker %(process)d] %(message)s")
    state = _WorkerState()
    state.channel = Channel(_worker_socket(), state.handle)
    try:
        state.channel.serve_forever()
    except WorkerError:
        pass  # ComfyUI went away

