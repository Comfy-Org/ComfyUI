"""ComfyUI's side: one worker process per custom-node pack, and the execution backend."""
from __future__ import annotations

import asyncio
import atexit
import itertools
import inspect
import logging
import os
import socket
import subprocess
import sys
import threading
from typing import Any, Optional

from . import without_settings
from .boundary import Closure, to_worker
from .channel import Channel, RemoteError, WorkerError

logger = logging.getLogger(__name__)


class Worker:
    """One pack's worker process, as ComfyUI sees it."""

    def __init__(self, pack_root: str, port: Optional[int] = None, token: Optional[str] = None) -> None:
        """A worker ComfyUI starts, or with ``port`` a pack server already listening there."""
        self.pack_root = pack_root
        self.port, self._token = port, token
        self._host_values = {}  # id -> Opaque / EngineFunction handed to a running pack function
        self._host_order = []
        self.proc: Optional[subprocess.Popen] = None
        self._calls: Optional[tuple] = None  # (runtime, loop) while one of this pack's nodes runs
        self._call_error: Optional[BaseException] = None
        self._side: Optional[Channel] = None  # schemas, INPUT_TYPES, HTTP: not behind a running node
        self._side_lock = threading.Lock()
        if port is not None:
            from . import server

            if token is None:  # started by hand (scripts/pack.sh): the token is in its run file
                info = server.read_info(port)
                if info is None or not info.get("token"):
                    raise WorkerError(f"no pack server is running on port {port}")
                self._token = info["token"]
            self.channel = Channel(server.connect(port, self._token), self._handle)
            self.pid = self._hello()
            logger.info("custom nodes from %s run in the pack server on port %d (pid %d)", pack_root, port, self.pid)
            return
        parent, child = socket.socketpair()
        # comfy_api/_impl/isolation/host.py -> ComfyUI's root
        comfy_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        env = without_settings(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([comfy_root] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
        self.proc = subprocess.Popen(
            # Imported, not run as __main__, so what it pickles names this module.
            [sys.executable, "-c", "import sys; from comfy_api._impl.isolation.worker import _worker_main; "
             "_worker_main(int(sys.argv[1]))", str(child.fileno())],
            pass_fds=(child.fileno(),), env=env, cwd=comfy_root)
        child.close()
        self.channel = Channel(parent, self._handle)
        self.pid = self.proc.pid
        self._hello()
        logger.info("custom nodes from %s run in worker process %d", pack_root, self.proc.pid)

    def side_request(self, message: dict) -> Any:
        """A request on a second connection, answered even while one of the pack's nodes runs."""
        from . import server

        with self._side_lock:
            if self._side is None:
                self._side = Channel(server.connect(self.port, self._token), None)
            return self._side.request(message)

    def alive(self) -> bool:
        """Whether the pack's process is still there: its connection is open (a process that just
        exited closes it before it's reaped) and, for a worker ComfyUI started, it hasn't exited."""
        if self._closed or (self.proc is not None and self.proc.poll() is not None):
            return False
        try:
            return self.channel.sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) != b""
        except BlockingIOError:
            return True
        except OSError:
            return False

    _closed = False

    def _hello(self) -> int:
        """Meet the pack's process; use shared memory for large tensors if it sees ours."""
        from .channel import probe_shared_memory
        name, release = probe_shared_memory()
        try:
            from . import torch_state  # the pack computes with ComfyUI's numeric settings, not PyTorch's defaults
            reply = self.channel.request({"op": "hello", "shm_probe": name, "torch_settings": torch_state.snapshot()})
        finally:
            release()
        self.channel.shared_memory = bool(reply.get("shm"))
        return reply["pid"]

    def release_segments(self) -> None:
        """Segments this side sent that the pack never read, and any the pack left behind."""
        from .channel import sweep
        for channel in (self.channel, self._side):
            if channel is not None:
                channel.release_unread()
        if getattr(self, "pid", None):
            sweep(self.pid)

    def stop(self) -> None:
        if self.proc is None:  # a pack server: disconnect; whoever started it stops it
            self._closed = True
            self.release_segments()
            for channel in (self.channel, self._side):
                if channel is not None:
                    channel.sock.close()
            return
        if self.alive():
            try:
                self.channel._send({"op": "shutdown", "id": 0})
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
        self.release_segments()

    # requests from the worker
    def _handle(self, message: dict) -> Any:
        if message.get("op") == "host_call":  # a pack function calling an engine function it was given
            return self._host_call(message)
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
        from comfy_api._impl import _runtime

        args = [self._closure(a) if isinstance(a, Closure) else a for a in message["args"]]
        kwargs = {k: self._closure(v) if isinstance(v, Closure) else v for k, v in message["kwargs"].items()}
        with _runtime.bind_runtime(runtime.refs, runtime.ctx, runtime.ops, pack=runtime.pack, plan=runtime.plan):
            result = await runtime.ops.apply("engine.call", None, {"name": message["name"], "args": args, "kwargs": kwargs})
        return self._out(result, runtime)

    def _closure(self, closure: Closure):
        worker = self

        def call_in_pack_process(*args: Any, **kwargs: Any) -> Any:
            return worker._invoke({"op": "invoke", "closure": closure.id}, args, kwargs)
        call_in_pack_process.__qualname__ = f"pack function {closure.id}"
        return call_in_pack_process

    def call_registered(self, kind: str, name: str, *args: Any, **kwargs: Any) -> Any:
        """Run a sampler or scheduler the pack registered (``ComfyExtension.get_samplers`` ...)."""
        return self._invoke({"op": "registered", "kind": kind, "name": name}, args, kwargs)

    def _invoke(self, message: dict, args: tuple, kwargs: dict) -> Any:
        if not self.alive():
            raise WorkerError(f"the worker for {self.pack_root} is gone")
        mark = len(self._host_order)
        try:
            from . import torch_state
            result = self.channel.request({**message, "args": self._to_pack(list(args)),
                                           "kwargs": self._to_pack(dict(kwargs)), "mode": torch_state.call_mode()})
            return self._from_pack(result)
        finally:  # what this call handed out lives as long as the call
            for hid in self._host_order[mark:]:
                self._host_values.pop(hid, None)
            del self._host_order[mark:]

    # Engine values and engine functions inside what a pack function is given (``_engine.call_view``)
    _host_values: dict
    _host_order: list

    def _to_pack(self, value: Any) -> Any:
        from comfy_api._impl import _engine
        from .boundary import HostValue
        if isinstance(value, (_engine.Opaque, _engine.EngineFunction)):
            hid = next(_host_ids)
            self._host_values[hid] = value
            self._host_order.append(hid)
            return HostValue(hid, isinstance(value, _engine.EngineFunction))
        if type(value) is dict:
            return {k: self._to_pack(v) for k, v in value.items()}
        if type(value) in (list, tuple):
            return type(value)(self._to_pack(v) for v in value)
        return value

    def _from_pack(self, value: Any) -> Any:
        from .boundary import HostValue
        if isinstance(value, Closure):  # a pack function handed to an engine function
            return self._closure(value)
        if isinstance(value, HostValue):
            try:
                return self._host_values[value.id]
            except KeyError:
                raise WorkerError("a pack function used an engine value after the call it came with") from None
        if type(value) is dict:
            return {k: self._from_pack(v) for k, v in value.items()}
        if type(value) in (list, tuple):
            return type(value)(self._from_pack(v) for v in value)
        return value

    def _host_call(self, message: dict) -> Any:
        from .boundary import HostValue
        fn = self._from_pack(HostValue(message["value"], True))
        if not callable(fn):
            raise WorkerError("a pack function called an engine value that isn't a function")
        return self._to_pack(fn(*self._from_pack(message["args"]), **self._from_pack(message["kwargs"])))

    def _out(self, value: Any, runtime: Any) -> Any:
        from comfy_api._impl import _runtime

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
                       "pack": plan.node_module, "settings": _node_settings(), "mode": _call_mode()}
            try:
                return self.channel.request(message)
            except RemoteError:
                if self._call_error is not None:  # e.g. the user's interrupt, raised by an engine call
                    raise self._call_error
                raise
        finally:
            self._calls = None


_host_ids = itertools.count(1)


def _call_mode() -> dict:
    from . import torch_state  # ComfyUI runs nodes in inference mode: so does the pack's process
    return torch_state.call_mode()


def _node_settings() -> dict:
    """ComfyUI settings a node's ``ui`` savers follow wherever the node runs."""
    try:
        from comfy.cli_args import args
    except ImportError:
        return {}
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


class OutOfProcessBackend:
    """The execution backend ``COMFY_PACK_ISOLATION`` installs."""

    def __init__(self) -> None:
        self.workers: dict[str, Worker] = {}
        self.packs: Any = None  # packs.PackManager, once ComfyUI sets packs up (setup_packs)
        self._lock = threading.Lock()
        self._in_process_noted: set[str] = set()
        atexit.register(self.stop)

    def setup_packs(self, folders: Optional[list] = None) -> None:
        """Pack isolation: find the packs and what they have, import none of them."""
        from .packs import PackManager
        self.packs = PackManager(self)
        self.packs.setup(folders)

    async def ensure_node_types(self, node_types: set) -> None:
        """Before a prompt is validated: start the packs its nodes come from, registering their nodes."""
        if self.packs is not None:
            await self.packs.ensure_node_types(node_types)

    def stop(self) -> None:
        for worker in list(self.workers.values()):
            worker.stop()
        self.workers.clear()
        if self.packs is not None:
            self.packs.stop()

    def worker_for(self, pack_root: str) -> Worker:
        with self._lock:
            worker = self.workers.get(pack_root)
            if worker is None or not worker.alive():
                if worker is not None:
                    logger.warning("worker for %s exited; starting a new one", pack_root)
                worker = self.workers[pack_root] = Worker(pack_root)
            return worker

    async def dispatch(self, plan: Any, local_call: Any, runtime: Any = None) -> Any:
        node_class = _node_class(plan)
        if node_class is not None and getattr(node_class, "ISOLATED_NODE", None) and self.packs is not None:
            with self.packs.using(node_class.ISOLATED_NODE[0]):
                worker = await asyncio.to_thread(self.packs.worker_for_node, node_class)
                return await self._run(worker, plan, runtime, None, ("node", node_class.ISOLATED_NODE[1]))
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
        return await self._run(self.worker_for(pack_root), plan, runtime, module_file, locator)

    async def _run(self, worker: Worker, plan: Any, runtime: Any, module_file: Optional[str], locator: tuple) -> Any:
        loop = asyncio.get_running_loop()
        output = await asyncio.to_thread(worker.run, plan, runtime, loop, module_file, locator)
        if output["kind"] == "plain":
            return output["value"]
        from comfy_api.index import io
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
    by_id = nodes.NODE_CLASS_MAPPINGS.get(plan.node_type)  # node ids needn't be Python names
    if by_id is not None and (not plan.node_module or getattr(by_id, "__module__", None) == plan.node_module):
        return by_id
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
