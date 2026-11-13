"""A pack process: one V4 pack, imported here and never in ComfyUI.

ComfyUI starts this file as a script, not with `python -m`: `-m` would import `comfy_api.v4`
first, and with it `folder_paths`, which reads ComfyUI's arguments at import. So this file
imports ComfyUI's modules inside its functions, after `main` has applied ComfyUI's arguments.

The launcher writes the start message to stdin: ComfyUI's arguments, the folders this process
uses as ComfyUI's output, input, temp and user folders, the epoch (how many of this pack's
processes have stopped before) and the socket. Before any pack code runs, each `eng` group gets
an instance whose calls are made in ComfyUI, and `util.folders` asks ComfyUI about model folders.
ComfyUI's requests (load the pack, run a node's method, give a node's schema, answer a route,
call one of the pack's objects) are served as tasks, which ComfyUI can cancel, on two lanes: the
main thread runs the "exec" lane, for what ComfyUI's prompt thread asks, and one more thread the
"server" lane, for what its server thread asks.

The process runs in a process group (and session) of its own, so the terminal's Ctrl+C reaches
only ComfyUI. When the socket closes, because ComfyUI has ended the pack or has itself ended, the
process ends every program the pack started and then its group, itself included.

What the pack writes to `sys.stdout` and `sys.stderr` and logs with `logging` is sent to ComfyUI,
which writes it where its own goes. The file descriptors are ComfyUI's: what is written to them
directly goes to its console.
"""

import asyncio
import contextlib
import functools
import importlib.util
import inspect
import io
import itertools
import logging
import os
import pickle
import signal
import socket
import sys
import threading
import types

import psutil
import torch

OFFERED = ("to", "models", "cleanup", "generate_noise")


class Pack:
    """The pack process's side of the channel: the objects ComfyUI holds, the nodes, the lanes."""

    peer = "ComfyUI"
    poll = None

    def __init__(self, sock: socket.socket, epoch: int):
        from comfy_api.v4._channel import Channel

        self.channel = Channel(sock, self)
        # ComfyUI counts this pack's processes; an object's key names the process that made it.
        self.epoch = epoch
        self.objects = {}
        self.object_ids = itertools.count()
        self.io_types = {}
        self.nodes = {}
        self.routes = []
        # chain -> the task serving ComfyUI's top-level request of that chain
        self.requests: dict[int, asyncio.Task] = {}
        self.loops: dict[str, asyncio.AbstractEventLoop] = {}
        self.lanes: set[threading.Thread] = set()

    def keep(self, obj, chain: int) -> tuple[int, int]:
        key = (self.epoch, next(self.object_ids))
        self.objects[key] = obj
        return key

    def forget(self, keys: list) -> None:
        """Let go of what `keep` kept for a message that was not sent."""
        for key in keys:
            del self.objects[key]

    def hold(self, obj, chain: int) -> None:
        """ComfyUI's objects are held for the whole request already."""

    def resolve(self, key: tuple[int, int], chain: int | None = None):
        if key[0] != self.epoch:
            raise LookupError(
                "this object was made by a pack process that has stopped; "
                "run the node that made it again"
            )
        return self.objects[key]

    def offers(self, obj) -> tuple[tuple[str, ...], dict]:
        """What ComfyUI uses of `obj` besides calling it, as `obj` has it now: the methods it calls
        on a model patch (`to`, `models`, `cleanup`) or a NOISE (`generate_noise`), and the seed."""
        methods = tuple(name for name in OFFERED if callable(getattr(obj, name, None)))
        return methods, {"seed": obj.seed} if hasattr(obj, "seed") else {}

    def remote(self, key, type_name: str, methods: tuple, values: dict):
        from comfy_api.v4._codec import Remote

        return Remote(self, key, type_name, methods, values)

    def release(self, keys: list) -> None:
        """ComfyUI no longer holds these objects; drop them on the exec lane, not the reader."""
        self.loops["exec"].call_soon_threadsafe(self._release, keys)

    def _release(self, keys: list) -> None:
        for key in keys:
            # A key of an earlier process of this pack is not here.
            self.objects.pop(key, None)

    def cancel(self, chain: int, lane: str) -> None:
        self.loops[lane].call_soon_threadsafe(self._cancel, chain)

    def _cancel(self, chain: int) -> None:
        task = self.requests.get(chain)
        if task is not None:
            task.cancel()

    def request(self, target: tuple, args: tuple = (), kwargs: dict | None = None):
        """Call ComfyUI from the lane thread that runs the current request, and wait."""
        from comfy_api.v4._channel import current_chain

        chain = current_chain(self)
        if chain is None or threading.current_thread() not in self.lanes:
            raise RuntimeError(
                "In a pack process, ComfyUI can be called only while it runs one of the pack's "
                "nodes or loads the pack, from the thread that runs it."
            )
        return self.channel.call(chain, target, args, kwargs or {})

    def serve(self, target: tuple, args: tuple, kwargs: dict):
        """Serve a call nested in one this thread is waiting on: ComfyUI calls one of our objects,
        or a method it offered."""
        function = self.resolve(target[1])
        if len(target) == 3:
            function = getattr(function, target[2])
        return function(*args, **kwargs)

    def schedule(self, call: dict) -> None:
        self.loops[call["lane"]].call_soon_threadsafe(self._start, call)

    def stopped(self) -> None:
        """ComfyUI has gone: end the programs the pack started, also those that left this
        process's group, and then the group, this process included."""
        sys.stdout.flush()
        sys.stderr.flush()
        for program in psutil.Process().children(recursive=True):
            with contextlib.suppress(psutil.Error):
                program.kill()
        os.killpg(0, signal.SIGKILL)

    def _start(self, call: dict) -> None:
        """Serve ComfyUI's request as a task of this lane's loop, so that ComfyUI can cancel it."""
        task = asyncio.get_running_loop().create_task(self._serve_request(call))
        self.requests.setdefault(call["chain"], task)
        task.add_done_callback(functools.partial(self._finished, call))

    def _finished(self, call: dict, task: asyncio.Task) -> None:
        if self.requests.get(call["chain"]) is task:
            del self.requests[call["chain"]]
        if task.cancelled():
            self.channel.reply(call, error=asyncio.CancelledError())
        elif task.exception() is not None:
            self.channel.reply(call, error=task.exception())
        else:
            self.channel.reply(call, task.result())

    async def _serve_request(self, call: dict):
        from comfy_api.v4._channel import enter_chain

        enter_chain(self, call["chain"])
        with (
            torch.inference_mode(call["inference"]),
            torch.set_grad_enabled(call["grad"]),
        ):
            args, kwargs = self.channel.decode(call)
        return await _in_mode(
            self._request(call["target"], args, kwargs), call["grad"], call["inference"]
        )

    async def _request(self, target: tuple, args: tuple, kwargs: dict):
        match target:
            case ("load",):
                return await self._load(*args)
            case ("node", node_id, method):
                return await self._run(node_id, method, *args)
            case ("route", index):
                return await self.routes[index](*args)
            case ("schema", node_id):
                return self.nodes[node_id].GET_SCHEMA()
        value = self.serve(target, args, kwargs)
        return await value if inspect.isawaitable(value) else value

    async def _load(self, entry: str, module_name: str) -> dict:
        """Import the pack as ComfyUI's loader would, and describe its nodes, routes and web
        folder."""
        from app import governance

        spec = governance.pack_module_spec(module_name, entry)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        extension = await _extension(module, entry)
        await extension.on_load()
        nodes = [self._node(node_cls) for node_cls in await extension.get_node_list()]
        # The handlers stay here; ComfyUI asks for a route by its place in this list.
        routes = await extension.get_routes()
        self.routes = [route.get("handler") for route in routes]
        return {
            "nodes": nodes,
            "routes": [(route.get("method"), route.get("path")) for route in routes],
            "web_directory": getattr(module, "WEB_DIRECTORY", None),
        }

    def _node(self, node_cls: type) -> tuple:
        """Keep a node class; its schema, and the arguments of each method ComfyUI's executor
        calls that it overrides."""
        from comfy_api.internal import first_real_override

        schema = node_cls.GET_SCHEMA()
        self.nodes[schema.node_id] = node_cls
        methods = {
            name: _arguments(method)
            for name in ("validate_inputs", "fingerprint_inputs", "check_lazy_status")
            if (method := first_real_override(node_cls, name)) is not None
        }
        return schema, methods

    async def _run(
        self,
        node_id: str,
        method: str,
        inputs: dict,
        hidden: dict,
        prefix: tuple | None = None,
    ):
        """Run a method of a node as ComfyUI's executor does, on a clone holding the hidden inputs,
        with the prefix the executor gives the nodes it expands into."""
        from comfy_api.internal import make_locked_method_func
        from comfy_execution.graph_utils import GraphBuilder

        if method == "execute" and prefix is not None:
            GraphBuilder.set_default_prefix(*prefix)
        node_cls = self.nodes[node_id]
        clone = node_cls.PREPARE_CLASS_CLONE({"hidden_inputs": hidden})
        name = node_cls.FUNCTION if method == "execute" else method
        result = make_locked_method_func(node_cls, name, clone)(**inputs)
        if inspect.isawaitable(result):
            result = await result
        if (
            method == "execute"
            and result.ui is not None
            and not isinstance(result.ui, dict)
        ):
            result.ui = result.ui.as_dict()
        return result


async def _extension(module: types.ModuleType, entry: str):
    """The V4 extension the pack's `comfy_entrypoint` returns."""
    from comfy_api.v4 import ComfyExtension

    entrypoint = getattr(module, "comfy_entrypoint", None)
    extension = None if entrypoint is None else entrypoint()
    if inspect.isawaitable(extension):
        extension = await extension
    if not isinstance(extension, ComfyExtension):
        raise TypeError(
            f"{entry} is not a V4 pack: only a pack whose comfy_entrypoint returns a "
            "comfy_api.v4.ComfyExtension can run in its own process"
        )
    return extension


def _arguments(method) -> tuple[list[str], bool]:
    """What ComfyUI's executor reads from a method's signature: its argument names, and `**`."""
    spec = inspect.getfullargspec(method)
    return spec.args, spec.varkw is not None


@types.coroutine
def _in_mode(coroutine, grad: bool, inference: bool):
    """Run each step of `coroutine` in the grad and inference modes of the thread that asked.

    These modes belong to the thread, and other requests run on it between this one's steps.
    """
    value, error = None, None
    while True:
        with torch.inference_mode(inference), torch.set_grad_enabled(grad):
            try:
                step = (
                    coroutine.send(value) if error is None else coroutine.throw(error)
                )
            except StopIteration as finished:
                return finished.value
        try:
            value, error = (yield step), None
        except BaseException as thrown:
            value, error = None, thrown


def _forwarder(pack: Pack, target: tuple, function):
    """A function with `function`'s name, signature and docstring that asks ComfyUI instead."""
    if inspect.iscoroutinefunction(function):

        async def forward(self, *args, **kwargs):
            return pack.request(target, args, kwargs)

    else:

        def forward(*args, **kwargs):
            return pack.request(target, args, kwargs)

    return functools.wraps(function)(forward)


def _forward_groups(pack: Pack) -> None:
    """Give each `eng` group an instance whose calls are made in ComfyUI."""
    from comfy_api.v4 import ComfyAPI_v4

    for group, cls in ComfyAPI_v4.__annotations__.items():
        calls = {
            name: _forwarder(pack, ("api", group, name), method)
            for name, method in inspect.getmembers(cls, inspect.iscoroutinefunction)
            if not name.startswith("_")
        }
        cls.inject_instance(type(cls.__name__, (cls,), calls)())


# What `util.folders` asks ComfyUI: its table of model folders and the files in them. The rest
# runs here, on the output, input, temp and user folders of the start message, inside whatever
# confines this process.
FOLDERS_IN_COMFYUI = frozenset(
    {
        "add_model_folder_path",
        "base_path",
        "clear_filename_list_cache",
        "folder_names_and_paths",
        "get_filename_list",
        "get_filename_list_",
        "get_folder_paths",
        "get_full_path",
        "get_full_path_or_raise",
        "models_dir",
        "supported_pt_extensions",
    }
)


def _forward_folders(pack: Pack) -> None:
    """Make each function and attribute of `util.folders` about model folders ask ComfyUI;
    attributes come as copies."""
    from comfy_api.v4.util import folders

    for name, function in inspect.getmembers(folders, inspect.isfunction):
        if name in FOLDERS_IN_COMFYUI:
            setattr(folders, name, _forwarder(pack, ("folders", name), function))
    here = folders.__getattr__

    def read(name: str):
        if name == "folder_names_and_paths":
            return _folder_table(pack)
        if name in FOLDERS_IN_COMFYUI:
            return pack.request(("folders", name))
        return here(name)

    folders.__getattr__ = read


def _folder_table(pack: Pack) -> dict:
    """A copy of ComfyUI's table of model folders, whose entries hold tuples and frozen sets.
    Assigning an entry assigns it in ComfyUI's table too; any other change is refused."""

    def frozen(entry):
        if type(entry) is not tuple:
            return entry
        return tuple(frozenset(p) if type(p) is set else tuple(p) for p in entry)

    def assign(self, name: str, entry) -> None:
        pack.request(("folders", "folder_names_and_paths", name), (entry,))
        dict.__setitem__(self, name, frozen(entry))

    def refuse(self, *args):
        raise TypeError(
            "In a pack process, util.folders.folder_names_and_paths changes only by assigning "
            "an entry: folder_names_and_paths[name] = (directories, extensions)"
        )

    changes = (
        "__delitem__",
        "__ior__",
        "setdefault",
        "pop",
        "popitem",
        "update",
        "clear",
    )
    table = type(
        "folder_names_and_paths",
        (dict,),
        {"__setitem__": assign, **dict.fromkeys(changes, refuse)},
    )
    snapshot = pack.request(("folders", "folder_names_and_paths"))
    return table({name: frozen(entry) for name, entry in snapshot.items()})


class _Output(io.TextIOBase):
    """`sys.stdout` or `sys.stderr` of a pack process: what is written goes to ComfyUI's, or to
    the file descriptor once ComfyUI is gone. Its reader thread must not wait to send."""

    def __init__(self, channel, name: str, stream):
        self.channel, self.name, self.stream = channel, name, stream

    def write(self, text: str) -> int:
        from comfy_api.v4._channel import PackStopped

        if not isinstance(text, str):
            raise TypeError(f"write() argument must be str, not {type(text).__name__}")
        if threading.current_thread() is not self.channel.reader:
            with contextlib.suppress(PackStopped):
                self.channel.notify({"output": self.name, "text": str(text)})
                return len(text)
        return self.stream.write(text)

    def flush(self) -> None:
        self.stream.flush()

    def fileno(self) -> int:
        return self.stream.fileno()

    def isatty(self) -> bool:
        return self.stream.isatty()

    @property
    def encoding(self) -> str:
        return self.stream.encoding

    def __getattr__(self, name: str):
        # What else ComfyUI's stream has, such as `buffer` and `reconfigure`.
        return getattr(self.stream, name)


class _Log(logging.Handler):
    """Hands the pack's log records to ComfyUI's logging, which formats and writes them."""

    def __init__(self, channel):
        super().__init__()
        self.channel = channel

    def emit(self, record: logging.LogRecord) -> None:
        from comfy_api.v4._channel import PackStopped

        exc_text = record.exc_text
        if record.exc_info:
            exc_text = logging.Formatter().formatException(record.exc_info)
        fields = ("name", "levelno", "levelname", "created", "msecs", "pathname")
        fields += ("filename", "module", "lineno", "funcName", "stack_info")
        sent = {field: getattr(record, field) for field in fields}
        sent |= {"msg": record.getMessage(), "exc_text": exc_text}
        try:
            self.channel.notify({"output": "log", "record": sent})
        except PackStopped:
            sys.__stderr__.write(f"{record.getMessage()}\n")


def main() -> None:
    # The start message comes from ComfyUI, which is trusted; what the pack sends is checked.
    start = pickle.load(sys.stdin.buffer)
    sys.argv = start["argv"]
    from comfy.cli_args import args

    vars(args).update(start["args"])
    from app import governance

    # As ComfyUI's main does: under a custom-node policy a pack is run from its source only.
    governance.initialize()
    import folder_paths

    for name, path in start["folders"].items():
        getattr(folder_paths, f"set_{name}_directory")(path)
    sock = socket.socket(fileno=start["socket"])
    # A process this one starts must not hold the socket, or ComfyUI would not see this one end:
    # not one it executes, and not a fork, which lets go of the channel.
    sock.set_inheritable(False)
    pack = Pack(sock, start["epoch"])
    os.register_at_fork(after_in_child=pack.channel.after_fork)
    _forward_groups(pack)
    _forward_folders(pack)
    exec_loop, server_loop = asyncio.new_event_loop(), asyncio.new_event_loop()
    server = threading.Thread(
        target=_run_lane, args=(server_loop,), name="pack server lane", daemon=True
    )
    pack.loops = {"exec": exec_loop, "server": server_loop}
    pack.lanes = {threading.current_thread(), server}
    server.start()
    pack.channel.start("pack reader")
    sys.stdout = _Output(pack.channel, "stdout", sys.stdout)
    sys.stderr = _Output(pack.channel, "stderr", sys.stderr)
    logging.basicConfig(
        level=start["log_level"], handlers=[_Log(pack.channel)], force=True
    )
    asyncio.set_event_loop(exec_loop)
    _run_lane(exec_loop)


def _run_lane(loop: asyncio.AbstractEventLoop) -> None:
    """Run a lane's loop for good. `SystemExit` or `KeyboardInterrupt` raised by pack code stops
    the loop; ComfyUI gets it as the failure of the request that raised it, and the lane goes on."""
    while True:
        with contextlib.suppress(SystemExit, KeyboardInterrupt):
            loop.run_forever()


if __name__ == "__main__":
    # Import ComfyUI's modules from its folder, as main.py does, not from this file's.
    sys.path[0] = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    main()
