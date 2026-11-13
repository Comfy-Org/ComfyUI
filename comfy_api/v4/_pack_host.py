"""V4 packs that run in their own process: ComfyUI's side.

For a pack `_pack_processes.isolated` names, the loader calls `load`, which starts the
pack's process; the process imports the pack (ComfyUI never does) and answers with its nodes'
schemas, its routes and its web folder. `load` returns a module that stands for the pack, whose
extension lists a node class per schema; each class's methods run the pack's node in its process.
The loader then registers the pack as any V4 pack.

While the pack runs, its `eng` and `util.folders` calls are served on the thread, and in the
node's context, where they would run in process. ComfyUI objects the pack is given stay here;
the pack holds handles to them until the request they were given in ends. The pack's objects
that ComfyUI holds are released in the pack when ComfyUI lets go of them.

The process can stop: it crashed, or ComfyUI ended it. Every wait for it then fails at once, and
the next node run or route request starts it again. Each node class's fingerprint holds the
process's epoch, so ComfyUI's cache runs the pack's nodes again instead of serving what a
stopped process made. When the user cancels, a request the pack is busy with for a node of the
prompt is cancelled in the pack, and the process is killed if the request has not ended `GRACE`
seconds later; while the pack waits on ComfyUI that time does not run. ComfyUI starts the process
with the launcher (`_pack_launch`) and acts on it only through what the launcher returned: when
ComfyUI ends the process, or sees it stop, it kills it, which ends the programs the pack started
too. A pack whose process stops `CRASHES` times within `CRASH_WINDOW` seconds without ComfyUI
ending it is disabled: `_packs.remove_pack` takes it away, and each later use fails saying why.
Each stop ComfyUI did not cause, and each kill after a cancel, is logged.

What the pack prints or logs with Python goes where ComfyUI's own output goes; output written to
its file descriptors directly (native libraries, a crash) goes to ComfyUI's console only, as it
would in process.

A pack process cannot register a cache provider: ComfyUI's cache would call it at every lookup.
"""

import asyncio
import concurrent.futures
import contextlib
import contextvars
import dataclasses
import functools
import inspect
import itertools
import logging
import math
import os
import queue
import sys
import threading
import time
import types

from comfy.cli_args import args
from comfy.model_management import (
    interrupt_current_processing,
    processing_interrupted,
)
from comfy_api._exceptions import InterruptProcessingException
from comfy_api.latest import io
from comfy_execution.utils import get_executing_context

from . import _channel, _codec, _pack_launch, _pack_nodes, _pack_processes, _packs

GRACE = 5.0
POLL = 0.1
# How long ComfyUI waits for a node's schema before it uses the last one the pack sent, and
# then how long it asks no more.
SCHEMA_WAIT = 1.0
SCHEMA_PAUSE = 10.0
CRASHES = 3
CRASH_WINDOW = 600.0


def _entry(module_path: str) -> str:
    """The file the loader imports for a pack: the file itself, or its folder's `__init__.py`."""
    if os.path.isfile(module_path):
        return module_path
    return os.path.join(module_path, "__init__.py")


@contextlib.contextmanager
def _serving(record: "_Chain | None"):
    """While ComfyUI answers one of the pack's calls in `record`'s chain."""
    call = object()
    if record is not None:
        record.calls.add(call)
    try:
        yield
    finally:
        if record is not None:
            record.calls.discard(call)


@dataclasses.dataclass
class _Chain:
    """One request to the pack and the calls nested in it, as ComfyUI sees them."""

    id: int
    lane: str
    # The process the request went to, and the node ComfyUI was running when it began.
    process: _pack_launch.Process
    node: str | None
    # The ComfyUI objects the pack was given in this chain; its handles name them by index.
    # Also the pack's objects ComfyUI sent back in it, held until the pack is done with them.
    objects: list = dataclasses.field(default_factory=list)
    # Where the pack's calls are served when no thread here is waiting in this chain.
    loop: asyncio.AbstractEventLoop | None = None
    context: contextvars.Context | None = None
    inbox: queue.SimpleQueue | None = None
    # When ComfyUI has cancelled the request: the time by which it must have ended (`inf` until
    # ComfyUI has answered the calls the pack was waiting on).
    deadline: float | None = None
    # Whether the caller stopped waiting; the record then stays until the answer comes.
    abandoned: bool = False
    # The id of the top-level request, for one ComfyUI awaits.
    request: int | None = None
    # The pack's calls of this chain that ComfyUI has to answer (added and removed on different
    # threads, so a set), how many of ComfyUI's calls to the pack are nested in them, and the
    # calls the pack was waiting on when ComfyUI cancelled.
    calls: set = dataclasses.field(default_factory=set)
    nested: int = 0
    awaited: set = dataclasses.field(default_factory=set)
    # Whether the cancel has been sent to the pack.
    told: bool = False


class PackProcess:
    """ComfyUI's side of one pack's process, through its restarts."""

    poll = POLL

    def __init__(self, module_path: str, module_name: str):
        self.module_path, self.module_name = module_path, module_name
        self.peer = f"pack {os.path.basename(os.path.normpath(module_path))}"
        self.io_types = {}
        self.chains: dict[int, _Chain] = {}
        self.chain_ids = itertools.count(1)
        # Counts the processes that have stopped; part of each node's fingerprint.
        self.epoch = 0
        # The process is loaded and answers; the next use starts a new one when it is not.
        self.ready = False
        self.removed = False
        # Why the pack was disabled, if it was.
        self.disabled: str | None = None
        self.killed = False
        self.crashes: list[float] = []
        self.lock = threading.Lock()
        self.starting: concurrent.futures.Future | None = None
        # The node ids, overridden methods and routes the pack declared when it was loaded.
        self.layout = None
        self.schemas: dict[str, io.Schema] = {}
        self.schemas_paused_until = 0.0

    def keep(self, obj, chain: int) -> tuple[int, int]:
        objects = self.chains[chain].objects
        objects.append(obj)
        return chain, len(objects) - 1

    def forget(self, keys: list) -> None:
        """Let go of what `keep` kept for a message that was not sent."""
        for chain, index in keys:
            self.chains[chain].objects[index] = None

    def hold(self, obj, chain: int) -> None:
        # Until the pack has decoded the message that sends `obj` back, a release must not reach it.
        self.chains[chain].objects.append(obj)

    def resolve(self, key: tuple[int, int], chain: int):
        record = self.chains.get(key[0])
        if record is None or key[0] != chain:
            raise ValueError(
                "a handle from another request; it is valid only in the request it was given in"
            )
        return record.objects[key[1]]

    def offers(self, obj) -> tuple[tuple, dict]:
        """A pack can only call ComfyUI's objects, or pass them back."""
        return (), {}

    def remote(
        self, key, type_name: str, methods: tuple, values: dict
    ) -> _codec.Remote:
        """One of the pack's objects; the pack is told when ComfyUI no longer holds it. Of what
        the pack says it offers, only what ComfyUI uses of such an object is kept."""
        from ._pack_main import OFFERED

        methods = tuple(name for name in methods if name in OFFERED)
        values = {name: values[name] for name in ("seed",) if name in values}
        return _codec.Remote(
            self, key, type_name, methods, values, self.channel.release
        )

    def serve(self, target: tuple, args: tuple, kwargs: dict):
        record = self.chains.get(_channel.current_chain(self))
        with _serving(record):
            match target:
                case ("api", group, name):
                    return _call_api(group, name, args, kwargs)
                case ("folders", "folder_names_and_paths", name):
                    return _assign_folder(self.module_path, name, *args)
                case ("folders", name):
                    return _folders(self.module_path, name, args, kwargs)
                case ("call", key):
                    return self.resolve(key, record and record.id)(*args, **kwargs)
            raise LookupError(f"ComfyUI serves no {target!r}")

    def schedule(self, call: dict) -> None:
        """Serve a call no thread here waits for: on its chain's loop, in its chain's context."""
        record = self.chains.get(call["chain"])
        if record is None or record.loop is None:
            self.channel.drop(call)
            self.channel.reply(
                call, error=RuntimeError("the request this call belongs to has ended")
            )
            return

        def answer() -> None:
            try:
                self.channel.answer(call)
            finally:
                record.calls.discard(call["id"])

        # Until it is answered: a cancel meanwhile waits for it.
        record.calls.add(call["id"])
        record.inbox.put(functools.partial(record.context.run, answer))
        record.loop.call_soon_threadsafe(_channel.serve_queued, record.inbox)

    def output(self, message: dict) -> None:
        """Write what the pack printed or logged where ComfyUI's own output goes."""
        match message["output"]:
            case "stdout":
                sys.stdout.write(message["text"])
            case "stderr":
                sys.stderr.write(message["text"])
            case "log":
                # The pack process formats its message: ComfyUI formats nothing of it again.
                record = logging.makeLogRecord(
                    {**message["record"], "args": None, "exc_info": None}
                )
                logging.getLogger(record.name).handle(record)

    def tick(self) -> None:
        """While the process runs, every `POLL` seconds: pass the user's cancel on to the requests
        it would interrupt in one process, and kill the process if one outlives its deadline."""
        interrupted = processing_interrupted()
        for record in list(self.chains.values()):
            # What runs for a node of the prompt; not a route, a validation or a restart the
            # server thread asked for.
            if record.deadline is None and interrupted and record.node is not None:
                self._cancel(record)
            elif record.deadline is not None:
                self._follow(record)

    def _follow(self, record: _Chain) -> None:
        """Tell the pack of the cancel until it has been told, and kill the process if the
        request outlives its deadline."""
        if not record.told:
            self._tell(record)
        if record.abandoned and record.request not in self.channel.waiters:
            # Its answer came after its loop closed, so nothing settled it.
            self.chains.pop(record.id, None)
            return
        if record.deadline == math.inf:
            if record.nested or not record.awaited & record.calls:
                record.deadline = time.monotonic() + GRACE
            return
        if record.process is not self.process or time.monotonic() <= record.deadline:
            return
        if not self.killed:
            logging.warning(
                f"{self.peer} had not stopped {GRACE:g} s after the cancel; "
                "its process is ended."
            )
        self.kill()

    def ended(self) -> bool:
        return self.process.poll() is not None

    def stopped(self) -> None:
        """The process's socket has closed: end it with what it started, and reap it; whatever
        it made is gone."""
        self.process.kill()
        code = self.process.wait()
        with self.lock:
            self.epoch += 1
            if not self.killed:
                self.crashes.append(time.monotonic())
            self.ready = False
        if not self.killed:
            logging.warning(
                f"{self.peer} stopped (exit code {code}); "
                "it starts again when it is next used."
            )

    def kill(self) -> None:
        self.killed = True
        self.process.kill()

    def stop(self) -> None:
        """End the process for good: ComfyUI no longer has the pack."""
        self.removed = True
        self.kill()

    def schema(self, node_id: str) -> io.Schema:
        """The node's schema as the pack defines it now, as in process; or the last one the pack
        sent while its process is stopped, or if it does not answer within `SCHEMA_WAIT`."""
        if self.ready and time.monotonic() >= self.schemas_paused_until:
            try:
                self.schemas[node_id] = self.request(
                    ("schema", node_id), lane="server", timeout=SCHEMA_WAIT
                )
            except _channel.PackStopped:
                pass
            except TimeoutError:
                self.schemas_paused_until = time.monotonic() + SCHEMA_PAUSE
        return self.schemas[node_id]

    def request(
        self,
        target: tuple,
        args: tuple = (),
        kwargs: dict | None = None,
        lane: str | None = None,
        timeout: float | None = None,
    ):
        """Call the pack and wait on this thread, in the chain this thread is in, or a new one.

        A new chain goes to the pack's exec lane while ComfyUI runs a node, else to its server
        lane: a patched model freed on the server thread does not wait for the pack's node.
        """
        nested = self.chains.get(_channel.current_chain(self))
        lane = lane or ("exec" if get_executing_context() else "server")
        with contextlib.nullcontext(nested) if nested else self._chain(lane) as record:
            # The pack has the turn while it answers.
            record.nested += bool(nested)
            try:
                return self.channel.call(
                    record.id, target, args, kwargs or {}, record.lane, timeout
                )
            except (Exception, InterruptProcessingException) as error:
                raise self._failure(record, error)
            finally:
                record.nested -= bool(nested)

    async def run(self, target: tuple, args: tuple, lane: str = "exec"):
        """Call the pack in a new chain and await the answer, starting its process again first if
        it has stopped. The pack's calls are served here meanwhile."""
        await self._started()
        return await self._call(target, args, lane)

    async def _call(self, target: tuple, args: tuple, lane: str):
        future = asyncio.get_running_loop().create_future()
        with self._chain(lane, future.get_loop()) as record:
            deliver = functools.partial(self._deliver, record, future)
            try:
                record.request = self.channel.call_async(
                    record.id, target, args, lane, deliver
                )
            except _channel.PackStopped as error:
                raise self._failure(record, error)
            try:
                return await future
            except asyncio.CancelledError:
                # The caller stopped waiting, unless the answer had already come.
                if future.cancelled():
                    self._abandon(record)
                raise

    def _deliver(self, record: _Chain, future: asyncio.Future, reply: dict) -> None:
        """On the reader thread: hand the answer to the loop that awaits it."""
        try:
            future.get_loop().call_soon_threadsafe(self._settle, record, future, reply)
        except RuntimeError:
            # The loop has closed, so nobody waits for this answer.
            self._drop_answer(record, reply)

    def _settle(self, record: _Chain, future: asyncio.Future, reply: dict) -> None:
        if future.done():  # the caller has stopped waiting
            return self._drop_answer(record, reply)
        try:
            future.set_result(self.channel.outcome(reply))
        except (Exception, InterruptProcessingException) as error:
            future.set_exception(self._failure(record, error))

    def _abandon(self, record: _Chain) -> None:
        """Nobody awaits the request any more: its record stays until its answer comes."""
        record.abandoned, record.loop = True, None
        if record.id in self.chains:
            self._cancel(record)

    def _drop_answer(self, record: _Chain, reply: dict) -> None:
        """Let go of the answer to a request nobody awaits any more, and of its chain."""
        self.channel.drop(reply)
        self.chains.pop(record.id, None)

    def _failure(self, record: _Chain, error: BaseException) -> BaseException:
        """What a failed request raises: ComfyUI's cancel if ComfyUI cancelled it, cleared as a
        check in one process clears it; or which process stopped, how, in which node."""
        if record.deadline is not None:
            interrupt_current_processing(False)
            return InterruptProcessingException()
        if isinstance(error, _channel.PackStopped):
            where = f" while running node {record.node}" if record.node else ""
            if self.removed:
                return _channel.PackStopped(f"{self.peer} was unloaded{where}")
            code = record.process.wait()
            return _channel.PackStopped(
                f"{self.peer} stopped (exit code {code}){where}"
            )
        return error

    def _cancel(self, record: _Chain) -> None:
        # A pack waiting on ComfyUI, which goes on with its own work after a cancel too, has
        # its grace period once ComfyUI has answered.
        record.awaited = set() if record.nested else record.calls.copy()
        record.deadline = math.inf if record.awaited else time.monotonic() + GRACE
        self._tell(record)

    def _tell(self, record: _Chain) -> None:
        try:
            record.told = self.channel.cancel(record.id, record.lane)
        except _channel.PackStopped:
            record.told = True

    @contextlib.contextmanager
    def _chain(self, lane: str, loop: asyncio.AbstractEventLoop | None = None):
        running = get_executing_context()
        record = _Chain(
            next(self.chain_ids), lane, self.process, running and running.node_id
        )
        self.chains[record.id] = record
        token = _channel.enter_chain(self, record.id)
        if loop is not None:
            record.loop, record.context = loop, contextvars.copy_context()
            record.inbox = _channel.inbox(self)
        try:
            yield record
        finally:
            _channel.leave_chain(token)
            if not record.abandoned:
                del self.chains[record.id]

    def _launch(self) -> None:
        import folder_paths

        folders = {
            name: getattr(folder_paths, f"get_{name}_directory")()
            for name in ("output", "input", "temp", "user")
        }
        start = {
            "argv": sys.argv,
            "args": vars(args),
            "folders": folders,
            "epoch": self.epoch,
            "log_level": logging.getLogger().getEffectiveLevel(),
            "pack": self.module_path,
            "name": self.module_name,
        }
        self.killed = False
        self.process, sock = _pack_launch.launcher()(start)
        self.channel = _channel.Channel(sock, self)
        self.channel.start(f"{self.peer} reader")

    async def _load(self) -> dict:
        """Start a process and load the pack in it; kill the process if loading fails."""
        epoch = self.epoch
        self._launch()
        try:
            loaded = await self._call(
                ("load",), (_entry(self.module_path), self.module_name), "exec"
            )
        except BaseException:
            self.kill()
            # Its `stopped` acts on `self.process`: it must be done before another starts.
            self.channel.reader.join(GRACE)
            raise
        self.schemas = {schema.node_id: schema for schema, _ in loaded["nodes"]}
        with self.lock:
            if self.epoch != epoch:
                raise _channel.PackStopped(
                    f"{self.peer} stopped (exit code {self.process.wait()}) after it loaded"
                )
            self.ready = True
        return loaded

    async def _started(self) -> None:
        """Start the process again if it has stopped: once, however many loops are waiting."""
        if self.ready:
            return
        if self.removed:
            why = (
                "no longer loaded"
                if self.disabled is None
                else f"disabled: {self.disabled}"
            )
            raise _channel.PackStopped(f"{self.peer} is {why}")
        with self.lock:
            starting, first = self.starting, self.starting is None
            if first:
                starting = self.starting = concurrent.futures.Future()
        if not first:
            return await asyncio.wrap_future(starting)
        try:
            await self._restart()
        except BaseException as error:
            starting.set_exception(error)
            raise
        else:
            starting.set_result(None)
        finally:
            with self.lock:
                self.starting = None

    async def _restart(self) -> None:
        recent = [t for t in self.crashes if time.monotonic() - t < CRASH_WINDOW]
        if len(recent) >= CRASHES:
            raise self._disable(
                f"its process stopped {len(recent)} times in {CRASH_WINDOW:.0f} s"
            )
        try:
            loaded = await self._load()
        except _channel.PackStopped:
            raise  # `stopped` counted it, unless ComfyUI ended the process
        except Exception:
            self.crashes.append(time.monotonic())
            raise
        if _layout(loaded) != self.layout:
            raise self._disable(
                "it declared other nodes or routes when its process started again"
            )

    def _disable(self, reason: str) -> _channel.PackStopped:
        """Take the pack away, which ends its process, and say why, now and at each later use."""
        self.disabled = reason
        logging.warning(f"{self.peer} in {self.module_path} is disabled: {reason}.")
        _packs.remove_pack(self.module_path)
        return _channel.PackStopped(f"{self.peer} is disabled: {reason}")


def _layout(loaded: dict) -> tuple:
    """What a pack's process declares that ComfyUI built on: node ids, their methods, routes."""
    nodes = [(schema.node_id, methods) for schema, methods in loaded["nodes"]]
    return nodes, loaded["routes"]


@functools.cache
def _sync_api():
    from comfy_api.v4 import ComfyAPISync

    return ComfyAPISync()


def _call_api(group: str, name: str, args: tuple, kwargs: dict):
    from comfy_api.v4 import ComfyAPI_v4

    if group not in ComfyAPI_v4.__annotations__ or name.startswith("_"):
        raise LookupError(f"eng.{group}.{name} is not a call of the API")
    if group == "caching":
        raise TypeError("cache providers must be registered in ComfyUI's process")
    return getattr(getattr(_sync_api(), group), name)(*args, **kwargs)


def _folders(pack_path: str, name: str, args: tuple, kwargs: dict):
    from comfy_api.v4.util import folders

    from ._pack_main import FOLDERS_IN_COMFYUI

    # The others are answered in the pack process, which may not see ComfyUI's folders.
    if name not in FOLDERS_IN_COMFYUI:
        raise LookupError(f"util.folders.{name} is not answered by ComfyUI")
    value = getattr(folders, name)
    if name in folders.__annotations__:
        return value
    if name == "add_model_folder_path":
        bound = inspect.signature(value).bind(*args, **kwargs)
        _refuse_directories(pack_path, [bound.arguments["full_folder_path"]])
    return value(*args, **kwargs)


def _assign_folder(pack_path: str, name: str, entry: tuple) -> None:
    from comfy_api.v4.util import folders

    directories, extensions = entry
    if not (
        type(directories) in (list, tuple)
        and type(extensions) in (set, frozenset)
        and all(type(item) is str for item in (*directories, *extensions))
    ):
        raise TypeError(
            "a model folder entry is (directories, extensions): a list of paths and a set "
            "of extensions"
        )
    _refuse_directories(pack_path, directories)
    folders.folder_names_and_paths[name] = (list(directories), set(extensions))


def _refuse_directories(pack_path: str, directories: list) -> None:
    """Raise `ValueError` unless ComfyUI may list model files in each of `directories` for the
    pack process loaded from `pack_path`: a directory it lists already, one inside its models
    folder, or one inside the pack's own folder when no link in that folder leads out of it.
    ComfyUI reads there with its own access to files, which the pack's process may not have, and
    follows links as it lists."""
    import folder_paths

    listed = {
        os.path.realpath(directory)
        for known, _ in folder_paths.folder_names_and_paths.values()
        for directory in known
    }
    for directory in directories:
        if not (
            os.path.realpath(directory) in listed
            or folder_paths.is_within_directory(folder_paths.models_dir, directory)
            or _inside_pack(pack_path, directory)
        ):
            raise ValueError(
                f"{directory!r} cannot be a model folder of a pack that runs in its own "
                "process: it is outside ComfyUI's models folder and the pack's own folder, "
                "or a link in it leads out of the pack's folder"
            )


def _inside_pack(pack_path: str, directory: str) -> bool:
    """Whether `directory` is in the pack's folder and no link in that folder leads out of it."""
    import folder_paths

    if not (
        os.path.isdir(pack_path)
        and folder_paths.is_within_directory(pack_path, directory)
    ):
        return False
    return all(
        folder_paths.is_within_directory(pack_path, os.path.join(root, name))
        for root, folders, files in os.walk(pack_path)
        for name in folders + files
    )


async def load(module_path: str, module_name: str) -> types.ModuleType:
    """Start the pack's process, load the pack there, and return a module that stands for it.

    The module has the pack's `WEB_DIRECTORY` and a `comfy_entrypoint` whose extension lists
    node classes built from the pack's schemas. Raises what loading raised in the pack process.
    """
    if module_path in _pack_processes.processes:
        raise RuntimeError(f"{module_path} already runs in its own process")
    process = PackProcess(module_path, module_name)
    # Listed while it loads too, so that ComfyUI's exit ends it.
    _pack_processes.processes[module_path] = process
    try:
        loaded = await process._load()
    except BaseException:
        _pack_processes.processes.pop(module_path, None)
        raise
    process.layout = _layout(loaded)
    extension = _pack_nodes.extension(process, module_name, loaded)
    module = types.ModuleType(module_name)
    module.WEB_DIRECTORY = _web_directory(module_path, loaded["web_directory"])
    module.comfy_entrypoint = lambda: extension
    return module


def _web_directory(module_path: str, web_directory) -> str | None:
    """The pack's web folder, if it is inside the folder the loader resolves it from: ComfyUI
    serves it to any browser with its own access to files."""
    import folder_paths

    folder = module_path if os.path.isdir(module_path) else os.path.dirname(module_path)
    if web_directory is None or (
        type(web_directory) is str
        and folder_paths.is_within_directory(
            folder, os.path.join(folder, web_directory)
        )
    ):
        return web_directory
    logging.warning(
        f"pack {os.path.basename(module_path)}: its web folder {web_directory!r} is not served: "
        "it is outside the pack's folder"
    )
    return None
