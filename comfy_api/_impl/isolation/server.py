"""A pack running in its own process, on its own port.

    PackServer.start("/path/to/pack", port=8301)   # loads the pack, then listens on 127.0.0.1:8301
    PackServer.stop(8301)
    PackServer.status()                            # every pack server this user started

ComfyUI starts these itself under pack isolation (through a launcher, ``launcher.py``). By hand:
``scripts/pack.sh start|stop|status|describe``, or ``python -m comfy_api._impl.isolation.server``.

The process imports the pack (a stand-in ``server.PromptServer`` collects the routes it
registers), then serves ComfyUI over ``channel``: the pack's node list, each node's schema, node
methods, and HTTP requests for the pack's routes (served by a private aiohttp app on loopback).
``describe`` imports a pack, writes what it has (nodes, routes, web directory) and exits.

Each server has a run file, ``<run dir>/<port>.json`` (pid, pack, address; readable only by this
user). A connection must present the token (from ``--token-file``, or made and written to the run
file) before anything is unpickled. The channel trusts the other side completely, so a server
listens on the loopback address unless told otherwise.
"""
from __future__ import annotations

import argparse
import asyncio
import hmac
import http.client
import importlib.util
import json
import logging
import os
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
from typing import Any, Optional

from . import without_settings
from .channel import Channel, WorkerError

RUN_DIR_ENV = "COMFY_PACK_RUN_DIR"
#: A pack process ComfyUI starts exits this many seconds after its last connection from ComfyUI
#: closes, as a browser's page processes exit when their channel to the browser closes. The OS
#: closes ComfyUI's connections however it ends (a crash or ``kill -9`` included), so pack
#: processes don't outlive it. Unset: the server runs until stopped (``scripts/pack.sh``).
ORPHAN_GRACE_ENV = "COMFY_PACK_ORPHAN_GRACE"
DEFAULT_ORPHAN_GRACE = 5.0
FIRST_CONNECTION_GRACE = 60.0  # ComfyUI connects as soon as start returns; allow for a slow launcher
MAGIC = b"COMFYPACK1"
TOKEN_BYTES = 32  # sent as 64 hex characters
COMFY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

logger = logging.getLogger(__name__)


def run_dir() -> str:
    path = os.environ.get(RUN_DIR_ENV) or os.path.join(tempfile.gettempdir(), f"comfy-packs-{os.getuid()}")
    os.makedirs(path, mode=0o700, exist_ok=True)
    return path


def _run_file(port: int) -> str:
    return os.path.join(run_dir(), f"{int(port)}.json")


def read_info(port: int) -> Optional[dict]:
    try:
        with open(_run_file(port)) as file:
            return json.load(file)
    except FileNotFoundError:
        return None


_children: dict[int, subprocess.Popen] = {}  # pid -> the pack processes this process started


def _alive(pid: int) -> bool:
    child = _children.get(pid)
    if child is not None:  # ours: collect its exit status, so it doesn't stay behind as a zombie
        if child.poll() is None:
            return True
        _children.pop(pid, None)
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # a zombie someone else hasn't collected yet has exited
        with open(f"/proc/{pid}/stat") as stat:
            return stat.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def _write_private(path: str, text: str) -> None:
    fd = os.open(path + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as file:
        file.write(text)
    os.replace(path + ".tmp", path)


def connect(port: int, token: str, host: str = "127.0.0.1", timeout: float = 10.0) -> socket.socket:
    """A connection to the pack server on ``port``, past the token check."""
    sock = socket.create_connection((host, int(port)), timeout=timeout)
    sock.settimeout(None)
    sock.sendall(MAGIC + token.encode("ascii"))
    return sock


def module_name_for(entry: str) -> str:
    """The name ComfyUI gives a custom-node directory it imports (``nodes.load_custom_node``)."""
    return entry.replace(".", "_x_")


# --------------------------------------------------------------------------- #
# The pack, inside its process
# --------------------------------------------------------------------------- #
class _PromptServerStandIn:
    """``server.PromptServer.instance`` for pack code: collects routes, forwards UI events."""

    def __init__(self) -> None:
        from aiohttp import web

        self.routes = web.RouteTableDef()
        self.app = web.Application()
        self.client_id = None
        self.last_node_id = None
        self.last_prompt_id = None
        self.supports: set = set()
        self.loop = None
        self.number = 0
        self.on_prompt_handlers: list = []  # ComfyUI's runs on its own prompts; a pack process sees none

    def send_sync(self, event: str, data: Any, sid: Any = None) -> None:
        from .worker import current_state
        state = current_state()
        if state is None:
            logger.debug("UI event %r outside a node; dropped", event)
            return
        state.channel.request({"op": "call", "name": "ui.send", "args": [event, data], "kwargs": {}})

    async def send(self, event: str, data: Any, sid: Any = None) -> None:
        self.send_sync(event, data, sid)

    def send_progress_text(self, text: str, node_id: Any, sid: Any = None) -> None:
        self.send_sync("progress_text", {"text": text, "node": node_id}, sid)

    def add_on_prompt_handler(self, handler: Any) -> None:
        self.on_prompt_handlers.append(handler)


def registration_names(registrations: dict) -> dict:
    from comfy_api._impl import registration
    return registration.names(registrations) if registrations else {}


def _install_server_stand_in() -> _PromptServerStandIn:
    instance = _PromptServerStandIn()
    module = types.ModuleType("server", "Stand-in for ComfyUI's server module, in a pack's process.")
    from protocol import BinaryEventTypes
    module.BinaryEventTypes = BinaryEventTypes
    module.PromptServer = type("PromptServer", (), {"instance": instance})
    sys.modules["server"] = module
    return instance


class LoadedPack:
    """A pack imported in this process: its nodes, routes and web directory."""

    def __init__(self, entry: str) -> None:
        from comfy_api._impl import legacy_imports

        legacy_imports.install()  # packs that import comfy_api.latest
        apply_folder_layout()  # ComfyUI's folders, when ComfyUI started this process
        self.entry = os.path.realpath(entry)
        self.module_name = module_name_for(self.entry)
        self.prompt_server = _install_server_stand_in()
        init = os.path.join(self.entry, "__init__.py")
        spec = importlib.util.spec_from_file_location(self.module_name, init, submodule_search_locations=[self.entry])
        module = importlib.util.module_from_spec(spec)
        sys.modules[self.module_name] = module
        spec.loader.exec_module(module)
        self.module = module
        self.registrations: dict = {}  # samplers, schedulers, model folders, node replacements
        self.nodes: dict[str, Any] = dict(getattr(module, "NODE_CLASS_MAPPINGS", None) or {})
        self.display_names: dict[str, str] = dict(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", None) or {})
        if hasattr(module, "comfy_entrypoint"):
            asyncio.run(self._load_extension(module.comfy_entrypoint))
        web = getattr(module, "WEB_DIRECTORY", None)
        self.web_directory = os.path.abspath(os.path.join(self.entry, web)) if web else None
        self._http_port: Optional[int] = None
        self._http_lock = threading.Lock()

    async def _load_extension(self, entrypoint: Any) -> None:
        extension = entrypoint()
        if asyncio.iscoroutine(extension):
            extension = await extension
        await extension.on_load()
        from comfy_api._impl import registration
        self.registrations = await registration.collect(extension)
        for cls in await extension.get_node_list():
            schema = cls.GET_SCHEMA()
            self.nodes[schema.node_id] = cls
            if schema.display_name:
                self.display_names[schema.node_id] = schema.display_name

    def routes(self) -> list[tuple[str, str]]:
        found = [(route.method, route.path) for route in self.prompt_server.routes
                 if hasattr(route, "method") and hasattr(route, "path")]
        for resource in self.prompt_server.app.router.resources():
            info = resource.get_info()
            path = info.get("path") or info.get("formatter")
            if "directory" in info:
                continue  # static files: ``static()``
            for route in resource:
                if path and route.method != "HEAD":
                    found.append((route.method, path))
        return sorted(set(found))

    def static(self) -> list[tuple[str, str]]:
        """Static-file routes (``web.static``): a URL prefix and a directory."""
        found = []
        for resource in self.prompt_server.app.router.resources():
            info = resource.get_info()
            if "directory" in info:
                found.append((info["prefix"], str(info["directory"])))
        for route in self.prompt_server.routes:
            if hasattr(route, "prefix") and hasattr(route, "path") and not hasattr(route, "method"):
                found.append((route.prefix, str(route.path)))
        return sorted(set(found))

    def describe(self) -> dict:
        return {"module": self.module_name, "entry": self.entry,
                "nodes": {node_id: self.display_names.get(node_id, node_id) for node_id in self.nodes},
                "routes": self.routes(), "static": self.static(), "web_directory": self.web_directory,
                "declared_routes": sorted([m, p] for m, p in self.declared_routes()),
                "registrations": registration_names(self.registrations)}

    def handle(self, message: dict) -> Any:
        from . import proxy

        op = message["op"]
        if op == "describe":
            return self.describe()
        if op == "nodes":
            return {node_id: proxy.describe_node(cls) for node_id, cls in self.nodes.items()}
        if op == "input_types":
            return proxy.plain(self.nodes[message["node"]].INPUT_TYPES())
        if op == "http":
            return self._http(message)
        if op == "route":
            return self._route(message)
        raise WorkerError(f"unknown request {op!r}")

    # V2 declared routes: ``routes`` in the pack's manifest, served at /ext/<namespace>/
    def declared_routes(self) -> dict[tuple[str, str], tuple[str, str]]:
        from comfy_api._impl import manifest as _manifest
        path = _manifest.path_for(self.entry)
        if path is None:
            return {}
        with open(path) as file:
            data = json.load(file)
        return {(r["method"].lower(), r["path"]): (r["module"], r["handler"]) for r in data.get("routes", ())}

    def _route(self, message: dict) -> dict:
        import importlib

        route = self.declared_routes().get((message["method"], message["path"]))
        if route is None:
            return {"status": 404, "body": json.dumps({"ok": False, "error": "unknown route"}),
                    "content_type": "application/json"}
        module = importlib.import_module(f"{self.module_name}.{route[0]}")
        result = getattr(module, route[1])({k: message[k] for k in ("method", "path", "query", "body")})
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)
        return _route_response(result)

    # the pack's routes, served by a private aiohttp app on loopback
    def _start_http(self) -> int:
        from aiohttp import web

        ready = threading.Event()
        holder: dict = {}

        def run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.prompt_server.loop = loop

            async def start() -> None:
                app = self.prompt_server.app
                app.add_routes(self.prompt_server.routes)
                runner = web.AppRunner(app)
                await runner.setup()
                site = web.TCPSite(runner, "127.0.0.1", 0)
                await site.start()
                holder["port"] = site._server.sockets[0].getsockname()[1]
                ready.set()
            loop.run_until_complete(start())
            loop.run_forever()

        threading.Thread(target=run, name="pack-http", daemon=True).start()
        ready.wait(30)
        return holder["port"]

    def _http(self, message: dict) -> dict:
        with self._http_lock:
            if self._http_port is None:
                self._http_port = self._start_http()
        conn = http.client.HTTPConnection("127.0.0.1", self._http_port, timeout=300)
        headers = {k: v for k, v in message["headers"] if k.lower() not in ("host", "content-length", "connection")}
        conn.request(message["method"], message["path_qs"], body=message["body"] or None, headers=headers)
        response = conn.getresponse()
        body = response.read()
        conn.close()
        return {"status": response.status, "reason": response.reason, "body": body,
                "headers": [(k, v) for k, v in response.getheaders()
                            if k.lower() not in ("transfer-encoding", "connection", "content-length")]}


ROUTE_BODY_MAX_BYTES = 1 << 20
ROUTE_CONTENT_TYPES = frozenset({"application/json", "text/plain"})


def _route_response(result: Any) -> dict:
    """A declared route's answer: ``{"status", "body" (str), "content_type"}``, as the overlay's."""
    if not isinstance(result, dict):
        raise TypeError("route handler must return a mapping")
    status = result.get("status", 200)
    if not isinstance(status, int) or isinstance(status, bool) or not 100 <= status <= 599:
        raise ValueError(f"route response status {status!r} is not an HTTP status")
    body = result.get("body", "")
    if not isinstance(body, str):
        raise TypeError("route response body must be a string")
    if len(body.encode("utf-8")) > ROUTE_BODY_MAX_BYTES:
        raise ValueError("route response is larger than 1 MB")
    content_type = result.get("content_type", "application/json")
    if content_type not in ROUTE_CONTENT_TYPES:
        raise ValueError(f"route response content type {content_type!r} is not one of {sorted(ROUTE_CONTENT_TYPES)}")
    return {"status": status, "body": body, "content_type": content_type}


# --------------------------------------------------------------------------- #
# The server process
# --------------------------------------------------------------------------- #
def _read_exact(sock: socket.socket, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            break
        data += chunk
    return data


def _serve(entry: str, port: int, host: str, token_file: Optional[str],
           orphan_grace: Optional[float] = None) -> None:
    from .worker import _WorkerState

    logging.basicConfig(level=logging.INFO, format=f"[pack server {os.getpid()} :{port}] %(message)s")
    if host not in ("127.0.0.1", "::1", "localhost"):
        logger.warning("listening on %s: anyone who can reach it and has the token can run code here", host)
    pack = LoadedPack(entry)  # loaded before listening: a server that accepts is ready
    if token_file:
        with open(token_file) as file:
            token = file.read().strip()
    else:
        token = secrets.token_hex(TOKEN_BYTES)
    expected = MAGIC + token.encode("ascii")
    listener = socket.create_server((host, port))
    info = {"pid": os.getpid(), "pack": pack.entry, "host": host, "port": port, "started": time.time(),
            "token": None if token_file else token}
    _write_private(_run_file(port), json.dumps(info))
    signal.signal(signal.SIGTERM, lambda *_: (_remove(port, os.getpid()), os._exit(0)))
    logger.info("serving %s (%d nodes)", pack.entry, len(pack.nodes))
    owner = {"open": 0, "since": time.monotonic(), "connected": False}
    owner_lock = threading.Lock()
    if orphan_grace is not None:
        threading.Thread(target=_exit_when_orphaned, args=(port, owner, owner_lock, orphan_grace),
                         name="pack-orphan-watch", daemon=True).start()

    def connection(conn: socket.socket, peer: Any) -> None:
        if not hmac.compare_digest(_read_exact(conn, len(expected)), expected):
            logger.warning("refused a connection from %s: wrong token", peer)
            conn.close()
            return
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)  # notice a peer that vanished
        with owner_lock:
            owner["open"] += 1
            owner["connected"] = True
        state = _WorkerState(pack)
        state.channel = Channel(conn, state.handle)
        try:
            state.channel.serve_forever()
        except WorkerError:
            return  # the other side went away; the orphan watch decides whether that was ComfyUI
        finally:
            conn.close()
            with owner_lock:
                owner["open"] -= 1
                owner["since"] = time.monotonic()
        logger.info("shut down")
        _remove(port, os.getpid())
        os._exit(0)

    while True:
        conn, peer = listener.accept()
        threading.Thread(target=connection, args=(conn, peer), daemon=True).start()


def _exit_when_orphaned(port: int, owner: dict, lock: threading.Lock, grace: float) -> None:
    """Exit once no connection from ComfyUI has been open for ``grace`` seconds."""
    while True:
        time.sleep(min(0.5, grace / 2) if grace > 0 else 0.5)
        with lock:
            if owner["open"]:
                continue
            limit = grace if owner["connected"] else max(grace, FIRST_CONNECTION_GRACE)
            idle = time.monotonic() - owner["since"]
        if idle >= limit:
            logger.info("ComfyUI's connections are gone (%.1fs); exiting", idle)
            _remove(port, os.getpid())
            from .channel import sweep
            sweep(os.getpid())  # segments we sent that nobody will read now
            os._exit(0)


def _remove(port: int, pid: int) -> None:
    info = read_info(port)
    if info is not None and info["pid"] == pid:
        os.unlink(_run_file(port))


FOLDERS_ENV = "COMFY_PACK_FOLDERS"


def folder_layout() -> Optional[dict]:
    """ComfyUI's folders (``--output-directory``, extra model paths...), for a pack's process."""
    folder_paths = sys.modules.get("folder_paths")
    if folder_paths is None:
        return None
    return {"models_dir": folder_paths.models_dir, "output": folder_paths.get_output_directory(),
            "input": folder_paths.get_input_directory(), "temp": folder_paths.get_temp_directory(),
            "user": folder_paths.get_user_directory(),
            "folders": {name: [list(paths), sorted(exts)] for name, (paths, exts) in folder_paths.folder_names_and_paths.items()}}


def apply_folder_layout() -> None:
    """In a pack's process: ComfyUI's folders, as ComfyUI handed them over (``util.folders``)."""
    import json
    raw = os.environ.get(FOLDERS_ENV)
    if not raw:
        return
    layout = json.loads(raw)
    import folder_paths
    folder_paths.models_dir = layout["models_dir"]
    folder_paths.set_output_directory(layout["output"])
    folder_paths.set_input_directory(layout["input"])
    folder_paths.set_temp_directory(layout["temp"])
    folder_paths.set_user_directory(layout["user"])
    for name, (paths, exts) in layout["folders"].items():
        folder_paths.folder_names_and_paths[name] = (list(paths), set(exts))
    folder_paths.filename_list_cache.clear()


def _python_env() -> dict:
    import json
    env = without_settings(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([COMFY_ROOT] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    layout = folder_layout()
    if layout is not None:
        env[FOLDERS_ENV] = json.dumps(layout)
    return env


class PackServer:
    """Start, stop, list and describe pack servers in plain local processes."""

    @staticmethod
    def start(entry: str, port: int, host: str = "127.0.0.1", token_file: Optional[str] = None,
              python: Optional[str] = None, wait: float = 300.0, orphan_grace: Optional[float] = None) -> dict:
        """``orphan_grace``: exit that many seconds after the last connection closes (ComfyUI's own
        pack processes); None runs until stopped."""
        entry = os.path.realpath(entry)
        if not os.path.isfile(os.path.join(entry, "__init__.py")):
            raise FileNotFoundError(f"no pack at {entry} (no __init__.py)")
        existing = read_info(port)
        if existing is not None and _alive(existing["pid"]):
            raise RuntimeError(f"port {port} already runs a pack server for {existing['pack']} (pid {existing['pid']})")
        log_path = os.path.join(run_dir(), f"{int(port)}.log")
        command = [python or sys.executable, "-m", "comfy_api._impl.isolation.server", "_serve",
                   "--pack", entry, "--port", str(port), "--host", host]
        if token_file:
            command += ["--token-file", token_file]
        if orphan_grace is not None:
            command += ["--orphan-grace", str(orphan_grace)]
        with open(log_path, "ab") as log:
            proc = subprocess.Popen(command, env=_python_env(), cwd=COMFY_ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        _children[proc.pid] = proc
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            info = read_info(port)
            if info is not None and info["pid"] == proc.pid:
                return info
            if proc.poll() is not None:
                raise RuntimeError(f"the pack server for {entry} exited ({proc.returncode}); see {log_path}")
            time.sleep(0.05)
        proc.kill()
        raise TimeoutError(f"the pack server for {entry} didn't start in {wait}s; see {log_path}")

    @staticmethod
    def stop(port: int, wait: float = 10.0) -> bool:
        """Stop the server on ``port``. False if none was running."""
        info = read_info(port)
        if info is None:
            return False
        pid = info["pid"]
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if _alive(pid):
                os.kill(pid, sig)
            deadline = time.monotonic() + wait / 2
            while _alive(pid) and time.monotonic() < deadline:
                time.sleep(0.05)
        _remove(port, pid)
        return True

    @staticmethod
    def status() -> list[dict]:
        servers = []
        for name in sorted(os.listdir(run_dir())):
            if name.endswith(".json"):
                info = read_info(int(name[:-5]))
                if info is not None:
                    servers.append({k: v for k, v in info.items() if k != "token"} | {"alive": _alive(info["pid"])})
        return servers

    @staticmethod
    def manifest(entry: str, python: Optional[str] = None, timeout: float = 600.0) -> dict:
        """A V2 pack's ``comfy-pack.json``, made by importing it in a short-lived process."""
        from comfy_api._impl import manifest as _manifest
        out = tempfile.NamedTemporaryFile(prefix="manifest-", suffix=".json", dir=run_dir(), delete=False)
        out.close()
        try:
            command = [python or sys.executable, "-m", "comfy_api._impl.isolation.server", "_manifest",
                       "--pack", os.path.realpath(entry), "--out", out.name]
            existing = _manifest.path_for(entry)
            if existing and os.path.basename(existing) == _manifest.FILE:
                command += ["--previous", existing]  # keep what review granted
            done = subprocess.run(command, env=_python_env(), cwd=COMFY_ROOT, capture_output=True, text=True,
                                  timeout=timeout)
            if done.returncode != 0:
                raise RuntimeError(f"making the manifest of {entry} failed:\n{done.stdout[-2000:]}{done.stderr[-4000:]}")
            with open(out.name) as file:
                return json.load(file)
        finally:
            os.unlink(out.name)

    @staticmethod
    def describe(entry: str, python: Optional[str] = None, timeout: float = 600.0) -> dict:
        """Import the pack in a short-lived process and return what it has."""
        out = tempfile.NamedTemporaryFile(prefix="describe-", suffix=".json", dir=run_dir(), delete=False)
        out.close()
        try:
            done = subprocess.run([python or sys.executable, "-m", "comfy_api._impl.isolation.server", "_describe",
                                   "--pack", os.path.realpath(entry), "--out", out.name],
                                  env=_python_env(), cwd=COMFY_ROOT, capture_output=True, text=True, timeout=timeout)
            if done.returncode != 0:
                raise RuntimeError(f"describing {entry} failed:\n{done.stdout[-2000:]}{done.stderr[-4000:]}")
            with open(out.name) as file:
                return json.load(file)
        finally:
            os.unlink(out.name)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="pack.sh", description="Pack isolation: run, describe, install and make manifests for custom-node packs.")
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start", help="start a pack on a port")
    start.add_argument("pack")
    start.add_argument("port", type=int)
    start.add_argument("--host", default="127.0.0.1")
    start.add_argument("--token-file")
    stop = sub.add_parser("stop", help="stop the pack on a port")
    stop.add_argument("port", type=int)
    sub.add_parser("status", help="list pack servers")
    describe = sub.add_parser("describe", help="print a pack's nodes, routes and web directory")
    describe.add_argument("pack")
    make = sub.add_parser("manifest", help="write a V2 pack's comfy-pack.json from its code (imports it)")
    make.add_argument("pack")
    make.add_argument("--print", action="store_true", help="print it instead of writing it")
    install = sub.add_parser("install", help="prepare a V2 pack: its environment and its weights")
    install.add_argument("pack")
    serve = sub.add_parser("_serve")  # the server process itself
    serve.add_argument("--pack", required=True)
    serve.add_argument("--port", type=int, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--token-file")
    serve.add_argument("--orphan-grace", type=float, default=None,
                       help=f"exit this many seconds after the last connection closes (default: ${ORPHAN_GRACE_ENV}, else never)")
    one_shot = sub.add_parser("_describe")
    one_shot.add_argument("--pack", required=True)
    one_shot.add_argument("--out", required=True)
    generate = sub.add_parser("_manifest")
    generate.add_argument("--pack", required=True)
    generate.add_argument("--out", required=True)
    generate.add_argument("--previous")
    args = parser.parse_args(argv)
    if args.command == "_serve":
        grace = args.orphan_grace
        if grace is None and os.environ.get(ORPHAN_GRACE_ENV, "").strip():
            grace = float(os.environ[ORPHAN_GRACE_ENV])
        _serve(args.pack, args.port, args.host, args.token_file, grace)
        return 0
    if args.command == "_describe":
        logging.basicConfig(level=logging.WARNING)
        _write_private(args.out, json.dumps(LoadedPack(args.pack).describe()))
        return 0
    if args.command == "_manifest":
        from comfy_api._impl import manifest as _manifest
        logging.basicConfig(level=logging.WARNING)
        previous = None
        if args.previous:
            with open(args.previous) as file:
                previous = json.load(file)
        _write_private(args.out, json.dumps(_manifest.generate(args.pack, previous)))
        return 0
    from .packs import entry_for
    if args.command == "start":
        info = PackServer.start(entry_for(args.pack), args.port, args.host, args.token_file)
        sys.stdout.write(f"started {info['pack']} on {info['host']}:{info['port']} (pid {info['pid']})\n")
        return 0
    if args.command == "stop":
        if PackServer.stop(args.port):
            sys.stdout.write(f"stopped the pack server on port {args.port}\n")
            return 0
        sys.stderr.write(f"no pack server on port {args.port}\n")
        return 1
    if args.command == "describe":
        sys.stdout.write(json.dumps(PackServer.describe(entry_for(args.pack)), indent=2) + "\n")
        return 0
    if args.command == "manifest":
        from comfy_api._impl import manifest as _manifest
        from . import envs
        entry = entry_for(args.pack)
        data = PackServer.manifest(entry, python=envs.python_for(entry))
        text = json.dumps(data, indent=2) + "\n"
        if args.print:
            sys.stdout.write(text)
        else:
            with open(os.path.join(entry, _manifest.FILE), "w", encoding="utf-8") as file:
                file.write(text)
            sys.stdout.write(f"wrote {os.path.join(entry, _manifest.FILE)}: {len(data['nodes'])} nodes\n")
        return 0
    if args.command == "install":
        from comfy_api._impl import manifest as _manifest
        from . import envs, weights
        entry = entry_for(args.pack)
        data = _manifest.read(entry)
        if data is None:
            sys.stderr.write(f"{entry} has no {_manifest.FILE}; run: pack.sh manifest {args.pack}\n")
            return 1
        python = envs.python_for(entry, data.get("requirements", []))
        paths = weights.ensure(data["name"], data.get("weights", []))
        sys.stdout.write(f"{data['name']} {data['version']}: python {python}; {len(paths)} weights ready\n")
        return 0
    for server in PackServer.status():
        state = "running" if server["alive"] else "dead"
        sys.stdout.write(f"{server['port']:>6}  {state:<8} pid {server['pid']:<8} {server['pack']}\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError, TimeoutError) as error:
        sys.stderr.write(f"pack.sh: {error}\n")
        sys.exit(1)
