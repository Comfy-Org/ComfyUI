"""Pack isolation's packs: where they are, what they have, and starting them when they're needed.

Under pack isolation ComfyUI imports no custom-node pack. At startup it finds the packs and learns what
each has (nodes, routes, static files, web directory) from a description, made once per version
of the pack by a short-lived process and cached. Web directories and static files are served from
disk. A pack's process is started, and its nodes registered, when a prompt uses one of its nodes
(before the prompt is validated) or when one of its routes is called. A pack that hasn't been used
for ``COMFY_PACK_IDLE_MINUTES`` (default 15; 0: never) is stopped, never while it's running a node
or answering a route, and started again when it's next needed. Every use resets its timer.

Routes: a V2 pack declares its endpoints (``routes`` in its ``secure-nodes.json``) relative to
``/ext/<namespace>/``, the namespace derived from the pack's name (``pack_namespace``, the same
as the secure overlay's), so ComfyUI routes ``/ext/<namespace>/...`` to the pack without knowing
its endpoints in advance. Routes a V1 pack registers on ``PromptServer.instance`` are found by its
description and forwarded as they are.

Where packs come from: ComfyUI's ``custom_nodes`` folders, or the folders in ``COMFY_PACKS_DIR``
(``os.pathsep``-separated) instead. A folder is either flat (``<folder>/<pack>/``) or a pack
corpus (``packs.json`` and ``<slug>/<key>/<Pack>-HEAD/``). A pack that has a ``v2/`` package is
loaded from it: that's its conversion.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional

from . import launcher as _launcher
from . import server

PACKS_DIR_ENV = "COMFY_PACKS_DIR"
TTL_ENV = "COMFY_PACK_IDLE_MINUTES"
DEFAULT_TTL_MINUTES = 15.0
DESCRIBE_VERSION = 3
logger = logging.getLogger(__name__)


@dataclass
class Pack:
    name: str  # what ComfyUI calls it: web extension name, python_module
    entry: str  # the directory that's imported (the pack, or its v2/ conversion)
    canonical: str = ""  # version-free identity of a corpus pack (its slug)

    @property
    def namespace(self) -> str:
        """``/ext/<namespace>/``: from a corpus slug, else the manifest's ``name`` (so it's the same
        wherever the pack is installed, and a converter can write it into the pack's frontend), else
        the folder name."""
        if self.canonical:
            return pack_namespace(self.canonical)
        from comfy_api._impl import manifest as _manifest
        data = _manifest.read(self.entry)
        return pack_namespace((data or {}).get("name") or self.name)

    @property
    def key(self) -> str:
        return server.module_name_for(self.entry)


ROUTE_MOUNT = "/ext"
QUERY_MAX = 64
REQUEST_MAX_BYTES = 1 << 20


def pack_namespace(canonical_id: str) -> str:
    """Where a pack's declared routes live: ``/ext/<namespace>/``. A readable slug plus 8 hex of
    SHA-256 over the canonical id, so it is stable across versions and can't be chosen by a pack."""
    import re
    slug = re.sub(r"[^a-z0-9]+", "-", canonical_id.lower()).strip("-")[:32].strip("-") or "pack"
    return f"{slug}-{hashlib.sha256(canonical_id.encode('utf-8')).hexdigest()[:8]}"


def entry_for(path: str) -> str:
    """The directory to import for a pack directory: its ``v2/`` conversion if it has one."""
    path = os.path.realpath(path)
    v2 = os.path.join(path, "v2")
    return v2 if os.path.isfile(os.path.join(v2, "__init__.py")) else path


def _pack_name(directory: str) -> str:
    name = os.path.basename(os.path.normpath(directory))
    return name[:-len("-HEAD")] if name.endswith("-HEAD") else name


def _corpus(folder: str) -> list[Pack]:
    """``packs.json`` + ``<slug>/<key>/<pack>/``: the pack corpus repository's layout."""
    with open(os.path.join(folder, "packs.json")) as file:
        manifest = json.load(file)
    packs = []
    for slug, entry in sorted(manifest.items()):
        directory = os.path.join(folder, slug, entry["key"], entry["pack"])
        if os.path.isfile(os.path.join(directory, "__init__.py")):
            packs.append(Pack(_pack_name(directory), entry_for(directory), slug))
        else:
            logger.warning("pack corpus: %s isn't there; skipped", directory)
    return packs


def find_packs(folders: list[str]) -> list[Pack]:
    packs: list[Pack] = []
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        if os.path.isfile(os.path.join(folder, "packs", "packs.json")):
            folder = os.path.join(folder, "packs")
        if os.path.isfile(os.path.join(folder, "packs.json")):
            packs.extend(_corpus(folder))
            continue
        for name in sorted(os.listdir(folder)):
            directory = os.path.join(folder, name)
            if name.startswith(".") or name.endswith(".disabled") or name == "__pycache__":
                continue
            if os.path.isfile(os.path.join(directory, "__init__.py")):
                packs.append(Pack(_pack_name(directory), entry_for(directory)))
            elif name.endswith(".py") and os.path.isfile(directory):
                logger.info("pack isolation runs packs that are directories; %s is skipped", directory)
    seen: dict[str, Pack] = {}
    for pack in packs:
        if pack.name in seen:
            logger.warning("two packs named %s (%s, %s); using the first", pack.name, seen[pack.name].entry, pack.entry)
        else:
            seen[pack.name] = pack
    return list(seen.values())


def pack_folders() -> list[str]:
    from . import setting
    value = (setting(PACKS_DIR_ENV) or "").strip()
    if value:
        return [os.path.realpath(p) for p in value.split(os.pathsep) if p]
    import folder_paths
    return [os.path.realpath(p) for p in folder_paths.get_folder_paths("custom_nodes")]


def fingerprint(entry: str) -> str:
    digest = hashlib.sha256(f"{DESCRIBE_VERSION}\0{entry}".encode())
    for root, dirs, files in os.walk(entry):
        dirs[:] = sorted(d for d in dirs if d not in ("__pycache__", ".git", "node_modules"))
        for name in sorted(files):
            path = os.path.join(root, name)
            try:
                stat = os.stat(path)
            except OSError:
                continue
            digest.update(f"{os.path.relpath(path, entry)}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()[:24]


def _cache_dir() -> str:
    import folder_paths
    path = os.path.join(folder_paths.get_user_directory(), "pack_isolation_index")
    os.makedirs(path, exist_ok=True)
    return path


def from_manifest(pack: Pack, data: dict) -> dict:
    """A V2 pack's description, read from its manifest: nothing is imported or started."""
    web = data.get("web_directory")
    return {"module": server.module_name_for(pack.entry), "entry": pack.entry,
            "nodes": {node_id: node.get("display_name") or node_id for node_id, node in data["nodes"].items()},
            "routes": [], "static": [], "web_directory": os.path.join(pack.entry, web) if web else None,
            "declared_routes": sorted([r["method"].lower(), r["path"]] for r in data.get("routes", [])),
            "manifest": data}


def describe(pack: Pack) -> dict:
    """What the pack has: a V2 pack's manifest, or for a V1 pack, the cache or a short-lived
    process that imports it (once per version of the pack)."""
    from comfy_api._impl import manifest as _manifest
    data = _manifest.read(pack.entry)
    if data is not None:
        return from_manifest(pack, data)
    path = os.path.join(_cache_dir(), f"{pack.name}-{fingerprint(pack.entry)}.json")
    try:
        with open(path) as file:
            return json.load(file)
    except (FileNotFoundError, ValueError):
        pass
    from . import envs
    description = server.PackServer.describe(pack.entry, python=envs.python_for(pack.entry))
    with open(path + ".tmp", "w") as file:
        json.dump(description, file)
    os.replace(path + ".tmp", path)
    return description


def _version_in(version: str, wanted: str) -> bool:
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version
    except ImportError:  # can't check ranges without packaging; presence is still checked
        return True
    try:
        return Version(version) in SpecifierSet(wanted)
    except Exception:
        return False


class PackManager:
    """The packs under pack isolation: the index, the running pack processes, the registered nodes."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.packs: dict[str, Pack] = {}
        self.descriptions: dict[str, dict] = {}
        self.node_packs: dict[str, str] = {}  # node id -> pack name
        self.started: dict[str, Any] = {}  # pack name -> Worker
        self.ports: dict[str, int] = {}
        self.failed: dict[str, str] = {}
        self.registered: set[str] = set()
        self.extensions_loaded: set[str] = set()
        self.missing_dependencies: dict[str, list[str]] = {}  # pack -> what it depends on that isn't here
        self.namespaces: dict[str, str] = {}  # /ext/<namespace>/ -> pack name
        self.launcher = _launcher.get_launcher()
        from . import setting
        self.ttl = float(setting(TTL_ENV, str(DEFAULT_TTL_MINUTES))) * 60
        self.last_used: dict[str, float] = {}
        self.busy: dict[str, int] = {}
        self._input_types_cache: dict[tuple[str, str], dict] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()
        self._watchdog: Optional[threading.Thread] = None
        self._stopping = threading.Event()

    # --------------------------------------------------------------- startup
    def setup(self, folders: Optional[list[str]] = None, workers: int = 4) -> None:
        packs = find_packs(folders if folders is not None else pack_folders())
        self.packs = {pack.name: pack for pack in packs}

        def one(pack: Pack) -> tuple[Pack, Optional[dict], Optional[str]]:
            try:
                return pack, describe(pack), None
            except Exception as error:
                return pack, None, str(error)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(one, packs))
        for pack, description, error in results:
            if description is None:
                self.failed[pack.name] = error
                logger.warning("pack isolation: can't load %s: %s", pack.entry, (error or "").splitlines()[-1:] or "")
                continue
            self.descriptions[pack.name] = description
            for node_id in description["nodes"]:
                owner = self.node_packs.setdefault(node_id, pack.name)
                if owner != pack.name:
                    logger.warning("pack isolation: node %s is in %s and %s; using %s", node_id, owner, pack.name, owner)
        self.namespaces = {self.packs[name].namespace: name for name in self.descriptions}
        self._check_dependencies()
        self._register_manifest_nodes()
        self._install_registrations()
        self._register_web_and_routes()
        if self.ttl > 0:
            self._watchdog = threading.Thread(target=self._stop_idle_packs, name="oop-pack-ttl", daemon=True)
            self._watchdog.start()
        logger.info("pack isolation: %d packs, %d nodes; packs start when a prompt or a route needs them%s",
                    len(self.descriptions), len(self.node_packs),
                    f" and stop after {self.ttl / 60:g} minutes unused" if self.ttl > 0 else "")

    def _check_dependencies(self) -> None:
        """Each V2 pack's ``depends_on``: installed, and in the version range it names."""
        by_name = {d["manifest"]["name"]: n for n, d in self.descriptions.items() if "manifest" in d}
        for name, description in self.descriptions.items():
            missing = []
            for dependency in description.get("manifest", {}).get("depends_on", []):
                target = by_name.get(dependency["pack"]) or (dependency["pack"] if dependency["pack"] in self.descriptions else None)
                wanted = dependency.get("version")
                have = self.descriptions[target].get("manifest", {}).get("version") if target else None
                if target is None:
                    missing.append(dependency["pack"] + (f" {wanted}" if wanted else ""))
                elif wanted and have and not _version_in(have, wanted):
                    missing.append(f"{dependency['pack']} {wanted} (installed: {have})")
            if missing:
                self.missing_dependencies[name] = missing
                logger.warning("pack isolation: %s needs %s; its nodes won't run until that's installed",
                               name, ", ".join(missing))

    def _register_manifest_nodes(self) -> None:
        """V2 packs' nodes, registered now from their manifests' schemas: no pack is imported."""
        import nodes
        from . import proxy
        count = 0
        for name, description in self.descriptions.items():
            data = description.get("manifest")
            if data is None:
                continue
            pack = self.packs[name]
            for node_id, node in data["nodes"].items():
                if self.node_packs.get(node_id) != name:
                    continue
                if node_id in nodes.NODE_CLASS_MAPPINGS and not getattr(nodes.NODE_CLASS_MAPPINGS[node_id], "ISOLATED_NODE", None):
                    logger.warning("pack isolation: %s from %s is already a node; skipped", node_id, name)
                    continue
                cls = proxy.make_proxy(node_id, {"kind": "manifest", "schema": node["schema"],
                                                 "is_async": node.get("async", True), "methods": node.get("methods", [])},
                                       pack.key, name, self._input_types(name))
                cls.RELATIVE_PYTHON_MODULE = f"custom_nodes.{name}"
                nodes.NODE_CLASS_MAPPINGS[node_id] = cls
                if description["nodes"].get(node_id, node_id) != node_id:
                    nodes.NODE_DISPLAY_NAME_MAPPINGS[node_id] = description["nodes"][node_id]
                count += 1
            self.registered.add(name)
        if count:
            logger.info("pack isolation: registered %d nodes from V2 pack manifests, importing no pack", count)

    def _install_registrations(self) -> None:
        """Packs' samplers, schedulers, model folders and node replacements, from their manifests or
        descriptions: in ComfyUI's lists now; a sampler or scheduler runs in its pack's process."""
        from comfy_api._impl import registration
        for name, description in self.descriptions.items():
            declared = description.get("manifest") or description.get("registrations") or {}
            if any(declared.get(kind) for kind in registration.KINDS):
                registration.install(declared, lambda kind, fn_name, pack=name: self._registered(pack, kind, fn_name), name)
            if declared.get("scheduler_providers"):
                try:
                    self._install_providers(name, declared)
                except Exception as error:
                    logger.warning("pack isolation: %s's scheduler providers: %s", name, error)

    def _install_providers(self, name: str, declared: dict) -> None:
        """The secure overlay's declarative schedulers: registered now, each call in a short-lived
        process on the pack's Python that loads only the provider's module, as the overlay runs each
        in a fresh guest."""
        import pathlib
        from comfy_api._impl import scheduler_providers as sp
        from . import envs
        pack = self.packs[name]
        providers = sp.decode(declared["scheduler_providers"], v2_dir=pathlib.Path(pack.entry), mount_name=pack.key,
                              module_source=sp.module_source)
        python_box: dict = {}

        def call(provider, projection, steps):
            with self.using(name):
                if "python" not in python_box:
                    python_box["python"] = envs.python_for(pack.entry, declared.get("requirements"))
                return sp.run_in_process(python_box["python"], provider, projection, steps)
        sp.register_providers(name, providers, call)
        logger.info("pack isolation: %s registered schedulers %s", name, ", ".join(p.name for p in providers))

    def _registered(self, pack: str, kind: str, name: str):
        """A sampler or scheduler registered by ``pack``, as a function that runs it in the pack's process."""
        def call(*args: Any, **kwargs: Any) -> Any:
            with self.using(pack):
                return self.ensure_started(pack).call_registered(kind, name, *args, **kwargs)
        call.__qualname__ = f"{pack} {kind} {name}"
        return call

    def _prepare(self, name: str) -> None:
        """Before a pack runs: what it depends on is installed, and the weights it declares are here."""
        missing = self.missing_dependencies.get(name)
        if missing:
            raise RuntimeError(f"{name} needs {', '.join(missing)}, which isn't installed")
        data = self.descriptions[name].get("manifest")
        if data and data.get("weights"):
            from . import weights
            weights.ensure(name, data["weights"])

    def _register_web_and_routes(self) -> None:
        import nodes
        for name, description in self.descriptions.items():
            web = description.get("web_directory")
            if web and os.path.isdir(web):
                nodes.EXTENSION_WEB_DIRS[name] = web
        try:
            from server import PromptServer
            routes = PromptServer.instance.routes
        except Exception:
            return
        self._register_static(PromptServer.instance)
        try:
            existing = list(routes)
        except TypeError:  # a stand-in server (tests) with a write-only route table
            existing = []
        taken = {(r.method, r.path) for r in existing if hasattr(r, "method") and hasattr(r, "path")}
        routes.route("*", ROUTE_MOUNT + "/{namespace}/{tail:.*}")(self._declared_route)
        for name, description in self.descriptions.items():
            for method, path in description.get("routes", ()):
                if (method, path) in taken:
                    logger.warning("pack isolation: %s %s from %s is already a route; not forwarded", method, path, name)
                    continue
                taken.add((method, path))
                routes.route(method, path)(self._route_handler(name))

    def _register_static(self, prompt_server: Any) -> None:
        """A pack's static-file routes are files: served from disk, the pack isn't started."""
        from aiohttp import web
        app = getattr(prompt_server, "app", None)
        if app is None or not hasattr(app, "add_routes") or getattr(app.router, "frozen", False):
            return
        for name, description in self.descriptions.items():
            for prefix, directory in description.get("static", ()):
                if os.path.isdir(directory):
                    try:
                        app.add_routes([web.static(prefix, directory)])
                    except Exception as error:
                        logger.warning("pack isolation: static %s from %s: %s", prefix, name, error)

    async def _declared_route(self, request: Any) -> Any:
        """``/ext/<namespace>/<path>``: a V2 pack's declared route, started if it isn't running."""
        from aiohttp import web
        name = self.namespaces.get(request.match_info["namespace"])
        if name is None:
            return web.json_response({"ok": False, "error": "unknown route"}, status=404)
        if request.content_length is not None and request.content_length > REQUEST_MAX_BYTES:
            return web.json_response({"ok": False, "error": "request body is too large"}, status=413)
        method, path = request.method.lower(), "/" + request.match_info["tail"]
        declared = {tuple(r) for r in self.descriptions[name].get("declared_routes", ())}
        if (method, path) not in declared:  # the manifest says what the pack answers: nothing to start
            return web.json_response({"ok": False, "error": "unknown route"}, status=404)
        body = await request.text()
        message = {"op": "route", "method": method, "path": path,
                   "query": dict(list(request.query.items())[:QUERY_MAX]), "body": body}
        with self.using(name):
            worker = await asyncio.to_thread(self.ensure_started, name)
            try:
                reply = await asyncio.to_thread(worker.side_request, message)
            except Exception as error:
                logger.warning("pack isolation: route %s of %s failed: %s", message["path"], name, error)
                return web.json_response({"ok": False, "error": "route handler failed"}, status=500)
        return web.Response(status=reply["status"], text=reply["body"], content_type=reply["content_type"])

    def _route_handler(self, name: str) -> Any:
        own = {path for _, path in self.descriptions[name].get("routes", ())}

        async def forward(request: Any) -> Any:
            with self.using(name):
                return await forward_in_use(request)

        async def forward_in_use(request: Any) -> Any:
            from aiohttp import web
            worker = await asyncio.to_thread(self.ensure_started, name)
            path_qs = request.path_qs
            matched = request.match_info.route.resource.canonical
            if matched.startswith("/api/") and matched not in own:  # ComfyUI's /api copy of the route
                path_qs = path_qs[len("/api"):]
            message = {"op": "http", "method": request.method, "path_qs": path_qs,
                       "headers": list(request.headers.items()), "body": await request.read()}
            reply = await asyncio.to_thread(worker.side_request, message)
            return web.Response(status=reply["status"], reason=reply["reason"], body=reply["body"],
                                headers=reply["headers"])
        forward.__name__ = f"oop_route_{name}"
        return forward

    # --------------------------------------------------------------- on demand
    async def ensure_node_types(self, node_types: set) -> None:
        """A prompt uses these node types: start their packs (registering their nodes); a use."""
        names = {self.node_packs[t] for t in node_types if t in self.node_packs}
        for name in sorted(names):
            with self.using(name):
                await asyncio.to_thread(self.ensure_started, name)

    @contextmanager
    def using(self, name: str):
        """A use of the pack: it isn't stopped meanwhile, and its idle timer restarts after."""
        with self._lock:
            self.busy[name] = self.busy.get(name, 0) + 1
            self.last_used[name] = time.monotonic()
        try:
            yield
        finally:
            with self._lock:
                self.busy[name] -= 1
                self.last_used[name] = time.monotonic()

    def _stop_idle_packs(self) -> None:
        interval = max(1.0, min(30.0, self.ttl / 4))
        while not self._stopping.wait(interval):
            now = time.monotonic()
            for name in list(self.started):
                with self._lock:
                    idle = self.busy.get(name, 0) == 0 and now - self.last_used.get(name, now) >= self.ttl
                if idle:
                    with self._locks[name]:
                        with self._lock:  # used since we looked?
                            if self.busy.get(name, 0) or time.monotonic() - self.last_used.get(name, 0) < self.ttl:
                                continue
                        worker = self.started.pop(name, None)
                        if worker is not None:
                            logger.info("pack isolation: %s unused for %g minutes; stopping it", name, self.ttl / 60)
                            self._stop_one(name, worker)

    def _stop_one(self, name: str, worker: Any) -> None:
        try:
            worker.stop()
            self.launcher.stop(self.ports[name])
        except Exception as error:
            logger.warning("pack isolation: stopping %s: %s", name, error)

    def ensure_started(self, name: str) -> Any:
        """The pack's worker, starting the pack and registering its nodes if it isn't running."""
        with self._lock:
            lock = self._locks.setdefault(name, threading.Lock())
        with lock:
            worker = self.started.get(name)
            if worker is not None and worker.alive():
                return worker
            pack = self.packs[name]
            self._prepare(name)
            if worker is not None:
                logger.warning("pack isolation: %s exited; starting it again", name)
                self.launcher.stop(self.ports[name])
            worker = self._start(pack)
            self.started[name] = worker
            if name not in self.extensions_loaded:
                from comfy_api._impl import _engine
                _engine.load_pack_extensions(pack.key, pack.entry)  # they run next to the engine
                self.extensions_loaded.add(name)
            if name not in self.registered:
                try:
                    self._register_nodes(pack, worker)
                except BaseException:
                    self._stop_one(name, self.started.pop(name))  # don't leave it running unregistered
                    raise
                self.registered.add(name)
            self.last_used.setdefault(name, time.monotonic())
            return worker

    def _start(self, pack: Pack) -> Any:
        from .host import Worker
        import secrets

        port = _launcher.free_port()
        token = secrets.token_hex(server.TOKEN_BYTES)
        token_file = os.path.join(server.run_dir(), f"{port}.token")
        server._write_private(token_file, token)
        logger.info("pack isolation: starting %s on port %d", pack.name, port)
        self.ports[pack.name] = port
        self.launcher.start(pack.entry, port, token_file)
        try:
            return Worker(pack.entry, port=port, token=token)
        except BaseException:
            self.launcher.stop(port)
            raise

    def _register_nodes(self, pack: Pack, worker: Any) -> None:
        import nodes
        from . import proxy

        descriptions = worker.side_request({"op": "nodes"})
        display = self.descriptions[pack.name]["nodes"]
        for node_id, description in descriptions.items():
            if self.node_packs.get(node_id, pack.name) != pack.name:
                continue
            if node_id in nodes.NODE_CLASS_MAPPINGS and not getattr(nodes.NODE_CLASS_MAPPINGS[node_id], "ISOLATED_NODE", None):
                logger.warning("pack isolation: %s from %s is already a node; skipped", node_id, pack.name)
                continue
            cls = proxy.make_proxy(node_id, description, pack.key, pack.name, self._input_types(pack.name))
            cls.RELATIVE_PYTHON_MODULE = f"custom_nodes.{pack.name}"
            nodes.NODE_CLASS_MAPPINGS[node_id] = cls
            if display.get(node_id) and display[node_id] != node_id:
                nodes.NODE_DISPLAY_NAME_MAPPINGS[node_id] = display[node_id]
        logger.info("pack isolation: %s registered %d nodes", pack.name, len(descriptions))

    def _input_types(self, name: str) -> Any:
        def input_types(node_id: str) -> dict:
            """A V1 node's INPUT_TYPES: asked of the pack while it runs; the last answer while it's
            stopped (listing nodes isn't a use, and doesn't start a pack)."""
            worker = self.started.get(name)
            key = (name, node_id)
            if worker is None or not worker.alive():
                if key in self._input_types_cache:
                    return self._input_types_cache[key]
                worker = self.ensure_started(name)
            self._input_types_cache[key] = worker.side_request({"op": "input_types", "node": node_id})
            return self._input_types_cache[key]
        return input_types

    def worker_for_node(self, node_class: Any) -> Any:
        name, _ = node_class.ISOLATED_NODE
        return self.ensure_started(name)

    def stop(self) -> None:
        self._stopping.set()
        for name, worker in list(self.started.items()):
            self._stop_one(name, worker)
        self.started.clear()
