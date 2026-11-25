"""The V4 packs ComfyUI has loaded, and the routes they serve.

ComfyUI's loader is wrapped in `_loading_pack` and calls `_add_pack` for each pack with a
`comfy_entrypoint` it loads; its server calls `_mount_routes` once it has added its own routes.
What a pack registers through the API while it loads is taken back if it does not load, or when
it is taken away. `_remove_pack` takes away a pack that fails to load after `_add_pack`. Taking
packs away is `_flush_pack_resources`, which `_remove_pack` calls for the packs of a path, and
`_flush_all_pack_resources`, which runs when ComfyUI exits. Packs do not call anything here.
"""

import asyncio
import atexit
import contextlib
import functools
import logging
import os
import re
import threading
from collections.abc import Callable, Iterable
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, get_args

from aiohttp import web
from aiohttp.web_urldispatcher import DynamicResource

from app import extension_packs
from comfy_api._exceptions import InterruptProcessingException

from . import _pack_processes
from ._extension import ComfyExtension, Route, RouteRequest

_METHODS = get_args(Route.__annotations__["method"])


class _Address(NamedTuple):
    method: str
    path: str
    """The literal path, or aiohttp's formatter of the pattern (`{name}` for each part)."""
    pattern: re.Pattern


@dataclass
class _Pack:
    name: str
    folder: str
    module_path: str
    nodes: list[type]
    routes: list[Route] = field(default_factory=list)
    addresses: list[_Address] = field(default_factory=list)
    # (table, key, value) entries the loader made for this pack in its folder tables
    folders: list[tuple[dict[str, str], str, str]] = field(default_factory=list)
    # node id: (class, display name) of the node, ComfyUI's or a loaded pack's, this pack's replaces
    replaced: dict[str, tuple[type, str | None]] = field(default_factory=dict)
    # The record of its loading, which a task the pack started while loading still has
    loading: "_Loading | None" = None


@dataclass
class _Loading:
    module_path: str
    added: bool = False
    # None while the loader loads it, then whether it is loaded: False once it is taken away
    loaded: bool | None = None
    # How to take back what the pack registered through the API, in the order it registered
    registered: list[Callable[[], None]] = field(default_factory=list)


_packs: dict[tuple[str, str], _Pack] = {}
# aiohttp takes no route once the application serves; `_mount_routes` sets this.
_serving = False
# The web folders of V4 packs that ComfyUI serves, as `_mount_routes` found them; aiohttp cannot
# drop a route.
_served_web: dict[str, str] = {}
# Packs taken away whose node classes stay registered while a queued or running prompt uses one:
# ComfyUI looks node classes up by id while it runs a prompt.
_retired: list[_Pack] = []
# The loader, the prompt thread (a pack disabled) and the host flush.
_flushing = threading.Lock()
# How often ComfyUI looks again whether the retired packs' nodes are still used.
_RETIRED_POLL = 1.0
# The pack ComfyUI's loader is loading in this task, or loaded in the task this one came from.
_loading: ContextVar[_Loading | None] = ContextVar("v4_pack_loading", default=None)
# A pack's task may register on one thread while the pack is taken away on another.
_registering = threading.Lock()


def _loading_pack(load_custom_node):
    """Wrap ComfyUI's `load_custom_node`, so that what a pack registers through the API while it
    loads is taken back if it does not load."""

    @functools.wraps(load_custom_node)
    async def load(module_path: str, *args, **kwargs) -> bool:
        pack = _Loading(module_path)
        token = _loading.set(pack)
        loaded = False
        try:
            loaded = await load_custom_node(module_path, *args, **kwargs)
            return loaded
        finally:
            _loading.reset(token)
            if not loaded:
                if pack.added:
                    _remove_pack(module_path)
                _take_back(pack)
            else:
                with _registering:
                    if pack.loaded is None:
                        pack.loaded = True

    return load


def _registered(take_back: Callable[[], None]) -> None:
    """Note how to take back what a pack has just registered, if it is the pack's loading that
    registered it: its import, its `on_load`, or a task or `asyncio.to_thread` call either
    started. From a pack that did not load, or has been taken away, it is taken back at once.
    Any other registration stays: one a loaded pack makes later from elsewhere, and one made
    on a thread a pack started itself, which does not say whose it is."""
    pack = _loading.get()
    if pack is None:
        return
    with _registering:
        if pack.loaded is not False:
            pack.registered.append(take_back)
            return
    take_back()


def _take_back(pack: _Loading) -> None:
    """Take back what the pack registered, last first, and from now on what it registers."""
    with _registering:
        pack.loaded = False
    while pack.registered:
        take_back = pack.registered.pop()
        try:
            take_back()
        except Exception:
            logging.exception("Could not take back what a pack registered")


def _address(method: str, info: dict) -> _Address:
    """A path as aiohttp's `get_info()` describes it."""
    path = info["path"] if "path" in info else info["formatter"]
    return _Address(method, path, info.get("pattern") or re.compile(re.escape(path)))


def _parse(method: str, path: str) -> _Address:
    """ValueError if aiohttp cannot parse `path`, or a `{name:regex}` part has a named group of
    its own: aiohttp fails a request in which such a group takes no part."""
    address = _address(method, web.UrlDispatcher().add_resource(path).get_info())
    if len(address.pattern.groupindex) > len(re.findall(r"\{\w+\}", address.path)):
        raise ValueError("a {name:regex} part has a named group of its own")
    return address


def _segments(address: _Address) -> tuple[list[str | None], bool]:
    """A pattern's segments, None for one with a `{name}` part, up to its first `{name:regex}`
    part; and whether it has one, whose regex may span segments."""
    segments = []
    for segment in address.path.split("/"):
        names = re.findall(r"\{(\w+)\}", segment)
        if any(
            f"(?P<{n}>{DynamicResource.GOOD})" not in address.pattern.pattern
            for n in names
        ):
            return segments, True
        segments.append(None if names else segment)
    return segments, False


def _overlap(first: _Address, second: _Address) -> bool:
    """Whether one request could reach both; `*` answers every method.

    Two patterns overlap unless a literal segment differs before either has a `{name:regex}`
    part, or they cannot have the same number of segments. This errs towards overlap.
    """
    if first.method != second.method and "*" not in (first.method, second.method):
        return False
    if "{" not in first.path:
        return bool(second.pattern.fullmatch(first.path))
    if "{" not in second.path:
        return bool(first.pattern.fullmatch(second.path))
    (first_segments, first_open), (second_segments, second_open) = (
        _segments(first),
        _segments(second),
    )
    # A `{name:regex}` part follows a "/", so its pattern has more segments than come before it.
    if (
        not first_open
        and len(first_segments) < len(second_segments) + second_open
        or not second_open
        and len(second_segments) < len(first_segments) + first_open
    ):
        return False
    return all(
        x == y or None in (x, y) for x, y in zip(first_segments, second_segments)
    )


def _conflict(addresses: list[_Address], served: list[_Address]) -> str | None:
    for address in addresses:
        for other in served:
            if _overlap(address, other):
                return (
                    f"{address.method} {address.path} is already served"
                    f" ({other.method} {other.path})"
                )
    return None


def _add_routes(pack: _Pack, routes: list[Route]) -> str | None:
    """Record `routes` on `pack`, or say why the pack is refused."""
    if routes and _serving:
        return "routes can only be added before ComfyUI starts serving"
    if not isinstance(routes, list):
        return f"get_routes returned {type(routes).__name__}, not a list of routes"
    loaded = [address for other in _packs.values() for address in other.addresses]
    for route in routes:
        if not isinstance(route, dict):
            return f"route {route!r} is not a dict of method, path and handler"
        method, path = route.get("method"), route.get("path")
        if method not in _METHODS:
            return f"route method {method!r} is not one of {', '.join(_METHODS)}"
        if not isinstance(path, str) or not path.startswith("/"):
            return f"route path {path!r} does not start with '/'"
        if not callable(route.get("handler")):
            return f"route {method} {path} has no handler"
        # aiohttp answers HEAD requests on a GET route.
        methods = ("GET", "HEAD") if method == "GET" else (method,)
        try:
            addresses = [_parse(m, p) for m in methods for p in (path, "/api" + path)]
        except ValueError as error:
            return f"route path {path!r}: {error}"
        # aiohttp prefers a literal path to a pattern, so one pack may have both.
        if any(
            _overlap(a, b) and ("{" in a.path) == ("{" in b.path)
            for a in addresses
            for b in pack.addresses
        ):
            return f"{method} {path} overlaps another of its routes"
        problem = _conflict(addresses, loaded)
        if problem is not None:
            return problem
        pack.routes.append(
            {"method": method, "path": path, "handler": route["handler"]}
        )
        pack.addresses += addresses
    return None


def _folders(module_path: str) -> list[tuple[dict[str, str], str, str]]:
    """The loader's `LOADED_MODULE_DIRS` and `EXTENSION_WEB_DIRS` entries for a pack: those that
    point into its folder, whichever name the loader keyed them by."""
    import nodes

    tables = (nodes.LOADED_MODULE_DIRS, nodes.EXTENSION_WEB_DIRS)
    if os.path.isfile(module_path):
        # The loader keys a one-file pack's entries by its path without the extension.
        key = os.path.splitext(module_path)[0]
        return [(table, key, table[key]) for table in tables if key in table]
    root = Path(module_path).absolute()
    return [
        (table, key, value)
        for table in tables
        for key, value in table.items()
        if Path(value).absolute().is_relative_to(root)
    ]


def _give_back_folders(module_path: str) -> None:
    """Undo the loader's folder entries for a pack that is not loaded; a loaded V4 pack whose
    entry it overwrote gets its own back."""
    loaded = {(id(t), k): v for pack in _packs.values() for t, k, v in pack.folders}
    for table, key, _ in _folders(module_path):
        if (id(table), key) in loaded:
            table[key] = loaded[id(table), key]
        else:
            del table[key]


def _unserve_web_folders_outside(module_path: str) -> None:
    """Take out the web folders the loader registered for a pack that runs in its own process,
    if they are outside the pack's folder: ComfyUI serves a web folder to any browser with its
    own access to files. They are the entries that changed while the pack loaded, the loader
    loading one pack at a time; what an entry was before is put back."""
    import folder_paths
    import nodes

    process = _pack_processes.processes.get(module_path)
    if process is None:
        return
    folder = module_path if os.path.isdir(module_path) else os.path.dirname(module_path)
    served, before = nodes.EXTENSION_WEB_DIRS, process.web_folders
    for name, web_folder in list(served.items()):
        if before.get(name) == web_folder:
            continue
        if folder_paths.is_within_directory(folder, web_folder):
            continue
        if name in before:
            served[name] = before[name]
        else:
            del served[name]
        logging.warning(
            f"Pack {name!r} in {module_path}: its web folder {web_folder!r} is not served: "
            "it is outside the pack's folder."
        )


async def _add_pack(
    module_path: str, extension: ComfyExtension, node_list: list[type]
) -> bool:
    """Record a V4 pack: its routes, the node classes in `node_list` (the loader registers them
    next) and the loader's module and web folder entries for it. True, recording nothing, for an
    extension of an older API.

    Gives back the folder entries the loader made for the pack (a loaded V4 pack whose entry it
    overwrote gets its own back) and records nothing if the pack must not load: raises if its
    `get_routes` or a node's schema does; returns False, with a warning, if a loaded pack has both
    its id or a route is refused (see `ComfyExtension.get_routes`), and with an error if its
    `pyproject.toml` gives no valid id (`app.extension_packs.pack_identity`).
    """

    if not isinstance(extension, ComfyExtension):
        return True
    if _pack_processes.isolated(module_path):
        _unserve_web_folders_outside(module_path)
    import nodes

    folder = nodes.get_module_name(module_path)
    try:
        name, _ = extension_packs.pack_identity(module_path)
    except ValueError as reason:
        logging.error(f"V4 pack {module_path} is not loaded: {reason}.")
        _refuse(module_path)
        return False
    try:
        routes = await extension.get_routes()
        node_ids = [node_cls.GET_SCHEMA().node_id for node_cls in node_list]
    except Exception:
        # The loader reports the error; the pack's folder entries go, as a refused pack's do.
        _refuse(module_path)
        raise
    pack = _Pack(name, folder, module_path, list(node_list))
    holder = _holder(name)
    if holder not in (None, os.path.abspath(module_path)):
        problem = f"its id {name!r} is already held by {holder}"
    else:
        problem = _add_routes(pack, routes)
    if problem is not None:
        logging.warning(f"Pack {name!r} in {module_path} is disabled: {problem}.")
        _refuse(module_path)
        return False
    pack.folders = _folders(module_path)
    pack.replaced = _replaced(node_ids)
    loading_pack = _loading.get()
    if loading_pack is not None:
        loading_pack.added = True
        pack.loading = loading_pack
    _packs[name, folder] = pack
    return True


def _holder(pack_id: str) -> str | None:
    """The folder of the loaded V4 pack, or the V4 pack ComfyUI verified before loading, that has
    this id."""
    holders = [p.module_path for p in _packs.values() if p.name == pack_id]
    holders.append(extension_packs.held_by(pack_id))
    return next((os.path.abspath(path) for path in holders if path is not None), None)


def _refuse(module_path: str) -> None:
    """Undo what the loader did for a pack that is not loaded, and end its process if it has one."""
    _give_back_folders(module_path)
    _pack_processes.unload(module_path)


def _replaced(node_ids: list[str]) -> dict[str, tuple[type, str | None]]:
    """The loaded nodes, with their display names, that nodes of these ids replace."""
    import nodes

    return {
        node_id: (
            nodes.NODE_CLASS_MAPPINGS[node_id],
            nodes.NODE_DISPLAY_NAME_MAPPINGS.get(node_id),
        )
        for node_id in node_ids
        if node_id in nodes.NODE_CLASS_MAPPINGS
    }


def _hand_on_replaced(pack: _Pack) -> None:
    """A pack that replaced one of `pack`'s nodes now replaces what `pack` replaced."""
    for other in [*_packs.values(), *_retired]:
        for node_id, (node_cls, _) in list(other.replaced.items()):
            if node_cls not in pack.nodes:
                continue
            if node_id in pack.replaced:
                other.replaced[node_id] = pack.replaced[node_id]
            else:
                del other.replaced[node_id]


def _server():
    from server import PromptServer

    return getattr(PromptServer, "instance", None)


def _in_use(node_classes: list[type]) -> bool:
    """Whether a queued or running prompt names one of these node classes."""
    import nodes

    server = _server()
    if server is None:
        return False
    ids = {i for i, c in nodes.NODE_CLASS_MAPPINGS.items() if c in node_classes}
    running, queued = server.prompt_queue.get_current_queue_volatile()
    return any(
        node.get("class_type") in ids
        for item in running + queued
        for node in item[2].values()
    )


def _remove_nodes(pack: _Pack) -> None:
    import nodes

    listed = {node_cls for other in _packs.values() for node_cls in other.nodes}
    for node_id, node_cls in list(nodes.NODE_CLASS_MAPPINGS.items()):
        if node_cls not in pack.nodes or node_cls in listed:
            continue
        nodes.NODE_DISPLAY_NAME_MAPPINGS.pop(node_id, None)
        if node_id not in pack.replaced:
            del nodes.NODE_CLASS_MAPPINGS[node_id]
            continue
        nodes.NODE_CLASS_MAPPINGS[node_id], display_name = pack.replaced[node_id]
        if display_name is not None:
            nodes.NODE_DISPLAY_NAME_MAPPINGS[node_id] = display_name


def _flush(packs: list[_Pack]) -> None:
    with _flushing:
        # Another thread may have taken one away meanwhile.
        for pack in [p for p in packs if _packs.get((p.name, p.folder)) is p]:
            _pack_processes.unload(pack.module_path)
            del _packs[pack.name, pack.folder]
            _hand_on_replaced(pack)
            _give_back_folders(pack.module_path)
            if pack.loading is not None:
                _take_back(pack.loading)
            _retired.append(pack)
        for pack in [p for p in _retired if not _in_use(p.nodes)]:
            _retired.remove(pack)
            _remove_nodes(pack)
        if _retired:
            _look_again()


def _look_again() -> None:
    """Flush again in `_RETIRED_POLL` seconds, on the server's loop, for the retired packs."""
    loop = _server().loop
    # A loop that has closed runs nothing more: ComfyUI is ending.
    with contextlib.suppress(RuntimeError):
        loop.call_soon_threadsafe(loop.call_later, _RETIRED_POLL, _flush, [])


def _remove_pack(module_path: str) -> None:
    """Disable the pack `_add_pack` recorded from `module_path`, as `_flush_pack_resources` does:
    a pack that fails to load after `_add_pack`.
    """
    _flush_pack_resources(
        (p.name, p.folder) for p in _packs.values() if p.module_path == module_path
    )


def _served(app: web.Application) -> list[_Address]:
    """What `app` answers. A static folder or a sub-application answers every path under its
    prefix; the web root, at "/", every path under each of its entries. A sub-application
    matched by domain answers what its own routes do, whatever the domain."""
    served = []
    for resource in app.router.resources():
        info = resource.get_info()
        if "rule" in info:
            served += _served(info["app"])
            continue
        if "prefix" not in info:
            served += [_address(route.method, info) for route in resource]
            continue
        method = "*" if "app" in info else "GET"
        prefix = info["prefix"]
        prefixes = (
            [prefix]
            if prefix
            else ["/" + entry for entry in os.listdir(info["directory"])]
        )
        # A prefix is literal text: a folder's entries need not be valid route syntax.
        served += [
            _Address(method, p + tail, re.compile(re.escape(p) + regex))
            for p in prefixes
            for tail, regex in (("", ""), ("/{tail}", "/.*"))
        ]
    return served


def _serve(pack: _Pack, route: Route):
    async def serve(request: web.Request) -> web.StreamResponse:
        if _packs.get((pack.name, pack.folder)) is not pack:
            raise web.HTTPNotFound()
        route_request = RouteRequest(
            method=request.method,
            path=request.path,
            match_info=dict(request.match_info),
            query=list(request.query.items()),
            headers={
                name.lower(): ", ".join(request.headers.getall(name))
                for name in request.headers
            },
            body=await request.read(),
            remote=request.remote,
        )
        try:
            response = await route["handler"](route_request)
        except (InterruptProcessingException, web.HTTPException, asyncio.TimeoutError) as error:
            # aiohttp would drop the connection for the first, a BaseException, and answer with
            # the second's own status; a pack in its own process cannot raise either.
            raise web.HTTPInternalServerError() from error
        return web.Response(
            status=response["status"],
            headers=response["headers"],
            body=response["body"],
        )

    return serve


def _mount_routes(server) -> None:
    """Serve every recorded pack's routes on `server.app`. ComfyUI calls this once, after it has
    added all of its own routes and before it serves.

    A pack with a route that `server.app` already answers (a route of ComfyUI or of an older-API
    pack, a static folder, a sub-application) is disabled, with a warning, as `_remove_pack` does;
    what answered that route goes on answering it. A route of a pack taken away later answers 404,
    and so does its web folder.
    Each `--pack-process` name that no loaded pack has is warned of, and on Windows that the
    option is not supported there and the named packs are not loaded.
    """
    global _serving, _served_web
    import nodes

    _pack_processes.warn_at_start()
    _served_web = {
        key: value
        for pack in _packs.values()
        for table, key, value in pack.folders
        if table is nodes.EXTENSION_WEB_DIRS and table.get(key) == value
    }
    if _served_web:
        server.app.middlewares.append(_unserved_web_folders)
    routed = any(pack.addresses for pack in _packs.values())
    served = _served(server.app) if routed else []
    for pack in list(_packs.values()):
        problem = _conflict(pack.addresses, served)
        if problem is not None:
            logging.warning(
                f"Pack {pack.name!r} in {pack.module_path} is disabled: {problem}."
            )
            _flush([pack])
    server.app.add_routes(
        web.route(route["method"], path, _serve(pack, route))
        for pack in _packs.values()
        for route in pack.routes
        for path in (route["path"], "/api" + route["path"])
    )
    _serving = True


@web.middleware
async def _unserved_web_folders(request: web.Request, handler):
    """Answer 404 under the web folder of a V4 pack taken away since ComfyUI started serving."""
    import nodes

    prefix, _, rest = request.path.partition("/extensions/")
    name = rest.partition("/")[0]
    # A static folder answers GET and HEAD; a route under its address answers the rest.
    if prefix == "" and name in _served_web and request.method in ("GET", "HEAD"):
        if nodes.EXTENSION_WEB_DIRS.get(name) != _served_web[name]:
            raise web.HTTPNotFound()
    return await handler(request)


def _flush_pack_resources(packs: Iterable[tuple[str, str]]) -> None:
    """Take away the packs named by (name, folder name): end the process of one that runs in its
    own, answer 404 to its routes and its web folder, and remove its nodes, putting back
    any node of another pack they replaced. The nodes stay while a queued or running prompt names
    one of them, and go once none does: ComfyUI looks again every `_RETIRED_POLL` seconds. A node
    that only another node expands into is not seen there, and fails as a missing node would.

    The cache providers, node replacements and model folder directories the pack registered
    through the API as it loaded, or in a task or `asyncio.to_thread` call its loading
    started, are taken back. What it registered from a node, a route or another thread, and
    anything it changed in ComfyUI directly, stays until ComfyUI restarts. Packs not loaded
    are ignored.
    """
    _flush([_packs[key] for key in set(packs) if key in _packs])


def _flush_all_pack_resources() -> None:
    """Take away every V4 pack, as `_flush_pack_resources` does."""
    _flush(list(_packs.values()))


@atexit.register
def _unload_all() -> None:
    """ComfyUI exits: take every pack away, and end the pack processes still loading, which no
    pack stands for yet."""
    _flush_all_pack_resources()
    _pack_processes.end_all()
