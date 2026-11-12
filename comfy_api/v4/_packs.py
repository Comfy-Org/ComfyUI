"""The V4 packs ComfyUI has loaded, and the routes they serve.

ComfyUI's loader calls `add_pack` for each pack it loads and `remove_pack` when loading fails
after that; its server calls `mount_routes` once it has added its own routes. Packs do not call
anything here.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, get_args

from aiohttp import web
from aiohttp.web_urldispatcher import DynamicResource

from comfy_api._exceptions import InterruptProcessingException

from ._extension import ComfyExtension, Route, RouteRequest

METHODS = get_args(Route.__annotations__["method"])


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
    # node id: (class, display name) of a loaded pack's node this pack's node replaces
    replaced: dict[str, tuple[type, str | None]] = field(default_factory=dict)


_packs: dict[tuple[str, str], _Pack] = {}
# aiohttp takes no route once the application serves; `mount_routes` sets this.
_serving = False


def _address(method: str, info: dict) -> _Address:
    """A path as aiohttp's `get_info()` describes it."""
    path = info["path"] if "path" in info else info["formatter"]
    return _Address(method, path, info.get("pattern") or re.compile(re.escape(path)))


def _parse(method: str, path: str) -> _Address:
    """ValueError if aiohttp cannot parse `path`."""
    return _address(method, web.UrlDispatcher().add_resource(path).get_info())


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

    Two patterns overlap unless a literal segment differs, or the number of segments does,
    before either has a `{name:regex}` part. This errs towards overlap.
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
    if len(first_segments) != len(second_segments) and not (first_open or second_open):
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
        if method not in METHODS:
            return f"route method {method!r} is not one of {', '.join(METHODS)}"
        if not isinstance(path, str) or not path.startswith("/"):
            return f"route path {path!r} does not start with '/'"
        if not callable(route.get("handler")):
            return f"route {method} {path} has no handler"
        try:
            addresses = [_parse(method, path), _parse(method, "/api" + path)]
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


def _identity(module_path: str) -> tuple[str, str]:
    import nodes
    from comfy_config import config_parser

    folder = nodes.get_module_name(module_path)
    try:
        name = config_parser.extract_node_configuration(module_path).project.name
    except Exception:
        # The loader loads a pack whose pyproject.toml it cannot read; it is named by its folder.
        name = ""
    return name or folder, folder


def _folders(
    module_path: str, name: str, folder: str
) -> list[tuple[dict[str, str], str, str]]:
    """The loader's `LOADED_MODULE_DIRS` and `EXTENSION_WEB_DIRS` entries for a pack."""
    import nodes

    tables = (nodes.LOADED_MODULE_DIRS, nodes.EXTENSION_WEB_DIRS)
    if os.path.isfile(module_path):
        # The loader keys a one-file pack's entries by its path without the extension.
        key = os.path.splitext(module_path)[0]
        return [(table, key, table[key]) for table in tables if key in table]
    root = Path(module_path).absolute()
    return [
        (table, key, table[key])
        for table in tables
        for key in {name, folder}
        if key in table and Path(table[key]).absolute().is_relative_to(root)
    ]


def _give_back_folders(module_path: str, name: str, folder: str) -> None:
    """Undo the loader's folder entries for a pack that is not loaded; a loaded pack whose entry
    it overwrote gets its own back."""
    loaded = {(id(t), k): v for pack in _packs.values() for t, k, v in pack.folders}
    for table, key, _ in _folders(module_path, name, folder):
        if (id(table), key) in loaded:
            table[key] = loaded[id(table), key]
        else:
            del table[key]


async def add_pack(
    module_path: str, extension: ComfyExtension, node_list: list[type]
) -> bool:
    """Record a V4 pack: its routes, the node classes in `node_list` (the loader registers them
    next) and the loader's module and web folder entries for it. True, recording nothing, for an
    extension of an older API.

    Gives back the folder entries the loader made for the pack (a loaded pack whose entry it
    overwrote gets its own back) and records nothing if the pack must not load: raises if its
    `get_routes` or a node's schema does; returns False, with a warning, if a loaded pack has both
    its name and its folder name or a route is refused (see `ComfyExtension.get_routes`). What its
    `on_load` registered stays.
    """
    import nodes

    if not isinstance(extension, ComfyExtension):
        return True
    name, folder = _identity(module_path)
    try:
        routes = await extension.get_routes()
        node_ids = [node_cls.GET_SCHEMA().node_id for node_cls in node_list]
    except Exception:
        # The loader reports the error; the pack's folder entries go, as a refused pack's do.
        _give_back_folders(module_path, name, folder)
        raise
    pack = _Pack(name, folder, module_path, list(node_list))
    loaded = _packs.get((name, folder))
    if loaded is not None:
        problem = f"{loaded.module_path} has the same name and folder name"
    else:
        problem = _add_routes(pack, routes)
    if problem is not None:
        logging.warning(f"Pack {name!r} in {module_path} is disabled: {problem}.")
        _give_back_folders(module_path, name, folder)
        return False
    pack.folders = _folders(module_path, name, folder)
    pack.replaced = {
        node_id: (
            nodes.NODE_CLASS_MAPPINGS[node_id],
            nodes.NODE_DISPLAY_NAME_MAPPINGS.get(node_id),
        )
        for node_id in node_ids
        if node_id in nodes.NODE_CLASS_MAPPINGS
    }
    _packs[name, folder] = pack
    return True


def _hand_on_replaced(pack: _Pack) -> None:
    """A loaded pack that replaced one of `pack`'s nodes now replaces what `pack` replaced."""
    for other in _packs.values():
        for node_id, (node_cls, _) in list(other.replaced.items()):
            if node_cls not in pack.nodes:
                continue
            if node_id in pack.replaced:
                other.replaced[node_id] = pack.replaced[node_id]
            else:
                del other.replaced[node_id]


def _remove(pack: _Pack) -> None:
    import nodes

    del _packs[pack.name, pack.folder]
    _hand_on_replaced(pack)
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
    _give_back_folders(pack.module_path, pack.name, pack.folder)


def remove_pack(module_path: str) -> None:
    """Disable the pack `add_pack` recorded from `module_path`: remove its nodes, putting back any
    node of another pack they replaced, give back the loader's folder entries for it as `add_pack`
    does on refusal, and never serve its routes. For a pack that fails to load after `add_pack`,
    before ComfyUI serves.
    """
    for pack in [p for p in _packs.values() if p.module_path == module_path]:
        _remove(pack)


def _served(app: web.Application) -> list[_Address]:
    """What `app` answers. A static folder or a sub-application answers every path under its
    prefix; the web root, at "/", every path under each of its entries. A sub-application
    matched by domain has no path of its own and is not counted."""
    served = []
    for resource in app.router.resources():
        info = resource.get_info()
        if "rule" in info:
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


def _serve(route: Route):
    async def serve(request: web.Request) -> web.StreamResponse:
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
        except InterruptProcessingException as error:
            # A BaseException: aiohttp would drop the connection instead of answering 500.
            raise web.HTTPInternalServerError() from error
        return web.Response(
            status=response["status"],
            headers=response["headers"],
            body=response["body"],
        )

    return serve


def mount_routes(server) -> None:
    """Serve every recorded pack's routes on `server.app`. ComfyUI calls this once, after it has
    added all of its own routes and before it serves.

    A pack with a route that `server.app` already answers (a route of ComfyUI or of an older-API
    pack, a static folder, a sub-application) is disabled, with a warning, as `remove_pack` does;
    what answered that route goes on answering it.
    """
    global _serving
    served = _served(server.app)
    for pack in list(_packs.values()):
        problem = _conflict(pack.addresses, served)
        if problem is not None:
            logging.warning(
                f"Pack {pack.name!r} in {pack.module_path} is disabled: {problem}."
            )
            _remove(pack)
    server.app.add_routes(
        web.route(route["method"], path, _serve(route))
        for pack in _packs.values()
        for route in pack.routes
        for path in (route["path"], "/api" + route["path"])
    )
    _serving = True
