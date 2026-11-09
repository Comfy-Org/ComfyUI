"""A V2 pack's manifest, ``comfy-pack.json``: everything ComfyUI needs to know about a pack without
importing it. See ``comfy_api/docs/pack-manifest.md``.

    {"format": "comfy-pack/1", "name": ..., "version": ..., "weights": [...], "requirements": [...],
     "depends_on": [...], "permissions": [...], "nodes": {node_id: {"schema": ..., ...}}, ...}

``read(entry)`` loads and checks it (and reads the older ``secure-nodes.json`` conversions ship);
``generate(entry)`` writes it from the pack's code, once, at publish time; ``encode_schema`` /
``decode_schema`` carry an ``io.Schema`` as JSON.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from enum import Enum
from typing import Any, Optional

FORMAT = "comfy-pack/1"
FILE = "comfy-pack.json"
LEGACY_FILE = "secure-nodes.json"  # what pack conversions wrote before comfy-pack.json
NODE_METHODS = ("validate_inputs", "fingerprint_inputs", "check_lazy_status")
_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    """A manifest that ComfyUI can't use."""


# a model folder a pack names: one plain name, which ComfyUI places under its models directory
_FOLDER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def check_folder_name(folder: Any) -> str:
    """A model folder name a pack gave, refused unless it's one plain name: no path separators, no '..', not
    absolute. Model files are read only from ComfyUI's model folders."""
    if not isinstance(folder, str) or not _FOLDER.match(folder) or ".." in folder:
        raise ManifestError(f"model folder {folder!r} must be a plain name (letters, digits, '_', '-', '.')")
    return folder


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def path_for(entry: str) -> Optional[str]:
    for name in (FILE, LEGACY_FILE):
        path = os.path.join(entry, name)
        if os.path.isfile(path):
            return path
    return None


def read(entry: str) -> Optional[dict]:
    """The pack's manifest, checked, or None if it has none (a V1 pack)."""
    path = path_for(entry)
    if path is None:
        return None
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    if os.path.basename(path) == LEGACY_FILE:
        if "format" not in data and "nodes" not in data:
            return None  # an early conversion's route list only: the pack is read by importing it
        data = _from_legacy(data, entry)
    check(data, path)
    return data


def _req(data: dict, key: str, kind: type, where: str) -> Any:
    value = data.get(key)
    if not isinstance(value, kind):
        raise ManifestError(f"{where}: {key!r} must be {kind.__name__}")
    return value


def check(data: Any, where: str = FILE) -> None:
    """Refuse a manifest ComfyUI can't act on, with the reason."""
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ManifestError(f"{where}: not a {FORMAT} manifest")
    if not _NAME.match(_req(data, "name", str, where)):
        raise ManifestError(f"{where}: name must be lowercase letters, digits, '.', '_' or '-'")
    _req(data, "version", str, where)
    for key in ("requirements", "permissions"):
        if not all(isinstance(item, str) and item for item in data.get(key, [])):
            raise ManifestError(f"{where}: {key!r} must be a list of strings")
    for index, dependency in enumerate(data.get("depends_on", [])):
        if not isinstance(dependency, dict) or not isinstance(dependency.get("pack"), str):
            raise ManifestError(f"{where}: depends_on[{index}] needs a 'pack' name")
    for index, weight in enumerate(data.get("weights", [])):
        if not isinstance(weight, dict) or not all(isinstance(weight.get(k), str) and weight[k] for k in ("folder", "filename")):
            raise ManifestError(f"{where}: weights[{index}] needs 'folder' and 'filename'")
        if ".." in weight["filename"].replace("\\", "/").split("/") or os.path.isabs(weight["filename"]):
            raise ManifestError(f"{where}: weights[{index}] filename can't leave its folder")
        try:
            check_folder_name(weight["folder"])
        except ManifestError as error:
            raise ManifestError(f"{where}: weights[{index}]: {error}") from None
        if weight.get("sha256") is not None and not _SHA256.match(weight["sha256"]):
            raise ManifestError(f"{where}: weights[{index}] sha256 must be 64 lowercase hex digits")
    nodes = _req(data, "nodes", dict, where)
    for node_id, node in nodes.items():
        if not isinstance(node, dict) or not isinstance(node.get("schema"), dict):
            raise ManifestError(f"{where}: node {node_id!r} needs a 'schema'")
    for key in ("schedulers", "samplers"):
        if not all(isinstance(n, str) and n for n in data.get(key, [])):
            raise ManifestError(f"{where}: {key!r} must be a list of names")
    if not isinstance(data.get("scheduler_providers", []), list):
        raise ManifestError(f"{where}: 'scheduler_providers' must be a list (the secure overlay's format)")
    if not isinstance(data.get("model_folders", {}), dict):
        raise ManifestError(f"{where}: 'model_folders' must be {{folder: [extensions] or null}}")
    for folder in data.get("model_folders", {}):
        try:
            check_folder_name(folder)
        except ManifestError as error:
            raise ManifestError(f"{where}: {error}") from None
    for index, replacement in enumerate(data.get("node_replacements", [])):
        if not isinstance(replacement, dict) or not all(isinstance(replacement.get(k), str) for k in ("old_node_id", "new_node_id")):
            raise ManifestError(f"{where}: node_replacements[{index}] needs 'old_node_id' and 'new_node_id'")
    for index, route in enumerate(data.get("routes", [])):
        if not isinstance(route, dict) or not all(isinstance(route.get(k), str) and route[k] for k in ("method", "path", "module", "handler")):
            raise ManifestError(f"{where}: routes[{index}] needs 'method', 'path', 'module' and 'handler'")
        if route["method"].lower() not in ("get", "post", "put", "patch", "delete"):
            raise ManifestError(f"{where}: routes[{index}] method must be get, post, put, patch or delete")
        if not route["path"].startswith("/") or any(c in route["path"] for c in "{}?#") or ".." in route["path"].split("/"):
            raise ManifestError(f"{where}: routes[{index}] path must be a literal path such as '/presets'")
    for index, extension in enumerate(data.get("engine_extensions", [])):
        if not isinstance(extension, dict) or not str(extension.get("file", "")).endswith(".py"):
            raise ManifestError(f"{where}: engine_extensions[{index}] needs a .py 'file'")


def _from_legacy(data: dict, entry: str) -> dict:
    """``secure-nodes.json`` (comfy-secure-nodes-v1) as a comfy-pack manifest."""
    if data.get("format") != "comfy-secure-nodes-v1":
        raise ManifestError(f"{LEGACY_FILE}: unknown format {data.get('format')!r}")
    root = os.path.dirname(os.path.normpath(entry)) if os.path.basename(os.path.normpath(entry)) == "v2" else entry
    identity = _pyproject(root)
    name = identity.get("name") or os.path.basename(root)
    if name.endswith("-HEAD"):
        name = name[:-len("-HEAD")]
    weights, seen = [], set()
    nodes = {}
    for node_id, node in data.get("nodes", {}).items():
        for weight in node.get("required_weights", []):
            item = {k: weight[k] for k in ("folder", "filename", "repo_id", "revision", "sha256") if weight.get(k)}
            key = json.dumps(item, sort_keys=True)
            if key not in seen:
                seen.add(key)
                weights.append(item)
        methods = node.get("methods", {})
        nodes[node_id] = {"display_name": node.get("schema", {}).get("attrs", {}).get("display_name") or node_id,
                          "module": node.get("module"), "class": node.get("class"), "schema": node["schema"],
                          "methods": [m for m in NODE_METHODS if methods.get(m)], "async": True}
    return {
        "format": FORMAT, "name": _normal_name(name), "display_name": identity.get("display_name") or name,
        "version": identity.get("version") or "0", "api": "2.1",
        "requirements": identity.get("requirements") or _requirements_txt(entry, root),
        "depends_on": [], "weights": weights, "permissions": [],
        "web_directory": data.get("web_directory"), "routes": data.get("routes", []),
        "static_directories": data.get("static_directories", {}), "nodes": nodes,
        "engine_extensions": _extensions(entry),
        "scheduler_providers": data.get("scheduler_providers", []),  # the overlay's declarative schedulers
    }


def registration_fields(pack: Any) -> dict:
    """Samplers, schedulers, model folders and node replacements the pack's extension declares."""
    from . import registration
    return registration.names(getattr(pack, "registrations", None) or {})


def _normal_name(name: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-")[:100] or "pack"


def _requirements_txt(*folders: str) -> list[str]:
    for folder in folders:
        path = os.path.join(folder, "requirements.txt")
        if os.path.isfile(path):
            with open(path, "rb") as file:
                raw = file.read()
            # pip reads a BOM'd UTF-16 or UTF-8 file (Windows editors write them); anything else as UTF-8
            encoding = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
            text = raw.decode(encoding, errors="replace")
            return [line.split("#", 1)[0].strip() for line in text.splitlines()
                    if line.split("#", 1)[0].strip() and not line.strip().startswith("-")]
    return []


def _pyproject(root: str) -> dict:
    path = os.path.join(root, "pyproject.toml")
    if not os.path.isfile(path):
        return {}
    import tomllib
    with open(path, "rb") as file:
        data = tomllib.load(file)
    project, comfy = data.get("project", {}), data.get("tool", {}).get("comfy", {})
    urls = project.get("urls", {})
    license_value = project.get("license")
    return {
        "name": project.get("name"), "version": project.get("version"), "description": project.get("description"),
        "license": license_value.get("text") if isinstance(license_value, dict) else license_value,
        "repository": urls.get("Repository") or urls.get("repository"),
        "requirements": [d.strip() for d in project.get("dependencies", []) if isinstance(d, str) and d.strip()],
        "display_name": comfy.get("DisplayName"), "publisher": comfy.get("PublisherId"),
        "depends_on": list(comfy.get("depends_on", [])), "weights": list(comfy.get("weights", [])),
        "requested_permissions": list(comfy.get("permissions", [])),
        "routes": list(comfy.get("routes", [])),
    }


def _extensions(entry: str) -> list[dict]:
    folder = os.path.join(entry, "engine")
    if not os.path.isdir(folder):
        folder = os.path.join(entry, "v2", "engine")
    found = []
    if os.path.isdir(folder):
        for name in sorted(os.listdir(folder)):
            if name.endswith(".py"):
                with open(os.path.join(folder, name), "rb") as file:
                    found.append({"file": name, "sha256": hashlib.sha256(file.read()).hexdigest()})
    return found


# --------------------------------------------------------------------------- #
# Writing (publish time; imports the pack)
# --------------------------------------------------------------------------- #
def generate(entry: str, previous: Optional[dict] = None) -> dict:
    """The manifest for a pack, from its code and its ``pyproject.toml``. Imports the pack, so it
    runs at publish time (``scripts/pack.sh manifest``), never when ComfyUI starts.

    ``permissions`` is what review granted, so it's kept from ``previous``, never derived."""
    from comfy_api._impl import _io
    from .isolation.server import LoadedPack
    from .internal import first_real_override

    entry = os.path.realpath(entry)
    root = os.path.dirname(entry) if os.path.basename(entry) == "v2" else entry
    identity = _pyproject(root)
    pack = LoadedPack(entry)
    nodes = {}
    for node_id, cls in sorted(pack.nodes.items()):
        if not (isinstance(cls, type) and issubclass(cls, _io.ComfyNode)):
            raise ManifestError(f"{node_id} isn't an io.ComfyNode; a V2 pack's nodes declare io.Schema")
        import inspect
        nodes[node_id] = {
            "display_name": pack.display_names.get(node_id, node_id),
            "module": cls.__module__.removeprefix(pack.module_name + ".") if cls.__module__ != pack.module_name else "",
            "class": cls.__qualname__,
            "schema": encode_schema(cls.define_schema()),
            "methods": [m for m in NODE_METHODS if first_real_override(cls, m) is not None],
            "async": inspect.iscoroutinefunction(cls.execute),
        }
    web = os.path.relpath(pack.web_directory, entry) if pack.web_directory else None
    name = identity.get("name") or os.path.basename(root)
    manifest = {
        "format": FORMAT, "name": _normal_name(name), "display_name": identity.get("display_name") or name,
        "version": identity.get("version") or "0", "publisher": identity.get("publisher"),
        "description": identity.get("description"), "license": identity.get("license"),
        "repository": identity.get("repository"), "api": "2.1",
        "weights": identity.get("weights", []),
        "requirements": identity.get("requirements") or _requirements_txt(entry, root),
        "depends_on": identity.get("depends_on", []),
        "permissions": list((previous or {}).get("permissions", [])),
        "requested_permissions": identity.get("requested_permissions", []),
        **registration_fields(pack),
        "engine_extensions": _extensions(entry), "web_directory": web,
        # endpoints, served at /ext/<namespace>/<path>: [[tool.comfy.routes]] in pyproject.toml (or kept
        # from the pack's manifest)
        "routes": identity.get("routes") or [{"method": m, "path": p, "module": mod, "handler": h}
                                             for (m, p), (mod, h) in sorted(pack.declared_routes().items())],
        "nodes": nodes,
    }
    manifest = {k: v for k, v in manifest.items() if v is not None}
    check(manifest)
    return manifest


# --------------------------------------------------------------------------- #
# io.Schema as JSON (from the secure overlay's pack manifest, with io.ModelFolder)
# --------------------------------------------------------------------------- #
def _io():
    from comfy_api._impl import _io_public as io
    return io


def _io_impl():
    from comfy_api._impl import _io
    return _io


def _io_class(io_type: str):
    io = _io()
    for value in vars(io).values():
        if isinstance(value, type) and getattr(value, "io_type", None) == io_type:
            return value
    return io.Custom(io_type)


def _encode_value(value: Any):
    io, io_impl = _io(), _io_impl()
    if isinstance(value, Enum):
        return {"$type": "enum", "enum": type(value).__name__, "name": value.name}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, io_impl.Input):
        return {"$type": "input", "value": _encode_input(value)}
    if isinstance(value, io_impl.Output):
        return {"$type": "output", "value": _encode_output(value)}
    if isinstance(value, io.ModelFolder):
        return {"$type": "model-folder", "folder": value.folder, "extensions": value.extensions}
    if isinstance(value, io.DynamicCombo.Option):
        return {"$type": "dynamic-option", "key": value.key, "inputs": [_encode_input(i) for i in value.inputs]}
    if isinstance(value, io.MatchType.Template):
        return {"$type": "match-template", "template_id": value.template_id,
                "allowed_types": [item.io_type for item in value.allowed_types]}
    if isinstance(value, io.Autogrow.TemplatePrefix):
        return {"$type": "autogrow-prefix", "input": _encode_input(value.input), "prefix": value.prefix,
                "min": value.min, "max": value.max}
    if isinstance(value, io.Autogrow.TemplateNames):
        return {"$type": "autogrow-names", "input": _encode_input(value.input), "names": value.names, "min": value.min}
    if isinstance(value, io.RemoteOptions):
        return {"$type": "remote-options", "attrs": _encode_value(vars(value))}
    if isinstance(value, io.PriceBadge):
        value.validate()
        return {"$type": "price-badge", "attrs": _encode_value(vars(value))}
    if isinstance(value, io.PriceBadgeDepends):
        value.validate()
        return {"$type": "price-badge-depends", "attrs": _encode_value(vars(value))}
    if isinstance(value, type) and getattr(value, "io_type", None):
        return {"$type": "io-class", "io_type": value.io_type}
    if isinstance(value, tuple):
        return {"$type": "tuple", "items": [_encode_value(item) for item in value]}
    if isinstance(value, list):
        return [_encode_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _encode_value(item) for key, item in value.items()}
    raise ManifestError(f"a manifest can't hold {type(value).__name__} in a schema")


def _decode_value(value: Any):
    io = _io()
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    tag = value.get("$type")
    if tag is None:
        return {key: _decode_value(item) for key, item in value.items()}
    if tag == "input":
        return _decode_input(value["value"])
    if tag == "output":
        return _decode_output(value["value"])
    if tag == "model-folder":
        return io.ModelFolder(value["folder"], value.get("extensions"))
    if tag == "dynamic-option":
        return io.DynamicCombo.Option(value["key"], [_decode_input(item) for item in value["inputs"]])
    if tag == "match-template":
        return io.MatchType.Template(value["template_id"], [_io_class(item) for item in value["allowed_types"]])
    if tag == "autogrow-prefix":
        return io.Autogrow.TemplatePrefix(_decode_input(value["input"]), value["prefix"], value["min"], value["max"])
    if tag == "autogrow-names":
        return io.Autogrow.TemplateNames(_decode_input(value["input"]), value["names"], value["min"])
    if tag == "remote-options":
        remote = object.__new__(io.RemoteOptions)
        remote.__dict__.update(_decode_value(value["attrs"]))
        return remote
    if tag in ("price-badge", "price-badge-depends"):
        cls = io.PriceBadge if tag == "price-badge" else io.PriceBadgeDepends
        result = cls(**_decode_value(value["attrs"]))
        result.validate()
        return result
    if tag == "io-class":
        return _io_class(value["io_type"])
    if tag == "enum":
        enum_type = getattr(io, value["enum"], None)
        if not isinstance(enum_type, type) or not issubclass(enum_type, Enum):
            raise ManifestError(f"unknown manifest enum {value['enum']!r}")
        return enum_type[value["name"]]
    if tag == "tuple":
        return tuple(_decode_value(item) for item in value["items"])
    raise ManifestError(f"unknown manifest value tag {tag!r}")


def _input_kind(value: Any) -> str:
    io = _io()
    for kind, cls in (("dynamic-combo", io.DynamicCombo.Input), ("autogrow", io.Autogrow.Input),
                      ("match-type", io.MatchType.Input), ("multi-type", io.MultiType.Input)):
        if isinstance(value, cls):
            return kind
    return "standard"


def _encode_input(value: Any) -> dict:
    return {"kind": _input_kind(value), "io_type": value.io_type, "attrs": _encode_value(vars(value))}


def _decode_input(value: dict) -> Any:
    io = _io()
    classes = {"dynamic-combo": io.DynamicCombo.Input, "autogrow": io.Autogrow.Input,
               "match-type": io.MatchType.Input, "multi-type": io.MultiType.Input}
    kind = value["kind"]
    if kind == "standard":
        cls = _io_class(value["io_type"]).Input
    elif kind in classes:
        cls = classes[kind]
    else:
        raise ManifestError(f"unknown manifest input kind {kind!r}")
    result = object.__new__(cls)
    result.__dict__.update(_decode_value(value["attrs"]))
    return result


def _encode_output(value: Any) -> dict:
    io = _io()
    kind = "match-type" if isinstance(value, io.MatchType.Output) else "standard"
    return {"kind": kind, "io_type": value.io_type, "attrs": _encode_value(vars(value))}


def _decode_output(value: dict) -> Any:
    io = _io()
    if value["kind"] == "match-type":
        cls = io.MatchType.Output
    elif value["kind"] == "standard":
        cls = _io_class(value["io_type"]).Output
    else:
        raise ManifestError(f"unknown manifest output kind {value['kind']!r}")
    result = object.__new__(cls)
    result.__dict__.update(_decode_value(value["attrs"]))
    return result


def encode_schema(schema: Any) -> dict:
    return {
        "attrs": {k: _encode_value(v) for k, v in vars(schema).items() if k not in {"inputs", "outputs", "hidden"}},
        "inputs": [_encode_input(value) for value in schema.inputs],
        "outputs": [_encode_output(value) for value in schema.outputs],
        "hidden": [value.name for value in schema.hidden],
    }


def decode_schema(value: dict) -> Any:
    io = _io()
    attrs = _decode_value(value["attrs"])
    attrs["inputs"] = [_decode_input(item) for item in value["inputs"]]
    attrs["outputs"] = [_decode_output(item) for item in value["outputs"]]
    attrs["hidden"] = [io.Hidden[name] for name in value["hidden"]]
    return io.Schema(**attrs)
