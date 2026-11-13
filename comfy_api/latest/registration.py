"""What a pack registers besides nodes: samplers, schedulers, model folders and node replacements.

A pack declares them on its ``ComfyExtension``::

    class Extension(ComfyExtension):
        async def get_schedulers(self):        # name -> fn(model_sampling: dict, steps: int) -> sigmas
            return {"beta57": beta57}
        async def get_samplers(self):          # name -> a sampling loop, as eng.sampler.custom takes
            return {"pack_euler": euler}
        async def get_model_folders(self):     # name -> file extensions (None: ComfyUI's model files)
            return {"onnx": ["onnx"]}
        async def get_node_replacements(self): # io.NodeReplace, for workflows that use old node ids
            return [io.NodeReplace(new_node_id="NewNode", old_node_id="OldNode")]

``collect`` reads them from an extension, ``names`` is the plain-data form a manifest keeps, and
``install`` registers them with ComfyUI. ``install`` takes a lookup for each pack function: the
function itself in ComfyUI's process, or one that runs it in the pack's process (pack isolation),
so a pack's samplers and schedulers are in ComfyUI's lists at startup without importing it.

What a scheduler gets as ``model_sampling`` is plain data and engine functions:
``{"sigma_min", "sigma_max", "sigmas", "sigma", "timestep", "percent_to_sigma"}``; ``sigmas`` is the
model's sigma table, and ``sigma(t)``, ``timestep(s)``, ``percent_to_sigma(p)`` call the engine.
"""
from __future__ import annotations

import logging
import os
import importlib
import inspect
import re
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)
KINDS = {"schedulers": "get_schedulers", "samplers": "get_samplers", "model_folders": "get_model_folders",
         "node_replacements": "get_node_replacements"}
SCHEDULER_OWNERS: dict[str, str] = {}


async def collect(extension: Any) -> dict:
    """The registrations an extension declares, checked."""
    found: dict[str, Any] = {}
    for kind, method in KINDS.items():
        getter = getattr(extension, method, None)
        value = await getter() if getter is not None else None
        if value is None:
            value = [] if kind == "node_replacements" else {}
        if kind in ("schedulers", "samplers"):
            if not isinstance(value, dict) or not all(isinstance(k, str) and k and callable(v) for k, v in value.items()):
                raise TypeError(f"{method}() returns {{name: function}}")
        elif kind == "model_folders":
            if not isinstance(value, dict) or not all(
                    isinstance(k, str) and k and (v is None or (isinstance(v, (list, tuple)) and all(isinstance(e, str) for e in v)))
                    for k, v in value.items()):
                raise TypeError("get_model_folders() returns {folder: [extensions] or None}")
        else:
            from ._io import NodeReplace
            if not isinstance(value, (list, tuple)) or not all(isinstance(v, NodeReplace) for v in value):
                raise TypeError("get_node_replacements() returns a list of io.NodeReplace")
        found[kind] = value
    return found


def names(registrations: dict) -> dict:
    """The plain-data form, as a manifest keeps it."""
    return {
        "schedulers": sorted(registrations.get("schedulers", {})),
        "samplers": sorted(registrations.get("samplers", {})),
        "model_folders": {k: (list(v) if v is not None else None) for k, v in registrations.get("model_folders", {}).items()},
        "node_replacements": [r.as_dict() if hasattr(r, "as_dict") else dict(r) for r in registrations.get("node_replacements", [])],
    }


def from_extension(registrations: dict) -> tuple[dict, Callable[[str, str], Callable]]:
    """``install``'s arguments for registrations collected in this process."""
    functions = {"scheduler": registrations.get("schedulers", {}), "sampler": registrations.get("samplers", {})}
    return names(registrations), lambda kind, name: functions[kind][name]


def install(declared: dict, function: Callable[[str, str], Callable], pack: str) -> list[str]:
    """Register ``declared`` (``names`` form) with ComfyUI. ``function(kind, name)`` gives the pack
    function for a sampler or scheduler when it's used. Returns what was registered."""
    done = []
    for folder, extensions in (declared.get("model_folders") or {}).items():
        _model_folder(folder, extensions)
        done.append(f"model folder {folder}")
    for name in declared.get("schedulers") or ():
        if _scheduler(name, lambda name=name: function("scheduler", name), pack):
            done.append(f"scheduler {name}")
    for name in declared.get("samplers") or ():
        if _sampler(name, lambda name=name: function("sampler", name), pack):
            done.append(f"sampler {name}")
    for replacement in declared.get("node_replacements") or ():
        _node_replacement(replacement)
        done.append(f"replacement {replacement['old_node_id']} -> {replacement['new_node_id']}")
    if done:
        logger.info("%s registered: %s", pack, ", ".join(done))
    return done


def _model_folder(folder: str, extensions: Optional[list]) -> None:
    import folder_paths

    if not isinstance(folder, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", folder) or ".." in folder:
        raise ValueError(f"model folder {folder!r} must be a plain name (letters, digits, '_', '-', '.')")
    exts = {("." + e.lower().lstrip(".")) for e in extensions} if extensions else set(folder_paths.supported_pt_extensions)
    if folder in folder_paths.folder_names_and_paths:
        paths, known = folder_paths.folder_names_and_paths[folder]
        folder_paths.folder_names_and_paths[folder] = (paths, set(known) | exts)
        return
    path = os.path.join(folder_paths.models_dir, folder)
    folder_paths.folder_names_and_paths[folder] = ([path], exts)


def _scheduler(name: str, function: Callable[[], Callable], pack: str) -> bool:
    import comfy.samplers
    from ._engine import pack_function

    owner = SCHEDULER_OWNERS.get(name)
    if owner is not None and owner != pack:  # a scheduler name belongs to the pack that registered it first
        logger.warning("%s: scheduler %r belongs to %s; not registered", pack, name, owner)
        return False
    if owner is None and name in comfy.samplers.SCHEDULER_HANDLERS and not getattr(comfy.samplers.SCHEDULER_HANDLERS[name].handler, "_comfy_pack", None):
        logger.warning("%s: scheduler %r is ComfyUI's own; not registered", pack, name)
        return False

    def handler(model_sampling: Any, steps: int) -> Any:
        from ._engine import model_sampling_view
        result = pack_function(function())(model_sampling_view(model_sampling), int(steps))
        return validate_scheduler_result(result, int(steps))
    handler._comfy_pack = pack
    SCHEDULER_OWNERS[name] = pack
    comfy.samplers.SCHEDULER_HANDLERS[name] = comfy.samplers.SchedulerHandler(handler)
    if name not in comfy.samplers.SCHEDULER_NAMES:
        comfy.samplers.SCHEDULER_NAMES.append(name)  # KSampler.SCHEDULERS is the same list
    return True


def validate_scheduler_result(values, steps: int):
    import torch

    try:
        result = torch.as_tensor(values, dtype=torch.float32, device="cpu")
    except Exception as error:
        raise TypeError("scheduler provider returned a non-numeric result") from error
    if result.ndim != 1 or not 2 <= len(result) <= min(65_536, steps + 1):
        raise ValueError("scheduler provider result must be a bounded 1-D sigma sequence")
    if not bool(torch.isfinite(result).all()) or bool((result < 0).any()):
        raise ValueError("scheduler provider returned invalid sigma values")
    if bool((result[1:] > result[:-1]).any()):
        raise ValueError("scheduler provider result must be non-increasing")
    if float(result[-1]) != 0.0:
        raise ValueError("scheduler provider result must terminate at zero")
    return result


def _sampler(name: str, function: Callable[[], Callable], pack: str) -> bool:
    import comfy.k_diffusion.sampling as k_diffusion_sampling
    import comfy.samplers
    from ._engine import pack_function

    existing = getattr(k_diffusion_sampling, f"sample_{name}", None)
    if name in comfy.samplers.SAMPLER_NAMES and not getattr(existing, "_comfy_pack", None):
        logger.warning("%s: sampler %r is ComfyUI's own; not registered", pack, name)
        return False

    def sample(model, x, sigmas, extra_args=None, callback=None, disable=None, **options):
        from ._engine import check_sigmas
        check_sigmas(sigmas)
        run = pack_function(function(), engine_args=(0, "model", "callback"))
        return run(model, x, sigmas, extra_args or {}, callback=callback, disable=disable, **options)
    sample._comfy_pack = pack
    sample.__name__ = f"sample_{name}"
    setattr(k_diffusion_sampling, f"sample_{name}", sample)  # comfy.samplers.ksampler(name) finds it here
    for listed in (comfy.samplers.KSAMPLER_NAMES, comfy.samplers.SAMPLER_NAMES):
        if name not in listed:
            listed.append(name)  # KSampler.SAMPLERS is SAMPLER_NAMES
    return True


def _node_replacement(data: dict) -> None:
    from ._io import NodeReplace
    try:
        from server import PromptServer
        manager = PromptServer.instance.node_replace_manager
    except Exception:  # no server (tests, describe): nothing to register with
        return
    manager.register(NodeReplace(**data))


# --------------------------------------------------------------------------- #
# Declared routes: [[tool.comfy.routes]] in the pack's pyproject.toml, served at /ext/<namespace>/<path>
# --------------------------------------------------------------------------- #
ROUTE_MOUNT = "/ext"
QUERY_MAX = 64
REQUEST_MAX_BYTES = 1 << 20
ROUTE_BODY_MAX_BYTES = 1 << 20
ROUTE_CONTENT_TYPES = frozenset({"application/json", "text/plain"})


def declarations(module_dir: str) -> dict:
    """The pack's ``pyproject.toml`` (``project`` and ``tool.comfy``), from the pack's directory or its
    parent when ComfyUI loaded a ``v2/`` entrypoint."""
    import tomllib

    for folder in (module_dir, os.path.dirname(module_dir)):
        path = os.path.join(folder, "pyproject.toml")
        if os.path.isfile(path):
            with open(path, "rb") as file:
                data = tomllib.load(file)
            return {"project": data.get("project", {}), "comfy": data.get("tool", {}).get("comfy", {})}
    return {"project": {}, "comfy": {}}


def pack_name(module_dir: str, declared: dict) -> str:
    """The pack's permanent name: ``[project] name``, else its folder's."""
    folder = os.path.dirname(module_dir) if os.path.basename(module_dir) == "v2" else module_dir
    name = declared["project"].get("name") or os.path.basename(os.path.normpath(folder))
    return re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-")[:100] or "pack"


def pack_namespace(name: str) -> str:
    """Where a pack's routes live, ``/ext/<namespace>/``: a readable slug plus 8 hex characters of
    SHA-256 over the pack's name, so it's the same wherever the pack is installed and can't collide."""
    import hashlib

    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:32].strip("-") or "pack"
    return f"{slug}-{hashlib.sha256(name.encode('utf-8')).hexdigest()[:8]}"


def install_routes(module_dir: str, package: str, prompt_server: Any) -> list[str]:
    """Serve the pack's declared routes on ComfyUI's server; a handler runs in ComfyUI's process."""
    declared = declarations(module_dir)
    routes = declared["comfy"].get("routes") or []
    if not routes:
        return []
    namespace = pack_namespace(pack_name(module_dir, declared))
    served = []
    for route in routes:
        method, path = str(route["method"]).upper(), str(route["path"])
        if not path.startswith("/") or "{" in path:
            raise ValueError(f"route path {path!r} must be literal and start with '/'")
        handler = getattr(importlib.import_module(f"{package}.{route['module']}"), route["handler"])
        url = f"{ROUTE_MOUNT}/{namespace}{path}"
        prompt_server.routes.route(method, url)(_serve(handler, method.lower(), path))
        served.append(f"{method} {url}")
    return served


def route_response(result: Any) -> dict:
    """A route handler's answer: ``{"status", "body" (text, at most 1 MB), "content_type"}``."""
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


def _serve(handler: Callable, method: str, path: str):
    async def serve(request: Any) -> Any:
        from aiohttp import web

        if request.content_length is not None and request.content_length > REQUEST_MAX_BYTES:
            return web.json_response({"ok": False, "error": "request body is too large"}, status=413)
        message = {"method": method, "path": path, "query": dict(list(request.query.items())[:QUERY_MAX]),
                   "body": await request.text()}
        try:
            result = handler(message)
            if inspect.isawaitable(result):
                result = await result
            reply = route_response(result)
        except Exception as error:
            logger.warning("route %s %s failed: %s", method, path, error)
            return web.json_response({"ok": False, "error": "route handler failed"}, status=500)
        return web.Response(status=reply["status"], text=reply["body"], content_type=reply["content_type"])
    return serve
