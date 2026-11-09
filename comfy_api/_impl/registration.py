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
so a V2 pack's samplers and schedulers are in ComfyUI's lists at startup without importing it.

What a scheduler gets as ``model_sampling`` is plain data and engine functions:
``{"sigma_min", "sigma_max", "sigmas", "sigma", "timestep", "percent_to_sigma"}``; ``sigmas`` is the
model's sigma table, and ``sigma(t)``, ``timestep(s)``, ``percent_to_sigma(p)`` call the engine.
"""
from __future__ import annotations

import logging
import os
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
    if owner is not None and owner != pack:  # the secure overlay's rule: no cross-pack collisions
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
