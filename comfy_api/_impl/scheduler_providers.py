# Lifted unchanged from the secure overlay (comfy_secure_nodes/schedulerproviders.py), its tested
# scheduler providers. Pack isolation adds ``register_providers`` (below): the same checks and
# registration, with each call in a short-lived process on the pack's own Python instead of a guest.
"""Declarative scheduler names backed by scalar-only pack sandboxes.

These providers exist for packs whose intended behavior is a scheduler name,
not a node.  The host owns registration and model-schedule projection.  The
pack owns the actual sigma algorithm, which runs in a fresh confined guest and
receives only finite scalar lists, an integer step count and sealed JSON
configuration.
"""
from __future__ import annotations

import asyncio
import copy
import json
import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MAX_PROVIDERS_PER_PACK = 32
MAX_CONFIG_BYTES = 64 * 1024
MAX_CONFIG_DEPTH = 16
MAX_SIGMAS = 65_536
MAX_STEPS = 10_000
_NAME = re.compile(r"^[a-z][a-z0-9_+.-]{0,63}$")
_PROJECTIONS = frozenset({"none", "simple", "model_sigmas"})


@dataclass(frozen=True)
class SchedulerProvider:
    name: str
    module_name: str
    module_file: Path
    function: str
    projection: str
    projection_steps: int | None
    min_steps: int
    max_steps: int
    config: dict[str, Any]


def _plain_json(value: Any, depth: int = 0) -> Any:
    if depth > MAX_CONFIG_DEPTH:
        raise ValueError("scheduler provider config is nested too deeply")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("scheduler provider config numbers must be finite")
        return value
    if isinstance(value, list):
        return [_plain_json(item, depth + 1) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str) or key in {
                "__proto__", "constructor", "prototype",
            }:
                raise ValueError("scheduler provider config has an unsafe key")
            result[key] = _plain_json(item, depth + 1)
        return result
    raise ValueError("scheduler provider config must contain only plain JSON")


def decode(
    declarations: Any,
    *,
    v2_dir: Path,
    mount_name: str,
    module_source,
) -> tuple[SchedulerProvider, ...]:
    """Validate manifest declarations without importing pack code."""
    if not isinstance(declarations, list):
        raise ValueError("scheduler_providers must be a list")
    if len(declarations) > MAX_PROVIDERS_PER_PACK:
        raise ValueError(
            f"scheduler_providers exceeds {MAX_PROVIDERS_PER_PACK} entries")
    result = []
    names = set()
    required = {
        "name", "module", "function", "projection", "min_steps",
        "max_steps", "config",
    }
    optional = {"projection_steps"}
    for index, item in enumerate(declarations):
        if not isinstance(item, dict):
            raise ValueError(f"scheduler_providers[{index}] must be an object")
        missing = required - set(item)
        unknown = set(item) - required - optional
        if missing or unknown:
            raise ValueError(
                f"scheduler_providers[{index}] has missing fields "
                f"{sorted(missing)} and unknown fields {sorted(unknown)}")
        name = item["name"]
        if not isinstance(name, str) or _NAME.fullmatch(name) is None:
            raise ValueError(f"scheduler_providers[{index}] has an unsafe name")
        if name in names:
            raise ValueError(f"duplicate scheduler provider name {name!r}")
        names.add(name)
        relative_module = item["module"]
        if not isinstance(relative_module, str):
            raise ValueError("scheduler provider module must be a string")
        source = module_source(v2_dir, relative_module)
        function = item["function"]
        if not isinstance(function, str) or not function.isidentifier():
            raise ValueError("scheduler provider function must be an identifier")
        projection = item["projection"]
        if projection not in _PROJECTIONS:
            raise ValueError(
                f"scheduler provider projection must be one of "
                f"{sorted(_PROJECTIONS)}")
        projection_steps = item.get("projection_steps")
        if projection == "simple":
            if (
                isinstance(projection_steps, bool)
                or not isinstance(projection_steps, int)
                or not 1 <= projection_steps <= MAX_STEPS
            ):
                raise ValueError(
                    "simple scheduler projection requires bounded "
                    "projection_steps")
        elif projection_steps is not None:
            raise ValueError(
                "projection_steps is valid only for the simple projection")
        min_steps = item["min_steps"]
        max_steps = item["max_steps"]
        if (
            isinstance(min_steps, bool)
            or isinstance(max_steps, bool)
            or not isinstance(min_steps, int)
            or not isinstance(max_steps, int)
            or not 1 <= min_steps <= max_steps <= MAX_STEPS
        ):
            raise ValueError(
                f"scheduler provider steps must satisfy 1 <= min <= max <= "
                f"{MAX_STEPS}")
        config = _plain_json(item["config"])
        if len(json.dumps(config, separators=(",", ":")).encode()) > MAX_CONFIG_BYTES:
            raise ValueError(
                f"scheduler provider config exceeds {MAX_CONFIG_BYTES} bytes")
        result.append(SchedulerProvider(
            name=name,
            module_name=f"{mount_name}.{relative_module}",
            module_file=source,
            function=function,
            projection=projection,
            projection_steps=projection_steps,
            min_steps=min_steps,
            max_steps=max_steps,
            config=config,
        ))
    return tuple(result)


_OWNERS: dict[str, str] = {}
_HANDLERS: dict[str, Any] = {}
_LOCK = threading.RLock()


def _samplers():
    import comfy.samplers

    return comfy.samplers


def preflight(packs) -> None:
    """Refuse built-in, legacy, and cross-pack name collisions atomically."""
    samplers = _samplers()
    proposed: dict[str, str] = {}
    with _LOCK:
        for pack in packs:
            for provider in pack.scheduler_providers:
                other = proposed.get(provider.name)
                if other is not None:
                    raise RuntimeError(
                        f"secure packs define duplicate scheduler "
                        f"{provider.name!r}: {other!r} and {pack.name!r}")
                proposed[provider.name] = pack.name
                owner = _OWNERS.get(provider.name)
                if owner is not None and owner != pack.name:
                    raise RuntimeError(
                        f"scheduler {provider.name!r} is owned by {owner!r}")
                if owner is None and provider.name in samplers.SCHEDULER_HANDLERS:
                    raise RuntimeError(
                        f"scheduler {provider.name!r} is already registered")


def _projection(provider: SchedulerProvider, model_sampling) -> list | None:
    import torch

    samplers = _samplers()
    if provider.projection == "none":
        return None
    if provider.projection == "simple":
        value = samplers.simple_scheduler(
            model_sampling, provider.projection_steps)
    else:
        value = getattr(model_sampling, "sigmas", None)
        if value is None:
            raise ValueError(
                "model-sigmas scheduler projection is unavailable")
    tensor = torch.as_tensor(value).detach().to(
        device="cpu", dtype=torch.float64)
    if tensor.ndim != 1 or not 2 <= len(tensor) <= MAX_SIGMAS:
        raise ValueError("scheduler projection must be a bounded 1-D sequence")
    if not bool(torch.isfinite(tensor).all()) or bool((tensor < 0).any()):
        raise ValueError("scheduler projection must contain finite sigmas")
    return tensor.tolist()


def _validate_result(values, steps: int):
    import torch

    try:
        result = torch.as_tensor(values, dtype=torch.float32, device="cpu")
    except Exception as error:
        raise TypeError("scheduler provider returned a non-numeric result") from error
    if result.ndim != 1 or not 2 <= len(result) <= min(MAX_SIGMAS, steps + 1):
        raise ValueError(
            "scheduler provider result must be a bounded 1-D sigma sequence")
    if not bool(torch.isfinite(result).all()) or bool((result < 0).any()):
        raise ValueError("scheduler provider returned invalid sigma values")
    if len(result) > 1 and bool((result[1:] > result[:-1]).any()):
        raise ValueError("scheduler provider result must be non-increasing")
    if float(result[-1]) != 0.0:
        raise ValueError("scheduler provider result must terminate at zero")
    return result


async def _invoke_async(pack, provider: SchedulerProvider, projection, steps):
    from .transport.host import GuestSession

    session = GuestSession(
        f"scheduler:{pack.name}:{provider.name}",
        runtime=pack.runtime,
    )
    await session.start()
    try:
        return await session.invoke_scheduler_provider(
            module_name=provider.module_name,
            module_file=str(provider.module_file),
            function_name=provider.function,
            projection=projection,
            steps=steps,
            config=copy.deepcopy(provider.config),
        )
    finally:
        await session.kill()


def _invoke(pack, provider: SchedulerProvider, model_sampling, steps: int):
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError("scheduler steps must be an integer")
    if not provider.min_steps <= steps <= provider.max_steps:
        raise ValueError(
            f"scheduler {provider.name!r} steps must be in "
            f"[{provider.min_steps}, {provider.max_steps}]")
    projected = _projection(provider, model_sampling)
    box = []
    errors = []

    def runner():
        try:
            box.append(asyncio.run(
                _invoke_async(pack, provider, projected, steps)))
        except BaseException as error:  # re-raised on the sampling thread
            errors.append(error)

    thread = threading.Thread(
        target=runner,
        name=f"secure-scheduler-{provider.name}",
        daemon=True,
    )
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    if len(box) != 1:
        raise RuntimeError("scheduler provider produced no result")
    return _validate_result(box[0], steps)


def register(pack) -> None:
    if not pack.scheduler_providers:
        return
    preflight((pack,))
    samplers = _samplers()
    replacements = {
        provider.name: samplers.SchedulerHandler(
            lambda model_sampling, steps, p=provider: _invoke(
                pack, p, model_sampling, steps),
            use_ms=True,
        )
        for provider in pack.scheduler_providers
    }
    with _LOCK:
        old_names = {
            name for name, owner in _OWNERS.items() if owner == pack.name
        }
        new_names = set(replacements)
        for name in old_names - new_names:
            _remove_name(samplers, name)
        for name, handler in replacements.items():
            samplers.SCHEDULER_HANDLERS[name] = handler
            if name not in samplers.SCHEDULER_NAMES:
                samplers.SCHEDULER_NAMES.append(name)
            if name not in samplers.KSampler.SCHEDULERS:
                samplers.KSampler.SCHEDULERS.append(name)
            _OWNERS[name] = pack.name
            _HANDLERS[name] = handler


def _remove_name(samplers, name: str) -> None:
    registered = _HANDLERS.get(name)
    if registered is not None and samplers.SCHEDULER_HANDLERS.get(name) is registered:
        samplers.SCHEDULER_HANDLERS.pop(name, None)
    while name in samplers.SCHEDULER_NAMES:
        samplers.SCHEDULER_NAMES.remove(name)
    if samplers.KSampler.SCHEDULERS is not samplers.SCHEDULER_NAMES:
        while name in samplers.KSampler.SCHEDULERS:
            samplers.KSampler.SCHEDULERS.remove(name)
    _OWNERS.pop(name, None)
    _HANDLERS.pop(name, None)


def unregister(pack_name: str) -> None:
    with _LOCK:
        names = [
            name for name, owner in _OWNERS.items() if owner == pack_name
        ]
        if not names:
            return
        samplers = _samplers()
        for name in names:
            _remove_name(samplers, name)


# --------------------------------------------------------------------------- #
# Pack isolation: providers called in a short-lived process (as the overlay calls each in a fresh guest)
# --------------------------------------------------------------------------- #
_RUNNER = r"""
import importlib.util, json, sys
request = json.loads(sys.stdin.read())
spec = importlib.util.spec_from_file_location("comfy_scheduler_provider", request["file"])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result = getattr(module, request["function"])(request["projection"], request["steps"], request["config"])
if hasattr(result, "tolist"):
    result = result.tolist()
sys.stdout.write(json.dumps([float(x) for x in result]))
"""


def run_in_process(python: str, provider: SchedulerProvider, projection, steps: int, timeout: float = 60.0) -> list:
    """Call one provider in a fresh process that loads only its module: no pack import."""
    import subprocess

    done = subprocess.run([python, "-c", _RUNNER], input=json.dumps({
        "file": str(provider.module_file), "function": provider.function, "projection": projection,
        "steps": steps, "config": provider.config}), capture_output=True, text=True, timeout=timeout)
    if done.returncode != 0:
        raise RuntimeError(f"scheduler {provider.name!r} failed: {done.stderr[-2000:]}")
    return json.loads(done.stdout)


def register_providers(pack_name: str, providers, call) -> list:
    """Register ``providers`` for ``pack_name``: ``call(provider, projection, steps) -> sigmas``.
    The overlay's checks apply: collisions (``preflight``), step bounds, projection, result."""
    if not providers:
        return []

    class _Pack:
        name = pack_name
        scheduler_providers = tuple(providers)
    preflight((_Pack,))
    samplers = _samplers()

    def handler_for(provider):
        def handler(model_sampling, steps):
            if isinstance(steps, bool) or not isinstance(steps, int):
                raise TypeError("scheduler steps must be an integer")
            if not provider.min_steps <= steps <= provider.max_steps:
                raise ValueError(f"scheduler {provider.name!r} steps must be in "
                                 f"[{provider.min_steps}, {provider.max_steps}]")
            return _validate_result(call(provider, _projection(provider, model_sampling), steps), steps)
        return samplers.SchedulerHandler(handler, use_ms=True)

    with _LOCK:
        for provider in providers:
            handler = handler_for(provider)
            samplers.SCHEDULER_HANDLERS[provider.name] = handler
            if provider.name not in samplers.SCHEDULER_NAMES:
                samplers.SCHEDULER_NAMES.append(provider.name)
            if provider.name not in samplers.KSampler.SCHEDULERS:
                samplers.KSampler.SCHEDULERS.append(provider.name)
            _OWNERS[provider.name] = pack_name
            _HANDLERS[provider.name] = handler
    return [provider.name for provider in providers]


def module_source(v2_dir: Path, relative_module: str) -> Path:
    """The file of a provider's module inside the pack, refusing anything that leaves it."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*", relative_module or ""):
        raise ValueError("scheduler provider module must be a dotted module name")
    path = (Path(v2_dir) / (relative_module.replace(".", "/") + ".py")).resolve()
    if Path(v2_dir).resolve() not in path.parents or not path.is_file():
        raise ValueError(f"scheduler provider module {relative_module!r} isn't in the pack")
    return path
