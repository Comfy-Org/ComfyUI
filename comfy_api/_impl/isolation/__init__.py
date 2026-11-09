"""Pack isolation: each custom-node pack in its own process, on its own Python environment, as a
browser runs each site in its own page process. ``COMFY_PACK_ISOLATION=true``. See ``README.md``.

Two reasons: packs' dependencies can't conflict with each other or with ComfyUI's, and a pack that
crashes can't take ComfyUI down.

ComfyUI is the browser process: it owns the engine, the models, the GPU and the UI server, and it
imports no pack. Each pack runs in a pack process that ComfyUI starts when a prompt uses one of its
nodes or the frontend calls one of its routes, stops when it has been idle for
``COMFY_PACK_IDLE_MINUTES``, and that exits when ComfyUI does. A pack that crashes fails its own
nodes; ComfyUI stays up.

Modules: ``envs`` (each pack's own Python environment, so packs' dependencies can't conflict),
``packs`` (finding packs, their descriptions, starting them on demand), ``launcher``
(how a pack process is started: locally, or by ``COMFY_PACK_LAUNCHER``), ``server`` (the pack
process: ``scripts/pack.sh``), ``proxy`` (the node classes ComfyUI registers for a pack's nodes),
``host`` (ComfyUI's side), ``worker`` (the pack's side), ``boundary`` (what crosses) and
``channel`` (the pipe).

The boundary follows the same rules as any out-of-process dispatch: engine objects (MODEL, CLIP,
VAE...) stay in ComfyUI and cross as handles the pack passes back to ``eng`` calls; tensors and
plain data cross as copies; anything else is refused. A function given to a ``model.set_*`` call
stays in the pack process and is called from sampling.

This isolates failures and resources, not trust: ComfyUI and its pack processes trust each other.
It is not a security sandbox.
"""
from __future__ import annotations

import os
from typing import Any, Optional

ENV = "COMFY_PACK_ISOLATION"
#: Each setting's earlier name, still read when the new one isn't set.
LEGACY_NAMES = {
    "COMFY_PACK_ISOLATION": "OOP_CUSTOM_NODES",
    "COMFY_PACKS_DIR": "OOP_PACKS_DIR",
    "COMFY_PACK_IDLE_MINUTES": "OOP_PACK_TTL_MINUTES",
    "COMFY_PACK_LAUNCHER": "OOP_PACK_LAUNCHER",
}


def setting(name: str, default: Optional[str] = None) -> Optional[str]:
    """A pack-isolation setting from the environment, by its name or its earlier name."""
    value = os.environ.get(name)
    if value is None and name in LEGACY_NAMES:
        value = os.environ.get(LEGACY_NAMES[name])
    return default if value is None else value


def without_settings(env: dict) -> dict:
    """``env`` for a process ComfyUI starts: pack isolation is ComfyUI's, not the pack's."""
    env = dict(env)
    env.pop(ENV, None)
    env.pop(LEGACY_NAMES[ENV], None)
    return env


def enabled() -> bool:
    return (setting(ENV) or "").strip().lower() in ("1", "true", "yes", "on")


from .boundary import Closure, from_worker, to_worker  # noqa: E402
from .channel import Channel, RemoteError, WorkerError  # noqa: E402
from .host import OutOfProcessBackend, Worker  # noqa: E402


def register(providers: Any) -> None:
    """Entry point for ``_runtime.load_execution_provider``."""
    providers.execution_backend = OutOfProcessBackend()


__all__ = ["ENV", "LEGACY_NAMES", "setting", "without_settings", "enabled", "register", "OutOfProcessBackend",
           "Worker", "Channel", "Closure", "RemoteError", "WorkerError", "to_worker", "from_worker"]
