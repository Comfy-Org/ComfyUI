"""
Custom-node SDK: resources as abstractions (refs), a brokered ``ctx``, and a
provider registry that lets a separable overlay swap the in-process default
implementation for an isolated one — without the node author changing a line.

Design (see docs/secure_custom_nodes_WIP.md):

* A node's heavyweight values cross ``execute()`` as opaque **refs**
  (``ImageRef``, ``ModelRef``, ``AssetRef`` ...), never raw buffers/paths.
* Side effects go through **ctx** domains (``ctx.assets``, ``ctx.progress`` ...)
  rather than ``folder_paths`` / ``PromptServer`` / ambient globals.
* Everything here ships a **default in-process implementation** so open-source
  ComfyUI behaves exactly as today (a ref just wraps the real object; ctx is a
  thin passthrough). Zero behavior change, better authoring API.
* The **overlay** (proprietary, cloud-only, loaded by ``COMFY_OVERLAY_MODULE``)
  calls ``providers.register_*`` at import to replace those defaults with the
  isolated engine: out-of-process guests, shm/CUDA-IPC refs, an enforcing
  broker. Uninstall the overlay -> pure OSS.

Nothing isolation-specific lives in this file. This is the *seam*, not the
engine.
"""
from __future__ import annotations

import contextvars
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Optional,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:  # keep this module import-safe / torch-free at import time
    import torch

    from ._io import NodeOutput

logger = logging.getLogger(__name__)

# Env var an operator points at a directory/module implementing ``register``.
OVERLAY_ENV = "COMFY_OVERLAY_MODULE"


# --------------------------------------------------------------------------- #
# Refs — opaque, typed handles. In OSS a ref is resolved by an identity table
# (zero-copy, zero-overhead: it holds the real object). The overlay swaps the
# resolver for shm/CUDA-IPC across a process boundary. The ref token itself
# carries nothing exploitable; the host table is authoritative.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Ref:
    """Base opaque resource handle. ``kind`` is the io_type (IMAGE, LATENT...)."""

    kind: str
    id: str

    def __repr__(self) -> str:  # keep ids short in logs, never leak contents
        return f"<{type(self).__name__} {self.id[:8]}>"


class _TypedRef(Ref):
    KIND: str = "ANY"

    @classmethod
    def _wrap(cls, ref: Ref) -> "Ref":
        return cls(kind=cls.KIND, id=ref.id)


class TensorRef(_TypedRef):
    KIND = "TENSOR"

    async def tensor(self) -> "torch.Tensor":
        return await current_runtime().refs.resolve(self)

    @classmethod
    async def from_tensor(cls, t: "torch.Tensor") -> "TensorRef":
        return cls._wrap(await current_runtime().refs.create(cls.KIND, t))  # type: ignore[return-value]


class ImageRef(TensorRef):
    KIND = "IMAGE"


class MaskRef(TensorRef):
    KIND = "MASK"


class LatentRef(_TypedRef):
    KIND = "LATENT"

    async def value(self) -> dict:
        return await current_runtime().refs.resolve(self)

    @classmethod
    async def from_value(cls, v: dict) -> "LatentRef":
        return cls._wrap(await current_runtime().refs.create(cls.KIND, v))  # type: ignore[return-value]


class CondRef(_TypedRef):
    KIND = "CONDITIONING"


class ModelRef(_TypedRef):
    KIND = "MODEL"


class ClipRef(_TypedRef):
    KIND = "CLIP"


class VaeRef(_TypedRef):
    KIND = "VAE"


class AudioRef(_TypedRef):
    KIND = "AUDIO"


class VideoRef(_TypedRef):
    KIND = "VIDEO"


class AssetRef(_TypedRef):
    """A file/model resolved by name+hash, tenant-scoped. Never a raw path."""

    KIND = "ASSET"


# --------------------------------------------------------------------------- #
# Runtime — per-execution binding of (ref resolver, ctx), set by the active
# ExecutionBackend. Refs read it via a contextvar, mirroring how the engine
# exposes get_executing_context(). Authors never construct it.
# --------------------------------------------------------------------------- #
@dataclass
class Runtime:
    refs: "RefResolver"
    ctx: "Context"


_active_runtime: "contextvars.ContextVar[Optional[Runtime]]" = contextvars.ContextVar(
    "comfy_sdk_runtime", default=None
)


def current_runtime() -> Runtime:
    rt = _active_runtime.get()
    if rt is None:
        raise RuntimeError(
            "No active Comfy SDK runtime. Ref/ctx access is only valid inside "
            "node execution."
        )
    return rt


def current_context() -> "Context":
    return current_runtime().ctx


class _RuntimeScope:
    """Context manager the ExecutionBackend uses to bind a runtime per node."""

    def __init__(self, runtime: Runtime) -> None:
        self._runtime = runtime
        self._token: Any = None

    def __enter__(self) -> Runtime:
        self._token = _active_runtime.set(self._runtime)
        return self._runtime

    def __exit__(self, *exc: Any) -> None:
        _active_runtime.reset(self._token)


def bind_runtime(refs: "RefResolver", ctx: "Context") -> _RuntimeScope:
    return _RuntimeScope(Runtime(refs=refs, ctx=ctx))


# --------------------------------------------------------------------------- #
# Provider interfaces. OSS ships the defaults below; the overlay overrides.
# --------------------------------------------------------------------------- #
@runtime_checkable
class RefResolver(Protocol):
    async def create(self, kind: str, obj: Any) -> Ref: ...
    async def resolve(self, ref: Ref) -> Any: ...
    async def release(self, ref: Ref) -> None: ...


@dataclass
class ExecutionPlan:
    """What the execution seam hands the backend to decide placement."""

    prompt_id: str
    node_id: str
    node_type: str
    tier: str = "default"  # overlay reads manifest tier; OSS is always "default"
    permissions: tuple[str, ...] = ()


@runtime_checkable
class ExecutionBackend(Protocol):
    async def dispatch(
        self,
        plan: ExecutionPlan,
        local_call: Callable[[], Awaitable["NodeOutput"]],
    ) -> "NodeOutput":
        """Run the node. Default just awaits ``local_call`` (in-process). The
        overlay routes ``tier == 'sandbox'`` nodes to a guest process instead,
        and calls ``local_call`` only for nodes that stay local."""
        ...


@runtime_checkable
class CtxProvider(Protocol):
    def build(self, plan: ExecutionPlan) -> "Context": ...


# --------------------------------------------------------------------------- #
# ctx — the brokered side-effect surface. Interfaces first; in-process defaults
# implement the important ones over real core. Domains not needed by the POC
# are declared and stubbed so the shape is fixed.
# --------------------------------------------------------------------------- #
class AssetsDomain(Protocol):
    async def resolve(self, folder: str, name: str) -> AssetRef: ...
    async def path(self, ref: AssetRef) -> str: ...
    async def list(self, folder: str) -> list[str]: ...


class ProgressDomain(Protocol):
    async def update(self, value: float, total: float,
                     preview: Optional[ImageRef] = None) -> None: ...


class ScratchDomain(Protocol):
    async def dir(self) -> str: ...


class EventsDomain(Protocol):
    async def emit(self, event: str, data: dict) -> None: ...


class StorageDomain(Protocol):
    async def get(self, key: str) -> Optional[str]: ...
    async def set(self, key: str, value: str) -> None: ...


class Context(Protocol):
    assets: AssetsDomain
    progress: ProgressDomain
    scratch: ScratchDomain
    events: EventsDomain
    storage: StorageDomain
    # Declared for the contract; overlay/full-SDK implement. Stubbed in OSS
    # default until wired: models, sample, serve, secrets, net.


# --------------------------------------------------------------------------- #
# Default in-process implementations (OSS). All heavy imports are lazy so this
# module stays import-safe.
# --------------------------------------------------------------------------- #
class InProcessRefResolver:
    """Identity table. A ref wraps the real object; resolve returns it as-is."""

    def __init__(self) -> None:
        self._table: dict[str, Any] = {}

    async def create(self, kind: str, obj: Any) -> Ref:
        rid = uuid.uuid4().hex
        self._table[rid] = obj
        return Ref(kind=kind, id=rid)

    async def resolve(self, ref: Ref) -> Any:
        if ref.id not in self._table:
            raise KeyError(f"unknown ref {ref!r}")
        return self._table[ref.id]

    async def release(self, ref: Ref) -> None:
        self._table.pop(ref.id, None)


class _InProcessAssets:
    async def resolve(self, folder: str, name: str) -> AssetRef:
        import folder_paths  # lazy

        full = folder_paths.get_full_path_or_raise(folder, name)
        return AssetRef._wrap(await current_runtime().refs.create("ASSET", full))  # type: ignore[return-value]

    async def path(self, ref: AssetRef) -> str:
        return await current_runtime().refs.resolve(ref)

    async def list(self, folder: str) -> list[str]:
        import folder_paths

        return list(folder_paths.get_filename_list(folder))


class _InProcessProgress:
    def __init__(self, node_id: Optional[str]) -> None:
        self._node_id = node_id

    async def update(self, value: float, total: float,
                     preview: Optional[ImageRef] = None) -> None:
        from comfy.utils import ProgressBar  # lazy

        pb = ProgressBar(total, node_id=self._node_id)
        pb.update_absolute(value, total)


class _InProcessScratch:
    async def dir(self) -> str:
        import folder_paths  # lazy

        return folder_paths.get_temp_directory()


class _InProcessEvents:
    async def emit(self, event: str, data: dict) -> None:
        from server import PromptServer  # lazy

        inst = getattr(PromptServer, "instance", None)
        if inst is not None:
            inst.send_sync(event, data)


class _StubDomain:
    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, item: str) -> Any:
        raise NotImplementedError(
            f"ctx.{self._name}.{item} is defined in the SDK contract but not yet "
            f"implemented by the in-process default. Provided by the full SDK / "
            f"overlay."
        )


@dataclass
class InProcessContext:
    assets: Any
    progress: Any
    scratch: Any
    events: Any
    storage: Any
    models: Any
    sample: Any
    serve: Any
    secrets: Any
    net: Any


class InProcessCtxProvider:
    def build(self, plan: ExecutionPlan) -> Context:
        return InProcessContext(  # type: ignore[return-value]
            assets=_InProcessAssets(),
            progress=_InProcessProgress(plan.node_id),
            scratch=_InProcessScratch(),
            events=_InProcessEvents(),
            storage=_StubDomain("storage"),
            models=_StubDomain("models"),
            sample=_StubDomain("sample"),
            serve=_StubDomain("serve"),
            secrets=_StubDomain("secrets"),
            net=_StubDomain("net"),
        )


class InProcessExecutionBackend:
    async def dispatch(
        self,
        plan: ExecutionPlan,
        local_call: Callable[[], Awaitable["NodeOutput"]],
    ) -> "NodeOutput":
        return await local_call()


# --------------------------------------------------------------------------- #
# Provider registry — the seam the overlay attaches to.
# --------------------------------------------------------------------------- #
class Providers:
    def __init__(self) -> None:
        self.execution_backend: ExecutionBackend = InProcessExecutionBackend()
        self.ctx_provider: CtxProvider = InProcessCtxProvider()
        self.ref_resolver_factory: Callable[[], RefResolver] = InProcessRefResolver
        self._overlay_name: Optional[str] = None

    # Overlay entry points -------------------------------------------------- #
    def register_execution_backend(self, impl: ExecutionBackend) -> None:
        logger.info("SDK: execution backend -> %s", type(impl).__name__)
        self.execution_backend = impl

    def register_ctx_provider(self, impl: CtxProvider) -> None:
        logger.info("SDK: ctx provider -> %s", type(impl).__name__)
        self.ctx_provider = impl

    def register_ref_resolver_factory(self, factory: Callable[[], RefResolver]) -> None:
        logger.info("SDK: ref resolver -> %s", getattr(factory, "__name__", factory))
        self.ref_resolver_factory = factory

    @property
    def overlay_active(self) -> bool:
        return self._overlay_name is not None


providers = Providers()


# --------------------------------------------------------------------------- #
# Overlay loader — the "sidecar" attach point. OSS core calls load_overlay()
# once at startup; if COMFY_OVERLAY_MODULE is unset it is a no-op and behavior
# is pure in-process. The overlay module (proprietary, separate repo) exposes
# ``register(providers)`` and installs its implementations. Mirrors how
# load_custom_node imports by file path.
# --------------------------------------------------------------------------- #
def load_overlay(spec: Optional[str] = None) -> bool:
    spec = spec if spec is not None else os.environ.get(OVERLAY_ENV)
    if not spec:
        return False

    import importlib
    import importlib.util
    import sys

    module = None
    if os.path.exists(spec):
        norm = os.path.normpath(spec)
        if os.path.isdir(norm):
            # Package directory: put its parent on sys.path and import by name
            # so intra-package relative imports resolve normally.
            pkg_name = os.path.basename(norm)
            parent = os.path.dirname(norm)
            if parent not in sys.path:
                sys.path.insert(0, parent)
            module = importlib.import_module(pkg_name)
        else:
            # Single .py file.
            modspec = importlib.util.spec_from_file_location("comfy_overlay", norm)
            if modspec and modspec.loader:
                module = importlib.util.module_from_spec(modspec)
                sys.modules["comfy_overlay"] = module
                modspec.loader.exec_module(module)
    else:
        module = importlib.import_module(spec)  # importable module name

    if module is None:
        logger.error("SDK overlay %r could not be loaded", spec)
        return False

    register = getattr(module, "register", None)
    if not callable(register):
        logger.error("SDK overlay %r has no register(providers) entrypoint", spec)
        return False

    register(providers)
    providers._overlay_name = getattr(module, "__name__", spec)
    logger.info("SDK overlay loaded: %s", providers._overlay_name)
    return True
