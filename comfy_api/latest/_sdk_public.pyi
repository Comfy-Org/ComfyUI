"""
Authoritative type contract for the secure custom-node SDK.

This is the backend counterpart to the frontend published API's generated
``comfy-api.d.ts``: the stable, versioned surface a node author compiles
against when importing ``from comfy_api.v0_0_3 import sdk`` (or a pinned
version). Type checkers prefer this stub over ``_sdk_public.py``, so it defines
the *contract*; the ``.py`` defines the in-process default *implementation*.

Two audiences, separated below:
  1. NODE AUTHORS — refs, ctx and its domains, and the ``ctx()`` accessor.
  2. HOST / OVERLAY — the provider seam (ExecutionBackend / CtxProvider /
     RefResolver / ExecutionPlan / providers). Node authors do not use these;
     they exist so a separable overlay can replace the defaults with an
     isolated engine. (Analogous to the frontend "host-plumbing declarations".)

If prose and this stub disagree, treat the stub + its implementation tests as
authoritative.
"""
from typing import Any, Awaitable, Callable, Optional, Protocol, TypeVar, runtime_checkable

# NOTE: the contract deliberately does NOT import torch or any backend module.
# Node code has no direct access to torch/CUDA/filesystem; it works through
# refs, operations, and ctx — all brokered.

# =========================================================================== #
# 1. NODE-AUTHOR SURFACE
# =========================================================================== #

# --- Refs: opaque, typed asset handles. A ref never exposes the buffer. The
#     PREFERRED interface is operations on the asset (below); raw buffer access
#     is a permissioned escape hatch. --- #
class Ref:
    kind: str
    id: str

_T = TypeVar("_T", bound="TensorRef")

class TensorRef(Ref):
    KIND: str
    # RAW ESCAPE HATCH — permissioned (`raw`/`tensor.read`), discouraged; forces
    # the dedicated tier under the overlay. Return is untyped by design (the
    # contract does not depend on torch).
    async def raw(self) -> Any: ...

class ImageRef(TensorRef):
    """IMAGE asset. Preferred interface = operations; the buffer stays engine-side."""
    KIND: str
    async def invert(self) -> "ImageRef": ...
    async def scale(self, factor: float) -> "ImageRef": ...

class MaskRef(TensorRef):
    """MASK asset."""
    KIND: str

_L = TypeVar("_L", bound="LatentRef")

class LatentRef(Ref):
    """LATENT — dict {samples, ...}."""
    KIND: str
    async def value(self) -> dict: ...
    @classmethod
    async def from_value(cls: type[_L], v: dict) -> _L: ...

class CondRef(Ref):
    """CONDITIONING."""
    KIND: str

class ModelRef(Ref):
    """MODEL — patch/hook via ctx.models; weights never materialize in-node."""
    KIND: str

class ClipRef(Ref):
    KIND: str

class VaeRef(Ref):
    KIND: str

class AudioRef(Ref):
    KIND: str

class VideoRef(Ref):
    KIND: str

class AssetRef(Ref):
    """A file/model resolved by name+hash, tenant-scoped. Never a raw path."""
    KIND: str

# --- ctx domains: the brokered side-effect surface. In-process these call core
#     directly (allow-all); under the overlay they are policy-checked and
#     tenant-scoped. Domains marked (overlay) raise NotImplementedError in the
#     OSS in-process default until wired. --- #
class AssetsDomain(Protocol):
    async def resolve(self, folder: str, name: str) -> AssetRef: ...
    async def path(self, ref: AssetRef) -> str: ...
    async def list(self, folder: str) -> list[str]: ...

class ProgressDomain(Protocol):
    async def update(
        self, value: float, total: float, preview: Optional[ImageRef] = ...
    ) -> None: ...

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
    # Declared in the contract; provided by the full SDK / overlay:
    models: Any   # ModelRef patch/hook/load
    sample: Any   # engine-side sampling with per-step callbacks
    serve: Any    # namespaced, authenticated routes
    secrets: Any  # brokered from the BYOK store
    net: Any      # policy-checked fetch

def ctx() -> Context:
    """The active Context. Valid only inside node execution."""
    ...

def current_context() -> Context: ...

# =========================================================================== #
# 2. HOST / OVERLAY SEAM  (not for node authors)
# =========================================================================== #
class ExecutionPlan:
    prompt_id: str
    node_id: str
    node_type: str
    tier: str
    permissions: tuple[str, ...]
    def __init__(
        self,
        prompt_id: str,
        node_id: str,
        node_type: str,
        tier: str = ...,
        permissions: tuple[str, ...] = ...,
    ) -> None: ...

@runtime_checkable
class RefResolver(Protocol):
    async def create(self, kind: str, obj: Any) -> Ref: ...
    async def resolve(self, ref: Ref) -> Any: ...
    async def release(self, ref: Ref) -> None: ...

@runtime_checkable
class ExecutionBackend(Protocol):
    async def dispatch(
        self, plan: ExecutionPlan, local_call: Callable[[], Awaitable[Any]]
    ) -> Any: ...

@runtime_checkable
class CtxProvider(Protocol):
    def build(self, plan: ExecutionPlan) -> Context: ...

class _Providers:
    execution_backend: ExecutionBackend
    ctx_provider: CtxProvider
    ref_resolver_factory: Callable[[], RefResolver]
    @property
    def overlay_active(self) -> bool: ...
    def register_execution_backend(self, impl: ExecutionBackend) -> None: ...
    def register_ctx_provider(self, impl: CtxProvider) -> None: ...
    def register_ref_resolver_factory(
        self, factory: Callable[[], RefResolver]
    ) -> None: ...

providers: _Providers
