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

    # --- RAW ESCAPE HATCH (permissioned, discouraged) -------------------- #
    # `raw()` returns the underlying buffer object. It is NOT the preferred
    # interface: prefer operations on the asset (see ImageRef below). Raw
    # access requires the `raw`/`tensor.read` capability and forces a node to
    # the dedicated tier under the overlay. The SDK contract does not depend on
    # torch; the return is deliberately untyped (Any) here.
    async def raw(self) -> Any:
        return await current_runtime().refs.resolve(self)

    @classmethod
    async def _from_raw(cls, obj: Any) -> "TensorRef":
        return cls._wrap(await current_runtime().refs.create(cls.KIND, obj))  # type: ignore[return-value]


class ImageRef(TensorRef):
    KIND = "IMAGE"

    # --- PREFERRED INTERFACE: operations on the asset, by name. The heavy
    #     compute runs engine-side (trusted plane); the node never receives a
    #     buffer. `op` is generic dispatch — core ships a tiny built-in set and
    #     an overlay can extend the vocabulary without changing this contract.
    async def op(self, name: str, **params: Any) -> "ImageRef":
        return await current_runtime().ops.apply(name, self, params)

    # Convenience wrappers over the two built-in primitives.
    async def invert(self) -> "ImageRef":
        return await self.op("invert")

    async def scale(self, factor: float) -> "ImageRef":
        return await self.op("scale", factor=factor)


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
    ops: "OpsProvider" = None  # engine-side operations (the preferred interface)


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

    @property
    def runtime(self) -> Runtime:
        return self._runtime


def bind_runtime(
    refs: "RefResolver", ctx: "Context", ops: "OpsProvider" = None
) -> _RuntimeScope:
    return _RuntimeScope(Runtime(refs=refs, ctx=ctx, ops=ops))


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
    # Work-unit payload for out-of-process backends: import spec of the node's
    # defining module and the (ref-wrapped) inputs. Populated by the execution
    # seam for SDK_REFS nodes; in-process dispatch ignores them.
    node_module: str = ""
    inputs: Optional[dict] = None


@runtime_checkable
class ExecutionBackend(Protocol):
    async def dispatch(
        self,
        plan: ExecutionPlan,
        local_call: Callable[[], Awaitable["NodeOutput"]],
        runtime: Optional[Runtime] = None,
    ) -> "NodeOutput":
        """Run the node. Default just awaits ``local_call`` (in-process). The
        overlay routes ``tier == 'sandbox'`` nodes to a guest process instead,
        and calls ``local_call`` only for nodes that stay local. ``runtime`` is
        the host-side binding (refs/ctx/ops) for this node so an out-of-process
        backend can serve brokered guest calls against the same ref table."""
        ...


@runtime_checkable
class CtxProvider(Protocol):
    def build(self, plan: ExecutionPlan) -> "Context": ...


@runtime_checkable
class OpsProvider(Protocol):
    """Engine-side operations on assets — the preferred node interface. The
    node passes/receives refs; the buffer math happens here, on the trusted
    plane (in-process default) or in the engine (overlay). Dispatch is generic
    (``apply(op, image, params)``) so the op vocabulary is data, not API
    surface: an overlay adds ops without changing this contract, and a node
    can probe ``supports(op)`` to choose a fallback (e.g. the ``raw`` tier)."""

    async def apply(self, op: str, image: "ImageRef", params: dict) -> "ImageRef": ...
    def supports(self, op: str) -> bool: ...


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


class OpNotSupported(NotImplementedError):
    """Raised by ``apply`` for an op this provider does not implement. Carries
    the capability name so a node can decide to fall back (e.g. to ``raw``)."""

    def __init__(self, op: str) -> None:
        self.op = op
        self.capability = f"ops.{op}"
        super().__init__(
            f"op {op!r} is not supported by this ops provider "
            f"(capability: {self.capability})"
        )


class InProcessOps:
    """Default operations. Runs in the trusted process; uses the real buffers
    via the resolver but never hands them to the node. Kept torch-free (tensor
    arithmetic works on the resolved objects directly). The op set is a
    registry: core ships two primitives; ``register_op`` extends it (an overlay
    subclasses or registers richer ops)."""

    def __init__(self) -> None:
        self._ops: dict[str, Callable[..., Awaitable["ImageRef"]]] = {
            "invert": self._invert,
            "scale": self._scale,
        }

    def register_op(self, name: str, fn: Callable[..., Awaitable["ImageRef"]]) -> None:
        self._ops[name] = fn

    def supports(self, op: str) -> bool:
        return op in self._ops

    async def apply(self, op: str, image: "ImageRef", params: dict) -> "ImageRef":
        fn = self._ops.get(op)
        if fn is None:
            raise OpNotSupported(op)
        return await fn(image, **params)

    async def _invert(self, image: "ImageRef") -> "ImageRef":
        t = await current_runtime().refs.resolve(image)
        return ImageRef._wrap(await current_runtime().refs.create("IMAGE", 1.0 - t))  # type: ignore[return-value]

    async def _scale(self, image: "ImageRef", factor: float) -> "ImageRef":
        t = await current_runtime().refs.resolve(image)
        return ImageRef._wrap(await current_runtime().refs.create("IMAGE", t * factor))  # type: ignore[return-value]


class InProcessExecutionBackend:
    async def dispatch(
        self,
        plan: ExecutionPlan,
        local_call: Callable[[], Awaitable["NodeOutput"]],
        runtime: Optional[Runtime] = None,
    ) -> "NodeOutput":
        return await local_call()


# --------------------------------------------------------------------------- #
# Input/output ref marshaling for SDK nodes (those declaring ``SDK_REFS``).
# Heavy inputs become refs before execute(); output refs resolve back to real
# objects for downstream (legacy) nodes. This is what makes execute() see
# assets, not buffers. Under the overlay this happens at the process boundary.
# --------------------------------------------------------------------------- #
def _looks_like_tensor(v: Any) -> bool:
    return type(v).__name__ == "Tensor" and hasattr(v, "shape")


def _is_plain_data(v: Any) -> bool:
    """Whether a value can cross a process boundary as data.

    Deliberately a whitelist. Everything else is a live engine object — a
    ModelPatcher, a conditioning list holding tensors, a VAE — and handing one
    to an out-of-process node is either impossible (it will not serialize) or
    exactly what the boundary exists to prevent.
    """
    if v is None or isinstance(v, (str, bool, int, float, bytes)):
        return True
    if isinstance(v, (list, tuple)):
        return all(_is_plain_data(x) for x in v)
    if isinstance(v, dict):
        return all(isinstance(k, str) and _is_plain_data(x) for k, x in v.items())
    return False


#: Live engine objects a node may receive, by the ref type that stands in for
#: them. Detection is duck-typed because these classes live in `comfy.*`, which
#: this module must not import.
def _ref_type_for(v: Any) -> tuple[type, str]:
    if _looks_like_tensor(v):
        return ImageRef, "IMAGE"
    if isinstance(v, dict) and "samples" in v:
        return LatentRef, "LATENT"
    if hasattr(v, "model_options") and hasattr(v, "load_device"):
        return ModelRef, "MODEL"          # ModelPatcher
    if hasattr(v, "encode_from_tokens") or hasattr(v, "tokenize"):
        return ClipRef, "CLIP"
    if hasattr(v, "decode") and hasattr(v, "encode"):
        return VaeRef, "VAE"
    return CondRef, "CONDITIONING"        # conditioning lists and anything else


async def wrap_inputs(resolver: "RefResolver", inputs: dict) -> dict:
    """Replace live engine objects with refs before a node sees them.

    An SDK_REFS node is handed handles, never the objects themselves, so the
    same node body works in-process and out-of-process. The rule is by
    capability rather than by an enumerated type list: if a value cannot cross
    as data, it becomes a handle. That is what lets a node take a MODEL or a
    CONDITIONING — which are live engine objects — and still run in a guest.
    """
    out = {}
    for k, v in inputs.items():
        if _is_plain_data(v):
            out[k] = v
            continue
        ref_cls, kind = _ref_type_for(v)
        out[k] = ref_cls._wrap(await resolver.create(kind, v))
    return out


async def unwrap_outputs(resolver: "RefResolver", node_output: Any) -> Any:
    args = getattr(node_output, "result", None)
    if not args:
        return node_output
    resolved = []
    for a in args:
        resolved.append(await resolver.resolve(a) if isinstance(a, Ref) else a)
    from ._io import NodeOutput

    # Rebuilding the NodeOutput must preserve everything that is not a result.
    # Dropping `ui` here silently made every SDK_REFS node unable to be an
    # output node: ComfyUI only sends the `executed` event that carries results
    # to the frontend for nodes returning ui data, so a PreviewImage-style node
    # would run correctly and then display nothing. `expand` and
    # `block_execution` matter for the same reason — they are node output, not
    # node results, and resolving refs has no business discarding them.
    return NodeOutput(
        *resolved,
        ui=getattr(node_output, "ui", None),
        expand=getattr(node_output, "expand", None),
        block_execution=getattr(node_output, "block_execution", None),
    )


# --------------------------------------------------------------------------- #
# Provider registry — the seam the overlay attaches to.
# --------------------------------------------------------------------------- #
class Providers:
    def __init__(self) -> None:
        self.execution_backend: ExecutionBackend = InProcessExecutionBackend()
        self.ctx_provider: CtxProvider = InProcessCtxProvider()
        self.ops_provider: OpsProvider = InProcessOps()
        self.ref_resolver_factory: Callable[[], RefResolver] = InProcessRefResolver
        self._overlay_name: Optional[str] = None

    # Overlay entry points -------------------------------------------------- #
    def register_execution_backend(self, impl: ExecutionBackend) -> None:
        logger.info("SDK: execution backend -> %s", type(impl).__name__)
        self.execution_backend = impl

    def register_ctx_provider(self, impl: CtxProvider) -> None:
        logger.info("SDK: ctx provider -> %s", type(impl).__name__)
        self.ctx_provider = impl

    def register_ops_provider(self, impl: OpsProvider) -> None:
        logger.info("SDK: ops provider -> %s", type(impl).__name__)
        self.ops_provider = impl

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
