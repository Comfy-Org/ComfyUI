"""What a node runs inside: its engine, and how it gets dispatched.

In process, an engine-API node gets its inputs as they are and ``eng`` calls the
engine directly. A provider module, named by ``COMFY_EXECUTION_PROVIDER``, can
replace how nodes are dispatched. One that runs pack code in another process
uses the handle machinery here (``Ref``, a ref table, ``wrap_value``) to carry
engine objects across its boundary. Handles are transport: pack code never
needs to know they exist.
"""
from __future__ import annotations

import contextvars
import importlib
import importlib.util
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)
PROVIDER_ENV = "COMFY_EXECUTION_PROVIDER"


@dataclass(frozen=True)
class Ref:
    """A handle a provider uses to carry a value across a process boundary.

    ``kind`` is the io type (IMAGE, MODEL...). In-process dispatch never makes one.
    """

    kind: str
    id: str

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.id[:8]}>"

    @classmethod
    def _wrap(cls, ref: "Ref") -> "Ref":
        return cls(kind=getattr(cls, "KIND", None) or ref.kind, id=ref.id)

    async def release(self) -> None:
        """Optional early free. Every handle is released when the node returns."""
        await current_runtime().refs.release(self)


@dataclass
class Runtime:
    refs: Any
    ctx: Any = None
    ops: Any = None
    pack: str = ""
    plan: Any = None


_active_runtime: contextvars.ContextVar[Optional[Runtime]] = contextvars.ContextVar(
    "comfy_engine_runtime", default=None)


def current_runtime() -> Runtime:
    rt = _active_runtime.get()
    if rt is None:
        raise RuntimeError("No active runtime: the engine is only available while a node is running")
    return rt


class _RuntimeScope:
    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        self._token: Any = None

    def __enter__(self) -> Runtime:
        self._token = _active_runtime.set(self.runtime)
        return self.runtime

    def __exit__(self, *exc: Any) -> None:
        _active_runtime.reset(self._token)


def bind_runtime(refs: Any, ctx: Any = None, ops: Any = None, pack: str = "", plan: Any = None) -> _RuntimeScope:
    return _RuntimeScope(Runtime(refs=refs, ctx=ctx, ops=ops, pack=pack, plan=plan))


class InProcessRefResolver:
    """The handle table. It records each handle's kind, and refuses a relabelled one."""

    def __init__(self) -> None:
        self._table: dict[str, tuple[str, Any]] = {}
        self._closed = False

    async def create(self, kind: str, obj: Any) -> Ref:
        if self._closed:
            raise RuntimeError("this node execution has ended; its ref table is closed")
        rid = uuid.uuid4().hex
        self._table[rid] = (kind, obj)
        return Ref(kind=kind, id=rid)

    async def resolve(self, ref: Ref) -> Any:
        entry = self._table.get(ref.id)
        if entry is None:
            raise KeyError(f"unknown ref {ref!r}")
        kind, obj = entry
        if ref.kind != kind:
            raise TypeError(
                f"ref {ref.id[:8]} was created as {kind} but presented as "
                f"{ref.kind}; the holder of a handle does not get to relabel it")
        return obj

    async def release(self, ref: Ref) -> None:
        self._table.pop(ref.id, None)

    def clear(self) -> int:
        """Drop every entry when the node ends, so large tensors go at a known point."""
        n = len(self._table)
        self._table.clear()
        self._closed = True
        return n


# --------------------------------------------------------------------------- #
# Handles for values
# --------------------------------------------------------------------------- #
def _looks_like_tensor(v: Any) -> bool:
    return type(v).__name__ == "Tensor" and hasattr(v, "shape")


def _is_plain_data(v: Any) -> bool:
    """Whether a value is data. Everything else is an engine object and gets a handle."""
    if v is None or isinstance(v, (str, bool, int, float, bytes)):
        return True
    if isinstance(v, (list, tuple)):
        return all(_is_plain_data(x) for x in v)
    if isinstance(v, dict):
        return all(isinstance(k, str) and _is_plain_data(x) for k, x in v.items())
    return False


def _is_conditioning(v: Any) -> bool:
    return (isinstance(v, (list, tuple)) and bool(v) and isinstance(v[0], (list, tuple))
            and len(v[0]) == 2 and _looks_like_tensor(v[0][0]) and isinstance(v[0][1], dict))


def _kind_of(v: Any) -> str:
    """The handle kind for a value. Duck-typed: this module doesn't import the engine."""
    if _looks_like_tensor(v):
        return "SIGMAS" if getattr(v, "ndim", None) == 1 else "IMAGE"
    if hasattr(v, "get_control") and hasattr(v, "set_extra_arg") and hasattr(v, "copy"):
        return "CONTROL_NET"
    if hasattr(v, "get_stream_source") and hasattr(v, "get_components") and hasattr(v, "save_to"):
        return "VIDEO"
    if hasattr(v, "model_patcher") and hasattr(v, "set_conds") and hasattr(v, "predict_noise"):
        return "GUIDER"
    if type(v).__name__ == "KSAMPLER" and callable(getattr(v, "sample", None)):
        return "SAMPLER"
    if type(v).__name__ == "ImageModelDescriptor" and hasattr(v, "scale"):
        return "UPSCALE_MODEL"
    if hasattr(v, "model_options") and hasattr(v, "load_device"):
        return "MODEL"
    if hasattr(v, "encode_from_tokens") or hasattr(v, "tokenize"):
        return "CLIP"
    if hasattr(v, "decode") and hasattr(v, "encode"):
        return "VAE"
    if isinstance(v, dict):
        if "samples" in v:
            return "LATENT"
        return "AUDIO" if "waveform" in v and "sample_rate" in v else "VALUE"
    if isinstance(v, (list, tuple)):
        return "CONDITIONING" if _is_conditioning(v) else "VALUE"
    return "OPAQUE"


#: Kinds whose handles stand for data a pack can read. The rest are engine objects.
DATA_KINDS = frozenset({"IMAGE", "MASK", "SIGMAS", "TENSOR", "LATENT", "CONDITIONING", "AUDIO", "VALUE"})

_handle_type: Callable[[Any], tuple[type, str]] = lambda v: (Ref, _kind_of(v))
_single_handle: Callable[[Any], bool] = lambda v: False


def set_handle_types(fn: Callable[[Any], tuple[type, str]], single: Optional[Callable[[Any], bool]] = None) -> None:
    """Let a provider choose the handle class for a value, and which structures get one handle."""
    global _handle_type, _single_handle
    _handle_type = fn
    _single_handle = single or (lambda v: False)


def handle_kind(value: Any) -> str:
    return _handle_type(value)[1]


async def wrap_value(resolver: Any, value: Any) -> Any:
    """Keep plain data; give anything else a handle, leaf by leaf."""
    if isinstance(value, Ref) or _is_plain_data(value):
        return value
    single = _single_handle(value)
    # A structured socket may carry several engine objects, so keep its shape.
    if isinstance(value, dict) and not (
        single or "samples" in value
        or ("waveform" in value and "sample_rate" in value)
        or set(value) >= {"model", "processor"}
    ):
        return {key: await wrap_value(resolver, item) for key, item in value.items()}
    if isinstance(value, (list, tuple)) and not (single or _is_conditioning(value)):
        wrapped = [await wrap_value(resolver, item) for item in value]
        return tuple(wrapped) if isinstance(value, tuple) else wrapped
    ref_cls, kind = _handle_type(value)
    return ref_cls._wrap(await resolver.create(kind, value))


async def wrap_inputs(resolver: Any, inputs: dict) -> dict:
    return {key: await wrap_value(resolver, value) for key, value in inputs.items()}


async def unwrap_outputs(resolver: Any, node_output: Any) -> Any:
    args = getattr(node_output, "result", None)
    if not args:
        return node_output

    async def resolve(value: Any) -> Any:
        if isinstance(value, Ref):
            value = await resolver.resolve(value)
        if isinstance(value, list):
            return [await resolve(item) for item in value]
        if isinstance(value, tuple):
            return tuple([await resolve(item) for item in value])
        if isinstance(value, dict):
            return {key: await resolve(item) for key, item in value.items()}
        return value

    resolved = [await resolve(value) for value in args]
    from comfy_api._impl._io import NodeOutput

    # Everything that isn't a result survives: an output node's ui depends on it.
    return NodeOutput(
        *resolved,
        ui=getattr(node_output, "ui", None),
        expand=getattr(node_output, "expand", None),
        block_execution=getattr(node_output, "block_execution", None),
    )


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
@dataclass
class ExecutionPlan:
    """One node invocation, as handed to the execution backend."""

    prompt_id: str
    node_id: str
    node_type: str
    node_module: str = ""
    inputs: Optional[dict] = None
    input_mode: str = "refs"
    prompt: Any = None
    extra_pnginfo: Any = None
    dynamic_prompt: Any = None
    method: str = "execute"
    execution_metadata: Any = None


class InProcessExecutionBackend:
    async def dispatch(self, plan: ExecutionPlan, local_call: Callable[[], Awaitable[Any]],
                       runtime: Optional[Runtime] = None) -> Any:
        return await local_call()


class CallNotSupported(LookupError):
    """The name is not a table row, or an extension the calling pack ships."""


class EngineCalls:
    """The default ops provider: engine calls go straight to the dispatcher."""

    async def apply(self, op: str, subject: Any, params: dict) -> Any:
        if op != "engine.call":
            raise CallNotSupported(f"unknown operation {op!r}")
        from . import _engine
        return await _engine.dispatch(subject, **params)

    def supports(self, op: str) -> bool:
        return op == "engine.call"


class _NoContext:
    def build(self, plan: ExecutionPlan) -> None:
        return None


class Providers:
    """How nodes are dispatched. Defaults run everything in this process."""

    frontend_runtime_config: dict = {}

    def __init__(self) -> None:
        self.execution_backend: Any = InProcessExecutionBackend()
        self.ctx_provider: Any = _NoContext()
        self.ops_provider: Any = EngineCalls()
        self.ref_resolver_factory: Callable[[], Any] = InProcessRefResolver
        self.load_custom_nodes = True

    @property
    def engaged(self) -> bool:
        """Whether every node goes through the backend, not only engine-API nodes."""
        return not (type(self.execution_backend) is InProcessExecutionBackend
                    and self.ref_resolver_factory is InProcessRefResolver)

    def uses_handles(self, node_class: Any) -> bool:
        """Whether this node's inputs become handles. Only a boundary provider says yes."""
        return False

    def node_engaged(self, node_class: Any) -> bool:
        """Whether the node runs with a bound runtime, so ``eng`` works inside it."""
        return bool(getattr(node_class, "ENGINE_API", False)) or self.uses_handles(node_class) or self.engaged

    def always_changed(self, node_class: Any) -> bool:
        """Whether a node's output can't be reused from cache."""
        return False

    def observes_execution(self, node_class: Any) -> bool:
        return False

    def build_plan(self, node_class: Any, **fields: Any) -> Any:
        return ExecutionPlan(**fields)


providers = Providers()


async def begin_node(node_class: Any, inputs: Any, *, method: str, prompt_id: Any, node_id: Any,
                     handles: bool, hidden: Any = None, metadata: Any = None) -> tuple:
    """Build the plan, handle table and runtime for one node invocation."""
    plan = providers.build_plan(
        node_class,
        prompt_id=str(prompt_id),
        node_id=str(node_id),
        node_type=getattr(node_class, "__name__", "node"),
        node_module=getattr(node_class, "__module__", "") or "",
        inputs=inputs,
        input_mode="refs" if handles else "values",
        method=method,
        prompt=getattr(hidden, "prompt", None),
        extra_pnginfo=getattr(hidden, "extra_pnginfo", None),
        dynamic_prompt=getattr(hidden, "dynprompt", None),
        execution_metadata=metadata,
    )
    refs = providers.ref_resolver_factory()
    scope = bind_runtime(refs, providers.ctx_provider.build(plan), providers.ops_provider,
                         pack=plan.node_module, plan=plan)
    if handles:
        inputs = await wrap_inputs(refs, inputs)
        plan.inputs = inputs
    return plan, refs, scope, inputs


def load_execution_provider(spec: Optional[str] = None) -> bool:
    """Import the module named by ``COMFY_EXECUTION_PROVIDER`` and call its ``register``.

    With no provider named, ``COMFY_PACK_ISOLATION=true`` selects ComfyUI's own: each custom-node
    pack runs in its own pack process (``comfy_api._impl.isolation``).
    """
    spec = spec if spec is not None else os.environ.get(PROVIDER_ENV)
    if not spec:
        from comfy_api._impl import isolation as out_of_process
        if out_of_process.enabled():
            spec = "comfy_api._impl.isolation"
    if not spec:
        return False
    module = None
    if os.path.exists(spec):
        norm = os.path.normpath(spec)
        if os.path.isdir(norm):
            parent = os.path.dirname(norm)
            if parent not in sys.path:
                sys.path.insert(0, parent)
            module = importlib.import_module(os.path.basename(norm))
        else:
            modspec = importlib.util.spec_from_file_location("comfy_execution_provider", norm)
            if modspec and modspec.loader:
                module = importlib.util.module_from_spec(modspec)
                sys.modules["comfy_execution_provider"] = module
                modspec.loader.exec_module(module)
    else:
        module = importlib.import_module(spec)
    register = getattr(module, "register", None)
    if not callable(register):
        raise RuntimeError(f"execution provider {spec!r} has no register(providers) entrypoint")
    register(providers)  # looked up here: importing the module may replace the registry
    logger.info("execution provider loaded: %s", getattr(module, "__name__", spec))
    return True
