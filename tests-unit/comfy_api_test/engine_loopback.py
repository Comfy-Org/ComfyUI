"""A pack in a separate-program shape, without the separate program.

``run_in_process`` is how ComfyUI runs an engine-API node: values in, values out,
``eng`` calling the engine directly.

``run_separate`` puts a boundary in the same process, with the rules a process
boundary enforces. Control data goes through real JSON: refs, scalars and
containers cross, and anything else is an error. Engine objects stay on the host
side and cross as refs. Data crosses as copies, so nothing can work by sharing
an object with the engine. Refs exist only on the wire: the pack's side gives
the pack the data behind a data ref, and keeps an engine ref only to pass back.
"""
from __future__ import annotations

import json
import types
import uuid
from typing import Any

from comfy_api.index import io
from comfy_api._impl import _engine, _runtime


def _strict(value: Any, closures: dict | None = None) -> Any:
    if isinstance(value, _runtime.Ref):
        return {"__ref__": {"kind": value.kind, "id": value.id, "cls": type(value).__name__}}
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (bytes, bytearray)):
        import base64
        return {"__bytes__": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, (types.FunctionType, types.MethodType)):
        if closures is None:
            raise TypeError("a function can only cross as a call argument")
        cid = uuid.uuid4().hex
        closures[cid] = value
        return {"__clo__": cid}
    if isinstance(value, (list, tuple)):
        return [_strict(v, closures) for v in value]
    if isinstance(value, dict):
        return {str(k): _strict(v, closures) for k, v in value.items()}
    raise TypeError(f"{type(value).__name__} cannot cross the wire")


def _wire(value: Any, closures: dict | None = None) -> Any:
    return json.loads(json.dumps(_strict(value, closures)))


def _unwire(value: Any, closure=None) -> Any:
    if isinstance(value, dict):
        token = value.get("__ref__")
        if token is not None:
            return _runtime.Ref(kind=token["kind"], id=token["id"])
        if "__clo__" in value:
            return closure(value["__clo__"])
        if set(value) == {"__bytes__"}:
            import base64
            return base64.b64decode(value["__bytes__"])
        return {k: _unwire(v, closure) for k, v in value.items()}
    if isinstance(value, list):
        return [_unwire(v, closure) for v in value]
    return value


def _copy_data(value: Any, on_object=None) -> Any:
    """A deep copy of tensors and containers, as a shared-memory channel gives."""
    if _runtime._looks_like_tensor(value):
        return value.detach().clone()
    if isinstance(value, (str, bytes, int, float, bool, type(None), _runtime.Ref, _engine.Opaque, _engine.EngineFunction)):
        return value  # engine values and engine functions cross as tokens, and come back as themselves
    if isinstance(value, dict):
        return {k: _copy_data(v, on_object) for k, v in value.items()}
    if type(value) in (list, tuple):
        return type(value)(_copy_data(v, on_object) for v in value)
    if on_object is not None:
        return on_object(value)
    raise TypeError(f"{type(value).__name__} can't cross as data")


class Host:
    """The engine's side: a ref table, and the in-process engine calls."""

    def __init__(self, pack: str) -> None:
        self.refs = _runtime.InProcessRefResolver()
        self.ops = _runtime.EngineCalls()
        self.pack = pack
        self.plan = None
        self.scope = lambda: _runtime.bind_runtime(self.refs, None, self.ops, pack=self.pack, plan=self.plan)

    def ref_for(self, engine_object: Any) -> _runtime.Ref:
        kind = _runtime.handle_kind(engine_object)
        rid = uuid.uuid4().hex
        self.refs._table[rid] = (kind, engine_object)
        return _runtime.Ref(kind=kind, id=rid)

    async def send(self, value: Any) -> Any:
        """Host value -> wire form: engine objects become refs, data is copied over."""
        return _wire(await _runtime.wrap_value(self.refs, value))

    async def receive(self, value: Any) -> Any:
        """Wire form from the pack's side -> host values. Engine refs stay refs for dispatch."""
        return _unwire(value)


class Guest:
    """The pack's side of the boundary."""

    def __init__(self, host: Host) -> None:
        self.host = host

    async def to_values(self, wired: Any) -> Any:
        """What the pack sees: the data behind data refs; engine refs kept to pass back."""
        value = _unwire(wired)
        return await self._read(value)

    async def _read(self, value: Any) -> Any:
        if isinstance(value, _runtime.Ref):
            if value.kind not in _runtime.DATA_KINDS:
                return value
            data = await self.host.refs.resolve(value)
            # Engine objects inside data (a ControlNet in conditioning) stay on the host.
            return _copy_data(data, on_object=self.host.ref_for)
        if type(value) is dict:
            return {k: await self._read(v) for k, v in value.items()}
        if type(value) in (list, tuple):
            return type(value)([await self._read(v) for v in value])
        return value

    async def to_wire(self, value: Any, closures: dict | None = None) -> Any:
        """Pack value -> wire form: data the pack holds is stored on the host as a copy."""
        async def store(v: Any) -> Any:
            if isinstance(v, _runtime.Ref) or _engine._is_function(v) or _runtime._is_plain_data(v):
                return v
            kind = _runtime.handle_kind(v)
            if kind == "VALUE" and type(v) is dict:
                return {k: await store(x) for k, x in v.items()}
            if kind == "VALUE" and type(v) in (list, tuple):
                return type(v)([await store(x) for x in v])
            # A tensor, latent, conditioning...: one copy on the host. Anything
            # that isn't data (an object the pack built) fails to copy.
            return await self.host.refs.create(kind, _copy_data(v))
        return _wire(await store(value), closures)


class GuestOps:
    """``eng`` calls from the pack's side: values out, values back."""

    def __init__(self, host: Host, guest: Guest, guest_scope) -> None:
        self.host = host
        self.guest = guest
        self.guest_scope = guest_scope
        self.closures: dict[str, Any] = {}
        self.calls: list[str] = []
        self.hook_calls = 0

    def _closure(self, cid: str):
        # What runs on the engine side when the sampler calls a pack function.
        def invoke(*args: Any, **kwargs: Any) -> Any:
            self.hook_calls += 1
            with self.guest_scope():
                result = self.closures[cid](*_copy_data(list(args)), **_copy_data(kwargs))
            return _copy_data(result)
        return invoke

    def supports(self, op: str) -> bool:
        return True

    async def apply(self, op: str, subject: Any, params: dict) -> Any:
        self.calls.append(params.get("name", op))
        sent = await self.guest.to_wire(params, self.closures)
        with self.host.scope():
            result = await self.host.ops.apply(op, None, _unwire(sent, self._closure))
            wired = await self.host.send(result)
        return await self.guest.to_values(wired)


def _plan(node_cls: Any, inputs: dict, method: str) -> Any:
    return _runtime.ExecutionPlan(prompt_id="t", node_id="1", node_type=node_cls.__name__,
                                  node_module=node_cls.__module__, inputs=inputs, method=method)


async def run_in_process(node_cls: Any, pack: str, inputs: dict, method: str = "execute") -> Any:
    with _runtime.bind_runtime(_runtime.InProcessRefResolver(), None, _runtime.EngineCalls(), pack=pack,
                               plan=_plan(node_cls, inputs, method)):
        output = await getattr(node_cls, method)(**inputs)
    return output, None


async def run_separate(node_cls: Any, pack: str, inputs: dict, method: str = "execute") -> Any:
    host = Host(pack)
    host.plan = _plan(node_cls, inputs, method)
    guest = Guest(host)
    box: dict[str, Any] = {}

    def scope():
        return _runtime.bind_runtime(_runtime.InProcessRefResolver(), None, box["ops"], pack="")

    box["ops"] = GuestOps(host, guest, scope)
    crossed = await guest.to_values(await host.send(inputs))
    with scope():
        output = await getattr(node_cls, method)(**crossed)
    results = _unwire(await guest.to_wire(list(getattr(output, "result", ()) or ())))
    decoded = io.NodeOutput(*results, ui=getattr(output, "ui", None))
    return await _runtime.unwrap_outputs(host.refs, decoded), box["ops"]


_backend = None


async def run_out_of_process(node_cls: Any, pack: str, inputs: dict, method: str = "execute") -> Any:
    """The node in a real worker process, as ``OOP_CUSTOM_NODES=true`` runs it."""
    global _backend
    import os
    import sys
    import pytest
    out_of_process = pytest.importorskip("comfy_secure_nodes.v21.isolation")

    if "<locals>" in node_cls.__qualname__ and node_cls.__module__.startswith("test"):
        import pytest
        pytest.skip("a class defined inside a test function can't be imported by a worker process")
    if _backend is None:
        os.environ["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
        _backend = out_of_process.OutOfProcessBackend()
    hidden = getattr(node_cls, "hidden", None)  # tests set hidden inputs on a class clone
    plan = _runtime.ExecutionPlan(prompt_id="t", node_id=str(getattr(hidden, "unique_id", None) or "1"),
                                  node_type=node_cls.__name__, node_module=node_cls.__module__, inputs=inputs,
                                  method=method, prompt=getattr(hidden, "prompt", None),
                                  extra_pnginfo=getattr(hidden, "extra_pnginfo", None))
    refs = _runtime.InProcessRefResolver()
    runtime = _runtime.Runtime(refs=refs, ctx=None, ops=_runtime.EngineCalls(), pack=pack, plan=plan)
    root = os.path.dirname(os.path.abspath(sys.modules[node_cls.__module__].__file__))
    output = await _backend.run_class(node_cls, root, plan, runtime)
    try:
        return await _runtime.unwrap_outputs(refs, output), None
    finally:
        refs.clear()


MODES = {"in_process": run_in_process, "separate": run_separate}
if __import__("os").environ.get("ENGINE_TEST_OOP") == "1":
    MODES["out_of_process"] = run_out_of_process
