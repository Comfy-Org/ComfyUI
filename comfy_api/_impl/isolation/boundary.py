"""What crosses between ComfyUI and a pack's worker process."""
from __future__ import annotations

import dataclasses
import itertools
import types
from typing import Any, Optional


@dataclasses.dataclass(frozen=True)
class Closure:
    """A function that stays in the worker; ComfyUI calls it by id."""
    id: int


@dataclasses.dataclass(frozen=True)
class HostValue:
    """An engine value or engine function that stays in ComfyUI, inside what a pack function is
    given (``_engine.Opaque`` / ``_engine.EngineFunction``). The pack passes it back; calling a
    callable one is a request to ComfyUI, made while the pack function runs."""
    id: int
    callable: bool = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not self.callable:
            raise TypeError("an engine value can't be called; pass it to the engine function it came with")
        from .worker import current_state
        state = current_state()
        if state is None or state.channel is None:
            raise RuntimeError("an engine function can only be called while the pack function it was given to runs")
        return state.channel.request({"op": "host_call", "value": self.id,  # pack functions cross as Closures
                                      "args": from_worker(list(args), state.functions, "argument"),
                                      "kwargs": from_worker(dict(kwargs), state.functions, "argument")})


def _is_tensor(value: Any) -> bool:
    return type(value).__name__ == "Tensor" and hasattr(value, "shape")


def _is_function(value: Any) -> bool:
    """A pack function: it stays in the pack's process and is called there. An object the pack made to be called
    (a class with ``__call__``, as FDGuidance's guidance calculator) is one too, as it is in ComfyUI."""
    import functools
    if isinstance(value, (types.FunctionType, types.MethodType, functools.partial)):
        return True
    return (not isinstance(value, type) and type(value).__module__ != "builtins" and not _is_tensor(value)
            and callable(getattr(type(value), "__call__", None)))


@dataclasses.dataclass(frozen=True)
class PackValue:
    """A node output that isn't data (a pack's own type, such as Impact's SEGS): it stays in the
    pack's process, and the pack's next node gets the object back. ``pid`` is that process."""
    id: int
    type: str
    pid: int

    def __repr__(self) -> str:
        return f"<{self.type} in pack process {self.pid}>"


_PLAIN = (str, bytes, bool, int, float, type(None))


def to_worker(value: Any, refs: Any, create) -> Any:
    """ComfyUI -> worker: data as it is (pickled, so copied); engine objects as handles."""
    from comfy_api._impl import _runtime

    if isinstance(value, _PLAIN) or _is_tensor(value) or isinstance(value, (_runtime.Ref, Closure, HostValue, PackValue)):
        return value
    if type(value).__module__ == "numpy":  # numpy arrays cross as data, both ways
        return value
    if isinstance(value, dict):  # dict subclasses (SavedResult...) cross as plain dicts
        return {k: to_worker(v, refs, create) for k, v in value.items()}
    if type(value) in (list, tuple):
        return type(value)(to_worker(v, refs, create) for v in value)
    return create(value)


def from_worker(value: Any, functions: Optional[dict] = None, path: str = "value") -> Any:
    """Worker -> ComfyUI: only data, handles ComfyUI gave out, and (as call arguments) functions."""
    from comfy_api._impl import _runtime

    if isinstance(value, _PLAIN) or _is_tensor(value) or isinstance(value, (_runtime.Ref, HostValue, PackValue)):
        return value
    if type(value).__module__ == "numpy":  # numpy arrays cross as data
        return value
    if functions is not None and _is_function(value):
        cid = next(_closure_ids)
        functions[cid] = value
        return Closure(cid)
    if isinstance(value, dict):  # dict subclasses (SavedResult...) cross as plain dicts
        return {k: from_worker(v, functions, f"{path}[{k!r}]") for k, v in value.items()}
    if type(value) in (list, tuple):
        return type(value)(from_worker(v, functions, f"{path}[{i}]") for i, v in enumerate(value))
    raise TypeError(f"{path}: a {type(value).__name__} can't leave a pack's process; "
                    "return tensors, plain data, or engine objects it was given")


_closure_ids = itertools.count(1)
_pack_value_ids = itertools.count(1)
_pack_values: dict[int, Any] = {}  # in a pack's process: the objects its PackValues stand for


def keep(value: Any) -> Any:
    """A node output, worker -> ComfyUI: data as ``from_worker`` sends it; anything else stays here
    as a ``PackValue``."""
    import os
    from comfy_api._impl import _runtime

    if isinstance(value, _PLAIN) or _is_tensor(value) or isinstance(value, (_runtime.Ref, HostValue, PackValue)):
        return value
    if type(value).__module__ == "numpy":
        return value
    if type(value).__name__ == "Size" and type(value).__module__.startswith("torch"):
        return tuple(value)  # a tensor's shape is plain data
    if type(value) is dict:
        return {k: keep(v) for k, v in value.items()}
    if type(value) in (list, tuple):
        return type(value)(keep(v) for v in value)
    pid = next(_pack_value_ids)
    _pack_values[pid] = value
    return PackValue(pid, type(value).__name__, os.getpid())


def restore(value: Any) -> Any:
    """Node inputs in a pack's process: PackValues back to the objects they stand for."""
    import os
    if isinstance(value, PackValue):
        if value.pid != os.getpid() or value.id not in _pack_values:
            raise ValueError(f"{value!r} was made by a pack process that has stopped, or by another pack; "
                             "run the node that made it again")
        return _pack_values[value.id]
    if type(value) is dict:
        return {k: restore(v) for k, v in value.items()}
    if type(value) in (list, tuple):
        return type(value)(restore(v) for v in value)
    return value
