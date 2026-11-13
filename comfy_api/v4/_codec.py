"""What crosses between ComfyUI and a pack process, and how.

A value arrives as a copy of the same type and value when it is:

- `None`, `bool`, `int`, `float`, `str`, `bytes`, `bytearray`, or a `list`, `tuple`, `dict`, `set`,
  `frozenset` or `OrderedDict` of crossing values (exactly these types, not subclasses);
- a tensor: dtype, shape, values, device, `requires_grad` and the `Parameter` class (a view or a
  transposed tensor arrives contiguous). It is an inference tensor when the receiving side decodes
  it in inference mode, which a call does in the mode of the thread that made it;
- a numpy array or numpy scalar of numbers, a `torch.Size`, `torch.device` or `torch.dtype`, a
  `pathlib` path;
- one of the value classes `_allowed` names (ComfyUI's V3 values, `ExecutionBlocker`, PIL images,
  `Fraction`, `BytesIO`), or a member of one of ComfyUI's enums;
- a V3 value that holds a file (`VideoFromFile`, `File3D`): it arrives holding the file's bytes,
  also when it held a path. The receiving side refuses one that holds a path, so ComfyUI never
  opens a path a pack chose.

A member of a pack's own enum that is also a `str` or an `int` (`StrEnum`, `IntEnum`, `str, Enum`)
arrives as its value, which it equals: ComfyUI's own schemas hold such members. Anything else stays
in the process that made it, such as a subclass of a type above, a pack's own class or function,
another enum's member, a sparse tensor, a `datetime` or a `complex`. The other side gets a `Remote`
that names its type: calling it is a round trip, any other use raises, and sent back it is the
object itself. The receiving side accepts only the classes named here, so a message cannot make
ComfyUI import a pack's module or build an object of any other class. Of those it builds only what
pickle makes of a value: an object made without its constructor and then given its attributes
once, an enum's member, an `OrderedDict` or a `Fraction` of whole numbers. A message cannot call a
constructor, nor change a class or an object that it did not make.

A tensor or numpy array arrives as a copy, in memory of the receiving process's own: a change made
in place on one side is not seen on the other. Its bytes are not pickled: `dumps` returns them as
blocks, views of the sender's memory to write after the message, and `loads` makes each tensor or
array on the block it was read into. A list, dict or value-class object that is twice in one
message arrives as one object, as pickle keeps it; a tensor, array or object that stays arrives
once for each place it is in.
"""

import enum
import fractions
import functools
import pathlib
import pickle
import re
import sys
import weakref
from collections import OrderedDict
from io import BytesIO

import numpy
import torch
from PIL import Image

PLAIN = frozenset(
    {
        type(None),
        bool,
        int,
        float,
        str,
        bytes,
        bytearray,
        list,
        tuple,
        dict,
        set,
        frozenset,
    }
)
_MODULES = ("comfy_api.latest._io", "comfy_api.latest._ui")
_CLASSES = frozenset(
    {
        ("comfy_execution.graph_utils", "ExecutionBlocker"),
        ("PIL.Image", "Image"),
        ("collections", "OrderedDict"),
        ("fractions", "Fraction"),
        ("_io", "BytesIO"),
    }
)
_PATHS = {
    cls.__name__: cls
    for cls in (
        pathlib.PosixPath,
        pathlib.WindowsPath,
        pathlib.PurePosixPath,
        pathlib.PureWindowsPath,
    )
}
NUMBERS = "biufc"


def _allowed(module: str, qualname: str) -> bool:
    return (
        (module, qualname) in _CLASSES
        or any(module == m or module.startswith(m + ".") for m in _MODULES)
        or (module, qualname) in _value_classes()
    )


@functools.cache
def _value_classes() -> frozenset[tuple[str, str]]:
    """The value classes V3 publishes, apart from those of a file: nothing else of their
    modules, whose other classes hold memory that is not Python's."""
    from comfy_api.latest import _input_impl, _util

    published = [getattr(m, name) for m in (_input_impl, _util) for name in m.__all__]
    return frozenset(
        (cls.__module__, cls.__qualname__)
        for cls in published
        if isinstance(cls, type) and cls not in _file_classes().values()
    )


@functools.cache
def _file_classes() -> dict[str, type]:
    """The value classes that hold a file, as a path or as its bytes."""
    from comfy_api.latest import InputImpl, Types

    return {"VideoFromFile": InputImpl.VideoFromFile, "File3D": Types.File3D}


def _file_source(value):
    if isinstance(value, _file_classes()["File3D"]):
        return value.get_source()
    return value.get_stream_source()


def _file_state(value) -> dict:
    """The value's attributes, with the path of its file replaced by the file's bytes: the other
    process may not be able to open the path, and must not open one this process chose."""
    source = _file_source(value)
    if not isinstance(source, str):
        return dict(vars(value))
    with open(source, "rb") as file:
        data = BytesIO(file.read())
    return {
        name: data if held is source else held for name, held in vars(value).items()
    }


@functools.cache
def _file_attributes(name: str) -> frozenset[str]:
    """The attributes a value of the file class `name` has, as its constructor sets them."""
    return frozenset(vars(_file_classes()[name](BytesIO())))


def _load_file_value(name: str, state: dict):
    cls = _file_classes().get(name)
    if (
        cls is None
        or type(state) is not dict
        or not state.keys() <= _file_attributes(name)
    ):
        raise pickle.UnpicklingError(f"{name!r} is not a value that holds a file")
    value = cls.__new__(cls)
    vars(value).update(state)
    if type(_file_source(value)) is not BytesIO:
        raise pickle.UnpicklingError(
            f"a {name} crosses with its file's bytes, not a path"
        )
    # ComfyUI names the files it saves of a 3D file with its format.
    file_format = getattr(value, "format", "")
    if not (type(file_format) is str and (file_format == "" or file_format.isalnum())):
        raise pickle.UnpicklingError(f"{file_format!r} is not the format of a file")
    return value


def _named(cls: type) -> bool:
    """Whether `cls` is found again from its module and qualified name."""
    found = sys.modules.get(cls.__module__)
    for part in cls.__qualname__.split("."):
        found = getattr(found, part, None)
    return found is cls


class Remote:
    """An object that stays in the other process. It can be called (a round trip) or sent back.

    The methods and values its side offered when it crossed (`side.offers`) can be used too:
    each method call is a round trip. Given `release`, it is called with the key once this
    process no longer holds the object. A copy of it is itself.
    """

    def __init__(
        self, side, key, type_name: str, methods: tuple, values: dict, release=None
    ):
        self._side, self._key, self._type = side, key, type_name
        self._methods, self._values = methods, values
        self._release = (
            None if release is None else weakref.finalize(self, release, key)
        )

    def __call__(self, *args, **kwargs):
        return self._request(("call", self._key), args, kwargs)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._values:
            return self._values[name]
        if name in self._methods:

            def method(*args, **kwargs):
                return self._request(("call", self._key, name), args, kwargs)

            return method
        raise AttributeError(self._refusal())

    def __repr__(self) -> str:
        return f"<{self._type} in {self._side.peer}>"

    def __format__(self, spec: str) -> str:
        # `str()` stays the repr, which ComfyUI's error report shows for a node's inputs.
        raise TypeError(self._refusal())

    def _refusal(self) -> str:
        where = f"{self._type} lives in {self._side.peer}"
        return f"{where}; this process can only call it or pass it on"

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def _request(self, target: tuple, args: tuple, kwargs: dict):
        if self._release is not None and not self._release.alive:
            # Garbage being torn down (a `__del__`) can call a part of it that the collector
            # has already released: the other side has dropped it, and nothing uses the result.
            return None
        return self._side.request(target, args, kwargs)


def _tensor(tensor: torch.Tensor, blocks: list) -> tuple:
    grad = (tensor.requires_grad, type(tensor) is torch.nn.Parameter)
    device = str(tensor.device)
    tensor = tensor.detach().cpu().resolve_conj().resolve_neg().contiguous()
    # Its memory as one row: a contiguous tensor's dimension of size 1 may have any stride.
    row = tensor.as_strided((tensor.numel(),), (1,))
    blocks.append(memoryview(row.view(torch.uint8).numpy()))
    dtype = str(tensor.dtype).removeprefix("torch.")
    return ("tensor", dtype, tuple(tensor.shape), device, *grad)


def _array(array: numpy.ndarray, blocks: list) -> tuple:
    contiguous = numpy.ascontiguousarray(array).reshape(-1)
    blocks.append(memoryview(contiguous.view(numpy.uint8)))
    return ("numpy", array.dtype.str, array.shape)


def _data(obj, cls: type, blocks: list) -> tuple | None:
    """The reference that sends `obj` by value, if it is a tensor, array or other data."""
    if (
        (cls is torch.Tensor or cls is torch.nn.Parameter)
        and obj.layout is torch.strided
        and not (obj.is_quantized or obj.is_nested)
    ):
        return _tensor(obj, blocks)
    if cls is numpy.ndarray and obj.dtype.kind in NUMBERS:
        return _array(obj, blocks)
    if isinstance(obj, numpy.generic) and obj.dtype.kind in NUMBERS:
        return ("numpy scalar", obj.dtype.str, obj.tobytes())
    if cls is torch.Size:
        return ("size", tuple(obj))
    if _PATHS.get(cls.__name__) is cls:
        return ("path", cls.__name__, str(obj))
    if cls is torch.device:
        return ("device", str(obj))
    if cls is torch.dtype:
        return ("dtype", str(obj).removeprefix("torch."))
    return None


def _dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name)
    if not isinstance(dtype, torch.dtype):
        raise pickle.UnpicklingError(f"{dtype!r} is not a dtype")
    return dtype


def _number(name: str) -> numpy.dtype:
    dtype = numpy.dtype(name)
    if dtype.kind not in NUMBERS:
        raise pickle.UnpicklingError(f"{dtype} is not a number")
    return dtype


def _load_tensor(
    block: memoryview | bytearray,
    dtype: str,
    shape: tuple,
    device: str,
    requires_grad: bool,
    parameter: bool,
) -> torch.Tensor:
    # `frombuffer` refuses an empty buffer.
    raw = (
        torch.frombuffer(block, dtype=torch.uint8)
        if block
        else torch.empty(0, dtype=torch.uint8)
    )
    tensor = raw.view(_dtype(dtype)).reshape(shape)
    if device != "cpu":
        tensor = tensor.to(device)
    if parameter:
        return torch.nn.Parameter(tensor, requires_grad=requires_grad)
    return tensor.requires_grad_(requires_grad)


def _io_class(cls: type) -> tuple | None:
    """Name an io type's class that its qualified name does not find.

    `comfytype` copies each io type's Input and Output classes, and `io.Custom` makes io types
    at run time, so these are named by their io type and rebuilt with `io.Custom` if needed.
    """
    from comfy_api.latest import _io

    if issubclass(cls, _io._ComfyType):
        parent, part = cls, None
    elif issubclass(cls, _io._IO_V3):
        parent = cls.Parent
        part = next(
            (p for p in ("Input", "Output") if getattr(parent, p, None) is cls), None
        )
        if part is None:
            return None
    else:
        return None
    return ("io", parent.__qualname__ if _named(parent) else None, parent.io_type, part)


def _load_io_class(side, qualname: str | None, io_type: str, part: str | None) -> type:
    from comfy_api.latest import _io, io

    if qualname is None:
        if io_type not in side.io_types:
            side.io_types[io_type] = io.Custom(io_type)
        parent = side.io_types[io_type]
    else:
        parent = _io
        for name in qualname.split("."):
            parent = getattr(parent, name)
        if not (isinstance(parent, type) and issubclass(parent, _io._ComfyType)):
            raise pickle.UnpicklingError(f"{qualname} is not an io type")
    if part not in (None, "Input", "Output"):
        raise pickle.UnpicklingError(f"{part!r} is not a part of an io type")
    return getattr(parent, part) if part else parent


class _Pickler(pickle.Pickler):
    def __init__(self, file, side, chain):
        super().__init__(file, protocol=5)
        self.side, self.chain = side, chain
        self.blocks: list[memoryview] = []
        # What `side` kept for this message; it lets go of it if the message is not made.
        self.kept = []

    def persistent_id(self, obj):
        cls = type(obj)
        if cls in PLAIN:
            return None
        data = _data(obj, cls, self.blocks)
        if data is not None:
            return data
        if cls is Remote:
            return self._sent_back(obj)
        if isinstance(obj, type):
            if _named(obj) and _allowed(obj.__module__, obj.__qualname__):
                return None
            named = _io_class(obj)
            if named is not None:
                return named
        elif _allowed(cls.__module__, cls.__qualname__):
            return None
        elif cls in _file_classes().values():
            return ("file", cls.__name__, _file_state(obj))
        elif isinstance(obj, enum.Enum) and isinstance(obj, (str, int)):
            return ("value", obj.value)
        key = self.side.keep(obj, self.chain)
        self.kept.append(key)
        return ("object", key, cls.__qualname__, *self.side.offers(obj))

    def _sent_back(self, remote: Remote) -> tuple:
        if remote._side is not self.side:
            raise TypeError(
                f"{remote._type} lives in {remote._side.peer}; only that pack's nodes can use it"
            )
        self.side.hold(remote, self.chain)
        return ("yours", remote._key)


def _whole_ratio(args: tuple) -> bool:
    """Whether `args` are what pickle gives `Fraction`: numerator and denominator, or before
    Python 3.12 their text. Text with an exponent would make `Fraction` compute the power."""
    if len(args) == 2:
        return all(type(arg) is int for arg in args)
    return (
        len(args) == 1
        and type(args[0]) is str
        and re.fullmatch(r"-?[0-9]+(/[0-9]+)?", args[0]) is not None
    )


def _call(function, args: tuple):
    """What a message may call: an enum class for a member, `OrderedDict` for an empty one, and
    `Fraction` for a ratio of whole numbers."""
    if isinstance(function, enum.EnumMeta) and len(args) == 1:
        return function(*args)
    if function is OrderedDict and args == ():
        return OrderedDict()
    if function is fractions.Fraction and _whole_ratio(args):
        return fractions.Fraction(*args)
    raise pickle.UnpicklingError(f"refused to call {function!r}")


def _attributes(state) -> bool:
    return type(state) is dict and all(type(name) is str for name in state)


def _refuse(self) -> None:
    raise pickle.UnpicklingError("refused an instruction a value does not need")


# The containers a message may fill.
_FILLED = (list, dict, set, OrderedDict)


class _Unpickler(pickle._Unpickler):
    """Python's own unpickler, whose instructions can be refused one by one; its memo is a dict,
    so an index in a message allocates nothing."""

    dispatch = dict(pickle._Unpickler.dispatch)

    def __init__(self, file, side, chain, blocks):
        super().__init__(file)
        self.side, self.chain = side, chain
        self.blocks = iter(blocks)
        # The objects this message made, by id; and those of them with no attributes yet.
        self.made, self.unbuilt = {}, {}

    def find_class(self, module: str, name: str):
        if _allowed(module, name):
            cls = super().find_class(module, name)
            if isinstance(cls, type) and (cls.__module__, cls.__qualname__) == (
                module,
                name,
            ):
                return cls
        raise pickle.UnpicklingError(f"refused to load {module}.{name}")

    def load_reduce(self) -> None:
        args = self.stack.pop()
        self.stack[-1] = _call(self.stack[-1], args)

    def load_newobj(self) -> None:
        args = self.stack.pop()
        cls = self.stack[-1]
        if args != () or not isinstance(cls, type) or issubclass(cls, enum.Enum):
            raise pickle.UnpicklingError(f"refused to make {cls!r} from {args!r}")
        made = cls.__new__(cls)
        self.made[id(made)] = self.unbuilt[id(made)] = made
        self.stack[-1] = made

    def load_build(self) -> None:
        state = self.stack.pop()
        made = self.stack[-1]
        if self.unbuilt.pop(id(made), None) is not made:
            raise pickle.UnpicklingError(
                f"refused to change {type(made).__qualname__}: this message did not make it"
            )
        if type(made) is Image.Image and type(state) is list:
            made.__setstate__(state)
        elif type(made) is BytesIO and type(state) is tuple and state[2:] == (None,):
            made.__setstate__(state)
        elif type(made) not in (BytesIO, Image.Image) and _attributes(state):
            vars(made).update(state)
        else:
            raise pickle.UnpicklingError(
                f"refused the state of a {type(made).__qualname__}"
            )

    def _fill(self, container) -> None:
        """Refuse to add to anything but a plain container or an object this message made: a
        class, or one of this process's objects sent back, would be changed through its own
        `append`, `add` or item assignment."""
        if (
            type(container) not in _FILLED
            and self.made.get(id(container)) is not container
        ):
            raise pickle.UnpicklingError(
                f"refused to change {type(container).__qualname__}: this message did not make it"
            )

    def load_setitem(self) -> None:
        self._fill(self.stack[-3])
        super().load_setitem()

    def load_append(self) -> None:
        self._fill(self.stack[-2])
        super().load_append()

    def load_setitems(self) -> None:
        self._fill(self.metastack[-1][-1])
        super().load_setitems()

    def load_appends(self) -> None:
        self._fill(self.metastack[-1][-1])
        super().load_appends()

    def load_additems(self) -> None:
        self._fill(self.metastack[-1][-1])
        super().load_additems()

    def _block(self) -> memoryview | bytearray:
        block = next(self.blocks, None)
        if block is None:
            raise pickle.UnpicklingError("a tensor or array came without its bytes")
        return block

    def persistent_load(self, pid):
        match pid:
            case ("tensor", *tensor):
                return _load_tensor(self._block(), *tensor)
            case ("numpy", dtype, shape):
                return numpy.frombuffer(self._block(), dtype=_number(dtype)).reshape(
                    shape
                )
            case ("numpy scalar", dtype, data):
                return numpy.frombuffer(data, dtype=_number(dtype))[0]
            case ("size", dims):
                return torch.Size(dims)
            case ("path", name, path) if name in _PATHS:
                return _PATHS[name](path)
            case ("yours", key):
                return self.side.resolve(key, self.chain)
            case ("object", key, type_name, methods, values):
                return self.side.remote(key, type_name, methods, values)
            case ("device", name):
                return torch.device(name)
            case ("dtype", name):
                return _dtype(name)
            case ("value", value):
                return value
            case ("io", qualname, io_type, part):
                return _load_io_class(self.side, qualname, io_type, part)
            case ("file", name, state):
                return _load_file_value(name, state)
        raise pickle.UnpicklingError(f"unknown reference {pid[0]!r}")


_Unpickler.dispatch.update(
    {
        pickle.REDUCE[0]: _Unpickler.load_reduce,
        pickle.NEWOBJ[0]: _Unpickler.load_newobj,
        pickle.BUILD[0]: _Unpickler.load_build,
        pickle.SETITEM[0]: _Unpickler.load_setitem,
        pickle.SETITEMS[0]: _Unpickler.load_setitems,
        pickle.APPEND[0]: _Unpickler.load_append,
        pickle.APPENDS[0]: _Unpickler.load_appends,
        pickle.ADDITEMS[0]: _Unpickler.load_additems,
        **dict.fromkeys(
            (
                code[0]
                for code in (
                    pickle.NEWOBJ_EX,
                    pickle.INST,
                    pickle.OBJ,
                    pickle.EXT1,
                    pickle.EXT2,
                    pickle.EXT4,
                )
            ),
            _refuse,
        ),
    }
)


def dumps(value, side, chain) -> tuple[bytes, list[memoryview]]:
    """Pickle `value` for the other side, and give the blocks of its tensors and arrays, in order.

    Objects that stay here are kept by `side` in `chain`; if pickling fails, `side.forget` lets go
    of those it kept for this message.
    """
    buffer = BytesIO()
    pickler = _Pickler(buffer, side, chain)
    try:
        pickler.dump(value)
    except BaseException:
        side.forget(pickler.kept)
        raise
    return buffer.getvalue(), pickler.blocks


def loads(data: bytes, side, chain: int | None = None, blocks=()):
    """Unpickle what the other side sent in `chain`, making its tensors and arrays on `blocks`.
    Refuses every class `_allowed` does not name."""
    return _Unpickler(BytesIO(data), side, chain, blocks).load()
