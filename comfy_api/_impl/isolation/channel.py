"""The channel: length-prefixed pickles over a socket, with strictly nested requests.

Large tensors and numpy arrays (``SHM_MIN_BYTES`` and up) don't go through the socket when both
sides can see the same shared memory (probed when they connect, ``probe_shared_memory``): they
cross as segments, written once by the sender and copied out and unlinked by the receiver as it
unpickles, the secure overlay's transfer rule (``shm.py`` is its carrier). Small values, and every
value on a channel without shared memory (a pack on another machine), are pickled inline.
"""
from __future__ import annotations

import io
import itertools
import os
import pickle
import secrets
import socket
import struct
import threading
import traceback
from typing import Any

SHM_MIN_BYTES = 64 << 10
_segments = itertools.count(1)
_PREFIX = f"cpack_{os.getpid()}_{secrets.token_hex(4)}_"
#: torch dtypes numpy has no type for: they cross as a same-size integer view
_VIEW_AS = {"bfloat16": "int16", "float8_e4m3fn": "uint8", "float8_e5m2": "uint8",
            "float8_e4m3fnuz": "uint8", "float8_e5m2fnuz": "uint8"}


def _layout_permutation(tensor: Any) -> list[int] | None:
    """The dimension order that makes a non-contiguous tensor contiguous without a copy, when its layout is a
    plain permutation (no gaps, no overlap); None for a contiguous tensor or any other layout."""
    if tensor.dim() < 2 or tensor.is_contiguous():
        return None
    perm = sorted(range(tensor.dim()), key=lambda d: (-tensor.stride(d), d))
    return perm if tensor.permute(perm).is_contiguous() else None


def segment_prefix(pid: int | None = None) -> str:
    """The names this process gives its segments (``sweep`` releases a dead peer's)."""
    return _PREFIX if pid is None else f"cpack_{pid}_"


def sweep(pid: int) -> int:
    """Unlink every segment a process that has gone made and nobody read. Linux lists them."""
    from . import shm
    root = "/dev/shm"
    if not os.path.isdir(root):
        return 0
    gone = 0
    for name in os.listdir(root):
        if name.startswith(f"cpack_{pid}_"):
            gone += bool(shm.unlink(name))
    return gone


class _Pickler(pickle.Pickler):
    def __init__(self, file: Any, channel: "Channel") -> None:
        super().__init__(file, protocol=pickle.HIGHEST_PROTOCOL)
        self.channel = channel

    def persistent_id(self, obj: Any) -> Any:
        if not self.channel.shared_memory:
            return None
        kind = type(obj)
        if kind.__name__ == "Tensor" and kind.__module__ == "torch" and str(obj.layout) == "torch.strided" \
                and obj.device.type != "meta":
            if obj.numel() * obj.element_size() < SHM_MIN_BYTES:
                return None
            return ("shm", self.channel._export_tensor(obj))
        if kind.__name__ == "ndarray" and kind.__module__ == "numpy" and not obj.dtype.hasobject:
            if obj.nbytes < SHM_MIN_BYTES:
                return None
            return ("shm", self.channel._export_array(obj))
        return None


class _Unpickler(pickle.Unpickler):
    def persistent_load(self, pid: Any) -> Any:
        tag, handle = pid
        if tag != "shm":
            raise pickle.UnpicklingError(f"unknown persistent id {tag!r}")
        return _import(handle)


def _import(handle: dict) -> Any:
    from . import shm
    if handle.get("framework") == "numpy":
        try:
            return shm.read_array(handle, copy=True)
        finally:
            shm.unlink(handle)
    import torch
    tensor = shm.read_tensor(handle, consume=True)
    if handle.get("torch_dtype") and str(tensor.dtype).removeprefix("torch.") != handle["torch_dtype"]:
        tensor = tensor.view(getattr(torch, handle["torch_dtype"]))
    perm = handle.get("permute")
    if perm is not None:  # the sender's layout back: it sent the tensor permuted to contiguous order
        if not (isinstance(perm, list) and sorted(perm) == list(range(tensor.dim()))
                and all(type(p) is int for p in perm)):
            raise ValueError(f"refused a tensor layout {perm!r} for {tensor.dim()} dimensions")
        inverse = [0] * len(perm)
        for position, dim in enumerate(perm):
            inverse[dim] = position
        tensor = tensor.permute(inverse)
    want = str(handle.get("device") or "cpu")  # restore the sender's placement, as the overlay does
    if want.startswith("cuda") and torch.cuda.is_available():
        tensor = tensor.to(want)
    elif want.startswith("mps") and torch.backends.mps.is_available():
        tensor = tensor.to("mps")
    if handle.get("requires_grad"):
        tensor.requires_grad_(True)
    return tensor


def probe_shared_memory() -> tuple[str, Any]:
    """A segment for the other side to look for: (name, release)."""
    from . import shm
    name = f"{_PREFIX}probe{next(_segments)}"
    shm.reserve(name, 1)
    return name, (lambda: shm.unlink(name))


def can_see(name: Any) -> bool:
    """Whether this process sees the other side's probe segment."""
    from . import shm
    try:
        return isinstance(name, str) and shm.exists(name)
    except Exception:
        return False


class WorkerError(RuntimeError):
    """The worker process failed or went away."""


class RemoteError(RuntimeError):
    def __init__(self, kind: str, message: str, trace: str = "") -> None:
        super().__init__(f"{kind}: {message}")
        self.kind, self.message, self.trace = kind, message, trace


_RAISABLE = {"TypeError": TypeError, "ValueError": ValueError, "KeyError": KeyError,
             "PermissionError": PermissionError, "NotImplementedError": NotImplementedError}


def _raise_remote(error: dict) -> None:
    if error["kind"] == "CallNotSupported":
        from comfy_api._impl._runtime import CallNotSupported
        raise CallNotSupported(error["message"])
    import builtins
    cls = _RAISABLE.get(error["kind"]) or getattr(builtins, error["kind"], None)
    if isinstance(cls, type) and issubclass(cls, Exception) and cls not in (SystemExit, KeyboardInterrupt):
        raise cls(error["message"])
    raise RemoteError(error["kind"], error["message"], error.get("trace", ""))


def _describe(exc: BaseException) -> dict:
    return {"kind": type(exc).__name__, "message": str(exc), "trace": traceback.format_exc()}


class Channel:
    """Either side of the socket. A side waiting for a reply serves the other side's nested
    requests in the meantime (an engine call during a node, a sampling function during that
    call), so the conversation can nest to any depth without threads waiting on each other."""

    def __init__(self, sock: socket.socket, handler) -> None:
        self.sock = sock
        self.handler = handler  # handler(message) -> result, for requests from the other side
        self._ids = itertools.count(1)
        self._send_lock = threading.Lock()
        self._recv_lock = threading.Lock()
        self.shared_memory = False  # set once both sides have seen the same segments
        self._written: list[str] = []  # segments sent and maybe not read yet

    def _export_tensor(self, tensor: Any) -> dict:
        from . import shm
        import torch
        dtype = str(tensor.dtype).removeprefix("torch.")
        data = tensor.detach()
        # a permuted layout (ComfyUI's movedim views: images channels-last) crosses as it is, so the other side
        # gets the same strides: kernels that pick their algorithm by memory format then give the same result
        # (VAE encode: a contiguous copy of the same pixels encoded differently, max 0.29 in the latent)
        perm = _layout_permutation(data)
        if perm is not None:
            data = data.permute(perm)
        if dtype in _VIEW_AS:
            data = data.contiguous().view(getattr(torch, _VIEW_AS[dtype]))
        handle = shm.write_tensor(data, name=self._name())
        handle.update(framework="torch", torch_dtype=dtype, device=str(tensor.device),
                      requires_grad=bool(tensor.requires_grad))
        if perm is not None:
            handle["permute"] = perm
        return handle

    def _export_array(self, array: Any) -> dict:
        from . import shm
        handle = shm.write_tensor(array, name=self._name())
        handle["framework"] = "numpy"
        return handle

    def _name(self) -> str:
        name = f"{_PREFIX}{next(_segments)}"
        self._written.append(name)
        return name

    def release_unread(self) -> None:
        """Unlink segments this side sent that the other side never read (it failed first)."""
        from . import shm
        written, self._written = self._written, []
        for name in written:
            shm.unlink(name)

    def _send(self, message: dict) -> None:
        buffer = io.BytesIO()
        buffer.write(b"\0" * 8)
        _Pickler(buffer, self).dump(message)
        data = buffer.getbuffer()
        struct.pack_into("!Q", data, 0, len(data) - 8)
        with self._send_lock:
            self.sock.sendall(data)

    def _recv(self) -> dict:
        with self._recv_lock:
            header = self._read(8)
            (size,) = struct.unpack("!Q", header)
            return _Unpickler(io.BytesIO(self._read(size))).load()

    def _read(self, size: int) -> bytes:
        chunks, remaining = [], size
        while remaining:
            chunk = self.sock.recv(min(remaining, 1 << 22))
            if not chunk:
                raise WorkerError("the other process closed the connection")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def request(self, message: dict) -> Any:
        rid = next(self._ids)
        mark = len(self._written)
        try:
            self._send({**message, "id": rid})
            return self._await_reply(rid, mark)
        except BaseException:
            self._release_from(mark)  # e.g. the other side went away before reading what we sent
            raise

    def _await_reply(self, rid: int, mark: int) -> Any:
        while True:
            incoming = self._recv()
            if incoming.get("reply") == rid:
                self._release_from(mark)  # the other side read them (or failed before it could)
                if "error" in incoming:
                    _raise_remote(incoming["error"])
                return incoming["result"]
            if "reply" in incoming:
                raise WorkerError(f"out-of-order reply {incoming['reply']} while waiting for {rid}")
            self.serve(incoming)

    def serve(self, message: dict) -> None:
        try:
            result = self.handler(message)
            reply = {"reply": message["id"], "result": result}
        except BaseException as exc:  # reported to the requester, which raises it
            reply = {"reply": message["id"], "error": _describe(exc)}
        self._send(reply)

    def _release_from(self, mark: int) -> None:
        from . import shm
        for name in self._written[mark:]:
            shm.unlink(name)  # already gone when the other side read it: unlink is idempotent
        del self._written[mark:]

    def serve_forever(self) -> None:
        while True:
            message = self._recv()
            self.release_unread()  # the requester read our last reply before it sent this
            if message.get("op") == "shutdown":
                return
            self.serve(message)
