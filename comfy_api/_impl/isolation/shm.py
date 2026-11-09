# Lifted unchanged from the secure overlay (comfy_secure_nodes/transport/shm.py), its tested
# shared-memory carrier, so pack isolation moves large tensors and arrays the same way.
"""Rung-2 out-of-band tensor channel: shared memory (CPU).

Big buffers must NEVER cross the JSON-RPC wire. When a node holds the ``raw``
capability and materializes a tensor, the bytes travel through POSIX shared
memory and only a tiny handle ({name, shape, dtype, nbytes}) crosses the wire.

CPU today (open-Q #7 / burndown Workstream T): there is one host→shm copy and
one shm→guest copy, but NOTHING is pickled and no buffer touches the socket —
that satisfies the "no full-buffer copy in the RPC channel" acceptance
criterion. It does NOT satisfy the static/constant no-copy contract in
``docs/resource-model.md``. Persistent shared CPU tensor backing remains work
to do; the experimental CUDA VMM path already maps GPU storage without copying,
while the legacy Torch-private CUDA importer still clones.

Lifetime is exactly-once ownership transfer, with a host backstop:
  * the producer writes and unregisters the segment from multiprocessing's
    resource_tracker, relinquishing ownership;
  * the CONSUMER copies out and unlinks (``read_tensor(consume=True)``);
  * the host releases anything still outstanding when a dispatch ends
    (``unlink``), because there is no refcount and a consumer that never reads
    would otherwise leave the segment with no owner at all.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import mmap
from multiprocessing import resource_tracker, shared_memory
import os
import re
import secrets
import stat
from typing import Any

import numpy as np

_DIRECTORY = ContextVar("guest_buffer_directory", default=os.environ.get("COMFY_GUEST_SHM_ROOT"))


@contextmanager
def directory(path):
    """Select one session's carrier without sharing /dev/shm between realms."""
    token = _DIRECTORY.set(path)
    try:
        yield
    finally:
        _DIRECTORY.reset(token)


def _segment_name(name):
    if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", name) is None:
        raise ValueError("invalid private buffer name")
    return name


class _FileSegment:
    """A POSIX-style buffer backed by a file in the guest's private mount.

    These are not global shm names. O_NOFOLLOW also prevents a guest-created
    symlink from becoming a file-open primitive in the controller.
    """
    _track = False

    def __init__(self, root, name=None, *, create=False, size=0):
        self.name = _segment_name(name or ("buf_" + secrets.token_hex(12)))
        self.root = root
        directory_fd = os.open(root, os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            flags = os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC
            if create:
                flags |= os.O_CREAT | os.O_EXCL
            fd = os.open(self.name, flags, 0o600, dir_fd=directory_fd)
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise ValueError("private buffer is not a regular file")
                if create:
                    os.ftruncate(fd, size)
                self._mapping = mmap.mmap(fd, 0)
            except BaseException:
                if create:
                    os.unlink(self.name, dir_fd=directory_fd)
                raise
            finally:
                os.close(fd)
        finally:
            os.close(directory_fd)
        self.buf = memoryview(self._mapping)

    def close(self):
        self.buf.release()
        self._mapping.close()

    def unlink(self):
        fd = os.open(self.root, os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            os.unlink(self.name, dir_fd=fd)
        finally:
            os.close(fd)


def _open_segment(name=None, *, create=False, size=0, **kwargs):
    root = _DIRECTORY.get()
    if root is not None:
        return _FileSegment(root, name, create=create, size=size)
    return shared_memory.SharedMemory(name=name, create=create, size=size, **kwargs)

try:  # POSIX only; absent on Windows, where `unlink` falls back to attach.
    from _posixshmem import shm_unlink as _shm_unlink
except ImportError:  # pragma: no cover - platform dependent
    _shm_unlink = None


def _relinquish_tracker(segment) -> None:
    if not getattr(shared_memory, "_USE_POSIX", False):
        return
    if getattr(segment, "_track", True) is False:
        return
    try:
        resource_tracker.unregister(
            segment._name, "shared_memory")  # type: ignore[attr-defined]
    except Exception:
        pass
    if hasattr(segment, "_track"):
        segment._track = False  # type: ignore[attr-defined]


def _open_untracked(name: str):
    try:
        return _open_segment(name=name, track=False)
    except TypeError:
        segment = _open_segment(name=name)
        _relinquish_tracker(segment)
        return segment


def _np(tensor: Any) -> np.ndarray:
    # torch tensor -> contiguous cpu numpy (no torch import needed here)
    arr = tensor.detach().cpu().contiguous().numpy() if hasattr(tensor, "detach") \
        else np.ascontiguousarray(tensor)
    # `np.ascontiguousarray` accepts ANYTHING. Handed a dict — a LATENT is
    # `{"samples": Tensor}` — it returns a 0-d array of dtype=object holding a
    # POINTER into this process's heap, and the segment is written without
    # error: measured as shape [1], dtype 'object', nbytes 8. The consumer then
    # fails far away with "can't convert np.ndarray of type numpy.object_", or
    # worse, on a same-process path appears to work.
    #
    # A channel that silently transports a pointer between address spaces is the
    # most dangerous failure this transport can have, so it is refused at the
    # point the mistake is made, naming the object.
    if arr.dtype == np.dtype("object"):
        raise TypeError(
            f"cannot put a {type(tensor).__name__} through the buffer channel: "
            f"it is not array data, so it would cross as a pointer into this "
            f"process's memory. Composite values (a LATENT dict, a conditioning "
            f"list) must be decomposed into their tensors by the caller."
        )
    return arr


def reserve(name: str, size: int, *, root=None) -> None:
    """Create a named segment for a peer and relinquish tracker ownership."""
    if root is None:
        seg = _open_segment(create=True, name=name, size=max(int(size), 1))
    else:
        seg = _FileSegment(root, name, create=True, size=max(int(size), 1))
    _relinquish_tracker(seg)
    seg.close()


def write_tensor(tensor: Any, name: str | None = None, *, create: bool = True) -> dict:
    """Copy ``tensor`` into a fresh shm segment; return the wire handle.

    ``name`` lets the caller supply a HOST-ISSUED segment name. A guest must
    use one: the host rejects a store whose segment it did not name, because a
    freely-chosen name is an arbitrary-shm-object primitive (see
    ``host._h_store``).
    """
    arr = _np(tensor)
    segment = _open_segment(
        create=create, name=name, size=max(arr.nbytes, 1) if create else 0)
    segment_name = segment.name
    available = segment.buf.nbytes
    try:
        if available < max(arr.nbytes, 1):
            raise ValueError(
                f"reserved segment holds {available} bytes but tensor needs "
                f"{arr.nbytes}")
        view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=segment.buf)
        view[...] = arr
        handle = {
            "kind": "shm",
            "handle": segment_name,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "nbytes": int(arr.nbytes),
            "strides": [s // arr.itemsize for s in arr.strides],
            "offset": 0,
        }
    except BaseException:
        _relinquish_tracker(segment)
        segment.close()
        if create:
            unlink(segment_name)
        raise
    _relinquish_tracker(segment)
    segment.close()
    return handle


def read_array(handle: dict, *, copy: bool = False):
    """Map a handle to a numpy array. Framework-free — the escape hatch.

    Default is a genuine **borrow**: a zero-copy view onto the shared segment,
    so writes are visible to every other holder. That is the property the whole
    channel exists for; copying by default would quietly turn a shared buffer
    into a private one and nothing would fail loudly.

    A plain ndarray cannot own the mapping (no ``__dict__``), and if the
    SharedMemory object is collected the mapping closes underneath the view —
    surfacing later as a segfault or silent zeros. Hence the thin subclass.
    """
    class _Mapped(np.ndarray):
        _seg = None

    seg = _open_segment(name=_name(handle))
    # A reader is a borrower, not the owner: it must not unlink at exit.
    _relinquish_tracker(seg)
    arr = np.ndarray(tuple(handle["shape"]), dtype=np.dtype(handle["dtype"]),
                     buffer=seg.buf, offset=handle.get("offset", 0))
    if copy:
        out = arr.copy()
        seg.close()
        return out
    view = arr.view(_Mapped)
    view._seg = seg
    return view


def read_tensor(handle: dict, *, consume: bool = True):
    """Map a handle back to an owned torch tensor (copied out of shm)."""
    import torch

    shm = _open_segment(name=_name(handle))
    try:
        # The descriptor decides how many bytes are read, and on the store path
        # it comes from the GUEST. numpy will happily map a shape/dtype larger
        # than the segment and read past it, so the host must check rather than
        # trust: an oversized descriptor is an out-of-bounds read inside the
        # trusted process.
        dtype = np.dtype(handle["dtype"])
        if dtype.hasobject:
            raise ValueError(f"refused object dtype {dtype!r} from a handle")
        shape = tuple(int(x) for x in handle["shape"])
        need = int(np.prod(shape)) * dtype.itemsize if shape else dtype.itemsize
        if need > shm.buf.nbytes:
            raise ValueError(
                f"handle describes {need} bytes but its segment holds "
                f"{shm.buf.nbytes}; refusing an out-of-bounds read")
        arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        out = torch.from_numpy(arr.copy())  # own the memory so we can unlink
    finally:
        shm.close()
        if consume:
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
        else:
            _relinquish_tracker(shm)
    return out


def unlink(handle: Any) -> bool:
    """Release a segment by handle or name. Idempotent; True if still present.

    This channel transfers ownership exactly once: the producer unregisters from
    the resource_tracker and the consumer unlinks after copying out. There is no
    refcount, so a consumer that never reads means nobody unlinks — which is why
    ownership belongs to the party that outlives the transfer, the host.

    The host calls this for every segment still outstanding when a dispatch
    ends, making the contract "released when no longer used" rather than
    "released if the guest behaves". A guest that crashes or times out between
    export and read is the case it covers.
    """
    name = handle if isinstance(handle, str) else _name(handle)
    root = _DIRECTORY.get()
    if root is not None:
        fd = os.open(root, os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            try:
                os.unlink(_segment_name(name), dir_fd=fd)
                return True
            except FileNotFoundError:
                return False
        finally:
            os.close(fd)

    # Unlink by NAME, without attaching. `shm_unlink` is the POSIX primitive for
    # exactly this: no mapping, no resource_tracker involvement.
    #
    # Attaching first would be wrong on both counts. It maps a segment purely to
    # destroy it, and a default SharedMemory attachment registers the observer
    # as an owner. Raw unlink avoids creating a tracker obligation merely to
    # release a name whose producer already relinquished that obligation.
    if _shm_unlink is not None:
        target = name if name.startswith("/") else "/" + name
        try:
            _shm_unlink(target)
            return True
        except FileNotFoundError:
            return False  # already consumed — the ordinary path

    # Windows has no POSIX shm and no resource_tracker for it; releasing there
    # is dropping the last handle, so attach-and-close is the correct analogue.
    try:
        seg = shared_memory.SharedMemory(name=name)
    except FileNotFoundError:
        return False
    seg.close()
    try:
        seg.unlink()
    except FileNotFoundError:
        return False
    return True


def _name(handle: dict) -> str:
    """Accept the current descriptor, and the pre-unification `__shm__` form."""
    return handle.get("handle") or handle["__shm__"]


def exists(name: str) -> bool:
    """Observe a segment name without acquiring cleanup ownership."""
    try:
        segment = _open_untracked(name)
    except FileNotFoundError:
        return False
    segment.close()
    return True


def is_handle(v: Any) -> bool:
    return isinstance(v, dict) and (
        v.get("kind") == "shm" or "handle" in v or "__shm__" in v)
