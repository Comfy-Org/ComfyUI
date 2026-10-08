"""Spike: persist a prefix-cache entry (attention KV up to n plus the DeltaNet state at n) to disk.

File layout: an 8-byte header length, a JSON header, then each tensor's raw bytes at a 4096-aligned
offset, the file padded to a 4096 multiple. Reads bypass the OS file cache (FILE_FLAG_NO_BUFFERING
on Windows, O_DIRECT on Linux) into one page-aligned host buffer, so a restore is timed from the
drive and not from RAM; set COMFY_LLM_PREFIX_CACHE_DISK_DIRECT=0 for a plain buffered read.
"""
import hashlib
import json
import os
import struct
import sys
import time
from array import array

import torch

ALIGN = 4096
CHUNK = 64 << 20
DIRECT = os.environ.get("COMFY_LLM_PREFIX_CACHE_DISK_DIRECT", "1") == "1"
_DTYPES = {str(d): d for d in (torch.bfloat16, torch.float16, torch.float32)}


def entry_path(root, key, ids):
    h = hashlib.sha256(repr(key).encode())
    h.update(array("i", ids).tobytes())
    return os.path.join(root, h.hexdigest()[:32] + ".llmkv")


def _round_up(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


def _host_buffer(nbytes):
    # page-aligned: unbuffered reads need a sector-aligned destination; pinned when the driver allows it
    try:
        buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        pinned = True
    except RuntimeError:
        buf = torch.empty(nbytes + ALIGN, dtype=torch.uint8)
        pinned = False
    off = (-buf.data_ptr()) % ALIGN
    return buf[off:off + nbytes], pinned


def save(path, key, n, kv, states):
    """kv: {layer: (key, value)} device tensors, used up to n; states: {layer: (conv, recurrent)}. Returns stats."""
    t0 = time.perf_counter()
    tensors = [(f"k{i}", k[:, :, :n]) for i, (k, _) in kv.items()] + [(f"v{i}", v[:, :, :n]) for i, (_, v) in kv.items()]
    tensors += [(f"c{i}", c) for i, (c, _) in states.items()] + [(f"r{i}", r) for i, (_, r) in states.items()]
    meta, off = [], 0
    for name, t in tensors:
        meta.append({"name": name, "dtype": str(t.dtype), "shape": list(t.shape), "offset": off})
        off = _round_up(off + t.nbytes)
    header = json.dumps({"key": repr(key), "n": n, "tensors": meta}).encode()
    head_len = _round_up(8 + len(header))
    total = head_len + off
    buf, pinned = _host_buffer(total)
    buf[:8].copy_(torch.frombuffer(bytearray(struct.pack("<Q", len(header))), dtype=torch.uint8))
    buf[8:8 + len(header)].copy_(torch.frombuffer(bytearray(header), dtype=torch.uint8))
    for m, (_, t) in zip(meta, tensors):
        o = head_len + m["offset"]
        buf[o:o + t.nbytes].view(t.dtype).view(t.shape).copy_(t, non_blocking=pinned)
    if tensors[0][1].device.type == "cuda":
        torch.cuda.synchronize(tensors[0][1].device)
    t1 = time.perf_counter()
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(memoryview(buf.numpy()))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    t2 = time.perf_counter()
    return {"disk_save_d2h_s": round(t1 - t0, 3), "disk_save_write_s": round(t2 - t1, 3), "disk_gb": round(total / 1e9, 3), "disk_pinned": pinned}


def _read_direct(path, buf):
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateFileW.restype = wintypes.HANDLE
        k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        k32.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        # GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN
        h = k32.CreateFileW(path, 0x80000000, 1, None, 3, 0x20000000 | 0x08000000, None)
        if h is None or h == wintypes.HANDLE(-1).value:
            raise OSError(ctypes.get_last_error(), "CreateFileW", path)
        try:
            base, done, got = buf.data_ptr(), 0, wintypes.DWORD()
            while done < buf.numel():
                want = min(CHUNK, buf.numel() - done)
                if not k32.ReadFile(h, base + done, want, ctypes.byref(got), None):
                    raise OSError(ctypes.get_last_error(), "ReadFile", path)
                if got.value == 0:
                    break
                done += got.value
        finally:
            k32.CloseHandle(h)
        return done
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECT", 0))
    try:
        mv, done = memoryview(buf.numpy()), 0
        while done < buf.numel():
            got = os.readv(fd, [mv[done:done + CHUNK]])
            if got == 0:
                break
            done += got
        return done
    finally:
        os.close(fd)


def load(path):
    """Read an entry into host memory. Returns (key repr, n, {layer: (key, value)}, {layer: (conv, recurrent)}, stats);
    the tensors are views of one page-aligned (pinned when possible) buffer, ready for non-blocking copies."""
    t0 = time.perf_counter()
    size = os.path.getsize(path)
    buf, pinned = _host_buffer(size)
    if DIRECT:
        done = _read_direct(path, buf)
    else:
        with open(path, "rb") as f:
            done = f.readinto(memoryview(buf.numpy()))
    if done != size:
        raise OSError(f"short read {done}/{size}: {path}")
    head = struct.unpack("<Q", bytes(buf[:8].numpy()))[0]
    header = json.loads(bytes(buf[8:8 + head].numpy()))
    head_len = _round_up(8 + head)
    out = {}
    for m in header["tensors"]:
        dt = _DTYPES[m["dtype"]]
        nb = dt.itemsize
        for d in m["shape"]:
            nb *= d
        o = head_len + m["offset"]
        out[m["name"]] = buf[o:o + nb].view(dt).view(m["shape"])
    kv = {int(k[1:]): (out[k], out["v" + k[1:]]) for k in out if k[0] == "k"}
    states = {int(k[1:]): (out[k], out["r" + k[1:]]) for k in out if k[0] == "c"}
    return header["key"], header["n"], kv, states, {"disk_read_s": round(time.perf_counter() - t0, 3), "disk_gb": round(size / 1e9, 3),
                                                    "disk_pinned": pinned, "disk_direct": DIRECT}
