"""Core engine extension: the ``eng.cache`` rows, large values kept in ComfyUI's memory between runs.

Per pack, least recently used first out, within ``COMFY_PACK_CACHE_MB`` (default 2048) across packs;
an entry may expire after ``ttl`` seconds. Values are tensors, plain data, or engine objects (a model
the pack loaded); they stay in ComfyUI's process, so a pack running elsewhere gets them back as it
gets any value. Nothing survives a restart: it's a cache.
"""
import collections
import os
import threading
import time

from comfy_engine import _runtime

_entries = collections.OrderedDict()  # (pack, key) -> (value, size, expires)
_lock = threading.Lock()


def _budget():
    return int(float(os.environ.get("COMFY_PACK_CACHE_MB", "2048")) * (1 << 20))


def _size(value):
    if _runtime._looks_like_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_size(v) for v in value.values()) + 64
    if isinstance(value, (list, tuple)):
        return sum(_size(v) for v in value) + 64
    if isinstance(value, (str, bytes)):
        return len(value)
    model_size = getattr(value, "model_size", None)  # a ModelPatcher
    if callable(model_size):
        try:
            return int(model_size())
        except Exception:
            pass
    return 64


def _key(key):
    return (_runtime.current_runtime().pack or "", str(key))


class EngineOps:
    def get(self, key, default=None):
        with _lock:
            entry = _entries.get(_key(key))
            if entry is None:
                return default
            if entry[2] is not None and entry[2] < time.monotonic():
                del _entries[_key(key)]
                return default
            _entries.move_to_end(_key(key))
            return entry[0]

    def set(self, key, value, ttl=None):
        if ttl is not None and not (isinstance(ttl, (int, float)) and ttl > 0):
            raise ValueError("ttl is a number of seconds")
        size = _size(value)
        budget = _budget()
        if size > budget:
            raise ValueError("the value is larger than the whole cache (COMFY_PACK_CACHE_MB)")
        with _lock:
            _entries.pop(_key(key), None)
            _entries[_key(key)] = (value, size, time.monotonic() + ttl if ttl else None)
            total = sum(entry[1] for entry in _entries.values())
            while total > budget:
                _, (_, dropped, _) = _entries.popitem(last=False)
                total -= dropped

    def delete(self, key):
        with _lock:
            return _entries.pop(_key(key), None) is not None
