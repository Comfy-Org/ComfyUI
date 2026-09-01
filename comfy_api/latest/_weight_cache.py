"""Bounded reuse of loaded model weights.

Loading weights is expensive and reusing them is not, so a capability that
loads a model keeps it until the cache bound is reached. This is the generic
half of that behavior; the model-specific loader and its entry type belong to
the capability that needs them.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Any, Callable, Generic, Optional, TypeVar

T = TypeVar("T")


def weight_identity(path: str) -> tuple[Any, ...]:
    """Identify a weight file by what would change its contents, not its name.

    A file can be replaced in place, so a path alone would serve stale weights
    after an update. Device and inode additionally separate two files that
    share a path across mounts.
    """
    status = os.stat(path)
    return (
        os.path.realpath(path),
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


class WeightCache(Generic[T]):
    """Bounded least-recently-used cache of loaded weights.

    ``discriminators`` separate entries loaded from the same file under
    different settings — a variant, an architecture, a label count — so one
    checkpoint serving two configurations does not alias.

    ``release`` is invoked on an entry as it is evicted, which is where a
    capability moves a model off the accelerator. It must not raise.

    Loads run under the cache lock: a model is costly enough that loading it
    twice concurrently is worse than making the second caller wait.
    """

    def __init__(
        self,
        load: Callable[..., T],
        max_entries: int = 2,
        release: Optional[Callable[[T], None]] = None,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self._load = load
        self._release = release
        self.max_entries = max_entries
        self._entries: "OrderedDict[tuple[Any, ...], T]" = OrderedDict()
        self._lock = threading.Lock()
        self.loads = 0
        self.hits = 0

    def get(self, path: str, *discriminators: Any) -> T:
        key = (weight_identity(path), discriminators)
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is not None:
                self.hits += 1
                self._entries[key] = entry
                return entry
            entry = self._load(path, *discriminators)
            self.loads += 1
            while len(self._entries) >= self.max_entries:
                _stale_key, stale = self._entries.popitem(last=False)
                if self._release is not None:
                    self._release(stale)
            self._entries[key] = entry
            return entry

    def clear(self) -> int:
        """Drop and release every entry, returning how many were held.

        Entries are released after the cache lock is dropped: moving a model
        off the accelerator is slow, and a caller asking for one weight should
        not wait behind the teardown of an unrelated one.
        """
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
        if self._release is not None:
            for entry in entries:
                self._release(entry)
        return len(entries)
