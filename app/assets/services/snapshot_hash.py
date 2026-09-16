"""Hashes a file and returns the stat it proved describes those exact bytes.
Identity, size and mtime are sampled before the read, on the open handle at
both ends of it, and once more afterwards; if any sample disagrees the file
moved under the reader and the result is discarded rather than returned.
Callers persist the stat that comes back, which is what makes a stored hash and
a stored size describe one observation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from blake3 import blake3


@dataclass(frozen=True, slots=True)
class _Snapshot:
    dev: int
    ino: int
    mtime_ns: int
    size: int


def _snapshot(stat_result: os.stat_result) -> _Snapshot:
    return _Snapshot(
        dev=stat_result.st_dev,
        ino=stat_result.st_ino,
        mtime_ns=stat_result.st_mtime_ns,
        size=stat_result.st_size,
    )


def snapshot_hash(
    path: str, chunk_size: int = 8 * 1024 * 1024
) -> tuple[str, os.stat_result] | None:
    try:
        pre_stat = _snapshot(os.stat(path))
        hasher = blake3()
        with open(path, "rb") as file:
            open_stat = _snapshot(os.fstat(file.fileno()))
            while chunk := file.read(chunk_size):
                hasher.update(chunk)
            post_hash_stat = _snapshot(os.fstat(file.fileno()))
        post_stat_result = os.stat(path)
    except FileNotFoundError:
        return None
    post_stat = _snapshot(post_stat_result)
    if len({pre_stat, open_stat, post_hash_stat, post_stat}) != 1:
        return None
    return hasher.hexdigest(), post_stat_result
