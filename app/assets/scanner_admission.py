from __future__ import annotations

import mimetypes
import os
import time
from dataclasses import dataclass
from typing import Final

from sqlalchemy.orm import Session

from app.assets.services.path_utils import compute_loader_path, get_name_and_tags_from_asset_path

PARTIAL_DOWNLOAD_EXTENSIONS = frozenset({
    ".part", ".partial", ".crdownload", ".download", ".tmp", ".aria2", ".!qb", ".opdownload",
})
_WATCH_SCAN_RETRIES: Final = 30
_WATCH_LIST_MAX_SIZE: Final = 256


@dataclass
class _WatchEntry:
    path: str
    last_stat: os.stat_result
    ticks: int = 0


_WATCH_LIST: list[_WatchEntry] = []


def _should_skip_extension(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in PARTIAL_DOWNLOAD_EXTENSIONS


def _two_stat_admit(paths_with_stats: list[tuple[str, os.stat_result]]) -> tuple[list[str], list[str]]:
    if not paths_with_stats:
        return [], []
    time.sleep(0.1)
    admitted: list[str] = []
    watched: list[str] = []
    for path, first_stat in paths_with_stats:
        try:
            second_stat = os.stat(path)
        except FileNotFoundError:
            continue
        if (second_stat.st_mtime_ns, second_stat.st_size) == (first_stat.st_mtime_ns, first_stat.st_size):
            _WATCH_LIST[:] = [entry for entry in _WATCH_LIST if entry.path != path]
            admitted.append(path)
        else:
            for entry in _WATCH_LIST:
                if entry.path == path:
                    entry.last_stat = second_stat
                    break
            else:
                _WATCH_LIST.append(_WatchEntry(path, second_stat))
                if len(_WATCH_LIST) > _WATCH_LIST_MAX_SIZE:
                    _ = _WATCH_LIST.pop(0)
            watched.append(path)
    return admitted, watched


def tick_watch_list(session: Session) -> None:
    from app.assets.scanner import seed_asset_specs, SeedAssetSpec

    remaining: list[_WatchEntry] = []
    for entry in _WATCH_LIST:
        try:
            current = os.stat(entry.path)
        except FileNotFoundError:
            continue
        if (current.st_mtime_ns, current.st_size) == (entry.last_stat.st_mtime_ns, entry.last_stat.st_size):
            name, tags = get_name_and_tags_from_asset_path(entry.path)
            spec: SeedAssetSpec = {
                "abs_path": entry.path,
                "size_bytes": current.st_size,
                "mtime_ns": current.st_mtime_ns,
                "info_name": name,
                "tags": tags,
                "fname": compute_loader_path(entry.path),
                "metadata": None,
                "mime_type": mimetypes.guess_type(entry.path, strict=False)[0],
                "job_id": None,
            }
            seed_asset_specs(session, [spec])
            continue
        entry.last_stat = current
        entry.ticks += 1
        if entry.ticks < _WATCH_SCAN_RETRIES:
            remaining.append(entry)
    _WATCH_LIST[:] = remaining
