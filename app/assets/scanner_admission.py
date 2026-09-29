"""Decides whether a file observed during a filesystem walk is settled enough to
catalog, and parks the ones that are not. Admission requires two stats taken a
moment apart to agree; a file still changing, or carrying a partial-download
extension, goes onto a bounded watch list that later scans re-check until it
settles, vanishes, or exhausts its retries. This is what keeps a model that is
still downloading out of the catalog.
"""

from __future__ import annotations

import logging
import mimetypes
import os
import time
from dataclasses import dataclass
from typing import Final

from app.assets.event_log import emit, error_type
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


def tick_watch_list() -> None:
    from app.assets.scanner import insert_asset_specs, SeedAssetSpec

    remaining: list[_WatchEntry] = []
    settled: list[SeedAssetSpec] = []
    unvisited = iter(list(_WATCH_LIST))
    try:
        for entry in unvisited:
            try:
                current = os.stat(entry.path)
            except OSError as exc:
                logging.warning("Dropping watched asset after stat failed: %s", entry.path)
                emit("scanner.watch_stat_failed", error_type=error_type(exc))
                continue
            if (current.st_mtime_ns, current.st_size) == (entry.last_stat.st_mtime_ns, entry.last_stat.st_size):
                try:
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
                except Exception as exc:
                    logging.warning(
                        "Dropping watched asset after spec construction failed: %s", entry.path
                    )
                    emit("scanner.watch_spec_failed", error_type=error_type(exc))
                    continue
                settled.append(spec)
                continue
            entry.last_stat = current
            entry.ticks += 1
            if entry.ticks < _WATCH_SCAN_RETRIES:
                remaining.append(entry)
        _created, seed_error = insert_asset_specs(settled, set())
        if seed_error is not None:
            # The batch reports only its first error, so failed entries can't be named here;
            # like any settled entry, they leave the watch list either way.
            logging.warning("Seeding settled watched assets failed for at least one entry")
            emit("scanner.watch_seed_failed", error_type=error_type(seed_error))
    finally:
        # Skipping this write wedges the list: drained entries stay on it and are re-attempted
        # every tick, while entries past the fault never reach the increment _WATCH_SCAN_RETRIES
        # needs to retire them. Draining the iterator keeps entries the loop never reached.
        _WATCH_LIST[:] = remaining + list(unvisited)
