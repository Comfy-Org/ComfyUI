"""Tracks the persisted hashing mode and carries the existing catalog across an
off-to-on switch. Turning hashing on enqueues every live content row for
verification and the queue is drained in the background; a path that cannot be
read is retried a bounded number of times and then has its stored hash cleared,
so the queue can empty and the mode can flip. An unverifiable digest never
survives into the persisted enabled state.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from dataclasses import dataclass
from typing import Final

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.assets import mode as _mode
from app.assets.database.models import AssetContent, AssetSystemState
from app.assets.database.queries.records import create_content, create_record, mark_content_missing
from app.assets.helpers import to_stored_hash
from app.assets.services.path_utils import compute_loader_path, get_name_and_tags_from_asset_path
from app.assets.services.snapshot_hash import snapshot_hash

_KEY = "hash_mode"
_MAX_VERIFY_ATTEMPTS: Final = 3


@dataclass
class _PendingEntry:
    path: str
    ticks: int = 0


_PENDING_QUEUE: deque[_PendingEntry] = deque()
_PENDING_PATHS: set[str] = set()
_off_to_on_transition_in_flight = False


def clear_transition_queue() -> None:
    global _off_to_on_transition_in_flight

    _PENDING_QUEUE.clear()
    _PENDING_PATHS.clear()
    _off_to_on_transition_in_flight = False


def pending_transition_count() -> int:
    return len(_PENDING_QUEUE)


def read_stored_mode(session: Session) -> str | None:
    row = session.get(AssetSystemState, _KEY)
    return row.value if row else None


def write_stored_mode(session: Session, value: str) -> None:
    row = session.get(AssetSystemState, _KEY)
    if row is None:
        session.add(AssetSystemState(key=_KEY, value=value))
    else:
        row.value = value
    session.flush()


def record_transition_intent(session: Session) -> str | None:
    stored = read_stored_mode(session)
    runtime = "on" if _mode.hashing_enabled() else "off"
    if stored is None:
        write_stored_mode(session, runtime)
        return None
    if stored == "off" and runtime == "on":
        return "off_to_on"
    if stored == "on" and runtime == "off":
        write_stored_mode(session, "off")
        return "on_to_off"
    return None


def enqueue_transition_work(session: Session, transition: str | None) -> None:
    global _off_to_on_transition_in_flight

    if transition != "off_to_on":
        return
    _off_to_on_transition_in_flight = True
    rows = session.scalars(
        select(AssetContent).where(AssetContent.is_missing.is_(False))
    )
    for row in rows:
        if row.path not in _PENDING_PATHS:
            _PENDING_QUEUE.append(_PendingEntry(row.path))
            _PENDING_PATHS.add(row.path)


def _retry_or_retire(session: Session, entry: _PendingEntry) -> None:
    entry.ticks += 1
    if entry.ticks < _MAX_VERIFY_ATTEMPTS:
        _PENDING_QUEUE.append(entry)
        _PENDING_PATHS.add(entry.path)
        return
    content = session.scalars(
        select(AssetContent).where(
            AssetContent.path == entry.path, AssetContent.is_missing.is_(False)
        )
    ).first()
    if content is not None:
        content.hash = None
    logging.warning(
        "Could not verify %s in %d attempts; clearing its stored hash so the hash-mode "
        "transition can complete",
        entry.path,
        _MAX_VERIFY_ATTEMPTS,
    )


def drain_transition_queue(session: Session) -> None:
    global _off_to_on_transition_in_flight

    pending_count = len(_PENDING_QUEUE)
    for _ in range(pending_count):
        entry = _PENDING_QUEUE.popleft()
        _PENDING_PATHS.discard(entry.path)
        path = entry.path
        try:
            snapshot = snapshot_hash(path)
        except OSError:
            _retry_or_retire(session, entry)
            continue
        if snapshot is None:
            # snapshot_hash returns None for vanished and unstable files; stat distinguishes them.
            try:
                os.stat(path)
            except FileNotFoundError:
                gone = session.scalars(
                    select(AssetContent).where(
                        AssetContent.path == path, AssetContent.is_missing.is_(False)
                    )
                ).first()
                if gone is not None:
                    mark_content_missing(session, gone.id)
            except OSError:
                _retry_or_retire(session, entry)
            else:
                _retry_or_retire(session, entry)
            continue
        digest, stat = snapshot
        stored_hash = to_stored_hash(digest)
        content = session.scalars(
            select(AssetContent).where(
                AssetContent.path == path, AssetContent.is_missing.is_(False)
            )
        ).first()
        if content is None:
            continue
        if content.hash is None:
            content.hash = stored_hash
            content.size_bytes = stat.st_size
            content.mtime_ns = stat.st_mtime_ns
        elif content.hash != stored_hash:
            try:
                name, tags = get_name_and_tags_from_asset_path(path)
            except ValueError:
                logging.warning(
                    "Skipping hash-mode split for out-of-root path: %s", path
                )
                continue
            mark_content_missing(session, content.id)
            replacement = create_content(
                session,
                path=path,
                hash=stored_hash,
                size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )
            create_record(
                session,
                content_id=replacement.id,
                name=name,
                loader_path=compute_loader_path(path),
                tags=tags,
            )
        else:
            content.size_bytes = stat.st_size
            content.mtime_ns = stat.st_mtime_ns
    if _off_to_on_transition_in_flight and not _PENDING_QUEUE:
        write_stored_mode(session, "on")
        _off_to_on_transition_in_flight = False
