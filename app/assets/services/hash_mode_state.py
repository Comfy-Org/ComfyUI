from __future__ import annotations

import logging
import os

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.assets import mode as _mode
from app.assets.database.models import AssetContent, AssetSystemState
from app.assets.database.queries.records import create_content, create_record, mark_content_missing
from app.assets.helpers import to_stored_hash
from app.assets.services.path_utils import compute_loader_path, get_name_and_tags_from_asset_path
from app.assets.services.snapshot_hash import snapshot_hash

_KEY = "hash_mode"
_PENDING_QUEUE: list[str] = []
_off_to_on_transition_in_flight = False


def clear_transition_queue() -> None:
    global _off_to_on_transition_in_flight

    _PENDING_QUEUE.clear()
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
        if row.path not in _PENDING_QUEUE:
            _PENDING_QUEUE.append(row.path)


def drain_transition_queue(session: Session) -> None:
    global _off_to_on_transition_in_flight

    pending_count = len(_PENDING_QUEUE)
    for _ in range(pending_count):
        path = _PENDING_QUEUE.pop(0)
        try:
            snapshot = snapshot_hash(path)
        except OSError:
            _PENDING_QUEUE.append(path)
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
                _PENDING_QUEUE.append(path)
            else:
                _PENDING_QUEUE.append(path)
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
