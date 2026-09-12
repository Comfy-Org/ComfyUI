"""Reconciles catalogued content against what is actually on disk: retiring rows
whose file is gone, splitting a row whose bytes changed, and recovering one
whose file came back. Recovery fires only when the returning file's hash
identifies exactly one missing row and no live row already occupies that path,
so a restored file can never leave two live rows describing one location.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.assets.database.models import AssetContent
from app.assets.database.queries.records import (
    create_content,
    create_record,
    mark_content_missing,
    unset_content_missing,
)
from app.assets.helpers import to_stored_hash
from app.assets.services.path_utils import compute_loader_path, get_name_and_tags_from_asset_path
from app.assets.services.snapshot_hash import snapshot_hash

_pending_verification_ids: list[str] = []
_pending_recovery_paths: list[str] = []


def clear_pending_verifications() -> None:
    _pending_verification_ids.clear()
    _pending_recovery_paths.clear()


def queue_pending_verification(content_id: str) -> None:
    if content_id not in _pending_verification_ids:
        _pending_verification_ids.append(content_id)


def pending_recovery_count() -> int:
    return len(_pending_recovery_paths)


def recover_missing_content(
    session: Session, path: str, stat_result: os.stat_result, hashing_is_enabled: bool
) -> Literal["recovered", "no_match", "unstable"]:
    if not hashing_is_enabled:
        return "no_match"
    occupied = session.scalar(
        sa.select(AssetContent.id)
        .where(AssetContent.path == path, AssetContent.is_missing.is_(False))
        .limit(1)
    )
    if occupied is not None:
        return "no_match"
    snapshot = snapshot_hash(path)
    if snapshot is None:
        if path not in _pending_recovery_paths:
            _pending_recovery_paths.append(path)
        return "unstable"
    digest, verified_stat = snapshot
    stored_hash = to_stored_hash(digest)
    matches = list(
        session.scalars(
            sa.select(AssetContent).where(
                AssetContent.path == path,
                AssetContent.is_missing.is_(True),
                AssetContent.hash == stored_hash,
            )
        )
    )
    if len(matches) == 1:
        recovered = matches[0]
        unset_content_missing(session, recovered.id)
        recovered.size_bytes = verified_stat.st_size
        recovered.mtime_ns = verified_stat.st_mtime_ns
        return "recovered"
    if len(matches) > 1:
        return "no_match"
    null_hash_matches = list(
        session.scalars(
            sa.select(AssetContent).where(
                AssetContent.path == path,
                AssetContent.is_missing.is_(True),
                AssetContent.hash.is_(None),
            )
        )
    )
    if len(null_hash_matches) != 1:
        return "no_match"
    candidate = null_hash_matches[0]
    if (candidate.size_bytes, candidate.mtime_ns) != (
        verified_stat.st_size,
        verified_stat.st_mtime_ns,
    ):
        return "no_match"
    unset_content_missing(session, candidate.id)
    candidate.hash = stored_hash
    candidate.size_bytes = verified_stat.st_size
    candidate.mtime_ns = verified_stat.st_mtime_ns
    return "recovered"


def is_path_under_prefixes(path: str, prefixes: list[str]) -> bool:
    candidate = Path(os.path.abspath(path))
    return any(candidate.is_relative_to(os.path.abspath(prefix)) for prefix in prefixes)


def split_content(session: Session, content: AssetContent, stat_result: os.stat_result, hash_value: str | None) -> AssetContent:
    mark_content_missing(session, content.id)
    name, tags = get_name_and_tags_from_asset_path(content.path)
    replacement = create_content(
        session,
        path=content.path,
        hash=hash_value,
        size_bytes=stat_result.st_size,
        mtime_ns=stat_result.st_mtime_ns,
    )
    create_record(
        session,
        content_id=replacement.id,
        name=name,
        loader_path=compute_loader_path(content.path),
        tags=tags,
    )
    return replacement


def detect_content_change(
    session: Session,
    content: AssetContent,
    stat_result: os.stat_result,
    hashing_is_enabled: bool,
) -> None:
    if content.mtime_ns == stat_result.st_mtime_ns:
        # Ruling #10: size drift with unchanged mtime is undefined behavior.
        return
    if hashing_is_enabled:
        queue_pending_verification(content.id)
        return
    if content.size_bytes == stat_result.st_size:
        # User identity rule: a same-size mtime bump (rsync, cloud sync, backup restore) is the
        # same file — never split, or the record's tags and metadata are destroyed.
        # The stored hash goes with the refreshed stat: OFF mode cannot prove the bytes, and a
        # refreshed stat alone would re-qualify the row to be served under a digest it may no
        # longer match.
        content.size_bytes = stat_result.st_size
        content.mtime_ns = stat_result.st_mtime_ns
        content.hash = None
        return
    split_content(session, content, stat_result, hash_value=None)


def drain_pending_verifications(session: Session, limit: int | None = None) -> int:
    queued_count = min(len(_pending_verification_ids), limit or len(_pending_verification_ids))
    processed = 0
    for _ in range(queued_count):
        content_id = _pending_verification_ids.pop(0)
        content = session.get(AssetContent, content_id)
        if content is None or content.is_missing:
            continue
        try:
            os.stat(content.path, follow_symlinks=True)
        except FileNotFoundError:
            mark_content_missing(session, content.id)
            processed += 1
            continue
        except OSError:
            queue_pending_verification(content_id)
            continue

        try:
            snapshot = snapshot_hash(content.path)
        except OSError:
            queue_pending_verification(content_id)
            continue
        if snapshot is None:
            queue_pending_verification(content_id)
            continue
        digest, verified_stat = snapshot
        stored_hash = to_stored_hash(digest)

        if content.hash == stored_hash or content.hash is None:
            content.hash = stored_hash
            content.size_bytes = verified_stat.st_size
            content.mtime_ns = verified_stat.st_mtime_ns
        else:
            split_content(session, content, verified_stat, hash_value=stored_hash)
        processed += 1
    return processed


def live_contents_under_prefixes(session: Session, prefixes: list[str]) -> list[AssetContent]:
    contents = session.scalars(
        sa.select(AssetContent).where(AssetContent.is_missing.is_(False))
    )
    return [content for content in contents if is_path_under_prefixes(content.path, prefixes)]
