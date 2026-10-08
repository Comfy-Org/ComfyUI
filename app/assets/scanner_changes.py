"""Reconciles catalogued content against what is actually on disk: retiring rows
whose file is gone, splitting a row whose bytes changed, and recovering one
whose file came back. Recovery never fires for a path a live row already
occupies, so a restored file can never leave two live rows describing one
location. With hashing on, the returning file's hash must identify exactly one
missing row; with hashing off, its size must match, and its modification time too
once REVIVE_WINDOW has passed since a scan marked the row missing.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import timedelta
from typing import Literal

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetContent
from app.assets.database.queries.records import (
    create_content,
    create_record,
    mark_content_missing,
    unset_content_missing,
)
from app.assets.helpers import get_utc_now, path_prefix_matcher, sql_path_under_prefix_batches, to_stored_hash
from app.assets.services.file_utils import get_mtime_ns
from app.assets.services.path_utils import compute_loader_path, get_name_and_tags_from_asset_path
from app.assets.services.snapshot_hash import snapshot_hash

# Within this long of a scan marking a row missing, a file back at its path and size
# revives it whatever its mtime; after that the mtime must match too.
REVIVE_WINDOW = timedelta(days=7)

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
    session: Session,
    path: str,
    snapshot: tuple[str, os.stat_result] | None,
    hashing_is_enabled: bool,
) -> Literal["recovered", "no_match", "unstable"]:
    """``snapshot`` is ``snapshot_hash(path)``, taken by the caller before its write
    transaction opens so the file is never hashed while the write lock is held."""
    if not hashing_is_enabled:
        return "no_match"
    occupied = session.scalar(
        sa.select(AssetContent.id)
        .where(AssetContent.path == path, AssetContent.is_missing == sa.false())
        .limit(1)
    )
    if occupied is not None:
        return "no_match"
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


def missing_content_ids_by_path(session: Session, paths: list[str]) -> dict[str, list[str]]:
    """The ids of the missing rows at ``paths``. No index serves a missing row's path, so
    callers run this before their write transaction opens."""
    by_path: dict[str, list[str]] = {}
    for start in range(0, len(paths), 500):
        chunk = paths[start : start + 500]
        for content_id, path in session.execute(
            sa.select(AssetContent.id, AssetContent.path).where(
                AssetContent.is_missing.is_(True), AssetContent.path.in_(chunk)
            )
        ):
            by_path.setdefault(path, []).append(content_id)
    return by_path


def recover_missing_content_by_stat(
    session: Session,
    path: str,
    stat_result: os.stat_result,
    candidate_ids: list[str],
) -> Literal["recovered", "no_match"]:
    """Hashing-off recovery: size and modification time are the identity a hashing-off
    scan checks on a live row, so a returning file that matches them restores its row.
    Within REVIVE_WINDOW of a scan marking it, the size alone does, as the bulk revive
    (scanner.revive_returned_references_safely) has it."""
    mtime_ns = get_mtime_ns(stat_result)
    since = get_utc_now() - REVIVE_WINDOW
    candidates = [session.get(AssetContent, content_id) for content_id in candidate_ids]
    matches = [
        candidate
        for candidate in candidates
        if candidate is not None
        and candidate.is_missing
        and candidate.path == path
        and candidate.size_bytes == stat_result.st_size
        and (candidate.mtime_ns == mtime_ns or (candidate.missing_since is not None and candidate.missing_since >= since))
        # A row whose records were all deleted while it was missing would come back
        # live with nothing to show it, and hold the path so no scan ever lists it.
        and session.scalar(sa.select(Asset.id).where(Asset.content_id == candidate.id).limit(1))
        is not None
    ]
    if not matches:
        return "no_match"
    # "= 0", not "IS 0", so the partial live-path index serves it instead of a table scan.
    occupied = session.scalar(
        sa.select(AssetContent.id)
        .where(AssetContent.path == path, AssetContent.is_missing == sa.false())
        .limit(1)
    )
    if occupied is not None:
        return "no_match"
    # Several match only when earlier offline cycles left copies of one file behind;
    # the newest is the one that was live last.
    recovered = max(matches, key=lambda candidate: (candidate.created_at, candidate.id))
    if recovered.mtime_ns != mtime_ns:
        # As a same-size mtime bump on a live row: keep the row, drop the unproven hash.
        recovered.mtime_ns, recovered.hash = mtime_ns, None
    unset_content_missing(session, recovered.id)
    return "recovered"


def is_path_under_prefixes(path: str, prefixes: list[str]) -> bool:
    return path_prefix_matcher(prefixes)(path)


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
        # Commit the previous entry's writes so this entry's hash runs with no transaction open.
        # That needs a create_session() session: on a write session the next read takes the lock.
        session.commit()
        content_id = _pending_verification_ids.pop(0)
        content = session.get(AssetContent, content_id)
        if content is None or content.is_missing:
            continue
        loaded = (content.hash, content.size_bytes, content.mtime_ns)
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
        # Skip a row another writer retired or changed while the file was hashed.
        content = session.get(AssetContent, content_id, populate_existing=True)
        if content is None or content.is_missing or (content.hash, content.size_bytes, content.mtime_ns) != loaded:
            continue

        if content.hash == stored_hash or content.hash is None:
            content.hash = stored_hash
            content.size_bytes = verified_stat.st_size
            content.mtime_ns = verified_stat.st_mtime_ns
        else:
            split_content(session, content, verified_stat, hash_value=stored_hash)
        processed += 1
    return processed


def live_contents_under_prefixes(session: Session, prefixes: list[str]) -> Iterator[AssetContent]:
    """Stream the live contents under the prefixes in batches; consume it inside the session."""
    seen: set[str] = set()
    for under_prefixes in sql_path_under_prefix_batches(AssetContent.path, prefixes):
        stmt = sa.select(AssetContent).where(AssetContent.is_missing.is_(False), under_prefixes)
        for content in session.scalars(stmt.execution_options(yield_per=500)):
            if content.id not in seen:
                seen.add(content.id)
                yield content
