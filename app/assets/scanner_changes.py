"""Reconciles catalogued content against what is actually on disk: retiring rows
whose file is gone, splitting a row whose bytes changed, and recovering one
whose file came back. Recovery fires only when the returning file's hash
identifies exactly one missing row and no live row already occupies that path,
so a restored file can never leave two live rows describing one location.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Literal, NamedTuple

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetContent
from app.assets.database.queries.records import (
    create_content,
    create_record,
    mark_content_missing,
    unset_content_missing,
)
from app.assets.helpers import sql_path_under_prefix, to_stored_hash
from app.assets.services.path_utils import compute_loader_path, get_name_and_tags_from_asset_path
from app.assets.services.snapshot_hash import snapshot_hash
from app.database.db import create_session, run_write_txn

_pending_verification_ids: list[str] = []
_pending_recovery_paths: list[str] = []


class PreparedRecovery(NamedTuple):
    path: str
    initial_stat: os.stat_result
    snapshot: tuple[str, os.stat_result] | None


class _PendingVerificationPreflight(NamedTuple):
    content_id: str
    path: str | None
    content_hash: str | None
    size_bytes: int | None
    mtime_ns: int | None
    outcome: Literal["drop", "gone", "retry", "ready"]


def prepare_missing_content_recovery(path: str, stat_result: os.stat_result) -> PreparedRecovery:
    return PreparedRecovery(path, stat_result, snapshot_hash(path))


def clear_pending_verifications() -> None:
    _pending_verification_ids.clear()
    _pending_recovery_paths.clear()


def queue_pending_verification(content_id: str) -> None:
    if content_id not in _pending_verification_ids:
        _pending_verification_ids.append(content_id)


def queue_pending_recovery(path: str) -> None:
    if path not in _pending_recovery_paths:
        _pending_recovery_paths.append(path)


def pending_recovery_count() -> int:
    return len(_pending_recovery_paths)


def recover_missing_content_from_preparation(
    session: Session,
    path: str,
    stat_result: os.stat_result,
    prepared: PreparedRecovery,
    pending_recovery_paths: list[str],
) -> Literal["recovered", "no_match", "unstable"]:
    occupied = session.scalar(
        sa.select(AssetContent.id)
        .where(AssetContent.path == path, AssetContent.is_missing.is_(False))
        .limit(1)
    )
    if occupied is not None:
        return "no_match"
    if prepared.snapshot is None or (
        prepared.initial_stat.st_size != stat_result.st_size
        or prepared.initial_stat.st_mtime_ns != stat_result.st_mtime_ns
    ):
        if path not in pending_recovery_paths:
            pending_recovery_paths.append(path)
        return "unstable"
    digest, verified_stat = prepared.snapshot
    if (
        verified_stat.st_size != stat_result.st_size
        or verified_stat.st_mtime_ns != stat_result.st_mtime_ns
    ):
        if path not in pending_recovery_paths:
            pending_recovery_paths.append(path)
        return "unstable"
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
    pending_verification_ids: list[str] | None = None,
) -> None:
    if content.mtime_ns == stat_result.st_mtime_ns:
        # Ruling #10: size drift with unchanged mtime is undefined behavior.
        return
    if hashing_is_enabled:
        if content.hash is None:
            session.execute(
                sa.update(Asset)
                .where(Asset.content_id == content.id)
                .values(system_metadata=None)
            )
        if pending_verification_ids is None:
            queue_pending_verification(content.id)
        elif content.id not in pending_verification_ids:
            pending_verification_ids.append(content.id)
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
        session.execute(
            sa.update(Asset)
            .where(Asset.content_id == content.id)
            .values(system_metadata=None)
        )
        return
    split_content(session, content, stat_result, hash_value=None)


def _preflight_pending_verification(
    content_id: str,
) -> _PendingVerificationPreflight:
    with create_session() as session:
        content = session.get(AssetContent, content_id)
        if content is None or content.is_missing:
            return _PendingVerificationPreflight(
                content_id, None, None, None, None, "drop"
            )
        path = content.path
        content_hash = content.hash
        size_bytes = content.size_bytes
        mtime_ns = content.mtime_ns
    try:
        os.stat(path, follow_symlinks=True)
    except FileNotFoundError:
        outcome: Literal["drop", "gone", "retry", "ready"] = "gone"
    except OSError:
        outcome = "retry"
    else:
        outcome = "ready"
    return _PendingVerificationPreflight(
        content_id, path, content_hash, size_bytes, mtime_ns, outcome
    )


def _apply_pending_verification(
    session: Session,
    preflight: _PendingVerificationPreflight,
    snapshot: tuple[str, os.stat_result] | None,
) -> Literal["drop", "processed", "retry"]:
    if preflight.outcome == "drop":
        return "drop"
    content = session.get(AssetContent, preflight.content_id)
    if (
        content is None
        or content.is_missing
        or content.path != preflight.path
        or content.hash != preflight.content_hash
        or content.size_bytes != preflight.size_bytes
        or content.mtime_ns != preflight.mtime_ns
    ):
        return "drop"
    assert preflight.path is not None
    if preflight.outcome == "gone":
        mark_content_missing(session, content.id)
        return "processed"
    if preflight.outcome == "retry" or snapshot is None:
        return "retry"
    digest, verified_stat = snapshot
    stored_hash = to_stored_hash(digest)
    if content.hash == stored_hash or content.hash is None:
        content.hash = stored_hash
        content.size_bytes = verified_stat.st_size
        content.mtime_ns = verified_stat.st_mtime_ns
    else:
        split_content(session, content, verified_stat, hash_value=stored_hash)
    return "processed"


def drain_pending_verifications(
    _session: Session | None = None,
    limit: int | None = None,
    interrupt_check: Callable[[], bool] | None = None,
) -> int:
    queued_count = min(len(_pending_verification_ids), limit or len(_pending_verification_ids))
    processed = 0
    for _ in range(queued_count):
        if interrupt_check and interrupt_check():
            break
        content_id = _pending_verification_ids[0]
        preflight = _preflight_pending_verification(content_id)
        snapshot: tuple[str, os.stat_result] | None = None
        if preflight.outcome == "ready":
            assert preflight.path is not None
            try:
                snapshot = snapshot_hash(preflight.path)
            except OSError:
                preflight = preflight._replace(outcome="retry")
        outcome = run_write_txn(
            lambda session: _apply_pending_verification(session, preflight, snapshot)
        )
        _pending_verification_ids.pop(0)
        if outcome == "retry":
            queue_pending_verification(content_id)
        elif outcome == "processed":
            processed += 1
    return processed


def live_contents_under_prefixes(session: Session, prefixes: list[str]) -> list[AssetContent]:
    if not prefixes:
        return []
    return list(
        session.scalars(
            sa.select(AssetContent).where(
                AssetContent.is_missing.is_(False),
                sa.or_(*(sql_path_under_prefix(AssetContent.path, prefix) for prefix in prefixes)),
            )
        )
    )
