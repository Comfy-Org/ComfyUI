"""Resolves a hash to a content row that can actually be served, and claims that
row so a concurrent writer cannot retire it mid-use. A row qualifies only if
its file exists, sits outside the temp directory, and its recorded size and
mtime still match the file — an unrecorded mtime excuses the mtime comparison
but never the size comparison.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import folder_paths
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.assets import mode
from app.assets.database.models import AssetContent


def is_temp_path(path: str) -> bool:
    try:
        temp_root = Path(os.path.abspath(folder_paths.get_temp_directory()))
        candidate = Path(os.path.abspath(path))
        return candidate.is_relative_to(temp_root)
    except OSError:
        return False


def _stat_consistent(content: AssetContent) -> bool:
    try:
        stat = os.stat(content.path)
    except FileNotFoundError:
        return False
    if content.mtime_ns is not None and stat.st_mtime_ns != content.mtime_ns:
        return False
    if stat.st_size != content.size_bytes:
        return False
    return True


def _qualifies(content: AssetContent) -> bool:
    return (
        not content.is_missing
        and os.path.isfile(content.path)
        and _stat_consistent(content)
        and not is_temp_path(content.path)
    )


def qualified_content_iterator(session: Session, hash: str) -> Iterator[AssetContent]:
    rows = session.scalars(
        select(AssetContent)
        .where(AssetContent.hash == hash, AssetContent.is_missing.is_(False))
        .order_by(AssetContent.created_at, AssetContent.id)
    )
    for row in rows:
        if _qualifies(row):
            yield row


def claim_qualified_content(session: Session, content_id: str, hash: str) -> bool:
    """Claim a live matching content row before attaching a record.

    The conditional update takes this session's SQLite write lock through commit.
    False means the row was retired or changed after lookup.
    """
    # This session's connection keeps the claim's write lock through this session's commit.
    result = session.connection().execute(
        update(AssetContent)
        .where(
            AssetContent.id == content_id,
            AssetContent.hash == hash,
            AssetContent.is_missing.is_(False),
        )
        .values(is_missing=False)
    )
    return result.rowcount == 1


def refresh_qualified_content(session: Session, content_id: str) -> AssetContent | None:
    content = session.get(AssetContent, content_id, populate_existing=True)
    if content is None or not _qualifies(content):
        return None
    return content


def lookup_for_from_hash(session: Session, hash: str) -> AssetContent | None:
    if not mode.hashing_enabled():
        return None
    return next(qualified_content_iterator(session, hash), None)


def lookup_for_view(session: Session, hash: str) -> AssetContent | None:
    return next(qualified_content_iterator(session, hash), None)
