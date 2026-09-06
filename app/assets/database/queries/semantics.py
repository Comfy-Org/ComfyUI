
from dataclasses import dataclass

import sqlalchemy as sa
from sqlalchemy.dialects import sqlite
from sqlalchemy.orm import Session

from app.assets.database.models import (
    Asset,
    AssetReference,
    AssetReferenceTag,
    AssetSemanticsVersion,
)
from app.assets.database.queries.common import (
    MAX_BIND_PARAMS,
    iter_chunks,
    iter_row_chunks,
)
from app.assets.database.queries.tags import ensure_tags_exist
from app.assets.helpers import get_utc_now

_SINGLETON_ROW_ID = 1

AUTOMATIC_TAG_ORIGIN = "automatic"


def get_semantics_version(session: Session) -> int:
    row = session.get(AssetSemanticsVersion, _SINGLETON_ROW_ID)
    return int(row.version) if row is not None else 0


def set_semantics_version(session: Session, version: int) -> None:
    row = session.get(AssetSemanticsVersion, _SINGLETON_ROW_ID)
    if row is None:
        session.add(
            AssetSemanticsVersion(
                id=_SINGLETON_ROW_ID, version=int(version), updated_at=get_utc_now()
            )
        )
    else:
        row.version = int(version)
        row.updated_at = get_utc_now()
    session.flush()


@dataclass(frozen=True)
class DerivedStateRow:

    reference_id: str
    file_path: str
    loader_path: str | None
    mtime_ns: int | None
    is_missing: bool
    needs_verify: bool
    size_bytes: int | None


def get_file_backed_references_page(
    session: Session,
    after_id: str | None,
    limit: int,
) -> list[DerivedStateRow]:
    """Soft-deleted references are deliberately included: their derived columns are as stale as anyone else's."""
    query = (
        sa.select(
            AssetReference.id,
            AssetReference.file_path,
            AssetReference.loader_path,
            AssetReference.mtime_ns,
            AssetReference.is_missing,
            AssetReference.needs_verify,
            Asset.size_bytes,
        )
        .join(Asset, Asset.id == AssetReference.asset_id)
        .where(AssetReference.file_path.isnot(None))
        .order_by(AssetReference.id.asc())
        .limit(limit)
    )
    if after_id is not None:
        query = query.where(AssetReference.id > after_id)

    return [
        DerivedStateRow(
            reference_id=row[0],
            file_path=row[1],
            loader_path=row[2],
            mtime_ns=row[3],
            is_missing=bool(row[4]),
            needs_verify=bool(row[5]),
            size_bytes=int(row[6]) if row[6] is not None else None,
        )
        for row in session.execute(query).all()
    ]


def get_tag_origins_by_reference(
    session: Session, reference_ids: list[str]
) -> dict[str, dict[str, str]]:
    if not reference_ids:
        return {}

    by_reference: dict[str, dict[str, str]] = {}
    for chunk in iter_chunks(reference_ids, MAX_BIND_PARAMS):
        rows = session.execute(
            sa.select(
                AssetReferenceTag.asset_reference_id,
                AssetReferenceTag.tag_name,
                AssetReferenceTag.origin,
            ).where(AssetReferenceTag.asset_reference_id.in_(chunk))
        ).all()
        for reference_id, tag_name, origin in rows:
            by_reference.setdefault(reference_id, {})[tag_name] = origin
    return by_reference


def bulk_set_loader_paths(
    session: Session, loader_paths: dict[str, str | None]
) -> None:
    if not loader_paths:
        return
    session.execute(
        sa.update(AssetReference),
        [
            {"id": reference_id, "loader_path": loader_path}
            for reference_id, loader_path in loader_paths.items()
        ],
    )


def bulk_add_automatic_tags(session: Session, links: list[tuple[str, str]]) -> None:
    if not links:
        return

    now = get_utc_now()
    ensure_tags_exist(session, {tag_name for _, tag_name in links})
    rows = [
        {
            "asset_reference_id": reference_id,
            "tag_name": tag_name,
            "origin": AUTOMATIC_TAG_ORIGIN,
            "added_at": now,
        }
        for reference_id, tag_name in links
    ]
    for chunk in iter_row_chunks(rows, cols_per_row=4):
        session.execute(
            sqlite.insert(AssetReferenceTag)
            .values(chunk)
            .on_conflict_do_nothing(
                index_elements=[
                    AssetReferenceTag.asset_reference_id,
                    AssetReferenceTag.tag_name,
                ]
            )
        )


def bulk_remove_automatic_tags(session: Session, links: list[tuple[str, str]]) -> None:
    """The origin is re-asserted below rather than trusted from the caller's snapshot."""
    if not links:
        return

    by_reference: dict[str, list[str]] = {}
    for reference_id, tag_name in links:
        by_reference.setdefault(reference_id, []).append(tag_name)

    for reference_id, tag_names in by_reference.items():
        for chunk in iter_chunks(tag_names, MAX_BIND_PARAMS):
            session.execute(
                sa.delete(AssetReferenceTag).where(
                    AssetReferenceTag.asset_reference_id == reference_id,
                    AssetReferenceTag.tag_name.in_(chunk),
                    AssetReferenceTag.origin == AUTOMATIC_TAG_ORIGIN,
                )
            )
