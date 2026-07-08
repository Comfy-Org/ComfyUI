"""Query functions for the unified AssetReference table.

This module replaces the separate asset_info.py and cache_state.py query modules,
providing a unified interface for the merged asset_references table.
"""

from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from typing import NamedTuple, Sequence

import sqlalchemy as sa
from sqlalchemy import delete, select
from sqlalchemy.dialects import sqlite
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, noload

from app.assets.database.models import (
    Asset,
    AssetReference,
    AssetReferenceMeta,
    AssetReferenceTag,
    Tag,
)
from app.assets.database.queries.common import (
    MAX_BIND_PARAMS,
    apply_metadata_filter,
    apply_tag_filters,
    build_prefix_like_conditions,
    build_visible_owner_clause,
    calculate_rows_per_statement,
    iter_chunks,
)
from app.assets.helpers import escape_sql_like_string, get_utc_now


def _check_is_scalar(v):
    if v is None:
        return True
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float, Decimal, str)):
        return True
    return False


def _scalar_to_row(key: str, ordinal: int, value) -> dict:
    """Convert a scalar value to a typed projection row."""
    if isinstance(value, bool):
        return {"key": key, "ordinal": ordinal, "val_bool": bool(value)}
    if isinstance(value, (int, float, Decimal)):
        num = value if isinstance(value, Decimal) else Decimal(str(value))
        return {"key": key, "ordinal": ordinal, "val_num": num}
    if isinstance(value, str):
        return {"key": key, "ordinal": ordinal, "val_str": value}
    return {"key": key, "ordinal": ordinal, "val_json": value}


def convert_metadata_to_rows(key: str, value) -> list[dict]:
    """Turn a metadata key/value into typed projection rows."""
    if value is None:
        return []

    if _check_is_scalar(value):
        return [_scalar_to_row(key, 0, value)]

    if isinstance(value, list):
        if all(_check_is_scalar(x) for x in value):
            return [_scalar_to_row(key, i, x) for i, x in enumerate(value) if x is not None]
        return [{"key": key, "ordinal": i, "val_json": x} for i, x in enumerate(value) if x is not None]

    return [{"key": key, "ordinal": 0, "val_json": value}]




def get_reference_by_id(
    session: Session,
    reference_id: str,
) -> AssetReference | None:
    return session.get(AssetReference, reference_id)


def get_reference_with_owner_check(
    session: Session,
    reference_id: str,
    owner_id: str,
) -> AssetReference:
    """Fetch a reference and verify ownership.

    Raises:
        ValueError: if reference not found or soft-deleted
        PermissionError: if owner_id doesn't match
    """
    ref = get_reference_by_id(session, reference_id=reference_id)
    if not ref or ref.deleted_at is not None:
        raise ValueError(f"AssetReference {reference_id} not found")
    if ref.owner_id and ref.owner_id != owner_id:
        raise PermissionError("not owner")
    return ref


def get_reference_by_file_path(
    session: Session,
    file_path: str,
) -> AssetReference | None:
    """Get a reference by its file path."""
    return (
        session.execute(
            select(AssetReference).where(AssetReference.file_path == file_path).limit(1)
        )
        .scalars()
        .first()
    )


def count_active_siblings(
    session: Session,
    asset_id: str,
    exclude_reference_id: str,
) -> int:
    """Count active (non-deleted) references to an asset, excluding one reference."""
    return (
        session.query(AssetReference)
        .filter(
            AssetReference.asset_id == asset_id,
            AssetReference.id != exclude_reference_id,
            AssetReference.deleted_at.is_(None),
        )
        .count()
    )


def reference_exists_for_asset_id(
    session: Session,
    asset_id: str,
) -> bool:
    q = (
        select(sa.literal(True))
        .select_from(AssetReference)
        .where(AssetReference.asset_id == asset_id)
        .where(AssetReference.deleted_at.is_(None))
        .limit(1)
    )
    return session.execute(q).first() is not None


def reference_exists(
    session: Session,
    reference_id: str,
) -> bool:
    """Return True if a reference with the given ID exists (not soft-deleted)."""
    q = (
        select(sa.literal(True))
        .select_from(AssetReference)
        .where(AssetReference.id == reference_id)
        .where(AssetReference.deleted_at.is_(None))
        .limit(1)
    )
    return session.execute(q).first() is not None


def insert_reference(
    session: Session,
    asset_id: str,
    name: str,
    owner_id: str = "",
    file_path: str | None = None,
    mtime_ns: int | None = None,
    preview_id: str | None = None,
) -> AssetReference | None:
    """Insert a new AssetReference. Returns None if unique constraint violated."""
    now = get_utc_now()
    try:
        with session.begin_nested():
            ref = AssetReference(
                asset_id=asset_id,
                name=name,
                owner_id=owner_id,
                file_path=file_path,
                mtime_ns=mtime_ns,
                preview_id=preview_id,
                created_at=now,
                updated_at=now,
                last_access_time=now,
            )
            session.add(ref)
            session.flush()
            return ref
    except IntegrityError:
        return None


def get_or_create_reference(
    session: Session,
    asset_id: str,
    name: str,
    owner_id: str = "",
    file_path: str | None = None,
    mtime_ns: int | None = None,
    preview_id: str | None = None,
) -> tuple[AssetReference, bool]:
    """Get existing or create new AssetReference.

    For filesystem references (file_path is set), uniqueness is by file_path.
    For API references (file_path is None), we look for matching
    asset_id + owner_id + name.

    Returns (reference, created).
    """
    ref = insert_reference(
        session,
        asset_id=asset_id,
        name=name,
        owner_id=owner_id,
        file_path=file_path,
        mtime_ns=mtime_ns,
        preview_id=preview_id,
    )
    if ref:
        return ref, True

    # Find existing - priority to file_path match, then name match
    if file_path:
        existing = get_reference_by_file_path(session, file_path)
    else:
        existing = (
            session.execute(
                select(AssetReference)
                .where(
                    AssetReference.asset_id == asset_id,
                    AssetReference.name == name,
                    AssetReference.owner_id == owner_id,
                    AssetReference.file_path.is_(None),
                )
                .limit(1)
            )
            .unique()
            .scalar_one_or_none()
        )
    if not existing:
        raise RuntimeError("Failed to find AssetReference after insert conflict.")
    return existing, False


def update_reference_timestamps(
    session: Session,
    reference: AssetReference,
    preview_id: str | None = None,
) -> None:
    """Update timestamps and optionally preview_id on existing AssetReference."""
    now = get_utc_now()
    if preview_id and reference.preview_id != preview_id:
        reference.preview_id = preview_id
    reference.updated_at = now


def list_references_page(
    session: Session,
    owner_id: str = "",
    limit: int = 100,
    offset: int = 0,
    name_contains: str | None = None,
    include_tags: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
    metadata_filter: dict | None = None,
    sort: str | None = None,
    order: str | None = None,
    after_cursor_value: object | None = None,
    after_cursor_id: str | None = None,
) -> tuple[list[AssetReference], dict[str, list[str]], int]:
    """List references with pagination, filtering, and sorting.

    When ``after_cursor_value``/``after_cursor_id`` are supplied the query uses
    keyset pagination — ``offset`` is ignored and a WHERE clause selects rows
    strictly after the given ``(sort_col, id)`` position in the active sort
    direction. The cursor value must already be typed for the column
    (datetime for time sorts, int for size, str for name); the caller decodes
    the opaque cursor string and resolves to the typed value.

    Returns (references, tag_map, total_count).
    """
    base = (
        select(AssetReference)
        .join(Asset, Asset.id == AssetReference.asset_id)
        .where(build_visible_owner_clause(owner_id))
        .where(AssetReference.is_missing == False)  # noqa: E712
        .where(AssetReference.deleted_at.is_(None))
        .options(noload(AssetReference.tags))
    )

    if name_contains:
        escaped, esc = escape_sql_like_string(name_contains)
        base = base.where(AssetReference.name.ilike(f"%{escaped}%", escape=esc))

    base = apply_tag_filters(base, include_tags, exclude_tags)
    base = apply_metadata_filter(base, metadata_filter)

    sort = (sort or "created_at").lower()
    order = (order or "desc").lower()
    sort_map = {
        "name": AssetReference.name,
        "created_at": AssetReference.created_at,
        "updated_at": AssetReference.updated_at,
        "last_access_time": AssetReference.last_access_time,
        "size": Asset.size_bytes,
    }
    sort_col = sort_map.get(sort, AssetReference.created_at)
    descending = order == "desc"

    # Keyset WHERE: (sort_col, id) strictly less-than / greater-than the cursor.
    # Equivalent to: sort_col <op> v  OR  (sort_col = v AND id <op> cursor_id).
    if after_cursor_value is not None and after_cursor_id is not None:
        if descending:
            keyset = sa.or_(
                sort_col < after_cursor_value,
                sa.and_(sort_col == after_cursor_value, AssetReference.id < after_cursor_id),
            )
        else:
            keyset = sa.or_(
                sort_col > after_cursor_value,
                sa.and_(sort_col == after_cursor_value, AssetReference.id > after_cursor_id),
            )
        base = base.where(keyset)

    # Secondary ORDER BY id (matching the primary direction) gives the keyset
    # comparison a deterministic tiebreaker on duplicate sort_col values.
    id_exp = AssetReference.id.desc() if descending else AssetReference.id.asc()
    sort_exp = sort_col.desc() if descending else sort_col.asc()

    base = base.order_by(sort_exp, id_exp).limit(limit)
    if after_cursor_id is None:
        base = base.offset(offset)

    count_stmt = (
        select(sa.func.count())
        .select_from(AssetReference)
        .join(Asset, Asset.id == AssetReference.asset_id)
        .where(build_visible_owner_clause(owner_id))
        .where(AssetReference.is_missing == False)  # noqa: E712
        .where(AssetReference.deleted_at.is_(None))
    )
    if name_contains:
        escaped, esc = escape_sql_like_string(name_contains)
        count_stmt = count_stmt.where(
            AssetReference.name.ilike(f"%{escaped}%", escape=esc)
        )
    count_stmt = apply_tag_filters(count_stmt, include_tags, exclude_tags)
    count_stmt = apply_metadata_filter(count_stmt, metadata_filter)

    total = int(session.execute(count_stmt).scalar_one() or 0)
    refs = session.execute(base).unique().scalars().all()

    id_list: list[str] = [r.id for r in refs]
    tag_map: dict[str, list[str]] = defaultdict(list)
    if id_list:
        rows = session.execute(
            select(AssetReferenceTag.asset_reference_id, Tag.name)
            .join(Tag, Tag.name == AssetReferenceTag.tag_name)
            .where(AssetReferenceTag.asset_reference_id.in_(id_list))
            .order_by(AssetReferenceTag.tag_name.asc())
        )
        for ref_id, tag_name in rows.all():
            tag_map[ref_id].append(tag_name)

    return list(refs), tag_map, total


def fetch_reference_asset_and_tags(
    session: Session,
    reference_id: str,
    owner_id: str = "",
) -> tuple[AssetReference, Asset, list[str]] | None:
    stmt = (
        select(AssetReference, Asset, Tag.name)
        .join(Asset, Asset.id == AssetReference.asset_id)
        .join(
            AssetReferenceTag,
            AssetReferenceTag.asset_reference_id == AssetReference.id,
            isouter=True,
        )
        .join(Tag, Tag.name == AssetReferenceTag.tag_name, isouter=True)
        .where(
            AssetReference.id == reference_id,
            AssetReference.deleted_at.is_(None),
            build_visible_owner_clause(owner_id),
        )
        .options(noload(AssetReference.tags))
        .order_by(Tag.name.asc())
    )

    rows = session.execute(stmt).all()
    if not rows:
        return None

    first_ref, first_asset, _ = rows[0]
    tags: list[str] = []
    seen: set[str] = set()
    for _ref, _asset, tag_name in rows:
        if tag_name and tag_name not in seen:
            seen.add(tag_name)
            tags.append(tag_name)
    return first_ref, first_asset, tags


def fetch_reference_and_asset(
    session: Session,
    reference_id: str,
    owner_id: str = "",
) -> tuple[AssetReference, Asset] | None:
    stmt = (
        select(AssetReference, Asset)
        .join(Asset, Asset.id == AssetReference.asset_id)
        .where(
            AssetReference.id == reference_id,
            AssetReference.deleted_at.is_(None),
            build_visible_owner_clause(owner_id),
        )
        .limit(1)
        .options(noload(AssetReference.tags))
    )
    pair = session.execute(stmt).first()
    if not pair:
        return None
    return pair[0], pair[1]


def update_reference_access_time(
    session: Session,
    reference_id: str,
    ts: datetime | None = None,
    only_if_newer: bool = True,
) -> None:
    ts = ts or get_utc_now()
    stmt = sa.update(AssetReference).where(AssetReference.id == reference_id)
    if only_if_newer:
        stmt = stmt.where(
            sa.or_(
                AssetReference.last_access_time.is_(None),
                AssetReference.last_access_time < ts,
            )
        )
    session.execute(stmt.values(last_access_time=ts))


def update_reference_name(
    session: Session,
    reference_id: str,
    name: str,
) -> None:
    """Update the name of an AssetReference."""
    now = get_utc_now()
    session.execute(
        sa.update(AssetReference)
        .where(AssetReference.id == reference_id)
        .values(name=name, updated_at=now)
    )


def update_reference_updated_at(
    session: Session,
    reference_id: str,
    ts: datetime | None = None,
) -> None:
    """Update the updated_at timestamp of an AssetReference."""
    ts = ts or get_utc_now()
    session.execute(
        sa.update(AssetReference)
        .where(AssetReference.id == reference_id)
        .values(updated_at=ts)
    )


def rebuild_metadata_projection(session: Session, ref: AssetReference) -> None:
    """Delete and rebuild AssetReferenceMeta rows from merged system+user metadata.

    The merged dict is ``{**system_metadata, **user_metadata}`` so user keys
    override system keys of the same name.
    """
    session.execute(
        delete(AssetReferenceMeta).where(
            AssetReferenceMeta.asset_reference_id == ref.id
        )
    )
    session.flush()

    merged = {**(ref.system_metadata or {}), **(ref.user_metadata or {})}
    if not merged:
        return

    rows: list[AssetReferenceMeta] = []
    for k, v in merged.items():
        for r in convert_metadata_to_rows(k, v):
            rows.append(
                AssetReferenceMeta(
                    asset_reference_id=ref.id,
                    key=r["key"],
                    ordinal=int(r["ordinal"]),
                    val_str=r.get("val_str"),
                    val_num=r.get("val_num"),
                    val_bool=r.get("val_bool"),
                    val_json=r.get("val_json"),
                )
            )
    if rows:
        session.add_all(rows)
        session.flush()


def set_reference_metadata(
    session: Session,
    reference_id: str,
    user_metadata: dict | None = None,
) -> None:
    ref = session.get(AssetReference, reference_id)
    if not ref:
        raise ValueError(f"AssetReference {reference_id} not found")

    ref.user_metadata = user_metadata or {}
    ref.updated_at = get_utc_now()
    session.flush()

    rebuild_metadata_projection(session, ref)


def set_reference_system_metadata(
    session: Session,
    reference_id: str,
    system_metadata: dict | None = None,
) -> None:
    """Set system_metadata on a reference and rebuild the merged projection."""
    ref = session.get(AssetReference, reference_id)
    if not ref:
        raise ValueError(f"AssetReference {reference_id} not found")

    ref.system_metadata = system_metadata or {}
    ref.updated_at = get_utc_now()
    session.flush()

    rebuild_metadata_projection(session, ref)


def delete_reference_by_id(
    session: Session,
    reference_id: str,
    owner_id: str,
) -> bool:
    stmt = sa.delete(AssetReference).where(
        AssetReference.id == reference_id,
        build_visible_owner_clause(owner_id),
    )
    return int(session.execute(stmt).rowcount or 0) > 0


def soft_delete_reference_by_id(
    session: Session,
    reference_id: str,
    owner_id: str,
) -> bool:
    """Mark a reference as soft-deleted by setting deleted_at timestamp.

    Returns True if the reference was found and marked deleted.
    """
    now = get_utc_now()
    stmt = (
        sa.update(AssetReference)
        .where(
            AssetReference.id == reference_id,
            AssetReference.deleted_at.is_(None),
            build_visible_owner_clause(owner_id),
        )
        .values(deleted_at=now)
    )
    return int(session.execute(stmt).rowcount or 0) > 0


def set_reference_preview(
    session: Session,
    reference_id: str,
    preview_reference_id: str | None = None,
) -> None:
    """Set or clear preview_id and bump updated_at. Raises on unknown IDs."""
    ref = session.get(AssetReference, reference_id)
    if not ref:
        raise ValueError(f"AssetReference {reference_id} not found")

    if preview_reference_id is None:
        ref.preview_id = None
    else:
        if not session.get(AssetReference, preview_reference_id):
            raise ValueError(f"Preview AssetReference {preview_reference_id} not found")
        ref.preview_id = preview_reference_id

    ref.updated_at = get_utc_now()
    session.flush()


class CacheStateRow(NamedTuple):
    """Row from reference query with cache state data."""

    reference_id: str
    file_path: str
    mtime_ns: int | None
    needs_verify: bool
    asset_id: str
    asset_hash: str | None
    size_bytes: int | None


def list_references_by_asset_id(
    session: Session,
    asset_id: str,
) -> Sequence[AssetReference]:
    return (
        session.execute(
            select(AssetReference)
            .where(AssetReference.asset_id == asset_id)
            .where(AssetReference.is_missing == False)  # noqa: E712
            .where(AssetReference.deleted_at.is_(None))
            .order_by(AssetReference.id.asc())
        )
        .scalars()
        .all()
    )


def list_all_file_paths_by_asset_id(
    session: Session,
    asset_id: str,
) -> list[str]:
    """Return every file_path for an asset, including soft-deleted/missing refs.

    Used for orphan cleanup where all on-disk files must be removed.
    """
    return list(
        session.execute(
            select(AssetReference.file_path)
            .where(AssetReference.asset_id == asset_id)
            .where(AssetReference.file_path.isnot(None))
        )
        .scalars()
        .all()
    )


def upsert_reference(
    session: Session,
    asset_id: str,
    file_path: str,
    name: str,
    mtime_ns: int,
    owner_id: str = "",
) -> tuple[bool, bool]:
    """Upsert a reference by file_path. Returns (created, updated).

    Also restores references that were previously marked as missing.
    """
    now = get_utc_now()
    vals = {
        "asset_id": asset_id,
        "file_path": file_path,
        "name": name,
        "owner_id": owner_id,
        "mtime_ns": int(mtime_ns),
        "is_missing": False,
        "created_at": now,
        "updated_at": now,
        "last_access_time": now,
    }
    ins = (
        sqlite.insert(AssetReference)
        .values(**vals)
        .on_conflict_do_nothing(index_elements=[AssetReference.file_path])
    )
    res = session.execute(ins)
    created = int(res.rowcount or 0) > 0

    if created:
        return True, False

    upd = (
        sa.update(AssetReference)
        .where(AssetReference.file_path == file_path)
        .where(
            sa.or_(
                AssetReference.asset_id != asset_id,
                AssetReference.mtime_ns.is_(None),
                AssetReference.mtime_ns != int(mtime_ns),
                AssetReference.is_missing == True,  # noqa: E712
                AssetReference.deleted_at.isnot(None),
            )
        )
        .values(
            asset_id=asset_id, mtime_ns=int(mtime_ns), is_missing=False,
            deleted_at=None, updated_at=now,
        )
    )
    res2 = session.execute(upd)
    updated = int(res2.rowcount or 0) > 0
    return False, updated


def mark_references_missing_outside_prefixes(
    session: Session,
    valid_prefixes: list[str],
) -> int:
    """Mark references as missing when file_path doesn't match any valid prefix.

    Returns number of references marked as missing.
    """
    if not valid_prefixes:
        return 0

    conds = build_prefix_like_conditions(valid_prefixes)
    matches_valid_prefix = sa.or_(*conds)
    result = session.execute(
        sa.update(AssetReference)
        .where(AssetReference.file_path.isnot(None))
        .where(AssetReference.deleted_at.is_(None))
        .where(~matches_valid_prefix)
        .where(AssetReference.is_missing == False)  # noqa: E712
        .values(is_missing=True)
    )
    return result.rowcount


def restore_references_by_paths(session: Session, file_paths: list[str]) -> int:
    """Restore references that were previously marked as missing.

    Returns number of references restored.
    """
    if not file_paths:
        return 0

    total = 0
    for chunk in iter_chunks(file_paths, MAX_BIND_PARAMS):
        result = session.execute(
            sa.update(AssetReference)
            .where(AssetReference.file_path.in_(chunk))
            .where(AssetReference.is_missing == True)  # noqa: E712
            .where(AssetReference.deleted_at.is_(None))
            .values(is_missing=False)
        )
        total += result.rowcount
    return total


def get_unreferenced_unhashed_asset_ids(session: Session) -> list[str]:
    """Get IDs of unhashed assets (hash=None) with no active references.

    An asset is considered unreferenced if it has no references,
    or all its references are marked as missing.

    Returns list of asset IDs that are unreferenced.
    """
    active_ref_exists = (
        sa.select(sa.literal(1))
        .where(AssetReference.asset_id == Asset.id)
        .where(AssetReference.is_missing == False)  # noqa: E712
        .where(AssetReference.deleted_at.is_(None))
        .correlate(Asset)
        .exists()
    )
    unreferenced_subq = sa.select(Asset.id).where(
        Asset.hash.is_(None), ~active_ref_exists
    )
    return [row[0] for row in session.execute(unreferenced_subq).all()]


def delete_assets_by_ids(session: Session, asset_ids: list[str]) -> int:
    """Delete assets and their references by ID.

    Returns number of assets deleted.
    """
    if not asset_ids:
        return 0
    total = 0
    for chunk in iter_chunks(asset_ids, MAX_BIND_PARAMS):
        session.execute(
            sa.delete(AssetReference).where(AssetReference.asset_id.in_(chunk))
        )
        result = session.execute(sa.delete(Asset).where(Asset.id.in_(chunk)))
        total += result.rowcount
    return total


def get_references_for_prefixes(
    session: Session,
    prefixes: list[str],
    *,
    include_missing: bool = False,
) -> list[CacheStateRow]:
    """Get all references with file paths matching any of the given prefixes.

    Args:
        session: Database session
        prefixes: List of absolute directory prefixes to match
        include_missing: If False (default), exclude references marked as missing

    Returns:
        List of cache state rows with joined asset data
    """
    if not prefixes:
        return []

    conds = build_prefix_like_conditions(prefixes)

    query = (
        sa.select(
            AssetReference.id,
            AssetReference.file_path,
            AssetReference.mtime_ns,
            AssetReference.needs_verify,
            AssetReference.asset_id,
            Asset.hash,
            Asset.size_bytes,
        )
        .join(Asset, Asset.id == AssetReference.asset_id)
        .where(AssetReference.file_path.isnot(None))
        .where(AssetReference.deleted_at.is_(None))
        .where(sa.or_(*conds))
    )

    if not include_missing:
        query = query.where(AssetReference.is_missing == False)  # noqa: E712

    rows = session.execute(
        query.order_by(AssetReference.asset_id.asc(), AssetReference.id.asc())
    ).all()

    return [
        CacheStateRow(
            reference_id=row[0],
            file_path=row[1],
            mtime_ns=row[2],
            needs_verify=row[3],
            asset_id=row[4],
            asset_hash=row[5],
            size_bytes=int(row[6]) if row[6] is not None else None,
        )
        for row in rows
    ]


def bulk_update_needs_verify(
    session: Session, reference_ids: list[str], value: bool
) -> int:
    """Set needs_verify flag for multiple references.

    Returns: Number of rows updated
    """
    if not reference_ids:
        return 0
    total = 0
    for chunk in iter_chunks(reference_ids, MAX_BIND_PARAMS):
        result = session.execute(
            sa.update(AssetReference)
            .where(AssetReference.id.in_(chunk))
            .values(needs_verify=value)
        )
        total += result.rowcount
    return total


def bulk_update_is_missing(
    session: Session, reference_ids: list[str], value: bool
) -> int:
    """Set is_missing flag for multiple references.

    Returns: Number of rows updated
    """
    if not reference_ids:
        return 0
    total = 0
    for chunk in iter_chunks(reference_ids, MAX_BIND_PARAMS):
        result = session.execute(
            sa.update(AssetReference)
            .where(AssetReference.id.in_(chunk))
            .values(is_missing=value)
        )
        total += result.rowcount
    return total


def update_is_missing_by_asset_id(
    session: Session, asset_id: str, value: bool
) -> int:
    """Set is_missing flag for ALL references belonging to an asset.

    Returns: Number of rows updated
    """
    result = session.execute(
        sa.update(AssetReference)
        .where(AssetReference.asset_id == asset_id)
        .where(AssetReference.deleted_at.is_(None))
        .values(is_missing=value)
    )
    return result.rowcount


def delete_references_by_ids(session: Session, reference_ids: list[str]) -> int:
    """Delete references by their IDs.

    Returns: Number of rows deleted
    """
    if not reference_ids:
        return 0
    total = 0
    for chunk in iter_chunks(reference_ids, MAX_BIND_PARAMS):
        result = session.execute(
            sa.delete(AssetReference).where(AssetReference.id.in_(chunk))
        )
        total += result.rowcount
    return total


def delete_orphaned_seed_asset(session: Session, asset_id: str) -> bool:
    """Delete a seed asset (hash is None) and its references.

    Returns: True if asset was deleted, False if not found or has a hash
    """
    asset = session.get(Asset, asset_id)
    if not asset:
        return False
    if asset.hash is not None:
        return False
    session.execute(
        sa.delete(AssetReference).where(AssetReference.asset_id == asset_id)
    )
    session.delete(asset)
    return True


class UnenrichedReferenceRow(NamedTuple):
    """Row for references needing enrichment."""

    reference_id: str
    asset_id: str
    file_path: str
    enrichment_level: int


def get_unenriched_references(
    session: Session,
    prefixes: list[str],
    max_level: int = 0,
    limit: int = 1000,
) -> list[UnenrichedReferenceRow]:
    """Get references that need enrichment (enrichment_level <= max_level).

    Args:
        session: Database session
        prefixes: List of absolute directory prefixes to scan
        max_level: Maximum enrichment level to include (0=stubs, 1=metadata done)
        limit: Maximum number of rows to return

    Returns:
        List of unenriched reference rows with file paths
    """
    if not prefixes:
        return []

    conds = build_prefix_like_conditions(prefixes)

    query = (
        sa.select(
            AssetReference.id,
            AssetReference.asset_id,
            AssetReference.file_path,
            AssetReference.enrichment_level,
        )
        .where(AssetReference.file_path.isnot(None))
        .where(AssetReference.deleted_at.is_(None))
        .where(sa.or_(*conds))
        .where(AssetReference.is_missing == False)  # noqa: E712
        .where(AssetReference.enrichment_level <= max_level)
        .order_by(AssetReference.id.asc())
        .limit(limit)
    )

    rows = session.execute(query).all()
    return [
        UnenrichedReferenceRow(
            reference_id=row[0],
            asset_id=row[1],
            file_path=row[2],
            enrichment_level=row[3],
        )
        for row in rows
    ]


def bulk_update_enrichment_level(
    session: Session,
    reference_ids: list[str],
    level: int,
) -> int:
    """Update enrichment level for multiple references.

    Returns: Number of rows updated
    """
    if not reference_ids:
        return 0
    result = session.execute(
        sa.update(AssetReference)
        .where(AssetReference.id.in_(reference_ids))
        .values(enrichment_level=level)
    )
    return result.rowcount


def bulk_insert_references_ignore_conflicts(
    session: Session,
    rows: list[dict],
) -> None:
    """Bulk insert reference rows with ON CONFLICT DO NOTHING on file_path.

    Each dict should have: id, asset_id, file_path, name, owner_id, mtime_ns, etc.
    The is_missing field is automatically set to False for new inserts.
    """
    if not rows:
        return
    enriched_rows = [{**row, "is_missing": False} for row in rows]
    ins = sqlite.insert(AssetReference).on_conflict_do_nothing(
        index_elements=[AssetReference.file_path]
    )
    for chunk in iter_chunks(enriched_rows, calculate_rows_per_statement(14)):
        session.execute(ins, chunk)


def get_references_by_paths_and_asset_ids(
    session: Session,
    path_to_asset: dict[str, str],
) -> set[str]:
    """Query references to find paths where our asset_id won the insert.

    Args:
        path_to_asset: Mapping of file_path -> asset_id we tried to insert

    Returns:
        Set of file_paths where our asset_id is present
    """
    if not path_to_asset:
        return set()

    pairs = list(path_to_asset.items())
    winners: set[str] = set()

    # Each pair uses 2 bind params, so chunk at MAX_BIND_PARAMS // 2
    for chunk in iter_chunks(pairs, MAX_BIND_PARAMS // 2):
        pairwise = sa.tuple_(AssetReference.file_path, AssetReference.asset_id).in_(
            chunk
        )
        result = session.execute(
            select(AssetReference.file_path).where(pairwise)
        )
        winners.update(result.scalars().all())

    return winners


def get_reference_ids_by_ids(
    session: Session,
    reference_ids: list[str],
) -> set[str]:
    """Query to find which reference IDs exist in the database."""
    if not reference_ids:
        return set()

    found: set[str] = set()
    for chunk in iter_chunks(reference_ids, MAX_BIND_PARAMS):
        result = session.execute(
            select(AssetReference.id).where(AssetReference.id.in_(chunk))
        )
        found.update(result.scalars().all())
    return found
