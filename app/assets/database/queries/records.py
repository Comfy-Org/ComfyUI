"""Owns every write to content rows, records and their tag links, plus the paged
reads that list them. Inserts that can lose a race — a content row at a path, a
tag, a tag link — run inside a savepoint and re-read the conflicting row, so a
concurrent writer settles the call instead of raising, while a genuine
constraint failure still surfaces. This is the sole writer of a content row's
path, which is what lets other modules trust raw-column path predicates.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Literal, NamedTuple, TypeAlias

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload, noload
from sqlalchemy.sql.elements import ColumnElement

from app.assets.database.models import Asset, AssetContent, AssetTag, Tag
from app.assets.helpers import escape_sql_like_string, get_utc_now

RecordSortField: TypeAlias = Literal[
    "name", "created_at", "updated_at", "size", "last_access_time"
]
RecordSortOrder: TypeAlias = Literal["asc", "desc"]


class RecordCursorBoundary(NamedTuple):
    value: datetime | int | str
    id: str


class RecordPageSpec(NamedTuple):
    all_tags: tuple[str, ...] = ()
    any_tags: tuple[str, ...] = ()
    none_tags: tuple[str, ...] = ()
    name_contains: str | None = None
    limit: int = 20
    offset: int = 0
    sort: RecordSortField = "created_at"
    order: RecordSortOrder = "desc"
    after: RecordCursorBoundary | None = None


_LIVE_PATH_UNIQUE_INDEX = "uq_asset_contents_path_live"


def _is_live_path_conflict(error: IntegrityError) -> bool:
    orig = error.orig
    message = str(orig)
    postgres_names_the_index = getattr(getattr(orig, "diag", None), "constraint_name", None) == _LIVE_PATH_UNIQUE_INDEX
    sqlite_names_the_column = "UNIQUE constraint failed" in message and "asset_contents.path" in message
    return postgres_names_the_index or sqlite_names_the_column


def create_content_reporting_insert(session: Session, path: str, hash: str | None = None, size_bytes: int = 0, mtime_ns: int | None = None) -> tuple[AssetContent, bool]:
    # The sole writer of asset_contents.path, which is what makes the raw-column SQL prefix
    # predicates sound — lifecycle's temp wipe HARD-DELETES every row its predicate admits.
    path = os.path.abspath(path)
    content = AssetContent(path=path, hash=hash, size_bytes=size_bytes, mtime_ns=mtime_ns)
    try:
        with session.begin_nested():
            session.add(content)
            session.flush()
            return content, True
    except IntegrityError as error:
        if not _is_live_path_conflict(error):
            raise
        winner = session.execute(sa.select(AssetContent).where(AssetContent.path == path, AssetContent.is_missing.is_(False))).scalar_one()
        return winner, False


def create_content(session: Session, path: str, hash: str | None = None, size_bytes: int = 0, mtime_ns: int | None = None) -> AssetContent:
    content, _ = create_content_reporting_insert(session, path, hash, size_bytes, mtime_ns)
    return content


def ensure_tag(session: Session, name: str) -> None:
    if session.get(Tag, name) is not None:
        return
    try:
        with session.begin_nested():
            session.add(Tag(name=name))
            session.flush()
    except IntegrityError:
        # Only a row that is now present proves a lost race; anything else is a real failure.
        if session.get(Tag, name) is None:
            raise


def ensure_tag_link(session: Session, *, asset_id: str, tag_name: str, origin: str) -> bool:
    if session.get(AssetTag, (asset_id, tag_name)) is not None:
        return False
    try:
        with session.begin_nested():
            session.add(AssetTag(asset_id=asset_id, tag_name=tag_name, origin=origin))
            session.flush()
    except IntegrityError:
        if session.get(AssetTag, (asset_id, tag_name)) is None:
            raise
        return False
    return True


def create_record(session: Session, content_id: str, name: str, mime_type: str | None = None, job_id: str | None = None, loader_path: str | None = None, tags: Sequence[str] | None = None, *, system_metadata: dict[str, Any] | None = None) -> Asset:
    record = Asset(content_id=content_id, name=name, mime_type=mime_type, job_id=job_id, loader_path=loader_path, system_metadata=system_metadata)
    session.add(record)
    session.flush()
    for tag_name in dict.fromkeys(tags or ()):
        ensure_tag(session, tag_name)
        ensure_tag_link(session, asset_id=record.id, tag_name=tag_name, origin="manual")
    session.flush()
    return record


def get_record_by_id(session: Session, id: str) -> Asset | None:
    return session.get(Asset, id)


def get_preview_file_paths_by_ids(
    session: Session,
    preview_ids: Sequence[str],
) -> dict[str, str]:
    if not preview_ids:
        return {}

    rows = session.execute(
        sa.select(Asset.id, AssetContent.path)
        .join(AssetContent, Asset.content_id == AssetContent.id)
        .where(
            Asset.id.in_(preview_ids),
            AssetContent.is_missing.is_(False),
        )
    )
    return {preview_id: path for preview_id, path in rows}


def get_record_by_path_or_none(session: Session, path: str) -> Asset | None:
    return session.scalar(
        sa.select(Asset)
        .join(AssetContent, Asset.content_id == AssetContent.id)
        .where(AssetContent.path == path, AssetContent.is_missing.is_(False))
        .order_by(Asset.created_at.desc(), Asset.id.desc())
        .limit(1)
    )


def fetch_record_tags(session: Session, record_id: str) -> list[str]:
    return list(
        session.scalars(
            sa.select(AssetTag.tag_name)
            .where(AssetTag.asset_id == record_id)
            .order_by(AssetTag.tag_name)
        )
    )


def update_record_access_time(
    session: Session,
    record_id: str,
    ts: datetime | None = None,
    only_if_newer: bool = True,
) -> None:
    ts = ts or get_utc_now()
    stmt = sa.update(Asset).where(Asset.id == record_id)
    if only_if_newer:
        stmt = stmt.where(
            sa.or_(
                Asset.last_access_time.is_(None),
                Asset.last_access_time < ts,
            )
        )
    session.execute(stmt.values(last_access_time=ts))


def bump_record_updated_at(session: Session, record_id: str) -> None:
    session.execute(
        sa.update(Asset).where(Asset.id == record_id).values(updated_at=get_utc_now())
    )


def build_record_tag_filter_clauses(
    all_tags: Sequence[str],
    any_tags: Sequence[str],
    none_tags: Sequence[str],
) -> tuple[ColumnElement[bool], ...]:
    clauses: list[ColumnElement[bool]] = []
    for tag_name in all_tags:
        clauses.append(
            sa.exists(
                sa.select(AssetTag.asset_id).where(
                    AssetTag.asset_id == Asset.id,
                    AssetTag.tag_name == tag_name,
                )
            )
        )
    if any_tags:
        clauses.append(
            sa.exists(
                sa.select(AssetTag.asset_id).where(
                    AssetTag.asset_id == Asset.id,
                    AssetTag.tag_name.in_(any_tags),
                )
            )
        )
    if none_tags:
        clauses.append(
            ~sa.exists(
                sa.select(AssetTag.asset_id).where(
                    AssetTag.asset_id == Asset.id,
                    AssetTag.tag_name.in_(none_tags),
                )
            )
        )
    return tuple(clauses)


def list_records_page(
    session: Session,
    spec: RecordPageSpec,
) -> tuple[list[Asset], dict[str, list[str]], int]:
    filters = list(build_record_tag_filter_clauses(spec.all_tags, spec.any_tags, spec.none_tags))
    if spec.name_contains:
        escaped_name, escape_character = escape_sql_like_string(spec.name_contains)
        filters.append(
            Asset.name.ilike(f"%{escaped_name}%", escape=escape_character)
        )

    sort_columns = {
        "name": Asset.name,
        "created_at": Asset.created_at,
        "updated_at": Asset.updated_at,
        "size": AssetContent.size_bytes,
        "last_access_time": Asset.last_access_time,
    }
    sort_column = sort_columns[spec.sort]
    descending = spec.order == "desc"
    sort_expression = sort_column.desc() if descending else sort_column.asc()
    id_expression = Asset.id.desc() if descending else Asset.id.asc()

    statement = (
        sa.select(Asset)
        .join(AssetContent, Asset.content_id == AssetContent.id)
        .where(*filters)
        .options(joinedload(Asset.content), noload(Asset.tags))
    )
    if spec.after is not None:
        comparison = (
            sort_column < spec.after.value
            if descending
            else sort_column > spec.after.value
        )
        tied_comparison = (
            Asset.id < spec.after.id
            if descending
            else Asset.id > spec.after.id
        )
        statement = statement.where(
            sa.or_(
                comparison,
                sa.and_(
                    sort_column == spec.after.value,
                    tied_comparison,
                ),
            )
        )

    statement = statement.order_by(sort_expression, id_expression).limit(spec.limit)
    if spec.after is None:
        statement = statement.offset(spec.offset)
    records = list(session.scalars(statement))

    total = session.scalar(
        sa.select(sa.func.count())
        .select_from(Asset)
        .join(AssetContent, Asset.content_id == AssetContent.id)
        .where(*filters)
    )

    record_ids = [record.id for record in records]
    tag_map: dict[str, list[str]] = {}
    if record_ids:
        rows = session.execute(
            sa.select(AssetTag.asset_id, AssetTag.tag_name)
            .join(Asset, AssetTag.asset_id == Asset.id)
            .join(AssetContent, Asset.content_id == AssetContent.id)
            .where(AssetTag.asset_id.in_(record_ids))
            .order_by(AssetTag.tag_name.asc())
        )
        for record_id, tag_name in rows:
            tag_map.setdefault(record_id, []).append(tag_name)

    return records, tag_map, int(total or 0)


def rename_record(session: Session, id: str, name: str) -> Asset:
    record = session.get(Asset, id)
    if record is None:
        raise LookupError(id)
    record.name = name
    record.updated_at = get_utc_now()
    session.flush()
    return record


def delete_record(session: Session, id: str) -> None:
    record = session.get(Asset, id)
    if record is None:
        return
    session.delete(record)
    session.flush()


def mark_content_missing(session: Session, content_id: str) -> None:
    content = session.get(AssetContent, content_id)
    if content is None:
        raise LookupError(content_id)
    content.is_missing = True
    ensure_tag(session, "missing")
    for record_id in session.scalars(sa.select(Asset.id).where(Asset.content_id == content_id)):
        ensure_tag_link(session, asset_id=record_id, tag_name="missing", origin="automatic")
    session.flush()


def unset_content_missing(session: Session, content_id: str) -> None:
    content = session.get(AssetContent, content_id)
    if content is None:
        raise LookupError(content_id)
    content.is_missing = False
    session.execute(sa.delete(AssetTag).where(AssetTag.tag_name == "missing", AssetTag.asset_id.in_(sa.select(Asset.id).where(Asset.content_id == content_id))))
    session.flush()
