"""Answers questions about tag usage: which tags exist with how many assets, and
how those counts narrow once a filter is applied. Both queries reuse the record
listing's own joins and filter clauses, so the counts a client sees always
describe the same assets the listing endpoint would return for that filter,
records with missing content included.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing import Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.assets.database.models import (
    Asset,
    AssetContent,
    AssetTag,
    Tag,
)
from app.assets.database.queries.records import (
    build_record_tag_filter_clauses,
)
from app.assets.helpers import escape_sql_like_string


@dataclass(frozen=True)
class AddTagsResult:
    added: list[str]
    already_present: list[str]
    total_tags: list[str]


@dataclass(frozen=True)
class RemoveTagsResult:
    removed: list[str]
    not_present: list[str]
    total_tags: list[str]
    protected: list[str] = field(default_factory=list)


def list_tags_with_usage(
    session: Session,
    prefix: str | None = None,
    limit: int = 100,
    offset: int = 0,
    include_zero: bool = True,
    order: str = "count_desc",
) -> tuple[list[tuple[str, int]], int]:
    prefix_filter = prefix.strip() if prefix else ""

    counts_sq = (
        select(
            AssetTag.tag_name.label("tag_name"),
            func.count(AssetTag.asset_id).label("cnt"),
        )
        .select_from(AssetTag)
        .join(Asset, Asset.id == AssetTag.asset_id)
        .join(AssetContent, Asset.content_id == AssetContent.id)
        .group_by(AssetTag.tag_name)
        .subquery()
    )

    q = (
        select(
            Tag.name,
            func.coalesce(counts_sq.c.cnt, 0).label("count"),
        )
        .select_from(Tag)
        .join(counts_sq, counts_sq.c.tag_name == Tag.name, isouter=True)
    )

    if prefix_filter:
        q = q.where(func.substr(Tag.name, 1, len(prefix_filter)) == prefix_filter)

    if not include_zero:
        q = q.where(func.coalesce(counts_sq.c.cnt, 0) > 0)

    if order == "name_asc":
        q = q.order_by(Tag.name.asc())
    else:
        q = q.order_by(func.coalesce(counts_sq.c.cnt, 0).desc(), Tag.name.asc())

    total_q = select(func.count()).select_from(Tag)
    if prefix_filter:
        total_q = total_q.where(func.substr(Tag.name, 1, len(prefix_filter)) == prefix_filter)
    if not include_zero:
        visible_tags_sq = (
            select(AssetTag.tag_name)
            .join(Asset, Asset.id == AssetTag.asset_id)
            .join(AssetContent, Asset.content_id == AssetContent.id)
            .group_by(AssetTag.tag_name)
        )
        total_q = total_q.where(Tag.name.in_(visible_tags_sq))

    rows = (session.execute(q.limit(limit).offset(offset))).all()
    total = (session.execute(total_q)).scalar_one()

    rows_norm = [(name, int(count or 0)) for (name, count) in rows]
    return rows_norm, int(total or 0)


def list_tag_counts_for_filtered_assets(
    session: Session,
    include_tags: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
    name_contains: str | None = None,
    limit: int = 100,
    # Appended last so pre-existing positional callers keep binding correctly.
    any_tags: Sequence[str] | None = None,
) -> dict[str, int]:
    """Return {tag_name: count} for the assets matching the given filters.

    Reuses build_record_tag_filter_clauses and the same Asset->AssetContent inner
    join as list_records_page, so /api/assets and /api/assets/tags/refine agree on
    which assets a given all/any/none + name_contains filter selects — including
    missing-content records, which stay catalog-visible.
    """
    filters = list(
        build_record_tag_filter_clauses(
            tuple(include_tags or ()),
            tuple(any_tags or ()),
            tuple(exclude_tags or ()),
        )
    )
    if name_contains:
        escaped, esc = escape_sql_like_string(name_contains)
        filters.append(Asset.name.ilike(f"%{escaped}%", escape=esc))

    asset_sq = (
        select(Asset.id)
        .join(AssetContent, Asset.content_id == AssetContent.id)
        .where(*filters)
        .subquery()
    )

    q = (
        select(
            AssetTag.tag_name,
            func.count(AssetTag.asset_id).label("cnt"),
        )
        .where(AssetTag.asset_id.in_(select(asset_sq.c.id)))
        .group_by(AssetTag.tag_name)
        .order_by(func.count(AssetTag.asset_id).desc(), AssetTag.tag_name.asc())
        .limit(limit)
    )

    rows = session.execute(q).all()
    return {tag_name: int(cnt) for tag_name, cnt in rows}
