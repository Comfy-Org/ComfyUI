"""Applies and removes tags on a record and answers tag listing and histogram
queries. Adding reports what it actually inserted, read back after the write,
so a caller that lost a race is told the tag was already present rather than
that it added one. Removal leaves automatically applied tags in place and
reports them as protected instead of silently skipping them.
"""

from typing import Sequence

from sqlalchemy import delete, select

from app.assets.database.queries import (
    AddTagsResult,
    RemoveTagsResult,
    list_tags_with_usage,
)
from app.assets.database.queries.records import (
    bump_record_updated_at,
    ensure_tag,
    ensure_tag_link,
)
from app.assets.database.queries.tags import list_tag_counts_for_filtered_assets
from app.assets.database.models import Asset, AssetTag
from app.assets.helpers import normalize_tags
from app.assets.services.schemas import TagUsage
from app.database.db import create_session


def apply_tags(
    reference_id: str,
    tags: list[str],
    origin: str = "manual",
) -> AddTagsResult:
    with create_session() as session:
        if session.get(Asset, reference_id) is None:
            raise ValueError(f"Asset {reference_id} not found")

        normalized_tags = normalize_tags(tags)
        requested_tags = set(normalized_tags)
        added: list[str] = []
        for tag_name in normalized_tags:
            ensure_tag(session, tag_name)
            if ensure_tag_link(
                session,
                asset_id=reference_id,
                tag_name=tag_name,
                origin=origin,
            ):
                added.append(tag_name)
        session.flush()
        if added:
            bump_record_updated_at(session, reference_id)
        total_tags = list(
            session.scalars(
                select(AssetTag.tag_name)
                .where(AssetTag.asset_id == reference_id)
                .order_by(AssetTag.tag_name)
            )
        )
        session.commit()

    return AddTagsResult(
        added=sorted(added),
        already_present=sorted((requested_tags & set(total_tags)) - set(added)),
        total_tags=total_tags,
    )


def remove_tags(
    reference_id: str,
    tags: list[str],
) -> RemoveTagsResult:
    with create_session() as session:
        if session.get(Asset, reference_id) is None:
            raise ValueError(f"Asset {reference_id} not found")

        requested_tags = set(normalize_tags(tags))
        removable_tags = set(
            session.scalars(
                select(AssetTag.tag_name).where(
                    AssetTag.asset_id == reference_id,
                    AssetTag.origin != "automatic",
                    AssetTag.tag_name.in_(requested_tags),
                )
            )
        )
        protected_tags = set(
            session.scalars(
                select(AssetTag.tag_name).where(
                    AssetTag.asset_id == reference_id,
                    AssetTag.origin == "automatic",
                    AssetTag.tag_name.in_(requested_tags),
                )
            )
        )
        if removable_tags:
            session.execute(
                delete(AssetTag).where(
                    AssetTag.asset_id == reference_id,
                    AssetTag.origin != "automatic",
                    AssetTag.tag_name.in_(removable_tags),
                )
            )
            bump_record_updated_at(session, reference_id)
        total_tags = list(
            session.scalars(
                select(AssetTag.tag_name)
                .where(AssetTag.asset_id == reference_id)
                .order_by(AssetTag.tag_name)
            )
        )
        session.commit()

    return RemoveTagsResult(
        removed=sorted(removable_tags),
        not_present=sorted(requested_tags - removable_tags - protected_tags),
        total_tags=total_tags,
        protected=sorted(protected_tags),
    )


def list_tags(
    prefix: str | None = None,
    limit: int = 100,
    offset: int = 0,
    order: str = "count_desc",
    include_zero: bool = True,
) -> tuple[list[TagUsage], int]:
    limit = max(1, min(1000, limit))
    offset = max(0, offset)

    with create_session() as session:
        rows, total = list_tags_with_usage(
            session,
            prefix=prefix,
            limit=limit,
            offset=offset,
            include_zero=include_zero,
            order=order,
        )

    return [TagUsage(name, count) for name, count in rows], total


def list_tag_histogram(
    include_tags: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
    name_contains: str | None = None,
    limit: int = 100,
    # Appended last so pre-existing positional callers keep binding correctly.
    any_tags: Sequence[str] | None = None,
) -> dict[str, int]:
    with create_session() as session:
        return list_tag_counts_for_filtered_assets(
            session,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            any_tags=any_tags,
            name_contains=name_contains,
            limit=limit,
        )
