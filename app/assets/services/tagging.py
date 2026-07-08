from typing import Sequence

from app.assets.database.queries import (
    AddTagsResult,
    RemoveTagsResult,
    add_tags_to_reference,
    get_reference_with_owner_check,
    list_tags_with_usage,
    remove_tags_from_reference,
)
from app.assets.database.queries.tags import list_tag_counts_for_filtered_assets
from app.assets.services.schemas import TagUsage
from app.database.db import create_session


def apply_tags(
    reference_id: str,
    tags: list[str],
    origin: str = "manual",
    owner_id: str = "",
) -> AddTagsResult:
    with create_session() as session:
        ref_row = get_reference_with_owner_check(session, reference_id, owner_id)

        result = add_tags_to_reference(
            session,
            reference_id=reference_id,
            tags=tags,
            origin=origin,
            create_if_missing=True,
            reference_row=ref_row,
        )
        session.commit()

    return result


def remove_tags(
    reference_id: str,
    tags: list[str],
    owner_id: str = "",
) -> RemoveTagsResult:
    with create_session() as session:
        get_reference_with_owner_check(session, reference_id, owner_id)

        result = remove_tags_from_reference(
            session,
            reference_id=reference_id,
            tags=tags,
        )
        session.commit()

    return result


def list_tags(
    prefix: str | None = None,
    limit: int = 100,
    offset: int = 0,
    order: str = "count_desc",
    include_zero: bool = True,
    owner_id: str = "",
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
            owner_id=owner_id,
        )

    return [TagUsage(name, count) for name, count in rows], total


def list_tag_histogram(
    owner_id: str = "",
    include_tags: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
    name_contains: str | None = None,
    metadata_filter: dict | None = None,
    limit: int = 100,
) -> dict[str, int]:
    with create_session() as session:
        return list_tag_counts_for_filtered_assets(
            session,
            owner_id=owner_id,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            name_contains=name_contains,
            metadata_filter=metadata_filter,
            limit=limit,
        )
