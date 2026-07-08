"""Shared utilities for database query modules."""

import os
from decimal import Decimal
from typing import Iterable, Sequence

import sqlalchemy as sa
from sqlalchemy import exists

from app.assets.database.models import AssetReference, AssetReferenceMeta, AssetReferenceTag
from app.assets.helpers import escape_sql_like_string, normalize_tags

MAX_BIND_PARAMS = 800


def calculate_rows_per_statement(cols: int) -> int:
    """Calculate how many rows can fit in one statement given column count."""
    return max(1, MAX_BIND_PARAMS // max(1, cols))


def iter_chunks(seq, n: int):
    """Yield successive n-sized chunks from seq."""
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def iter_row_chunks(rows: list[dict], cols_per_row: int) -> Iterable[list[dict]]:
    """Yield chunks of rows sized to fit within bind param limits."""
    if not rows:
        return
    yield from iter_chunks(rows, calculate_rows_per_statement(cols_per_row))


def build_visible_owner_clause(owner_id: str) -> sa.sql.ClauseElement:
    """Build owner visibility predicate for reads.

    Owner-less rows are visible to everyone.
    """
    owner_id = (owner_id or "").strip()
    if owner_id == "":
        return AssetReference.owner_id == ""
    return AssetReference.owner_id.in_(["", owner_id])


def build_prefix_like_conditions(
    prefixes: list[str],
) -> list[sa.sql.ColumnElement]:
    """Build LIKE conditions for matching file paths under directory prefixes."""
    conds = []
    for p in prefixes:
        base = os.path.abspath(p)
        if not base.endswith(os.sep):
            base += os.sep
        escaped, esc = escape_sql_like_string(base)
        conds.append(AssetReference.file_path.like(escaped + "%", escape=esc))
    return conds


def apply_tag_filters(
    stmt: sa.sql.Select,
    include_tags: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
) -> sa.sql.Select:
    """include_tags: every tag must be present; exclude_tags: none may be present."""
    include_tags = normalize_tags(include_tags)
    exclude_tags = normalize_tags(exclude_tags)

    if include_tags:
        for tag_name in include_tags:
            stmt = stmt.where(
                exists().where(
                    (AssetReferenceTag.asset_reference_id == AssetReference.id)
                    & (AssetReferenceTag.tag_name == tag_name)
                )
            )

    if exclude_tags:
        stmt = stmt.where(
            ~exists().where(
                (AssetReferenceTag.asset_reference_id == AssetReference.id)
                & (AssetReferenceTag.tag_name.in_(exclude_tags))
            )
        )
    return stmt


def apply_metadata_filter(
    stmt: sa.sql.Select,
    metadata_filter: dict | None = None,
) -> sa.sql.Select:
    """Apply filters using asset_reference_meta projection table."""
    if not metadata_filter:
        return stmt

    def _exists_for_pred(key: str, *preds) -> sa.sql.ClauseElement:
        return sa.exists().where(
            AssetReferenceMeta.asset_reference_id == AssetReference.id,
            AssetReferenceMeta.key == key,
            *preds,
        )

    def _exists_clause_for_value(key: str, value) -> sa.sql.ClauseElement:
        if value is None:
            return sa.not_(
                sa.exists().where(
                    AssetReferenceMeta.asset_reference_id == AssetReference.id,
                    AssetReferenceMeta.key == key,
                )
            )

        if isinstance(value, bool):
            return _exists_for_pred(key, AssetReferenceMeta.val_bool == bool(value))
        if isinstance(value, (int, float, Decimal)):
            num = value if isinstance(value, Decimal) else Decimal(str(value))
            return _exists_for_pred(key, AssetReferenceMeta.val_num == num)
        if isinstance(value, str):
            return _exists_for_pred(key, AssetReferenceMeta.val_str == value)
        return _exists_for_pred(key, AssetReferenceMeta.val_json == value)

    for k, v in metadata_filter.items():
        if isinstance(v, list):
            ors = [_exists_clause_for_value(k, elem) for elem in v]
            if ors:
                stmt = stmt.where(sa.or_(*ors))
        else:
            stmt = stmt.where(_exists_clause_for_value(k, v))
    return stmt
