import os
from datetime import datetime, timezone

import sqlalchemy as sa
from sqlalchemy.sql import ColumnElement


def sql_path_under_prefix(
    column: ColumnElement[str], prefix: str
) -> ColumnElement[bool]:
    """SQL predicate for ``Path(column).is_relative_to(prefix)`` on this platform.

    Case-SENSITIVE and component-bounded: prefix ``/a/b`` matches ``/a/b`` and
    ``/a/b/c`` but not ``/a/bc``, ``/a/b-other`` or ``/a/B/c``.

    ``LIKE`` cannot express this. SQLite's ``LIKE`` is ASCII case-insensitive by
    default, so ``'/data/TEMP/f' LIKE '/data/temp/%'`` is TRUE — which let the
    temp wipe hard-delete records under a case-different persistent directory,
    and let the enrichment scan mutate rows outside the requested root.
    ``GLOB`` is case-sensitive but carries its own metacharacters (``*``, ``?``,
    ``[``) with no ESCAPE clause, so every caller would need bracket-quoting.
    ``substr(column, 1, n) = <prefix>`` compares under the column's BINARY
    collation and has no metacharacters at all, so a path containing ``%``,
    ``_``, ``*``, ``?`` or ``[`` needs no escaping and cannot inject.

    Only the PREFIX is normalized here. That is sound because the column holds
    normalized absolute paths — ``records.create_content`` is the sole writer
    and normalizes there. Normalizing the column in SQL is not an option anyway:
    it would need a per-row Python call and would defeat the index.
    """
    base = os.path.abspath(prefix)
    stem = base if base.endswith(os.sep) else base + os.sep
    return sa.or_(
        column == base,
        sa.func.substr(column, 1, len(stem)) == stem,
    )


def escape_sql_like_string(s: str, escape: str = "!") -> tuple[str, str]:
    """Escapes %, _ and the escape char in a LIKE prefix.

    Returns (escaped_prefix, escape_char).
    """
    s = s.replace(escape, escape + escape)  # escape the escape char first
    s = s.replace("%", escape + "%").replace("_", escape + "_")  # escape LIKE wildcards
    return s, escape


def get_utc_now() -> datetime:
    """Naive UTC timestamp (no tzinfo). We always treat DB datetimes as UTC."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def normalize_tags(tags: list[str] | None) -> list[str]:
    """
    Normalize a list of tags by:
      - Stripping whitespace.
      - Removing exact duplicates while preserving order and case.
    """
    return list(dict.fromkeys(t.strip() for t in (tags or []) if (t or "").strip()))


def to_stored_hash(digest: str) -> str:
    return f"blake3:{digest}"


def validate_blake3_hash(s: str) -> str:
    """Validate and normalize a blake3 hash string.

    Returns canonical 'blake3:<hex>' or raises ValueError.
    """
    s = s.strip().lower()
    if not s or ":" not in s:
        raise ValueError("hash must be 'blake3:<hex>'")
    algo, digest = s.split(":", 1)
    if (
        algo != "blake3"
        or len(digest) != 64
        or any(c for c in digest if c not in "0123456789abcdef")
    ):
        raise ValueError("hash must be 'blake3:<hex>'")
    return f"{algo}:{digest}"
