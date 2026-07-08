import os
from datetime import datetime, timezone
from typing import Sequence


def select_best_live_path(states: Sequence) -> str:
    """
    Return the best on-disk path among cache states:
      1) Prefer a path that exists with needs_verify == False (already verified).
      2) Otherwise, pick the first path that exists.
      3) Otherwise return empty string.
    """
    alive = [
        s
        for s in states
        if getattr(s, "file_path", None) and os.path.isfile(s.file_path)
    ]
    if not alive:
        return ""
    for s in alive:
        if not getattr(s, "needs_verify", False):
            return s.file_path
    return alive[0].file_path


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
      - Stripping whitespace and converting to lowercase.
      - Removing duplicates.
    """
    return list(dict.fromkeys(t.strip().lower() for t in (tags or []) if (t or "").strip()))


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
