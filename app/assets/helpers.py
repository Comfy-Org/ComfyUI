"""Small conversions the asset system needs in more than one layer: canonical
stored-hash strings, UTC timestamps, tag normalization, and path containment
checks. The SQL predicate is deliberately case-sensitive and component-bounded,
because callers use it to choose rows for hard deletion; the Python matcher
follows ``Path.is_relative_to`` instead, including its platform case rules.
"""

import functools
import os
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta, timezone

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
    base, stem = _base_and_stem(prefix)
    return sa.or_(
        column == base,
        sa.func.substr(column, 1, len(stem)) == stem,
    )


def _base_and_stem(prefix: str) -> tuple[str, str]:
    base = os.path.abspath(prefix)
    return base, base if base.endswith(os.sep) else base + os.sep


def stored_path_under_prefixes(prefixes: list[str]) -> Callable[[str], bool]:
    """The Python twin of sql_path_under_prefix OR'd over ``prefixes``: the same
    case-sensitive string test on a stored path, for filtering rows already fetched.

    Unlike path_prefix_matcher, it neither normalizes nor normcases the path.
    """
    pairs = [_base_and_stem(prefix) for prefix in prefixes]
    bases = frozenset(base for base, _ in pairs)
    stems = tuple(stem for _, stem in pairs)
    return lambda path: path in bases or path.startswith(stems)


# Each prefix adds two terms to one flat OR, and SQLite rejects an expression deeper than
# 1000, so about 500 prefixes in one statement fail with "Expression tree is too large".
# SQLAlchemy flattens nested ORs, so more prefixes than this are split across statements,
# or filtered in Python with stored_path_under_prefixes.
PREFIX_BATCH_SIZE = 200


def sql_path_under_prefix_batches(
    column: ColumnElement[str], prefixes: list[str]
) -> list[ColumnElement[bool]]:
    """sql_path_under_prefix OR'd over each run of at most PREFIX_BATCH_SIZE prefixes.

    Run one statement per predicate and merge. Nested or overlapping prefixes can put a
    row in more than one batch, so the caller dedupes.
    """
    return [
        sa.or_(*(sql_path_under_prefix(column, p) for p in prefixes[i:i + PREFIX_BATCH_SIZE]))
        for i in range(0, len(prefixes), PREFIX_BATCH_SIZE)
    ]


def path_prefix_matcher(prefixes: Iterable[str]) -> Callable[[str], bool]:
    """Return ``path -> Path(path).is_relative_to(<any prefix>)``, with the prefixes
    normalized once.

    The startup prune tests every catalogued row against every owned prefix, and
    ``Path.is_relative_to`` walks the path's parents on each call, so a pathlib
    check there costs rows x prefixes x depth. A normcase'd, separator-bounded
    string prefix keeps its component bounds and platform case rules.
    """
    # abspath keeps exactly two leading separators, and pathlib treats that "//" as an
    # anchor of its own: "//server/f" is not under "/". Such paths are only matched
    # against prefixes with the same anchor.
    double = os.sep * 2
    exact: dict[bool, set[str]] = {False: set(), True: set()}
    stems: dict[bool, list[str]] = {False: [], True: []}
    for prefix in prefixes:
        base = os.path.normcase(os.path.abspath(prefix))
        is_double = base.startswith(double)
        exact[is_double].add(base)
        stems[is_double].append(base if base.endswith(os.sep) else base + os.sep)
    stem_tuples = {key: tuple(value) for key, value in stems.items()}

    def matches(path: str) -> bool:
        candidate = os.path.normcase(os.path.abspath(path))
        is_double = candidate.startswith(double)
        return candidate in exact[is_double] or candidate.startswith(stem_tuples[is_double])

    return matches


@functools.lru_cache(maxsize=None)
def cached_prefix_matcher(prefixes: tuple[str, ...]) -> Callable[[str], bool]:
    """path_prefix_matcher, built once per distinct prefix tuple, for callers that check
    every scanned file against the same folders.

    Keyed on the raw prefixes: normalizing them in the key would bring back the per-call
    cost this avoids. A folder-config change is just a new key.

    Precondition: the prefixes are absolute. That is not checked. main.py and
    extra_model_paths make their folders absolute, but folder_paths' setters and
    add_model_folder_path store whatever they are given, so a custom node can register a
    relative one. A relative prefix is resolved against the working directory once, on
    first use, and that resolution is then frozen for as long as the key is unchanged.
    """
    return path_prefix_matcher(prefixes)


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


def mtime_ns_to_utc(mtime_ns: int, ctime_ns: int) -> datetime:
    """Naive UTC time of a file mtime. A future mtime falls back to the ctime (creation time on
    Windows), capped at now, so the file is dated no later than it arrived."""
    now = get_utc_now()
    made = datetime(1970, 1, 1) + timedelta(microseconds=mtime_ns // 1000)
    if made <= now:
        return made
    return min(datetime(1970, 1, 1) + timedelta(microseconds=ctime_ns // 1000), now)


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
