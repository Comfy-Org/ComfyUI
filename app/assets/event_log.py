"""Structured event log lines for the assets system.

Every line is ``[assets-event] <event> <compact-json>`` on the standard logging
INFO channel, mirroring the ``assets.seed.*`` events the seeder already puts on
the PromptServer bus. A log-tailing launcher can pick assets health signals out
of core's output without parsing prose, and the existing human-readable lines
stay exactly as they are.

The field vocabulary is closed. Only the names in :data:`ALLOWED_FIELDS` may be
carried, each has a validator, and no value may contain a path separator — so
file names, paths, asset ids and content hashes cannot ride along.
"""

import json
import logging
import os
import re
import traceback
from collections.abc import Callable
from typing import Any

TAG = "[assets-event]"

MAX_STRING_LENGTH = 64
FORBIDDEN_STRING_CHARS = ("/", "\\", ":")

EVENT_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")

ROOTS = frozenset({"models", "input", "output", "user", "temp"})
PHASES = frozenset({"fast", "enrich", "full"})
STAGES = frozenset({"mark_missing", "pruning", "fast_scan", "enrich", "finalize"})
ROUTES = frozenset(
    {
        "get_asset_route",
        "upload_asset",
        "update_asset_route",
        "delete_asset_route",
        "add_asset_tags",
        "delete_asset_tags",
        "parse_multipart_upload",
    }
)
SIZE_BUCKETS = frozenset({"lt_1m", "lt_100m", "lt_1g", "ge_1g"})


class EventLogError(ValueError):
    """An emit() call that would break the closed event vocabulary."""


def _is_safe_string(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) <= MAX_STRING_LENGTH
        and not any(char in value for char in FORBIDDEN_STRING_CHARS)
    )


def _one_of(allowed: frozenset[str]) -> Callable[[Any], bool]:
    def validate(value: Any) -> bool:
        return _is_safe_string(value) and value in allowed

    return validate


def _is_count(value: Any) -> bool:
    # bool subclasses int, so it has to be excluded before the int check.
    return isinstance(value, int) and not isinstance(value, bool)


def _is_flag(value: Any) -> bool:
    return isinstance(value, bool)


ALLOWED_FIELDS: dict[str, Callable[[Any], bool]] = {
    "root": _one_of(ROOTS),
    "phase": _one_of(PHASES),
    "stage": _one_of(STAGES),
    "route": _one_of(ROUTES),
    "size_bucket": _one_of(SIZE_BUCKETS),
    "elapsed_ms": _is_count,
    "created": _is_count,
    "enriched": _is_count,
    "skipped": _is_count,
    "marked_missing": _is_count,
    "hash_failed": _is_count,
    "enrich_failed": _is_count,
    "permission_denied": _is_count,
    "count": _is_count,
    "error_type": _is_safe_string,
    "hashing_enabled": _is_flag,
}

_warned_call_sites: set[tuple[str, int]] = set()


def _find_problem(event: Any, fields: dict[str, Any]) -> str | None:
    if not isinstance(event, str) or EVENT_NAME_PATTERN.match(event) is None:
        return "invalid event name"
    for name, value in fields.items():
        validate = ALLOWED_FIELDS.get(name)
        if validate is None:
            return f"field {name!r} is not in the allowed vocabulary"
        if not validate(value):
            return f"field {name!r} has a value its validator rejected"
    return None


def _strict_mode() -> bool:
    return (
        "PYTEST_CURRENT_TEST" in os.environ
        or os.environ.get("COMFYUI_ASSETS_EVENT_LOG_STRICT") == "1"
    )


def _caller_call_site() -> tuple[str, int]:
    """Identify emit()'s caller so a bad call site warns at most once."""
    caller = traceback.extract_stack(limit=3)[0]
    return (caller.filename, caller.lineno or 0)


def emit(event: str, *, root: str | None = None, **fields: Any) -> None:
    """Log one tagged event line.

    An invalid call raises in strict mode (under pytest, or with
    COMFYUI_ASSETS_EVENT_LOG_STRICT=1) so a bad call site fails the test suite.
    In production it warns once per call site and drops the event, so a
    vocabulary mistake can never break a running server.
    """
    if root is not None:
        fields["root"] = root

    problem = _find_problem(event, fields)
    if problem is None:
        logging.info(
            "%s %s %s",
            TAG,
            event,
            json.dumps(fields, sort_keys=True, separators=(",", ":")),
        )
        return

    if _strict_mode():
        raise EventLogError(problem)

    call_site = _caller_call_site()
    if call_site not in _warned_call_sites:
        _warned_call_sites.add(call_site)
        logging.warning(
            "Dropped an invalid assets event at %s:%d: %s",
            call_site[0],
            call_site[1],
            problem,
        )


def error_type(exc: BaseException) -> str:
    """The only sanctioned description of an exception: its class name.

    Stringifying the exception itself is banned here, because FileNotFoundError
    and friends embed the path that triggered them.
    """
    return type(exc).__name__
