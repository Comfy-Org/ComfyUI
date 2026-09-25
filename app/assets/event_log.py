"""Structured event log lines for the assets system.

Every line is ``[assets-event] <event> key=value ...`` on the standard logging
INFO channel, with fields sorted by name and omitted when an event has none. This
mirrors the ``assets.seed.*`` events the seeder already puts on the PromptServer
bus. A log-tailing launcher can pick assets health signals out of core's output
without parsing prose, and the existing human-readable lines stay exactly as
they are.

The field vocabulary is closed. Only the names in :data:`ALLOWED_FIELDS` may be
carried, each has a validator, and no string value may contain a path separator,
logfmt delimiter or line break — so file names, paths, asset ids and content
hashes cannot ride along.
"""

import logging
import os
import re
import traceback
from collections.abc import Callable
from typing import Any

from app.assets.failures import NO_WINERROR, REASONS, describe_failure

TAG = "[assets-event]"

MAX_STRING_LENGTH = 64
FORBIDDEN_STRING_CHARS = ("/", "\\", ":", " ", "=", '"', "\n", "\r")

ROOTS = frozenset({"models", "input", "output", "user", "temp"})
PHASES = frozenset({"fast", "enrich", "full"})
STAGES = frozenset({"mark_missing", "pruning", "fast_scan", "enrich", "finalize"})
SITES = frozenset({
    "discovery",
    "enrich",
    "reference",
    "seed_observation",
    "walk_root",
    "walk_dir",
    "hash",
    "metadata",
    "batch_insert",
    "watch_stat",
    "watch_spec",
    "watch_seed",
})
ALLOWED_EVENTS = frozenset({
    "assets.enabled",
    "seeder.scan_started",
    "seeder.scan_completed",
    "seeder.scan_failed",
    "seeder.scan_cancelled",
    "seeder.marked_missing",
    "seeder.batch_insert_failed",
    "scanner.hash_failed",
    "scanner.enrich_failed",
    "scanner.hash_discarded_modified",
    "scanner.fast_scan_failed",
    "scanner.temp_sync_failed",
    "scanner.mark_missing_failed",
    "scanner.stat_failed",
    "scanner.invalid_mtime",
    "scanner.watch_stat_failed",
    "scanner.watch_spec_failed",
    "scanner.watch_seed_failed",
    "scanner.failure_bucket",
    "scanner.root_unreachable",
    "scanner.walk_failed",
    "scanner.metadata_failed",
})


class EventLogError(ValueError):
    """An emit() call that would break the closed event vocabulary."""


def _is_safe_string(value: Any) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= MAX_STRING_LENGTH
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


def _matches(pattern: str) -> Callable[[Any], bool]:
    compiled = re.compile(pattern)

    def validate(value: Any) -> bool:
        return _is_safe_string(value) and compiled.fullmatch(value) is not None

    return validate


def _is_winerror(value: Any) -> bool:
    return _is_count(value) and (value == NO_WINERROR or 0 <= value <= 0xFFFF)


def _is_line(value: Any) -> bool:
    return _is_count(value) and value >= 0


ALLOWED_FIELDS: dict[str, Callable[[Any], bool]] = {
    "root": _one_of(ROOTS),
    "phase": _one_of(PHASES),
    "stage": _one_of(STAGES),
    "elapsed_ms": _is_count,
    "created": _is_count,
    "enriched": _is_count,
    "skipped": _is_count,
    "hash_failed": _is_count,
    "enrich_failed": _is_count,
    "permission_denied": _is_count,
    "count": _is_count,
    "error_type": _is_safe_string,
    "hashing_enabled": _is_flag,
    "site": _one_of(SITES),
    "reason": _one_of(REASONS),
    # A pattern rather than this interpreter's errno table, so a name emitted on one
    # platform (WSAESTALE on Windows) validates everywhere; mirrors the launcher tap.
    "errno_name": _matches(r"E[A-Z0-9]{1,23}|WSA[A-Z0-9_]{1,24}|none"),
    "winerror": _is_winerror,
    "exc_fp": _matches(r"[0-9a-f]{12}"),
    "exc_class": _matches(r"[A-Za-z_][A-Za-z0-9_.]*"),
    "exc_site": _matches(r"[A-Za-z_][A-Za-z0-9_.]*"),
    "exc_line": _is_line,
}

_warned_call_sites: set[tuple[str, int]] = set()


def _find_problem(event: Any, fields: dict[str, Any]) -> str | None:
    if not isinstance(event, str) or event not in ALLOWED_EVENTS:
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
    """Identify the caller outside this module, so a bad call site warns at most once
    whether it called emit() or emit_failure()."""
    for frame in reversed(traceback.extract_stack(limit=6)):
        if frame.filename != __file__:
            return (frame.filename, frame.lineno or 0)
    return (__file__, 0)


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
        pairs = " ".join(
            f"{name}={str(value).lower() if isinstance(value, bool) else value}"
            for name, value in sorted(fields.items())
        )
        line = f"{TAG} {event}" + (f" {pairs}" if pairs else "")
        logging.info("%s", line)
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


def emit_failure(event: str, exc: BaseException, *, root: str | None = None, **fields: Any) -> None:
    """emit() for a failure: ``error_type`` plus the classified reason and the
    fingerprint of the code that raised ``exc``, none of it read from the message."""
    failure = describe_failure(exc)
    emit(
        event,
        root=root,
        error_type=error_type(exc),
        reason=failure.reason,
        errno_name=failure.errno_name,
        winerror=failure.winerror,
        exc_fp=failure.exc_fp,
        exc_class=failure.exc_class,
        exc_site=failure.exc_site,
        exc_line=failure.exc_line,
        **fields,
    )
