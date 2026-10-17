"""Machine-readable diagnostic log lines for launchers.

When a launcher passes ``--feature-flag structured_log_events``, each :func:`emit`
call writes one ``[comfy-event] <namespace>.<event> key=value ...`` line to stdout,
with fields sorted by name. A launcher that tails the process output can then read
timings and failure classes without parsing prose. Nothing is sent anywhere.

When the flag is off, which is the default, :func:`emit` returns before doing any
work, and the output is byte-identical to a build without this module.

The lines use a dedicated logger that does not propagate and is not filtered by
``--verbose``, so ``--verbose WARNING`` does not hide them. They are written to
whatever ``sys.stdout`` is at emit time, so they also reach the in-app log buffer.

The vocabulary is closed. Namespaces, event names and field names are fixed sets,
and every field has a validator. String fields come from closed sets, so
paths, file names, prompt text and exception messages cannot be carried. The
assets system keeps its own ``[assets-event]`` lines (``app/assets/event_log.py``).
"""

import logging
import os
import sys
import traceback
from collections.abc import Callable
from typing import Any

from comfy_api.feature_flags import SERVER_FEATURE_FLAGS

TAG = "[comfy-event]"
FEATURE_FLAG = "structured_log_events"

NAMESPACES = frozenset({"execution", "models", "nodes", "server", "startup", "perf"})
ALLOWED_EVENTS = frozenset({
    "perf.timing",
})

# Fields an event cannot be emitted without.
REQUIRED_FIELDS: dict[str, frozenset[str]] = {
    "perf.timing": frozenset({"op"}),
}

# Every perf.timing op a call site may pass. Launchers only check an op's shape,
# so an op outside this set would reach them unreviewed.
PERF_OPS: frozenset[str] = frozenset()

OUTCOMES = frozenset({"ok", "error", "cancelled"})


class EventLogError(ValueError):
    """An emit() call that would break the closed event vocabulary."""


def _one_of(allowed: frozenset[str]) -> Callable[[Any], bool]:
    def validate(value: Any) -> bool:
        return isinstance(value, str) and value in allowed

    return validate


def _is_op(value: Any) -> bool:
    return isinstance(value, str) and value in PERF_OPS


def _is_non_negative_int(value: Any) -> bool:
    # bool subclasses int, so it has to be excluded before the int check.
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


ALLOWED_FIELDS: dict[str, Callable[[Any], bool]] = {
    "op": _is_op,
    "outcome": _one_of(OUTCOMES),
    "duration_ms": _is_non_negative_int,
    "cpu_ms": _is_non_negative_int,
    "count": _is_non_negative_int,
    "bytes_total": _is_non_negative_int,
}


class _CurrentStdoutHandler(logging.StreamHandler):
    """A stream handler bound to ``sys.stdout`` as it is at emit time.

    ``app.logger.setup_logger`` swaps ``sys.stdout`` for its log interceptor after
    this module may already have been imported. Binding at emit time keeps the
    lines in the in-app log buffer.
    """

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, _value):
        pass


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("comfy.events")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, _CurrentStdoutHandler) for h in logger.handlers):
        handler = _CurrentStdoutHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


_logger = _build_logger()
_warned_call_sites: set[tuple[str, int]] = set()
_STRICT_ENV = os.environ.get("COMFYUI_EVENT_LOG_STRICT") == "1"


def enabled() -> bool:
    """True when the launcher opted in with ``--feature-flag structured_log_events``."""
    return SERVER_FEATURE_FLAGS.get(FEATURE_FLAG) is True


def _find_problem(event: Any, fields: dict[str, Any]) -> str | None:
    if not isinstance(event, str) or event not in ALLOWED_EVENTS:
        return "invalid event name"
    missing = REQUIRED_FIELDS.get(event, frozenset()) - fields.keys()
    if missing:
        return f"required fields {sorted(missing)} are missing"
    for name, value in fields.items():
        validate = ALLOWED_FIELDS.get(name)
        if validate is None:
            return f"field {name!r} is not in the allowed vocabulary"
        if not validate(value):
            return f"field {name!r} has a value its validator rejected"
    return None


def _strict_mode() -> bool:
    # Checked on every call while the flag is off, so the common production case
    # (pytest never imported) costs one dict lookup.
    return _STRICT_ENV or (
        sys.modules.get("pytest") is not None and "PYTEST_CURRENT_TEST" in os.environ
    )


def _caller_call_site() -> tuple[str, int]:
    """Identify emit()'s caller so a bad call site warns at most once."""
    caller = traceback.extract_stack(limit=3)[0]
    return (caller.filename, caller.lineno or 0)


def _format_line(event: str, fields: dict[str, Any]) -> str:
    # No field takes a bool yet; launchers only accept lowercase true/false, the
    # same rendering as [assets-event], so a future flag field cannot drift.
    pairs = " ".join(
        f"{name}={str(value).lower() if isinstance(value, bool) else value}"
        for name, value in sorted(fields.items())
    )
    return f"{TAG} {event}" + (f" {pairs}" if pairs else "")


def emit(event: str, **fields: Any) -> None:
    """Write one tagged event line if the launcher opted in.

    Fields passed as None are omitted, so an optional value needs no branch at
    the call site. An invalid call raises in strict mode (under pytest, or with
    COMFYUI_EVENT_LOG_STRICT=1), even when the flag is off, so a bad call site
    fails the test suite whichever way it runs. In production it warns once per
    call site and drops the event, so a vocabulary mistake cannot break a running
    server.
    """
    is_enabled = enabled()
    if not is_enabled and not _strict_mode():
        return

    fields = {name: value for name, value in fields.items() if value is not None}
    problem = _find_problem(event, fields)
    if problem is None:
        if is_enabled:
            _logger.info("%s", _format_line(event, fields))
        return

    if _strict_mode():
        raise EventLogError(problem)

    call_site = _caller_call_site()
    if call_site not in _warned_call_sites:
        _warned_call_sites.add(call_site)
        logging.warning(
            "Dropped an invalid diagnostic event at %s:%d: %s",
            call_site[0],
            call_site[1],
            problem,
        )
