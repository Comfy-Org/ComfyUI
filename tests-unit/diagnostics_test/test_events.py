"""Tests for the ``[comfy-event]`` diagnostic lines (``comfy/diagnostics/events.py``)."""

import json
import logging
import re
import subprocess
import sys
from pathlib import Path

import pytest

from comfy.diagnostics import events
from comfy.diagnostics.events import (
    ALLOWED_EVENTS,
    ALLOWED_FIELDS,
    FEATURE_FLAG,
    NAMESPACES,
    TAG,
    EventLogError,
    emit,
)
from comfy_api.feature_flags import CLI_FEATURE_FLAG_REGISTRY, SERVER_FEATURE_FLAGS

REPO_ROOT = Path(__file__).resolve().parents[2]

# The line grammar below is the CONTRACT shared with launchers that tail core's
# output. It is the ``[assets-event]`` grammar with a different tag, an event of
# exactly ``<namespace>.<event>``, and field names that may contain digits
# (``p95_ms``). ``fixtures/core_event_lines.txt`` is copied byte for byte by
# consumers, so neither may change without them.
EVENT_LINE_PATTERN = re.compile(
    r"^\[comfy-event\] (?P<event>[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*)"
    r"(?P<fields>(?: [a-z][a-z0-9_]*=[^ =]+)*)$"
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "core_event_lines.txt"

# What launchers accept as an op; every registered op has to fit it.
OP_SHAPE = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*){1,3}")
REGISTERED_PERF_OPS = events.PERF_OPS
# Ops the tests below emit, standing in for the registry.
TEST_PERF_OPS = frozenset({
    "assets.scan.collect_paths",
    "assets.scan.enrich_hash",
    "assets.scan.insert",
    "assets.scan.mark_missing",
    "startup.db",
    "server.http.window",
})

# One valid value per allowed field, covering every enum member.
VALID_VALUES: dict[str, list[object]] = {
    "op": ["assets.scan.insert", "startup.db", "server.http.window"],
    "outcome": ["ok", "error", "cancelled"],
    "duration_ms": [0, 412],
    "cpu_ms": [0, 380],
    "count": [0, 1204],
    "bytes_total": [0, 73400320],
}


@pytest.fixture(autouse=True)
def test_perf_ops(monkeypatch):
    monkeypatch.setattr(events, "PERF_OPS", TEST_PERF_OPS)


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setitem(SERVER_FEATURE_FLAGS, FEATURE_FLAG, True)


@pytest.fixture
def production(monkeypatch):
    """Leave strict mode so invalid calls warn-and-drop instead of raising.

    Patched rather than clearing PYTEST_CURRENT_TEST, which pytest sets again
    after fixture setup.
    """
    monkeypatch.setattr(events, "_strict_mode", lambda: False)
    monkeypatch.setattr(events, "_warned_call_sites", set())


def fixture_lines() -> list[str]:
    return FIXTURE_PATH.read_text(encoding="utf-8").splitlines()


def parse_fields(raw: str) -> dict[str, bool | int | str]:
    fields: dict[str, bool | int | str] = {}
    for pair in raw.split():
        name, value = pair.split("=", maxsplit=1)
        if value in ("true", "false"):
            fields[name] = value == "true"
        elif value.isdigit():
            fields[name] = int(value)
        else:
            fields[name] = value
    return fields


def emit_line(capsys: pytest.CaptureFixture[str], event: str, **fields: object) -> str:
    """Emit one event and return the single line it wrote to stdout."""
    capsys.readouterr()
    emit(event, **fields)
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert len(lines) == 1, out
    assert out.endswith("\n")
    return lines[0]


# --- the shared cross-repo fixture -------------------------------------------------


def test_shared_fixture_file_holds_three_newline_terminated_lines():
    raw = FIXTURE_PATH.read_text(encoding="utf-8")

    assert raw.endswith("\n")
    assert len(raw.splitlines()) == 3


@pytest.mark.parametrize("line", fixture_lines())
def test_emit_reproduces_each_shared_fixture_line_byte_for_byte(capsys, flag_on, line):
    match = EVENT_LINE_PATTERN.match(line)
    assert match is not None, line
    fields = parse_fields(match.group("fields"))

    assert emit_line(capsys, match.group("event"), **fields) == line


# --- line shape and routing ---------------------------------------------------------


def test_fields_are_serialized_as_sorted_logfmt_on_stdout(capsys, flag_on):
    line = emit_line(capsys, "perf.timing", op="startup.db", duration_ms=7, outcome="ok")

    assert line == "[comfy-event] perf.timing duration_ms=7 op=startup.db outcome=ok"
    assert EVENT_LINE_PATTERN.match(line) is not None


def test_bools_render_lowercase_like_assets_events():
    assert events._format_line("perf.timing", {"flag": True, "off": False}) == (
        "[comfy-event] perf.timing flag=true off=false"
    )


def test_the_line_survives_a_warning_console_level(capsys, caplog, flag_on, monkeypatch):
    """--verbose WARNING raises the root level; the event logger ignores it."""
    monkeypatch.setattr(logging.getLogger(), "level", logging.WARNING)

    with caplog.at_level(logging.WARNING):
        line = emit_line(capsys, "perf.timing", op="startup.db", duration_ms=7)

    assert line.startswith(TAG)
    # The dedicated logger does not propagate, so root handlers (console with its
    # level prefix, file logs) never see a second copy.
    assert not [r for r in caplog.records if TAG in r.getMessage()]


def test_the_line_follows_sys_stdout_when_it_is_replaced_after_import(flag_on, monkeypatch):
    """app.logger swaps sys.stdout for its interceptor after this module is imported."""

    class Sink:
        def __init__(self):
            self.chunks: list[str] = []

        def write(self, data):
            self.chunks.append(data)

        def flush(self):
            pass

    sink = Sink()
    monkeypatch.setattr(sys, "stdout", sink)

    emit("perf.timing", op="startup.db", duration_ms=7)

    assert "".join(sink.chunks) == "[comfy-event] perf.timing duration_ms=7 op=startup.db\n"


# --- the feature flag ---------------------------------------------------------------


def test_the_flag_is_registered_for_launchers_as_an_off_by_default_bool():
    info = CLI_FEATURE_FLAG_REGISTRY[FEATURE_FLAG]

    assert info["type"] == "bool"
    assert info["default"] is False


def test_list_feature_flags_advertises_the_flag():
    """Launchers only pass the flag when this listing names it."""
    result = subprocess.run(
        [sys.executable, "main.py", "--list-feature-flags"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )

    listed = json.loads(result.stdout)
    assert listed[FEATURE_FLAG] == {
        "type": "bool",
        "default": False,
        "description": CLI_FEATURE_FLAG_REGISTRY[FEATURE_FLAG]["description"],
    }


@pytest.mark.parametrize("value", [None, False, "true", 1])
def test_anything_but_true_leaves_emit_disabled(monkeypatch, value):
    if value is None:
        monkeypatch.delitem(SERVER_FEATURE_FLAGS, FEATURE_FLAG, raising=False)
    else:
        monkeypatch.setitem(SERVER_FEATURE_FLAGS, FEATURE_FLAG, value)

    assert events.enabled() is False


def test_off_outside_strict_mode_writes_nothing_and_does_no_work(capsys, caplog, production, monkeypatch):
    monkeypatch.delitem(SERVER_FEATURE_FLAGS, FEATURE_FLAG, raising=False)

    def fail(*_args, **_kwargs):
        raise AssertionError("emit did work while the flag was off")

    monkeypatch.setattr(events, "_find_problem", fail)
    monkeypatch.setattr(events, "_format_line", fail)

    with caplog.at_level(logging.DEBUG):
        emit("perf.timing", op="startup.db", duration_ms=7)
        emit("not.an_event", secret="/home/x")

    assert capsys.readouterr() == ("", "")
    assert caplog.records == []


def test_off_under_pytest_still_rejects_a_bad_call_site(capsys, monkeypatch):
    """Tests run with the flag off by default, and must still catch bad call sites."""
    monkeypatch.delitem(SERVER_FEATURE_FLAGS, FEATURE_FLAG, raising=False)

    emit("perf.timing", op="startup.db", duration_ms=7)
    with pytest.raises(EventLogError):
        emit("perf.timing", op="startup.db", path="/home/x")

    assert capsys.readouterr().out == ""


# --- the closed vocabulary ----------------------------------------------------------


def test_every_event_belongs_to_a_registered_namespace():
    for event in ALLOWED_EVENTS:
        namespace, _, name = event.partition(".")
        assert namespace in NAMESPACES, event
        assert re.fullmatch(r"[a-z][a-z0-9_]*", name), event


def test_every_field_name_fits_the_line_grammar():
    for name in ALLOWED_FIELDS:
        assert re.fullmatch(r"[a-z][a-z0-9_]*", name), name


def test_every_registered_op_fits_the_shape_launchers_accept():
    for op in REGISTERED_PERF_OPS:
        assert len(op) <= 64 and OP_SHAPE.fullmatch(op), op


def test_the_valid_value_matrix_covers_every_allowed_field():
    assert set(VALID_VALUES) == set(ALLOWED_FIELDS)


@pytest.mark.parametrize(
    ("field", "value"),
    [(field, value) for field, values in VALID_VALUES.items() for value in values],
)
def test_each_valid_value_is_emitted(capsys, flag_on, field, value):
    line = emit_line(capsys, "perf.timing", **{"op": "startup.db", field: value})

    assert EVENT_LINE_PATTERN.match(line) is not None
    assert f" {field}={value}" in line


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("op", "assets.scan.unregistered"),
        ("op", "Assets.scan"),
        ("op", "assets"),
        ("op", "a.b.c.d.e"),
        ("op", "assets.scan/../x"),
        ("op", "assets.scan x"),
        ("op", "a." + "b" * 70),
        ("op", 3),
        ("outcome", "timeout"),
        ("outcome", "OK"),
        ("duration_ms", -1),
        ("duration_ms", 1.5),
        ("duration_ms", True),
        ("duration_ms", "5"),
        ("bytes_total", -10),
    ],
)
def test_invalid_values_are_rejected(flag_on, field, value):
    with pytest.raises(EventLogError, match=field):
        emit("perf.timing", **{"op": "startup.db", field: value})


def test_fields_passed_as_none_are_omitted(capsys, flag_on):
    line = emit_line(capsys, "perf.timing", op="startup.db", count=None, duration_ms=3)

    assert line == "[comfy-event] perf.timing duration_ms=3 op=startup.db"


def test_a_missing_required_field_is_rejected(flag_on):
    with pytest.raises(EventLogError, match="op"):
        emit("perf.timing", duration_ms=3)
    with pytest.raises(EventLogError, match="op"):
        emit("perf.timing", op=None, duration_ms=3)


@pytest.mark.parametrize("event", ["timing", "perf.unknown", "assets.enabled", "perf.timing.x", 3])
def test_unknown_events_are_rejected(flag_on, event):
    with pytest.raises(EventLogError, match="event"):
        emit(event, op="startup.db")


def test_unknown_fields_are_rejected(flag_on):
    with pytest.raises(EventLogError, match="error_message"):
        emit("perf.timing", op="startup.db", error_message="boom")


# --- production mode ----------------------------------------------------------------


def test_production_drops_an_invalid_event_and_warns_once_per_call_site(
    capsys, caplog, flag_on, production
):
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            emit("perf.timing", op="startup.db", path="/home/x")

    assert capsys.readouterr().out == ""
    warnings = [r.getMessage() for r in caplog.records]
    assert len(warnings) == 1
    assert "test_events.py" in warnings[0]
    assert "/home/x" not in warnings[0]
