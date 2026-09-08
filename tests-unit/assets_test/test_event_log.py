"""Tests for the structured assets event log lines (``app/assets/event_log.py``)."""

import logging
import re
from pathlib import Path

import pytest

from app.assets import event_log
from app.assets.event_log import ALLOWED_FIELDS, TAG, EventLogError, emit, error_type

# The line grammar below is the CONTRACT shared with the desktop launcher's log
# tap: Comfy-Org/Comfy-Desktop `src/main/lib/assetsTap.ts` holds the equivalent
# regex, and `tests-unit/assets_test/fixtures/assets_event_lines.txt` is a
# byte-identical copy of that repo's `src/main/lib/__fixtures__/assets-event-lines.txt`.
# Neither side may change without the other.
EVENT_LINE_PATTERN = re.compile(
    r"^\[assets-event\] (?P<event>[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*)"
    r"(?P<fields>(?: [a-z_]+=[^ =]+)*)$"
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "assets_event_lines.txt"

# One valid value per allowed field, covering every enum member so the desktop
# tap's mirrored validator matrix has a counterpart on this side.
VALID_VALUES: dict[str, list[object]] = {
    "root": ["models", "input", "output", "user", "temp"],
    "phase": ["fast", "enrich", "full"],
    "stage": ["mark_missing", "pruning", "fast_scan", "enrich", "finalize"],
    "elapsed_ms": [0, 8123],
    "created": [0, 12],
    "enriched": [4],
    "skipped": [3],
    "hash_failed": [2],
    "enrich_failed": [0],
    "permission_denied": [0],
    "count": [1],
    "error_type": ["ValueError", "FileNotFoundError"],
    "hashing_enabled": [True, False],
    "site": ["discovery", "enrich"],
}


@pytest.fixture(autouse=True)
def autoclean_unit_test_assets():
    """Shadow the conftest fixture of the same name.

    The conftest version reaches a running server to delete test-tagged assets,
    which transitively boots ComfyUI for every test in this directory. Nothing
    here touches a server or creates an asset, so the boot is pure cost.
    """
    yield


def fixture_lines() -> list[str]:
    return FIXTURE_PATH.read_text(encoding="utf-8").splitlines()


def parse_fields(raw: str) -> dict[str, bool | int | str]:
    fields: dict[str, bool | int | str] = {}
    for pair in raw.split():
        name, value = pair.split("=", maxsplit=1)
        if value == "true":
            fields[name] = True
        elif value == "false":
            fields[name] = False
        elif value.removeprefix("-").isdigit():
            fields[name] = int(value)
        else:
            fields[name] = value
    return fields


def emit_line(caplog: pytest.LogCaptureFixture, event: str, **fields: object) -> str:
    """Emit one event and return the single tagged line it produced."""
    caplog.clear()
    with caplog.at_level(logging.INFO):
        emit(event, **fields)
    tagged = [r.getMessage() for r in caplog.records if r.getMessage().startswith(TAG)]
    assert len(tagged) == 1, tagged
    return tagged[0]


def go_to_production_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave strict mode so invalid calls warn-and-drop instead of raising."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("COMFYUI_ASSETS_EVENT_LOG_STRICT", raising=False)
    event_log._warned_call_sites.clear()


# --- the shared cross-repo fixture -------------------------------------------------


def test_shared_fixture_file_holds_three_newline_terminated_lines():
    raw = FIXTURE_PATH.read_text(encoding="utf-8")

    assert raw.endswith("\n")
    assert len(raw.splitlines()) == 3


@pytest.mark.parametrize("line", fixture_lines())
def test_emit_reproduces_each_shared_fixture_line_byte_for_byte(caplog, line):
    """Given a canonical line, When its fields are re-emitted, Then the bytes match."""
    match = EVENT_LINE_PATTERN.match(line)
    assert match is not None, line
    fields = parse_fields(match.group("fields"))

    assert emit_line(caplog, match.group("event"), **fields) == line


# --- line shape ---------------------------------------------------------------------


def test_fields_are_serialized_as_sorted_logfmt(caplog):
    line = emit_line(caplog, "seeder.scan_started", root="models", phase="fast")

    assert line == "[assets-event] seeder.scan_started phase=fast root=models"


def test_a_fieldless_event_still_matches_the_shared_pattern(caplog):
    line = emit_line(caplog, "scanner.hash_discarded_modified")

    assert line == "[assets-event] scanner.hash_discarded_modified"
    assert EVENT_LINE_PATTERN.match(line) is not None


def test_the_emitted_record_is_a_single_line(caplog):
    line = emit_line(caplog, "seeder.scan_failed", error_type="ValueError")

    assert "\n" not in line
    assert "\r" not in line


# --- error_type ---------------------------------------------------------------------


def test_error_type_is_the_class_name_and_the_path_never_reaches_the_line(caplog):
    exc = FileNotFoundError("/home/x/model.safetensors")

    assert error_type(exc) == "FileNotFoundError"

    line = emit_line(caplog, "seeder.scan_failed", error_type=error_type(exc))
    assert "/home/x/model.safetensors" not in line
    assert "model.safetensors" not in line


# --- the closed vocabulary ----------------------------------------------------------


def test_the_valid_value_matrix_covers_every_allowed_field():
    assert set(VALID_VALUES) == set(ALLOWED_FIELDS)


@pytest.mark.parametrize(
    ("field", "value"),
    [(field, value) for field, values in VALID_VALUES.items() for value in values],
)
def test_every_allowed_field_value_round_trips(caplog, field, value):
    line = emit_line(caplog, "seeder.scan_completed", **{field: value})

    match = EVENT_LINE_PATTERN.match(line)
    assert match is not None, line
    assert parse_fields(match.group("fields")) == {field: value}


def test_unknown_field_raises_under_pytest():
    with pytest.raises(EventLogError):
        emit("seeder.scan_started", path="/home/x/models")


@pytest.mark.parametrize("value", ["a/b", "a\\b", "a:b", "a b", "a=b", 'a"b'])
def test_a_string_value_carrying_a_forbidden_character_raises(value):
    with pytest.raises(EventLogError):
        emit("seeder.scan_failed", error_type=value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("root", "checkpoints"),
        ("root", 1),
        ("phase", "quick"),
        ("phase", None),
        ("stage", "scanning"),
        ("site", "reference"),
        ("error_type", "x" * 65),
        ("error_type", 7),
        ("elapsed_ms", "8123"),
        ("count", 1.5),
        ("created", True),
        ("hashing_enabled", 1),
        ("hashing_enabled", "true"),
    ],
    ids=[
        "bad-root",
        "non-string-root",
        "bad-phase",
        "none-phase",
        "bad-stage",
        "bad-site",
        "oversized-string",
        "non-string-error-type",
        "string-into-int-field",
        "float-into-int-field",
        "bool-into-int-field",
        "int-into-bool-field",
        "string-into-bool-field",
    ],
)
def test_every_validator_rejects_its_bad_value(field, value):
    with pytest.raises(EventLogError):
        emit("seeder.scan_completed", **{field: value})


@pytest.mark.parametrize(
    "event",
    ["", "Seeder.scan_started", "seeder..scan", "9seeder.scan", "seeder.scan-started", "seeder scan", "seeder."],
)
def test_an_invalid_event_name_raises(event):
    with pytest.raises(EventLogError):
        emit(event)


# --- strict mode vs production mode -------------------------------------------------


def test_the_env_var_enables_strict_mode_without_pytest(monkeypatch):
    go_to_production_mode(monkeypatch)
    monkeypatch.setenv("COMFYUI_ASSETS_EVENT_LOG_STRICT", "1")

    with pytest.raises(EventLogError):
        emit("seeder.scan_started", path="/home/x")


def test_an_env_var_value_other_than_1_is_not_strict(caplog, monkeypatch):
    go_to_production_mode(monkeypatch)
    monkeypatch.setenv("COMFYUI_ASSETS_EVENT_LOG_STRICT", "true")

    with caplog.at_level(logging.WARNING):
        emit("seeder.scan_started", path="/home/x")

    assert [r for r in caplog.records if r.levelno == logging.WARNING]


def test_production_mode_warns_once_for_repeated_calls_from_one_call_site(caplog, monkeypatch):
    go_to_production_mode(monkeypatch)
    caplog.clear()

    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            emit("seeder.scan_started", path="/home/x/models")

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert not [r for r in caplog.records if r.getMessage().startswith(TAG)]
    assert "/home/x/models" not in caplog.text


def test_production_mode_warns_once_per_distinct_call_site(caplog, monkeypatch):
    go_to_production_mode(monkeypatch)
    caplog.clear()

    with caplog.at_level(logging.WARNING):
        emit("seeder.scan_started", path="/home/x/models")
        emit("seeder.scan_started", path="/home/x/models")

    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 2


def test_production_mode_still_emits_valid_events_after_a_dropped_one(caplog, monkeypatch):
    go_to_production_mode(monkeypatch)

    with caplog.at_level(logging.INFO):
        emit("seeder.scan_started", path="/home/x/models")
        emit("seeder.scan_started", phase="fast")

    tagged = [r.getMessage() for r in caplog.records if r.getMessage().startswith(TAG)]
    assert tagged == ["[assets-event] seeder.scan_started phase=fast"]
