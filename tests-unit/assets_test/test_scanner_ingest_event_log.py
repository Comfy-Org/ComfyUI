import json
import logging
import re
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.assets import scanner
from app.assets.event_log import TAG
from app.assets.scanner import UnenrichedContent
from app.assets.seeder import Progress
from app.assets.services import ingest


EVENT_LINE_PATTERN = re.compile(
    rf"^{re.escape(TAG)} (?P<event>[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*) "
    r"(?P<fields>\{.*\})$"
)
EventFields = dict[str, bool | int | str]


@pytest.fixture(autouse=True)
def autoclean_unit_test_assets():
    yield


def tagged_events(caplog: pytest.LogCaptureFixture) -> list[tuple[str, EventFields]]:
    events: list[tuple[str, EventFields]] = []
    for record in caplog.records:
        match = EVENT_LINE_PATTERN.match(record.getMessage())
        if match is not None:
            events.append((match.group("event"), json.loads(match.group("fields"))))
    return events


def events_named(
    caplog: pytest.LogCaptureFixture, event_name: str
) -> list[EventFields]:
    return [fields for event, fields in tagged_events(caplog) if event == event_name]


def tagged_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.getMessage().startswith(TAG)]


def hash_session(path: Path) -> Mock:
    stat_result = path.stat()
    content = SimpleNamespace(hash=None, mtime_ns=stat_result.st_mtime_ns)
    record = SimpleNamespace(system_metadata=None, mime_type=None)
    session = Mock()
    session.get.side_effect = lambda _model, row_id: content if row_id == "content" else record
    return session


def run_hash_failure(session: Mock, path: Path, progress: Progress) -> bool:
    return scanner.enrich_asset(
        session,
        file_path=str(path),
        content_id="content",
        record_id="record",
        extract_metadata=False,
        compute_hash=True,
        progress=progress,
    )


@pytest.mark.parametrize(
    ("operation", "event_name", "expected_fields", "expected_result"),
    [
        pytest.param(
            lambda: scanner.sync_root_safely("models"),
            "scanner.fast_scan_failed",
            {"error_type": "FileNotFoundError", "root": "models"},
            set(),
            id="fast-scan",
        ),
        pytest.param(
            scanner.sync_temp_references_safely,
            "scanner.temp_sync_failed",
            {"error_type": "FileNotFoundError", "root": "temp"},
            None,
            id="temp-sync",
        ),
        pytest.param(
            lambda: scanner.mark_missing_outside_prefixes_safely([]),
            "scanner.mark_missing_failed",
            {"error_type": "FileNotFoundError"},
            0,
            id="mark-missing",
        ),
    ],
)
def test_scanner_safe_failures_emit_exception_type_without_path(
    operation,
    event_name: str,
    expected_fields: EventFields,
    expected_result,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret_path = "/private/assets/secret.safetensors"

    def fail_session():
        raise FileNotFoundError(secret_path)

    monkeypatch.setattr(scanner, "create_session", fail_session)

    with caplog.at_level(logging.INFO):
        result = operation()

    assert result == expected_result
    assert events_named(caplog, event_name) == [expected_fields]
    assert all(secret_path not in line for line in tagged_lines(caplog))


def test_permission_error_in_reference_sync_increments_scan_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_path = "/private/assets/unreadable.safetensors"
    content = SimpleNamespace(id="content", path=secret_path)
    progress = Progress()

    def deny_stat(*_args, **_kwargs):
        raise PermissionError(secret_path)

    monkeypatch.setattr(scanner, "os", SimpleNamespace(stat=deny_stat, path=scanner.os.path))
    monkeypatch.setattr(scanner, "live_contents_under_prefixes", lambda _session, _prefixes: [content])

    scanner.sync_prefixes_with_filesystem(Mock(), ["/private/assets"], progress=progress)

    assert progress.permission_denied == 1


def test_hash_failures_emit_once_per_scan_and_count_every_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"model")
    progress = Progress()

    def fail_hash(_path: str):
        raise FileNotFoundError(str(path))

    monkeypatch.setattr(scanner, "snapshot_hash", fail_hash)
    session = hash_session(path)

    with caplog.at_level(logging.INFO):
        assert run_hash_failure(session, path, progress) is False
        assert run_hash_failure(session, path, progress) is False

    assert events_named(caplog, "scanner.hash_failed") == [
        {"error_type": "FileNotFoundError"}
    ]
    assert progress.hash_failed == 2


def test_hash_failure_first_occurrence_resets_with_new_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"model")

    def fail_hash(_path: str):
        raise OSError("hash unavailable")

    monkeypatch.setattr(scanner, "snapshot_hash", fail_hash)
    session = hash_session(path)

    with caplog.at_level(logging.INFO):
        run_hash_failure(session, path, Progress())
        run_hash_failure(session, path, Progress())

    assert events_named(caplog, "scanner.hash_failed") == [
        {"error_type": "OSError"},
        {"error_type": "OSError"},
    ]


def test_hash_failure_tagged_line_does_not_leak_exception_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "private-model.safetensors"
    path.write_bytes(b"model")

    def fail_hash(_path: str):
        raise FileNotFoundError(str(path))

    monkeypatch.setattr(scanner, "snapshot_hash", fail_hash)

    with caplog.at_level(logging.INFO):
        run_hash_failure(hash_session(path), path, Progress())

    assert events_named(caplog, "scanner.hash_failed") == [
        {"error_type": "FileNotFoundError"}
    ]
    assert any(str(path) in record.getMessage() for record in caplog.records)
    assert all(str(path) not in line for line in tagged_lines(caplog))


def test_modified_during_hash_emits_fieldless_discard_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "changing.safetensors"
    path.write_bytes(b"model")
    monkeypatch.setattr(scanner, "snapshot_hash", lambda _path: None)

    with caplog.at_level(logging.INFO):
        updated = scanner.enrich_asset(
            hash_session(path),
            file_path=str(path),
            content_id="content",
            record_id="record",
            extract_metadata=False,
            compute_hash=True,
            progress=Progress(),
        )

    assert updated is False
    assert events_named(caplog, "scanner.hash_discarded_modified") == [{}]


def test_enrich_failures_emit_once_per_scan_and_reset_with_new_scan(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    rows = [
        UnenrichedContent("content-1", "record-1", "/private/assets/one.bin"),
        UnenrichedContent("content-2", "record-2", "/private/assets/two.bin"),
    ]
    monkeypatch.setattr(scanner, "create_session", lambda: nullcontext(Mock()))

    def fail_enrich(*_args, **_kwargs):
        raise FileNotFoundError("/private/assets/secret.bin")

    monkeypatch.setattr(scanner, "enrich_asset", fail_enrich)

    with caplog.at_level(logging.INFO):
        first_result = scanner.enrich_assets_batch(rows, progress=Progress())
        second_result = scanner.enrich_assets_batch(rows[:1], progress=Progress())

    assert first_result == (0, ["record-1", "record-2"])
    assert second_result == (0, ["record-1"])
    assert events_named(caplog, "scanner.enrich_failed") == [
        {"error_type": "FileNotFoundError"},
        {"error_type": "FileNotFoundError"},
    ]


def test_enrich_exception_counts_one_failure_per_raising_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        UnenrichedContent("content-1", "record-1", "/private/assets/one.bin"),
        UnenrichedContent("content-2", "record-2", "/private/assets/two.bin"),
    ]
    monkeypatch.setattr(scanner, "create_session", lambda: nullcontext(Mock()))

    def fail_enrich(*_args, **_kwargs):
        raise FileNotFoundError("/private/assets/secret.bin")

    monkeypatch.setattr(scanner, "enrich_asset", fail_enrich)
    progress = Progress()

    enriched, failed_ids = scanner.enrich_assets_batch(rows, progress=progress)

    assert enriched == 0
    assert failed_ids == ["record-1", "record-2"]
    assert progress.enrich_failed == 2


def test_benign_enrich_no_op_is_skipped_without_counting_a_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deleted = tmp_path / "gone.safetensors"
    rows = [UnenrichedContent("content-1", "record-1", str(deleted))]
    monkeypatch.setattr(scanner, "create_session", lambda: nullcontext(Mock()))
    progress = Progress()

    enriched, failed_ids = scanner.enrich_assets_batch(rows, progress=progress)

    assert enriched == 0
    assert failed_ids == ["record-1"]
    assert progress.enrich_failed == 0


def test_both_register_output_failures_emit_the_shared_line_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret_path = str(tmp_path / "private-output.png")

    def fail_session():
        raise FileNotFoundError(secret_path)

    monkeypatch.setattr(ingest, "create_session", fail_session)

    with caplog.at_level(logging.INFO):
        assert ingest.register_cached_output(secret_path) is None
        assert ingest.register_executed_output(secret_path) is None

    assert events_named(caplog, "ingest.register_output_failed") == [
        {"error_type": "FileNotFoundError"},
        {"error_type": "FileNotFoundError"},
    ]
    assert all(EVENT_LINE_PATTERN.match(line) is not None for line in tagged_lines(caplog))
    assert all(secret_path not in line for line in tagged_lines(caplog))


def test_orphan_cleanup_failure_emits_exception_type(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = Mock()
    session.scalar.side_effect = FileNotFoundError("/private/assets/orphan.bin")

    with caplog.at_level(logging.INFO):
        ingest._discard_unreferenced_content(session, "content-id")

    assert events_named(caplog, "ingest.discard_orphan_failed") == [
        {"error_type": "FileNotFoundError"}
    ]
