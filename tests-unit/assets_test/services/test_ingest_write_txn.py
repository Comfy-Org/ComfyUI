import os

import logging
import sqlite3

import folder_paths
from sqlalchemy.exc import IntegrityError, OperationalError

import app.assets.services.ingest as ingest
from app.assets.database.queries.records import create_content, create_record


def _output_path(name: str) -> str:
    output_dir = folder_paths.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, name)


def test_cached_registration_skips_extraction_when_live_content_is_missing(
    mock_create_session, monkeypatch
) -> None:
    path = _output_path("cached-missing-no-extraction.bin")
    with open(path, "wb") as file:
        file.write(b"output")

    def extraction_must_not_run(*_args, **_kwargs):
        raise AssertionError("missing content must not trigger metadata extraction")

    monkeypatch.setattr(ingest, "_extract_system_metadata_sync", extraction_must_not_run)
    try:
        assert ingest.register_cached_output(path) is None
    finally:
        os.unlink(path)


def test_cached_registration_skips_extraction_when_reusing_a_sibling(
    mock_create_session, monkeypatch
) -> None:
    path = _output_path("cached-sibling-no-extraction.bin")
    with open(path, "wb") as file:
        file.write(b"output")

    def extraction_must_not_run(*_args, **_kwargs):
        raise AssertionError("sibling metadata must be reused without extraction")

    with mock_create_session() as session:
        content = create_content(session, path, size_bytes=6)
        create_record(
            session,
            content.id,
            "sibling.bin",
            system_metadata={"source": "sibling"},
        )
        session.commit()

    monkeypatch.setattr(ingest, "_extract_system_metadata_sync", extraction_must_not_run)
    try:
        result = ingest.register_cached_output(path)
        assert result is not None
    finally:
        os.unlink(path)


def test_executed_registration_uses_the_write_transaction_runner(
    mock_create_session, monkeypatch
) -> None:
    path = _output_path("executed-write-transaction.bin")
    with open(path, "wb") as file:
        file.write(b"output")

    calls: list[None] = []

    def record_call(work):
        calls.append(None)
        with mock_create_session() as session:
            result = work(session)
            session.commit()
            return result

    monkeypatch.setattr(ingest, "run_write_txn", record_call, raising=False)
    try:
        result = ingest.register_executed_output(path)
        assert result is not None
        assert len(calls) == 1
    finally:
        os.unlink(path)


def _registration_failure_event(caplog) -> str:
    events = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("[assets-event] ingest.register_failed")
    ]
    assert len(events) == 1
    return events[0]


def test_executed_registration_reports_exhausted_locked_retries(monkeypatch, caplog) -> None:
    path = _output_path("executed-locked-retries.bin")
    with open(path, "wb") as file:
        file.write(b"output")

    def exhausted_retries(_work):
        raise OperationalError("INSERT", {}, sqlite3.OperationalError("database is locked"))

    monkeypatch.setattr(ingest, "run_write_txn", exhausted_retries)
    try:
        with caplog.at_level(logging.INFO):
            assert ingest.register_executed_output(path) is None
        assert _registration_failure_event(caplog) == (
            "[assets-event] ingest.register_failed error_type=OperationalError output_kind=executed"
        )
    finally:
        os.unlink(path)


def test_executed_registration_reports_non_retryable_write_failure(monkeypatch, caplog) -> None:
    path = _output_path("executed-non-retryable.bin")
    with open(path, "wb") as file:
        file.write(b"output")

    def non_retryable_failure(_work):
        raise IntegrityError("INSERT", {}, sqlite3.IntegrityError("constraint failed"))

    monkeypatch.setattr(ingest, "run_write_txn", non_retryable_failure)
    try:
        with caplog.at_level(logging.INFO):
            assert ingest.register_executed_output(path) is None
        assert _registration_failure_event(caplog) == (
            "[assets-event] ingest.register_failed error_type=IntegrityError output_kind=executed"
        )
    finally:
        os.unlink(path)


def test_executed_registration_reports_preflight_os_error(monkeypatch, caplog) -> None:
    monkeypatch.setattr(ingest.os, "stat", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("gone")))

    with caplog.at_level(logging.INFO):
        assert ingest.register_executed_output("/missing/output.bin") is None
    assert _registration_failure_event(caplog) == (
        "[assets-event] ingest.register_failed error_type=OSError output_kind=executed"
    )
