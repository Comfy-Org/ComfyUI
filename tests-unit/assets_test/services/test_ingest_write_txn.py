import os

import logging
import sqlite3

import folder_paths
import pytest
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


def _seed_cached_content(mock_create_session, path: str) -> str:
    with mock_create_session() as session:
        content = create_content(session, path, size_bytes=os.path.getsize(path))
        session.commit()
        return content.id


def _apply_cached_preflight(session, preflight, path: str, system_metadata: dict[str, int]) -> None:
    ingest._apply_cached_registration(
        session,
        preflight,
        os.path.basename(path),
        ["output"],
        None,
        None,
        path,
        system_metadata,
    )


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


def test_reused_upload_falls_back_after_four_stale_preflights(monkeypatch, caplog) -> None:
    preflight = object()
    prepared = object()
    attempts: list[object] = []
    fallback_result = object()

    monkeypatch.setattr(ingest, "_preflight_upload_record", lambda *_args: preflight)
    monkeypatch.setattr(ingest, "_prepare_upload_record", lambda _preflight: prepared)

    def stale_apply(_session, observed_prepared):
        attempts.append(observed_prepared)
        raise ingest._PreflightStale

    monkeypatch.setattr(ingest, "_apply_reused_upload_record", stale_apply)
    monkeypatch.setattr(
        ingest,
        "_reuse_qualified_content_in_txn",
        lambda *_args: fallback_result,
    )
    monkeypatch.setattr(ingest, "run_write_txn", lambda work: work(object()))

    spec = ingest._UploadRecordSpec("asset", [], None, {}, None)
    with caplog.at_level(logging.WARNING):
        result = ingest._reuse_qualified_content("blake3:hash", spec)

    assert result is fallback_result
    assert attempts == [prepared, prepared, prepared, prepared]
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_cached_registration_reports_terminal_write_failure(
    mock_create_session, monkeypatch, caplog
) -> None:
    path = _output_path("cached-terminal-failure.bin")
    with open(path, "wb") as file:
        file.write(b"output")
    with mock_create_session() as session:
        create_content(session, path, size_bytes=6)
        session.commit()

    def non_retryable_failure(_work):
        raise IntegrityError("INSERT", {}, sqlite3.IntegrityError("constraint failed"))

    monkeypatch.setattr(ingest, "run_write_txn", non_retryable_failure)
    try:
        with caplog.at_level(logging.INFO):
            assert ingest.register_cached_output(path) is None
        assert _registration_failure_event(caplog) == (
            "[assets-event] ingest.register_failed error_type=IntegrityError output_kind=cached"
        )
    finally:
        os.unlink(path)


def test_cached_registration_restarts_when_content_vanishes_after_preflight(
    mock_create_session, monkeypatch
) -> None:
    direct_path = _output_path("cached-direct-content-vanished.bin")
    public_path = _output_path("cached-public-content-vanished.bin")
    for path in (direct_path, public_path):
        with open(path, "wb") as file:
            file.write(b"output")
    try:
        direct_content_id = _seed_cached_content(mock_create_session, direct_path)
        direct_preflight = ingest._preflight_cached_registration(direct_path)
        assert direct_preflight is not None
        with mock_create_session() as session:
            content = session.get(ingest.AssetContent, direct_content_id)
            assert content is not None
            ingest.mark_content_missing(session, content.id)
            session.commit()
        with mock_create_session() as session:
            with pytest.raises(ingest._PreflightStale):
                _apply_cached_preflight(session, direct_preflight, direct_path, {})

        public_content_id = _seed_cached_content(mock_create_session, public_path)
        real_apply = ingest._apply_cached_registration
        mutated = False

        def vanish_then_apply(session, *args):
            nonlocal mutated
            if not mutated:
                mutated = True
                with mock_create_session() as mutation_session:
                    content = mutation_session.get(ingest.AssetContent, public_content_id)
                    assert content is not None
                    ingest.mark_content_missing(mutation_session, content.id)
                    mutation_session.commit()
            return real_apply(session, *args)

        monkeypatch.setattr(ingest, "_apply_cached_registration", vanish_then_apply)
        assert ingest.register_cached_output(public_path) is None
        assert mutated is True
    finally:
        for path in (direct_path, public_path):
            os.unlink(path)


def test_cached_registration_restarts_when_sibling_appears_after_preflight(
    mock_create_session, monkeypatch
) -> None:
    direct_path = _output_path("cached-direct-sibling-appeared.bin")
    public_path = _output_path("cached-public-sibling-appeared.bin")
    for path in (direct_path, public_path):
        with open(path, "wb") as file:
            file.write(b"output")
    try:
        direct_content_id = _seed_cached_content(mock_create_session, direct_path)
        direct_preflight = ingest._preflight_cached_registration(direct_path)
        assert direct_preflight is not None
        with mock_create_session() as session:
            create_record(
                session,
                direct_content_id,
                "sibling.bin",
                system_metadata={"generation": 1},
            )
            session.commit()
        with mock_create_session() as session:
            with pytest.raises(ingest._PreflightStale):
                _apply_cached_preflight(session, direct_preflight, direct_path, {})

        public_content_id = _seed_cached_content(mock_create_session, public_path)
        real_apply = ingest._apply_cached_registration
        extraction_count = 0
        mutated = False

        def extract_metadata(*_args, **_kwargs):
            nonlocal extraction_count
            extraction_count += 1
            return {"generation": 0}

        def add_sibling_then_apply(session, *args):
            nonlocal mutated
            if not mutated:
                mutated = True
                with mock_create_session() as mutation_session:
                    create_record(
                        mutation_session,
                        public_content_id,
                        "sibling.bin",
                        system_metadata={"generation": 2},
                    )
                    mutation_session.commit()
            return real_apply(session, *args)

        monkeypatch.setattr(ingest, "_extract_system_metadata_sync", extract_metadata)
        monkeypatch.setattr(ingest, "_apply_cached_registration", add_sibling_then_apply)
        result = ingest.register_cached_output(public_path)
        assert result is not None
        with mock_create_session() as session:
            record = session.get(ingest.Asset, result.id)
            assert record is not None
            assert record.system_metadata == {"generation": 2}
        assert extraction_count == 1
        assert mutated is True
    finally:
        for path in (direct_path, public_path):
            os.unlink(path)


def test_cached_registration_restarts_when_file_changes_after_preflight(
    mock_create_session, monkeypatch
) -> None:
    direct_path = _output_path("cached-direct-stat-changed.bin")
    public_path = _output_path("cached-public-stat-changed.bin")
    for path in (direct_path, public_path):
        with open(path, "wb") as file:
            file.write(b"old")
    try:
        _seed_cached_content(mock_create_session, direct_path)
        direct_preflight = ingest._preflight_cached_registration(direct_path)
        assert direct_preflight is not None
        with open(direct_path, "wb") as file:
            file.write(b"new bytes")
        with mock_create_session() as session:
            with pytest.raises(ingest._PreflightStale):
                _apply_cached_preflight(session, direct_preflight, direct_path, {})

        _seed_cached_content(mock_create_session, public_path)
        real_apply = ingest._apply_cached_registration
        extraction_sizes: list[int] = []
        mutated = False

        def extract_metadata(path, *_args, **_kwargs):
            size = os.path.getsize(path)
            extraction_sizes.append(size)
            return {"size": size}

        def rewrite_then_apply(session, *args):
            nonlocal mutated
            if not mutated:
                mutated = True
                with open(public_path, "wb") as file:
                    file.write(b"new public bytes")
            return real_apply(session, *args)

        monkeypatch.setattr(ingest, "_extract_system_metadata_sync", extract_metadata)
        monkeypatch.setattr(ingest, "_apply_cached_registration", rewrite_then_apply)
        result = ingest.register_cached_output(public_path)
        assert result is not None
        with mock_create_session() as session:
            record = session.get(ingest.Asset, result.id)
            assert record is not None
            assert record.system_metadata == {"size": len(b"new public bytes")}
        assert extraction_sizes == [len(b"old"), len(b"new public bytes")]
        assert mutated is True
    finally:
        for path in (direct_path, public_path):
            os.unlink(path)
