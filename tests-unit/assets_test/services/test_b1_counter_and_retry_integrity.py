import logging
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

import app.database.db as db_mod
from app.assets import mode as mode_module
from app.assets import scanner
from app.assets import scanner_changes
from app.assets.database.models import AssetContent
from app.assets.database.queries.records import create_record
from app.assets.seeder import _ScanState

_LOCKED_ERROR = OperationalError("COMMIT", {}, sqlite3.OperationalError("database is locked"))


def _fail_commit_once_then_succeed(engine):
    real_factory = sessionmaker(bind=engine)
    remaining = {"count": 1}

    def factory():
        session = real_factory()
        real_commit = session.commit

        def commit():
            if remaining["count"] > 0:
                remaining["count"] -= 1
                session.rollback()
                raise _LOCKED_ERROR
            return real_commit()

        session.commit = commit
        return session

    return factory


def _fail_run_write_txn_at(real_run_write_txn, fail_index: int):
    calls = {"count": -1}

    def wrapper(work):
        calls["count"] += 1
        if calls["count"] == fail_index:
            raise _LOCKED_ERROR
        return real_run_write_txn(work)

    return wrapper


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    monkeypatch.setattr(db_mod.time, "sleep", lambda _seconds: None)


def test_group_a_transient_locked_error_retried_preserves_queue_once(
    db_engine, tmp_path: Path, monkeypatch, session
):
    """A commit-time locked failure that later succeeds must publish the
    verification queue exactly once, never per attempt."""
    path = tmp_path / "drifted.bin"
    path.write_bytes(b"drifted bytes")
    stale_mtime_ns = path.stat().st_mtime_ns - 5_000_000_000
    content = AssetContent(
        path=str(path), hash=None, size_bytes=path.stat().st_size, mtime_ns=stale_mtime_ns
    )
    session.add(content)
    session.flush()
    create_record(session, content.id, "drifted.bin")
    session.commit()
    content_id = content.id

    scanner_changes.clear_pending_verifications()
    monkeypatch.setattr(db_mod, "WriteSession", _fail_commit_once_then_succeed(db_engine))

    class _HashingOn:
        enable_asset_hashing = True

    mode_module.init(_HashingOn())
    try:
        with patch("folder_paths.get_input_directory", return_value=str(tmp_path)):
            survivors = scanner.sync_root_safely("input", _ScanState())
    finally:
        mode_module.init(None)

    assert survivors == {os.path.abspath(str(path))}
    assert scanner_changes._pending_verification_ids == [content_id]
    scanner_changes.clear_pending_verifications()


def test_group_a_sync_permission_diagnostic_published_exactly_once_after_retry(
    db_engine, tmp_path: Path, monkeypatch, session, caplog
):
    """A commit-time locked failure that later succeeds must publish the
    permission-denied diagnostic exactly once: one counter bump, one emit."""
    path = tmp_path / "unreadable.bin"
    path.write_bytes(b"unreadable")
    content = AssetContent(
        path=str(path), hash=None, size_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns
    )
    session.add(content)
    session.flush()
    session.commit()

    real_stat = os.stat

    def deny_stat(candidate_path, *args, **kwargs):
        if str(candidate_path) == str(path):
            raise PermissionError(str(path))
        return real_stat(candidate_path, *args, **kwargs)

    monkeypatch.setattr(scanner, "os", SimpleNamespace(stat=deny_stat, path=scanner.os.path))
    monkeypatch.setattr(db_mod, "WriteSession", _fail_commit_once_then_succeed(db_engine))

    progress = _ScanState()
    with (
        patch("folder_paths.get_input_directory", return_value=str(tmp_path)),
        caplog.at_level(logging.INFO),
    ):
        scanner.sync_root_safely("input", progress)

    assert progress.permission_denied == 1
    stat_failed_lines = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("[assets-event] scanner.stat_failed")
    ]
    assert stat_failed_lines == [
        "[assets-event] scanner.stat_failed error_type=PermissionError site=reference_stat"
    ]


def test_b1_counter_integrity_under_a_locked_failure_at_row_n(
    db_engine, tmp_path: Path, session, monkeypatch
):
    """Injecting a locked failure at the middle row of a batch must not
    double-mark the surviving rows, and must match a clean run exactly."""
    paths = [tmp_path / f"row-{i}.bin" for i in range(3)]
    rows = []
    for i, path in enumerate(paths):
        path.write_bytes(f"payload-{i}".encode())
        stat = path.stat()
        content = AssetContent(
            path=str(path), hash=None, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns
        )
        session.add(content)
        session.flush()
        record = create_record(session, content.id, path.name)
        rows.append(scanner.UnenrichedContent(content.id, record.id, str(path), True))
    session.commit()
    content_ids = [row.content_id for row in rows]

    real_run_write_txn = scanner.run_write_txn
    monkeypatch.setattr(
        scanner, "run_write_txn", _fail_run_write_txn_at(real_run_write_txn, fail_index=1)
    )

    enriched, failed_ids = scanner.enrich_assets_batch(
        rows, extract_metadata=False, compute_hash=True
    )

    assert enriched == 2
    assert failed_ids == [rows[1].record_id]

    hashes_after_failure = {
        content_id: session.get(AssetContent, content_id).hash for content_id in content_ids
    }
    assert hashes_after_failure[content_ids[1]] is None
    assert hashes_after_failure[content_ids[0]] is not None
    assert hashes_after_failure[content_ids[2]] is not None

    monkeypatch.setattr(scanner, "run_write_txn", real_run_write_txn)
    retry_enriched, retry_failed_ids = scanner.enrich_assets_batch(
        [rows[1]], extract_metadata=False, compute_hash=True
    )
    assert retry_enriched == 1
    assert retry_failed_ids == []
    session.expire_all()
    final_hashes = {
        content_id: session.get(AssetContent, content_id).hash for content_id in content_ids
    }
    assert all(value is not None for value in final_hashes.values())
    assert len(session.scalars(scanner.sa.select(AssetContent)).all()) == 3
