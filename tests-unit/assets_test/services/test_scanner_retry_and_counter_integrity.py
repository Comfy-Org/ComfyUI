import os
import sqlite3
from pathlib import Path
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
    calls = {"count": 0}

    def wrapper(work):
        call_index = calls["count"]
        calls["count"] += 1
        if call_index == fail_index:
            raise _LOCKED_ERROR
        return real_run_write_txn(work)

    return wrapper, calls


def _seed_enrichment_rows(tmp_path: Path, count: int) -> list[scanner.UnenrichedContent]:
    paths = [tmp_path / f"row-{index}.bin" for index in range(count)]
    for index, path in enumerate(paths):
        path.write_bytes(f"payload-{index}".encode())

    def seed(write_session):
        rows: list[scanner.UnenrichedContent] = []
        for path in paths:
            stat_result = path.stat()
            content = AssetContent(
                path=str(path),
                hash=None,
                size_bytes=stat_result.st_size,
                mtime_ns=stat_result.st_mtime_ns,
            )
            write_session.add(content)
            write_session.flush()
            record = create_record(write_session, content.id, path.name)
            rows.append(
                scanner.UnenrichedContent(
                    content.id,
                    record.id,
                    str(path),
                    needs_hash=True,
                    observed_size_bytes=content.size_bytes,
                    observed_mtime_ns=content.mtime_ns,
                )
            )
        return rows

    return db_mod.run_write_txn(seed)


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    monkeypatch.setattr(db_mod.time, "sleep", lambda _seconds: None)


def test_scanner_sync_transient_locked_error_retried_preserves_queue_once(
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



def test_enrichment_batch_failure_rolls_back_every_row_and_counts_each_failure(
    tmp_path: Path, session, monkeypatch
) -> None:
    rows = _seed_enrichment_rows(tmp_path, 3)
    real_run_write_txn = scanner.run_write_txn
    failing_run_write_txn, calls = _fail_run_write_txn_at(
        real_run_write_txn, fail_index=0
    )
    monkeypatch.setattr(
        scanner, "run_write_txn", failing_run_write_txn
    )
    progress = _ScanState()

    enriched, failed_ids, _consumed = scanner.enrich_assets_batch(
        rows,
        extract_metadata=False,
        compute_hash=True,
        progress=progress,
    )

    assert calls["count"] == 1
    assert enriched == 0
    assert failed_ids == [row.record_id for row in rows]
    assert progress.enrich_failed == 3
    session.expire_all()
    assert all(
        session.get(AssetContent, row.content_id).hash is None for row in rows
    )


def test_enrichment_later_batch_failure_preserves_first_batch(
    tmp_path: Path, session, monkeypatch
) -> None:
    rows = _seed_enrichment_rows(tmp_path, 30)
    real_run_write_txn = scanner.run_write_txn
    failing_run_write_txn, calls = _fail_run_write_txn_at(
        real_run_write_txn, fail_index=1
    )
    monkeypatch.setattr(scanner, "run_write_txn", failing_run_write_txn)

    enriched, failed_ids, _consumed = scanner.enrich_assets_batch(
        rows, extract_metadata=False, compute_hash=True
    )

    assert calls["count"] == 2
    assert enriched == scanner.MAX_WRITE_BATCH
    assert failed_ids == [row.record_id for row in rows[scanner.MAX_WRITE_BATCH :]]
    session.expire_all()
    assert all(
        session.get(AssetContent, row.content_id).hash is not None
        for row in rows[: scanner.MAX_WRITE_BATCH]
    )
    assert all(
        session.get(AssetContent, row.content_id).hash is None
        for row in rows[scanner.MAX_WRITE_BATCH :]
    )
