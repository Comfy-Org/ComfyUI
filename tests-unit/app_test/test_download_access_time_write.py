"""A download's access-time write must not wait on another writer, and must not fail the download.

Scans hold the write lock for a whole insert or enrich batch. The access time is advisory, so the
download path gives up on it after a short bounded wait instead of blocking for the full busy
timeout, or raising "database is locked" once that runs out.
"""

import sqlite3
import threading
import time

import pytest
from sqlalchemy import text

from app.assets.api import routes as asset_routes
from app.assets.database.models import Asset
from app.assets.database.queries.records import create_content, create_record
from app.assets.services import asset_management
from app.database import db as db_module


@pytest.fixture
def file_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "comfyui.db")
    monkeypatch.setattr(db_module.args, "database_url", f"sqlite:///{db_path}")
    monkeypatch.setattr(db_module, "Session", None)
    monkeypatch.setattr(db_module, "WriteSession", None)
    monkeypatch.setattr(db_module, "_db_lock", None)
    db_module._init_file_db(db_module.args.database_url)
    yield db_path
    db_module.Session.kw["bind"].dispose()
    db_module.WriteSession.kw["bind"].dispose()
    db_module._db_lock.release(force=True)


@pytest.fixture
def record_id(file_db, tmp_path):
    asset = tmp_path / "asset.png"
    asset.write_bytes(b"png")
    with db_module.create_write_session() as session:
        content = create_content(session, str(asset), size_bytes=3, mtime_ns=1)
        record = create_record(session, content.id, "asset.png")
        other = create_content(session, str(tmp_path / "other.png"), size_bytes=1, mtime_ns=1)
        create_record(session, other.id, "other.png")
        session.commit()
        return record.id


def _last_access_time(record_id):
    with db_module.create_session() as session:
        return session.get(Asset, record_id).last_access_time


def _hold_write_lock(db_path, seconds, started):
    holder = sqlite3.connect(db_path, isolation_level=None, timeout=0)
    holder.execute("BEGIN IMMEDIATE")
    holder.execute("UPDATE assets SET name = name")
    started.set()
    time.sleep(seconds)
    holder.execute("COMMIT")
    holder.close()


def test_uncontended_download_records_the_access_time(record_id):
    before = _last_access_time(record_id)

    result = asset_routes._resolve_download(record_id)

    assert result.download_name == "asset.png"
    assert _last_access_time(record_id) != before


def test_download_skips_the_access_time_while_another_writer_holds_the_lock(record_id, file_db):
    before = _last_access_time(record_id)
    started = threading.Event()
    holder = threading.Thread(target=_hold_write_lock, args=(file_db, 2.0, started))
    holder.start()
    started.wait()
    t0 = time.perf_counter()
    try:
        result = asset_routes._resolve_download(record_id)
        elapsed = time.perf_counter() - t0
    finally:
        holder.join()

    assert result.download_name == "asset.png"
    assert elapsed < 0.3, f"download waited {elapsed:.2f}s on another writer's lock"
    assert _last_access_time(record_id) == before


def test_resolve_does_not_write(record_id, file_db):
    started = threading.Event()
    holder = threading.Thread(target=_hold_write_lock, args=(file_db, 1.0, started))
    holder.start()
    started.wait()
    t0 = time.perf_counter()
    try:
        asset_management.resolve_asset_for_download(record_id)
        elapsed = time.perf_counter() - t0
    finally:
        holder.join()

    assert elapsed < 0.3, elapsed


def test_writer_commits_between_read_and_access_time_write(record_id, file_db, monkeypatch):
    # pysqlite opens no transaction for the reads, so a commit in between leaves no stale snapshot.
    real_isfile = asset_management.os.path.isfile

    def isfile_then_concurrent_commit(path):
        other = sqlite3.connect(file_db, isolation_level=None)
        other.execute("BEGIN IMMEDIATE")
        other.execute("UPDATE assets SET name = name WHERE id != ?", (record_id,))
        other.execute("COMMIT")
        other.close()
        return real_isfile(path)

    monkeypatch.setattr(asset_management.os.path, "isfile", isfile_then_concurrent_commit)
    before = _last_access_time(record_id)

    result = asset_routes._resolve_download(record_id)

    assert result.download_name == "asset.png"
    assert _last_access_time(record_id) != before


def test_bounded_session_restores_the_pooled_busy_timeout(file_db):
    with db_module.create_write_session() as session:
        default = session.execute(text("PRAGMA busy_timeout")).scalar_one()

    with db_module.create_bounded_write_session(50) as session:
        assert session.execute(text("PRAGMA busy_timeout")).scalar_one() == 50

    with db_module.create_write_session() as session:
        assert session.execute(text("PRAGMA busy_timeout")).scalar_one() == default
