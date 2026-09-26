"""A download's access-time write must never fail the download.

Scans hold the write lock for a whole insert or enrich batch. The access time is advisory, so if
the write can't get the lock within the busy timeout, the file is served anyway.
"""

import sqlite3
import threading
import time

import pytest

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

    result = asset_management.resolve_asset_for_download(record_id)

    assert result.download_name == "asset.png"
    assert _last_access_time(record_id) != before


def test_download_is_served_when_the_lock_outlasts_the_busy_timeout(record_id, file_db):
    # pysqlite's default busy timeout is 5s; hold the lock longer so the write gives up.
    before = _last_access_time(record_id)
    started = threading.Event()
    holder = threading.Thread(target=_hold_write_lock, args=(file_db, 6.5, started))
    holder.start()
    started.wait()
    try:
        result = asset_management.resolve_asset_for_download(record_id)
    finally:
        holder.join()

    assert result.download_name == "asset.png"
    assert _last_access_time(record_id) == before


def test_access_time_is_recorded_once_a_short_lock_is_released(record_id, file_db):
    before = _last_access_time(record_id)
    started = threading.Event()
    holder = threading.Thread(target=_hold_write_lock, args=(file_db, 1.0, started))
    holder.start()
    started.wait()
    try:
        result = asset_management.resolve_asset_for_download(record_id)
    finally:
        holder.join()

    assert result.download_name == "asset.png"
    assert _last_access_time(record_id) != before
