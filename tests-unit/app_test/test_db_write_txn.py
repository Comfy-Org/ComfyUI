import sqlite3

import pytest
from sqlalchemy import text

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


def _other_writer_can_begin(db_path):
    other = sqlite3.connect(db_path, timeout=0, isolation_level=None)
    try:
        other.execute("BEGIN IMMEDIATE")
        other.execute("ROLLBACK")
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        other.close()


def test_file_db_uses_wal(file_db):
    with db_module.create_session() as session:
        assert session.execute(text("PRAGMA journal_mode")).scalar_one() == "wal"


def test_write_session_takes_the_write_lock_before_its_first_write(file_db):
    with db_module.create_write_session() as session:
        session.execute(text("SELECT 1")).scalar_one()
        assert _other_writer_can_begin(file_db) is False


def test_read_session_does_not_take_the_write_lock(file_db):
    with db_module.create_session() as session:
        session.execute(text("SELECT 1")).scalar_one()
        assert _other_writer_can_begin(file_db) is True
