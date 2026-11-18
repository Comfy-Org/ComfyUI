import sqlite3
from contextlib import closing

import pytest
from sqlalchemy import event, text
from sqlalchemy.engine import Engine

from app.database import db as db_module


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


@pytest.fixture
def extra_wal_synchronous(monkeypatch):
    # Not SQLite's FULL or WAL default on any build, so only the hooks can produce it.
    monkeypatch.setattr(db_module, "WAL_SYNCHRONOUS", "EXTRA")


@pytest.mark.parametrize("factory", ["Session", "WriteSession"])
def test_wal_connections_get_the_wal_synchronous_setting(extra_wal_synchronous, file_db, factory):
    engine = getattr(db_module, factory).kw["bind"]
    # The read engine's first may be the connection opened before WAL was on; holding it
    # makes the second a fresh one through the engine's connect hook.
    with closing(engine.raw_connection()) as first, closing(engine.raw_connection()) as second:
        values = [c.cursor().execute("PRAGMA synchronous").fetchone()[0] for c in (first, second)]
    assert values == [3, 3]  # EXTRA


@pytest.fixture
def refused_wal():
    """Stand in for a filesystem that refuses WAL: the request leaves the rollback journal on."""

    def refuse(conn, cursor, statement, parameters, context, executemany):
        if statement == "PRAGMA journal_mode=WAL":
            statement = "PRAGMA journal_mode=DELETE"
        return statement, parameters

    event.listen(Engine, "before_cursor_execute", refuse, retval=True)
    yield
    event.remove(Engine, "before_cursor_execute", refuse)


@pytest.fixture
def refused_wal_db(extra_wal_synchronous, refused_wal, file_db):
    """The file database initialised while WAL is refused (fixtures resolve in this order)."""
    return file_db


@pytest.mark.parametrize("factory", ["Session", "WriteSession"])
def test_synchronous_is_left_alone_when_wal_is_refused(refused_wal_db, factory):
    with closing(sqlite3.connect(":memory:")) as plain:
        build_default = plain.execute("PRAGMA synchronous").fetchone()[0]
    engine = getattr(db_module, factory).kw["bind"]
    # Two, so the second is a fresh connection through the engine's connect hooks.
    with closing(engine.raw_connection()) as first, closing(engine.raw_connection()) as second:
        for conn in (first, second):
            cursor = conn.cursor()
            assert cursor.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
            assert cursor.execute("PRAGMA synchronous").fetchone()[0] == build_default


def test_write_session_takes_the_write_lock_before_its_first_write(file_db):
    with db_module.create_write_session() as session:
        session.execute(text("SELECT 1")).scalar_one()
        assert _other_writer_can_begin(file_db) is False


def test_read_session_does_not_take_the_write_lock(file_db):
    with db_module.create_session() as session:
        session.execute(text("SELECT 1")).scalar_one()
        assert _other_writer_can_begin(file_db) is True


def test_backup_gives_up_when_the_destination_stays_locked(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    destination = tmp_path / "destination.db"
    with sqlite3.connect(source) as conn:
        conn.execute("CREATE TABLE t (x)")
    with sqlite3.connect(destination) as conn:
        conn.execute("CREATE TABLE u (y)")
    holder = sqlite3.connect(destination, isolation_level=None, timeout=0)
    holder.execute("BEGIN IMMEDIATE")
    monkeypatch.setattr(db_module, "_BACKUP_TIMEOUT_SECONDS", 0.0)
    try:
        with pytest.raises(TimeoutError):
            db_module._backup_database(str(source), str(destination))
    finally:
        holder.execute("ROLLBACK")
        holder.close()


def test_backup_past_its_deadline_still_completes_when_nothing_blocks_it(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    destination = tmp_path / "destination.db"
    with sqlite3.connect(source) as conn:
        conn.execute("CREATE TABLE t (x)")
        conn.execute("INSERT INTO t VALUES (1)")
    monkeypatch.setattr(db_module, "_BACKUP_TIMEOUT_SECONDS", -1.0)

    db_module._backup_database(str(source), str(destination))

    with sqlite3.connect(destination) as conn:
        assert conn.execute("SELECT x FROM t").fetchall() == [(1,)]
