import gc
import sqlite3
import threading
import time
from contextlib import closing

import pytest
from sqlalchemy import event, func, select, text

from app.assets.database.models import Tag
from app.database import db as db_module

_WAIT_SECONDS = 30


@pytest.fixture
def fresh_memory_db(monkeypatch):
    monkeypatch.setattr(db_module.args, "database_url", "sqlite:///:memory:")
    monkeypatch.setattr(db_module, "Session", None)
    monkeypatch.setattr(db_module, "WriteSession", None)
    monkeypatch.setattr(db_module, "_memory_db_anchor", None)
    yield
    for factory in (db_module.Session, db_module.WriteSession):
        if factory is not None:
            factory.kw["bind"].dispose()
    if db_module._memory_db_anchor is not None:
        db_module._memory_db_anchor.close()


def _sqlite_shares_memdb():
    # Probed independently of the product, so a wrong fallback fails these tests instead of skipping them.
    if sqlite3.sqlite_version_info < (3, 36, 0):
        return False
    try:
        sqlite3.connect("file:/comfyui-test-probe?vfs=memdb", uri=True).close()
    except sqlite3.OperationalError:
        return False
    return True


requires_memdb = pytest.mark.skipif(
    not _sqlite_shares_memdb(), reason=f"SQLite {sqlite3.sqlite_version} cannot share a memdb database"
)


@pytest.fixture
def memory_db(fresh_memory_db):
    db_module.init_db()
    assert db_module._memory_db_anchor is not None


def _memdb_connection(timeout):
    url = db_module.WriteSession.kw["bind"].url
    return sqlite3.connect(f"{url.database}?vfs=memdb", uri=True, timeout=timeout, isolation_level=None)


def _tag_count():
    with db_module.create_session() as session:
        return session.scalar(select(func.count()).select_from(Tag))


def _run_threads(target, count):
    errors = []

    def run(n):
        try:
            target(n)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=run, args=(n,)) for n in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(_WAIT_SECONDS)
    assert not any(thread.is_alive() for thread in threads)
    return errors


@requires_memdb
def test_concurrent_write_transactions_with_savepoints_all_commit(memory_db):
    # The seeder's batch inserts and output registration nest savepoints in concurrent
    # write transactions; on one shared connection they unwind each other's savepoints.
    # Odd workers write through the read engine's deferred transactions, as tagging does.
    threads, per_thread = 4, 200
    start = threading.Barrier(threads, timeout=_WAIT_SECONDS)

    def write(worker):
        factory = db_module.create_session if worker % 2 else db_module.create_write_session
        start.wait()
        for i in range(per_thread):
            with factory() as session, session.begin():
                with session.begin_nested():
                    session.add(Tag(name=f"w{worker}-{i}"))

    assert _run_threads(write, threads) == []
    assert _tag_count() == threads * per_thread


@requires_memdb
def test_database_outlives_its_pooled_connections(memory_db):
    with db_module.create_write_session() as session, session.begin():
        session.add(Tag(name="kept"))

    db_module.Session.kw["bind"].dispose()
    db_module.WriteSession.kw["bind"].dispose()
    gc.collect()

    assert _tag_count() == 1


@requires_memdb
def test_each_init_gets_its_own_database(memory_db):
    with db_module.create_write_session() as session, session.begin():
        session.add(Tag(name="first"))
    first = (db_module._memory_db_anchor, db_module.Session, db_module.WriteSession)

    db_module.init_db()

    assert _tag_count() == 0
    first[0].close()
    first[1].kw["bind"].dispose()
    first[2].kw["bind"].dispose()


@requires_memdb
def test_a_read_during_a_write_waits_and_sees_only_committed_rows(memory_db):
    # An API read while the scanner holds a write transaction: it must neither fail on the
    # lock nor see the uncommitted rows (the writer rolls back here).
    written, reading = threading.Event(), threading.Event()
    counts = []
    read_engine = db_module.Session.kw["bind"]

    def signal_reading(*_):
        reading.set()

    def write(_):
        with db_module.create_write_session() as session, session.begin():
            session.add(Tag(name="uncommitted"))
            session.flush()
            written.set()
            assert reading.wait(_WAIT_SECONDS)
            time.sleep(0.1)
            assert counts == []  # the reader is waiting on the lock, not reading around it
            session.rollback()

    def read(_):
        assert written.wait(_WAIT_SECONDS)
        counts.append(_tag_count())

    event.listen(read_engine, "before_cursor_execute", signal_reading)
    try:
        errors = _run_threads(lambda n: (write, read)[n](n), 2)
    finally:
        event.remove(read_engine, "before_cursor_execute", signal_reading)
    assert errors == []
    assert counts == [0]


@requires_memdb
def test_read_session_does_not_take_the_write_lock(memory_db):
    with db_module.create_session() as session:
        session.execute(text("SELECT 1")).scalar_one()
        other = _memdb_connection(timeout=0)
        try:
            other.execute("BEGIN IMMEDIATE")
            other.execute("ROLLBACK")
        finally:
            other.close()


@requires_memdb
def test_write_session_takes_the_write_lock_before_its_first_write(memory_db):
    with db_module.create_write_session() as session:
        session.execute(text("SELECT 1")).scalar_one()
        other = _memdb_connection(timeout=0)
        try:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                other.execute("BEGIN IMMEDIATE")
        finally:
            other.close()


@requires_memdb
@pytest.mark.parametrize("factory", ["Session", "WriteSession"])
def test_memdb_connections_enforce_foreign_keys(memory_db, factory):
    engine = getattr(db_module, factory).kw["bind"]
    # Two, so the second is a fresh connection through the engine's connect hooks.
    with closing(engine.raw_connection()) as first, closing(engine.raw_connection()) as second:
        assert [c.cursor().execute("PRAGMA foreign_keys").fetchone()[0] for c in (first, second)] == [1, 1]


@requires_memdb
def test_sqlite_3_36_uses_memdb(fresh_memory_db, monkeypatch):
    monkeypatch.setattr(db_module.sqlite3, "sqlite_version_info", (3, 36, 0))

    db_module.init_db()

    assert db_module._memory_db_anchor is not None


def _assert_shared_connection_fallback(caplog):
    db_module.init_db()

    assert "cannot share an in-memory database" in caplog.text
    assert db_module._memory_db_anchor is None
    assert db_module.Session.kw["bind"].url.database == ":memory:"

    def write(_):
        with db_module.create_write_session() as session, session.begin():
            session.add(Tag(name="from another thread"))

    assert _run_threads(write, 1) == []
    assert _tag_count() == 1


def test_falls_back_to_a_shared_connection_without_memdb(fresh_memory_db, monkeypatch, caplog):
    real_connect = sqlite3.connect

    def connect_without_memdb(database, *args, **kwargs):
        if "vfs=memdb" in str(database):
            raise sqlite3.OperationalError("no such vfs: memdb")
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(db_module.sqlite3, "connect", connect_without_memdb)
    _assert_shared_connection_fallback(caplog)


def test_falls_back_to_a_shared_connection_before_sqlite_3_36(fresh_memory_db, monkeypatch, caplog):
    # SQLite 3.23-3.35 can open a memdb database but not share it between connections.
    monkeypatch.setattr(db_module.sqlite3, "sqlite_version_info", (3, 35, 5))
    _assert_shared_connection_fallback(caplog)
