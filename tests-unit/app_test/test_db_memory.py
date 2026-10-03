import gc
import sqlite3
import threading

import pytest
from sqlalchemy import func, select

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


@pytest.fixture
def memory_db(fresh_memory_db):
    db_module.init_db()
    if db_module._memory_db_anchor is None:
        pytest.skip(f"SQLite {sqlite3.sqlite_version} cannot share a memdb database")


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


def test_concurrent_write_transactions_with_savepoints_all_commit(memory_db):
    # The seeder's batch inserts and output registration nest savepoints in concurrent
    # write transactions; on one shared connection they unwind each other's savepoints.
    threads, per_thread = 4, 200
    start = threading.Barrier(threads, timeout=_WAIT_SECONDS)

    def write(worker):
        start.wait()
        for i in range(per_thread):
            with db_module.create_write_session() as session, session.begin():
                with session.begin_nested():
                    session.add(Tag(name=f"w{worker}-{i}"))

    assert _run_threads(write, threads) == []
    assert _tag_count() == threads * per_thread


def test_database_outlives_its_pooled_connections(memory_db):
    with db_module.create_write_session() as session, session.begin():
        session.add(Tag(name="kept"))

    db_module.Session.kw["bind"].dispose()
    db_module.WriteSession.kw["bind"].dispose()
    gc.collect()

    assert _tag_count() == 1


def test_each_init_gets_its_own_database(memory_db):
    with db_module.create_write_session() as session, session.begin():
        session.add(Tag(name="first"))

    db_module.init_db()

    assert _tag_count() == 0


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
