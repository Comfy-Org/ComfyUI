import sqlite3
import threading

import pytest
from sqlalchemy import func, select

from app.assets.database.models import Tag
from app.database import db as db_module


@pytest.fixture
def memory_db(monkeypatch):
    monkeypatch.setattr(db_module.args, "database_url", "sqlite:///:memory:")
    monkeypatch.setattr(db_module, "Session", None)
    monkeypatch.setattr(db_module, "WriteSession", None)
    monkeypatch.setattr(db_module, "_memory_db_anchor", None, raising=False)
    db_module.init_db()


def _tag_count():
    with db_module.create_session() as session:
        return session.scalar(select(func.count()).select_from(Tag))


def test_concurrent_write_transactions_with_savepoints_all_commit(memory_db):
    # The seeder's batch inserts and output registration nest savepoints in concurrent
    # write transactions; on one shared connection they unwind each other's savepoints.
    threads, per_thread = 4, 50
    errors = []
    start = threading.Barrier(threads)

    def write(worker):
        start.wait()
        for i in range(per_thread):
            try:
                with db_module.create_write_session() as session, session.begin():
                    with session.begin_nested():
                        session.add(Tag(name=f"w{worker}-{i}"))
            except Exception as e:
                errors.append(e)

    workers = [threading.Thread(target=write, args=(n,)) for n in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert errors == []
    assert _tag_count() == threads * per_thread


def test_database_outlives_its_pooled_connections(memory_db):
    with db_module.create_write_session() as session, session.begin():
        session.add(Tag(name="kept"))

    db_module.Session.kw["bind"].dispose()
    db_module.WriteSession.kw["bind"].dispose()

    assert _tag_count() == 1


def test_each_init_gets_its_own_database(memory_db):
    with db_module.create_write_session() as session, session.begin():
        session.add(Tag(name="first"))

    db_module.init_db()

    assert _tag_count() == 0


def test_falls_back_to_a_shared_connection_without_memdb(monkeypatch, caplog):
    real_connect = sqlite3.connect

    def connect_without_memdb(database, *args, **kwargs):
        if "vfs=memdb" in str(database):
            raise sqlite3.OperationalError("no such vfs: memdb")
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(db_module.sqlite3, "connect", connect_without_memdb)
    monkeypatch.setattr(db_module.args, "database_url", "sqlite:///:memory:")
    monkeypatch.setattr(db_module, "Session", None)
    monkeypatch.setattr(db_module, "_memory_db_anchor", None, raising=False)

    db_module.init_db()

    assert "no memdb VFS" in caplog.text
    assert db_module.Session.kw["bind"].url.database == ":memory:"
    with db_module.create_write_session() as session, session.begin():
        session.add(Tag(name="fallback"))
    assert _tag_count() == 1
