import sqlite3
import threading
import time

from sqlalchemy import event, text

import app.database.db as db_mod
from app.database.db import create_session


def _invoke_writer(work):
    def _legacy(legacy_work):
        with create_session() as session:
            result = legacy_work(session)
            session.commit()
            return result

    return db_mod.run_write_txn(work) if hasattr(db_mod, "run_write_txn") else _legacy(work)


def test_write_transaction_waits_for_held_writer_before_select_then_mutate(tmp_path, monkeypatch):
    database_path = tmp_path / "assets.db"
    monkeypatch.setattr(db_mod.args, "database_url", f"sqlite:///{database_path}")
    monkeypatch.setattr(db_mod, "Session", None)
    monkeypatch.setattr(db_mod, "WriteSession", None)
    monkeypatch.setattr(db_mod, "_db_lock", None)
    db_mod.init_db()

    reader_engine = db_mod.Session.kw["bind"]
    writer_engine = db_mod.WriteSession.kw["bind"]

    def begin_deferred(connection):
        connection.exec_driver_sql("BEGIN")

    event.listen(reader_engine, "begin", begin_deferred)
    writer_started = threading.Event()

    def hold_write_lock():
        connection = sqlite3.connect(database_path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("INSERT INTO tags (name) VALUES (?)", ("promotion-holder",))
            writer_started.set()
            time.sleep(2)
            connection.rollback()
        finally:
            connection.close()

    holder = threading.Thread(target=hold_write_lock)
    holder.start()
    try:
        assert writer_started.wait(timeout=5)

        def select_then_mutate(session):
            session.execute(text("SELECT name FROM tags LIMIT 1"))
            session.execute(text("INSERT INTO tags (name) VALUES (:name)"), {"name": "promotion-work"})
            return "written"

        started_at = time.monotonic()
        result = _invoke_writer(select_then_mutate)
        elapsed = time.monotonic() - started_at
    finally:
        holder.join(timeout=5)
        assert not holder.is_alive()
        event.remove(reader_engine, "begin", begin_deferred)
        reader_engine.dispose()
        writer_engine.dispose()
        db_mod._db_lock.release(force=True)

    assert result == "written"
    assert elapsed >= 2
