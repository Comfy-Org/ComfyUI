import os
import shutil
import sqlite3
import threading
import time

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import event, text
from sqlalchemy.exc import OperationalError

import app.database.db as db_mod


_PRE_HEAD = "0006_add_loader_path"
_WAL_SENTINEL = "wal-resident-sentinel"


def _dispose_runtime_engines():
    for session_factory in (db_mod.Session, getattr(db_mod, "WriteSession", None)):
        if session_factory is not None:
            session_factory.kw["bind"].dispose()


@pytest.fixture
def file_database(tmp_path, monkeypatch):
    database_path = str(tmp_path / "assets.db")
    monkeypatch.setattr(db_mod.args, "database_url", f"sqlite:///{database_path}")
    monkeypatch.setattr(db_mod, "Session", None)
    monkeypatch.setattr(db_mod, "_db_lock", None)
    if hasattr(db_mod, "WriteSession"):
        monkeypatch.setattr(db_mod, "WriteSession", None)
    db_mod.init_db()
    yield database_path
    _dispose_runtime_engines()
    db_mod._db_lock.release(force=True)


@pytest.fixture
def memory_database(monkeypatch):
    monkeypatch.setattr(db_mod.args, "database_url", "sqlite:///:memory:")
    monkeypatch.setattr(db_mod, "Session", None)
    if hasattr(db_mod, "WriteSession"):
        monkeypatch.setattr(db_mod, "WriteSession", None)
    db_mod.init_db()
    yield
    _dispose_runtime_engines()


def _make_config(db_path: str) -> Config:
    root = os.path.join(os.path.dirname(__file__), "../..")
    config = Config(os.path.abspath(os.path.join(root, "alembic.ini")))
    config.set_main_option("script_location", os.path.abspath(os.path.join(root, "alembic_db")))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return config


def _crash_style_wal_database(tmp_path) -> str:
    source_path = str(tmp_path / "source.db")
    target_path = str(tmp_path / "crash.db")
    command.upgrade(_make_config(source_path), _PRE_HEAD)

    writer = sqlite3.connect(source_path)
    reader = sqlite3.connect(source_path)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        reader.execute("BEGIN")
        reader.execute("SELECT name FROM tags LIMIT 1").fetchone()
        writer.execute("INSERT INTO tags (name) VALUES (?)", (_WAL_SENTINEL,))
        writer.commit()
        shutil.copy(source_path, target_path)
        shutil.copy(source_path + "-wal", target_path + "-wal")
    finally:
        writer.close()
        reader.close()

    assert os.path.getsize(target_path + "-wal") > 0
    return target_path


def _migrate_crash_style_database(database_path: str, monkeypatch) -> None:
    monkeypatch.setattr(db_mod.args, "database_url", f"sqlite:///{database_path}")
    monkeypatch.setattr(db_mod, "Session", None)
    if hasattr(db_mod, "WriteSession"):
        monkeypatch.setattr(db_mod, "WriteSession", None)
    db_mod._migrate_and_bind(f"sqlite:///{database_path}", database_path, db_exists=True)


def test_file_database_configures_runtime_pragmas(file_database):
    with db_mod.create_session() as session:
        pragmas = (
            session.execute(text("PRAGMA journal_mode")).scalar_one(),
            session.execute(text("PRAGMA busy_timeout")).scalar_one(),
            session.execute(text("PRAGMA foreign_keys")).scalar_one(),
        )

    assert pragmas == ("wal", 30000, 1)


def test_runtime_connection_rejects_non_wal_journal_mode():
    class Cursor:
        def execute(self, statement):
            self.statement = statement
            return self

        def fetchone(self):
            return ("delete",)

        def close(self):
            return None

    class Connection:
        def __init__(self):
            self.cursor_instance = Cursor()

        def cursor(self):
            return self.cursor_instance

    with pytest.raises(RuntimeError, match="network filesystems"):
        db_mod._configure_runtime_connection(Connection(), "network.db")


def test_run_write_txn_passes_through_successful_result(memory_database):
    run_write_txn = db_mod.run_write_txn

    assert run_write_txn(lambda _session: "written") == "written"


def test_run_write_txn_retries_locked_operational_errors_then_succeeds(
    memory_database, monkeypatch
):
    run_write_txn = db_mod.run_write_txn
    monkeypatch.setattr(db_mod.time, "sleep", lambda _seconds: None)
    attempts = 0

    def work(_session):
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise OperationalError("SELECT 1", {}, sqlite3.OperationalError("database is locked"))
        return "written"

    assert run_write_txn(work) == "written"
    assert attempts == 3


def test_begin_immediate_caps_a_single_wait_at_the_busy_timeout(monkeypatch):
    clock = {"now": 0.0}

    class Connection:
        def execute(self, _statement):
            raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(db_mod.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        db_mod.time,
        "sleep",
        lambda seconds: clock.__setitem__("now", clock["now"] + seconds),
    )
    db_mod._attempt_lock_deadline.value = 60.0

    try:
        with pytest.raises(OperationalError, match="database is locked"):
            db_mod._begin_immediate(Connection())
    finally:
        db_mod._attempt_lock_deadline.value = None

    assert clock["now"] == pytest.approx(30.0)


def test_begin_immediate_and_write_retries_share_locked_only_classification(
    memory_database, monkeypatch
):
    clock = {"now": 0.0}
    attempts = 0

    class Connection:
        def execute(self, _statement):
            raise sqlite3.OperationalError("database is busy")

    monkeypatch.setattr(db_mod.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        db_mod.time,
        "sleep",
        lambda seconds: clock.__setitem__("now", clock["now"] + seconds),
    )

    with pytest.raises(sqlite3.OperationalError, match="database is busy"):
        db_mod._begin_immediate(Connection())

    def work(_session):
        nonlocal attempts
        attempts += 1
        raise OperationalError("INSERT", {}, sqlite3.OperationalError("database is busy"))

    with pytest.raises(OperationalError, match="database is busy"):
        db_mod.run_write_txn(work)

    assert attempts == 1


def test_run_write_txn_closes_when_rollback_fails_without_masking_work_error(monkeypatch):
    class Session:
        closed = False

        def rollback(self):
            raise RuntimeError("rollback failure")

        def close(self):
            self.closed = True

    session = Session()
    monkeypatch.setattr(db_mod, "WriteSession", lambda: session)

    def work(_session):
        raise ValueError("work failure")

    with pytest.raises(ValueError, match="work failure"):
        db_mod.run_write_txn(work)

    assert session.closed


def test_run_write_txn_reraises_nonretryable_operational_error_without_retry(memory_database):
    run_write_txn = db_mod.run_write_txn
    attempts = 0

    def work(_session):
        nonlocal attempts
        attempts += 1
        raise OperationalError("INSERT", {}, sqlite3.OperationalError("readonly database"))

    with pytest.raises(OperationalError, match="readonly database"):
        run_write_txn(work)

    assert attempts == 1


def test_run_write_txn_reraises_non_operational_error_without_retry(memory_database):
    run_write_txn = db_mod.run_write_txn
    attempts = 0

    def work(_session):
        nonlocal attempts
        attempts += 1
        raise ValueError("body failure")

    with pytest.raises(ValueError, match="body failure"):
        run_write_txn(work)

    assert attempts == 1


def test_run_write_txn_reraises_terminal_locked_error_after_five_attempts(
    memory_database, monkeypatch
):
    run_write_txn = db_mod.run_write_txn
    monkeypatch.setattr(db_mod.time, "sleep", lambda _seconds: None)
    attempts = 0

    def work(_session):
        nonlocal attempts
        attempts += 1
        raise OperationalError("INSERT", {}, sqlite3.OperationalError("database is locked"))

    with pytest.raises(OperationalError, match="database is locked"):
        run_write_txn(work)

    assert attempts == 5


def test_run_write_txn_deadline_gates_attempt_starts(memory_database, monkeypatch):
    run_write_txn = db_mod.run_write_txn
    clock_calls = 0

    def monotonic():
        nonlocal clock_calls
        clock_calls += 1
        return 0.0 if clock_calls == 1 else 60.1

    monkeypatch.setattr(db_mod.time, "monotonic", monotonic)
    monkeypatch.setattr(db_mod.time, "sleep", lambda _seconds: None)
    attempts = 0

    def work(_session):
        nonlocal attempts
        attempts += 1
        raise OperationalError("INSERT", {}, sqlite3.OperationalError("database is locked"))

    with pytest.raises(OperationalError, match="database is locked"):
        run_write_txn(work)

    assert attempts == 1


def test_run_write_txn_held_lock_respects_remaining_deadline(file_database, monkeypatch):
    run_write_txn = db_mod.run_write_txn
    monkeypatch.setattr(db_mod, "_WRITE_TXN_LOCK_RETRY_DEADLINE_SECONDS", 2)
    writer_started = threading.Event()

    def hold_lock():
        holder = sqlite3.connect(file_database)
        try:
            holder.execute("BEGIN IMMEDIATE")
            holder.execute("INSERT INTO tags (name) VALUES (?)", ("deadline-holder",))
            writer_started.set()
            time.sleep(5)
            holder.rollback()
        finally:
            holder.close()

    holder = threading.Thread(target=hold_lock)
    holder.start()
    try:
        assert writer_started.wait(timeout=5)
        started_at = time.monotonic()
        with pytest.raises(OperationalError, match="database is locked"):
            run_write_txn(
                lambda session: session.execute(text("INSERT INTO tags (name) VALUES ('blocked')"))
            )
        elapsed = time.monotonic() - started_at
    finally:
        holder.join(timeout=6)

    assert not holder.is_alive()
    assert 1.5 <= elapsed < 3


def test_run_write_txn_reopens_immediate_transaction_after_intermediate_commit(
    file_database, monkeypatch
):
    run_write_txn = db_mod.run_write_txn
    clock = {"now": 0.0}
    monkeypatch.setattr(db_mod.time, "monotonic", lambda: clock["now"])
    seen_timeouts = []

    def work(session):
        seen_timeouts.append(session.execute(text("PRAGMA busy_timeout")).scalar_one())
        session.commit()
        clock["now"] = 31.0
        seen_timeouts.append(session.execute(text("PRAGMA busy_timeout")).scalar_one())

    run_write_txn(work)

    assert seen_timeouts == [0, 0]


def test_run_write_txn_uses_a_fresh_session_for_each_attempt(memory_database, monkeypatch):
    run_write_txn = db_mod.run_write_txn
    monkeypatch.setattr(db_mod.time, "sleep", lambda _seconds: None)
    sessions = []

    def work(session):
        sessions.append(session)
        if len(sessions) < 3:
            raise OperationalError("INSERT", {}, sqlite3.OperationalError("database is locked"))
        return "written"

    assert run_write_txn(work) == "written"
    assert sessions[0] is not sessions[1]
    assert sessions[1] is not sessions[2]


def test_run_write_txn_reentrancy_guard_resets_after_failure(memory_database):
    run_write_txn = db_mod.run_write_txn

    def nested_work(_session):
        return run_write_txn(lambda _nested_session: None)

    with pytest.raises(RuntimeError):
        run_write_txn(nested_work)

    assert run_write_txn(lambda _session: "reset") == "reset"


def test_memory_database_uses_degraded_write_transaction_wiring(memory_database):
    run_write_txn = db_mod.run_write_txn

    assert db_mod.WriteSession is db_mod.Session
    assert run_write_txn(lambda _session: "written") == "written"


def test_migration_backup_checkpoints_crash_style_wal_before_copy(tmp_path, monkeypatch):
    database_path = _crash_style_wal_database(tmp_path)
    _migrate_crash_style_database(database_path, monkeypatch)
    try:
        with sqlite3.connect(database_path + ".bkp") as backup:
            rows = backup.execute("SELECT name FROM tags WHERE name = ?", (_WAL_SENTINEL,)).fetchall()
    finally:
        _dispose_runtime_engines()

    assert rows == [(_WAL_SENTINEL,)]


def test_failed_migration_restores_crash_style_wal_backup_and_removes_sidecars(
    tmp_path, monkeypatch
):
    database_path = _crash_style_wal_database(tmp_path)

    def fail_upgrade(_config, _target_revision):
        with open(database_path + "-wal", "wb") as wal_file:
            wal_file.write(b"stale wal")
        with open(database_path + "-shm", "wb") as shm_file:
            shm_file.write(b"stale shm")
        raise RuntimeError("upgrade failure")

    monkeypatch.setattr(db_mod.command, "upgrade", fail_upgrade)
    monkeypatch.setattr(db_mod.args, "database_url", f"sqlite:///{database_path}")
    monkeypatch.setattr(db_mod, "Session", None)
    if hasattr(db_mod, "WriteSession"):
        monkeypatch.setattr(db_mod, "WriteSession", None)

    with pytest.raises(RuntimeError, match="upgrade failure"):
        db_mod._migrate_and_bind(f"sqlite:///{database_path}", database_path, db_exists=True)

    with sqlite3.connect(database_path) as restored:
        rows = restored.execute("SELECT name FROM tags WHERE name = ?", (_WAL_SENTINEL,)).fetchall()

    assert rows == [(_WAL_SENTINEL,)]
    assert not os.path.exists(database_path + "-wal")
    assert not os.path.exists(database_path + "-shm")


def test_migration_uses_wal_only_for_runtime_engines(tmp_path, monkeypatch):
    database_path = _crash_style_wal_database(tmp_path)
    created_engine_commands = []
    original_create_engine = db_mod.create_engine

    def create_traced_engine(*args, **kwargs):
        engine = original_create_engine(*args, **kwargs)
        commands = []

        def trace_connection(dbapi_connection, _connection_record):
            dbapi_connection.set_trace_callback(commands.append)

        event.listen(engine, "connect", trace_connection, insert=True)
        created_engine_commands.append(commands)
        return engine

    monkeypatch.setattr(db_mod, "create_engine", create_traced_engine)
    _migrate_crash_style_database(database_path, monkeypatch)
    try:
        with db_mod.create_session() as reader:
            assert reader.execute(text("PRAGMA journal_mode")).scalar_one() == "wal"
        with db_mod.WriteSession() as writer:
            assert writer.execute(text("PRAGMA journal_mode")).scalar_one() == "wal"
    finally:
        _dispose_runtime_engines()

    inspection_commands = "\n".join(created_engine_commands[0]).lower()
    runtime_commands = "\n".join(
        command_text for commands in created_engine_commands[1:] for command_text in commands
    ).lower()
    assert "journal_mode=delete" in inspection_commands
    assert "journal_mode=wal" not in inspection_commands
    assert "journal_mode=wal" in runtime_commands
