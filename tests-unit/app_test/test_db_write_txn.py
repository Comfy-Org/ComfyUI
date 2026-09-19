import logging
import os
import shutil
import sqlite3
import threading

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
    monkeypatch.setattr(db_mod.args, "enable_assets", True)
    monkeypatch.setattr(db_mod.args, "database_url", f"sqlite:///{database_path}")
    monkeypatch.setattr(db_mod, "Session", None)
    monkeypatch.setattr(db_mod, "_db_lock", None)
    monkeypatch.setattr(db_mod, "WriteSession", None)
    db_mod.init_db()
    yield database_path
    _dispose_runtime_engines()
    db_mod._db_lock.release(force=True)


@pytest.fixture
def memory_database(monkeypatch):
    monkeypatch.setattr(db_mod.args, "enable_assets", True)
    monkeypatch.setattr(db_mod.args, "database_url", "sqlite:///:memory:")
    monkeypatch.setattr(db_mod, "Session", None)
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
    monkeypatch.setattr(db_mod.args, "enable_assets", True)
    monkeypatch.setattr(db_mod.args, "database_url", f"sqlite:///{database_path}")
    monkeypatch.setattr(db_mod, "Session", None)
    monkeypatch.setattr(db_mod, "WriteSession", None)
    db_mod._migrate_and_bind(f"sqlite:///{database_path}", database_path, db_exists=True)


def test_file_database_configures_runtime_pragmas(file_database):
    with db_mod.create_session() as session:
        pragmas = (
            session.execute(text("PRAGMA journal_mode")).scalar_one(),
            session.execute(text("PRAGMA busy_timeout")).scalar_one(),
            session.execute(text("PRAGMA foreign_keys")).scalar_one(),
            session.execute(text("PRAGMA query_only")).scalar_one(),
        )

    assert pragmas == ("wal", 30000, 1, 1)


def test_reader_session_cannot_write(file_database):
    with pytest.raises(OperationalError, match="readonly"):
        with db_mod.create_session() as reader:
            reader.execute(text("INSERT INTO tags (name) VALUES ('ro')"))
            reader.commit()

    db_mod.run_write_txn(
        lambda session: session.execute(text("INSERT INTO tags (name) VALUES ('rw')"))
    )

    with db_mod.create_session() as fresh_reader:
        names = (
            fresh_reader.execute(text("SELECT name FROM tags WHERE name IN ('ro', 'rw')"))
            .scalars()
            .all()
        )

    assert names == ["rw"]


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


@pytest.mark.parametrize(
    ("sqlite_errorname", "sqlite_errorcode", "message", "is_retryable"),
    [
        pytest.param("SQLITE_BUSY", None, "contention", True, id="busy-name"),
        pytest.param("SQLITE_BUSY_SNAPSHOT", None, "snapshot contention", True, id="busy-snapshot-name"),
        pytest.param("SQLITE_BUSY_TIMEOUT", None, "timed contention", True, id="busy-timeout-name"),
        pytest.param("SQLITE_BUSY_RECOVERY", None, "recovery contention", True, id="busy-recovery-name"),
        pytest.param("SQLITE_LOCKED", None, "table contention", True, id="locked-name"),
        pytest.param("SQLITE_LOCKED_SHAREDCACHE", None, "shared-cache contention", True, id="locked-sharedcache-name"),
        pytest.param(None, sqlite3.SQLITE_BUSY_SNAPSHOT, "snapshot contention", True, id="busy-snapshot-code"),
        pytest.param(None, sqlite3.SQLITE_LOCKED_VTAB, "virtual table locked", False, id="locked-vtab-code"),
        pytest.param(None, None, "database table is locked", True, id="python-310-locked-fallback"),
        pytest.param(None, None, "database is busy", False, id="python-310-busy-fallback"),
    ],
)
def test_begin_immediate_and_write_retries_agree_on_sqlite_result_classification(
    memory_database,
    monkeypatch,
    sqlite_errorname,
    sqlite_errorcode,
    message,
    is_retryable,
):
    def make_error():
        error = sqlite3.OperationalError(message)
        if sqlite_errorname is not None:
            error.sqlite_errorname = sqlite_errorname
        if sqlite_errorcode is not None:
            error.sqlite_errorcode = sqlite_errorcode
        return error

    clock = {"now": 0.0}

    class Connection:
        def execute(self, _statement):
            raise make_error()

    monkeypatch.setattr(db_mod.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        db_mod.time,
        "sleep",
        lambda _seconds: clock.__setitem__("now", 31.0),
    )

    if is_retryable:
        with pytest.raises(OperationalError):
            db_mod._begin_immediate(Connection())
    else:
        with pytest.raises(sqlite3.OperationalError):
            db_mod._begin_immediate(Connection())

    monkeypatch.setattr(db_mod.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(db_mod.time, "sleep", lambda _seconds: None)
    attempts = 0

    def work(_session):
        nonlocal attempts
        attempts += 1
        raise OperationalError("INSERT", {}, make_error())

    with pytest.raises(OperationalError):
        db_mod.run_write_txn(work)

    assert attempts == (5 if is_retryable else 1)


def test_retryable_lock_error_uses_the_python_310_message_fallback():
    assert db_mod._is_retryable_lock_error(sqlite3.OperationalError("database table is locked"))
    assert not db_mod._is_retryable_lock_error(sqlite3.OperationalError("database is busy"))


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


def test_run_write_txn_retries_when_rollback_fails_after_handled_lock(monkeypatch, caplog):
    class Session:
        def __init__(self, rollback_fails):
            self.closed = False
            self.rollback_fails = rollback_fails

        def commit(self):
            return None

        def rollback(self):
            if self.rollback_fails:
                raise RuntimeError("rollback failure")

        def close(self):
            self.closed = True

    first_session = Session(rollback_fails=True)
    second_session = Session(rollback_fails=False)
    sessions = iter((first_session, second_session))
    monkeypatch.setattr(db_mod, "WriteSession", lambda: next(sessions))
    monkeypatch.setattr(db_mod.time, "sleep", lambda _seconds: None)
    attempts = 0

    def work(_session):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OperationalError("INSERT", {}, sqlite3.OperationalError("database is locked"))
        return "written"

    with caplog.at_level(logging.WARNING):
        assert db_mod.run_write_txn(work) == "written"

    assert first_session.closed
    assert second_session.closed
    assert attempts == 2
    assert "rollback failed" in caplog.text


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


def test_run_write_txn_reraises_in_callback_locked_error_after_five_attempts(
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
    monkeypatch.setattr(db_mod, "_WRITE_TXN_LOCK_RETRY_DEADLINE_SECONDS", 0.5)
    lock_held = threading.Event()
    release = threading.Event()

    def hold_lock():
        holder = sqlite3.connect(file_database)
        try:
            holder.execute("BEGIN IMMEDIATE")
            holder.execute("INSERT INTO tags (name) VALUES (?)", ("deadline-holder",))
            lock_held.set()
            release.wait(timeout=30)
            holder.rollback()
        finally:
            holder.close()

    holder = threading.Thread(target=hold_lock)
    holder.start()
    try:
        assert lock_held.wait(timeout=5)
        with pytest.raises(OperationalError, match="database is locked"):
            run_write_txn(
                lambda session: session.execute(text("INSERT INTO tags (name) VALUES ('blocked')"))
            )
        assert not release.is_set(), (
            "refusal must come from the retry deadline, not from the holder releasing"
        )
        assert holder.is_alive(), "holder must still own the write lock at refusal time"
    finally:
        release.set()
        holder.join(timeout=6)

    assert not holder.is_alive()


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


def test_memory_database_leaves_its_shared_session_writable(memory_database):
    with db_mod.create_session() as session:
        query_only = session.execute(text("PRAGMA query_only")).scalar_one()

    assert query_only == 0


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
    monkeypatch.setattr(db_mod.args, "enable_assets", True)
    monkeypatch.setattr(db_mod.args, "database_url", f"sqlite:///{database_path}")
    monkeypatch.setattr(db_mod, "Session", None)
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


@pytest.fixture
def file_database_without_assets(tmp_path, monkeypatch):
    database_path = str(tmp_path / "assets.db")
    monkeypatch.setattr(db_mod.args, "enable_assets", False)
    monkeypatch.setattr(db_mod.args, "database_url", f"sqlite:///{database_path}")
    monkeypatch.setattr(db_mod, "Session", None)
    monkeypatch.setattr(db_mod, "_db_lock", None)
    monkeypatch.setattr(db_mod, "WriteSession", None)
    db_mod.init_db()
    yield database_path
    _dispose_runtime_engines()
    db_mod._db_lock.release(force=True)


def test_disabled_assets_startup_leaves_journal_mode_unpromoted(file_database_without_assets):
    with db_mod.create_session() as session:
        session.execute(text("SELECT 1"))

    connection = sqlite3.connect(file_database_without_assets)
    try:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        connection.close()

    assert journal_mode.lower() != "wal"


def test_disabled_assets_startup_leaves_its_shared_session_writable(file_database_without_assets):
    with db_mod.create_session() as session:
        query_only = session.execute(text("PRAGMA query_only")).scalar_one()
        session.execute(text("INSERT INTO tags (name) VALUES ('shared-writer')"))
        session.commit()

    with db_mod.create_session() as fresh_reader:
        names = (
            fresh_reader.execute(text("SELECT name FROM tags WHERE name = 'shared-writer'"))
            .scalars()
            .all()
        )

    assert query_only == 0
    assert names == ["shared-writer"]


def test_disabled_assets_startup_writes_no_wal_sidecars(file_database_without_assets):
    with db_mod.create_session() as session:
        session.execute(text("SELECT 1"))

    directory = os.path.dirname(file_database_without_assets)
    sidecars = [name for name in os.listdir(directory) if name.endswith(("-wal", "-shm"))]

    assert sidecars == []


def test_disabled_assets_startup_survives_a_filesystem_that_rejects_wal(tmp_path, monkeypatch):
    database_path = str(tmp_path / "assets.db")
    monkeypatch.setattr(db_mod.args, "enable_assets", False)
    monkeypatch.setattr(db_mod.args, "database_url", f"sqlite:///{database_path}")
    monkeypatch.setattr(db_mod, "Session", None)
    monkeypatch.setattr(db_mod, "_db_lock", None)
    monkeypatch.setattr(db_mod, "WriteSession", None)

    def reject_wal(dbapi_connection, db_path):
        raise RuntimeError(f"SQLite WAL could not be enabled for database '{db_path}'.")

    monkeypatch.setattr(db_mod, "_configure_runtime_connection", reject_wal)
    try:
        db_mod.init_db()
        with db_mod.create_session() as session:
            session.execute(text("SELECT 1"))
    finally:
        _dispose_runtime_engines()
        if db_mod._db_lock is not None:
            db_mod._db_lock.release(force=True)


def test_begin_time_contention_does_not_reach_the_in_callback_attempt_count(
    file_database, monkeypatch
):
    """Contention at BEGIN IMMEDIATE spends its deadline polling, not on the backoff table.

    The 5-attempt regime is reachable only when the lock error surfaces from inside the
    callback, so a test that fabricates it there cannot detect this difference.
    """
    run_write_txn = db_mod.run_write_txn
    monkeypatch.setattr(db_mod, "_SQLITE_BUSY_TIMEOUT_MS", 200)
    monkeypatch.setattr(db_mod, "_WRITE_TXN_LOCK_RETRY_DEADLINE_SECONDS", 0.5)

    lock_held = threading.Event()
    release = threading.Event()

    def hold_lock():
        holder = sqlite3.connect(file_database)
        try:
            holder.execute("BEGIN IMMEDIATE")
            holder.execute("INSERT INTO tags (name) VALUES (?)", ("begin-contention-holder",))
            lock_held.set()
            release.wait(timeout=30)
            holder.rollback()
        finally:
            holder.close()

    attempts = 0

    def work(session):
        nonlocal attempts
        attempts += 1
        session.execute(text("INSERT INTO tags (name) VALUES ('contended')"))

    holder = threading.Thread(target=hold_lock)
    holder.start()
    try:
        assert lock_held.wait(timeout=5)
        with pytest.raises(OperationalError, match="database is locked"):
            run_write_txn(work)
        assert holder.is_alive(), "holder must still own the write lock at refusal time"
    finally:
        release.set()
        holder.join(timeout=6)

    assert not holder.is_alive()
    assert 0 < attempts < 5, (
        f"BEGIN-time contention consumed the deadline in {attempts} attempts; "
        "the 5-attempt backoff table governs only in-callback lock errors"
    )
