import logging
import os
import random
import shutil
import threading
import time
from typing import Callable, TypeVar
from app.logger import log_startup_warning
from utils.install_util import get_missing_requirements_message
from filelock import FileLock, Timeout
from comfy.cli_args import args, database_default_path

_DB_AVAILABLE = False
Session = None
WriteSession = None
_attempt_lock_deadline = threading.local()
_write_txn_state = threading.local()
_WRITE_TXN_LOCK_RETRY_DEADLINE_SECONDS = 60
_WRITE_TXN_BACKOFF_SECONDS = (0.05, 0.1, 0.2, 0.4)
_SQLITE_BUSY_TIMEOUT_MS = 30000
T = TypeVar("T")


try:
    from alembic import command
    from alembic.config import Config
    from alembic.runtime.migration import MigrationContext
    from alembic.script import ScriptDirectory
    from sqlalchemy import create_engine, event
    from sqlalchemy.exc import OperationalError
    from sqlalchemy.orm import Session as SQLAlchemySession, sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.database.models import Base
    import app.assets.database.models  # noqa: F401 — register models with Base.metadata
    import blake3  # noqa: F401 — verify the hard dependency is importable at startup

    _DB_AVAILABLE = True
except ImportError as e:
    log_startup_warning(
        f"""
------------------------------------------------------------------------
Error importing dependencies: {e}
{get_missing_requirements_message()}
This error is happening because ComfyUI now uses a local sqlite database.
------------------------------------------------------------------------
""".strip()
    )


def dependencies_available():
    """
    Temporary function to check if the dependencies are available
    """
    return _DB_AVAILABLE


def can_create_session():
    """
    Temporary function to check if the database is available to create a session
    During initial release there may be environmental issues (or missing dependencies) that prevent the database from being created
    """
    return dependencies_available() and Session is not None


def get_alembic_config():
    root_path = os.path.join(os.path.dirname(__file__), "../..")
    config_path = os.path.abspath(os.path.join(root_path, "alembic.ini"))
    scripts_path = os.path.abspath(os.path.join(root_path, "alembic_db"))

    config = Config(config_path)
    config.set_main_option("script_location", scripts_path)
    config.set_main_option("sqlalchemy.url", get_database_url())

    return config


def get_database_url():
    if args.database_url is not None:
        return args.database_url

    import folder_paths

    db_path = os.path.join(folder_paths.get_user_directory(), "comfyui.db")
    return f"sqlite:///{db_path}"


def get_legacy_default_db_path():
    return database_default_path


def get_db_path():
    url = get_database_url()
    if url.startswith("sqlite:///"):
        return url.split("///", 1)[1]
    else:
        raise ValueError(f"Unsupported database URL '{url}'.")


def copy_legacy_default_db(db_path):
    if args.database_url is not None:
        return

    legacy_db_path = get_legacy_default_db_path()
    if legacy_db_path is None:
        return

    if os.path.abspath(legacy_db_path) == os.path.abspath(db_path):
        return

    if os.path.exists(db_path) or not os.path.exists(legacy_db_path):
        return

    backup_path = legacy_db_path + ".bak"
    if os.path.exists(backup_path):
        return

    os.replace(legacy_db_path, backup_path)
    shutil.copy(backup_path, db_path)
    logging.info(
        f"Renamed legacy database '{legacy_db_path}' to '{backup_path}' and copied it to '{db_path}'"
    )


def prepare_file_db_path(db_path):
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    copy_legacy_default_db(db_path)


_db_lock = None

def _acquire_file_lock(db_path):
    """Acquire an OS-level file lock to prevent multi-process access.

    Uses filelock for cross-platform support (macOS, Linux, Windows).
    The OS automatically releases the lock when the process exits, even on crashes.
    """
    global _db_lock
    lock_path = db_path + ".lock"
    _db_lock = FileLock(lock_path)
    try:
        _db_lock.acquire(timeout=0)
    except Timeout:
        raise RuntimeError(
            f"Could not acquire lock on database '{db_path}'. "
            "Another ComfyUI process may already be using it. "
            "Use --database-url to specify a separate database file."
        )


def _is_memory_db(db_url):
    """Check if the database URL refers to an in-memory SQLite database."""
    return db_url in ("sqlite:///:memory:", "sqlite://")


def init_db():
    db_url = get_database_url()
    logging.debug(f"Database URL: {db_url}")

    if _is_memory_db(db_url):
        _init_memory_db(db_url)
    else:
        _init_file_db(db_url)


def _init_memory_db(db_url):
    """Initialize an in-memory SQLite database using metadata.create_all.

    Alembic migrations don't work with in-memory SQLite because each
    connection gets its own separate database — tables created by Alembic's
    internal connection are lost immediately.
    """
    engine = create_engine(
        db_url,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)

    global Session, WriteSession
    Session = sessionmaker(bind=engine)
    # A second engine would create a separate memory database; this test path is single-threaded.
    WriteSession = Session


def _init_file_db(db_url):
    """Initialize a file-backed SQLite database using Alembic migrations."""
    db_path = get_db_path()
    prepare_file_db_path(db_path)
    db_exists = os.path.exists(db_path)

    # Lock BEFORE any migration work — deliberately diverging from upstream master, whose
    # "it would block Alembic" rationale is false (the lock guards a separate `<db>.lock`
    # file). Only this order makes revision inspection, backup, upgrade and the failure-path
    # restore mutually exclusive between processes.
    _acquire_file_lock(db_path)
    try:
        _migrate_and_bind(db_url, db_path, db_exists)
    except Exception:
        _db_lock.release()
        raise


_DESTRUCTIVE_REVISION = "0007_record_content_split"


def _upgrade_discards_the_catalog(script, target_rev, current_rev):
    return any(
        revision.revision == _DESTRUCTIVE_REVISION
        for revision in script.iterate_revisions(upper=target_rev, lower=current_rev)
    )


def _configure_runtime_connection(dbapi_connection, db_path):
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
        journal_mode = cursor.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if journal_mode.lower() != "wal":
            raise RuntimeError(
                f"SQLite WAL could not be enabled for database '{db_path}'. "
                "SQLite WAL is not supported on network filesystems."
            )
        cursor.execute(f"PRAGMA busy_timeout={_SQLITE_BUSY_TIMEOUT_MS}")
    finally:
        cursor.close()


def _writer_busy_timeout_ms():
    deadline = getattr(_attempt_lock_deadline, "value", None)
    if deadline is None:
        return _SQLITE_BUSY_TIMEOUT_MS
    remaining_ms = int((deadline - time.monotonic()) * 1000)
    return max(1, min(_SQLITE_BUSY_TIMEOUT_MS, remaining_ms))


def _configure_writer_busy_timeout(dbapi_connection):
    cursor = dbapi_connection.execute(f"PRAGMA busy_timeout = {_writer_busy_timeout_ms()}")
    cursor.close()


def _migrate_and_bind(db_url, db_path, db_exists):
    config = get_alembic_config()
    inspection_engine = create_engine(db_url)

    @event.listens_for(inspection_engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    try:
        with inspection_engine.connect() as inspection_connection:
            context = MigrationContext.configure(inspection_connection)
            current_rev = context.get_current_revision()
            script = ScriptDirectory.from_config(config)
            target_rev = script.get_current_head()
            needs_upgrade = target_rev is not None and current_rev != target_rev

            if target_rev is None:
                logging.warning("No target revision found.")
            elif needs_upgrade and db_exists:
                # WAL persists in the file, so Phase M makes the main-file backup self-contained.
                inspection_connection.rollback()
                inspection_connection.exec_driver_sql("PRAGMA wal_checkpoint(TRUNCATE)")
                journal_mode = inspection_connection.exec_driver_sql(
                    "PRAGMA journal_mode=DELETE"
                ).scalar_one()
                if journal_mode.lower() != "delete":
                    raise RuntimeError(
                        f"SQLite journal mode could not be reset before backing up '{db_path}'."
                    )
    finally:
        inspection_engine.dispose()

    if needs_upgrade:
        backup_path = db_path + ".bkp" if db_exists else None
        if backup_path is not None:
            shutil.copy(db_path, backup_path)
        try:
            command.upgrade(config, target_rev)
            logging.info(f"Database upgraded from {current_rev} to {target_rev}")
        except Exception:
            if backup_path is not None:
                for sidecar_path in (db_path + "-wal", db_path + "-shm"):
                    if os.path.exists(sidecar_path):
                        os.remove(sidecar_path)
                shutil.copy(backup_path, db_path)
                os.remove(backup_path)
            logging.exception("Error upgrading database: ")
            raise

        if backup_path is not None and _upgrade_discards_the_catalog(script, target_rev, current_rev):
            log_startup_warning(
                f"The asset catalog was rebuilt from scratch by migration "
                f"{_DESTRUCTIVE_REVISION}: manual tags, user metadata, previews, renames, "
                f"API-created records and job_id links from the previous database were "
                f"discarded. The database from before the upgrade was kept at {backup_path}."
            )

    # Redundant with busy_timeout by design: both set pysqlite's 30-second limit.
    reader_engine = create_engine(db_url, connect_args={"timeout": 30})

    @event.listens_for(reader_engine, "connect")
    def set_reader_sqlite_pragma(dbapi_connection, connection_record):
        _configure_runtime_connection(dbapi_connection, db_path)

    writer_engine = create_engine(db_url, connect_args={"timeout": 30})

    @event.listens_for(writer_engine, "connect")
    def set_writer_sqlite_pragma(dbapi_connection, connection_record):
        dbapi_connection.isolation_level = None
        _configure_runtime_connection(dbapi_connection, db_path)

    @event.listens_for(writer_engine, "begin")
    def begin_immediate(connection):
        connection.exec_driver_sql("BEGIN IMMEDIATE")

    @event.listens_for(writer_engine, "checkout")
    def cap_writer_busy_timeout(dbapi_connection, connection_record, connection_proxy):
        _configure_writer_busy_timeout(dbapi_connection)

    @event.listens_for(writer_engine, "begin", insert=True)
    def cap_writer_busy_timeout_before_begin(connection):
        connection.exec_driver_sql(f"PRAGMA busy_timeout = {_writer_busy_timeout_ms()}")

    with reader_engine.connect():
        pass
    with writer_engine.connect():
        pass
    global Session, WriteSession
    Session = sessionmaker(bind=reader_engine)
    WriteSession = sessionmaker(bind=writer_engine)


def create_session():
    return Session()


def run_write_txn(work: Callable[["SQLAlchemySession"], T]) -> T:
    """Run a write callback with bounded lock retries; its own work is not deadline-limited."""
    if getattr(_write_txn_state, "active", False):
        raise RuntimeError("run_write_txn cannot be nested")

    # Nested helpers could commit the outer callback's work.
    _write_txn_state.active = True
    retry_deadline = time.monotonic() + _WRITE_TXN_LOCK_RETRY_DEADLINE_SECONDS
    locked_error = None
    try:
        for attempt in range(len(_WRITE_TXN_BACKOFF_SECONDS) + 1):
            if attempt > 0:
                if time.monotonic() >= retry_deadline:
                    raise locked_error
                backoff_seconds = _WRITE_TXN_BACKOFF_SECONDS[attempt - 1]
                time.sleep(random.uniform(backoff_seconds * 0.5, backoff_seconds * 1.5))
                if time.monotonic() >= retry_deadline:
                    raise locked_error

            _attempt_lock_deadline.value = retry_deadline
            connection = None
            session = None
            try:
                if WriteSession is Session:
                    session = WriteSession()
                else:
                    writer_engine = getattr(WriteSession, "kw", {}).get("bind")
                    if writer_engine is None:
                        session = WriteSession()
                    else:
                        connection = writer_engine.connect()
                        _configure_writer_busy_timeout(connection.connection.driver_connection)
                        connection.begin()
                        session = SQLAlchemySession(bind=connection, join_transaction_mode="control_fully")
                result = work(session)
                session.commit()
                if connection is not None and connection.in_transaction():
                    connection.commit()
                return result
            except OperationalError as exc:
                if "locked" not in str(exc.orig):
                    raise
                locked_error = exc
            finally:
                if session is not None:
                    session.rollback()
                    session.close()
                if connection is not None:
                    connection.rollback()
                    connection.close()
                _attempt_lock_deadline.value = None

            if attempt == len(_WRITE_TXN_BACKOFF_SECONDS) or time.monotonic() >= retry_deadline:
                raise locked_error
    finally:
        _attempt_lock_deadline.value = None
        _write_txn_state.active = False
