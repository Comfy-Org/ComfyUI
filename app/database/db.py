import logging
import os
import shutil
import sqlite3
import time
from contextlib import closing
from app.logger import log_startup_warning
from utils.install_util import get_missing_requirements_message
from filelock import FileLock, Timeout
from comfy.cli_args import args, database_default_path

_DB_AVAILABLE = False
Session = None
WriteSession = None


try:
    from alembic import command
    from alembic.config import Config
    from alembic.runtime.migration import MigrationContext
    from alembic.script import ScriptDirectory
    from sqlalchemy import create_engine, event
    from sqlalchemy.exc import OperationalError
    from sqlalchemy.orm import sessionmaker
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

    if os.path.exists(legacy_db_path + "-wal"):
        # Fold committed WAL pages back into the file before it is renamed and copied.
        try:
            with closing(sqlite3.connect(legacy_db_path)) as legacy:
                busy, _, _ = legacy.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        except sqlite3.Error:
            busy = 1
        if busy:
            logging.warning(
                f"Not relocating legacy database '{legacy_db_path}': its WAL could not be "
                f"checkpointed, so it may still be in use."
            )
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


_BACKUP_TIMEOUT_SECONDS = 5.0
_SQLITE_BUSY, _SQLITE_LOCKED = 5, 6  # sqlite3 only exports these names from Python 3.11


def _backup_database(source_path, destination_path):
    # A plain file copy misses committed pages still in the WAL file.
    # sqlite3's backup() retries a locked database forever, so bound it: another
    # client holding the destination must not hang startup.
    deadline = time.monotonic() + _BACKUP_TIMEOUT_SECONDS

    def give_up_when_locked_too_long(status, remaining, total):
        if status in (_SQLITE_BUSY, _SQLITE_LOCKED) and time.monotonic() > deadline:
            raise TimeoutError(f"'{destination_path}' stayed locked; database backup abandoned")

    with closing(sqlite3.connect(source_path)) as source:
        with closing(sqlite3.connect(destination_path)) as destination:
            source.backup(destination, progress=give_up_when_locked_too_long)
    shutil.copymode(source_path, destination_path)


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
    WriteSession = Session


def _init_file_db(db_url):
    """Initialize a file-backed SQLite database using Alembic migrations."""
    db_path = get_db_path()
    prepare_file_db_path(db_path)

    # Lock before legacy import, migration inspection, backup, upgrade, and failure recovery.
    # The separate `<db>.lock` file does not block Alembic; this ordering keeps the sequence
    # process-exclusive.
    _acquire_file_lock(db_path)
    try:
        copy_legacy_default_db(db_path)
        db_exists = os.path.exists(db_path)
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


def _migrate_and_bind(db_url, db_path, db_exists):
    config = get_alembic_config()

    # Check if we need to upgrade
    engine = create_engine(db_url)
    write_engine = create_engine(db_url)

    # Enable foreign key enforcement for SQLite
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    # Writes go through a separate engine whose transactions take the write lock up front.
    # pysqlite otherwise defers BEGIN until the first INSERT/UPDATE/DELETE, so a transaction
    # that reads first can fail to upgrade to a write lock without waiting on busy_timeout.
    # Following SQLAlchemy's pysqlite recipe, the driver's own transaction handling is
    # switched off so the only BEGIN issued is the BEGIN IMMEDIATE below.
    @event.listens_for(write_engine, "connect")
    def configure_write_connection(dbapi_connection, connection_record):
        dbapi_connection.isolation_level = None
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    @event.listens_for(write_engine, "begin")
    def begin_immediate(connection):
        connection.exec_driver_sql("BEGIN IMMEDIATE")

    conn = engine.connect()

    try:
        journal_mode = conn.exec_driver_sql("PRAGMA journal_mode=WAL").scalar_one()
    except OperationalError:
        logging.warning("Could not enable SQLite WAL mode; continuing with the default journal mode.")
    else:
        if journal_mode.lower() != "wal":
            logging.warning("SQLite WAL mode unavailable; continuing with %s journal mode.", journal_mode)

    context = MigrationContext.configure(conn)
    current_rev = context.get_current_revision()

    script = ScriptDirectory.from_config(config)
    target_rev = script.get_current_head()

    if target_rev is None:
        logging.warning("No target revision found.")
    elif current_rev != target_rev:
        # Backup the database pre upgrade
        backup_path = db_path + ".bkp"
        if db_exists:
            _backup_database(db_path, backup_path)
        else:
            backup_path = None

        try:
            command.upgrade(config, target_rev)
            logging.info(f"Database upgraded from {current_rev} to {target_rev}")
        except Exception as e:
            logging.exception("Error upgrading database: ")
            if backup_path:
                # Restore the database from backup if upgrade fails
                try:
                    _backup_database(backup_path, db_path)
                    os.remove(backup_path)
                except Exception:
                    logging.exception(
                        f"Restoring the database from its pre-upgrade backup, or removing the "
                        f"backup afterwards, failed; the pre-upgrade copy is kept at {backup_path}"
                    )
            raise e

        if backup_path and _upgrade_discards_the_catalog(script, target_rev, current_rev):
            log_startup_warning(
                f"The asset catalog was rebuilt from scratch by migration "
                f"{_DESTRUCTIVE_REVISION}: manual tags, user metadata, previews, renames, "
                f"API-created records and job_id links from the previous database were "
                f"discarded. Record deletions were also discarded, so files still on disk "
                f"will be catalogued again. The database from before the upgrade was kept "
                f"at {backup_path}."
            )

    conn.close()

    global Session, WriteSession
    Session = sessionmaker(bind=engine)
    WriteSession = sessionmaker(bind=write_engine)


def create_session():
    return Session()


def create_write_session():
    """A session whose transactions open with BEGIN IMMEDIATE. Do filesystem work before
    using it: the write lock is held from the first statement until commit. Do not open
    one inside another: the inner one waits out busy_timeout for the outer's lock, then
    fails with "database is locked", indistinguishable from real contention. Rule out a
    nested session before investigating lock contention."""
    return WriteSession()
