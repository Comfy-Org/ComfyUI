import importlib
import logging
import os
import shutil
import sqlite3
import threading
import time
from contextlib import closing, suppress
from app.assets.event_log import error_kind
from app.logger import log_startup_warning
from utils.install_util import get_missing_requirements_message
from filelock import FileLock, Timeout
from comfy.cli_args import args, database_default_path

_DB_AVAILABLE = False
Session = None
WriteSession = None

# The packages the imports below need; keep in step with them.
_DEPENDENCIES = ("sqlalchemy", "alembic", "blake3")

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


def missing_dependencies():
    """Names of the database packages that fail to import."""
    missing = []
    for name in _DEPENDENCIES:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    return missing


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
_cannot_move_aside = False  # set when the corrupt database stays in use, so recovery stops
_DAILY_BACKUP_INTERVAL_SECONDS = 24 * 60 * 60
_LOCK_WAIT_SECONDS = 5.0

def _acquire_file_lock(db_path):
    """Acquire an OS-level file lock to prevent multi-process access.

    Uses filelock for cross-platform support (macOS, Linux, Windows).
    The OS automatically releases the lock when the process exits, even on crashes.
    If the lock is held, waits up to _LOCK_WAIT_SECONDS for it to be released, since a
    relaunch can start while the previous process is still exiting. The lock is never
    taken from a holder.
    """
    global _db_lock
    lock_path = db_path + ".lock"
    _db_lock = FileLock(lock_path)
    try:
        _db_lock.acquire(timeout=0)
    except Timeout:
        logging.info(f"Database lock is held; waiting up to {_LOCK_WAIT_SECONDS:g}s for it to be released")
        try:
            _db_lock.acquire(timeout=_LOCK_WAIT_SECONDS)
        except Timeout:
            raise RuntimeError(
                f"Could not acquire lock on database '{db_path}'. "
                "Another ComfyUI process may already be using it. "
                "Use --database-url to specify a separate database file."
            )


def lock_holder_db_path():
    """The database path if another process holds its lock, else None.

    Never waits and never keeps the lock: a free lock is taken and released at once.
    A missing lock file means no holder, so none is created.
    """
    try:
        db_path = get_db_path()
    except ValueError:
        return None
    lock_path = db_path + ".lock"
    if not os.path.exists(lock_path):
        return None
    probe = FileLock(lock_path)
    try:
        probe.acquire(timeout=0)
        probe.release()
    except Timeout:
        return db_path
    except Exception as e:
        # The check is advisory, so it must never stop startup.
        logging.debug(f"Could not check the database lock '{lock_path}': {e}")
    return None


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
        try:
            _migrate_and_bind(db_url, db_path, os.path.exists(db_path))
        except Exception as e:
            if not is_recoverable_corruption(e):
                raise
            _quarantine_and_restore(db_path, e)
            _migrate_and_bind(db_url, db_path, os.path.exists(db_path))
    except Exception:
        _db_lock.release()
        raise


def _is_default_db():
    """The default database file, also when --database-url names it; any other database may be managed elsewhere."""
    import folder_paths

    default_path = os.path.join(folder_paths.get_user_directory(), "comfyui.db")
    try:
        return os.path.normcase(os.path.abspath(get_db_path())) == os.path.normcase(os.path.abspath(default_path))
    except ValueError:  # not a SQLite file URL
        return False


def is_recoverable_corruption(error):
    # Only the default database (see _is_default_db).
    # SQLITE_CORRUPT or SQLITE_NOTADB, never "database is locked". The error can come from
    # another file (the pre-upgrade backup), so the live database must fail its check too.
    if _cannot_move_aside or not _is_default_db() or error_kind(error) != "database_corrupt":
        return False
    return not _passes_integrity_check(get_db_path())


def recover_from_corruption(error):
    """Replace a default database found corrupt after init_db. False for any other error."""
    if not is_recoverable_corruption(error):
        return False
    db_path = get_db_path()
    for session_factory in (Session, WriteSession):
        session_factory.kw["bind"].dispose()  # Windows refuses to rename an open file
    try:
        _quarantine_and_restore(db_path, error)
    except Exception:  # e.g. held open on Windows: asset startup now logs the corruption, as before
        logging.exception(f"Could not recover the corrupt database '{db_path}'; continuing without recovery")
        if not os.path.exists(db_path):
            raise  # a move that couldn't be undone: fail rather than run on a new empty file
        global _cannot_move_aside
        _cannot_move_aside = True
        return True
    _migrate_and_bind(get_database_url(), db_path, os.path.exists(db_path))
    return True


def _connect_existing(path):  # sqlite3.connect would create a missing file
    if not os.path.exists(path):
        raise sqlite3.OperationalError("unable to open database file")
    return sqlite3.connect(path)


def _passes_integrity_check(path):
    # Only after an error: unlike quick_check it also finds an index that disagrees with its table.
    try:
        with closing(_connect_existing(path)) as conn:
            return conn.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    except sqlite3.Error as e:
        return error_kind(e) == "database_locked"  # another process has it: not shown to be corrupt


def _quarantine_and_restore(db_path, error):
    """Rename a corrupt database aside and restore the daily backup in its place if it is sound.
    With no sound backup, no database file is left, so the caller creates a new one. On a failure,
    every move is undone, so the corrupt database stays in place to be recovered next time."""
    quarantine_path = f"{db_path}.corrupt-{time.strftime('%Y%m%d-%H%M%S')}"
    backup_path, restore_path = db_path + ".daily-backup", db_path + ".restore-tmp"
    outcome = "Database recreated empty: there was no sound daily backup"
    try:
        try:  # before anything moves: a kill during the slow copy and check leaves it to recover again
            shutil.copyfile(backup_path, restore_path)
            sound = _passes_integrity_check(restore_path)
            taken = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(backup_path)))
        except FileNotFoundError:
            sound = False  # no backup; any other error stops here, before anything moves
        moved = []
        try:
            # With its WAL and journal: when another connection keeps one, SQLite would replay it.
            for suffix in ("-wal", "-shm", "-journal", ""):
                if os.path.exists(db_path + suffix):
                    os.replace(db_path + suffix, quarantine_path + suffix)
                    moved.append((db_path + suffix, quarantine_path + suffix))
            if sound:
                # Atomic: a crash leaves either no database or the whole backup, never part of it.
                os.replace(restore_path, db_path)
                outcome = f"Database restored from the daily backup taken {taken}; later changes are lost"
        except OSError:
            logging.exception(f"Could not move the corrupt database '{db_path}' aside or restore the backup")
            for src, dst in reversed(moved):
                try:
                    os.replace(dst, src)
                except OSError:
                    logging.exception(f"Could not move '{dst}' back to '{src}'")
            raise error
    finally:
        with suppress(OSError):  # a leftover copy must not undo a finished restore
            os.remove(restore_path)
    log_startup_warning(
        f"Database quarantined: '{db_path}' was corrupt ({getattr(error, 'orig', error)}) and was "
        f"moved to '{quarantine_path}'. {outcome}. The asset catalog is rebuilt by rescanning your files."
    )


def start_daily_backup():
    """Keep a compact copy of the default database, refreshed once a day, off the startup path."""
    if not _is_default_db():
        return
    db_path = get_db_path()
    threading.Thread(
        target=_keep_daily_backup, args=(db_path, db_path + ".daily-backup"), name="database-daily-backup", daemon=True
    ).start()


def _keep_daily_backup(db_path, backup_path):
    # A daemon thread: exiting mid-backup leaves only a .tmp file, which the next backup removes.
    while True:
        due_in = 0  # no backup yet
        with suppress(OSError):
            age = time.time() - os.path.getmtime(backup_path)
            due_in = _DAILY_BACKUP_INTERVAL_SECONDS - age if age >= 0 else 0  # dated in the future: the clock moved back
        if due_in <= 0:
            _write_daily_backup(db_path, backup_path)
            due_in = _DAILY_BACKUP_INTERVAL_SECONDS  # also the retry delay after a failure
        time.sleep(due_in)


def _write_daily_backup(db_path, backup_path):
    tmp_path = backup_path + ".tmp"
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)  # left by a process that exited mid-backup
        with closing(_connect_existing(db_path)) as conn:  # a deleted database must not back up as empty
            conn.execute("VACUUM INTO ?", (tmp_path,))
        # VACUUM INTO copies a row that breaks a constraint; only a copy a restore accepts replaces the backup.
        if not _passes_integrity_check(tmp_path):
            raise sqlite3.DatabaseError("the new backup failed its integrity check")
        os.replace(tmp_path, backup_path)
        logging.info(f"Database daily backup written to '{backup_path}'")
    except Exception:
        logging.exception("Database daily backup failed; keeping the previous one")


# NORMAL: commits skip the fsync that held the write lock. A power loss or OS crash can
# roll back recent commits but cannot corrupt the database. "FULL" is SQLite's default.
WAL_SYNCHRONOUS = "NORMAL"


def _set_wal_synchronous(dbapi_connection, connection_record=None):
    dbapi_connection.execute(f"PRAGMA synchronous={WAL_SYNCHRONOUS}")


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
        _migrate(conn, engine, write_engine, config, db_path, db_exists)
    except BaseException:
        # Close every handle: Windows refuses to rename the file aside while one is open.
        conn.close()
        engine.dispose()
        write_engine.dispose()
        raise


def _migrate(conn, engine, write_engine, config, db_path, db_exists):
    try:
        journal_mode = conn.exec_driver_sql("PRAGMA journal_mode=WAL").scalar_one()
    except OperationalError:
        logging.warning("Could not enable SQLite WAL mode; continuing with the default journal mode.")
    else:
        if journal_mode.lower() != "wal":
            logging.warning("SQLite WAL mode unavailable; continuing with %s journal mode.", journal_mode)
        else:
            # Only in WAL mode: with a rollback journal, NORMAL risks corruption on power loss.
            event.listen(engine, "connect", _set_wal_synchronous)
            event.listen(write_engine, "connect", _set_wal_synchronous)
            _set_wal_synchronous(conn.connection.dbapi_connection)  # opened before the hooks

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
