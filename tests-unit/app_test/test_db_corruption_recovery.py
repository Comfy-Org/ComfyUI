import glob
import os
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from filelock import FileLock

import app.logger
from app.assets import lifecycle
from app.assets.manager import AssetsEnabled
from app.assets.event_log import error_kind
from app.database import db as db_module
from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import main  # noqa: E402

_PAGE = 4096


def _config(db_path: str) -> Config:
    root = os.path.join(os.path.dirname(__file__), "../..")
    cfg = Config(os.path.abspath(os.path.join(root, "alembic.ini")))
    cfg.set_main_option("script_location", os.path.abspath(os.path.join(root, "alembic_db")))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return cfg


def _head() -> str:
    return ScriptDirectory.from_config(_config("unused.db")).get_current_head()


def _revision(db_path: str) -> str:
    with closing(sqlite3.connect(db_path)) as conn:
        return conn.execute("SELECT version_num FROM alembic_version").fetchone()[0]


def _make_db(db_path: str, revision: str = "head", marker: str | None = None) -> None:
    command.upgrade(_config(db_path), revision)
    with closing(sqlite3.connect(db_path)) as conn:
        if marker:
            conn.execute("CREATE TABLE marker (value TEXT)")
            conn.execute("INSERT INTO marker VALUES (?)", (marker,))
            conn.commit()
        conn.execute("PRAGMA journal_mode=DELETE")  # one self-contained file to corrupt


def _use_wal(db_path: str) -> None:
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")


def _overwrite_page_of(db_path: str, table: str) -> None:
    with closing(sqlite3.connect(db_path)) as conn:
        page = conn.execute("SELECT rootpage FROM sqlite_master WHERE name = ?", (table,)).fetchone()[0]
    with open(db_path, "r+b") as f:
        f.seek((page - 1) * _PAGE)
        f.write(b"\xa5" * _PAGE)


def _overwrite_header(db_path: str) -> None:
    with open(db_path, "r+b") as f:
        f.write(b"\xa5" * 100)


def _truncate(db_path: str) -> None:
    with open(db_path, "r+b") as f:
        f.truncate(os.path.getsize(db_path) // 2)


def _break_schema(db_path: str) -> None:
    # "malformed database schema (...)": SQLITE_CORRUPT with its own message.
    with open(db_path, "r+b") as f:
        data = f.read()
        f.seek(data.index(b"CREATE TABLE alembic_version"))
        f.write(b"CREATX")


@pytest.fixture(autouse=True)
def boot_state(monkeypatch):
    """Undo what a boot binds: sessions, their engines, the database lock and recovery state."""
    monkeypatch.setattr(db_module, "_recovery_attempted", False)
    monkeypatch.setattr(db_module, "Session", None)
    monkeypatch.setattr(db_module, "WriteSession", None)
    monkeypatch.setattr(db_module, "_db_lock", None)
    yield
    for session_factory in (db_module.Session, db_module.WriteSession):
        if session_factory is not None:
            session_factory.kw["bind"].dispose()
    if db_module._db_lock is not None:
        db_module._db_lock.release(force=True)


@pytest.fixture(autouse=True)
def boot_events(monkeypatch):
    """What a boot did, in order: "startup" for each asset startup that succeeded, then "backup"."""
    events = []
    monkeypatch.setattr(main, "start_daily_backup", lambda: events.append("backup"))
    monkeypatch.setattr(_AssetsOn, "events", events)
    # The real asset startup, minus its effects outside the database.
    monkeypatch.setattr(lifecycle, "start_asset_seeder", lambda: False)
    monkeypatch.setattr(lifecycle, "cleanup_temp_filesystem", lambda: None)
    return events


@pytest.fixture(autouse=True)
def startup_warnings(monkeypatch):
    warnings = []
    monkeypatch.setattr(app.logger, "STARTUP_WARNINGS", warnings)
    return warnings


@pytest.fixture
def default_db(tmp_path, monkeypatch):
    """The default database (no --database-url), in a temporary user directory."""
    monkeypatch.setattr(db_module.args, "database_url", None)
    monkeypatch.setattr("folder_paths.get_user_directory", lambda: str(tmp_path))
    return str(tmp_path / "comfyui.db")


@pytest.fixture
def explicit_db(tmp_path, monkeypatch):
    path = str(tmp_path / "explicit.db")
    monkeypatch.setattr(db_module.args, "database_url", f"sqlite:///{path}")
    return path


class _AssetsOn(AssetsEnabled):
    events: list[str] = []

    def __init__(self):
        super().__init__(cli_args)

    def startup(self):
        super().startup()
        self.events.append("startup")


def _boot():
    main.setup_database(_AssetsOn())


def _quarantined(db_path: str) -> list[str]:
    return sorted(p for p in glob.glob(db_path + ".corrupt-*") if not p.endswith(("-wal", "-shm", "-journal")))


@pytest.mark.parametrize(
    "corrupt", [lambda p: _overwrite_page_of(p, "alembic_version"), _overwrite_header, _truncate, _break_schema]
)
def test_corrupt_database_is_quarantined_and_recreated_at_boot(default_db, startup_warnings, corrupt):
    _make_db(default_db, marker="lost")
    corrupt(default_db)
    corrupt_file = os.stat(default_db).st_ino

    _boot()

    assert _revision(default_db) == _head()
    [quarantined] = _quarantined(default_db)
    assert os.stat(quarantined).st_ino == corrupt_file  # moved aside, not deleted
    with closing(sqlite3.connect(default_db)) as conn:
        assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'marker'").fetchone() is None
    assert any("Database quarantined" in w and "recreated empty" in w for w in startup_warnings)


def test_sound_daily_backup_is_restored(default_db, startup_warnings):
    _make_db(default_db + ".daily-backup", marker="from backup")
    _make_db(default_db)
    _overwrite_page_of(default_db, "alembic_version")

    _boot()

    with closing(sqlite3.connect(default_db)) as conn:
        assert conn.execute("SELECT value FROM marker").fetchone() == ("from backup",)
    assert os.path.exists(default_db + ".daily-backup")
    assert not os.path.exists(default_db + ".restore-tmp")
    assert any("restored from the daily backup" in w for w in startup_warnings)


class _Killed(BaseException):
    pass


def test_kill_while_checking_the_backup_leaves_the_database_to_recover_again(default_db, monkeypatch, startup_warnings):
    _make_db(default_db + ".daily-backup", marker="from backup")
    _make_db(default_db)
    _overwrite_page_of(default_db, "alembic_version")
    corrupt_file = os.stat(default_db).st_ino
    real_check = db_module._passes_integrity_check

    def _killed_during_restore_check(path):
        if path.endswith(".restore-tmp"):
            raise _Killed  # the process dies during the slow copy and check of a big backup
        return real_check(path)

    monkeypatch.setattr(db_module, "_passes_integrity_check", _killed_during_restore_check)
    with pytest.raises(_Killed):
        _boot()
    assert os.stat(default_db).st_ino == corrupt_file and _quarantined(default_db) == []

    monkeypatch.setattr(db_module, "_passes_integrity_check", real_check)
    db_module._db_lock.release(force=True)  # the killed process's lock
    _boot()

    with closing(sqlite3.connect(default_db)) as conn:
        assert conn.execute("SELECT value FROM marker").fetchone() == ("from backup",)
    assert any("restored from the daily backup" in w for w in startup_warnings)


def test_copy_left_by_a_killed_recovery_is_replaced(default_db, startup_warnings):
    # A SIGKILL skips cleanup, so the next recovery finds the old .restore-tmp.
    _make_db(default_db + ".daily-backup", marker="from backup")
    _make_db(default_db)
    _overwrite_page_of(default_db, "alembic_version")
    with open(default_db + ".restore-tmp", "wb") as f:
        f.write(b"half a copy")

    _boot()

    with closing(sqlite3.connect(default_db)) as conn:
        assert conn.execute("SELECT value FROM marker").fetchone() == ("from backup",)
    assert not os.path.exists(default_db + ".restore-tmp")
    assert any("restored from the daily backup" in w for w in startup_warnings)


def _refuse_restore(monkeypatch):  # monkeypatch or a monkeypatch.context()
    # E.g. a virus scanner holding the fresh copy, after the corrupt database has moved aside.
    real_replace = os.replace

    def _restore_refused(src, dst):
        if src.endswith(".restore-tmp"):
            raise PermissionError("[WinError 32] The process cannot access the file")
        return real_replace(src, dst)

    monkeypatch.setattr(db_module.os, "replace", _restore_refused)


@pytest.mark.parametrize("table", ["alembic_version", "assets"])  # found by init, then by asset startup
def test_failed_restore_is_undone_and_recovered_on_the_next_launch(default_db, monkeypatch, startup_warnings, table):
    _make_db(default_db + ".daily-backup", marker="from backup")
    _make_db(default_db)
    _overwrite_page_of(default_db, table)
    corrupt_file = os.stat(default_db).st_ino
    with monkeypatch.context() as held:
        _refuse_restore(held)
        if table == "alembic_version":
            with pytest.raises(SystemExit):  # init can't continue on a corrupt database, as before
                _boot()
        else:
            _boot()  # asset startup logs the corruption and carries on, as before

    assert os.stat(default_db).st_ino == corrupt_file  # moved back
    assert glob.glob(default_db + ".corrupt-*") == [] and not os.path.exists(default_db + ".restore-tmp")

    # The next launch, once the file is no longer held.
    for session_factory in (db_module.Session, db_module.WriteSession):
        if session_factory is not None:
            session_factory.kw["bind"].dispose()
    if db_module._db_lock is not None:
        db_module._db_lock.release(force=True)
    monkeypatch.setattr(db_module, "_recovery_attempted", False)
    monkeypatch.setattr(db_module, "_db_lock", None)
    _boot()

    with closing(sqlite3.connect(default_db)) as conn:
        assert conn.execute("SELECT value FROM marker").fetchone() == ("from backup",)


@pytest.mark.parametrize("table", ["alembic_version", "assets"])
def test_restore_that_cannot_be_undone_fails_the_launch(default_db, monkeypatch, table):
    # Moved aside, not restored, not moved back: running on would mean a new empty database.
    _make_db(default_db + ".daily-backup")
    _make_db(default_db)
    _overwrite_page_of(default_db, table)
    real_replace = os.replace

    def _restore_and_undo_refused(src, dst):
        if src.endswith(".restore-tmp") or ".corrupt-" in src:
            raise PermissionError("[WinError 32] The process cannot access the file")
        return real_replace(src, dst)

    monkeypatch.setattr(db_module.os, "replace", _restore_and_undo_refused)

    with pytest.raises(SystemExit):
        _boot()

    assert not os.path.exists(default_db)
    assert len(_quarantined(default_db)) == 1


@pytest.mark.parametrize("table", ["alembic_version", "assets"])
def test_unreadable_backup_stops_recovery_before_anything_moves(default_db, monkeypatch, caplog, table):
    # E.g. a virus scanner holding the backup: not the same as having none.
    _make_db(default_db + ".daily-backup", marker="from backup")
    _make_db(default_db)
    _overwrite_page_of(default_db, table)
    corrupt_file = os.stat(default_db).st_ino

    def _backup_held(src, dst):
        raise PermissionError("[WinError 32] The process cannot access the file")

    monkeypatch.setattr(db_module.shutil, "copyfile", _backup_held)

    if table == "alembic_version":
        with pytest.raises(SystemExit):  # init can't continue on a corrupt database, as before
            _boot()
    else:
        _boot()  # asset startup logs the corruption and carries on, as before
        assert "Could not recover the corrupt database" in caplog.text

    assert os.stat(default_db).st_ino == corrupt_file
    assert glob.glob(default_db + ".corrupt-*") == []


def test_sidecars_move_before_the_database(default_db, monkeypatch):
    # A kill between moves must never leave an old WAL beside whatever is at the live path.
    _make_db(default_db)
    for suffix in ("-wal", "-shm", "-journal"):
        with open(default_db + suffix, "wb") as f:
            f.write(b"sidecar")
    moved = []
    real_replace = os.replace
    monkeypatch.setattr(db_module.os, "replace", lambda src, dst: moved.append(src[len(default_db):]) or real_replace(src, dst))

    db_module._quarantine_and_restore(default_db, sqlite3.DatabaseError("database disk image is malformed"))

    assert moved == ["-wal", "-shm", "-journal", ""]


def test_database_that_wont_move_keeps_its_wal(default_db, monkeypatch):
    _make_db(default_db)
    with open(default_db + "-wal", "wb") as f:
        f.write(b"recent commits")
    real_replace = os.replace

    def _database_held(src, dst):
        if os.path.abspath(src) == os.path.abspath(default_db):
            raise PermissionError("[WinError 32] The process cannot access the file")
        return real_replace(src, dst)

    monkeypatch.setattr(db_module.os, "replace", _database_held)

    with pytest.raises(sqlite3.DatabaseError):
        db_module._quarantine_and_restore(default_db, sqlite3.DatabaseError("database disk image is malformed"))

    with open(default_db + "-wal", "rb") as f:  # moved aside first, then back
        assert f.read() == b"recent commits"
    assert glob.glob(default_db + ".corrupt-*") == []


def test_failed_undo_is_reported(default_db, monkeypatch, caplog):
    _make_db(default_db + ".daily-backup")
    _make_db(default_db)
    real_replace = os.replace

    def _restore_and_undo_refused(src, dst):
        if src.endswith(".restore-tmp") or ".corrupt-" in src:
            raise PermissionError("[WinError 32] The process cannot access the file")
        return real_replace(src, dst)

    monkeypatch.setattr(db_module.os, "replace", _restore_and_undo_refused)

    with pytest.raises(sqlite3.DatabaseError):
        db_module._quarantine_and_restore(default_db, sqlite3.DatabaseError("database disk image is malformed"))

    [quarantined] = _quarantined(default_db)
    assert f"Could not move '{quarantined}' back to '{default_db}'" in caplog.text


def test_corrupt_daily_backup_is_not_restored(default_db, startup_warnings):
    backup = default_db + ".daily-backup"
    _make_db(backup, marker="from backup")
    _overwrite_page_of(backup, "marker")
    with open(backup, "rb") as f:
        backup_bytes = f.read()
    _make_db(default_db)
    _overwrite_page_of(default_db, "alembic_version")

    _boot()

    assert _revision(default_db) == _head()
    with closing(sqlite3.connect(default_db)) as conn:
        assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'marker'").fetchone() is None
    with open(backup, "rb") as f:
        assert f.read() == backup_bytes
    assert not os.path.exists(default_db + ".restore-tmp")
    assert any("recreated empty" in w for w in startup_warnings)


def test_corruption_found_by_an_upgrade_is_quarantined(default_db, startup_warnings):
    # A database from an older release fails only once the upgrade reads its tables.
    _make_db(default_db, revision="0006_add_loader_path")
    _overwrite_page_of(default_db, "assets")

    _boot()

    assert _revision(default_db) == _head()
    assert len(_quarantined(default_db)) == 1


def test_locked_database_is_not_mistaken_for_corruption(default_db):
    _make_db(default_db)
    database_file = os.stat(default_db).st_ino
    holder = sqlite3.connect(default_db)
    holder.execute("BEGIN EXCLUSIVE")
    try:
        with pytest.raises(SystemExit):
            _boot()
    finally:
        holder.close()
    assert _quarantined(default_db) == []
    assert os.stat(default_db).st_ino == database_file  # the same file, not recreated


def test_explicit_database_url_is_not_quarantined(explicit_db):
    _make_db(explicit_db)
    _overwrite_header(explicit_db)

    with pytest.raises(SystemExit):
        _boot()

    assert _quarantined(explicit_db) == []


@pytest.mark.parametrize("spelling", ["{db}", "{dir}/../{base}/comfyui.db"])
def test_explicit_url_naming_the_default_file_is_recovered(default_db, monkeypatch, spelling):
    # A launcher may pin --database-url to the default file; it is still the default database.
    folder = os.path.dirname(default_db)
    path = spelling.format(db=default_db, dir=folder, base=os.path.basename(folder))
    monkeypatch.setattr(db_module.args, "database_url", f"sqlite:///{path}")
    _make_db(default_db)
    _overwrite_header(default_db)

    _boot()

    assert len(_quarantined(default_db)) == 1
    assert _revision(default_db) == _head()


@pytest.mark.parametrize("url, backed_up", [
    ("sqlite:///{default_db}", True),
    ("sqlite:///{other}", False),
    ("sqlite:///:memory:", False),
    ("sqlite://", False),
])
def test_backup_only_for_the_default_file(default_db, monkeypatch, url, backed_up):
    monkeypatch.setattr(db_module.args, "database_url", url.format(default_db=default_db, other=default_db + ".other"))
    started = []
    monkeypatch.setattr(db_module, "threading", SimpleNamespace(Thread=lambda **kw: started.append(kw) or _NoThread()))

    db_module.start_daily_backup()

    assert bool(started) == backed_up


class _NoThread:
    def start(self):
        pass


@pytest.mark.parametrize("table", ["asset_system_state", "assets", "asset_contents"])
def test_corruption_first_hit_by_asset_startup_is_recovered(default_db, boot_events, monkeypatch, table):
    # init_db reads none of these tables; asset startup is the first to.
    _make_db(default_db + ".daily-backup", marker="from backup")
    _make_db(default_db)
    _overwrite_page_of(default_db, table)
    corrupt_file = os.stat(default_db).st_ino
    open_at_rename = []
    real_replace = os.replace

    def _replace(src, dst):
        if os.path.abspath(src) == os.path.abspath(default_db) and os.path.isdir("/proc/self/fd"):
            open_at_rename.append(os.path.realpath(src) in _open_files())  # Windows refuses to rename an open file
        return real_replace(src, dst)

    monkeypatch.setattr(db_module.os, "replace", _replace)

    _boot()

    [quarantined] = _quarantined(default_db)
    assert os.stat(quarantined).st_ino == corrupt_file  # kept, not deleted
    with closing(sqlite3.connect(default_db)) as conn:
        assert conn.execute("SELECT value FROM marker").fetchone() == ("from backup",)
    assert boot_events == ["startup", "backup"]
    if os.path.isdir("/proc/self/fd"):
        assert open_at_rename == [False]


def test_corruption_again_after_recovery_is_logged_as_before(default_db, boot_events, monkeypatch):
    # Recovery is attempted once per launch; after that, asset startup logs corruption and carries on.
    _make_db(default_db)
    monkeypatch.setattr(db_module, "_passes_integrity_check", lambda path: path.endswith(".restore-tmp"))

    def _wipe(session):
        raise sqlite3.DatabaseError("database disk image is malformed")

    monkeypatch.setattr(lifecycle, "wipe_temp_db_rows", _wipe)

    _boot()

    assert boot_events == ["startup", "backup"]
    assert len(_quarantined(default_db)) == 1


def test_failed_reopen_after_recovery_fails_the_launch(default_db, monkeypatch):
    _make_db(default_db)
    _overwrite_page_of(default_db, "assets")
    db_module.init_db()

    def _reopen_fails(*a):
        raise RuntimeError("the new database would not open")

    monkeypatch.setattr(db_module, "_migrate_and_bind", _reopen_fails)
    with pytest.raises(RuntimeError, match="would not open"):
        db_module.recover_from_corruption(sqlite3.DatabaseError("database disk image is malformed"))


def test_healthy_boot_starts_the_backup_after_asset_startup(default_db, boot_events, monkeypatch):
    _make_db(default_db)
    checked = []
    monkeypatch.setattr(db_module, "_passes_integrity_check", checked.append)

    _boot()

    assert boot_events == ["startup", "backup"]
    assert checked == []  # no integrity scan on the startup path
    assert _quarantined(default_db) == []


def test_assets_off_boot_starts_no_backup(default_db, boot_events):
    _make_db(default_db)

    class _AssetsOff:
        enabled = False

        def startup(self):
            pass

    main.setup_database(_AssetsOff())

    assert boot_events == []


def test_older_backup_is_restored_and_upgraded(default_db):
    _make_db(default_db + ".daily-backup", revision="0006_add_loader_path", marker="old backup")
    _make_db(default_db)
    _overwrite_header(default_db)

    _boot()

    assert _revision(default_db) == _head()
    with closing(sqlite3.connect(default_db)) as conn:
        assert conn.execute("SELECT value FROM marker").fetchone() == ("old backup",)


def test_explicit_database_url_with_corrupt_asset_tables_still_boots(explicit_db, boot_events):
    # Not recovered (not the default database), so asset startup logs it and carries on.
    _make_db(explicit_db)
    _overwrite_page_of(explicit_db, "assets")

    _boot()

    assert boot_events == ["startup", "backup"]
    assert _quarantined(explicit_db) == []


def test_index_damage_quick_check_misses_counts_as_corruption(live_db):
    with closing(sqlite3.connect(live_db)) as conn:
        conn.execute("CREATE INDEX marker_value ON marker (value)")
        conn.commit()
        page = conn.execute("SELECT rootpage FROM sqlite_master WHERE name = 'marker'").fetchone()[0]
    with open(live_db, "r+b") as f:  # change the row, not its index entry
        f.seek((page - 1) * _PAGE)
        data = bytearray(f.read(_PAGE))
        at = data.index(b"live")
        data[at:at + 4] = b"lime"
        f.seek((page - 1) * _PAGE)
        f.write(data)
    with closing(sqlite3.connect(live_db)) as conn:
        assert conn.execute("PRAGMA quick_check").fetchone() == ("ok",)

    assert not db_module._passes_integrity_check(live_db)


def test_recovery_after_init_leaves_a_sound_database(default_db):
    _make_db(default_db)
    db_module.init_db()

    assert not db_module.recover_from_corruption(sqlite3.DatabaseError("database disk image is malformed"))
    assert _quarantined(default_db) == []


def test_corruption_error_on_a_sound_database_is_logged_at_asset_startup(default_db, boot_events, monkeypatch):
    _make_db(default_db)
    seeded = []
    monkeypatch.setattr(lifecycle, "start_asset_seeder", lambda: seeded.append(True))

    def _wipe(session):
        raise sqlite3.DatabaseError("database disk image is malformed")

    monkeypatch.setattr(lifecycle, "wipe_temp_db_rows", _wipe)

    _boot()

    assert boot_events == ["startup", "backup"]
    assert seeded == [True]  # the previous flow: logged, then the seeder starts anyway
    assert _quarantined(default_db) == []


def test_a_live_database_locked_by_another_process_is_not_called_corrupt(default_db):
    # E.g. a damaged pre-upgrade .bkp raised the error while a database tool has the live file locked.
    _make_db(default_db)
    holder = sqlite3.connect(default_db, isolation_level=None)
    holder.execute("BEGIN EXCLUSIVE")
    try:
        assert not db_module.is_recoverable_corruption(sqlite3.DatabaseError("file is not a database"))
    finally:
        holder.close()


def test_recovery_after_init_ignores_other_errors(default_db):
    _make_db(default_db)
    db_module.init_db()
    _overwrite_page_of(default_db, "assets")  # so only the error decides

    assert not db_module.recover_from_corruption(RuntimeError("database disk image is malformed"))
    assert not db_module.recover_from_corruption(sqlite3.OperationalError("database is locked"))
    assert _quarantined(default_db) == []


def test_recovery_after_init_ignores_an_explicit_database_url(explicit_db):
    _make_db(explicit_db)
    db_module.init_db()
    _overwrite_header(explicit_db)

    assert not db_module.recover_from_corruption(sqlite3.DatabaseError("file is not a database"))
    assert _quarantined(explicit_db) == []


@pytest.mark.parametrize("url", ["sqlite://", "sqlite:///:memory:"])
def test_recovery_after_init_ignores_an_in_memory_database(monkeypatch, url):
    monkeypatch.setattr(db_module.args, "database_url", url)

    assert not db_module.recover_from_corruption(sqlite3.DatabaseError("file is not a database"))


def test_schema_corruption_is_recognised_by_its_message():
    # Python 3.10's sqlite3 has no error codes, so only the message identifies it.
    assert error_kind(sqlite3.DatabaseError("malformed database schema (t) - near \"x\"")) == "database_corrupt"


def test_failed_quarantine_raises_the_corruption(default_db, monkeypatch, startup_warnings):
    _make_db(default_db)
    _overwrite_header(default_db)

    def _refuse(src, dst):
        raise PermissionError("in use by another process")

    monkeypatch.setattr(db_module.os, "replace", _refuse)

    with pytest.raises(Exception) as raised:
        db_module.init_db()
    assert error_kind(raised.value) == "database_corrupt"
    assert startup_warnings == []


def test_unmovable_database_found_corrupt_after_init_starts_as_before(default_db, boot_events, monkeypatch):
    # As on Windows when another program has the file open: asset startup logs the corruption and
    # carries on, instead of failing the launch.
    _make_db(default_db)
    _overwrite_page_of(default_db, "assets")
    seeded = []
    monkeypatch.setattr(lifecycle, "start_asset_seeder", lambda: seeded.append(True))
    real_replace = os.replace

    def _held_open(src, dst):
        if os.path.abspath(src) == os.path.abspath(default_db):
            raise PermissionError("[WinError 32] The process cannot access the file")
        return real_replace(src, dst)

    monkeypatch.setattr(db_module.os, "replace", _held_open)
    monkeypatch.setattr(db_module, "_recovery_attempted", False)

    _boot()

    assert boot_events == ["startup", "backup"]
    assert seeded == [True]
    assert _quarantined(default_db) == []


def test_failed_sidecar_move_leaves_the_database_in_place(default_db, monkeypatch):
    _make_db(default_db)
    with open(default_db + "-wal", "wb") as f:
        f.write(b"held by another program")
    database_file = os.stat(default_db).st_ino
    real_replace = os.replace

    def _wal_held(src, dst):
        if src.endswith("-wal"):
            raise PermissionError("[WinError 32] The process cannot access the file")
        return real_replace(src, dst)

    monkeypatch.setattr(db_module.os, "replace", _wal_held)
    error = sqlite3.DatabaseError("database disk image is malformed")

    with pytest.raises(sqlite3.DatabaseError):
        db_module._quarantine_and_restore(default_db, error)

    assert os.stat(default_db).st_ino == database_file
    assert _quarantined(default_db) == []


def test_second_failure_propagates_and_releases_the_lock(default_db, monkeypatch):
    _make_db(default_db)
    _overwrite_header(default_db)
    real_migrate = db_module._migrate_and_bind
    calls = []

    def _fail_after_recovery(*a):
        calls.append(a)
        if len(calls) == 1:
            return real_migrate(*a)
        raise RuntimeError("restored database failed too")

    monkeypatch.setattr(db_module, "_migrate_and_bind", _fail_after_recovery)

    with pytest.raises(RuntimeError, match="restored database failed too"):
        db_module.init_db()
    assert len(calls) == 2
    contender = FileLock(default_db + ".lock")
    contender.acquire(timeout=0)
    contender.release()


def test_damaged_pre_upgrade_backup_does_not_quarantine_a_sound_database(default_db, monkeypatch):
    # The pre-upgrade copy is written into the existing .bkp, whose own damage reads as
    # "file is not a database".
    _make_db(default_db, revision="0006_add_loader_path")
    with open(default_db + ".bkp", "wb") as f:
        f.write(b"\xa5" * 8192)

    checked = []
    real_check = db_module._passes_integrity_check
    monkeypatch.setattr(db_module, "_passes_integrity_check", lambda p: checked.append((p, real_check(p))) or checked[-1][1])

    with pytest.raises(SystemExit):
        _boot()

    assert checked == [(default_db, True)]  # the corruption error was raised, and vetoed
    assert _quarantined(default_db) == []
    assert _revision(default_db) == "0006_add_loader_path"


def test_wal_kept_by_another_connection_does_not_reach_the_restored_database(default_db):
    # Another connection (a database browser, say) stops SQLite checkpointing and deleting the WAL.
    _make_db(default_db + ".daily-backup", marker="from backup")
    _make_db(default_db)
    other = sqlite3.connect(default_db)
    other.execute("PRAGMA journal_mode=WAL")
    other.execute("PRAGMA wal_autocheckpoint=0")
    other.execute("CREATE TABLE marker (value TEXT)")
    other.execute("INSERT INTO marker VALUES ('in the corrupt database wal')")
    other.commit()
    page = other.execute("SELECT rootpage FROM sqlite_master WHERE name = 'alembic_version'").fetchone()[0]
    with open(default_db, "r+b") as f:  # a page the WAL doesn't hold
        f.seek((page - 1) * _PAGE)
        f.write(b"\xa5" * _PAGE)
    try:
        if os.name == "nt":
            # Windows refuses to rename a file another connection has open: launch fails as before.
            with pytest.raises(SystemExit):
                _boot()
            assert _quarantined(default_db) == []
            return
        _boot()
    finally:
        other.close()

    [quarantined] = _quarantined(default_db)
    with closing(sqlite3.connect(default_db)) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert conn.execute("SELECT value FROM marker").fetchall() == [("from backup",)]
    assert os.path.exists(quarantined + "-wal") and os.path.exists(quarantined + "-shm")  # kept together


def test_journal_of_another_connection_moves_with_the_corrupt_database(default_db):
    _make_db(default_db + ".daily-backup", marker="from backup")
    _make_db(default_db)
    other = sqlite3.connect(default_db, isolation_level=None)  # rollback-journal mode
    other.execute("BEGIN IMMEDIATE")
    other.execute("CREATE TABLE marker (value TEXT)")  # its -journal now exists
    page = other.execute("SELECT rootpage FROM sqlite_master WHERE name = 'alembic_version'").fetchone()[0]
    with open(default_db, "r+b") as f:
        f.seek((page - 1) * _PAGE)
        f.write(b"\xa5" * _PAGE)
    assert os.path.exists(default_db + "-journal")
    try:
        if os.name == "nt":
            # Windows refuses to rename a file another connection has open: launch fails as before.
            with pytest.raises(SystemExit):
                _boot()
            return
        _boot()
    finally:
        other.close()

    [quarantined] = _quarantined(default_db)
    assert os.path.exists(quarantined + "-journal")
    with closing(sqlite3.connect(default_db)) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert conn.execute("SELECT value FROM marker").fetchall() == [("from backup",)]


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="needs /proc")
def test_failed_open_leaves_no_handle_on_the_database(default_db):
    # Windows refuses to rename a file this process still has open.
    _make_db(default_db)
    _overwrite_page_of(default_db, "alembic_version")

    with pytest.raises(Exception) as raised:
        db_module._migrate_and_bind(db_module.get_database_url(), default_db, True)

    # `raised` keeps the traceback, and the failed engines it references, alive during the check.
    assert error_kind(raised.value) == "database_corrupt"
    assert os.path.realpath(default_db) not in _open_files()


def _open_files() -> set[str]:
    return {os.path.realpath(f"/proc/self/fd/{fd}") for fd in os.listdir("/proc/self/fd")}


# Daily backup


@pytest.fixture
def live_db(tmp_path):
    path = str(tmp_path / "comfyui.db")
    _make_db(path, marker="live")
    return path


def test_daily_backup_is_a_sound_copy(live_db):
    backup = live_db + ".daily-backup"
    db_module._write_daily_backup(live_db, backup)

    with closing(sqlite3.connect(backup)) as conn:
        assert conn.execute("PRAGMA quick_check").fetchone() == ("ok",)
        assert conn.execute("SELECT value FROM marker").fetchone() == ("live",)
    assert not os.path.exists(backup + ".tmp")


def test_corrupt_database_never_replaces_the_backup(live_db):
    backup = live_db + ".daily-backup"
    with open(backup, "wb") as f:
        f.write(b"previous backup")
    _overwrite_page_of(live_db, "marker")

    db_module._write_daily_backup(live_db, backup)

    with open(backup, "rb") as f:
        assert f.read() == b"previous backup"
    assert not os.path.exists(backup + ".tmp")  # the failed copy is removed


def test_failed_backup_keeps_the_previous_one(live_db, monkeypatch):
    backup = live_db + ".daily-backup"
    with open(backup, "wb") as f:
        f.write(b"previous backup")

    def _fail(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(db_module.os, "replace", _fail)

    db_module._write_daily_backup(live_db, backup)

    with open(backup, "rb") as f:
        assert f.read() == b"previous backup"
    assert not os.path.exists(backup + ".tmp")  # the failed copy is removed


def _end_backup_thread_after_one_pass(monkeypatch):
    """The started backup thread, which ends after one pass instead of sleeping for a day."""
    monkeypatch.setattr(db_module, "time", SimpleNamespace(time=time.time, sleep=lambda s: sys.exit()))
    started = []

    def _thread(**kwargs):
        started.append(threading.Thread(**kwargs))
        return started[-1]

    monkeypatch.setattr(db_module, "threading", SimpleNamespace(Thread=_thread))
    return started


def test_backup_is_written_off_the_startup_thread(default_db, monkeypatch):
    _make_db(default_db)
    _use_wal(default_db)
    started = _end_backup_thread_after_one_pass(monkeypatch)
    release = threading.Event()
    writers = []

    def _blocked_write(db_path, backup_path):
        writers.append((threading.current_thread().name, db_path, backup_path))
        assert release.wait(10)

    monkeypatch.setattr(db_module, "_write_daily_backup", _blocked_write)

    db_module.start_daily_backup()

    [thread] = started
    assert thread.daemon  # never keeps ComfyUI from exiting
    deadline = time.monotonic() + 10
    while not writers and time.monotonic() < deadline:
        time.sleep(0.01)
    assert thread.is_alive()  # start_daily_backup returned while the write is still blocked
    release.set()
    thread.join(10)
    assert writers == [("database-daily-backup", default_db, default_db + ".daily-backup")]


def test_started_backup_writes_a_sound_copy(default_db, monkeypatch):
    _make_db(default_db, marker="live")
    _use_wal(default_db)
    started = _end_backup_thread_after_one_pass(monkeypatch)

    db_module.start_daily_backup()
    started[0].join(30)

    backup = default_db + ".daily-backup"
    with closing(sqlite3.connect(backup)) as conn:
        assert conn.execute("PRAGMA quick_check").fetchone() == ("ok",)
        assert conn.execute("SELECT value FROM marker").fetchone() == ("live",)


def test_no_backup_without_wal(default_db, monkeypatch, caplog):
    # Without WAL, VACUUM INTO's read lock would block every write for the length of the copy.
    _make_db(default_db)  # rollback-journal mode
    written = []
    monkeypatch.setattr(db_module, "_write_daily_backup", lambda *a: written.append(a))
    started = _end_backup_thread_after_one_pass(monkeypatch)

    db_module.start_daily_backup()
    started[0].join(30)

    assert written == []
    assert caplog.text.count("isn't in WAL mode") == 1


class _Stop(Exception):
    pass


@pytest.mark.parametrize("age_hours, first_write_hour", [(None, 0), (25, 0), (1, 23), (-12, 0), (-2400, 0)])
def test_backup_is_refreshed_once_a_day_while_running(tmp_path, monkeypatch, age_hours, first_write_hour):
    db_path = str(tmp_path / "comfyui.db")
    _use_wal(db_path)
    backup = db_path + ".daily-backup"
    now = [1_000_000.0]
    if age_hours is not None:
        open(backup, "w").close()
        os.utime(backup, (now[0] - age_hours * 3600,) * 2)
    writes = []

    def _write(*a):  # a successful write stamps the backup with the fake clock
        writes.append(round(now[0] - 1_000_000.0) // 3600)
        open(backup, "w").close()
        os.utime(backup, (now[0], now[0]))

    def _sleep(seconds):
        assert seconds <= 24 * 3600  # longer waits overflow time.sleep on Windows
        now[0] += seconds
        if len(writes) >= 3:
            raise _Stop

    monkeypatch.setattr(db_module, "time", SimpleNamespace(time=lambda: now[0], sleep=_sleep))
    monkeypatch.setattr(db_module, "_write_daily_backup", _write)

    with pytest.raises(_Stop):
        db_module._keep_daily_backup(db_path, backup)

    assert writes == [first_write_hour, first_write_hour + 24, first_write_hour + 48]


def test_failed_backup_is_retried_a_day_later(tmp_path, monkeypatch):
    db_path = str(tmp_path / "comfyui.db")
    _use_wal(db_path)
    now = [1_000_000.0]
    writes = []

    def _sleep(seconds):
        assert seconds > 0
        now[0] += seconds
        if len(writes) >= 3:
            raise _Stop

    monkeypatch.setattr(db_module, "time", SimpleNamespace(time=lambda: now[0], sleep=_sleep))
    # Every write fails, so the backup never appears.
    monkeypatch.setattr(db_module, "_write_daily_backup", lambda *a: writes.append(round(now[0] - 1_000_000.0) // 3600))

    with pytest.raises(_Stop):
        db_module._keep_daily_backup(db_path, db_path + ".daily-backup")

    assert writes == [0, 24, 48]


def test_no_backup_for_an_explicit_database_url(explicit_db, monkeypatch):
    started = []
    monkeypatch.setattr(db_module, "threading", SimpleNamespace(Thread=lambda **kw: started.append(kw)))

    db_module.start_daily_backup()

    assert started == []


def test_backup_includes_commits_still_in_the_wal(live_db):
    writer = sqlite3.connect(live_db)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute("INSERT INTO marker VALUES ('in the wal')")
    writer.commit()
    try:
        db_module._write_daily_backup(live_db, live_db + ".daily-backup")
    finally:
        writer.close()

    with closing(sqlite3.connect(live_db + ".daily-backup")) as conn:
        assert ("in the wal",) in conn.execute("SELECT value FROM marker").fetchall()


def test_copy_breaking_a_constraint_never_replaces_the_backup(live_db):
    # VACUUM INTO copies such a row; the restore's integrity check would then reject the backup.
    backup = live_db + ".daily-backup"
    with open(backup, "wb") as f:
        f.write(b"previous backup")
    with closing(sqlite3.connect(live_db)) as conn:
        conn.execute("CREATE TABLE sized (n INTEGER CHECK (n >= 0))")
        conn.execute("PRAGMA ignore_check_constraints=ON")  # as bit rot inside a value would
        conn.execute("INSERT INTO sized VALUES (-1)")
        conn.commit()

    db_module._write_daily_backup(live_db, backup)

    with open(backup, "rb") as f:
        assert f.read() == b"previous backup"
    assert not os.path.exists(backup + ".tmp")  # the failed copy is removed


def test_deleted_database_never_replaces_the_backup(tmp_path):
    db_path = str(tmp_path / "comfyui.db")
    backup = db_path + ".daily-backup"
    with open(backup, "wb") as f:
        f.write(b"previous backup")

    db_module._write_daily_backup(db_path, backup)

    with open(backup, "rb") as f:
        assert f.read() == b"previous backup"
    assert not os.path.exists(db_path)  # not recreated empty
    assert not db_module._passes_integrity_check(db_path)
    assert not os.path.exists(db_path)


def test_backup_replaces_the_previous_one(live_db):
    backup = live_db + ".daily-backup"
    db_module._write_daily_backup(live_db, backup)
    with closing(sqlite3.connect(live_db)) as conn:
        conn.execute("UPDATE marker SET value = 'a day later'")
        conn.commit()

    db_module._write_daily_backup(live_db, backup)

    with closing(sqlite3.connect(backup)) as conn:
        assert conn.execute("SELECT value FROM marker").fetchall() == [("a day later",)]
    assert not os.path.exists(backup + ".tmp")


def test_stale_partial_backup_is_removed(live_db):
    backup = live_db + ".daily-backup"
    with open(backup + ".tmp", "wb") as f:
        f.write(b"left by a process that exited mid-backup")

    db_module._write_daily_backup(live_db, backup)

    assert not os.path.exists(backup + ".tmp")
    with closing(sqlite3.connect(backup)) as conn:
        assert conn.execute("SELECT value FROM marker").fetchone() == ("live",)


def test_comfyui_launches_on_a_corrupt_default_database(tmp_path, table="assets"):
    db_path = tmp_path / "user" / "comfyui.db"
    db_path.parent.mkdir()
    _make_db(str(db_path))
    _overwrite_page_of(str(db_path), table)
    repo_root = Path(__file__).resolve().parents[2]

    launch = subprocess.run(
        [
            sys.executable, "main.py", "--cpu", "--enable-assets", "--quick-test-for-ci",
            "--disable-all-custom-nodes", "--disable-partner-nodes",
            f"--base-directory={tmp_path}", f"--front-end-root={tmp_path}",
        ],
        cwd=repo_root, capture_output=True, text=True, timeout=120,
    )

    assert launch.returncode == 0, launch.stderr[-3000:]
    assert "Database quarantined" in launch.stderr
    assert len(_quarantined(str(db_path))) == 1
    assert _revision(str(db_path)) == _head()
