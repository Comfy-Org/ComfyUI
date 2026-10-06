import glob
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import closing
from pathlib import Path

import pytest
import torch
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from filelock import FileLock

import app.logger
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
    """Undo what a boot binds: sessions, their engines and the database lock."""
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


class _AssetsOn:
    enabled = True
    events: list[str] = []

    def startup(self):
        with db_module.create_session() as session:
            session.connection().exec_driver_sql("SELECT * FROM assets").fetchall()
        self.events.append("startup")


def _boot():
    main.setup_database(_AssetsOn())


def _quarantined(db_path: str) -> list[str]:
    return sorted(glob.glob(db_path + ".corrupt-*"))


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
    holder = sqlite3.connect(default_db)
    holder.execute("BEGIN EXCLUSIVE")
    try:
        with pytest.raises(SystemExit):
            _boot()
    finally:
        holder.close()
    assert _quarantined(default_db) == []
    assert _revision(default_db) == _head()


def test_explicit_database_url_is_not_quarantined(explicit_db):
    _make_db(explicit_db)
    _overwrite_header(explicit_db)

    with pytest.raises(SystemExit):
        _boot()

    assert _quarantined(explicit_db) == []


def test_corruption_first_hit_by_asset_startup_is_recovered(default_db, boot_events):
    # init_db doesn't read the assets table; the first query that does is in asset startup.
    _make_db(default_db + ".daily-backup", marker="from backup")
    _make_db(default_db)
    _overwrite_page_of(default_db, "assets")

    _boot()

    [quarantined] = _quarantined(default_db)
    with closing(sqlite3.connect(default_db)) as conn:
        assert conn.execute("SELECT value FROM marker").fetchone() == ("from backup",)
    assert boot_events == ["startup", "backup"]
    if os.path.isdir("/proc/self/fd"):  # Windows refuses to rename a file this process has open
        assert os.path.realpath(quarantined) not in _open_files()


def test_healthy_boot_starts_the_backup_after_asset_startup(default_db, boot_events):
    _make_db(default_db)

    _boot()

    assert boot_events == ["startup", "backup"]
    assert _quarantined(default_db) == []


def test_older_backup_is_restored_and_upgraded(default_db):
    _make_db(default_db + ".daily-backup", revision="0006_add_loader_path", marker="old backup")
    _make_db(default_db)
    _overwrite_header(default_db)

    _boot()

    assert _revision(default_db) == _head()
    with closing(sqlite3.connect(default_db)) as conn:
        assert conn.execute("SELECT value FROM marker").fetchone() == ("old backup",)


def test_recovery_after_init_ignores_other_errors(default_db):
    _make_db(default_db)
    db_module.init_db()

    assert not db_module.recover_from_corruption(RuntimeError("database disk image is malformed"))
    assert not db_module.recover_from_corruption(sqlite3.OperationalError("database is locked"))
    assert _quarantined(default_db) == []


def test_recovery_after_init_ignores_an_explicit_database_url(explicit_db):
    _make_db(explicit_db)
    db_module.init_db()
    _overwrite_header(explicit_db)

    assert not db_module.recover_from_corruption(sqlite3.DatabaseError("file is not a database"))
    assert _quarantined(explicit_db) == []


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


def test_damaged_pre_upgrade_backup_does_not_quarantine_a_sound_database(default_db):
    # The pre-upgrade copy is written into the existing .bkp, whose own damage reads as
    # "file is not a database".
    _make_db(default_db, revision="0006_add_loader_path")
    with open(default_db + ".bkp", "wb") as f:
        f.write(b"\xa5" * 8192)

    with pytest.raises(SystemExit):
        _boot()

    assert _quarantined(default_db) == []
    assert _revision(default_db) == "0006_add_loader_path"


@pytest.mark.parametrize("with_backup", [False, True])
def test_crash_left_wal_does_not_reach_the_new_database(default_db, with_backup):
    if with_backup:
        _make_db(default_db + ".daily-backup")
    _make_db(default_db)
    writer = sqlite3.connect(default_db)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute("CREATE TABLE marker (value TEXT)")
    writer.execute("INSERT INTO marker VALUES ('in the wal')")
    writer.commit()
    page = writer.execute("SELECT rootpage FROM sqlite_master WHERE name = 'alembic_version'").fetchone()[0]
    crashed = default_db + ".crashed"
    shutil.copyfile(default_db, crashed)
    shutil.copyfile(default_db + "-wal", crashed + "-wal")  # as a crash leaves them
    writer.close()
    with open(crashed, "r+b") as f:  # a page the WAL doesn't hold
        f.seek((page - 1) * _PAGE)
        f.write(b"\xa5" * _PAGE)
    os.replace(crashed, default_db)
    os.replace(crashed + "-wal", default_db + "-wal")

    _boot()

    assert len(_quarantined(default_db)) == 1
    assert _revision(default_db) == _head()
    with closing(sqlite3.connect(default_db)) as conn:
        assert conn.execute("PRAGMA quick_check").fetchone() == ("ok",)
        assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'marker'").fetchone() is None


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="needs /proc")
def test_failed_open_leaves_no_handle_on_the_database(default_db):
    # Windows refuses to rename a file this process still has open.
    _make_db(default_db)
    _overwrite_page_of(default_db, "alembic_version")

    with pytest.raises(Exception) as raised:
        db_module._migrate_and_bind(db_module.get_database_url(), default_db, True)

    assert raised.value is not None  # its traceback still references the failed engines
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
    assert not os.path.exists(backup + ".tmp")


def test_failed_backup_removes_its_partial_file(live_db, monkeypatch):
    backup = live_db + ".daily-backup"

    def _fail(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(db_module.os, "replace", _fail)

    db_module._write_daily_backup(live_db, backup)

    assert not os.path.exists(backup)
    assert not os.path.exists(backup + ".tmp")


def test_backup_is_written_off_the_startup_thread(default_db, monkeypatch):
    _make_db(default_db)
    release = threading.Event()
    writers = []

    def _blocked_write(db_path, backup_path):
        writers.append((threading.current_thread().name, db_path, backup_path))
        release.wait(10)

    monkeypatch.setattr(db_module, "_write_daily_backup", _blocked_write)

    db_module.start_daily_backup()  # returns while the write is still blocked

    deadline = time.monotonic() + 10
    while not writers and time.monotonic() < deadline:
        time.sleep(0.01)
    release.set()
    assert writers == [("database-daily-backup", default_db, default_db + ".daily-backup")]


def test_started_backup_writes_a_sound_copy(default_db):
    _make_db(default_db, marker="live")

    db_module.start_daily_backup()

    backup = default_db + ".daily-backup"
    deadline = time.monotonic() + 30
    while not os.path.exists(backup) and time.monotonic() < deadline:
        time.sleep(0.05)
    with closing(sqlite3.connect(backup)) as conn:
        assert conn.execute("PRAGMA quick_check").fetchone() == ("ok",)
        assert conn.execute("SELECT value FROM marker").fetchone() == ("live",)


class _Stop(Exception):
    pass


@pytest.mark.parametrize("age_hours, writes_first", [(None, True), (25, True), (1, False)])
def test_backup_is_refreshed_once_a_day_while_running(tmp_path, monkeypatch, age_hours, writes_first):
    db_path = str(tmp_path / "comfyui.db")
    backup = db_path + ".daily-backup"
    if age_hours is not None:
        open(backup, "w").close()
        then = time.time() - age_hours * 3600
        os.utime(backup, (then, then))
    events = []

    def _sleep(seconds):
        events.append(("sleep", round(seconds / 3600)))
        if len(events) >= 4:
            raise _Stop

    monkeypatch.setattr(db_module.time, "sleep", _sleep)
    monkeypatch.setattr(db_module, "_write_daily_backup", lambda *a: events.append(("write", a)))

    with pytest.raises(_Stop):
        db_module._keep_daily_backup(db_path, backup)

    if writes_first:
        assert events[:2] == [("write", (db_path, backup)), ("sleep", 24)]
    else:
        assert events[0] == ("sleep", 23)


def test_no_backup_for_an_explicit_database_url(explicit_db, monkeypatch):
    started = []
    monkeypatch.setattr(db_module.threading, "Thread", lambda **kw: started.append(kw))

    db_module.start_daily_backup()

    assert started == []


def test_stale_partial_backup_is_removed(live_db):
    backup = live_db + ".daily-backup"
    with open(backup + ".tmp", "wb") as f:
        f.write(b"left by a process that exited mid-backup")

    db_module._write_daily_backup(live_db, backup)

    assert not os.path.exists(backup + ".tmp")
    with closing(sqlite3.connect(backup)) as conn:
        assert conn.execute("SELECT value FROM marker").fetchone() == ("live",)


@pytest.mark.parametrize("table", ["alembic_version", "asset_system_state"])
def test_comfyui_launches_on_a_corrupt_default_database(tmp_path, table):
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
