"""Each way the asset database can fail to open stops startup with a message that names
the database and the fix. Every case drives the real init_db against a real file."""

import errno
from contextlib import closing
import logging
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from alembic import command
from alembic.script import ScriptDirectory
from sqlalchemy.engine import URL, make_url
from filelock import FileLock

import folder_paths
from app.database import db as db_module
from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import main  # noqa: E402


class _AssetsOn:
    enabled = True

    def startup(self):
        pass


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = str(tmp_path / "comfyui.db")
    monkeypatch.setattr(db_module.args, "database_url", f"sqlite:///{path}")
    monkeypatch.setattr(db_module.args, "disable_assets", False)
    monkeypatch.setattr(db_module, "Session", None)
    monkeypatch.setattr(db_module, "WriteSession", None)
    monkeypatch.setattr(db_module, "_db_lock", None)
    monkeypatch.setattr(db_module, "_LOCK_WAIT_SECONDS", 0.1)
    yield path
    if db_module._db_lock is not None:
        db_module._db_lock.release(force=True)


def _startup_error(caplog, asset_manager=None, *, kind, level=logging.ERROR):
    with caplog.at_level(level), pytest.raises(SystemExit) as stopped:
        main.skip_assets_for_a_newer_database()  # as at startup: not newer, so assets stay on
        assert not main.args.disable_assets
        main.setup_database(asset_manager or _AssetsOn())
    assert stopped.value.code == 1
    assert "--disable-assets" in caplog.text
    assert f"ASSETS_STARTUP_FAILED: {kind}\n" in caplog.text
    return caplog.text


def _sqlalchemy_quotes_question_marks():
    return make_url(URL.create("sqlite", database="/a?/b.db").render_as_string()).database == "/a?/b.db"


def _stamp(path, revision, journal_mode="delete"):
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(f"PRAGMA journal_mode={journal_mode}")
        conn.execute("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        conn.execute("INSERT INTO alembic_version VALUES (?)", (revision,))
        conn.commit()


def test_lock_held_by_another_comfyui(db_path, caplog):
    holder = FileLock(db_path + ".lock")
    holder.acquire(timeout=0)
    try:
        error = _startup_error(caplog, kind="in_use")
    finally:
        holder.release()

    # Launchers match "already using this database"; keep that phrase.
    assert f"Another ComfyUI is already using this database: '{db_path}'." in error
    assert "Close the other ComfyUI" in error
    assert "Or give this ComfyUI its own database" not in error  # it already has --database-url


def test_database_locked_by_another_program(db_path, caplog):
    sqlite3.connect(db_path).close()
    other = sqlite3.connect(db_path, isolation_level=None)
    other.execute("BEGIN EXCLUSIVE")
    try:
        error = _startup_error(caplog, kind="locked")
    finally:
        other.close()

    assert f"The asset database '{db_path}' is locked by another program (database is locked)." in error
    assert "Close any program that has it open" in error


def test_corrupt_database(db_path, caplog):
    with open(db_path, "wb") as f:
        f.write(b"not a database" * 1000)

    error = _startup_error(caplog, kind="corrupt")

    assert f"The asset database '{db_path}' is corrupt (file is not a database)." in error
    assert "Move that file aside, or delete it, and start again" in error


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions enforced")
def test_corrupt_read_only_database_is_reported_as_corrupt(db_path, caplog):
    with open(db_path, "wb") as f:
        f.write(b"not a database" * 1000)
    os.chmod(db_path, 0o444)

    error = _startup_error(caplog, kind="corrupt")

    assert f"The asset database '{db_path}' is corrupt" in error


def _stamp_left_by_killed_writer(path, revision):
    """Committed but never checkpointed: the stamp lives only in the -wal a killed newer ComfyUI left."""
    script = (
        "import os, sqlite3, sys; conn = sqlite3.connect(sys.argv[1]); "
        "conn.execute('PRAGMA journal_mode=wal'); conn.execute('PRAGMA wal_autocheckpoint=0'); "
        "conn.execute('CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)'); "
        "conn.execute('INSERT INTO alembic_version VALUES (?)', (sys.argv[2],)); conn.commit(); "
        "os.kill(os.getpid(), 9)"
    )
    subprocess.run([sys.executable, "-c", script, path, revision])
    assert os.path.getsize(path + "-wal") > 0


def _files(directory):
    return {name: Path(directory, name).read_bytes() for name in os.listdir(directory) if not name.endswith(".lock")}


@pytest.mark.parametrize("left_by", ["rollback_journal", "wal", "killed_wal_writer"])
def test_database_from_a_newer_comfyui_turns_assets_off_and_is_left_as_it_is(db_path, caplog, left_by):
    if left_by == "killed_wal_writer":
        _stamp_left_by_killed_writer(db_path, "0099_from_a_newer_release")
    else:
        _stamp(db_path, "0099_from_a_newer_release", "wal" if left_by == "wal" else "delete")
    before = _files(os.path.dirname(db_path))

    with caplog.at_level(logging.WARNING):
        main.skip_assets_for_a_newer_database()

    assert main.args.disable_assets
    assert "ASSETS_DISABLED: newer_revision\n" in caplog.text
    assert "ASSETS_STARTUP_FAILED" not in caplog.text
    assert f"The asset database '{db_path}' was upgraded by a newer version of ComfyUI (revision '0099_from_a_newer_release')" in caplog.text
    assert "run the newer version again" in caplog.text
    assert "move or rename that database file" in caplog.text
    after = _files(os.path.dirname(db_path))
    # The database and any WAL it had are untouched (not checkpointed, not switched to WAL, no backup).
    # A read can only add the -shm index, or an empty -wal next to a WAL database that had none.
    for name in ("comfyui.db", "comfyui.db-wal"):
        assert after.get(name) == before.get(name) or (name not in before and after[name] == b"")
    assert set(after) <= {"comfyui.db", "comfyui.db-wal", "comfyui.db-shm"}
    assert not os.path.exists(db_path + ".lock")  # only init_db takes the lock, and it never ran
    assert db_module.Session is None


@pytest.mark.parametrize(
    "folder",
    [
        "100%20x",
        pytest.param(
            "a?b#c",
            marks=pytest.mark.skipif(
                sys.platform == "win32" or not _sqlalchemy_quotes_question_marks(),
                reason="? isn't allowed in Windows paths, and SQLAlchemy before 2.1 can't put one in a URL",
            ),
        ),
    ],
)
def test_newer_database_in_a_folder_with_uri_characters(tmp_path, monkeypatch, db_path, caplog, folder):
    path = tmp_path / folder / "comfyui.db"
    path.parent.mkdir()
    _stamp(str(path), "0099_from_a_newer_release")
    monkeypatch.setattr(db_module.args, "database_url", URL.create("sqlite", database=str(path)).render_as_string())

    with caplog.at_level(logging.WARNING):
        main.skip_assets_for_a_newer_database()

    assert main.args.disable_assets
    assert f"The asset database '{path}' was upgraded by a newer version of ComfyUI" in caplog.text


def test_newer_database_the_check_cannot_read_still_stops_as_newer_revision(db_path, caplog, monkeypatch):
    _stamp(db_path, "0099_from_a_newer_release")
    monkeypatch.setattr(db_module, "_unknown_revisions", lambda path: None)  # e.g. a hot rollback journal

    error = _startup_error(caplog, kind="newer_revision")

    assert "ASSETS_DISABLED" not in error

    assert f"The asset database '{db_path}' was last used by a newer version of ComfyUI" in error
    assert "0099_from_a_newer_release" in error
    assert "Update ComfyUI" in error


def test_disable_assets_never_reads_the_database(db_path, monkeypatch, caplog):
    _stamp(db_path, "0099_from_a_newer_release")
    monkeypatch.setattr(main.args, "disable_assets", True)
    monkeypatch.setattr(main, "newer_database", lambda: pytest.fail("checked the database with assets off"))

    with caplog.at_level(logging.WARNING):
        main.skip_assets_for_a_newer_database()

    assert "ASSETS_DISABLED" not in caplog.text


def test_newer_database_at_the_legacy_path_stays_where_the_newer_comfyui_left_it(tmp_path, monkeypatch, db_path, caplog):
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    legacy = tmp_path / "install" / "user" / "comfyui.db"
    legacy.parent.mkdir(parents=True)
    _stamp(str(legacy), "0099_from_a_newer_release", "wal")
    before = legacy.read_bytes()
    monkeypatch.setattr(db_module.args, "database_url", None)
    monkeypatch.setattr(db_module, "get_legacy_default_db_path", lambda: str(legacy))
    monkeypatch.setattr(folder_paths, "get_user_directory", lambda: str(user_dir))

    with caplog.at_level(logging.WARNING):
        main.skip_assets_for_a_newer_database()

    assert main.args.disable_assets
    assert f"The asset database '{legacy}' was upgraded by a newer version of ComfyUI" in caplog.text
    assert legacy.read_bytes() == before
    assert not (legacy.parent / "comfyui.db.bak").exists()
    assert not (user_dir / "comfyui.db").exists()


def test_database_at_this_versions_head_keeps_assets_on(db_path):
    _stamp(db_path, ScriptDirectory.from_config(db_module.get_alembic_config()).get_current_head())

    main.skip_assets_for_a_newer_database()

    assert not main.args.disable_assets


@pytest.mark.parametrize("beside_legacy", ["user_database", "legacy_backup"])
def test_newer_legacy_database_that_will_not_be_moved_keeps_assets_on(tmp_path, monkeypatch, db_path, beside_legacy):
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    legacy = tmp_path / "install" / "user" / "comfyui.db"
    legacy.parent.mkdir(parents=True)
    _stamp(str(legacy), "0099_from_a_newer_release")
    if beside_legacy == "user_database":
        _stamp(str(user_dir / "comfyui.db"), ScriptDirectory.from_config(db_module.get_alembic_config()).get_current_head())
    else:
        legacy.with_name("comfyui.db.bak").write_bytes(b"")
    monkeypatch.setattr(db_module.args, "database_url", None)
    monkeypatch.setattr(db_module, "get_legacy_default_db_path", lambda: str(legacy))
    monkeypatch.setattr(folder_paths, "get_user_directory", lambda: str(user_dir))

    main.skip_assets_for_a_newer_database()

    assert not main.args.disable_assets


def test_newer_legacy_database_is_not_checked_with_an_explicit_database_url(tmp_path, monkeypatch, db_path):
    legacy = tmp_path / "install" / "user" / "comfyui.db"
    legacy.parent.mkdir(parents=True)
    _stamp(str(legacy), "0099_from_a_newer_release")
    monkeypatch.setattr(db_module, "get_legacy_default_db_path", lambda: str(legacy))

    main.skip_assets_for_a_newer_database()

    assert not main.args.disable_assets


def test_any_unknown_revision_among_several_turns_assets_off(db_path, caplog):
    head = ScriptDirectory.from_config(db_module.get_alembic_config()).get_current_head()
    _stamp(db_path, head)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("INSERT INTO alembic_version VALUES ('0099_from_a_newer_release')")
        conn.commit()

    with caplog.at_level(logging.WARNING):
        main.skip_assets_for_a_newer_database()

    assert main.args.disable_assets
    assert "(revision '0099_from_a_newer_release')" in caplog.text
    assert head not in caplog.text


def test_revision_cannot_add_lines_to_the_log(db_path, caplog):
    _stamp(db_path, "0099\n[ERROR] ASSETS_STARTUP_FAILED: corrupt \u00e9")

    with caplog.at_level(logging.WARNING):
        main.skip_assets_for_a_newer_database()

    assert main.args.disable_assets
    assert "\n[ERROR] ASSETS_STARTUP_FAILED" not in caplog.text
    # Escaped to ASCII, so it also survives a console that can't encode it.
    assert "(revision '0099\\n[ERROR] ASSETS_STARTUP_FAILED: corrupt \\xe9')" in caplog.text


def test_failed_upgrade(db_path, caplog):
    _stamp(db_path, "0006_add_loader_path")  # but none of 0006's tables, so the next migration fails

    error = _startup_error(caplog, kind="other")

    assert f"Could not open or upgrade the asset database '{db_path}': no such table" in error
    assert "If the database is damaged, move that file aside and start again" in error
    assert "delete it" not in error
    assert "Error upgrading database" not in error  # its traceback is at DEBUG
    assert "Run with --verbose DEBUG for the full error." in error


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions enforced")
def test_folder_not_writable(tmp_path, db_path, caplog):
    os.chmod(tmp_path, 0o555)
    try:
        error = _startup_error(caplog, kind="not_writable")
    finally:
        os.chmod(tmp_path, 0o755)

    assert f"ComfyUI can't create, open or write the asset database '{db_path}' ([Errno 13] Permission denied" in error
    assert "Make sure its folder is a writable directory" in error


def test_database_path_is_a_directory(db_path, caplog):
    os.mkdir(db_path)

    error = _startup_error(caplog, kind="not_writable")

    assert f"ComfyUI can't create, open or write the asset database '{db_path}'" in error
    assert "the database path is a writable file" in error
    assert "no other program has it open" in error  # SQLite says the same for a file another program holds
    assert "delete it" not in error


@pytest.mark.parametrize(
    "parts",
    [
        ("taken", "comfyui.db"),
        pytest.param(("taken", "sub", "comfyui.db"), marks=pytest.mark.skipif(sys.platform == "win32", reason="Windows reports this as a missing path")),
    ],
)
def test_folder_path_runs_through_a_file(tmp_path, monkeypatch, db_path, caplog, parts):
    (tmp_path / "taken").write_text("")
    path = str(tmp_path.joinpath(*parts))
    monkeypatch.setattr(db_module.args, "database_url", f"sqlite:///{path}")

    error = _startup_error(caplog, kind="path_blocked")

    assert f"A file is in the way of the folder for the asset database '{path}'" in error
    assert "delete it" not in error


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions enforced")
def test_read_only_database_file_that_needs_an_upgrade(db_path, caplog):
    config = db_module.get_alembic_config()
    command.upgrade(config, "0006_add_loader_path")
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")  # as ComfyUI leaves it
    os.chmod(db_path, 0o444)

    error = _startup_error(caplog, kind="not_writable")

    assert f"ComfyUI can't create, open or write the asset database '{db_path}' (attempt to write a readonly database)" in error
    assert "delete it" not in error


def test_file_held_open_by_another_process_on_windows(monkeypatch, db_path, caplog):
    def _sharing_violation(path):
        error = PermissionError(errno.EACCES, "The process cannot access the file", path)
        error.winerror = 32  # ERROR_SHARING_VIOLATION
        raise error

    monkeypatch.setattr(db_module, "prepare_file_db_path", _sharing_violation)

    error = _startup_error(caplog, kind="locked")

    assert f"The asset database '{db_path}' is locked by another program" in error


def test_failure_after_the_database_opened_doesnt_suggest_deleting_it(db_path, caplog):
    class _StartupFails(_AssetsOn):
        def startup(self):
            raise RuntimeError("hash mode state unreadable")

    error = _startup_error(caplog, _StartupFails(), kind="other", level=logging.DEBUG)

    assert "Asset database startup failed" in error  # the --verbose DEBUG detail the message points to

    assert f"Could not open or upgrade the asset database '{db_path}': hash mode state unreadable" in error
    assert "delete it" not in error


@pytest.mark.parametrize(
    "url", ["postgresql://user:secret@localhost/comfy", "sqlite:relative.db", "sqlite://user:secret@host/x.db"]
)
def test_database_url_that_is_not_sqlite_file_url(monkeypatch, db_path, caplog, url):
    monkeypatch.setattr(db_module.args, "database_url", url)

    error = _startup_error(caplog, kind="unsupported_url", level=logging.DEBUG)

    assert "--database-url must start with sqlite:///, like sqlite:///path/to/comfyui.db" in error
    assert "or be left out to use the default database" in error
    assert url not in error
    assert "secret" not in caplog.text


def test_default_database_in_use_suggests_its_own_database(tmp_path, monkeypatch, db_path, caplog):
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    monkeypatch.setattr(db_module.args, "database_url", None)
    monkeypatch.setattr(db_module, "get_legacy_default_db_path", lambda: None)
    monkeypatch.setattr(folder_paths, "get_user_directory", lambda: str(user_dir))
    holder = FileLock(str(user_dir / "comfyui.db.lock"))
    holder.acquire(timeout=0)
    try:
        error = _startup_error(caplog, kind="in_use")
    finally:
        holder.release()

    assert "Or give this ComfyUI its own database: --database-url sqlite:///path/to/another.db" in error


def test_no_second_database_suggested_for_a_broken_one(tmp_path, monkeypatch, db_path, caplog):
    # A fresh file would quietly start a second, empty catalog instead of fixing this one.
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    (user_dir / "comfyui.db").write_bytes(b"not a database" * 1000)
    monkeypatch.setattr(db_module.args, "database_url", None)
    monkeypatch.setattr(db_module, "get_legacy_default_db_path", lambda: None)
    monkeypatch.setattr(folder_paths, "get_user_directory", lambda: str(user_dir))

    error = _startup_error(caplog, kind="corrupt")

    assert "comfyui-2.db" not in error


def test_opens_from_an_install_folder_with_a_percent_sign(tmp_path, monkeypatch, db_path):
    # Alembic config values go through ConfigParser interpolation, where % is special.
    root = tmp_path / "100% tmp"
    try:
        os.symlink(Path(main.__file__).parent, root, target_is_directory=True)
    except OSError:
        pytest.skip("can't create a directory symlink here")
    monkeypatch.setattr(db_module, "__file__", str(root / "app" / "database" / "db.py"))

    db_module.init_db()

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT version_num FROM alembic_version").fetchone()[0] == _head()


@pytest.mark.parametrize(
    "folder",
    [
        "100%20x",
        pytest.param(
            "what?",
            marks=pytest.mark.skipif(
                sys.platform == "win32" or not _sqlalchemy_quotes_question_marks(),
                reason="? isn't allowed in Windows paths, and SQLAlchemy before 2.1 can't put one in a URL",
            ),
        ),
    ],
)
def test_default_database_path_is_used_literally(tmp_path, monkeypatch, db_path, folder):
    # SQLAlchemy 2.1 decodes %xx in a URL and ends the path at ?, so the path must be quoted.
    user_dir = tmp_path / folder
    monkeypatch.setattr(db_module.args, "database_url", None)
    monkeypatch.setattr(db_module, "get_legacy_default_db_path", lambda: None)
    monkeypatch.setattr(folder_paths, "get_user_directory", lambda: str(user_dir))

    db_module.init_db()

    assert db_module.get_db_path() == str(user_dir / "comfyui.db")
    with sqlite3.connect(user_dir / "comfyui.db") as conn:
        assert conn.execute("SELECT version_num FROM alembic_version").fetchone()[0] == _head()


def _revision(path):
    with sqlite3.connect(path) as conn:
        return conn.execute("SELECT version_num FROM alembic_version").fetchone()[0]


def _head():
    return ScriptDirectory(str(Path(main.__file__).parent / "alembic_db")).get_current_head()
