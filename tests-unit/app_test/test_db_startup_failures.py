"""Each way the asset database can fail to open stops startup with a message that names
the database and the fix. Every case drives the real init_db against a real file."""

import errno
import logging
import os
import sqlite3
import sys
import types
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
    monkeypatch.setattr(db_module, "Session", None)
    monkeypatch.setattr(db_module, "WriteSession", None)
    monkeypatch.setattr(db_module, "_db_lock", None)
    monkeypatch.setattr(db_module, "_LOCK_WAIT_SECONDS", 0.1)
    yield path
    if db_module._db_lock is not None:
        db_module._db_lock.release(force=True)


def _startup_error(caplog, asset_manager=None):
    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as stopped:
        main.setup_database(asset_manager or _AssetsOn())
    assert stopped.value.code == 1
    assert "--disable-assets" in caplog.text
    return caplog.text


def _stamp(path, revision):
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        conn.execute("INSERT INTO alembic_version VALUES (?)", (revision,))


def test_lock_held_by_another_comfyui(db_path, caplog):
    holder = FileLock(db_path + ".lock")
    holder.acquire(timeout=0)
    try:
        error = _startup_error(caplog)
    finally:
        holder.release()

    assert f"Another ComfyUI is already using the asset database '{db_path}'." in error
    assert "Close the other ComfyUI" in error
    assert "to run both, give this one its own database" in error


def test_database_locked_by_another_program(db_path, caplog):
    sqlite3.connect(db_path).close()
    other = sqlite3.connect(db_path, isolation_level=None)
    other.execute("BEGIN EXCLUSIVE")
    try:
        error = _startup_error(caplog)
    finally:
        other.close()

    assert f"The asset database '{db_path}' is locked by another program (database is locked)." in error
    assert "Close any program that has it open" in error


def test_corrupt_database(db_path, caplog):
    with open(db_path, "wb") as f:
        f.write(b"not a database" * 1000)

    error = _startup_error(caplog)

    assert f"The asset database '{db_path}' is corrupt (file is not a database)." in error
    assert "Move that file aside, or delete it, and start again" in error


def test_database_from_a_newer_comfyui(db_path, caplog):
    _stamp(db_path, "0099_from_a_newer_release")

    error = _startup_error(caplog)

    assert f"The asset database '{db_path}' was last used by a newer version of ComfyUI" in error
    assert "0099_from_a_newer_release" in error
    assert "Update ComfyUI" in error


def test_failed_upgrade(db_path, caplog):
    _stamp(db_path, "0006_add_loader_path")  # but none of 0006's tables, so the next migration fails

    error = _startup_error(caplog)

    assert f"Could not open or upgrade the asset database '{db_path}': no such table" in error
    assert "If the database is damaged, move that file aside and start again" in error
    assert "delete it" not in error


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions enforced")
def test_folder_not_writable(tmp_path, db_path, caplog):
    os.chmod(tmp_path, 0o555)
    try:
        error = _startup_error(caplog)
    finally:
        os.chmod(tmp_path, 0o755)

    assert f"ComfyUI can't create, open or write the asset database '{db_path}' ([Errno 13] Permission denied" in error
    assert "Make sure its folder is a writable directory" in error


def test_database_path_is_a_directory(db_path, caplog):
    os.mkdir(db_path)

    error = _startup_error(caplog)

    assert f"ComfyUI can't create, open or write the asset database '{db_path}'" in error
    assert "the database path is a writable file" in error
    assert "delete it" not in error


def test_folder_path_runs_through_a_file(tmp_path, monkeypatch, db_path, caplog):
    (tmp_path / "taken").write_text("")
    path = str(tmp_path / "taken" / "sub" / "comfyui.db")
    monkeypatch.setattr(db_module.args, "database_url", f"sqlite:///{path}")

    error = _startup_error(caplog)

    assert f"ComfyUI can't create, open or write the asset database '{path}'" in error
    assert "delete it" not in error


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions enforced")
def test_read_only_database_file_that_needs_an_upgrade(db_path, caplog):
    config = db_module.get_alembic_config()
    command.upgrade(config, "0006_add_loader_path")
    os.chmod(db_path, 0o444)

    error = _startup_error(caplog)

    assert f"ComfyUI can't create, open or write the asset database '{db_path}' (attempt to write a readonly database)" in error
    assert "delete it" not in error


def _start_and_stop(db_path):
    """Open the database the way a previous run would, then let it go."""
    main.setup_database(_AssetsOn())
    db_module._db_lock.release(force=True)
    db_module._db_lock = None


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions enforced")
def test_read_only_database_file_at_the_current_revision(db_path, caplog):
    _start_and_stop(db_path)
    os.chmod(db_path, 0o444)

    error = _startup_error(caplog)

    assert f"ComfyUI can't create, open or write the asset database '{db_path}' (attempt to write a readonly database)" in error
    assert "delete it" not in error


def test_new_database_passes_the_write_check(db_path):
    main.setup_database(_AssetsOn())

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT version_num FROM alembic_version").fetchone()[0] == _head()


def test_another_reader_holding_the_database_open_doesnt_stop_startup(db_path):
    _start_and_stop(db_path)
    reader = sqlite3.connect(db_path, isolation_level=None)
    reader.execute("BEGIN")
    reader.execute("SELECT count(*) FROM alembic_version").fetchone()
    try:
        main.setup_database(_AssetsOn())
    finally:
        reader.close()


def test_file_held_open_by_another_process_on_windows(monkeypatch, db_path, caplog):
    def _sharing_violation(path):
        error = PermissionError(errno.EACCES, "The process cannot access the file", path)
        error.winerror = 32  # ERROR_SHARING_VIOLATION
        raise error

    monkeypatch.setattr(db_module, "prepare_file_db_path", _sharing_violation)

    error = _startup_error(caplog)

    assert f"The asset database '{db_path}' is locked by another program" in error


def test_failure_after_the_database_opened_doesnt_suggest_deleting_it(db_path, caplog):
    class _StartupFails(_AssetsOn):
        def startup(self):
            raise RuntimeError("hash mode state unreadable")

    error = _startup_error(caplog, _StartupFails())

    assert f"Could not open or upgrade the asset database '{db_path}': hash mode state unreadable" in error
    assert "delete it" not in error


def test_database_url_that_is_not_sqlite(monkeypatch, db_path, caplog):
    monkeypatch.setattr(db_module.args, "database_url", "postgresql://user:secret@localhost/comfy")

    error = _startup_error(caplog)

    assert "--database-url must start with sqlite:///, like sqlite:///path/to/comfyui.db." in error
    assert "secret" not in error


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


def _sqlalchemy_quotes_question_marks():
    return make_url(URL.create("sqlite", database="/a?/b.db").render_as_string()).database == "/a?/b.db"


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


def _head():
    return ScriptDirectory(str(Path(main.__file__).parent / "alembic_db")).get_current_head()


class _UnquotedURL:
    """SQLAlchemy 2.0's URL rendering: the database path goes in as is."""

    @staticmethod
    def create(drivername, database):
        return types.SimpleNamespace(render_as_string=lambda: f"{drivername}:///{database}")


@pytest.mark.skipif(sys.platform == "win32", reason="? isn't allowed in Windows paths")
def test_question_mark_in_the_default_path_stops_startup_on_older_sqlalchemy(tmp_path, monkeypatch, db_path, caplog):
    user_dir = tmp_path / "what?"
    monkeypatch.setattr(db_module.args, "database_url", None)
    monkeypatch.setattr(db_module, "get_legacy_default_db_path", lambda: None)
    monkeypatch.setattr(folder_paths, "get_user_directory", lambda: str(user_dir))
    monkeypatch.setattr(db_module, "URL", _UnquotedURL)

    error = _startup_error(caplog)

    assert f"The asset database path '{user_dir / 'comfyui.db'}' contains a '?'" in error
    assert "upgrade SQLAlchemy to 2.1 or newer" in error
    assert not (tmp_path / "what").exists()

