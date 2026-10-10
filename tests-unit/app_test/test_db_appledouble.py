import os
import shutil
import sqlite3
import stat
import sys

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

from app.database import db as db_module

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# What the Finder writes beside 0001_assets.py on a non-HFS volume: an AppleDouble
# header, which contains NUL bytes and so cannot be compiled as Python.
_APPLEDOUBLE = b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X        " + b"\x00" * 64


def _config(scripts_path: str, db_path: str) -> Config:
    cfg = Config(os.path.join(_REPO_ROOT, "alembic.ini"))
    cfg.set_main_option("script_location", scripts_path)
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return cfg


def _head(scripts_path: str) -> str:
    return ScriptDirectory.from_config(_config(scripts_path, "unused.db")).get_current_head()


def _current_revision(db_path: str) -> str:
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT version_num FROM alembic_version").fetchone()[0]


@pytest.fixture
def scripts(tmp_path, monkeypatch):
    """A copy of alembic_db that get_alembic_config() points at, and a database path."""
    scripts_path = str(tmp_path / "alembic_db")
    shutil.copytree(
        os.path.join(_REPO_ROOT, "alembic_db"),
        scripts_path,
        # A checkout on an exFAT volume has its own ._ files; start from a clean copy.
        ignore=shutil.ignore_patterns("__pycache__", "._*"),
    )
    db_path = str(tmp_path / "comfyui.db")
    monkeypatch.setattr(db_module.args, "database_url", f"sqlite:///{db_path}")
    monkeypatch.setattr(db_module, "get_alembic_config", lambda: _config(scripts_path, db_path))
    monkeypatch.setattr(db_module, "Session", None)
    monkeypatch.setattr(db_module, "WriteSession", None)
    monkeypatch.setattr(db_module, "_db_lock", None)
    yield scripts_path, db_path
    for factory in (db_module.Session, db_module.WriteSession):
        if factory is not None:
            factory.kw["bind"].dispose()
    if db_module._db_lock is not None:
        db_module._db_lock.release(force=True)


@pytest.fixture
def temp_root(tmp_path, monkeypatch):
    """The directory tempfile.gettempdir() returns, so copies land in tmp_path."""
    root = tmp_path / "100% tmp"  # % is ConfigParser syntax in Alembic options
    root.mkdir()
    monkeypatch.setattr(db_module.tempfile, "tempdir", str(root))
    return root


def _copies(temp_root) -> list[str]:
    return [str(p) for p in temp_root.glob("comfyui-alembic-versions-*")]


def _relaunch():
    """Run _init_file_db again, as the next launch of the same install would."""
    for factory in (db_module.Session, db_module.WriteSession):
        factory.kw["bind"].dispose()
    db_module._db_lock.release(force=True)
    db_module.Session = db_module.WriteSession = db_module._db_lock = None
    db_module._init_file_db(db_module.args.database_url)


def _tree(path: str) -> dict[str, bytes]:
    files = {}
    for root, dirs, names in os.walk(path):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in names:
            with open(os.path.join(root, name), "rb") as f:
                files[os.path.relpath(os.path.join(root, name), path)] = f.read()
    return files


def _plant_appledouble(scripts_path: str) -> str:
    planted = os.path.join(scripts_path, "versions", "._0001_assets.py")
    with open(planted, "wb") as f:
        f.write(_APPLEDOUBLE)
    return planted


def test_appledouble_file_breaks_alembic_on_its_own(scripts):
    scripts_path, db_path = scripts
    _plant_appledouble(scripts_path)

    # Python 3.12+ raises SyntaxError for NUL bytes in source; 3.10 and 3.11 raise ValueError.
    with pytest.raises((SyntaxError, ValueError), match="null bytes"):
        ScriptDirectory.from_config(_config(scripts_path, db_path)).get_current_head()


def test_upgrade_ignores_appledouble_files_and_leaves_the_install_unchanged(scripts, temp_root):
    scripts_path, db_path = scripts
    head = _head(scripts_path)
    command.upgrade(_config(scripts_path, db_path), "0006_add_loader_path")
    _plant_appledouble(scripts_path)
    # A real install also has bytecode and non-revision files beside the revisions.
    os.makedirs(os.path.join(scripts_path, "versions", "__pycache__"), exist_ok=True)
    with open(os.path.join(scripts_path, "versions", "README"), "w") as f:
        f.write("not a revision")
    install = _tree(scripts_path)

    # Alembic must read revisions from the filtered copy only: if it also scanned
    # <script_location>/versions it would load the planted file and raise SyntaxError.
    db_module._init_file_db(db_module.args.database_url)

    assert _current_revision(db_path) == head
    assert os.path.exists(db_path + ".bkp")
    assert _tree(scripts_path) == install
    assert len(_copies(temp_root)) == 1


def test_next_launch_replaces_the_copy(scripts, temp_root):
    scripts_path, db_path = scripts
    head = _head(scripts_path)
    _plant_appledouble(scripts_path)
    db_module._init_file_db(db_module.args.database_url)
    (copy,) = _copies(temp_root)
    # A copy left by an earlier launch, here one with a revision this install no longer has.
    stale = os.path.join(copy, "0099_stale.py")
    with open(stale, "wb") as f:
        f.write(_APPLEDOUBLE)

    _relaunch()

    assert _copies(temp_root) == [copy]
    assert not os.path.exists(stale)
    assert _current_revision(db_path) == head


def test_each_install_gets_its_own_copy(scripts, temp_root, tmp_path, monkeypatch):
    scripts_path, db_path = scripts
    other = str(tmp_path / "other" / "alembic_db")
    shutil.copytree(scripts_path, other)
    _plant_appledouble(scripts_path)
    _plant_appledouble(other)
    db_module._init_file_db(db_module.args.database_url)

    monkeypatch.setattr(db_module, "get_alembic_config", lambda: _config(other, db_path))
    _relaunch()

    assert len(_copies(temp_root)) == 2


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions, as non-root"
)
def test_copy_of_a_read_only_install_stays_removable(scripts, temp_root):
    scripts_path, db_path = scripts
    head = _head(scripts_path)
    _plant_appledouble(scripts_path)
    versions = os.path.join(scripts_path, "versions")
    read_only = [os.path.join(versions, name) for name in os.listdir(versions)] + [versions]
    for path in read_only:
        os.chmod(path, os.stat(path).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    try:
        db_module._init_file_db(db_module.args.database_url)
        _relaunch()
    finally:
        for path in read_only:
            os.chmod(path, os.stat(path).st_mode | stat.S_IWUSR)

    (copy,) = _copies(temp_root)
    # Other users can't write code into the copy, whatever the source's mode, and the copied
    # files don't keep a read-only mode (Windows can't delete read-only files).
    assert stat.S_IMODE(os.stat(copy).st_mode) & 0o077 == 0
    assert all(os.access(os.path.join(copy, name), os.W_OK) for name in os.listdir(copy))
    assert _current_revision(db_path) == head


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions, as non-root"
)
@pytest.mark.parametrize("taken_by", ["another user", "a symlink"])
def test_launch_works_when_the_copy_path_is_taken(scripts, temp_root, tmp_path, taken_by):
    scripts_path, db_path = scripts
    head = _head(scripts_path)
    _plant_appledouble(scripts_path)
    db_module._init_file_db(db_module.args.database_url)
    (copy,) = _copies(temp_root)
    if taken_by == "a symlink":
        shutil.rmtree(copy)
        os.symlink(tmp_path, copy)
    else:
        os.chmod(copy, 0o500)  # this user can no longer empty it, as with another user's copy
    try:
        _relaunch()
    finally:
        if not os.path.islink(copy):
            os.chmod(copy, 0o700)

    assert _current_revision(db_path) == head


def test_versions_without_appledouble_files_are_used_in_place(scripts, temp_root):
    scripts_path, db_path = scripts

    db_module._init_file_db(db_module.args.database_url)

    assert _copies(temp_root) == []
    assert _current_revision(db_path) == _head(scripts_path)
