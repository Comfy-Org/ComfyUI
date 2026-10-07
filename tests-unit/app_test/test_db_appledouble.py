import os
import shutil
import sqlite3

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


def test_upgrade_ignores_appledouble_files_and_leaves_the_install_unchanged(scripts):
    scripts_path, db_path = scripts
    head = _head(scripts_path)
    command.upgrade(_config(scripts_path, db_path), "0006_add_loader_path")
    _plant_appledouble(scripts_path)
    install = _tree(scripts_path)

    # Alembic must read revisions from the filtered copy only: if it also scanned
    # <script_location>/versions it would load the planted file and raise SyntaxError.
    db_module._init_file_db(db_module.args.database_url)

    assert _current_revision(db_path) == head
    assert os.path.exists(db_path + ".bkp")
    assert _tree(scripts_path) == install


def test_versions_without_appledouble_files_are_used_in_place(scripts, monkeypatch):
    scripts_path, db_path = scripts

    def _no_copy():
        raise AssertionError("copied the versions dir without any ._ files in it")

    monkeypatch.setattr(db_module.tempfile, "mkdtemp", _no_copy)
    db_module._init_file_db(db_module.args.database_url)

    assert _current_revision(db_path) == _head(scripts_path)
