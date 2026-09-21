import logging
import os
import sqlite3

import pytest
import torch
from alembic import command
from alembic.config import Config
from filelock import FileLock, Timeout

from app.database import db as db_module
from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import main  # noqa: E402

_PRE_HEAD = "0006_add_loader_path"


def _make_config(db_path: str) -> Config:
    root = os.path.join(os.path.dirname(__file__), "../..")
    cfg = Config(os.path.abspath(os.path.join(root, "alembic.ini")))
    cfg.set_main_option("script_location", os.path.abspath(os.path.join(root, "alembic_db")))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return cfg


def _current_revision(db_path: str) -> str:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT version_num FROM alembic_version").fetchall()
    assert len(rows) == 1
    return rows[0][0]


@pytest.fixture
def stale_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "comfyui.db")
    command.upgrade(_make_config(db_path), _PRE_HEAD)

    monkeypatch.setattr(db_module.args, "database_url", f"sqlite:///{db_path}")
    monkeypatch.setattr(db_module, "Session", None)
    monkeypatch.setattr(db_module, "_db_lock", None)
    yield db_path
    if db_module._db_lock is not None:
        db_module._db_lock.release(force=True)


def test_init_file_db_migrates_when_lock_is_free(stale_db):
    db_module._init_file_db(db_module.args.database_url)

    assert _current_revision(stale_db) != _PRE_HEAD
    assert os.path.exists(stale_db + ".bkp")


def test_successful_init_keeps_holding_the_lock(stale_db):
    db_module._init_file_db(db_module.args.database_url)

    contender = FileLock(stale_db + ".lock")
    with pytest.raises(Timeout):
        contender.acquire(timeout=0)


def test_failed_init_releases_the_lock(stale_db, monkeypatch):
    def _explode():
        raise RuntimeError("alembic config exploded")

    monkeypatch.setattr(db_module, "get_alembic_config", _explode)

    with pytest.raises(RuntimeError, match="alembic config exploded"):
        db_module._init_file_db(db_module.args.database_url)

    contender = FileLock(stale_db + ".lock")
    try:
        contender.acquire(timeout=0)
    except Timeout:
        pytest.fail(
            "a failed init stranded the lock; setup_database logs and CONTINUES when assets are "
            "disabled, so this process would block every other instance for its whole lifetime "
            "over a database it never opened"
        )
    contender.release()


def test_held_lock_blocks_before_any_migration_work(stale_db):
    holder = FileLock(stale_db + ".lock")
    holder.acquire(timeout=0)
    try:
        with pytest.raises(RuntimeError, match="Another ComfyUI process may already be using it"):
            db_module._init_file_db(db_module.args.database_url)

        assert not os.path.exists(stale_db + ".bkp")
        assert _current_revision(stale_db) == _PRE_HEAD
        assert db_module.Session is None
    finally:
        holder.release()


def test_setup_database_routes_file_lock_to_lock_guidance(monkeypatch, caplog):
    monkeypatch.setattr(main, "dependencies_available", lambda: True)

    def _raise_file_lock():
        raise RuntimeError(
            "Could not acquire lock on database 'x.db'. "
            "Another ComfyUI process may already be using it. "
            "Use --database-url to specify a separate database file."
        )

    monkeypatch.setattr(main, "init_db", _raise_file_lock)
    monkeypatch.setattr(main.args, "enable_assets", False)

    with caplog.at_level(logging.ERROR):
        result = main.setup_database(None)

    assert result is None
    assert "Database is locked. Another ComfyUI process is already using this database." in caplog.text
    assert "Failed to initialize database." not in caplog.text


class _RecordingManager:
    """Stands in for AssetsEnabled so a degrade can be observed without a database."""

    def __init__(self) -> None:
        self.disabled_with: BaseException | None = None

    def startup(self) -> None:
        return

    def disable(self, exc: BaseException) -> None:
        self.disabled_with = exc


def test_setup_database_disables_assets_for_file_lock_instead_of_exiting(monkeypatch, caplog):
    monkeypatch.setattr(main, "dependencies_available", lambda: True)

    def _raise_file_lock():
        raise RuntimeError(
            "Could not acquire lock on database 'x.db'. "
            "Another ComfyUI process may already be using it. "
            "Use --database-url to specify a separate database file."
        )

    monkeypatch.setattr(main, "init_db", _raise_file_lock)
    monkeypatch.setattr(main.args, "enable_assets", True)
    manager = _RecordingManager()

    with caplog.at_level(logging.ERROR):
        main.setup_database(manager)

    assert isinstance(manager.disabled_with, RuntimeError), (
        "a locked database must degrade to assets-off, not stop ComfyUI booting"
    )
    assert "Database is locked. Another ComfyUI process is already using this database." in caplog.text
    assert "Starting with assets disabled" in caplog.text


def test_setup_database_continues_for_driver_lock_when_assets_are_disabled(monkeypatch, caplog):
    monkeypatch.setattr(main, "dependencies_available", lambda: True)

    def _raise_driver_lock():
        raise Exception("sqlite3.OperationalError: database is locked")

    monkeypatch.setattr(main, "init_db", _raise_driver_lock)
    monkeypatch.setattr(main.args, "enable_assets", False)

    with caplog.at_level(logging.ERROR):
        main.setup_database(None)

    assert "Database is locked. Another ComfyUI process is already using this database." in caplog.text


def test_setup_database_gives_wal_its_own_guidance_and_never_suggests_memory(monkeypatch, caplog):
    monkeypatch.setattr(main, "dependencies_available", lambda: True)

    def _raise_wal():
        raise main.WalUnavailableError(
            "SQLite WAL could not be enabled for database '/mnt/nas/comfy.db'. "
            "SQLite WAL is not supported on network filesystems."
        )

    monkeypatch.setattr(main, "init_db", _raise_wal)
    monkeypatch.setattr(main.args, "enable_assets", True)
    manager = _RecordingManager()

    with caplog.at_level(logging.ERROR):
        main.setup_database(manager)

    assert ":memory:" not in caplog.text, (
        "an in-memory database silently discards the whole asset catalogue on restart; "
        "it must never be offered as a remedy for a network filesystem"
    )
    assert "keep the database on a local disk" in caplog.text
    assert "pip install -r requirements.txt" not in caplog.text
    assert isinstance(manager.disabled_with, main.WalUnavailableError)
