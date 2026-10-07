import os

import pytest
import torch
from filelock import FileLock, Timeout

import app.logger
from app.database import db as db_module
from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import main  # noqa: E402


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = str(tmp_path / "comfyui.db")
    monkeypatch.setattr(db_module.args, "database_url", f"sqlite:///{path}")
    return path


@pytest.fixture
def startup_warnings(monkeypatch):
    warnings = []
    monkeypatch.setattr(app.logger, "STARTUP_WARNINGS", warnings)
    return warnings


class _AssetsOff:
    enabled = False

    def __init__(self):
        self.started = False

    def startup(self):
        self.started = True


def test_probe_without_a_lock_file_creates_none(db_path):
    assert db_module.lock_holder_db_path() is None
    assert not os.path.exists(db_path + ".lock")


def test_probe_reports_a_held_lock(db_path):
    holder = FileLock(db_path + ".lock")
    holder.acquire(timeout=0)
    try:
        assert db_module.lock_holder_db_path() == db_path
    finally:
        holder.release()


def test_probe_does_not_keep_a_free_lock(db_path, monkeypatch):
    open(db_path + ".lock", "a").close()
    probes = []

    class _KeptAlive(FileLock):
        # Holding a reference stops garbage collection from releasing a lock the probe kept.
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            probes.append(self)

    monkeypatch.setattr(db_module, "FileLock", _KeptAlive)

    assert db_module.lock_holder_db_path() is None
    assert len(probes) == 1
    assert not os.path.exists(db_path + ".lock.json")

    contender = FileLock(db_path + ".lock")
    try:
        contender.acquire(timeout=0)
    except Timeout:
        pytest.fail("the probe kept the lock it was only meant to test")
    contender.release()


def test_probe_skips_non_sqlite_databases(monkeypatch):
    monkeypatch.setattr(db_module.args, "database_url", "postgresql://localhost/comfy")

    assert db_module.lock_holder_db_path() is None


def test_probe_treats_a_failed_check_as_free(db_path, monkeypatch):
    open(db_path + ".lock", "a").close()

    def _unexpected(self, *args, **kwargs):
        raise NotImplementedError("no locking on this filesystem")

    monkeypatch.setattr(FileLock, "acquire", _unexpected)

    assert db_module.lock_holder_db_path() is None


def test_assets_off_leaves_the_database_alone(db_path, startup_warnings, monkeypatch):
    monkeypatch.setattr(main, "dependencies_available", lambda: True)

    def _init_db():
        pytest.fail("init_db ran with assets off")

    monkeypatch.setattr(main, "init_db", _init_db)
    asset_manager = _AssetsOff()

    main.setup_database(asset_manager)

    assert asset_manager.started
    assert not os.path.exists(db_path)
    assert not os.path.exists(db_path + ".lock")
    assert not os.path.exists(db_path + ".lock.json")
    assert startup_warnings == []


def test_assets_off_warns_and_continues_when_another_process_holds_the_lock(
    db_path, startup_warnings, monkeypatch
):
    monkeypatch.setattr(main, "dependencies_available", lambda: True)
    monkeypatch.setattr(main, "init_db", lambda: pytest.fail("init_db ran with assets off"))
    asset_manager = _AssetsOff()
    holder = FileLock(db_path + ".lock")
    holder.acquire(timeout=0)
    with open(db_path + ".lock.json", "w") as f:
        f.write('{"pid": 1}')
    try:
        main.setup_database(asset_manager)
    finally:
        holder.release()

    assert asset_manager.started
    with open(db_path + ".lock.json") as f:
        assert f.read() == '{"pid": 1}'
    assert len(startup_warnings) == 1
    warning = startup_warnings[0]
    assert "Another ComfyUI is already using this install's asset database" in warning
    assert db_path in warning
    assert "This ComfyUI was started without --enable-assets, so it doesn't need that database and will start anyway" in warning
    assert "A future version will refuse to start two ComfyUIs on the same asset database" in warning


def test_probe_treats_a_failed_release_as_free(db_path, monkeypatch):
    open(db_path + ".lock", "a").close()

    def _unexpected(self, *args, **kwargs):
        raise OSError("unlock failed")

    monkeypatch.setattr(FileLock, "release", _unexpected)

    assert db_module.lock_holder_db_path() is None
