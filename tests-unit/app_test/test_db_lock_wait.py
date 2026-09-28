import logging
import subprocess
import sys
from pathlib import Path

import pytest
from filelock import FileLock

from app.database import db as db_module

REPO_ROOT = Path(__file__).resolve().parents[2]

RELEASE_SCRIPT = (
    "import sys, time; "
    "from filelock import FileLock; "
    "lock = FileLock(sys.argv[1]); lock.acquire(timeout=0); "
    "print('held', flush=True); "
    "time.sleep(float(sys.argv[2])); "
    "lock.release()"
)


@pytest.fixture(autouse=True)
def restore_module_lock(monkeypatch):
    monkeypatch.setattr(db_module, "_db_lock", None)
    yield
    if db_module._db_lock is not None:
        db_module._db_lock.release(force=True)


def test_waits_for_a_holder_that_is_shutting_down(tmp_path, caplog):
    db_path = str(tmp_path / "comfyui.db")
    holder = subprocess.Popen(
        [sys.executable, "-c", RELEASE_SCRIPT, db_path + ".lock", "0.5"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"

        with caplog.at_level(logging.INFO):
            db_module._acquire_file_lock(db_path)
    finally:
        holder.kill()
        holder.wait()

    assert db_module._db_lock.is_locked
    assert "Database lock acquired after waiting " in caplog.text


def test_gives_up_when_the_lock_stays_held(tmp_path, monkeypatch, caplog):
    db_path = str(tmp_path / "comfyui.db")
    monkeypatch.setattr(db_module, "_LOCK_WAIT_SECONDS", 0.3)
    holder = FileLock(db_path + ".lock")
    holder.acquire(timeout=0)
    try:
        with caplog.at_level(logging.INFO), pytest.raises(RuntimeError, match="Could not acquire lock"):
            db_module._acquire_file_lock(db_path)
    finally:
        holder.release()

    assert "Database lock acquired after waiting" not in caplog.text


def test_free_lock_is_taken_without_waiting(tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        db_module._acquire_file_lock(str(tmp_path / "comfyui.db"))

    assert db_module._db_lock.is_locked
    assert "Database lock acquired after waiting" not in caplog.text
