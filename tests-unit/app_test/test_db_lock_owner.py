import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import psutil
import pytest
from filelock import FileLock

from app.database import db as db_module

REPO_ROOT = Path(__file__).resolve().parents[2]

HOLD_SCRIPT = (
    "import sys, time; "
    "from app.database import db; "
    "db._acquire_file_lock(sys.argv[1]); "
    "print('held', flush=True); "
    "time.sleep(60)"
)


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
    monkeypatch.setattr(db_module, "_LOCK_WAIT_SECONDS", 0.5)
    yield
    if db_module._db_lock is not None:
        db_module._db_lock.release(force=True)


def _expect_lock_failure(db_path: str, caplog) -> str:
    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="Could not acquire lock"):
        db_module._acquire_file_lock(db_path)
    return caplog.text


def test_acquiring_the_lock_records_this_process_as_owner(tmp_path):
    db_path = str(tmp_path / "comfyui.db")

    db_module._acquire_file_lock(db_path)

    owner = json.loads(Path(db_path + ".lock.owner").read_text(encoding="utf-8"))
    process = psutil.Process()
    assert owner == {
        "pid": process.pid,
        "started": process.create_time(),
        "process": db_module._process_label(process.name(), process.cmdline()),
    }


def test_lock_failure_names_the_recorded_holder(tmp_path, caplog):
    db_path = str(tmp_path / "comfyui.db")
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLD_SCRIPT, db_path, "--api-key=holder-secret-value"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        holder_process = psutil.Process(holder.pid)
        expected = (
            f"Database lock held by pid {holder.pid} "
            f"(started {datetime.fromtimestamp(holder_process.create_time()).isoformat(timespec='seconds')}): "
            f"{holder_process.name()}"
        )
        record = Path(db_path + ".lock.owner").read_text(encoding="utf-8")

        log = _expect_lock_failure(db_path, caplog)
    finally:
        holder.kill()
        holder.wait()

    assert expected in log
    for value in ("holder-secret-value", db_path, "HOLD_SCRIPT", "_acquire_file_lock"):
        assert value not in record
        assert value not in log


@pytest.mark.parametrize(
    ("cmdline", "label"),
    [
        (["/venv/bin/python", "/home/user/ComfyUI/main.py", "--api-key", "secret"], "python main.py"),
        (["C:\\py\\python.exe", "-s", "ComfyUI\\main.py", "--listen"], "python main.py"),
        (["/venv/bin/python", "-m", "pip", "install", "https://token@host/pkg"], "python -m pip"),
        (["/venv/bin/python", "-c", "print('secret')"], "python"),
        (["/venv/bin/python"], "python"),
    ],
)
def test_process_label_keeps_no_argument_values(cmdline, label):
    assert db_module._process_label("python", cmdline) == label


def test_lock_failure_without_an_owner_record(tmp_path, caplog):
    db_path = str(tmp_path / "comfyui.db")
    holder = FileLock(db_path + ".lock")
    holder.acquire(timeout=0)
    try:
        log = _expect_lock_failure(db_path, caplog)
    finally:
        holder.release()

    assert "Database lock held by a process that did not record itself" in log


def test_lock_failure_ignores_a_record_whose_process_has_changed(tmp_path, caplog):
    db_path = str(tmp_path / "comfyui.db")
    stale = {"pid": psutil.Process().pid, "started": psutil.Process().create_time() - 3600, "cmdline": ["old"]}
    Path(db_path + ".lock.owner").write_text(json.dumps(stale), encoding="utf-8")
    holder = FileLock(db_path + ".lock")
    holder.acquire(timeout=0)
    try:
        log = _expect_lock_failure(db_path, caplog)
    finally:
        holder.release()

    assert "Database lock held by a process that did not record itself" in log
    assert "old" not in log


def test_failed_init_forgets_the_owner_record(tmp_path, monkeypatch):
    db_path = tmp_path / "comfyui.db"
    monkeypatch.setattr(db_module.args, "database_url", f"sqlite:///{db_path}")

    def _explode(*_args):
        raise RuntimeError("migration exploded")

    monkeypatch.setattr(db_module, "_migrate_and_bind", _explode)

    with pytest.raises(RuntimeError, match="migration exploded"):
        db_module._init_file_db(db_module.args.database_url)

    assert not Path(str(db_path) + ".lock.owner").exists()


def test_waits_for_a_holder_that_is_shutting_down(tmp_path, monkeypatch, caplog):
    db_path = str(tmp_path / "comfyui.db")
    monkeypatch.setattr(db_module, "_LOCK_WAIT_SECONDS", 5.0)
    holder = subprocess.Popen(
        [sys.executable, "-c", RELEASE_SCRIPT, db_path + ".lock", "0.5"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"

        with caplog.at_level(logging.WARNING):
            db_module._acquire_file_lock(db_path)
    finally:
        holder.kill()
        holder.wait()

    assert "another ComfyUI was still shutting down" in caplog.text
    assert "Database lock released after " in caplog.text
    owner = json.loads(Path(db_path + ".lock.owner").read_text(encoding="utf-8"))
    assert owner["pid"] == psutil.Process().pid


def test_free_lock_is_taken_without_waiting(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        db_module._acquire_file_lock(str(tmp_path / "comfyui.db"))

    assert "Database lock released after" not in caplog.text
