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


@pytest.fixture(autouse=True)
def restore_module_lock(monkeypatch):
    monkeypatch.setattr(db_module, "_db_lock", None)
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
    assert owner == {"pid": process.pid, "started": process.create_time(), "cmdline": process.cmdline()}


def test_lock_failure_names_the_recorded_holder(tmp_path, caplog):
    db_path = str(tmp_path / "comfyui.db")
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLD_SCRIPT, db_path],
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
            f"{' '.join(holder_process.cmdline())}"
        )

        log = _expect_lock_failure(db_path, caplog)
    finally:
        holder.kill()
        holder.wait()

    assert expected in log


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
