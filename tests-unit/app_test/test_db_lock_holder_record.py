import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from app.database import db as db_module

REPO_ROOT = Path(__file__).resolve().parents[2]

# Takes the lock as ComfyUI does, reports it, then exits the way stdin asks. On Linux it
# first renames itself, since a process name may contain spaces and parentheses.
HOLDER_SCRIPT = (
    "import os, sys; "
    "sys.platform.startswith('linux') and open('/proc/self/comm', 'w').write('a) b (c'); "
    "from app.database import db; "
    "db._acquire_file_lock(sys.argv[1]); "
    "print(os.getpid(), flush=True); "
    "how = sys.stdin.readline().strip(); "
    "os._exit(0) if how == 'crash' else sys.exit(0)"
)


@pytest.fixture(autouse=True)
def isolated_lock(monkeypatch):
    monkeypatch.setattr(db_module, "_db_lock", None)
    yield
    if db_module._db_lock is not None:
        db_module._db_lock.release(force=True)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "comfyui.db")


def _read_record(db_path):
    with open(db_path + ".lock.json", encoding="utf-8") as f:
        return json.load(f)


def _independent_start_token(pid):
    if sys.platform.startswith("linux"):
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        start_ticks = re.match(r".*\)\s+(.*)", Path(f"/proc/{pid}/stat").read_text(), re.S).group(1).split()[19]
        return f"{boot_id}:{start_ticks}"
    if sys.platform == "win32":
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", f"(Get-Process -Id {pid}).StartTime.ToFileTimeUtc()"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    out = subprocess.run(
        ["ps", "-o", "lstart=", "-p", str(pid)],
        capture_output=True, text=True, check=True, env={**os.environ, "TZ": "UTC", "LC_ALL": "C"},
    )
    return out.stdout.strip()


def _start_holder(db_path):
    """The holder process and its pid. A Windows venv's python.exe is a launcher that runs
    the interpreter as a child, so Popen.pid is not the holder's pid there."""
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLDER_SCRIPT, db_path],
        cwd=REPO_ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    line = holder.stdout.readline()
    if not line.strip().isdigit():
        holder.kill()
        holder.communicate()
        pytest.fail(f"the lock holder did not start: {line!r}")
    return holder, int(line)


def _stop_holder(holder, how):
    holder.stdin.write(how + "\n")
    holder.stdin.flush()
    assert holder.wait(timeout=30) == 0


def test_record_describes_the_lock_holder(db_path, monkeypatch):
    # The macOS token must not depend on the holder's time zone or locale.
    monkeypatch.setenv("TZ", "America/New_York")
    monkeypatch.setenv("LC_ALL", "de_DE.UTF-8")

    db_module._acquire_file_lock(db_path)

    record = _read_record(db_path)
    assert record == {
        "version": 1,
        "pid": os.getpid(),
        "started": _independent_start_token(os.getpid()),
        "db": db_path,
        "main": record["main"],
        "argv": sys.argv,
    }
    assert os.path.realpath(record["main"]) == os.path.realpath(REPO_ROOT / "main.py")
    assert sorted(os.listdir(os.path.dirname(db_path))) == ["comfyui.db.lock", "comfyui.db.lock.json"]


def test_record_names_the_database_by_absolute_path(tmp_path, monkeypatch):
    # A copied record then names a database other than the one it sits beside.
    monkeypatch.chdir(tmp_path)

    db_module._acquire_file_lock("comfyui.db")

    assert _read_record("comfyui.db")["db"] == str(tmp_path / "comfyui.db")


def test_another_process_reads_the_record_while_the_lock_is_held(db_path):
    holder, pid = _start_holder(db_path)
    try:
        record = _read_record(db_path)
        assert record["pid"] == pid
        assert record["started"] == _independent_start_token(pid)
    finally:
        _stop_holder(holder, "exit")


def test_a_failed_acquire_leaves_the_holders_record(db_path, monkeypatch):
    monkeypatch.setattr(db_module, "_LOCK_WAIT_SECONDS", 0.2)
    holder, pid = _start_holder(db_path)
    try:
        with pytest.raises(RuntimeError, match="Could not acquire lock"):
            db_module._acquire_file_lock(db_path)
        assert _read_record(db_path)["pid"] == pid
    finally:
        _stop_holder(holder, "exit")


@pytest.mark.parametrize("how", ["exit", "crash"])
def test_a_previous_holders_record_stays_until_the_next_holder_replaces_it(db_path, how):
    holder, pid = _start_holder(db_path)
    _stop_holder(holder, how)
    assert _read_record(db_path)["pid"] == pid

    db_module._acquire_file_lock(db_path)

    assert _read_record(db_path)["pid"] == os.getpid()


@pytest.mark.parametrize("token", [lambda: None, lambda: 1 / 0], ids=["empty", "raises"])
def test_no_record_without_a_start_token(db_path, monkeypatch, token):
    monkeypatch.setattr(db_module, "_process_start_token", token)

    db_module._acquire_file_lock(db_path)

    assert db_module._db_lock.is_locked
    assert os.listdir(os.path.dirname(db_path)) == ["comfyui.db.lock"]


def test_a_failed_write_keeps_the_lock_and_leaves_no_files(db_path, monkeypatch, caplog):
    def _fail(src, dst):
        raise PermissionError("denied")

    monkeypatch.setattr(db_module.os, "replace", _fail)

    with caplog.at_level(logging.WARNING):
        db_module._acquire_file_lock(db_path)

    assert db_module._db_lock.is_locked
    assert os.listdir(os.path.dirname(db_path)) == ["comfyui.db.lock"]
    assert "Could not record the database lock holder" in caplog.text
