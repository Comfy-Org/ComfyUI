"""Two ComfyUIs launched together on one corrupt database: the database lock lets only one
recover it, and the other fails to start as it does for any database already in use."""

import glob
import os
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.request
from contextlib import closing
from pathlib import Path

from alembic import command
from alembic.config import Config

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_HELD = "Database is locked. Another ComfyUI process is already using this database."


def _make_corrupt_db(db_path: Path, table: str) -> None:
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "alembic_db"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    command.upgrade(cfg, "head")
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("PRAGMA journal_mode=DELETE")
        page = conn.execute("SELECT rootpage FROM sqlite_master WHERE name = ?", (table,)).fetchone()[0]
    with open(db_path, "r+b") as f:
        f.seek((page - 1) * 4096)
        f.write(b"\xa5" * 4096)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _launch(base: Path, port: int, log: Path) -> subprocess.Popen:
    with open(log, "w") as out:
        return subprocess.Popen(
            [
                sys.executable, "main.py", "--cpu", "--enable-assets",
                "--disable-all-custom-nodes", "--disable-partner-nodes",
                f"--base-directory={base}", f"--front-end-root={base}",
                "--listen", "127.0.0.1", "--port", str(port),
            ],
            cwd=REPO_ROOT, stdout=out, stderr=subprocess.STDOUT,
        )


def _serving(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=2) as response:
            return response.status == 200
    except OSError:
        return False


def test_two_launches_on_a_corrupt_database_recover_it_once(tmp_path, table="asset_system_state"):
    # asset_system_state: recovered after init, while the lock is held.
    db_path = tmp_path / "user" / "comfyui.db"
    db_path.parent.mkdir()
    _make_corrupt_db(db_path, table)
    ports = [_free_port(), _free_port()]
    logs = [tmp_path / "first.log", tmp_path / "second.log"]

    launches = [_launch(tmp_path, port, log) for port, log in zip(ports, logs)]
    try:
        deadline = time.monotonic() + 180
        serving = [False, False]
        while time.monotonic() < deadline:
            exited = [p.poll() is not None for p in launches]
            serving = [not done and _serving(port) for done, port in zip(exited, ports)]
            if all(exited) or (any(exited) and any(serving)):
                break
            time.sleep(0.5)
        output = [log.read_text() for log in logs]

        assert sorted(serving) == [False, True], output
        loser = serving.index(False)
        assert launches[loser].returncode == 1, output[loser][-3000:]
        assert LOCK_HELD in output[loser]
        winner = serving.index(True)
        assert "Database quarantined" in output[winner]
        assert "Database quarantined" not in output[loser]
    finally:
        for p in launches:
            if p.poll() is None:
                p.terminate()
                p.wait(timeout=30)

    assert len([p for p in glob.glob(str(db_path) + ".corrupt-*") if not p.endswith(("-wal", "-shm", "-journal"))]) == 1
    assert os.path.exists(db_path)
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
