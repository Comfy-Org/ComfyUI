"""Real ComfyUI starts: assets are on unless --disable-assets, a database that can't be
opened stops startup, and one a newer ComfyUI upgraded turns assets off for the run."""

import contextlib
import os
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests
from alembic.script import ScriptDirectory

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPRECATED = "--enable-assets is deprecated and does nothing"
HEAD = ScriptDirectory(str(REPO_ROOT / "alembic_db")).get_current_head()


def _comfy_args(base: Path, *flags: str, database_url: bool = True) -> list[str]:
    # An explicit database unless asked: without one, startup relocates the checkout's own user/comfyui.db.
    if not database_url and (REPO_ROOT / "user" / "comfyui.db").exists():
        pytest.skip("this checkout has a user/comfyui.db that a start without --database-url would relocate")
    return [
        sys.executable,
        str(REPO_ROOT / "main.py"),
        "--cpu",
        "--disable-all-custom-nodes",
        "--disable-partner-nodes",
        f"--base-directory={base}",
        f"--front-end-root={base}",
        *([f"--database-url=sqlite:///{_db(base)}"] if database_url else []),
        *flags,
    ]


def _db(base: Path) -> Path:
    return base / "user" / "comfyui.db"


def _quick_start(base: Path, *flags: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        _comfy_args(base, "--quick-test-for-ci", *flags),
        cwd=base,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _revision(db: Path) -> str:
    with sqlite3.connect(db) as conn:
        return conn.execute("SELECT version_num FROM alembic_version").fetchone()[0]


@pytest.mark.parametrize(("flags", "hashing"), [((), "false"), (("--enable-asset-hashing",), "true")])
def test_assets_are_on_by_default_and_hashing_stays_opt_in(tmp_path, flags, hashing):
    result = _quick_start(tmp_path, *flags)
    output = result.stdout + result.stderr

    assert result.returncode == 0, result.stderr
    assert _revision(_db(tmp_path)) == HEAD
    assert [line.split("assets.enabled ", 1)[1] for line in output.splitlines() if "[assets-event] assets.enabled " in line] == [
        f"hashing_enabled={hashing}"
    ]
    assert DEPRECATED not in result.stderr


def test_enable_assets_is_accepted_and_says_once_that_it_does_nothing(tmp_path):
    result = _quick_start(tmp_path, "--enable-assets")

    assert result.returncode == 0, result.stderr
    assert result.stderr.count(DEPRECATED) == 1
    assert _revision(_db(tmp_path)) == HEAD


def test_disable_assets_leaves_the_database_alone(tmp_path):
    result = _quick_start(tmp_path, "--disable-assets")

    assert result.returncode == 0, result.stderr
    assert not _db(tmp_path).exists()


def test_starts_with_a_percent_sign_in_the_database_path(tmp_path):
    # Alembic config values go through ConfigParser interpolation, where % is special.
    base = tmp_path / "50% data"
    base.mkdir()

    result = _quick_start(base)

    assert result.returncode == 0, result.stderr
    assert _revision(_db(base)) == HEAD


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def default_server(tmp_path):
    yield from _serve(tmp_path)


@pytest.fixture
def disabled_server(tmp_path):
    yield from _serve(tmp_path, "--disable-assets")


def _serve(tmp_path, *flags, database_url=True):
    port = _free_port()
    log_path = tmp_path / "server.log"
    with open(log_path, "w") as log:
        server = subprocess.Popen(
            _comfy_args(tmp_path, *flags, "--listen", "127.0.0.1", "--port", str(port), database_url=database_url),
            cwd=tmp_path,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            assert server.poll() is None, log_path.read_text()[-4000:]
            with contextlib.suppress(requests.RequestException):
                if requests.get(f"{base_url}/system_stats", timeout=1).status_code == 200:
                    break
            time.sleep(0.25)
        else:
            raise AssertionError(f"server did not start\n{log_path.read_text()[-4000:]}")
        yield base_url
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


def test_default_server_serves_assets_and_registers_uploads(default_server):
    upload = requests.post(
        f"{default_server}/upload/image",
        files={"image": ("default-on.png", b"default-on-bytes", "image/png")},
        data={"type": "input"},
        timeout=10,
    )
    assert upload.status_code == 200
    assert upload.json()["asset"]["name"] == "default-on.png"
    view = requests.get(f"{default_server}/view", params={"filename": "default-on.png", "type": "input"}, timeout=10)
    assert view.content == b"default-on-bytes"

    listed = requests.get(f"{default_server}/api/assets", timeout=10)

    assert listed.status_code == 200
    assert "default-on.png" in [asset["name"] for asset in listed.json()["assets"]]
    assert requests.get(f"{default_server}/features", timeout=10).json()["assets"] is True


def test_disabled_server_reports_assets_off(disabled_server):
    disabled = requests.get(f"{disabled_server}/api/assets", timeout=10)

    assert disabled.status_code == 503
    assert "--disable-assets" in disabled.json()["error"]["message"]
    assert requests.get(f"{disabled_server}/features", timeout=10).json()["assets"] is False


def _stamp_newer(db: Path) -> None:
    db.parent.mkdir(parents=True)
    with contextlib.closing(sqlite3.connect(db)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")  # as a newer ComfyUI leaves it
        conn.execute("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        conn.execute("INSERT INTO alembic_version VALUES ('0099_from_a_newer_release')")
        conn.commit()


def test_database_from_a_newer_comfyui_runs_without_assets_and_is_left_as_it_is(tmp_path):
    # The default database in a custom user directory: the directory must be known before it opens.
    user_dir = tmp_path / "elsewhere" / "user"
    db = user_dir / "comfyui.db"
    _stamp_newer(db)
    before = db.read_bytes()

    server = _serve(tmp_path, f"--user-directory={user_dir}", database_url=False)
    base_url = next(server)
    try:
        upload = requests.post(
            f"{base_url}/upload/image",
            files={"image": ("newer.png", b"newer-bytes", "image/png")},
            data={"type": "input"},
            timeout=10,
        )
        assert upload.status_code == 200
        assert requests.get(f"{base_url}/api/assets", timeout=10).status_code == 503
        assert requests.get(f"{base_url}/features", timeout=10).json()["assets"] is False
    finally:
        server.close()

    log = (tmp_path / "server.log").read_text()
    assert "ASSETS_DISABLED: newer_revision\n" in log
    assert f"The asset database '{db}' was upgraded by a newer version of ComfyUI (revision '0099_from_a_newer_release')" in log
    assert "ASSETS_STARTUP_FAILED" not in log
    assert "Traceback" not in log
    assert db.read_bytes() == before
    wal = user_dir / "comfyui.db-wal"
    assert not wal.exists() or wal.stat().st_size == 0  # nothing written during the run either
    # Reading a WAL database can add an empty -wal and the -shm index; the lock file stays on POSIX only.
    assert set(os.listdir(user_dir)) - {"comfyui.db.lock", "comfyui.db-wal", "comfyui.db-shm"} == {"comfyui.db"}
