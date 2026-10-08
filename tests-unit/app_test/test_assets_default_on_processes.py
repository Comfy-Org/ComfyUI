"""Real ComfyUI starts: assets are on unless --disable-assets, and a database that can't be
opened stops startup."""

import contextlib
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


def _comfy_args(base: Path, *flags: str) -> list[str]:
    # Always an explicit database: without one, startup relocates the checkout's own user/comfyui.db.
    return [
        sys.executable,
        str(REPO_ROOT / "main.py"),
        "--cpu",
        "--disable-all-custom-nodes",
        "--disable-partner-nodes",
        f"--base-directory={base}",
        f"--front-end-root={base}",
        f"--database-url=sqlite:///{_db(base)}",
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


def test_corrupt_database_stops_startup(tmp_path):
    _db(tmp_path).parent.mkdir()
    _db(tmp_path).write_bytes(b"not a database" * 1000)

    result = _quick_start(tmp_path)

    assert result.returncode == 1, result.stderr
    assert f"The asset database '{_db(tmp_path)}' is corrupt" in result.stderr
    assert "--disable-assets" in result.stderr
    assert "Traceback" not in result.stderr


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


def _serve(tmp_path, *flags):
    port = _free_port()
    log_path = tmp_path / "server.log"
    with open(log_path, "w") as log:
        server = subprocess.Popen(
            _comfy_args(tmp_path, *flags, "--listen", "127.0.0.1", "--port", str(port)),
            cwd=tmp_path,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            assert server.poll() is None, log_path.read_text()[-4000:]
            with contextlib.suppress(requests.ConnectionError):
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
