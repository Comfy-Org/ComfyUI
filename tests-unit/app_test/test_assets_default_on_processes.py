"""Real ComfyUI starts: assets are off unless --enable-assets (and --disable-assets wins), and a
database that can't be opened stops startup."""

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
ENABLED_EVENT = "[assets-event] assets.enabled "
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
def test_enable_assets_turns_assets_on_and_hashing_stays_opt_in(tmp_path, flags, hashing):
    result = _quick_start(tmp_path, "--enable-assets", *flags)
    output = result.stdout + result.stderr

    assert result.returncode == 0, result.stderr
    assert _revision(_db(tmp_path)) == HEAD
    assert [line.split("assets.enabled ", 1)[1] for line in output.splitlines() if ENABLED_EVENT in line] == [
        f"hashing_enabled={hashing}"
    ]
    assert "--enable-assets is deprecated" not in output


@pytest.mark.parametrize("flags", [(), ("--disable-assets",), ("--enable-assets", "--disable-assets")])
def test_assets_off_leaves_the_database_alone(tmp_path, flags):
    result = _quick_start(tmp_path, *flags)

    assert result.returncode == 0, result.stderr
    assert not _db(tmp_path).exists()
    assert ENABLED_EVENT not in result.stdout + result.stderr


def test_database_from_a_newer_comfyui_stops_startup_without_a_traceback(tmp_path):
    _db(tmp_path).parent.mkdir()
    with sqlite3.connect(_db(tmp_path)) as conn:
        conn.execute("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        conn.execute("INSERT INTO alembic_version VALUES ('0099_from_a_newer_release')")

    result = _quick_start(tmp_path, "--enable-assets")

    assert result.returncode == 1, result.stderr
    assert "ASSETS_STARTUP_FAILED: newer_revision" in result.stderr
    assert "Traceback" not in result.stderr


def test_starts_with_a_percent_sign_in_the_database_path(tmp_path):
    # Alembic config values go through ConfigParser interpolation, where % is special.
    base = tmp_path / "50% data"
    base.mkdir()

    result = _quick_start(base, "--enable-assets")

    assert result.returncode == 0, result.stderr
    assert _revision(_db(base)) == HEAD


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def enabled_server(tmp_path):
    yield from _serve(tmp_path, "--enable-assets")


@pytest.fixture
def default_server(tmp_path):
    yield from _serve(tmp_path)


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


def test_enabled_server_serves_assets_and_registers_uploads(enabled_server):
    upload = requests.post(
        f"{enabled_server}/upload/image",
        files={"image": ("enabled.png", b"enabled-bytes", "image/png")},
        data={"type": "input"},
        timeout=10,
    )
    assert upload.status_code == 200
    assert upload.json()["asset"]["name"] == "enabled.png"
    view = requests.get(f"{enabled_server}/view", params={"filename": "enabled.png", "type": "input"}, timeout=10)
    assert view.content == b"enabled-bytes"

    listed = requests.get(f"{enabled_server}/api/assets", timeout=10)

    assert listed.status_code == 200
    assert "enabled.png" in [asset["name"] for asset in listed.json()["assets"]]
    assert requests.get(f"{enabled_server}/features", timeout=10).json()["assets"] is True


def test_default_server_reports_assets_off(default_server):
    disabled = requests.get(f"{default_server}/api/assets", timeout=10)

    assert disabled.status_code == 503
    assert "--enable-assets" in disabled.json()["error"]["message"]
    assert requests.get(f"{default_server}/features", timeout=10).json()["assets"] is False


def _listed_names(base_url):
    return [asset["name"] for asset in requests.get(f"{base_url}/api/assets", timeout=10).json()["assets"]]


# main.py always loads the checkout's extra_model_paths.yaml, whose model folders the startup scan would walk.
@pytest.mark.skipif((REPO_ROOT / "extra_model_paths.yaml").exists(), reason="checkout has an extra_model_paths.yaml")
def test_running_a_prompt_rescans_the_output_folder(enabled_server, tmp_path):
    log_path = tmp_path / "server.log"
    deadline = time.monotonic() + 60
    while requests.get(f"{enabled_server}/api/assets/seed/status", timeout=10).json()["state"] != "IDLE":
        assert time.monotonic() < deadline, f"startup scan did not finish\n{log_path.read_text()[-4000:]}"
        time.sleep(0.25)
    # Written behind ComfyUI's back, so only a scan of the output folder can find it.
    (tmp_path / "output").mkdir(exist_ok=True)
    (tmp_path / "output" / "undeclared.png").write_bytes(b"undeclared-bytes")
    assert "undeclared.png" not in _listed_names(enabled_server)

    prompt = {
        "1": {"class_type": "EmptyImage", "inputs": {"width": 8, "height": 8, "batch_size": 1, "color": 0}},
        "2": {"class_type": "PreviewImage", "inputs": {"images": ["1", 0]}},
    }
    assert requests.post(f"{enabled_server}/prompt", json={"prompt": prompt}, timeout=10).status_code == 200

    deadline = time.monotonic() + 60
    while "undeclared.png" not in _listed_names(enabled_server):
        assert time.monotonic() < deadline, f"the output folder was not rescanned after the prompt\n{log_path.read_text()[-4000:]}"
        time.sleep(0.25)
