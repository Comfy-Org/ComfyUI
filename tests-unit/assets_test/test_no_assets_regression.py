import contextlib
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(("127.0.0.1", 0))
        return server_socket.getsockname()[1]


@pytest.fixture
def no_assets_server(tmp_path: Path):
    for directory in ("models", "custom_nodes", "input", "output", "temp", "user"):
        (tmp_path / directory).mkdir()
    port = _free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "main.py",
            f"--base-directory={tmp_path}",
            "--listen",
            "127.0.0.1",
            "--port",
            str(port),
            "--cpu",
        ],
        cwd=Path(__file__).resolve().parents[2],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base_url = f"http://127.0.0.1:{port}"
    for _ in range(120):
        if process.poll() is not None:
            raise RuntimeError(f"No-assets server exited with {process.returncode}")
        try:
            if requests.get(f"{base_url}/system_stats", timeout=1).status_code == 200:
                break
        except requests.ConnectionError:
            pass
        time.sleep(0.25)
    else:
        raise RuntimeError("No-assets server did not start")
    yield base_url
    process.terminate()
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=15)


def test_no_assets_keeps_legacy_upload_and_view(no_assets_server: str):
    assert requests.get(f"{no_assets_server}/api/assets", timeout=10).status_code == 503

    upload = requests.post(
        f"{no_assets_server}/upload/image",
        files={"image": ("legacy.png", b"legacy-bytes", "image/png")},
        data={"type": "output"},
        timeout=10,
    )

    assert upload.status_code == 200
    filename = upload.json()["name"]
    view = requests.get(
        f"{no_assets_server}/view",
        params={"filename": filename, "type": "output"},
        timeout=10,
    )
    assert view.status_code == 200
    assert view.content == b"legacy-bytes"
