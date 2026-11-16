"""Real ComfyUI server helpers for the /api/v2 end-to-end tests (shared with other e2e suites).

`V2Server(root, *extra_args)` runs `main.py` on a free loopback port with every directory under `root`:
`a.db`, `in/`, `out/`, `temp/`, `user/`, `previews/`, `models/` (siblings; ComfyUI's real temp dir is `temp/temp`).
Logs go to `root/server-<n>.log`, one file per start, so a restart keeps the earlier log.
"""
from contextlib import contextmanager
import io
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Iterator

from PIL import Image
import requests

COMFYUI_ROOT = Path(__file__).parents[2]
EXTRA_MODEL_PATHS = Path(__file__).parent / "extra_model_paths.yaml"
TIMEOUT = (5, 30)  # (connect, read) seconds for every request
TERMINAL = {"succeeded", "failed", "canceled", "expired"}


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class V2Server:
    def __init__(self, root: Path, *extra_args: str):
        self.root, self.extra_args = root, extra_args
        self.process: subprocess.Popen | None = None
        self.logs: list[Path] = []
        self.base = ""
        for name in ("in", "out", "temp", "user", "previews", "models"):
            (root / name).mkdir(parents=True, exist_ok=True)

    @property
    def input_dir(self) -> Path:
        return self.root / "in"

    def command(self, port: int) -> list[str]:
        r = self.root
        return [
            sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(port), "--cpu",
            "--database-url", f"sqlite:///{r / 'a.db'}",
            "--output-directory", str(r / "out"), "--input-directory", str(r / "in"),
            "--temp-directory", str(r / "temp"), "--user-directory", str(r / "user"),
            "--previews-directory", str(r / "previews"), "--models-directory", str(r / "models"),
            "--extra-model-paths-config", str(EXTRA_MODEL_PATHS), "--disable-api-nodes",
            *self.extra_args,
        ]

    def start(self) -> None:
        port = free_port()
        self.base = f"http://127.0.0.1:{port}"
        log_path = self.root / f"server-{len(self.logs)}.log"
        self.logs.append(log_path)
        with log_path.open("w", encoding="utf-8") as log:
            self.process = subprocess.Popen(
                self.command(port), cwd=COMFYUI_ROOT, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
        try:
            self._wait_ready(log_path)
        except BaseException:
            self.stop()
            raise

    def _wait_ready(self, log_path: Path) -> None:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"server exited during startup:\n{log_path.read_text(encoding='utf-8')}")
            try:
                if requests.get(f"{self.base}/system_stats", timeout=2).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.25)
        raise TimeoutError(f"server readiness timed out:\n{log_path.read_text(encoding='utf-8')}")

    def stop(self) -> None:
        process, self.process = self.process, None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)

    def restart(self) -> None:
        self.stop()
        self.start()

    def url(self, path: str) -> str:
        return self.base + path


@contextmanager
def running(root: Path, *extra_args: str) -> Iterator[V2Server]:
    server = V2Server(root, *extra_args)
    try:
        server.start()
        yield server
    finally:
        server.stop()


def png_bytes(color: tuple[int, int, int], size: int = 8) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buffer, format="PNG")
    return buffer.getvalue()


def upload(server: V2Server, data: bytes, file_path: str, headers: dict | None = None) -> requests.Response:
    return requests.post(
        server.url("/api/v2/assets"), headers=headers, timeout=TIMEOUT,
        data={"file_path": file_path, "content_type": "image/png"},
        files={"file": (Path(file_path).name, data, "image/png")},
    )


def asset_ref(asset_id: str) -> dict:
    return {"__type": "core/ASSET", "info": {"id": asset_id}}


def wait_job(server: V2Server, job_id: str, statuses=TERMINAL, timeout: float = 60, headers: dict | None = None) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = requests.get(server.url(f"/api/v2/jobs/{job_id}"), headers=headers, timeout=TIMEOUT)
        assert resp.status_code == 200, resp.text
        job = resp.json()
        if job["status"] in statuses:
            return job
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} never reached {statuses}: {job}")


def sse_frames(resp: requests.Response, timeout: float = 60) -> list[tuple[str, dict]]:
    """Read an SSE response to EOF -> [(event, data)]. Comment frames are skipped; raises past `timeout`."""
    deadline, buffer = time.monotonic() + timeout, b""
    for chunk in resp.iter_content(chunk_size=None):
        buffer += chunk
        if time.monotonic() > deadline:
            raise TimeoutError(f"SSE stream still open after {timeout}s: {buffer!r}")
    frames = []
    for block in buffer.replace(b"\r\n", b"\n").split(b"\n\n"):
        fields = dict(line.split(b": ", 1) for line in block.split(b"\n") if line and not line.startswith(b":"))
        if fields:
            frames.append((fields[b"event"].decode(), json.loads(fields[b"data"])))
    return frames
