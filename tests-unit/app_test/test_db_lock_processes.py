"""Real ComfyUI processes sharing one install's database: an assets-off instance must
not block an assets-on one, and must warn when another process holds the database."""

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_HELD = "Another ComfyUI is already using this database"
IN_USE = "Another ComfyUI is already using this install's asset database"

HOLD_SCRIPT = (
    "import sys; "
    "from filelock import FileLock; "
    "lock = FileLock(sys.argv[1]); lock.acquire(timeout=0); "
    "print('held', flush=True); "
    "sys.stdin.readline()"
)


def _comfy_args(base: Path, *flags: str) -> list[str]:
    return [
        sys.executable,
        "main.py",
        "--cpu",
        "--disable-all-custom-nodes",
        "--disable-partner-nodes",
        f"--base-directory={base}",
        f"--front-end-root={base}",
        f"--database-url=sqlite:///{base / 'comfyui.db'}",
        *flags,
    ]


def _quick_start(base: Path, *flags: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        _comfy_args(base, "--quick-test-for-ci", *flags),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_until_serving(proc: subprocess.Popen, port: int, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise AssertionError(f"ComfyUI exited with code {proc.returncode} before serving")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(0.2)
    raise AssertionError("ComfyUI did not start serving")


@pytest.fixture
def held_lock(tmp_path):
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLD_SCRIPT, str(tmp_path / "comfyui.db.lock")],
        cwd=REPO_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        yield
    finally:
        holder.kill()
        holder.wait()


def test_running_assets_off_instance_does_not_block_an_assets_on_start(tmp_path):
    # A previous assets-on run leaves the database and its lock file behind, as on a real install.
    first = _quick_start(tmp_path, "--enable-assets")
    assert first.returncode == 0, first.stderr

    port = _free_port()
    server_log = tmp_path / "server.log"
    with open(server_log, "w") as log:
        server = subprocess.Popen(
            _comfy_args(tmp_path, "--disable-assets", "--listen", "127.0.0.1", "--port", str(port)),
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    try:
        try:
            _wait_until_serving(server, port)
        except AssertionError as e:
            raise AssertionError(f"{e}\n{server_log.read_text()[-4000:]}") from None

        second = _quick_start(tmp_path, "--enable-assets")

        assert server.poll() is None
    finally:
        server.terminate()
        server.wait(timeout=30)

    assert second.returncode == 0, second.stderr
    assert LOCK_HELD not in second.stderr


@pytest.mark.parametrize("flags", [(), ("--disable-assets",)])
def test_assets_off_start_warns_and_continues_when_the_database_is_held(tmp_path, held_lock, flags):
    result = _quick_start(tmp_path, *flags)

    assert result.returncode == 0, result.stderr
    assert IN_USE in result.stderr
    assert str(tmp_path / "comfyui.db") in result.stderr
    assert "doesn't use that database and will start anyway" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("flags", [("--enable-assets",), ("--windows-standalone-build", "--enable-assets")])
def test_assets_on_start_still_fails_when_the_database_is_held(tmp_path, held_lock, flags):
    result = _quick_start(tmp_path, *flags)

    assert result.returncode == 1, result.stderr
    assert LOCK_HELD in result.stderr


def test_portable_second_instance_starts_without_assets_when_database_is_held(tmp_path, held_lock):
    result = _quick_start(tmp_path, "--windows-standalone-build")

    assert result.returncode == 0, result.stderr
    assert IN_USE in result.stderr
    assert "doesn't use that database and will start anyway" in result.stderr


def test_simultaneous_portable_starts_share_one_database_lock(tmp_path):
    script = """
import os
import time
from pathlib import Path
import main
original_init_db = main.init_db
def init_db():
    original_init_db()
    time.sleep(6)
main.init_db = init_db
Path(os.environ["COMFY_READY"]).touch()
while not Path(os.environ["COMFY_GO"]).exists():
    time.sleep(0.01)
try:
    main.start_comfyui()
except SystemExit:
    print(f"assets={main.feature_flags.SERVER_FEATURE_FLAGS['assets']}", flush=True)
    raise
"""
    go = tmp_path / "go"
    processes = []
    try:
        for index in range(2):
            env = os.environ.copy()
            env["COMFY_READY"] = str(tmp_path / f"ready-{index}")
            env["COMFY_GO"] = str(go)
            processes.append(subprocess.Popen(
                [sys.executable, "-c", script, *_comfy_args(tmp_path, "--quick-test-for-ci", "--windows-standalone-build")[2:]],
                cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            ))
        deadline = time.monotonic() + 120
        while not all((tmp_path / f"ready-{index}").exists() for index in range(2)):
            assert all(process.poll() is None for process in processes)
            assert time.monotonic() < deadline
            time.sleep(0.1)
        go.touch()
        results = [process.communicate(timeout=120) for process in processes]
        assert all(process.returncode == 0 for process in processes), results
        assert sorted("assets=True" in stdout for stdout, _ in results) == [False, True]
        assert any(IN_USE in stderr for _, stderr in results)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()
