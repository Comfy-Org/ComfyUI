"""Pack processes die with ComfyUI, as a browser's page processes die with the browser.

ComfyUI holds a connection to every pack process it starts. However ComfyUI ends (``kill -9``
included), the OS closes that connection, and the pack process exits a few seconds later.
"""
import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from comfy_api._impl.isolation import launcher, server

COMFY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytestmark = pytest.mark.skipif(os.name != "posix", reason="kills processes with POSIX signals")


@pytest.fixture
def pack(tmp_path, monkeypatch):
    root = tmp_path / "tiny_pack"
    root.mkdir()
    (root / "__init__.py").write_text("NODE_CLASS_MAPPINGS = {}\n")
    run = tmp_path / "run"
    monkeypatch.setenv(server.RUN_DIR_ENV, str(run))
    return str(root)


def _gone(pid, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not server._alive(pid):
            return True
        time.sleep(0.1)
    return False


def _owner(code):
    """A stand-in for ComfyUI: a separate process that starts a pack and holds its connection."""
    proc = subprocess.Popen([sys.executable, "-c", textwrap.dedent(code)], cwd=COMFY, stdout=subprocess.PIPE,
                            env=dict(os.environ, PYTHONPATH=COMFY), text=True)
    line = proc.stdout.readline()
    assert line.strip().isdigit(), line
    return proc, int(line)


def test_a_pack_process_exits_when_comfyui_is_killed(pack):
    owner, pid = _owner(f"""
        import time
        from comfy_api._impl.isolation import launcher, server
        port = launcher.free_port()
        info = server.PackServer.start({pack!r}, port, orphan_grace=1.0)
        connection = server.connect(port, info["token"])
        print(info["pid"], flush=True)
        time.sleep(600)
    """)
    try:
        assert server._alive(pid)
        time.sleep(2.0)  # past the grace: a held connection keeps it running
        assert server._alive(pid)
        owner.send_signal(signal.SIGKILL)
        owner.wait(10)
        assert _gone(pid, 15), "the pack process outlived ComfyUI"
    finally:
        owner.kill()
        if server._alive(pid):
            os.kill(pid, signal.SIGKILL)


def test_the_built_in_launcher_starts_pack_processes_that_die_with_comfyui(pack, tmp_path):
    token = tmp_path / "token"
    token.write_text("ab" * 32)
    owner, pid = _owner(f"""
        import time
        from comfy_api._impl.isolation import launcher, server
        port = launcher.free_port()
        launcher.get_launcher().start({pack!r}, port, {str(token)!r})
        connection = server.connect(port, {"ab" * 32!r})
        print(server.read_info(port)["pid"], flush=True)
        time.sleep(600)
    """)
    try:
        owner.send_signal(signal.SIGKILL)
        owner.wait(10)
        assert _gone(pid, server.DEFAULT_ORPHAN_GRACE + 10)
    finally:
        owner.kill()
        if server._alive(pid):
            os.kill(pid, signal.SIGKILL)


def test_a_worker_process_exits_when_comfyui_is_killed(pack):
    owner, pid = _owner(f"""
        import time
        from comfy_api._impl.isolation.host import Worker
        worker = Worker({pack!r})
        print(worker.pid, flush=True)
        time.sleep(600)
    """)
    try:
        owner.send_signal(signal.SIGKILL)
        owner.wait(10)
        assert _gone(pid, 10), "the worker outlived ComfyUI"
    finally:
        owner.kill()
        if server._alive(pid):
            os.kill(pid, signal.SIGKILL)


def test_a_pack_started_by_hand_runs_until_stopped(pack):
    port = launcher.free_port()
    info = server.PackServer.start(pack, port)  # as scripts/pack.sh starts one: no owner
    try:
        server.connect(port, info["token"]).close()
        time.sleep(1.5)
        assert server._alive(info["pid"])
    finally:
        assert server.PackServer.stop(port)
    assert _gone(info["pid"], 10)
