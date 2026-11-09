"""How ComfyUI starts and stops a pack's process. The one place to plug in another runner.

The contract, for a pack directory ``entry``, a port and a token file:

    start: when it returns, a pack server for ``entry`` accepts connections on 127.0.0.1:<port>
           and admits the token in <token file>. (Run ``python -m comfy_api._impl.isolation.server
           _serve --pack <entry> --port <port> --token-file <token file>`` wherever the pack
           should run: here, a container, a gVisor sandbox... and make the port reachable.)
    stop:  the server on <port> is gone.

Built in: ``LocalLauncher``, plain processes on this machine. Any other runner is an executable
named by ``COMFY_PACK_LAUNCHER``, called as ``<launcher> start <entry> <port> <token file>`` and
``<launcher> stop <port>``. ComfyUI needs no changes for either. For a V2 pack the launcher also gets
``COMFY_PACK_PERMISSIONS`` (what review granted the pack, to enforce) and ``COMFY_PACK_REQUIREMENTS``.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time

from . import server

LAUNCHER_ENV = "COMFY_PACK_LAUNCHER"
PERMISSIONS_ENV = "COMFY_PACK_PERMISSIONS"  # JSON list, from the pack's manifest, for a launcher to enforce
REQUIREMENTS_ENV = "COMFY_PACK_REQUIREMENTS"  # JSON list of pip specifiers, for a launcher's environment


def _manifest(entry: str):
    from comfy_api._impl import manifest
    return manifest.read(entry)


class LocalLauncher:
    """Each pack in a plain local Python process (``server.PackServer``), on its own environment
    when it has requirements (``envs``)."""

    def start(self, entry: str, port: int, token_file: str) -> None:
        from . import envs
        data = _manifest(entry)
        python = envs.python_for(entry, data.get("requirements", []) if data is not None else None)
        server.PackServer.start(entry, port, token_file=token_file, python=python,
                                orphan_grace=server.DEFAULT_ORPHAN_GRACE)

    def stop(self, port: int) -> None:
        server.PackServer.stop(port)


class CommandLauncher:
    """``COMFY_PACK_LAUNCHER=/path/to/launcher``: any executable that keeps the contract."""

    def __init__(self, command: str, wait: float = 300.0) -> None:
        self.command = command
        self.wait = wait

    def start(self, entry: str, port: int, token_file: str) -> None:
        env = dict(os.environ, COMFY_ROOT=server.COMFY_ROOT)
        env.setdefault(server.ORPHAN_GRACE_ENV, str(server.DEFAULT_ORPHAN_GRACE))
        layout = server.folder_layout()  # ComfyUI's folders, for util.folders in the pack (the launcher passes env on)
        if layout is not None:
            env[server.FOLDERS_ENV] = json.dumps(layout)
        data = _manifest(entry)
        if data is not None:  # what the launcher enforces: the permissions review granted the pack
            env[PERMISSIONS_ENV] = json.dumps(data.get("permissions", []))
            env[REQUIREMENTS_ENV] = json.dumps(data.get("requirements", []))
        done = subprocess.run([self.command, "start", entry, str(port), token_file], env=env,
                              capture_output=True, text=True, timeout=self.wait)
        if done.returncode != 0:
            raise RuntimeError(f"{self.command} start {entry} failed ({done.returncode}):\n"
                               f"{done.stdout[-2000:]}{done.stderr[-4000:]}")
        deadline = time.monotonic() + self.wait
        while True:  # the contract says it's listening; allow for a forwarded port to come up
            try:
                socket.create_connection(("127.0.0.1", port), timeout=2).close()
                return
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{self.command} started {entry} but nothing listens on port {port}")
                time.sleep(0.1)

    def stop(self, port: int) -> None:
        subprocess.run([self.command, "stop", str(port)], capture_output=True, timeout=60)


def get_launcher() -> "LocalLauncher | CommandLauncher":
    from . import setting
    command = (setting(LAUNCHER_ENV) or "").strip()
    return CommandLauncher(command) if command else LocalLauncher()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
