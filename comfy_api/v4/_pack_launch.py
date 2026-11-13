"""How ComfyUI starts a pack's process, and the launcher it can be given instead.

ComfyUI starts each pack process, and starts it again after it stops, by calling the launcher
with the start message: a dict that the pack process reads from its stdin, with the keys
`argv`, `args`, `folders`, `epoch` and `log_level`, and `pack` and `name`, the path ComfyUI's
loader found the pack at and the name it gave it. `folders` holds the `output`, `input`, `temp`
and `user` folders: those the pack process uses as ComfyUI's, and that `util.folders` returns in
it. They are ComfyUI's own unless the launcher changes them.

`--pack-launcher module:function` names a launcher to use instead of `launch`. It is called as
`launch` is, `launcher(start)`, on any of ComfyUI's threads, and must return what `launch`
returns: a process object and ComfyUI's end of a connected stream socket whose other end the
pack process was given, as an inherited descriptor whose number is `start["socket"]`. ComfyUI
does nothing to a pack process except through those two:

- `kill()` ends the pack process and every program it started, at once. ComfyUI calls it when it
  ends the pack or sees the socket close, possibly many times, and after the process has ended.
- `poll()` returns the exit code once the process has ended, else None.
- `wait(timeout=None)` waits for the process to end and returns its exit code, or raises
  `subprocess.TimeoutExpired`, as `Popen.wait` does. ComfyUI calls it, without a timeout, after
  `kill()` or once the socket has closed, possibly from more than one thread.

If ComfyUI itself is killed, nothing in it runs: the pack process sees its socket close and ends
its programs and itself, unless it cannot run. A launcher that starts the process another way
must end it when ComfyUI's process is gone, as `launch`'s watchdog does.

A launcher that only changes the command, the environment or the folders calls `launch` with
them, and keeps its guarantees.
"""

import contextlib
import functools
import importlib
import os
import pickle
import signal
import socket
import subprocess
import sys
import threading
from typing import Protocol

import psutil

from comfy.cli_args import args


# The watchdog: once its stdin closes it waits a second, so that a pack process that can run ends
# the programs that left its group, then kills the group whose id is its `$0`. Written for any
# POSIX shell: dash, Ubuntu's `/bin/sh`, takes the signal only with `-s`.
WATCHDOG = 'read _; sleep 1; kill -s KILL -- "-$0" 2>/dev/null'


class Process(Protocol):
    """What a launcher returns for the pack process it started; see the module's docstring."""

    def kill(self) -> None: ...

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...


def pack_main() -> str:
    """The script a pack process runs, as ComfyUI sees it."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_pack_main.py")


def command() -> list[str]:
    """The command that runs a pack process with ComfyUI's interpreter."""
    return [sys.executable, pack_main()]


def launch(
    start: dict,
    argv: list[str] | None = None,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> tuple["LocalProcess", socket.socket]:
    """Start a pack process on this machine and give it `start`; return it and ComfyUI's end of
    the socket. `argv` (by default `command()`) must run the pack process in the process it
    starts, as a wrapper that execs it does, or in that process's group; `env` and `cwd` are as
    for `subprocess.Popen`, ComfyUI's own by default."""
    ours, theirs = socket.socketpair()
    # In a process group of its own, which ComfyUI ends with the programs the pack started.
    process = subprocess.Popen(
        argv or command(),
        stdin=subprocess.PIPE,
        pass_fds=(theirs.fileno(),),
        start_new_session=True,
        env=env,
        cwd=cwd,
    )
    with process.stdin:
        pickle.dump({**start, "socket": theirs.fileno()}, process.stdin)
    theirs.close()
    return LocalProcess(process), ours


class LocalProcess:
    """A pack process on this machine: the leader of a process group of its own, which a
    watchdog outside the group ends once ComfyUI's end of a pipe closes, as when ComfyUI is
    killed. Killing it ends the group, and the descendants that left the group while the process
    is there to name them. Not ended: a program that left the group and whose pack process then
    stopped by itself, or was ended by the watchdog while ComfyUI could not act; and a program
    that left the group and is no longer a descendant of the pack process, as a daemon is not."""

    def __init__(self, process: subprocess.Popen):
        self.process = process
        self.pid = process.pid
        self.lock = threading.Lock()
        self.watched = True
        watch, self.alive = os.pipe()
        # A shell, so the group ends also while the pack's interpreter cannot run.
        self.watchdog = subprocess.Popen(
            ["/bin/sh", "-c", WATCHDOG, str(self.pid)],
            stdin=watch,
            start_new_session=True,
        )
        os.close(watch)

    def poll(self) -> int | None:
        """The exit code once the process has ended, else None. An ended process is reaped
        only after its group is ended: until then its id names the group."""
        if self.process.returncode is None:
            if not _ended(self.pid):
                return None
            self.kill()
        return self.process.poll()

    def kill(self) -> None:
        """End the process and the programs it started: its process group, and its descendants
        that left the group. Does nothing once the process has been reaped: its group has been
        ended, and the id may be another program's."""
        if self.process.returncode is not None:
            return
        strays = _descendants(self.pid)
        # Gone, or only the process is left and not yet reaped (macOS then refuses).
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(self.pid, signal.SIGKILL)
        for stray in strays:
            with contextlib.suppress(psutil.Error):
                stray.kill()

    def wait(self, timeout: float | None = None) -> int:
        if timeout is not None:
            self.process.wait(timeout)
        with self.lock:
            if self.watched:
                # Before the process is reaped, while its group id cannot belong to another group.
                self.watchdog.kill()
                self.watchdog.wait()
                os.close(self.alive)
                self.watched = False
        return self.process.wait()


def _ended(pid: int) -> bool:
    """Whether the child `pid` has ended and waits to be reaped."""
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def _descendants(pid: int) -> list[psutil.Process]:
    """The processes `pid` started, and those they started; none once `pid` has ended."""
    try:
        return psutil.Process(pid).children(recursive=True)
    except psutil.Error:
        return []


def launcher():
    """The launcher `--pack-launcher` names, or `launch`."""
    if not args.pack_launcher:
        return launch
    return resolve(args.pack_launcher)


def start_process(start: dict) -> tuple[Process, socket.socket]:
    """Start a pack process with the launcher. Raises `TypeError`, naming the launcher, if it
    returned anything but a process object and a socket, after killing what it returned as the
    process if it can."""
    returned = launcher()(start)
    process, sock = (
        returned if type(returned) is tuple and len(returned) == 2 else (None, None)
    )
    methods = ("kill", "poll", "wait")
    if isinstance(sock, socket.socket) and all(
        callable(getattr(process, name, None)) for name in methods
    ):
        return process, sock
    with contextlib.suppress(Exception):
        process.kill()
    raise TypeError(
        f"--pack-launcher {args.pack_launcher or 'launch'} returned {returned!r}, not a "
        "process object (with kill, poll and wait) and a socket"
    )


@functools.cache
def resolve(name: str):
    """The function `module:function` names; raises `RuntimeError` saying why it cannot be
    used."""
    module_name, _, function_name = name.partition(":")
    try:
        if not module_name or not function_name:
            raise ValueError("expected module:function")
        function = getattr(importlib.import_module(module_name), function_name)
        if not callable(function):
            raise TypeError(f"{function_name} is not callable")
    except Exception as error:
        raise RuntimeError(
            f"--pack-launcher {name}: cannot use it as the pack launcher: "
            f"{type(error).__name__}: {error}"
        ) from error
    return function
