import codecs
import os
import re
import selectors
import signal
import subprocess
import time


LOG_LIMIT = 32768
LINE_LIMIT = 8192
STEP = re.compile(r"^step ([0-9]+)/([0-9]+):")


def _stop(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("TensorFold could not stop its worker after SIGKILL. Restart ComfyUI before sampling again.") from error


def run_native(command, *, cwd, interrupt, progress, environment=None, timeout=21600):
    interrupt()
    if os.name != "posix":
        raise RuntimeError("TensorFold media subprocesses currently require Linux.")
    env = os.environ.copy()
    if environment:
        env.update(environment)
    process = subprocess.Popen(
        command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True, bufsize=0,
    )
    tail = bytearray()
    pending = ""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    deadline = time.monotonic() + timeout
    try:
        with selectors.DefaultSelector() as selector:
            os.set_blocking(process.stdout.fileno(), False)
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                interrupt()
                if time.monotonic() >= deadline:
                    raise TimeoutError("TensorFold generation exceeded its execution deadline.")
                events = selector.select(0.1)
                if not events and process.poll() is not None:
                    break
                for key, _ in events:
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    tail.extend(chunk)
                    del tail[:-LOG_LIMIT]
                    pending += decoder.decode(chunk)
                    lines = pending.split("\n")
                    pending = lines.pop()[-LINE_LIMIT:]
                    for line in lines:
                        match = STEP.match(line)
                        if match is not None:
                            current, total = map(int, match.groups())
                            if 0 < current <= total <= 100000:
                                progress(current, total)
            while process.poll() is None:
                interrupt()
                if time.monotonic() >= deadline:
                    raise TimeoutError("TensorFold generation exceeded its execution deadline.")
                try:
                    process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    continue
        if process.returncode != 0:
            detail = tail.decode("utf-8", errors="replace")[-4096:].strip()
            raise RuntimeError(f"TensorFold exited with status {process.returncode}: {detail}")
        interrupt()
        return tail.decode("utf-8", errors="replace")
    finally:
        try:
            _stop(process)
        finally:
            process.stdout.close()
