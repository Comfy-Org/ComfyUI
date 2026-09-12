from __future__ import annotations

import atexit
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
import psutil

from comfy.cli_args import args


_GIB = 1024 ** 3
_MIB = 1024 ** 2
_PREFIX = "[Model ReadAhead]"
_EXTENSIONS = (".safetensors", ".sft")
_MIN_FILE_BYTES = 2 * _GIB
_CHUNK_BYTES = 32 * _MIB
_BASE_RESERVE_BYTES = 3 * _GIB
_MODEL_RESERVE_RATIO = 0.50
_MODEL_RESERVE_MIN_BYTES = 2 * _GIB
_MODEL_RESERVE_MAX_BYTES = 7 * _GIB
_MIN_BUDGET_BYTES = 1 * _GIB
_STOP_TIMEOUT_S = 1.0
_PROCESS_WAIT_S = 0.35
_MONITOR_INTERVAL_S = 0.05


@dataclass(frozen=True)
class _Request:
    generation: int
    path: str
    size: int
    budget: int
    reserve: int


def _available_memory() -> int | None:
    try:
        return int(psutil.virtual_memory().available)
    except (OSError, RuntimeError, psutil.Error) as exc:
        logging.warning("%s RAM query failed; skipping read-ahead: %s", _PREFIX, exc)
        return None


def _model_reserve(file_size: int) -> int:
    reserve = int(file_size * _MODEL_RESERVE_RATIO)
    reserve = max(_MODEL_RESERVE_MIN_BYTES, reserve)
    return min(_MODEL_RESERVE_MAX_BYTES, reserve)


class _SequentialReadAhead:
    """Warm large model files into the OS file cache without blocking model loading.

    The actual sequential read runs in a short-lived child Python process so it can be
    terminated independently when sampling starts, a newer model is requested, RAM
    pressure rises, or ComfyUI shuts down.
    """

    _CHILD_READER_CODE = r"""
import sys
path = sys.argv[1]
chunk = int(sys.argv[2])
limit = int(sys.argv[3])
buf = bytearray(chunk)
total = 0
with open(path, "rb", buffering=0) as handle:
    while total < limit:
        want = min(chunk, limit - total)
        count = handle.readinto(memoryview(buf)[:want])
        if not count:
            break
        total += count
print(total, flush=True)
"""

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._pending: _Request | None = None
        self._active: _Request | None = None
        self._active_process: subprocess.Popen[str] | None = None
        self._active_process_generation: int | None = None
        self._generation = 0
        self._cancel_generation: int | None = None
        self._cancel_reason = ""
        self._shutdown_requested = False
        self._worker_thread = threading.Thread(
            target=self._worker,
            name="Comfy-Model-ReadAhead",
            daemon=True,
        )
        self._worker_thread.start()

    @staticmethod
    def _candidate(path: str) -> tuple[str, int] | None:
        path = os.path.abspath(path)
        if not path.lower().endswith(_EXTENSIONS):
            return None
        try:
            size = os.path.getsize(path)
        except OSError:
            return None
        if size < _MIN_FILE_BYTES:
            return None
        return path, size

    @staticmethod
    def _plan(size: int) -> tuple[int, int, int] | None:
        available = _available_memory()
        if available is None:
            return None
        reserve = _BASE_RESERVE_BYTES + _model_reserve(size)
        budget = min(size, max(0, available - reserve))
        if budget < size:
            budget = (budget // _CHUNK_BYTES) * _CHUNK_BYTES
        if budget < _MIN_BUDGET_BYTES:
            return None
        return budget, reserve, available

    def request(self, path: str) -> bool:
        candidate = self._candidate(path)
        if candidate is None:
            return False
        path, size = candidate
        plan = self._plan(size)
        if plan is None:
            return False
        budget, reserve, available = plan
        normalized = os.path.normcase(path)

        with self._cv:
            if self._shutdown_requested:
                return False
            if self._active is not None and os.path.normcase(self._active.path) == normalized:
                return False
            if self._pending is not None and os.path.normcase(self._pending.path) == normalized:
                return False

            # A newer model request supersedes any currently active read. The worker
            # notices this immediately and the tracked child process can be terminated.
            if self._active is not None:
                self._cancel_generation = self._active.generation
                self._cancel_reason = "newer model requested"

            self._generation += 1
            self._pending = _Request(self._generation, path, size, budget, reserve)
            process = self._matching_active_process_locked(self._cancel_generation)
            self._cv.notify_all()

        if process is not None:
            self._terminate_process(process)

        logging.info(
            "%s queued %s: file %.2f GiB, budget %.2f GiB, RAM reserve %.2f GiB (avail %.2f GiB)",
            _PREFIX,
            os.path.basename(path),
            size / _GIB,
            budget / _GIB,
            reserve / _GIB,
            available / _GIB,
        )
        return True

    def _matching_active_process_locked(self, generation: int | None) -> subprocess.Popen[str] | None:
        if generation is None:
            return None
        if self._active_process_generation != generation:
            return None
        return self._active_process

    def _cancel_active(self, reason: str, timeout_s: float, *, clear_pending: bool) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_s)

        with self._cv:
            if clear_pending:
                self._pending = None
            active = self._active
            if active is None:
                return True
            generation = active.generation
            self._cancel_generation = generation
            self._cancel_reason = reason
            process = self._matching_active_process_locked(generation)
            self._cv.notify_all()

        # Do not hold the condition while terminating. The worker needs the same
        # condition to clear its active state.
        if process is not None:
            self._terminate_process(process)

        while True:
            with self._cv:
                if self._active is None or self._active.generation != generation:
                    return True
                remaining = deadline - time.monotonic()
                process = self._matching_active_process_locked(generation)
                if remaining <= 0:
                    break

            # Handles the race where cancellation was requested just before Popen()
            # became visible to the controller.
            if process is not None:
                self._terminate_process(process)

            with self._cv:
                if self._active is None or self._active.generation != generation:
                    return True
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    self._cv.wait(timeout=min(_MONITOR_INTERVAL_S, remaining))

        if process is not None:
            self._terminate_process(process)

        # Give the worker one final short window to observe the terminated process and
        # release its active state. Never block model execution indefinitely.
        final_deadline = time.monotonic() + _PROCESS_WAIT_S
        with self._cv:
            while self._active is not None and self._active.generation == generation:
                remaining = final_deadline - time.monotonic()
                if remaining <= 0:
                    logging.warning("%s helper did not stop after forced cancellation (%s)", _PREFIX, reason)
                    return False
                self._cv.wait(timeout=remaining)
        return True

    def stop_for_sampling(self, timeout_s: float = _STOP_TIMEOUT_S) -> bool:
        return self._cancel_active("sampling starting", timeout_s, clear_pending=True)

    def shutdown(self, timeout_s: float = _STOP_TIMEOUT_S) -> bool:
        with self._cv:
            if self._shutdown_requested:
                worker = self._worker_thread
            else:
                self._shutdown_requested = True
                self._pending = None
                worker = self._worker_thread
                self._cv.notify_all()

        stopped = self._cancel_active("shutdown", timeout_s, clear_pending=True)
        if worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=max(0.0, timeout_s))
        if worker.is_alive():
            logging.warning("%s worker thread did not exit during shutdown", _PREFIX)
            return False
        return stopped

    def _cancel_state(self, request: _Request) -> tuple[bool, str]:
        with self._cv:
            if self._shutdown_requested:
                return True, "shutdown"
            if self._cancel_generation == request.generation:
                return True, self._cancel_reason or "cancelled"
            return False, ""

    def _worker(self) -> None:
        while True:
            with self._cv:
                while self._pending is None and not self._shutdown_requested:
                    self._cv.wait()
                if self._shutdown_requested and self._pending is None:
                    return
                request = self._pending
                self._pending = None
                if request is None:
                    continue
                self._active = request

            try:
                self._read_isolated(request)
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                logging.warning("%s read failed for %s: %s", _PREFIX, os.path.basename(request.path), exc)
            finally:
                with self._cv:
                    if self._active == request:
                        self._active = None
                    if self._active_process_generation == request.generation:
                        self._active_process = None
                        self._active_process_generation = None
                    if self._cancel_generation == request.generation:
                        self._cancel_generation = None
                        self._cancel_reason = ""
                    self._cv.notify_all()

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=_PROCESS_WAIT_S)
            return
        except (OSError, subprocess.TimeoutExpired):
            pass
        try:
            process.kill()
            process.wait(timeout=_PROCESS_WAIT_S)
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _read_isolated(self, request: _Request) -> None:
        cancelled, reason = self._cancel_state(request)
        if cancelled:
            logging.info("%s skip %s; %s", _PREFIX, os.path.basename(request.path), reason)
            return

        started = time.perf_counter()
        command = [
            sys.executable,
            "-c",
            self._CHILD_READER_CODE,
            request.path,
            str(_CHUNK_BYTES),
            str(request.budget),
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        logging.info(
            "%s start %s (budget %.2f/%.2f GiB)",
            _PREFIX,
            os.path.basename(request.path),
            request.budget / _GIB,
            request.size / _GIB,
        )

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=creationflags,
        )

        with self._cv:
            if self._active == request:
                self._active_process = process
                self._active_process_generation = request.generation
            self._cv.notify_all()

        stop_reason = ""
        try:
            while process.poll() is None:
                cancelled, reason = self._cancel_state(request)
                if cancelled:
                    stop_reason = reason
                    self._terminate_process(process)
                    break

                available = _available_memory()
                if available is None:
                    stop_reason = "RAM query unavailable"
                    self._terminate_process(process)
                    break
                if available < request.reserve:
                    stop_reason = (
                        f"available RAM {available / _GIB:.2f} GiB < "
                        f"{request.reserve / _GIB:.2f} GiB reserve"
                    )
                    self._terminate_process(process)
                    break
                time.sleep(_MONITOR_INTERVAL_S)

            stdout, stderr = process.communicate(timeout=1.0)
        except subprocess.TimeoutExpired:
            self._terminate_process(process)
            stdout, stderr = process.communicate()
        finally:
            with self._cv:
                if self._active_process is process:
                    self._active_process = None
                    self._active_process_generation = None
                self._cv.notify_all()

        # The controller may terminate the child directly to make cancellation
        # synchronous. If that happened between poll() iterations, recover the
        # cancellation reason before interpreting the non-zero exit as an error.
        if not stop_reason and process.returncode != 0:
            cancelled, reason = self._cancel_state(request)
            if cancelled:
                stop_reason = reason

        elapsed = max(time.perf_counter() - started, 1e-9)
        if stop_reason:
            logging.info(
                "%s stop %s after %.2fs; %s",
                _PREFIX,
                os.path.basename(request.path),
                elapsed,
                stop_reason,
            )
            return
        if process.returncode != 0:
            detail = (stderr or "").strip()[-500:]
            raise RuntimeError(f"helper reader exit={process.returncode}: {detail}")
        lines = (stdout or "").strip().splitlines()
        if not lines:
            raise RuntimeError("helper reader returned no byte count")
        try:
            total = int(lines[-1])
        except ValueError as exc:
            raise RuntimeError("helper reader returned an invalid byte count") from exc
        if total < 0 or total > request.budget:
            raise RuntimeError(f"helper reader returned invalid byte count: {total}")
        logging.info(
            "%s done %s: warmed %.2f/%.2f GiB in %.2fs (%.2f GiB/s)",
            _PREFIX,
            os.path.basename(request.path),
            total / _GIB,
            request.size / _GIB,
            elapsed,
            total / elapsed / _GIB,
        )


_reader: _SequentialReadAhead | None = None

if args.enable_model_readahead and os.name != "nt":
    logging.warning("%s --enable-model-readahead is currently supported on Windows only", _PREFIX)
elif args.enable_model_readahead:
    try:
        _reader = _SequentialReadAhead()
        logging.info("%s enabled (Windows, opt-in)", _PREFIX)
    except (OSError, RuntimeError) as exc:
        logging.warning("%s could not initialize; normal model loading will be used: %s", _PREFIX, exc)


def request(path: str) -> bool:
    if _reader is None:
        return False
    return _reader.request(path)


def stop_for_sampling() -> bool:
    if _reader is None:
        return True
    return _reader.stop_for_sampling()


def shutdown() -> bool:
    if _reader is None:
        return True
    return _reader.shutdown()


if _reader is not None:
    atexit.register(shutdown)
