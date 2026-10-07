"""Preview generators by MIME type, and the worker pool that runs them.

Custom nodes register generators through ``comfy_api``; Core's own are fallbacks
that a registration for the same MIME type replaces. Workers are daemon threads, not
a ThreadPoolExecutor, whose workers are joined at exit: a generator that never
returns must not block shutdown.
"""

import logging
import queue
import threading
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from comfy_api.latest._previews import PreviewGenerator

PREVIEW_WORKERS = 2

_registered: dict[str, "PreviewGenerator"] = {}
_core: dict[str, "PreviewGenerator"] = {}
_lock = threading.Lock()

_jobs: "queue.SimpleQueue[tuple[Callable[[], Any], Future]]" = queue.SimpleQueue()
_workers_started = False


def register_preview_generator(generator: "PreviewGenerator") -> None:
    with _lock:
        for mime_type in generator.mime_types:
            replaced = _registered.get(mime_type)
            if replaced is not None and replaced is not generator:
                logging.warning("Preview generator for %s replaced by %r", mime_type, generator)
            _registered[mime_type] = generator


def unregister_preview_generator(generator: "PreviewGenerator") -> None:
    with _lock:
        for mime_type in [m for m, g in _registered.items() if g is generator]:
            del _registered[mime_type]


def set_core_preview_generator(generator: "PreviewGenerator") -> None:
    with _lock:
        for mime_type in generator.mime_types:
            _core[mime_type] = generator


def get_preview_generator(mime_type: str | None) -> "PreviewGenerator | None":
    if not mime_type:
        return None
    return _registered.get(mime_type) or _core.get(mime_type)


def preview_deadline_seconds(count: int) -> float:
    """How long a caller waits for ``count`` previews before giving up on the rest."""
    return min(5.0 + 0.5 * count, 30.0)


def _work() -> None:
    while True:
        fn, future = _jobs.get()
        if not future.set_running_or_notify_cancel():
            continue
        try:
            future.set_result(fn())
        except BaseException as exc:
            future.set_exception(exc)


def submit_preview_job(fn: Callable[[], Any]) -> Future:
    """Run ``fn`` on a preview worker. Cancel the future to drop it if it hasn't started."""
    global _workers_started
    with _lock:
        if not _workers_started:
            for i in range(PREVIEW_WORKERS):
                threading.Thread(target=_work, name=f"preview-worker-{i}", daemon=True).start()
            _workers_started = True
    future: Future = Future()
    _jobs.put((fn, future))
    return future
