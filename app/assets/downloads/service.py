"""Projects downloads onto the task API the browser already speaks.

The frontend has one contract for background work: ``GET``/``DELETE
/api/tasks/{id}`` plus ``asset_download`` websocket events. It uses that
contract against Cloud today. This service makes local model downloads arrive
through the same door, so the browser never learns that a download is involved
in anything beyond the task's name.

Nothing here is persisted. Task ids are *derived* from the backend's own
identifiers, so a ComfyUI restart rebuilds exactly the same ids by enumerating
the backend again -- which is what lets comfy-cli stay the only journal.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from app.assets.downloads import destination
from app.assets.downloads.backend import (
    CancelOutcome,
    DownloadBackend,
    DownloadPhase,
    DownloadRequest,
    DownloadSnapshot,
)

TASK_NAME = "download_file"
WS_EVENT = "asset_download"

# Fixed namespace so a download keeps its task id across restarts, releases and
# machines. Never change it: every id this server has ever handed out is derived
# from it.
_TASK_NAMESPACE = uuid.UUID("6f9b4b7e-9d1f-5a3e-9a1d-2f1c7a2b5c84")

_ACTIVE_SWEEP_S = 1.0
_IDLE_SWEEP_S = 2.0

_STATUSES = {
    DownloadPhase.PENDING: "created",
    DownloadPhase.TRANSFERRING: "running",
    # Still running as far as the task API is concerned: the bytes have landed
    # but nothing has yet proved a loader can see them. _finalize decides.
    DownloadPhase.TRANSFERRED: "running",
    DownloadPhase.FAILED: "failed",
    DownloadPhase.CANCELLED: "cancelled",
}


@dataclass
class _Tracked:
    task_id: str
    snapshot: DownloadSnapshot
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None
    announced: bool = False


class DownloadTaskService:
    def __init__(
        self,
        backend: DownloadBackend,
        notify: Callable[[str, dict[str, Any]], None],
        refresh_catalog: Callable[[], None] | None = None,
    ):
        self._backend = backend
        self._notify = notify
        self._refresh_catalog = refresh_catalog
        self._tracked: dict[str, _Tracked] = {}
        self._by_task: dict[str, str] = {}
        self._sweeper: asyncio.Task | None = None
        self._sweeping: asyncio.Task | None = None
        self._change_hint: object | None = None
        self._swept_once = False
        self._last_swept_at = 0.0
        self._last_failure: str | None = None
        self._last_sweep_error: Exception | None = None

    async def start(self, url: str, folder_name: str, filename: str | None) -> dict[str, Any]:
        """Submit a download and return its task, ready for a 202 response."""
        if filename:
            path = destination.resolve(folder_name, filename)
            request = DownloadRequest(url, os.path.dirname(path), os.path.basename(path))
        else:
            request = DownloadRequest(url, destination.directory(folder_name))
        snapshot = await self._backend.start(request)
        tracked = self._track(snapshot)
        # The submitter is the one client guaranteed to be listening, and the
        # first sweep is up to a second away.
        self._emit(tracked)
        return {"task_id": tracked.task_id, "status": tracked.status}

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        handle = await self._resolve(task_id)
        tracked = self._tracked.get(handle) if handle else None
        return _task_response(tracked) if tracked else None

    async def cancel_task(self, task_id: str) -> CancelOutcome:
        handle = await self._resolve(task_id)
        if handle is None:
            return CancelOutcome.MISSING
        outcome = await self._backend.cancel(handle)
        if outcome is CancelOutcome.CANCELLING:
            await self.sweep()
        return outcome

    async def _resolve(self, task_id: str) -> str | None:
        """Find the download a task id names, enumerating once if it is unknown.

        An unknown id is the normal way a task started before this process -- or
        by another client -- is discovered, so it has to trigger a lookup. The
        floor keeps a client polling an id that will never exist from spawning
        an enumeration per request.
        """
        handle = self._by_task.get(task_id)
        if handle is not None:
            return handle
        if time.monotonic() - self._last_swept_at >= _ACTIVE_SWEEP_S:
            await self.sweep()
            handle = self._by_task.get(task_id)
            if handle is not None:
                return handle
        # Only a transport that answered can say a task does not exist. The
        # frontend treats 404 as proof the row is gone and settles a pending
        # cancellation on it, so guessing that while the backend is down would
        # retire a download that is still running.
        if self._last_sweep_error is not None:
            raise self._last_sweep_error
        return None

    def start_sweeping(self) -> None:
        if self._sweeper is None or self._sweeper.done():
            self._sweeper = asyncio.create_task(self._sweep_forever())

    async def stop_sweeping(self) -> None:
        running = [t for t in (self._sweeper, self._sweeping) if t is not None]
        self._sweeper = self._sweeping = None
        for task in running:
            task.cancel()
        for task in running:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    async def sweep(self) -> None:
        """Re-read every download the backend knows about and publish changes.

        Callers arriving while one is already running join it rather than
        starting another: an enumeration spawns a comfy-cli process, and the
        unknown-task lookup is reachable from every request, so without this a
        burst of requests becomes a burst of subprocesses. The shield keeps one
        caller's cancellation from aborting the sweep the others are waiting on.
        """
        if self._sweeping is None or self._sweeping.done():
            self._sweeping = asyncio.create_task(self._sweep_once())
        await asyncio.shield(self._sweeping)

    async def _sweep_once(self) -> None:
        """Downloads this ComfyUI never started are included on purpose: the
        local agent runs ``comfy model download`` itself, and the complaint this
        feature answers (PM-1883) is precisely that those are invisible."""
        # Stamped before the work as well as after, so requests arriving during
        # a slow enumeration see a recent sweep rather than queueing their own.
        self._last_swept_at = time.monotonic()
        try:
            # Read before enumerating, so a record written while we enumerate
            # is not mistaken for one this sweep already covered.
            hint = self._backend.change_hint()
            known = set(self._tracked)
            snapshots = await self._backend.list()
            listed = set()
            for snapshot in snapshots:
                listed.add(snapshot.handle)
                await self._update(snapshot)
            for handle in known - listed:
                self._forget(handle)
        except Exception as e:
            self._last_sweep_error = e
            raise
        finally:
            self._last_swept_at = time.monotonic()
        self._last_sweep_error = None
        self._change_hint = hint
        self._swept_once = True

    async def _sweep_forever(self) -> None:
        while True:
            try:
                if await self._backend.available() and self._should_sweep():
                    await self.sweep()
                self._last_failure = None
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # A broken transport fails identically every tick, so only the
                # first of a run is worth a stack trace.
                if str(e) != self._last_failure:
                    self._last_failure = str(e)
                    logging.exception("Model download sweep failed; retrying.")
            await asyncio.sleep(_ACTIVE_SWEEP_S if self._has_active() else _IDLE_SWEEP_S)

    def _should_sweep(self) -> bool:
        """Skip the enumeration when nothing can have changed.

        An idle server would otherwise spawn comfy-cli once a second forever
        just to be told there is nothing to report.
        """
        if self._has_active() or not self._swept_once:
            return True
        hint = self._backend.change_hint()
        return hint is None or hint != self._change_hint

    def _has_active(self) -> bool:
        return any(not t.snapshot.is_terminal for t in self._tracked.values())

    def _track(self, snapshot: DownloadSnapshot) -> _Tracked:
        tracked = _Tracked(
            task_id=self._task_id(snapshot),
            snapshot=snapshot,
            status=_STATUSES[snapshot.phase],
        )
        self._tracked[snapshot.handle] = tracked
        self._by_task[tracked.task_id] = snapshot.handle
        return tracked

    def _forget(self, handle: str) -> None:
        tracked = self._tracked.pop(handle, None)
        if tracked is not None:
            # No tombstone: once the backend has pruned a record, GET must 404.
            # The frontend treats that as proof the task is gone and settles a
            # pending cancellation on it.
            self._by_task.pop(tracked.task_id, None)

    def _task_id(self, snapshot: DownloadSnapshot) -> str:
        identity = f"{self._backend.key}:{snapshot.handle}:{snapshot.started_at.isoformat()}"
        return str(uuid.uuid5(_TASK_NAMESPACE, identity))

    async def _update(self, snapshot: DownloadSnapshot) -> None:
        tracked = self._tracked.get(snapshot.handle)
        adopted = tracked is None
        if adopted:
            tracked = self._track(snapshot)
        previous = (tracked.status, tracked.snapshot.bytes_completed, tracked.snapshot.bytes_total)
        tracked.snapshot = snapshot

        if snapshot.phase is DownloadPhase.TRANSFERRED:
            if tracked.status not in ("completed", "failed", "cancelled"):
                await self._finalize(tracked, publish=not adopted or self._swept_once)
        else:
            tracked.status = _STATUSES[snapshot.phase]
            tracked.error = snapshot.error

        if adopted and snapshot.is_terminal and not self._swept_once:
            # Retained history, not news. comfy-cli keeps finished records for a
            # week, and announcing them at startup would reopen a toast for
            # every download the user has ever run. A record first seen after
            # that is a download someone really did start, often one that
            # failed fast, and staying silent about it is the bug this feature
            # exists to fix.
            tracked.announced = True
            return

        changed = (tracked.status, snapshot.bytes_completed, snapshot.bytes_total) != previous
        if changed or not tracked.announced or not snapshot.is_terminal:
            self._emit(tracked)

    async def _finalize(self, tracked: _Tracked, *, publish: bool) -> None:
        """Decide whether a finished transfer is actually a finished download.

        The bytes arriving is not the promise this API makes. BE-10028 is the
        case where they arrive somewhere the running ComfyUI cannot see, and a
        progress UI that reported 100% for a file no loader lists would be
        worse than no progress UI at all.
        """
        path = tracked.snapshot.destination
        seen = await asyncio.to_thread(destination.inspect, path)
        if not seen.ok:
            tracked.status = "failed"
            tracked.error = f"The download finished but this server's model loaders cannot use it: {seen.problem}."
            return

        if publish:
            # Records retained from previous runs still need a status, but
            # their catalog refresh happened long ago.
            destination.refresh_listing(seen.folder)
        tracked.status = "completed"
        tracked.error = None
        tracked.result = {
            "success": True,
            "file_path": path,
            "filename": (seen.name or os.path.basename(path)).replace(os.sep, "/"),
            "bytes_downloaded": tracked.snapshot.bytes_completed,
        }
        if publish and self._refresh_catalog is not None:
            self._refresh_catalog()

    def _emit(self, tracked: _Tracked) -> None:
        snapshot = tracked.snapshot
        total = snapshot.bytes_total or 0
        # A fraction, not a percentage: Cloud's AssetDownloadMessage documents
        # `progress` as 0.0-1.0 and the toast renders `progress * 100`.
        if tracked.status == "completed":
            progress = 1.0
        elif total > 0:
            progress = min(1.0, round(snapshot.bytes_completed / total, 4))
        else:
            progress = 0.0
        payload = {
            "task_id": tracked.task_id,
            "asset_name": os.path.basename(snapshot.destination),
            "bytes_total": total,
            "bytes_downloaded": snapshot.bytes_completed,
            "progress": progress,
            "status": tracked.status,
        }
        if tracked.error:
            payload["error"] = tracked.error
        tracked.announced = True
        self._notify(WS_EVENT, payload)


def _task_response(tracked: _Tracked) -> dict[str, Any]:
    snapshot = tracked.snapshot
    folder = destination.folder_for_path(snapshot.destination)
    response = {
        "id": tracked.task_id,
        "task_name": TASK_NAME,
        "idempotency_key": snapshot.handle,
        # Deliberately no url: a resolved download url can carry a presigned
        # token, and this is read by the browser.
        "payload": {"destination": snapshot.destination, "folder": folder},
        "status": tracked.status,
        "create_time": _rfc3339(snapshot.started_at),
        "update_time": _rfc3339(snapshot.updated_at),
    }
    if snapshot.phase is not DownloadPhase.PENDING:
        response["started_at"] = _rfc3339(snapshot.started_at)
    if tracked.status in ("completed", "failed", "cancelled"):
        response["completed_at"] = _rfc3339(snapshot.updated_at)
    if tracked.result is not None:
        response["result"] = tracked.result
    if tracked.error:
        response["error_message"] = tracked.error
    return response


def _rfc3339(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
