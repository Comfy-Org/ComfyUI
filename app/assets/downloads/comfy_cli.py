"""comfy-cli as both the download worker and the download journal.

``comfy model download --background`` already detaches a worker, records its
progress in ``<workspace>/.comfy-downloads/<id>.json`` under a documented
``download-state/1`` schema, throttles those writes to 1s, reconciles workers
that died, and cancels by sentinel plus process group (BE-4759). ComfyUI reuses
all of it rather than growing a second downloader and a second task registry.

Everything specific to that CLI -- argv, the ``--json`` envelope, the journal
layout, the foreground/background distinction -- is confined to this module.
Above it only :class:`~app.assets.downloads.backend.DownloadSnapshot` is
visible.

The CLI is invoked, not imported: it is a separate distribution with its own
dependency set, which ComfyUI does not require.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from typing import Any

from app.assets.downloads.backend import (
    CancelOutcome,
    DownloadBackendUnavailable,
    DownloadPhase,
    DownloadRejected,
    DownloadRequest,
    DownloadSnapshot,
)

JOURNAL_DIRNAME = ".comfy-downloads"

_PHASES = {
    "starting": DownloadPhase.PENDING,
    "downloading": DownloadPhase.TRANSFERRING,
    "completed": DownloadPhase.TRANSFERRED,
    "failed": DownloadPhase.FAILED,
    "cancelled": DownloadPhase.CANCELLED,
}

# A metadata round trip to Hugging Face or CivitAI happens in the foreground of
# `model download`, before the worker detaches, so submission is the one call
# that can legitimately take a while. The read-only verbs are local.
_SUBMIT_TIMEOUT_S = 120.0
_QUERY_TIMEOUT_S = 30.0

# How often the CLI is asked to enumerate rather than the journal being read
# directly. Only the CLI reconciles a worker that died and prunes finished
# records; progress itself is already in the journal, so paying for a
# subprocess at the progress cadence buys nothing.
_RECONCILE_INTERVAL_S = 5.0

# How long a failed executable lookup is trusted. comfy-cli is a separate
# install, so "not there" is a normal state that can change under a running
# server without re-probing the PATH on every sweep.
_LOOKUP_TTL_S = 60.0


class ComfyCliBackend:
    """Drives ``comfy model download`` and reads its journal."""

    key = "comfy-cli-model-download"

    def __init__(self, workspace: str, executable: str | None = None):
        self.workspace = os.path.abspath(workspace)
        self._executable = executable
        self._looked_up_at: float | None = None
        self._reconciled_at: float | None = None
        # `model downloads` prunes finished records, so two of those would race
        # over the same directory. Only that verb is serialised: holding the
        # lock across a submit would stall progress reads and cancels behind a
        # metadata round trip that can take a minute.
        self._prune_lock = asyncio.Lock()
        # A submit resolves metadata in comfy-cli's foreground, so each one can
        # hold an interpreter for up to _SUBMIT_TIMEOUT_S. Queue them rather
        # than forking one per concurrent request.
        self._submits = asyncio.Semaphore(4)

    @property
    def journal_dir(self) -> str:
        return os.path.join(self.workspace, JOURNAL_DIRNAME)

    def resolve_executable(self) -> str | None:
        if self._executable:
            return self._executable
        if self._looked_up_at is not None and time.monotonic() - self._looked_up_at < _LOOKUP_TTL_S:
            return None
        self._looked_up_at = time.monotonic()
        # The common install puts comfy-cli in the same environment as ComfyUI,
        # which is not necessarily on PATH when ComfyUI was launched by
        # interpreter path.
        found = shutil.which("comfy") or shutil.which("comfy", path=os.path.dirname(sys.executable))
        self._executable = found
        return found

    async def available(self) -> bool:
        return self.resolve_executable() is not None

    async def start(self, request: DownloadRequest) -> DownloadSnapshot:
        # comfy-cli joins --relative-path onto its workspace with pathlib, so an
        # absolute path replaces the workspace entirely. That is how ComfyUI's
        # folder_paths directory wins over the CLI's own idea of where models
        # live (BE-10028). Omitting --filename lets the CLI resolve a name only
        # the source knows, without giving up the directory.
        args = ["model", "download", "--url", request.url, "--relative-path", request.directory]
        if request.filename:
            args += ["--filename", request.filename]
        async with self._submits:
            envelope = await self._run(*args, "--background", timeout=_SUBMIT_TIMEOUT_S)
        handle = (envelope.get("data") or {}).get("download_id")
        if not handle:
            raise DownloadRejected("DOWNLOAD_NOT_STARTED", "comfy-cli did not report a download id.")

        record = await asyncio.to_thread(self._read_journal, handle)
        if record is None:
            raise DownloadRejected("DOWNLOAD_NOT_STARTED", f"comfy-cli started {handle} but wrote no journal entry.")
        return _snapshot(record)

    async def list(self) -> list[DownloadSnapshot]:
        if self._reconciled_at is not None and time.monotonic() - self._reconciled_at < _RECONCILE_INTERVAL_S:
            records = await asyncio.to_thread(self._read_all_journals)
            return [_snapshot(record) for record in records]

        async with self._prune_lock:
            self._reconciled_at = time.monotonic()
            envelope = await self._run("model", "downloads", timeout=_QUERY_TIMEOUT_S)
        rows = (envelope.get("data") or {}).get("downloads") or []
        records = await asyncio.to_thread(self._read_journals, [row.get("id") for row in rows])
        return [_snapshot({**records.get(row.get("id"), {}), **row}) for row in rows if row.get("id")]

    async def get(self, handle: str) -> DownloadSnapshot | None:
        try:
            envelope = await self._run("model", "download-status", handle, timeout=_QUERY_TIMEOUT_S)
        except DownloadRejected as e:
            if e.code == "download_not_found":
                return None
            raise
        row = envelope.get("data") or {}
        record = await asyncio.to_thread(self._read_journal, handle)
        return _snapshot({**(record or {}), **row})

    async def cancel(self, handle: str) -> CancelOutcome:
        snapshot = await self.get(handle)
        if snapshot is None:
            return CancelOutcome.MISSING
        # Refusing from a terminal phase here, rather than letting the CLI
        # report "already finished" as success, is what stops the browser from
        # re-offering Cancel on a row that can never change again.
        if snapshot.is_terminal or not snapshot.cancellable:
            return CancelOutcome.NOT_CANCELLABLE
        try:
            await self._run("model", "download-cancel", handle, timeout=_QUERY_TIMEOUT_S)
        except DownloadRejected as e:
            if e.code == "download_not_found":
                return CancelOutcome.MISSING
            if e.code == "model_download_foreground_cancel":
                return CancelOutcome.NOT_CANCELLABLE
            raise
        return CancelOutcome.CANCELLING

    async def _run(self, *args: str, timeout: float) -> dict[str, Any]:
        executable = self.resolve_executable()
        if executable is None:
            raise DownloadBackendUnavailable(
                "comfy-cli is not installed, so this server cannot download models. "
                "Install it with `pip install comfy-cli`."
            )
        argv = [executable, "--json", "--skip-prompt", "--workspace", self.workspace, *args]
        stdout, stderr, code = await self._communicate(argv, timeout)

        envelope = _parse_envelope(stdout)
        if envelope is None:
            detail = stderr.strip().splitlines()[-1] if stderr.strip() else f"exit code {code}"
            raise DownloadRejected("DOWNLOAD_BACKEND_ERROR", f"comfy-cli {' '.join(args[:2])} failed: {detail}")
        if not envelope.get("ok"):
            error = envelope.get("error") or {}
            raise DownloadRejected(
                error.get("code") or "DOWNLOAD_BACKEND_ERROR",
                error.get("message") or f"comfy-cli {' '.join(args[:2])} failed.",
                error.get("hint"),
            )
        return envelope

    async def _communicate(self, argv: list[str], timeout: float) -> tuple[str, str, int | None]:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.workspace,
        )
        try:
            out, err = await asyncio.wait_for(process.communicate(), timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            process.kill()
            await process.wait()
            raise
        return out.decode("utf-8", "replace"), err.decode("utf-8", "replace"), process.returncode

    def _read_journal(self, handle: str) -> dict[str, Any] | None:
        path = os.path.join(self.journal_dir, f"{handle}.json")
        try:
            with open(path, encoding="utf-8") as f:
                record = json.load(f)
        except (OSError, ValueError):
            return None
        return record if isinstance(record, dict) else None

    def _read_all_journals(self) -> list[dict[str, Any]]:
        try:
            names = [e.name for e in os.scandir(self.journal_dir) if e.name.endswith(".json")]
        except FileNotFoundError:
            return []
        # Any other OSError means the journal could not be read, not that it is
        # empty. Returning [] would retire every tracked task and make the route
        # answer an authoritative 404, which the frontend reads as proof the
        # download is gone.
        found = self._read_journals([name[: -len(".json")] for name in names])
        return [record for record in found.values() if record.get("id")]

    def _read_journals(self, handles: list[str | None]) -> dict[str, dict[str, Any]]:
        found = {}
        for handle in handles:
            record = self._read_journal(handle) if handle else None
            if record is not None:
                found[handle] = record
        return found

    def change_hint(self) -> tuple[int, int] | None:
        """Record count and newest write time across the journal.

        Spawning a process every second to learn that nothing is downloading is
        the cost this avoids: a record appearing, disappearing or being updated
        all move this value, and a worker rewrites its own file as it goes.
        """
        try:
            entries = list(os.scandir(self.journal_dir))
        except FileNotFoundError:
            # No journal yet is a definite answer, not a failed probe: comfy-cli
            # only creates the directory on its first write. Reporting None here
            # would mean "cannot tell", and an install that has never downloaded
            # anything would enumerate forever.
            return (0, 0)
        except OSError:
            return None
        newest = 0
        for entry in entries:
            if entry.name.endswith(".json"):
                try:
                    newest = max(newest, entry.stat().st_mtime_ns)
                except OSError:
                    continue
        return len(entries), newest


def _parse_envelope(stdout: str) -> dict[str, Any] | None:
    """Last JSON object on stdout. In ``--json`` mode that is the envelope;
    ``--json-stream`` would put event lines before it."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        if isinstance(parsed, dict) and parsed.get("type") == "envelope":
            return parsed
    return None


def _snapshot(record: dict[str, Any]) -> DownloadSnapshot:
    status = record.get("status")
    phase = _PHASES.get(status)
    if phase is None:
        logging.warning("Unrecognised comfy-cli download status %r; treating as failed.", status)
        phase = DownloadPhase.FAILED
    now = datetime.now(timezone.utc)
    started = _timestamp(record.get("started_at")) or now
    return DownloadSnapshot(
        handle=record["id"],
        phase=phase,
        destination=record.get("dest") or "",
        bytes_completed=int(record.get("completed_bytes") or 0),
        bytes_total=record.get("total_bytes"),
        started_at=started,
        updated_at=_timestamp(record.get("updated_at")) or started,
        error=record.get("error"),
        # A foreground record's pid is a user's own CLI process sharing a
        # terminal's process group, which comfy-cli refuses to signal.
        cancellable=record.get("kind", "background") == "background",
    )


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
