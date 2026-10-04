"""Runs the filesystem scan on a background thread so startup never waits for it,
exposing pause, resume, cancel and progress to the API. A run seeds
newly-observed files first, then enriches records in batches, and settles any
pending hash-mode transition at the start of the enrich phase so a server that
receives no prompts still completes the switch. An enrichment pass ends when
its ordered candidate cursor is exhausted.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, TypedDict

from app.assets.event_log import emit, error_kind, error_type
from app.assets.scanner import (
    RootType,
    build_asset_specs,
    enrich_assets_batch,
    get_owned_prefixes,
    get_scan_prefixes_for_root,
    get_unenriched_assets_for_roots,
    insert_asset_specs,
    list_root,
    live_references_safely,
    mark_missing_outside_prefixes_safely,
    mark_unlisted_references_missing_safely,
    verifies_catalogued_files,
    sync_root_safely,
    unlisted_references,
    sync_temp_references_safely,
    drain_pending_verifications,
    tick_watch_list,
)
from app.assets.services.hash_mode_state import drain_transition_queue, pending_transition_count
from app.database.db import create_session, dependencies_available


class ScanInProgressError(Exception):
    """Raised when an operation cannot proceed because a scan is running."""


class PruneCancelledError(Exception):
    """A standalone prune stopped by a cancel. The batches before it stay committed;
    ``marked`` counts them."""

    def __init__(self, marked: int) -> None:
        super().__init__(f"prune cancelled after marking {marked}")
        self.marked = marked


class State(Enum):
    """Seeder state machine states."""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    CANCELLING = "CANCELLING"


class ScanPhase(Enum):
    """Scan phase options."""

    FAST = "fast"  # Phase 1: filesystem only (stubs)
    ENRICH = "enrich"  # Phase 2: metadata + hash
    FULL = "full"  # Both phases sequentially


class PendingScan(TypedDict):
    roots: tuple[RootType, ...]
    phase: ScanPhase
    compute_hashes: bool


class _ScanStage(Enum):
    MARK_MISSING = "mark_missing"
    PRUNING = "pruning"
    FAST_SCAN = "fast_scan"
    ENRICH = "enrich"
    FINALIZE = "finalize"


@dataclass
class Progress:
    """Public snapshot of a scan's progress. Carries counters only."""

    scanned: int = 0
    total: int = 0
    created: int = 0
    skipped: int = 0
    hash_failed: int = 0
    enrich_failed: int = 0
    permission_denied: int = 0


@dataclass
class ScanStatus:
    """Current status of the asset seeder."""

    state: State
    progress: Progress | None
    errors: list[str] = field(default_factory=list)


ProgressCallback = Callable[[Progress], None]


@dataclass
class _ScanState:
    """Mutable in-flight state for one scan; never exposed outside this module.

    Satisfies scanner.py's `_ScanProgress` Protocol. `cancel_stage` stores the
    stage's string value (not the `_ScanStage` enum) so scanner.py never needs
    to import `_ScanStage`. Take a `Progress` snapshot before exposing state.
    """

    scanned: int = 0
    total: int = 0
    created: int = 0
    skipped: int = 0
    hash_failed: int = 0
    enrich_failed: int = 0
    permission_denied: int = 0
    # Rows the fast scan marked missing because their file was not found. Pruning reports
    # its own count, and temp retirement is routine, so neither is included.
    missing_marked: int = 0
    recovered: int = 0
    # Directories the input/output walks (or the output rescan's listing) listed; the
    # models listing is not counted.
    dirs_listed: int = 0
    # os.stat calls on files in the reference sync, output-listing check, discovery,
    # admission, seed, watch-list and enrich loops. Hashing's own stability stats are
    # not counted.
    files_statted: int = 0
    # Time blocked at the pause gate.
    paused_s: float = 0.0
    cancel_stage: str | None = None
    _emitted_keys: set[str] = field(default_factory=set)

    def mark_emitted(self, key: str) -> bool:
        """Return True the first call with `key` this scan, False every call after."""
        if key in self._emitted_keys:
            return False
        self._emitted_keys.add(key)
        return True


def _snapshot_progress(state: _ScanState) -> Progress:
    """Build the public counters-only `Progress` snapshot from live scan state."""
    return Progress(
        scanned=state.scanned,
        total=state.total,
        created=state.created,
        skipped=state.skipped,
        hash_failed=state.hash_failed,
        enrich_failed=state.enrich_failed,
        permission_denied=state.permission_denied,
    )


class _AssetSeeder:
    """Background asset scanning manager.

    Spawns ephemeral daemon threads for scanning.
    Each scan creates a new thread that exits when complete.
    Use the module-level ``asset_seeder`` instance.
    """

    def __init__(self) -> None:
        # RLock is required because _run_scan() drains pending work while
        # holding _lock and re-enters start() which also acquires _lock.
        self._lock = threading.RLock()
        self._state = State.IDLE
        self._scan_state: _ScanState | None = None
        self._last_progress: Progress | None = None
        self._errors: list[str] = []
        self._thread: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._run_gate = threading.Event()
        self._run_gate.set()  # Start unpaused (set = running, clear = paused)
        # Clear while a standalone prune holds the seeder (set = no prune running).
        self._prune_idle = threading.Event()
        self._prune_idle.set()
        # Set by shutdown(): a standalone prune that has not started by then does not.
        self._shutting_down = False
        self._roots: tuple[RootType, ...] = ()
        self._phase: ScanPhase = ScanPhase.FULL
        self._compute_hashes: bool = False
        self._prune_first: bool = False
        self._progress_callback: ProgressCallback | None = None
        self._event_sink: Callable[[str, dict[str, Any]], None] | None = None
        self._disabled: bool = False
        self._pending_scan: PendingScan | None = None

    def set_event_sink(self, sink: Callable[[str, dict[str, Any]], None] | None) -> None:
        self._event_sink = sink

    def disable(self) -> None:
        """Disable the asset seeder, preventing any scans from starting."""
        self._disabled = True
        logging.info("Asset seeder disabled")

    def is_disabled(self) -> bool:
        """Check if the asset seeder is disabled."""
        return self._disabled

    def start(
        self,
        roots: tuple[RootType, ...] = ("models", "input", "output"),
        phase: ScanPhase = ScanPhase.FULL,
        progress_callback: ProgressCallback | None = None,
        prune_first: bool = False,
        compute_hashes: bool = False,
        *,
        _start_paused: bool = False,
    ) -> bool:
        """Start a background scan for the given roots.

        Args:
            roots: Tuple of root types to scan (models, input, output)
            phase: Scan phase to run (FAST, ENRICH, or FULL for both)
            progress_callback: Optional callback called with progress updates
            prune_first: If True, prune orphaned assets before scanning
            compute_hashes: If True, compute blake3 hashes (slow)
            _start_paused: Start with phase work blocked until resume()

        Returns:
            True if scan was started, False if already running
        """
        if self._disabled:
            logging.debug("Asset seeder is disabled, skipping start")
            return False
        logging.info("Seeder start (roots=%s, phase=%s)", roots, phase.value)
        with self._lock:
            if self._state != State.IDLE:
                logging.info("Asset seeder already running, skipping start")
                return False
            self._state = State.PAUSED if _start_paused else State.RUNNING
            self._scan_state = _ScanState()
            self._errors = []
            self._roots = roots
            self._phase = phase
            self._prune_first = prune_first
            self._compute_hashes = compute_hashes
            self._progress_callback = progress_callback
            self._cancel_event.clear()
            if _start_paused:
                self._run_gate.clear()
            else:
                self._run_gate.set()
            self._thread = threading.Thread(
                target=self._run_scan,
                name="_AssetSeeder",
                daemon=True,
            )
            self._thread.start()
            return True

    def start_fast(
        self,
        roots: tuple[RootType, ...] = ("models", "input", "output"),
        progress_callback: ProgressCallback | None = None,
        prune_first: bool = False,
    ) -> bool:
        """Start a fast scan (phase 1 only) - creates stub records.

        Args:
            roots: Tuple of root types to scan
            progress_callback: Optional callback for progress updates
            prune_first: If True, prune orphaned assets before scanning

        Returns:
            True if scan was started, False if already running
        """
        return self.start(
            roots=roots,
            phase=ScanPhase.FAST,
            progress_callback=progress_callback,
            prune_first=prune_first,
            compute_hashes=False,
        )

    def enqueue_scan(
        self,
        roots: tuple[RootType, ...],
        phase: ScanPhase,
        compute_hashes: bool = False,
    ) -> bool:
        with self._lock:
            if self.start(
                roots=roots,
                phase=phase,
                prune_first=False,
                compute_hashes=compute_hashes,
            ):
                return True
            if self._pending_scan is not None:
                existing_roots = set(self._pending_scan["roots"])
                existing_roots.update(roots)
                self._pending_scan["roots"] = tuple(existing_roots)
                self._pending_scan["compute_hashes"] = (
                    self._pending_scan["compute_hashes"] or compute_hashes
                )
                if self._pending_scan["phase"] is not phase:
                    self._pending_scan["phase"] = ScanPhase.FULL
            else:
                self._pending_scan = {
                    "roots": roots,
                    "phase": phase,
                    "compute_hashes": compute_hashes,
                }
            logging.info(
                "Scan queued (roots=%s, phase=%s)",
                self._pending_scan["roots"],
                self._pending_scan["phase"].value,
            )
        return False

    def cancel(self) -> bool:
        """Request cancellation of the current scan.

        Returns:
            True if cancellation was requested, False if not running or paused
        """
        with self._lock:
            if self._state not in (State.RUNNING, State.PAUSED):
                return False
            logging.info("Asset seeder cancelling (was %s)", self._state.value)
            self._state = State.CANCELLING
            self._cancel_event.set()
            self._run_gate.set()  # Unblock if paused so thread can exit
            return True

    def stop(self) -> bool:
        """Stop the current scan (alias for cancel).

        Returns:
            True if stop was requested, False if not running
        """
        return self.cancel()

    def pause(self) -> bool:
        """Pause the current scan.

        The scan will complete its current batch before pausing.

        Returns:
            True if pause was requested, False if not running
        """
        with self._lock:
            if self._state != State.RUNNING:
                return False
            logging.info("Asset seeder pausing")
            self._state = State.PAUSED
            self._run_gate.clear()
            return True

    def resume(self) -> bool:
        """Resume a paused scan.

        This is a noop if the scan is not in the PAUSED state

        Returns:
            True if resumed, False if not paused
        """
        with self._lock:
            if self._state != State.PAUSED:
                return False
            logging.info("Asset seeder resuming")
            self._state = State.RUNNING
            self._run_gate.set()
        self._emit_event("assets.seed.resumed", {})
        return True

    def restart(
        self,
        roots: tuple[RootType, ...] | None = None,
        phase: ScanPhase | None = None,
        progress_callback: ProgressCallback | None = None,
        prune_first: bool | None = None,
        compute_hashes: bool | None = None,
        timeout: float = 5.0,
    ) -> bool:
        """Cancel any running scan and start a new one.

        Args:
            roots: Roots to scan (defaults to previous roots)
            phase: Scan phase (defaults to previous phase)
            progress_callback: Progress callback (defaults to previous)
            prune_first: Prune before scan (defaults to previous)
            compute_hashes: Compute hashes (defaults to previous)
            timeout: Max seconds to wait for current scan to stop

        Returns:
            True if new scan was started, False if failed to stop previous
        """
        logging.info("Asset seeder restart requested")
        with self._lock:
            prev_roots = self._roots
            prev_phase = self._phase
            prev_callback = self._progress_callback
            prev_prune = self._prune_first
            prev_hashes = self._compute_hashes

        self.cancel()
        if not self.wait(timeout=timeout):
            return False

        cb = progress_callback if progress_callback is not None else prev_callback
        return self.start(
            roots=roots if roots is not None else prev_roots,
            phase=phase if phase is not None else prev_phase,
            progress_callback=cb,
            prune_first=prune_first if prune_first is not None else prev_prune,
            compute_hashes=(
                compute_hashes if compute_hashes is not None else prev_hashes
            ),
        )

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for the current scan to complete.

        Args:
            timeout: Maximum seconds to wait, or None for no timeout

        Returns:
            True if scan completed, False if timeout expired or no scan running
        """
        with self._lock:
            thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def get_status(self) -> ScanStatus:
        """Get the current status and progress of the seeder."""
        with self._lock:
            progress = (
                _snapshot_progress(self._scan_state)
                if self._scan_state is not None
                else replace(self._last_progress)
                if self._last_progress is not None
                else None
            )
            return ScanStatus(
                state=self._state,
                progress=progress,
                errors=list(self._errors),
            )

    def shutdown(self, timeout: float = 5.0) -> bool:
        """Gracefully shutdown: cancel any running scan and wait for thread.

        Args:
            timeout: Maximum seconds to wait for thread to exit

        Returns:
            True if the scan thread joined cleanly; False on timeout.
        """
        with self._lock:
            self._shutting_down = True
        self.cancel()
        # A standalone prune stops at its next batch once cancelled. One deadline
        # covers both waits.
        deadline = time.monotonic() + timeout
        joined = self.wait(timeout=timeout)
        joined = self.wait_for_standalone_prune(max(0.0, deadline - time.monotonic())) and joined
        if not joined:
            logging.warning(
                "Asset seeder thread did not exit within %ss",
                timeout,
            )
        with self._lock:
            if joined:
                self._thread = None
        return joined

    def mark_missing_outside_prefixes(self) -> int | None:
        """Mark references as missing when outside all known root prefixes.

        This is a non-destructive soft-delete operation. Assets and their
        metadata are preserved, but references are flagged as missing.
        They can be restored if the file reappears in a future scan.

        This operation is decoupled from scanning to prevent partial scans
        from accidentally marking assets belonging to other roots.

        Should be called explicitly when cleanup is desired, typically after
        a full scan of all roots or during maintenance.

        Returns:
            Number of references marked as missing, or None when the marking
            itself failed. Zero and None are deliberately distinct: zero means
            nothing was outside the known prefixes, None means the answer is
            unknown, so callers must not report a failed prune as a clean one.

        Raises:
            ScanInProgressError: If a scan is currently running
            PruneCancelledError: If a cancel stopped it part way
        """
        with self._lock:
            if self._state != State.IDLE:
                raise ScanInProgressError(
                    "Cannot mark missing assets while scan is running"
                )
            if self._shutting_down:
                raise PruneCancelledError(0)
            self._state = State.RUNNING
            self._cancel_event.clear()
            self._prune_idle.clear()

        try:
            if not dependencies_available():
                logging.warning(
                    "Database dependencies not available, skipping mark missing"
                )
                return 0

            all_prefixes = get_owned_prefixes()
            # Not pausable (the API waits on it), but a cancel or shutdown stops it
            # between batches: it runs on a worker thread that exit would wait for.
            stopped = False

            def should_stop() -> bool:
                nonlocal stopped
                stopped = self._cancel_event.is_set()
                return stopped

            marked = mark_missing_outside_prefixes_safely(all_prefixes, should_stop)
            if marked is None:
                return None
            emit(
                "seeder.marked_missing",
                count=marked,
                stage=_ScanStage.MARK_MISSING.value,
            )
            if stopped:
                logging.info("Marking missing assets cancelled after marking %d", marked)
                raise PruneCancelledError(marked)
            if marked > 0:
                logging.info("Marked %d references as missing", marked)
            return marked
        finally:
            # The API runs this off the event loop, so a prompt can finish meanwhile
            # and queue its output rescan; start it now. Not after a cancel: shutdown
            # cancels, and a scan started here would run on into teardown. It stays
            # queued for the next scan to start.
            with self._lock:
                try:
                    if self._cancel_event.is_set():
                        self._reset_to_idle()
                    else:
                        self._finish_and_start_pending()
                finally:
                    self._prune_idle.set()

    def standalone_prune_running(self) -> bool:
        return not self._prune_idle.is_set()

    def wait_for_standalone_prune(self, timeout: float | None = None) -> bool:
        """Block until no standalone prune holds the seeder. True unless it timed out."""
        return self._prune_idle.wait(timeout)

    def _reset_to_idle(self) -> None:
        """Reset state to IDLE, preserving last progress. Caller must hold _lock."""
        if self._scan_state is not None:
            self._last_progress = _snapshot_progress(self._scan_state)
        self._state = State.IDLE
        self._scan_state = None

    def _is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancel_event.is_set()

    def _is_paused_or_cancelled(self) -> bool:
        """Non-blocking check: True if paused or cancelled.

        Use as interrupt_check for I/O-bound work (e.g. hashing) so that
        file handles are released immediately on pause rather than held
        open while blocked. The caller is responsible for blocking on
        _check_pause_and_cancel() afterward.
        """
        cancelled = self._cancel_event.is_set()
        if cancelled:
            self._record_cancel_stage(_ScanStage.ENRICH)
        return not self._run_gate.is_set() or cancelled

    def _record_cancel_stage(self, stage: _ScanStage) -> None:
        with self._lock:
            if self._scan_state is not None and self._scan_state.cancel_stage is None:
                self._scan_state.cancel_stage = stage.value

    def _check_pause_and_cancel(self, stage: _ScanStage) -> bool:
        """Block while paused, then check if cancelled.

        Call this at checkpoint locations in scan loops. It will:
        1. Block indefinitely while paused (until resume or cancel)
        2. Return True if cancelled, False to continue

        Returns:
            True if scan should stop, False to continue
        """
        if not self._run_gate.is_set():
            self._emit_event("assets.seed.paused", {})
        # Re-checked so a pause landing just after the check above still blocks, and is timed.
        if not self._run_gate.is_set():
            t_paused = time.perf_counter()
            self._run_gate.wait()  # Blocks until resume or cancel
            if self._scan_state is not None:
                self._scan_state.paused_s += time.perf_counter() - t_paused
        cancelled = self._is_cancelled()
        if cancelled:
            self._record_cancel_stage(stage)
        return cancelled

    def _emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit a WebSocket event if server is available."""
        try:
            if self._event_sink is not None:
                self._event_sink(event_type, data)
        except Exception:
            pass

    def _update_progress(
        self,
        scanned: int | None = None,
        total: int | None = None,
        created: int | None = None,
        skipped: int | None = None,
    ) -> None:
        """Update progress counters (thread-safe)."""
        callback: ProgressCallback | None = None
        progress: Progress | None = None

        with self._lock:
            if self._scan_state is None:
                return
            if scanned is not None:
                self._scan_state.scanned = scanned
            if total is not None:
                self._scan_state.total = total
            if created is not None:
                self._scan_state.created = created
            if skipped is not None:
                self._scan_state.skipped = skipped
            if self._progress_callback:
                callback = self._progress_callback
                progress = _snapshot_progress(self._scan_state)

        if callback and progress:
            try:
                callback(progress)
            except Exception:
                pass

    _MAX_ERRORS = 200

    def _add_error(self, message: str) -> None:
        """Add an error message (thread-safe), capped at _MAX_ERRORS."""
        with self._lock:
            if len(self._errors) < self._MAX_ERRORS:
                self._errors.append(message)

    def _log_scan_config(self, roots: tuple[RootType, ...]) -> None:
        """Log the directories that will be scanned."""
        import folder_paths

        for root in roots:
            if root == "models":
                logging.info(
                    "Asset scan [models] directory: %s",
                    os.path.abspath(folder_paths.models_dir),
                )
            else:
                prefixes = get_scan_prefixes_for_root(root)
                if prefixes:
                    logging.info("Asset scan [%s] directories: %s", root, prefixes)

    def _run_scan(self) -> None:
        """Main scan loop running in background thread."""
        t_start = time.perf_counter()
        # Per-thread CPU clock on Windows, macOS and Linux; excludes time blocked while paused.
        cpu_start = time.thread_time()
        roots = self._roots
        phase = self._phase
        root = roots[0] if len(roots) == 1 else None
        cancelled = False
        total_created = 0
        total_enriched = 0
        skipped_existing = 0
        total_paths = 0

        try:
            if not dependencies_available():
                self._add_error("Database dependencies not available")
                self._emit_event(
                    "assets.seed.error",
                    {"message": "Database dependencies not available"},
                )
                return

            emit("seeder.scan_started", phase=phase.value, root=root)
            assert self._scan_state is not None
            scan_state = self._scan_state

            if self._prune_first:
                all_prefixes = get_owned_prefixes()
                marked = mark_missing_outside_prefixes_safely(
                    all_prefixes, lambda: self._check_pause_and_cancel(_ScanStage.PRUNING)
                )
                marked_count = 0 if marked is None else marked
                if marked is None:
                    self._add_error(
                        "Marking missing assets failed; scan continued with the prune incomplete"
                    )
                else:
                    emit(
                        "seeder.marked_missing",
                        count=marked_count,
                        stage=_ScanStage.PRUNING.value,
                    )
                if marked_count > 0:
                    logging.info(
                        "Marked %d refs as missing before scan", marked_count
                    )
                sync_temp_references_safely(
                    scan_state, lambda: self._check_pause_and_cancel(_ScanStage.PRUNING)
                )

            if self._check_pause_and_cancel(_ScanStage.PRUNING):
                logging.info("Asset scan cancelled after pruning phase")
                cancelled = True
                return

            self._log_scan_config(roots)

            # Phase 1: Fast scan (stub records)
            if phase in (ScanPhase.FAST, ScanPhase.FULL):
                created, skipped, paths = self._run_fast_phase(roots)
                total_created, skipped_existing, total_paths = created, skipped, paths

                if self._check_pause_and_cancel(_ScanStage.FAST_SCAN):
                    cancelled = True
                    return

                self._emit_event(
                    "assets.seed.fast_complete",
                    {
                        "roots": list(roots),
                        "created": total_created,
                        "skipped": skipped_existing,
                        "total": total_paths,
                    },
                )

            # Phase 2: Enrichment scan (metadata + hashes)
            if phase in (ScanPhase.ENRICH, ScanPhase.FULL):
                if self._check_pause_and_cancel(_ScanStage.ENRICH):
                    cancelled = True
                    return

                enrich_cancelled, total_enriched = self._run_enrich_phase(roots)

                if enrich_cancelled:
                    cancelled = True
                    return

                self._emit_event(
                    "assets.seed.enrich_complete",
                    {
                        "roots": list(roots),
                        "enriched": total_enriched,
                    },
                )

            # Deliberately non-blocking, unlike every other checkpoint: no work
            # remains, so pausing here would hold the scan open with nothing to
            # do until main.py's next resume (a whole gc_collect_interval away).
            if self._is_cancelled():
                self._record_cancel_stage(_ScanStage.FINALIZE)
                cancelled = True
                return

            elapsed = time.perf_counter() - t_start
            cpu = time.thread_time() - cpu_start
            logging.info(
                "Scan(%s, %s) done %.3fs: created=%d enriched=%d skipped=%d",
                roots,
                phase.value,
                elapsed,
                total_created,
                total_enriched,
                skipped_existing,
            )
            emit(
                "seeder.scan_completed",
                phase=phase.value,
                elapsed_ms=round(elapsed * 1000),
                cpu_ms=round(cpu * 1000),
                paused_ms=round(scan_state.paused_s * 1000),
                dirs_listed_count=scan_state.dirs_listed,
                files_statted_count=scan_state.files_statted,
                created=total_created,
                enriched=total_enriched,
                skipped=skipped_existing,
                hash_failed=scan_state.hash_failed,
                enrich_failed=scan_state.enrich_failed,
                permission_denied=scan_state.permission_denied,
                missing_marked_count=scan_state.missing_marked,
                recovered_count=scan_state.recovered,
                root=root,
            )

            self._emit_event(
                "assets.seed.completed",
                {
                    "phase": phase.value,
                    "total": total_paths,
                    "created": total_created,
                    "enriched": total_enriched,
                    "skipped": skipped_existing,
                    "elapsed": round(elapsed, 3),
                },
            )

        except Exception as e:
            self._add_error(f"Scan failed: {e}")
            logging.exception("Asset scan failed")
            emit(
                "seeder.scan_failed",
                phase=phase.value,
                error_type=error_type(e),
                error_kind=error_kind(e),
                root=root,
            )
            self._emit_event("assets.seed.error", {"message": str(e)})
        finally:
            try:
                if cancelled:
                    stage = self._scan_state.cancel_stage if self._scan_state else None
                    if stage is not None:
                        emit(
                            "seeder.scan_cancelled",
                            phase=phase.value,
                            stage=stage,
                            root=root,
                        )
                        self._emit_event(
                            "assets.seed.cancelled",
                            {
                                "scanned": self._scan_state.scanned if self._scan_state else 0,
                                "total": total_paths,
                                "created": total_created,
                            },
                        )
            finally:
                with self._lock:
                    self._finish_and_start_pending()

    def _finish_and_start_pending(self) -> None:
        """Reset to IDLE, then start the scan queued while this run held the seeder,
        paused if this run was. Caller must hold _lock."""
        start_paused = self._state is State.PAUSED
        self._reset_to_idle()
        pending = self._pending_scan
        if pending is not None:
            self._pending_scan = None
            if not self.start(
                roots=pending["roots"],
                phase=pending["phase"],
                prune_first=False,
                compute_hashes=pending["compute_hashes"],
                _start_paused=start_paused,
            ):
                logging.warning(
                    "Pending scan could not start (roots=%s, phase=%s)",
                    pending["roots"],
                    pending["phase"].value,
                )

    @staticmethod
    def _emit_marked_missing(root: RootType, marked: int) -> None:
        """Report the rows a scan marked missing because their file was not found."""
        if marked > 0:
            emit(
                "seeder.marked_missing",
                count=marked,
                stage=_ScanStage.FAST_SCAN.value,
                root=root,
            )

    def _run_fast_phase(self, roots: tuple[RootType, ...]) -> tuple[int, int, int]:
        """Run phase 1: fast scan to create stub records.

        Returns:
            Tuple of (total_created, skipped_existing, total_paths)
        """
        t_fast_start = time.perf_counter()
        total_created = 0
        skipped_existing = 0

        should_stop = lambda: self._check_pause_and_cancel(_ScanStage.FAST_SCAN)
        verify = verifies_catalogued_files(roots)
        live_by_root: dict[RootType, dict[str, list]] = {}
        existing_paths: set[str] = set()
        t_sync = time.perf_counter()
        assert self._scan_state is not None
        scan_state = self._scan_state
        for r in roots:
            if self._check_pause_and_cancel(_ScanStage.FAST_SCAN):
                return total_created, skipped_existing, 0
            if verify:
                marked_before = scan_state.missing_marked
                existing_paths.update(sync_root_safely(r, scan_state, should_stop))
                self._emit_marked_missing(r, scan_state.missing_marked - marked_before)
            else:
                live_by_root[r] = live_references_safely(r)
                existing_paths.update(live_by_root[r])
        logging.debug(
            "Fast scan: sync_root phase took %.3fs (%d existing paths)",
            time.perf_counter() - t_sync,
            len(existing_paths),
        )

        if self._check_pause_and_cancel(_ScanStage.FAST_SCAN):
            return total_created, skipped_existing, 0

        t_collect = time.perf_counter()
        paths: list[str] = []
        for r in ("models", "input", "output"):
            if r not in roots:
                continue
            walk = list_root(r, should_stop)
            # A cancel during the walk leaves it partial.
            if should_stop():
                return total_created, skipped_existing, 0
            paths.extend(walk.files)
            scan_state.dirs_listed += walk.dirs_listed
            if r not in live_by_root:
                continue
            vanished, unlisted = unlisted_references(live_by_root[r], walk.listings, scan_state)
            marked_before = scan_state.missing_marked
            mark_unlisted_references_missing_safely(r, vanished, scan_state, should_stop)
            self._emit_marked_missing(r, scan_state.missing_marked - marked_before)
            logging.debug(
                "Fast scan: %s listing: %d dirs listed, %d rows retired, "
                "%d rows skipped (not listed, still on disk)",
                r,
                walk.dirs_listed,
                len(vanished),
                unlisted,
            )
        logging.debug(
            "Fast scan: collect_paths took %.3fs (%d paths found)",
            time.perf_counter() - t_collect,
            len(paths),
        )
        total_paths = len(paths)
        self._update_progress(total=total_paths)

        self._emit_event(
            "assets.seed.started",
            {"roots": list(roots), "total": total_paths, "phase": "fast"},
        )

        # Use stub specs (no metadata extraction, no hashing)
        t_specs = time.perf_counter()
        specs, tag_pool, skipped_existing = build_asset_specs(
            paths,
            existing_paths,
            enable_metadata_extraction=False,
            progress=scan_state,
            should_stop=should_stop,
        )
        logging.debug(
            "Fast scan: build_asset_specs took %.3fs (%d specs, %d skipped)",
            time.perf_counter() - t_specs,
            len(specs),
            skipped_existing,
        )
        self._update_progress(skipped=skipped_existing)

        if self._check_pause_and_cancel(_ScanStage.FAST_SCAN):
            return total_created, skipped_existing, total_paths

        batch_size = 500
        last_progress_time = time.perf_counter()
        progress_interval = 1.0

        for i in range(0, len(specs), batch_size):
            if self._check_pause_and_cancel(_ScanStage.FAST_SCAN):
                logging.info(
                    "Fast scan cancelled after %d/%d files (created=%d)",
                    i,
                    len(specs),
                    total_created,
                )
                return total_created, skipped_existing, total_paths

            batch = specs[i : i + batch_size]
            batch_tags = {t for spec in batch for t in spec["tags"]}
            created = 0
            try:
                created, batch_error = insert_asset_specs(batch, batch_tags, scan_state)
                total_created += created
                if batch_error is not None:
                    raise batch_error
            except MemoryError:
                # Recording this as a batch failure would march the scan through
                # every remaining batch while the process is out of memory.
                raise
            except Exception as e:
                self._add_error(
                    f"Batch insert encountered an error at offset {i} "
                    f"after creating {created}: {e}"
                )
                logging.exception(
                    "Batch insert encountered an error at offset %d after creating %d",
                    i,
                    created,
                )
                emit(
                    "seeder.batch_insert_failed",
                    error_type=error_type(e),
                    error_kind=error_kind(e),
                )

            scanned = i + len(batch)
            now = time.perf_counter()
            self._update_progress(scanned=scanned, created=total_created)

            if now - last_progress_time >= progress_interval:
                self._emit_event(
                    "assets.seed.progress",
                    {
                        "phase": "fast",
                        "scanned": scanned,
                        "total": len(specs),
                        "created": total_created,
                    },
                )
                last_progress_time = now

        self._update_progress(scanned=len(specs), created=total_created)
        tick_watch_list(scan_state)
        logging.info(
            "Fast scan complete: %.3fs total (created=%d, skipped=%d, total_paths=%d)",
            time.perf_counter() - t_fast_start,
            total_created,
            skipped_existing,
            total_paths,
        )
        return total_created, skipped_existing, total_paths

    def _run_enrich_phase(self, roots: tuple[RootType, ...]) -> tuple[bool, int]:
        """Run phase 2: enrich existing records with metadata and hashes.

        Returns:
            Tuple of (cancelled, total_enriched)
        """
        total_enriched = 0
        scan_state = self._scan_state
        with create_session() as session:
            drain_pending_verifications(session)
            session.commit()
            tick_watch_list(scan_state)
            for _ in range(3):
                drain_transition_queue(session)
                session.commit()
                if pending_transition_count() == 0:
                    break
        batch_size = 100
        last_progress_time = time.perf_counter()
        progress_interval = 1.0

        self._emit_event(
            "assets.seed.started",
            {"roots": list(roots), "phase": "enrich"},
        )

        last_seen_id: str | None = None

        while True:
            if self._check_pause_and_cancel(_ScanStage.ENRICH):
                logging.info("Enrich scan cancelled after %d assets", total_enriched)
                return True, total_enriched

            # Fetch next batch of unenriched assets
            unenriched = get_unenriched_assets_for_roots(
                roots,
                compute_hashes=self._compute_hashes,
                limit=batch_size,
                last_seen_id=last_seen_id,
            )

            if not unenriched:
                break

            enriched, _failed_ids, consumed = enrich_assets_batch(
                unenriched,
                extract_metadata=True,
                compute_hash=self._compute_hashes,
                interrupt_check=self._is_paused_or_cancelled,
                progress=scan_state,
            )
            total_enriched += enriched
            if consumed > 0:
                last_seen_id = unenriched[consumed - 1].record_id

            now = time.perf_counter()
            if now - last_progress_time >= progress_interval:
                self._emit_event(
                    "assets.seed.progress",
                    {
                        "phase": "enrich",
                        "enriched": total_enriched,
                    },
                )
                last_progress_time = now

        return False, total_enriched


asset_seeder = _AssetSeeder()
