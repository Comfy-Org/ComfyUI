"""Background asset seeder with thread management and cancellation support."""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from app.assets.scanner import (
    ENRICHMENT_METADATA,
    ENRICHMENT_STUB,
    RootType,
    build_asset_specs,
    collect_paths_for_roots,
    enrich_assets_batch,
    get_all_known_prefixes,
    get_prefixes_for_root,
    get_unenriched_assets_for_roots,
    insert_asset_specs,
    mark_missing_outside_prefixes_safely,
    sync_root_safely,
)
from app.database.db import dependencies_available


class ScanInProgressError(Exception):
    """Raised when an operation cannot proceed because a scan is running."""


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


@dataclass
class Progress:
    """Progress information for a scan operation."""

    scanned: int = 0
    total: int = 0
    created: int = 0
    skipped: int = 0


@dataclass
class ScanStatus:
    """Current status of the asset seeder."""

    state: State
    progress: Progress | None
    errors: list[str] = field(default_factory=list)


ProgressCallback = Callable[[Progress], None]


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
        self._progress: Progress | None = None
        self._last_progress: Progress | None = None
        self._errors: list[str] = []
        self._thread: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._run_gate = threading.Event()
        self._run_gate.set()  # Start unpaused (set = running, clear = paused)
        self._roots: tuple[RootType, ...] = ()
        self._phase: ScanPhase = ScanPhase.FULL
        self._compute_hashes: bool = False
        self._prune_first: bool = False
        self._progress_callback: ProgressCallback | None = None
        self._disabled: bool = False
        self._pending_enrich: dict | None = None

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
    ) -> bool:
        """Start a background scan for the given roots.

        Args:
            roots: Tuple of root types to scan (models, input, output)
            phase: Scan phase to run (FAST, ENRICH, or FULL for both)
            progress_callback: Optional callback called with progress updates
            prune_first: If True, prune orphaned assets before scanning
            compute_hashes: If True, compute blake3 hashes (slow)

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
            self._state = State.RUNNING
            self._progress = Progress()
            self._errors = []
            self._roots = roots
            self._phase = phase
            self._prune_first = prune_first
            self._compute_hashes = compute_hashes
            self._progress_callback = progress_callback
            self._cancel_event.clear()
            self._run_gate.set()  # Ensure unpaused when starting
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

    def start_enrich(
        self,
        roots: tuple[RootType, ...] = ("models", "input", "output"),
        progress_callback: ProgressCallback | None = None,
        compute_hashes: bool = False,
    ) -> bool:
        """Start an enrichment scan (phase 2 only) - extracts metadata and hashes.

        Args:
            roots: Tuple of root types to scan
            progress_callback: Optional callback for progress updates
            compute_hashes: If True, compute blake3 hashes

        Returns:
            True if scan was started, False if already running
        """
        return self.start(
            roots=roots,
            phase=ScanPhase.ENRICH,
            progress_callback=progress_callback,
            prune_first=False,
            compute_hashes=compute_hashes,
        )

    def enqueue_enrich(
        self,
        roots: tuple[RootType, ...] = ("models", "input", "output"),
        compute_hashes: bool = False,
    ) -> bool:
        """Start an enrichment scan now, or queue it for after the current scan.

        If the seeder is idle, starts immediately. Otherwise, the enrich
        request is stored and will run automatically when the current scan
        finishes.

        Args:
            roots: Tuple of root types to scan
            compute_hashes: If True, compute blake3 hashes

        Returns:
            True if started immediately, False if queued for later
        """
        with self._lock:
            if self.start_enrich(roots=roots, compute_hashes=compute_hashes):
                return True
            if self._pending_enrich is not None:
                existing_roots = set(self._pending_enrich["roots"])
                existing_roots.update(roots)
                self._pending_enrich["roots"] = tuple(existing_roots)
                self._pending_enrich["compute_hashes"] = (
                    self._pending_enrich["compute_hashes"] or compute_hashes
                )
            else:
                self._pending_enrich = {
                    "roots": roots,
                    "compute_hashes": compute_hashes,
                }
            logging.info("Enrich scan queued (roots=%s)", self._pending_enrich["roots"])
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
            src = self._progress or self._last_progress
            return ScanStatus(
                state=self._state,
                progress=Progress(
                    scanned=src.scanned,
                    total=src.total,
                    created=src.created,
                    skipped=src.skipped,
                )
                if src
                else None,
                errors=list(self._errors),
            )

    def shutdown(self, timeout: float = 5.0) -> None:
        """Gracefully shutdown: cancel any running scan and wait for thread.

        Args:
            timeout: Maximum seconds to wait for thread to exit
        """
        self.cancel()
        self.wait(timeout=timeout)
        with self._lock:
            self._thread = None

    def mark_missing_outside_prefixes(self) -> int:
        """Mark references as missing when outside all known root prefixes.

        This is a non-destructive soft-delete operation. Assets and their
        metadata are preserved, but references are flagged as missing.
        They can be restored if the file reappears in a future scan.

        This operation is decoupled from scanning to prevent partial scans
        from accidentally marking assets belonging to other roots.

        Should be called explicitly when cleanup is desired, typically after
        a full scan of all roots or during maintenance.

        Returns:
            Number of references marked as missing

        Raises:
            ScanInProgressError: If a scan is currently running
        """
        with self._lock:
            if self._state != State.IDLE:
                raise ScanInProgressError(
                    "Cannot mark missing assets while scan is running"
                )
            self._state = State.RUNNING

        try:
            if not dependencies_available():
                logging.warning(
                    "Database dependencies not available, skipping mark missing"
                )
                return 0

            all_prefixes = get_all_known_prefixes()
            marked = mark_missing_outside_prefixes_safely(all_prefixes)
            if marked > 0:
                logging.info("Marked %d references as missing", marked)
            return marked
        finally:
            with self._lock:
                self._reset_to_idle()

    def _reset_to_idle(self) -> None:
        """Reset state to IDLE, preserving last progress. Caller must hold _lock."""
        self._last_progress = self._progress
        self._state = State.IDLE
        self._progress = None

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
        return not self._run_gate.is_set() or self._cancel_event.is_set()

    def _check_pause_and_cancel(self) -> bool:
        """Block while paused, then check if cancelled.

        Call this at checkpoint locations in scan loops. It will:
        1. Block indefinitely while paused (until resume or cancel)
        2. Return True if cancelled, False to continue

        Returns:
            True if scan should stop, False to continue
        """
        if not self._run_gate.is_set():
            self._emit_event("assets.seed.paused", {})
        self._run_gate.wait()  # Blocks if paused
        return self._is_cancelled()

    def _emit_event(self, event_type: str, data: dict) -> None:
        """Emit a WebSocket event if server is available."""
        try:
            from server import PromptServer

            if hasattr(PromptServer, "instance") and PromptServer.instance:
                PromptServer.instance.send_sync(event_type, data)
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
            if self._progress is None:
                return
            if scanned is not None:
                self._progress.scanned = scanned
            if total is not None:
                self._progress.total = total
            if created is not None:
                self._progress.created = created
            if skipped is not None:
                self._progress.skipped = skipped
            if self._progress_callback:
                callback = self._progress_callback
                progress = Progress(
                    scanned=self._progress.scanned,
                    total=self._progress.total,
                    created=self._progress.created,
                    skipped=self._progress.skipped,
                )

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
                prefixes = get_prefixes_for_root(root)
                if prefixes:
                    logging.info("Asset scan [%s] directories: %s", root, prefixes)

    def _run_scan(self) -> None:
        """Main scan loop running in background thread."""
        t_start = time.perf_counter()
        roots = self._roots
        phase = self._phase
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

            if self._prune_first:
                all_prefixes = get_all_known_prefixes()
                marked = mark_missing_outside_prefixes_safely(all_prefixes)
                if marked > 0:
                    logging.info("Marked %d refs as missing before scan", marked)

            if self._check_pause_and_cancel():
                logging.info("Asset scan cancelled after pruning phase")
                cancelled = True
                return

            self._log_scan_config(roots)

            # Phase 1: Fast scan (stub records)
            if phase in (ScanPhase.FAST, ScanPhase.FULL):
                created, skipped, paths = self._run_fast_phase(roots)
                total_created, skipped_existing, total_paths = created, skipped, paths

                if self._check_pause_and_cancel():
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
                if self._check_pause_and_cancel():
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

            elapsed = time.perf_counter() - t_start
            logging.info(
                "Scan(%s, %s) done %.3fs: created=%d enriched=%d skipped=%d",
                roots,
                phase.value,
                elapsed,
                total_created,
                total_enriched,
                skipped_existing,
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
            self._emit_event("assets.seed.error", {"message": str(e)})
        finally:
            if cancelled:
                self._emit_event(
                    "assets.seed.cancelled",
                    {
                        "scanned": self._progress.scanned if self._progress else 0,
                        "total": total_paths,
                        "created": total_created,
                    },
                )
            with self._lock:
                self._reset_to_idle()
                pending = self._pending_enrich
                if pending is not None:
                    self._pending_enrich = None
                    if not self.start_enrich(
                        roots=pending["roots"],
                        compute_hashes=pending["compute_hashes"],
                    ):
                        logging.warning(
                            "Pending enrich scan could not start (roots=%s)",
                            pending["roots"],
                        )

    def _run_fast_phase(self, roots: tuple[RootType, ...]) -> tuple[int, int, int]:
        """Run phase 1: fast scan to create stub records.

        Returns:
            Tuple of (total_created, skipped_existing, total_paths)
        """
        t_fast_start = time.perf_counter()
        total_created = 0
        skipped_existing = 0

        existing_paths: set[str] = set()
        t_sync = time.perf_counter()
        for r in roots:
            if self._check_pause_and_cancel():
                return total_created, skipped_existing, 0
            existing_paths.update(sync_root_safely(r))
        logging.debug(
            "Fast scan: sync_root phase took %.3fs (%d existing paths)",
            time.perf_counter() - t_sync,
            len(existing_paths),
        )

        if self._check_pause_and_cancel():
            return total_created, skipped_existing, 0

        t_collect = time.perf_counter()
        paths = collect_paths_for_roots(roots)
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
            compute_hashes=False,
        )
        logging.debug(
            "Fast scan: build_asset_specs took %.3fs (%d specs, %d skipped)",
            time.perf_counter() - t_specs,
            len(specs),
            skipped_existing,
        )
        self._update_progress(skipped=skipped_existing)

        if self._check_pause_and_cancel():
            return total_created, skipped_existing, total_paths

        batch_size = 500
        last_progress_time = time.perf_counter()
        progress_interval = 1.0

        for i in range(0, len(specs), batch_size):
            if self._check_pause_and_cancel():
                logging.info(
                    "Fast scan cancelled after %d/%d files (created=%d)",
                    i,
                    len(specs),
                    total_created,
                )
                return total_created, skipped_existing, total_paths

            batch = specs[i : i + batch_size]
            batch_tags = {t for spec in batch for t in spec["tags"]}
            try:
                created = insert_asset_specs(batch, batch_tags)
                total_created += created
            except Exception as e:
                self._add_error(f"Batch insert failed at offset {i}: {e}")
                logging.exception("Batch insert failed at offset %d", i)

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
        batch_size = 100
        last_progress_time = time.perf_counter()
        progress_interval = 1.0

        # Get the target enrichment level based on compute_hashes
        if not self._compute_hashes:
            target_max_level = ENRICHMENT_STUB
        else:
            target_max_level = ENRICHMENT_METADATA

        self._emit_event(
            "assets.seed.started",
            {"roots": list(roots), "phase": "enrich"},
        )

        skip_ids: set[str] = set()
        consecutive_empty = 0
        max_consecutive_empty = 3

        # Hash checkpoints survive across batches so interrupted hashes
        # can be resumed without re-reading the entire file.
        hash_checkpoints: dict[str, object] = {}

        while True:
            if self._check_pause_and_cancel():
                logging.info("Enrich scan cancelled after %d assets", total_enriched)
                return True, total_enriched

            # Fetch next batch of unenriched assets
            unenriched = get_unenriched_assets_for_roots(
                roots,
                max_level=target_max_level,
                limit=batch_size,
            )

            # Filter out previously failed references
            if skip_ids:
                unenriched = [r for r in unenriched if r.reference_id not in skip_ids]

            if not unenriched:
                break

            enriched, failed_ids = enrich_assets_batch(
                unenriched,
                extract_metadata=True,
                compute_hash=self._compute_hashes,
                interrupt_check=self._is_paused_or_cancelled,
                hash_checkpoints=hash_checkpoints,
            )
            total_enriched += enriched
            skip_ids.update(failed_ids)

            if enriched == 0:
                consecutive_empty += 1
                if consecutive_empty >= max_consecutive_empty:
                    logging.warning(
                        "Enrich phase stopping: %d consecutive batches with no progress (%d skipped)",
                        consecutive_empty,
                        len(skip_ids),
                    )
                    break
            else:
                consecutive_empty = 0

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
