"""Covers the seeder's event sink, idle reset, and ``enqueue_scan``'s start-now-or-queue-and-merge behaviour."""

import threading
from typing import Any
from unittest.mock import call, patch

import pytest

from app.assets.seeder import Progress, ScanPhase, _AssetSeeder, State


@pytest.fixture()
def seeder():
    """Fresh seeder instance for each test."""
    return _AssetSeeder()


class TestEventSink:
    def test_delivers_event_and_payload_to_sink(self, seeder):
        received: list[tuple[str, dict[str, Any]]] = []

        def record_event(event_type: str, data: dict[str, Any]) -> None:
            received.append((event_type, data))

        event_type = "assets.seed.started"
        payload = {"roots": ["models"], "total": 1, "phase": "fast"}
        seeder.set_event_sink(record_event)

        seeder._emit_event(event_type, payload)

        assert received == [(event_type, payload)]

    def test_noop_when_no_sink_is_set(self, seeder):
        assert seeder._emit_event("assets.seed.resumed", {}) is None

    def test_swallows_raising_sink_exception(self, seeder):
        def raise_from_sink(event_type: str, data: dict[str, Any]) -> None:
            raise RuntimeError("boom")

        seeder.set_event_sink(raise_from_sink)

        assert seeder._emit_event("assets.seed.error", {"message": "failure"}) is None


# ---------------------------------------------------------------------------
# _reset_to_idle
# ---------------------------------------------------------------------------


class TestResetToIdle:
    def test_sets_idle_and_clears_progress(self, seeder):
        """_reset_to_idle should move state to IDLE and snapshot progress."""
        progress = Progress(scanned=10, total=20, created=5, skipped=3)
        seeder._state = State.RUNNING
        seeder._progress = progress

        with seeder._lock:
            seeder._reset_to_idle()

        assert seeder._state is State.IDLE
        assert seeder._progress is None
        assert seeder._last_progress is progress

    def test_noop_when_progress_already_none(self, seeder):
        """_reset_to_idle should handle None progress gracefully."""
        seeder._state = State.CANCELLING
        seeder._progress = None

        with seeder._lock:
            seeder._reset_to_idle()

        assert seeder._state is State.IDLE
        assert seeder._progress is None
        assert seeder._last_progress is None


# ---------------------------------------------------------------------------
# enqueue_scan – immediate start when idle
# ---------------------------------------------------------------------------


class TestEnqueueScanStartsImmediately:
    def test_starts_when_idle(self, seeder):
        with patch.object(seeder, "start", return_value=True) as mock:
            assert (
                seeder.enqueue_scan(
                    roots=("output",), phase=ScanPhase.ENRICH, compute_hashes=True
                )
                is True
            )
            mock.assert_called_once_with(
                roots=("output",),
                phase=ScanPhase.ENRICH,
                prune_first=False,
                compute_hashes=True,
            )

    def test_no_pending_when_started_immediately(self, seeder):
        with patch.object(seeder, "start", return_value=True):
            seeder.enqueue_scan(roots=("output",), phase=ScanPhase.ENRICH)
        assert seeder._pending_scan is None


# ---------------------------------------------------------------------------
# enqueue_scan – queuing when busy
# ---------------------------------------------------------------------------


class TestEnqueueScanQueuesWhenBusy:
    def test_queues_when_busy(self, seeder):
        """enqueue_scan should store a pending request when seeder is busy."""
        with patch.object(seeder, "start", return_value=False):
            result = seeder.enqueue_scan(
                roots=("models",), phase=ScanPhase.ENRICH, compute_hashes=False
            )

        assert result is False
        assert seeder._pending_scan == {
            "roots": ("models",),
            "phase": ScanPhase.ENRICH,
            "compute_hashes": False,
        }

    def test_queues_preserves_compute_hashes_true(self, seeder):
        with patch.object(seeder, "start", return_value=False):
            seeder.enqueue_scan(
                roots=("input",), phase=ScanPhase.ENRICH, compute_hashes=True
            )

        assert seeder._pending_scan["compute_hashes"] is True


# ---------------------------------------------------------------------------
# enqueue_scan – merging when a pending request already exists
# ---------------------------------------------------------------------------


class TestEnqueueScanMergesPending:
    def _make_busy(self, seeder):
        return patch.object(seeder, "start", return_value=False)

    def test_merges_roots(self, seeder):
        """A second enqueue should merge roots with the existing pending request."""
        with self._make_busy(seeder):
            seeder.enqueue_scan(roots=("models",), phase=ScanPhase.ENRICH)
            seeder.enqueue_scan(roots=("output",), phase=ScanPhase.FAST)

        merged = set(seeder._pending_scan["roots"])
        assert merged == {"models", "output"}
        assert seeder._pending_scan["phase"] is ScanPhase.FULL

    def test_merges_overlapping_roots(self, seeder):
        """Duplicate roots should be deduplicated."""
        with self._make_busy(seeder):
            seeder.enqueue_scan(roots=("models", "input"), phase=ScanPhase.ENRICH)
            seeder.enqueue_scan(roots=("input", "output"), phase=ScanPhase.ENRICH)

        merged = set(seeder._pending_scan["roots"])
        assert merged == {"models", "input", "output"}

    def test_compute_hashes_sticky_true(self, seeder):
        """Once compute_hashes is True it should stay True after merging."""
        with self._make_busy(seeder):
            seeder.enqueue_scan(
                roots=("models",), phase=ScanPhase.ENRICH, compute_hashes=True
            )
            seeder.enqueue_scan(
                roots=("output",), phase=ScanPhase.ENRICH, compute_hashes=False
            )

        assert seeder._pending_scan["compute_hashes"] is True

    def test_compute_hashes_upgrades_to_true(self, seeder):
        """A later enqueue with compute_hashes=True should upgrade the pending request."""
        with self._make_busy(seeder):
            seeder.enqueue_scan(
                roots=("models",), phase=ScanPhase.ENRICH, compute_hashes=False
            )
            seeder.enqueue_scan(
                roots=("output",), phase=ScanPhase.ENRICH, compute_hashes=True
            )

        assert seeder._pending_scan["compute_hashes"] is True

    def test_compute_hashes_stays_false(self, seeder):
        """If both enqueues have compute_hashes=False it stays False."""
        with self._make_busy(seeder):
            seeder.enqueue_scan(
                roots=("models",), phase=ScanPhase.ENRICH, compute_hashes=False
            )
            seeder.enqueue_scan(
                roots=("output",), phase=ScanPhase.ENRICH, compute_hashes=False
            )

        assert seeder._pending_scan["compute_hashes"] is False

    def test_triple_merge(self, seeder):
        """Three successive enqueues should all merge correctly."""
        with self._make_busy(seeder):
            seeder.enqueue_scan(
                roots=("models",), phase=ScanPhase.ENRICH, compute_hashes=False
            )
            seeder.enqueue_scan(
                roots=("input",), phase=ScanPhase.ENRICH, compute_hashes=False
            )
            seeder.enqueue_scan(
                roots=("output",), phase=ScanPhase.ENRICH, compute_hashes=True
            )

        merged = set(seeder._pending_scan["roots"])
        assert merged == {"models", "input", "output"}
        assert seeder._pending_scan["compute_hashes"] is True


class TestPendingScanDrain:
    @patch("app.assets.seeder.dependencies_available", return_value=True)
    @patch("app.assets.seeder.get_owned_prefixes", return_value=[])
    @patch("app.assets.seeder.sync_root_safely", return_value=set())
    @patch("app.assets.seeder.collect_paths_for_roots", return_value=[])
    @patch("app.assets.seeder.build_asset_specs", return_value=([], {}, 0))
    def test_pending_scan_starts_after_scan(self, *_mocks):
        seeder = _AssetSeeder()

        seeder._pending_scan = {
            "roots": ("output",),
            "phase": ScanPhase.FULL,
            "compute_hashes": True,
        }

        real_start = seeder.start
        with patch.object(seeder, "start", wraps=real_start) as mock_start:
            seeder.start_fast(roots=("models",))
            seeder.wait(timeout=5)
            seeder.wait(timeout=5)

            assert mock_start.call_args_list[-1] == call(
                roots=("output",),
                phase=ScanPhase.FULL,
                prune_first=False,
                compute_hashes=True,
            )

        assert seeder._pending_scan is None

    @patch("app.assets.seeder.dependencies_available", return_value=True)
    @patch("app.assets.seeder.get_owned_prefixes", return_value=[])
    @patch("app.assets.seeder.sync_root_safely", return_value=set())
    @patch("app.assets.seeder.collect_paths_for_roots", return_value=[])
    @patch("app.assets.seeder.build_asset_specs", return_value=([], {}, 0))
    def test_pending_cleared_even_when_start_fails(self, *_mocks):
        seeder = _AssetSeeder()
        seeder._pending_scan = {
            "roots": ("output",),
            "phase": ScanPhase.ENRICH,
            "compute_hashes": False,
        }

        real_start = seeder.start

        def start_initial_scan_only(**kwargs):
            if kwargs["roots"] == ("models",):
                return real_start(**kwargs)
            return False

        with patch.object(seeder, "start", side_effect=start_initial_scan_only):
            seeder.start_fast(roots=("models",))
            seeder.wait(timeout=5)

        assert seeder._pending_scan is None

    @patch("app.assets.seeder.dependencies_available", return_value=True)
    @patch("app.assets.seeder.get_owned_prefixes", return_value=[])
    @patch("app.assets.seeder.sync_root_safely", return_value=set())
    @patch("app.assets.seeder.collect_paths_for_roots", return_value=[])
    @patch("app.assets.seeder.build_asset_specs", return_value=([], {}, 0))
    def test_no_drain_when_no_pending(self, *_mocks):
        seeder = _AssetSeeder()
        assert seeder._pending_scan is None

        real_start = seeder.start
        with patch.object(seeder, "start", wraps=real_start) as mock_start:
            seeder.start_fast(roots=("models",))
            seeder.wait(timeout=5)

            assert mock_start.call_count == 1


# ---------------------------------------------------------------------------
# Thread-safety of enqueue_scan
# ---------------------------------------------------------------------------


class TestEnqueueScanThreadSafety:
    def test_concurrent_enqueues(self, seeder):
        """Multiple threads enqueuing should not lose roots."""
        with patch.object(seeder, "start", return_value=False):
            barrier = threading.Barrier(3)

            def enqueue(root):
                barrier.wait()
                seeder.enqueue_scan(
                    roots=(root,), phase=ScanPhase.ENRICH, compute_hashes=False
                )

            threads = [
                threading.Thread(target=enqueue, args=(r,))
                for r in ("models", "input", "output")
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

        merged = set(seeder._pending_scan["roots"])
        assert merged == {"models", "input", "output"}
