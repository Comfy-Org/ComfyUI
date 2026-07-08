"""Tests for app.assets.seeder – enqueue_enrich and pending-queue behaviour."""

import threading
from unittest.mock import patch

import pytest

from app.assets.seeder import Progress, _AssetSeeder, State


@pytest.fixture()
def seeder():
    """Fresh seeder instance for each test."""
    return _AssetSeeder()


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
# enqueue_enrich – immediate start when idle
# ---------------------------------------------------------------------------


class TestEnqueueEnrichStartsImmediately:
    def test_starts_when_idle(self, seeder):
        """enqueue_enrich should delegate to start_enrich and return True when idle."""
        with patch.object(seeder, "start_enrich", return_value=True) as mock:
            assert seeder.enqueue_enrich(roots=("output",), compute_hashes=True) is True
            mock.assert_called_once_with(roots=("output",), compute_hashes=True)

    def test_no_pending_when_started_immediately(self, seeder):
        """No pending request should be stored when start_enrich succeeds."""
        with patch.object(seeder, "start_enrich", return_value=True):
            seeder.enqueue_enrich(roots=("output",))
        assert seeder._pending_enrich is None


# ---------------------------------------------------------------------------
# enqueue_enrich – queuing when busy
# ---------------------------------------------------------------------------


class TestEnqueueEnrichQueuesWhenBusy:
    def test_queues_when_busy(self, seeder):
        """enqueue_enrich should store a pending request when seeder is busy."""
        with patch.object(seeder, "start_enrich", return_value=False):
            result = seeder.enqueue_enrich(roots=("models",), compute_hashes=False)

        assert result is False
        assert seeder._pending_enrich == {
            "roots": ("models",),
            "compute_hashes": False,
        }

    def test_queues_preserves_compute_hashes_true(self, seeder):
        with patch.object(seeder, "start_enrich", return_value=False):
            seeder.enqueue_enrich(roots=("input",), compute_hashes=True)

        assert seeder._pending_enrich["compute_hashes"] is True


# ---------------------------------------------------------------------------
# enqueue_enrich – merging when a pending request already exists
# ---------------------------------------------------------------------------


class TestEnqueueEnrichMergesPending:
    def _make_busy(self, seeder):
        """Patch start_enrich to always return False (seeder busy)."""
        return patch.object(seeder, "start_enrich", return_value=False)

    def test_merges_roots(self, seeder):
        """A second enqueue should merge roots with the existing pending request."""
        with self._make_busy(seeder):
            seeder.enqueue_enrich(roots=("models",))
            seeder.enqueue_enrich(roots=("output",))

        merged = set(seeder._pending_enrich["roots"])
        assert merged == {"models", "output"}

    def test_merges_overlapping_roots(self, seeder):
        """Duplicate roots should be deduplicated."""
        with self._make_busy(seeder):
            seeder.enqueue_enrich(roots=("models", "input"))
            seeder.enqueue_enrich(roots=("input", "output"))

        merged = set(seeder._pending_enrich["roots"])
        assert merged == {"models", "input", "output"}

    def test_compute_hashes_sticky_true(self, seeder):
        """Once compute_hashes is True it should stay True after merging."""
        with self._make_busy(seeder):
            seeder.enqueue_enrich(roots=("models",), compute_hashes=True)
            seeder.enqueue_enrich(roots=("output",), compute_hashes=False)

        assert seeder._pending_enrich["compute_hashes"] is True

    def test_compute_hashes_upgrades_to_true(self, seeder):
        """A later enqueue with compute_hashes=True should upgrade the pending request."""
        with self._make_busy(seeder):
            seeder.enqueue_enrich(roots=("models",), compute_hashes=False)
            seeder.enqueue_enrich(roots=("output",), compute_hashes=True)

        assert seeder._pending_enrich["compute_hashes"] is True

    def test_compute_hashes_stays_false(self, seeder):
        """If both enqueues have compute_hashes=False it stays False."""
        with self._make_busy(seeder):
            seeder.enqueue_enrich(roots=("models",), compute_hashes=False)
            seeder.enqueue_enrich(roots=("output",), compute_hashes=False)

        assert seeder._pending_enrich["compute_hashes"] is False

    def test_triple_merge(self, seeder):
        """Three successive enqueues should all merge correctly."""
        with self._make_busy(seeder):
            seeder.enqueue_enrich(roots=("models",), compute_hashes=False)
            seeder.enqueue_enrich(roots=("input",), compute_hashes=False)
            seeder.enqueue_enrich(roots=("output",), compute_hashes=True)

        merged = set(seeder._pending_enrich["roots"])
        assert merged == {"models", "input", "output"}
        assert seeder._pending_enrich["compute_hashes"] is True


# ---------------------------------------------------------------------------
# Pending enrich drains after scan completes
# ---------------------------------------------------------------------------


class TestPendingEnrichDrain:
    """Verify that _run_scan drains _pending_enrich via start_enrich."""

    @patch("app.assets.seeder.dependencies_available", return_value=True)
    @patch("app.assets.seeder.get_all_known_prefixes", return_value=[])
    @patch("app.assets.seeder.sync_root_safely", return_value=set())
    @patch("app.assets.seeder.collect_paths_for_roots", return_value=[])
    @patch("app.assets.seeder.build_asset_specs", return_value=([], {}, 0))
    def test_pending_enrich_starts_after_scan(self, *_mocks):
        """After a fast scan finishes, the pending enrich should be started."""
        seeder = _AssetSeeder()

        seeder._pending_enrich = {
            "roots": ("output",),
            "compute_hashes": True,
        }

        with patch.object(seeder, "start_enrich", return_value=True) as mock_start:
            seeder.start_fast(roots=("models",))
            seeder.wait(timeout=5)

            mock_start.assert_called_once_with(
                roots=("output",),
                compute_hashes=True,
            )

        assert seeder._pending_enrich is None

    @patch("app.assets.seeder.dependencies_available", return_value=True)
    @patch("app.assets.seeder.get_all_known_prefixes", return_value=[])
    @patch("app.assets.seeder.sync_root_safely", return_value=set())
    @patch("app.assets.seeder.collect_paths_for_roots", return_value=[])
    @patch("app.assets.seeder.build_asset_specs", return_value=([], {}, 0))
    def test_pending_cleared_even_when_start_fails(self, *_mocks):
        """_pending_enrich should be cleared even if start_enrich returns False."""
        seeder = _AssetSeeder()
        seeder._pending_enrich = {
            "roots": ("output",),
            "compute_hashes": False,
        }

        with patch.object(seeder, "start_enrich", return_value=False):
            seeder.start_fast(roots=("models",))
            seeder.wait(timeout=5)

        assert seeder._pending_enrich is None

    @patch("app.assets.seeder.dependencies_available", return_value=True)
    @patch("app.assets.seeder.get_all_known_prefixes", return_value=[])
    @patch("app.assets.seeder.sync_root_safely", return_value=set())
    @patch("app.assets.seeder.collect_paths_for_roots", return_value=[])
    @patch("app.assets.seeder.build_asset_specs", return_value=([], {}, 0))
    def test_no_drain_when_no_pending(self, *_mocks):
        """start_enrich should not be called when there is no pending request."""
        seeder = _AssetSeeder()
        assert seeder._pending_enrich is None

        with patch.object(seeder, "start_enrich", return_value=True) as mock_start:
            seeder.start_fast(roots=("models",))
            seeder.wait(timeout=5)

            mock_start.assert_not_called()


# ---------------------------------------------------------------------------
# Thread-safety of enqueue_enrich
# ---------------------------------------------------------------------------


class TestEnqueueEnrichThreadSafety:
    def test_concurrent_enqueues(self, seeder):
        """Multiple threads enqueuing should not lose roots."""
        with patch.object(seeder, "start_enrich", return_value=False):
            barrier = threading.Barrier(3)

            def enqueue(root):
                barrier.wait()
                seeder.enqueue_enrich(roots=(root,), compute_hashes=False)

            threads = [
                threading.Thread(target=enqueue, args=(r,))
                for r in ("models", "input", "output")
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

        merged = set(seeder._pending_enrich["roots"])
        assert merged == {"models", "input", "output"}
