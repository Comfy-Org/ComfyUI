"""Unit tests for the _AssetSeeder background scanning class."""

import threading
import time
from unittest.mock import patch

import pytest

from app.assets.database.queries.asset_reference import UnenrichedReferenceRow
from app.assets.seeder import _AssetSeeder, Progress, ScanInProgressError, ScanPhase, State


@pytest.fixture
def fresh_seeder():
    """Create a fresh _AssetSeeder instance for testing."""
    seeder = _AssetSeeder()
    yield seeder
    seeder.shutdown(timeout=1.0)


@pytest.fixture
def mock_dependencies():
    """Mock all external dependencies for isolated testing."""
    with (
        patch("app.assets.seeder.dependencies_available", return_value=True),
        patch("app.assets.seeder.sync_root_safely", return_value=set()),
        patch("app.assets.seeder.collect_paths_for_roots", return_value=[]),
        patch("app.assets.seeder.build_asset_specs", return_value=([], set(), 0)),
        patch("app.assets.seeder.insert_asset_specs", return_value=0),
        patch("app.assets.seeder.get_unenriched_assets_for_roots", return_value=[]),
        patch("app.assets.seeder.enrich_assets_batch", return_value=(0, 0)),
    ):
        yield


class TestSeederStateTransitions:
    """Test state machine transitions."""

    def test_initial_state_is_idle(self, fresh_seeder: _AssetSeeder):
        assert fresh_seeder.get_status().state == State.IDLE

    def test_start_transitions_to_running(
        self, fresh_seeder: _AssetSeeder, mock_dependencies
    ):
        barrier = threading.Event()
        reached = threading.Event()

        def slow_collect(*args):
            reached.set()
            barrier.wait(timeout=5.0)
            return []

        with patch(
            "app.assets.seeder.collect_paths_for_roots", side_effect=slow_collect
        ):
            started = fresh_seeder.start(roots=("models",))
            assert started is True
            assert reached.wait(timeout=2.0)
            assert fresh_seeder.get_status().state == State.RUNNING

            barrier.set()

    def test_start_while_running_returns_false(
        self, fresh_seeder: _AssetSeeder, mock_dependencies
    ):
        barrier = threading.Event()
        reached = threading.Event()

        def slow_collect(*args):
            reached.set()
            barrier.wait(timeout=5.0)
            return []

        with patch(
            "app.assets.seeder.collect_paths_for_roots", side_effect=slow_collect
        ):
            fresh_seeder.start(roots=("models",))
            assert reached.wait(timeout=2.0)

            second_start = fresh_seeder.start(roots=("models",))
            assert second_start is False

            barrier.set()

    def test_cancel_transitions_to_cancelling(
        self, fresh_seeder: _AssetSeeder, mock_dependencies
    ):
        barrier = threading.Event()
        reached = threading.Event()

        def slow_collect(*args):
            reached.set()
            barrier.wait(timeout=5.0)
            return []

        with patch(
            "app.assets.seeder.collect_paths_for_roots", side_effect=slow_collect
        ):
            fresh_seeder.start(roots=("models",))
            assert reached.wait(timeout=2.0)

            cancelled = fresh_seeder.cancel()
            assert cancelled is True
            assert fresh_seeder.get_status().state == State.CANCELLING

            barrier.set()

    def test_cancel_when_idle_returns_false(self, fresh_seeder: _AssetSeeder):
        cancelled = fresh_seeder.cancel()
        assert cancelled is False

    def test_state_returns_to_idle_after_completion(
        self, fresh_seeder: _AssetSeeder, mock_dependencies
    ):
        fresh_seeder.start(roots=("models",))
        completed = fresh_seeder.wait(timeout=5.0)
        assert completed is True
        assert fresh_seeder.get_status().state == State.IDLE


class TestSeederWait:
    """Test wait() behavior."""

    def test_wait_blocks_until_complete(
        self, fresh_seeder: _AssetSeeder, mock_dependencies
    ):
        fresh_seeder.start(roots=("models",))
        completed = fresh_seeder.wait(timeout=5.0)
        assert completed is True
        assert fresh_seeder.get_status().state == State.IDLE

    def test_wait_returns_false_on_timeout(
        self, fresh_seeder: _AssetSeeder, mock_dependencies
    ):
        barrier = threading.Event()

        def slow_collect(*args):
            barrier.wait(timeout=10.0)
            return []

        with patch(
            "app.assets.seeder.collect_paths_for_roots", side_effect=slow_collect
        ):
            fresh_seeder.start(roots=("models",))
            completed = fresh_seeder.wait(timeout=0.1)
            assert completed is False

            barrier.set()

    def test_wait_when_idle_returns_true(self, fresh_seeder: _AssetSeeder):
        completed = fresh_seeder.wait(timeout=1.0)
        assert completed is True


class TestSeederProgress:
    """Test progress tracking."""

    def test_get_status_returns_progress_during_scan(
        self, fresh_seeder: _AssetSeeder
    ):
        barrier = threading.Event()
        reached = threading.Event()

        def slow_build(*args, **kwargs):
            reached.set()
            barrier.wait(timeout=5.0)
            return ([], set(), 0)

        paths = ["/path/file1.safetensors", "/path/file2.safetensors"]

        with (
            patch("app.assets.seeder.dependencies_available", return_value=True),
            patch("app.assets.seeder.sync_root_safely", return_value=set()),
            patch("app.assets.seeder.collect_paths_for_roots", return_value=paths),
            patch("app.assets.seeder.build_asset_specs", side_effect=slow_build),
            patch("app.assets.seeder.insert_asset_specs", return_value=0),
            patch("app.assets.seeder.get_unenriched_assets_for_roots", return_value=[]),
            patch("app.assets.seeder.enrich_assets_batch", return_value=(0, 0)),
        ):
            fresh_seeder.start(roots=("models",))
            assert reached.wait(timeout=2.0)

            status = fresh_seeder.get_status()
            assert status.state == State.RUNNING
            assert status.progress is not None
            assert status.progress.total == 2

            barrier.set()

    def test_progress_callback_is_invoked(
        self, fresh_seeder: _AssetSeeder, mock_dependencies
    ):
        progress_updates: list[Progress] = []

        def callback(p: Progress):
            progress_updates.append(p)

        with patch(
            "app.assets.seeder.collect_paths_for_roots",
            return_value=[f"/path/file{i}.safetensors" for i in range(10)],
        ):
            fresh_seeder.start(roots=("models",), progress_callback=callback)
            fresh_seeder.wait(timeout=5.0)

        assert len(progress_updates) > 0


class TestSeederCancellation:
    """Test cancellation behavior."""

    def test_scan_commits_partial_progress_on_cancellation(
        self, fresh_seeder: _AssetSeeder
    ):
        insert_count = 0
        barrier = threading.Event()
        first_insert_done = threading.Event()

        def slow_insert(specs, tags):
            nonlocal insert_count
            insert_count += 1
            if insert_count == 1:
                first_insert_done.set()
            if insert_count >= 2:
                barrier.wait(timeout=5.0)
            return len(specs)

        paths = [f"/path/file{i}.safetensors" for i in range(1500)]
        specs = [
            {
                "abs_path": p,
                "size_bytes": 100,
                "mtime_ns": 0,
                "info_name": f"file{i}",
                "tags": [],
                "fname": f"file{i}",
                "metadata": None,
                "hash": None,
                "mime_type": None,
            }
            for i, p in enumerate(paths)
        ]

        with (
            patch("app.assets.seeder.dependencies_available", return_value=True),
            patch("app.assets.seeder.sync_root_safely", return_value=set()),
            patch("app.assets.seeder.collect_paths_for_roots", return_value=paths),
            patch(
                "app.assets.seeder.build_asset_specs", return_value=(specs, set(), 0)
            ),
            patch("app.assets.seeder.insert_asset_specs", side_effect=slow_insert),
            patch("app.assets.seeder.get_unenriched_assets_for_roots", return_value=[]),
            patch("app.assets.seeder.enrich_assets_batch", return_value=(0, 0)),
        ):
            fresh_seeder.start(roots=("models",))
            assert first_insert_done.wait(timeout=2.0)

            fresh_seeder.cancel()
            barrier.set()
            fresh_seeder.wait(timeout=5.0)

            assert 1 <= insert_count < 3  # 1500 paths / 500 batch = 3; cancel stopped early


class TestSeederErrorHandling:
    """Test error handling behavior."""

    def test_database_errors_captured_in_status(self, fresh_seeder: _AssetSeeder):
        with (
            patch("app.assets.seeder.dependencies_available", return_value=True),
            patch("app.assets.seeder.sync_root_safely", return_value=set()),
            patch(
                "app.assets.seeder.collect_paths_for_roots",
                return_value=["/path/file.safetensors"],
            ),
            patch(
                "app.assets.seeder.build_asset_specs",
                return_value=(
                    [
                        {
                            "abs_path": "/path/file.safetensors",
                            "size_bytes": 100,
                            "mtime_ns": 0,
                            "info_name": "file",
                            "tags": [],
                            "fname": "file",
                            "metadata": None,
                            "hash": None,
                            "mime_type": None,
                        }
                    ],
                    set(),
                    0,
                ),
            ),
            patch(
                "app.assets.seeder.insert_asset_specs",
                side_effect=Exception("DB connection failed"),
            ),
            patch("app.assets.seeder.get_unenriched_assets_for_roots", return_value=[]),
            patch("app.assets.seeder.enrich_assets_batch", return_value=(0, 0)),
        ):
            fresh_seeder.start(roots=("models",))
            fresh_seeder.wait(timeout=5.0)

            status = fresh_seeder.get_status()
            assert len(status.errors) > 0
            assert "DB connection failed" in status.errors[0]

    def test_dependencies_unavailable_captured_in_errors(
        self, fresh_seeder: _AssetSeeder
    ):
        with patch("app.assets.seeder.dependencies_available", return_value=False):
            fresh_seeder.start(roots=("models",))
            fresh_seeder.wait(timeout=5.0)

            status = fresh_seeder.get_status()
            assert len(status.errors) > 0
            assert "dependencies" in status.errors[0].lower()

    def test_thread_crash_resets_state_to_idle(self, fresh_seeder: _AssetSeeder):
        with (
            patch("app.assets.seeder.dependencies_available", return_value=True),
            patch(
                "app.assets.seeder.sync_root_safely",
                side_effect=RuntimeError("Unexpected crash"),
            ),
        ):
            fresh_seeder.start(roots=("models",))
            fresh_seeder.wait(timeout=5.0)

            status = fresh_seeder.get_status()
            assert status.state == State.IDLE
            assert len(status.errors) > 0


class TestSeederThreadSafety:
    """Test thread safety of concurrent operations."""

    def test_concurrent_start_calls_spawn_only_one_thread(
        self, fresh_seeder: _AssetSeeder, mock_dependencies
    ):
        barrier = threading.Event()

        def slow_collect(*args):
            barrier.wait(timeout=5.0)
            return []

        with patch(
            "app.assets.seeder.collect_paths_for_roots", side_effect=slow_collect
        ):
            results = []

            def try_start():
                results.append(fresh_seeder.start(roots=("models",)))

            threads = [threading.Thread(target=try_start) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            barrier.set()

            assert sum(results) == 1

    def test_get_status_safe_during_scan(
        self, fresh_seeder: _AssetSeeder, mock_dependencies
    ):
        barrier = threading.Event()
        reached = threading.Event()

        def slow_collect(*args):
            reached.set()
            barrier.wait(timeout=5.0)
            return []

        with patch(
            "app.assets.seeder.collect_paths_for_roots", side_effect=slow_collect
        ):
            fresh_seeder.start(roots=("models",))
            assert reached.wait(timeout=2.0)

            statuses = []
            for _ in range(100):
                statuses.append(fresh_seeder.get_status())

            barrier.set()

            assert all(
                s.state in (State.RUNNING, State.IDLE, State.CANCELLING)
                for s in statuses
            )


class TestSeederMarkMissing:
    """Test mark_missing_outside_prefixes behavior."""

    def test_mark_missing_when_idle(self, fresh_seeder: _AssetSeeder):
        with (
            patch("app.assets.seeder.dependencies_available", return_value=True),
            patch(
                "app.assets.seeder.get_all_known_prefixes",
                return_value=["/models", "/input", "/output"],
            ),
            patch(
                "app.assets.seeder.mark_missing_outside_prefixes_safely", return_value=5
            ) as mock_mark,
        ):
            result = fresh_seeder.mark_missing_outside_prefixes()
            assert result == 5
            mock_mark.assert_called_once_with(["/models", "/input", "/output"])

    def test_mark_missing_raises_when_running(
        self, fresh_seeder: _AssetSeeder, mock_dependencies
    ):
        barrier = threading.Event()
        reached = threading.Event()

        def slow_collect(*args):
            reached.set()
            barrier.wait(timeout=5.0)
            return []

        with patch(
            "app.assets.seeder.collect_paths_for_roots", side_effect=slow_collect
        ):
            fresh_seeder.start(roots=("models",))
            assert reached.wait(timeout=2.0)

            with pytest.raises(ScanInProgressError):
                fresh_seeder.mark_missing_outside_prefixes()

            barrier.set()

    def test_mark_missing_returns_zero_when_dependencies_unavailable(
        self, fresh_seeder: _AssetSeeder
    ):
        with patch("app.assets.seeder.dependencies_available", return_value=False):
            result = fresh_seeder.mark_missing_outside_prefixes()
            assert result == 0

    def test_prune_first_flag_triggers_mark_missing_before_scan(
        self, fresh_seeder: _AssetSeeder
    ):
        call_order = []

        def track_mark(prefixes):
            call_order.append("mark_missing")
            return 3

        def track_sync(root):
            call_order.append(f"sync_{root}")
            return set()

        with (
            patch("app.assets.seeder.dependencies_available", return_value=True),
            patch("app.assets.seeder.get_all_known_prefixes", return_value=["/models"]),
            patch("app.assets.seeder.mark_missing_outside_prefixes_safely", side_effect=track_mark),
            patch("app.assets.seeder.sync_root_safely", side_effect=track_sync),
            patch("app.assets.seeder.collect_paths_for_roots", return_value=[]),
            patch("app.assets.seeder.build_asset_specs", return_value=([], set(), 0)),
            patch("app.assets.seeder.insert_asset_specs", return_value=0),
            patch("app.assets.seeder.get_unenriched_assets_for_roots", return_value=[]),
            patch("app.assets.seeder.enrich_assets_batch", return_value=(0, 0)),
        ):
            fresh_seeder.start(roots=("models",), prune_first=True)
            fresh_seeder.wait(timeout=5.0)

            assert call_order[0] == "mark_missing"
            assert "sync_models" in call_order


class TestSeederPhases:
    """Test phased scanning behavior."""

    def test_start_fast_only_runs_fast_phase(self, fresh_seeder: _AssetSeeder):
        """Verify start_fast only runs the fast phase."""
        fast_called = []
        enrich_called = []

        def track_fast(*args, **kwargs):
            fast_called.append(True)
            return ([], set(), 0)

        def track_enrich(*args, **kwargs):
            enrich_called.append(True)
            return []

        with (
            patch("app.assets.seeder.dependencies_available", return_value=True),
            patch("app.assets.seeder.sync_root_safely", return_value=set()),
            patch("app.assets.seeder.collect_paths_for_roots", return_value=[]),
            patch("app.assets.seeder.build_asset_specs", side_effect=track_fast),
            patch("app.assets.seeder.insert_asset_specs", return_value=0),
            patch("app.assets.seeder.get_unenriched_assets_for_roots", side_effect=track_enrich),
            patch("app.assets.seeder.enrich_assets_batch", return_value=(0, 0)),
        ):
            fresh_seeder.start_fast(roots=("models",))
            fresh_seeder.wait(timeout=5.0)

            assert len(fast_called) == 1
            assert len(enrich_called) == 0

    def test_start_enrich_only_runs_enrich_phase(self, fresh_seeder: _AssetSeeder):
        """Verify start_enrich only runs the enrich phase."""
        fast_called = []
        enrich_called = []

        def track_fast(*args, **kwargs):
            fast_called.append(True)
            return ([], set(), 0)

        def track_enrich(*args, **kwargs):
            enrich_called.append(True)
            return []

        with (
            patch("app.assets.seeder.dependencies_available", return_value=True),
            patch("app.assets.seeder.sync_root_safely", return_value=set()),
            patch("app.assets.seeder.collect_paths_for_roots", return_value=[]),
            patch("app.assets.seeder.build_asset_specs", side_effect=track_fast),
            patch("app.assets.seeder.insert_asset_specs", return_value=0),
            patch("app.assets.seeder.get_unenriched_assets_for_roots", side_effect=track_enrich),
            patch("app.assets.seeder.enrich_assets_batch", return_value=(0, 0)),
        ):
            fresh_seeder.start_enrich(roots=("models",))
            fresh_seeder.wait(timeout=5.0)

            assert len(fast_called) == 0
            assert len(enrich_called) == 1

    def test_full_scan_runs_both_phases(self, fresh_seeder: _AssetSeeder):
        """Verify full scan runs both fast and enrich phases."""
        fast_called = []
        enrich_called = []

        def track_fast(*args, **kwargs):
            fast_called.append(True)
            return ([], set(), 0)

        def track_enrich(*args, **kwargs):
            enrich_called.append(True)
            return []

        with (
            patch("app.assets.seeder.dependencies_available", return_value=True),
            patch("app.assets.seeder.sync_root_safely", return_value=set()),
            patch("app.assets.seeder.collect_paths_for_roots", return_value=[]),
            patch("app.assets.seeder.build_asset_specs", side_effect=track_fast),
            patch("app.assets.seeder.insert_asset_specs", return_value=0),
            patch("app.assets.seeder.get_unenriched_assets_for_roots", side_effect=track_enrich),
            patch("app.assets.seeder.enrich_assets_batch", return_value=(0, 0)),
        ):
            fresh_seeder.start(roots=("models",), phase=ScanPhase.FULL)
            fresh_seeder.wait(timeout=5.0)

            assert len(fast_called) == 1
            assert len(enrich_called) == 1


class TestSeederPauseResume:
    """Test pause/resume behavior."""

    def test_pause_transitions_to_paused(
        self, fresh_seeder: _AssetSeeder, mock_dependencies
    ):
        barrier = threading.Event()
        reached = threading.Event()

        def slow_collect(*args):
            reached.set()
            barrier.wait(timeout=5.0)
            return []

        with patch(
            "app.assets.seeder.collect_paths_for_roots", side_effect=slow_collect
        ):
            fresh_seeder.start(roots=("models",))
            assert reached.wait(timeout=2.0)

            paused = fresh_seeder.pause()
            assert paused is True
            assert fresh_seeder.get_status().state == State.PAUSED

            barrier.set()

    def test_pause_when_idle_returns_false(self, fresh_seeder: _AssetSeeder):
        paused = fresh_seeder.pause()
        assert paused is False

    def test_resume_returns_to_running(
        self, fresh_seeder: _AssetSeeder, mock_dependencies
    ):
        barrier = threading.Event()
        reached = threading.Event()

        def slow_collect(*args):
            reached.set()
            barrier.wait(timeout=5.0)
            return []

        with patch(
            "app.assets.seeder.collect_paths_for_roots", side_effect=slow_collect
        ):
            fresh_seeder.start(roots=("models",))
            assert reached.wait(timeout=2.0)

            fresh_seeder.pause()
            assert fresh_seeder.get_status().state == State.PAUSED

            resumed = fresh_seeder.resume()
            assert resumed is True
            assert fresh_seeder.get_status().state == State.RUNNING

            barrier.set()

    def test_resume_when_not_paused_returns_false(
        self, fresh_seeder: _AssetSeeder, mock_dependencies
    ):
        barrier = threading.Event()
        reached = threading.Event()

        def slow_collect(*args):
            reached.set()
            barrier.wait(timeout=5.0)
            return []

        with patch(
            "app.assets.seeder.collect_paths_for_roots", side_effect=slow_collect
        ):
            fresh_seeder.start(roots=("models",))
            assert reached.wait(timeout=2.0)

            resumed = fresh_seeder.resume()
            assert resumed is False

            barrier.set()

    def test_cancel_while_paused_works(
        self, fresh_seeder: _AssetSeeder, mock_dependencies
    ):
        barrier = threading.Event()
        reached_checkpoint = threading.Event()

        def slow_collect(*args):
            reached_checkpoint.set()
            barrier.wait(timeout=5.0)
            return []

        with patch(
            "app.assets.seeder.collect_paths_for_roots", side_effect=slow_collect
        ):
            fresh_seeder.start(roots=("models",))
            assert reached_checkpoint.wait(timeout=2.0)

            fresh_seeder.pause()
            assert fresh_seeder.get_status().state == State.PAUSED

            cancelled = fresh_seeder.cancel()
            assert cancelled is True

            barrier.set()
            fresh_seeder.wait(timeout=5.0)
            assert fresh_seeder.get_status().state == State.IDLE

class TestSeederStopRestart:
    """Test stop and restart behavior."""

    def test_stop_is_alias_for_cancel(
        self, fresh_seeder: _AssetSeeder, mock_dependencies
    ):
        barrier = threading.Event()
        reached = threading.Event()

        def slow_collect(*args):
            reached.set()
            barrier.wait(timeout=5.0)
            return []

        with patch(
            "app.assets.seeder.collect_paths_for_roots", side_effect=slow_collect
        ):
            fresh_seeder.start(roots=("models",))
            assert reached.wait(timeout=2.0)

            stopped = fresh_seeder.stop()
            assert stopped is True
            assert fresh_seeder.get_status().state == State.CANCELLING

            barrier.set()

    def test_restart_cancels_and_starts_new_scan(
        self, fresh_seeder: _AssetSeeder, mock_dependencies
    ):
        barrier = threading.Event()
        reached = threading.Event()
        start_count = 0

        def slow_collect(*args):
            nonlocal start_count
            start_count += 1
            if start_count == 1:
                reached.set()
                barrier.wait(timeout=5.0)
            return []

        with patch(
            "app.assets.seeder.collect_paths_for_roots", side_effect=slow_collect
        ):
            fresh_seeder.start(roots=("models",))
            assert reached.wait(timeout=2.0)

            barrier.set()
            restarted = fresh_seeder.restart()
            assert restarted is True

            fresh_seeder.wait(timeout=5.0)
            assert start_count == 2

    def test_restart_preserves_previous_params(self, fresh_seeder: _AssetSeeder):
        """Verify restart uses previous params when not overridden."""
        collected_roots = []

        def track_collect(roots):
            collected_roots.append(roots)
            return []

        with (
            patch("app.assets.seeder.dependencies_available", return_value=True),
            patch("app.assets.seeder.sync_root_safely", return_value=set()),
            patch("app.assets.seeder.collect_paths_for_roots", side_effect=track_collect),
            patch("app.assets.seeder.build_asset_specs", return_value=([], set(), 0)),
            patch("app.assets.seeder.insert_asset_specs", return_value=0),
            patch("app.assets.seeder.get_unenriched_assets_for_roots", return_value=[]),
            patch("app.assets.seeder.enrich_assets_batch", return_value=(0, 0)),
        ):
            fresh_seeder.start(roots=("input", "output"))
            fresh_seeder.wait(timeout=5.0)

            fresh_seeder.restart()
            fresh_seeder.wait(timeout=5.0)

            assert len(collected_roots) == 2
            assert collected_roots[0] == ("input", "output")
            assert collected_roots[1] == ("input", "output")

    def test_restart_can_override_params(self, fresh_seeder: _AssetSeeder):
        """Verify restart can override previous params."""
        collected_roots = []

        def track_collect(roots):
            collected_roots.append(roots)
            return []

        with (
            patch("app.assets.seeder.dependencies_available", return_value=True),
            patch("app.assets.seeder.sync_root_safely", return_value=set()),
            patch("app.assets.seeder.collect_paths_for_roots", side_effect=track_collect),
            patch("app.assets.seeder.build_asset_specs", return_value=([], set(), 0)),
            patch("app.assets.seeder.insert_asset_specs", return_value=0),
            patch("app.assets.seeder.get_unenriched_assets_for_roots", return_value=[]),
            patch("app.assets.seeder.enrich_assets_batch", return_value=(0, 0)),
        ):
            fresh_seeder.start(roots=("models",))
            fresh_seeder.wait(timeout=5.0)

            fresh_seeder.restart(roots=("input",))
            fresh_seeder.wait(timeout=5.0)

            assert len(collected_roots) == 2
            assert collected_roots[0] == ("models",)
            assert collected_roots[1] == ("input",)


class TestEnqueueEnrichHandoff:
    """Test that the drain of _pending_enrich is atomic with start_enrich."""

    def test_pending_enrich_runs_after_scan_completes(
        self, fresh_seeder: _AssetSeeder, mock_dependencies
    ):
        """A queued enrich request runs automatically when a scan finishes."""
        enrich_roots_seen: list[tuple] = []
        original_start = fresh_seeder.start

        def tracking_start(*args, **kwargs):
            phase = kwargs.get("phase")
            roots = kwargs.get("roots", args[0] if args else None)
            result = original_start(*args, **kwargs)
            if phase == ScanPhase.ENRICH and result:
                enrich_roots_seen.append(roots)
            return result

        fresh_seeder.start = tracking_start

        # Start a fast scan, then enqueue an enrich while it's running
        barrier = threading.Event()
        reached = threading.Event()

        def slow_collect(*args):
            reached.set()
            barrier.wait(timeout=5.0)
            return []

        with patch(
            "app.assets.seeder.collect_paths_for_roots", side_effect=slow_collect
        ):
            fresh_seeder.start(roots=("models",), phase=ScanPhase.FAST)
            assert reached.wait(timeout=2.0)

            queued = fresh_seeder.enqueue_enrich(
                roots=("input",), compute_hashes=True
            )
            assert queued is False  # queued, not started immediately

            barrier.set()

        # Wait for the original scan + the auto-started enrich scan
        deadline = time.monotonic() + 5.0
        while fresh_seeder.get_status().state != State.IDLE and time.monotonic() < deadline:
            time.sleep(0.05)

        assert enrich_roots_seen == [("input",)]

    def test_enqueue_enrich_during_drain_does_not_lose_work(
        self, fresh_seeder: _AssetSeeder, mock_dependencies
    ):
        """enqueue_enrich called concurrently with drain cannot drop work.

        Simulates the race: another thread calls enqueue_enrich right as the
        scan thread is draining _pending_enrich.  The enqueue must either be
        picked up by the draining scan or successfully start its own scan.
        """
        barrier = threading.Event()
        reached = threading.Event()
        enrich_started = threading.Event()

        enrich_call_count = 0

        def slow_collect(*args):
            reached.set()
            barrier.wait(timeout=5.0)
            return []

        # Track how many times start_enrich actually fires
        real_start_enrich = fresh_seeder.start_enrich
        enrich_roots_seen: list[tuple] = []

        def tracking_start_enrich(**kwargs):
            nonlocal enrich_call_count
            enrich_call_count += 1
            enrich_roots_seen.append(kwargs.get("roots"))
            result = real_start_enrich(**kwargs)
            if result:
                enrich_started.set()
            return result

        fresh_seeder.start_enrich = tracking_start_enrich

        with patch(
            "app.assets.seeder.collect_paths_for_roots", side_effect=slow_collect
        ):
            # Start a scan
            fresh_seeder.start(roots=("models",), phase=ScanPhase.FAST)
            assert reached.wait(timeout=2.0)

            # Queue an enrich while scan is running
            fresh_seeder.enqueue_enrich(roots=("output",), compute_hashes=False)

            # Let scan finish — drain will fire start_enrich atomically
            barrier.set()

        # Wait for drain to complete and the enrich scan to start
        assert enrich_started.wait(timeout=5.0), "Enrich scan was never started from drain"
        assert ("output",) in enrich_roots_seen

    def test_concurrent_enqueue_during_drain_not_lost(
        self, fresh_seeder: _AssetSeeder,
    ):
        """A second enqueue_enrich arriving while drain is in progress is not lost.

        Because the drain now holds _lock through the start_enrich call,
        a concurrent enqueue_enrich will block until start_enrich has
        transitioned state to RUNNING, then the enqueue will queue its
        payload as _pending_enrich for the *next* drain.
        """
        scan_barrier = threading.Event()
        scan_reached = threading.Event()
        enrich_barrier = threading.Event()
        enrich_reached = threading.Event()

        collect_call = 0

        def gated_collect(*args):
            nonlocal collect_call
            collect_call += 1
            if collect_call == 1:
                # First call: the initial fast scan
                scan_reached.set()
                scan_barrier.wait(timeout=5.0)
            return []

        enrich_call = 0

        def gated_get_unenriched(*args, **kwargs):
            nonlocal enrich_call
            enrich_call += 1
            if enrich_call == 1:
                # First enrich batch: signal and block
                enrich_reached.set()
                enrich_barrier.wait(timeout=5.0)
            return []

        with (
            patch("app.assets.seeder.dependencies_available", return_value=True),
            patch("app.assets.seeder.sync_root_safely", return_value=set()),
            patch("app.assets.seeder.collect_paths_for_roots", side_effect=gated_collect),
            patch("app.assets.seeder.build_asset_specs", return_value=([], set(), 0)),
            patch("app.assets.seeder.insert_asset_specs", return_value=0),
            patch("app.assets.seeder.get_unenriched_assets_for_roots", side_effect=gated_get_unenriched),
            patch("app.assets.seeder.enrich_assets_batch", return_value=(0, 0)),
        ):
            # 1. Start fast scan
            fresh_seeder.start(roots=("models",), phase=ScanPhase.FAST)
            assert scan_reached.wait(timeout=2.0)

            # 2. Queue enrich while fast scan is running
            queued = fresh_seeder.enqueue_enrich(
                roots=("input",), compute_hashes=False
            )
            assert queued is False

            # 3. Let the fast scan finish — drain will start the enrich scan
            scan_barrier.set()

            # 4. Wait until the drained enrich scan is running
            assert enrich_reached.wait(timeout=5.0)

            # 5. Now enqueue another enrich while the drained scan is running
            queued2 = fresh_seeder.enqueue_enrich(
                roots=("output",), compute_hashes=True
            )
            assert queued2 is False  # should be queued, not started

            # Verify _pending_enrich was set (the second enqueue was captured)
            with fresh_seeder._lock:
                assert fresh_seeder._pending_enrich is not None
                assert "output" in fresh_seeder._pending_enrich["roots"]

            # Let the enrich scan finish
            enrich_barrier.set()

        deadline = time.monotonic() + 5.0
        while fresh_seeder.get_status().state != State.IDLE and time.monotonic() < deadline:
            time.sleep(0.05)


def _make_row(ref_id: str, asset_id: str = "a1") -> UnenrichedReferenceRow:
    return UnenrichedReferenceRow(
        reference_id=ref_id, asset_id=asset_id,
        file_path=f"/fake/{ref_id}.bin", enrichment_level=0,
    )


class TestEnrichPhaseDefensiveLogic:
    """Test skip_ids filtering and consecutive_empty termination."""

    def test_failed_refs_are_skipped_on_subsequent_batches(
        self, fresh_seeder: _AssetSeeder,
    ):
        """References that fail enrichment are filtered out of future batches."""
        row_a = _make_row("r1")
        row_b = _make_row("r2")
        call_count = 0

        def fake_get_unenriched(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return [row_a, row_b]
            return []

        enriched_refs: list[list[str]] = []

        def fake_enrich(rows, **kwargs):
            ref_ids = [r.reference_id for r in rows]
            enriched_refs.append(ref_ids)
            # r1 always fails, r2 succeeds
            failed = [r.reference_id for r in rows if r.reference_id == "r1"]
            enriched = len(rows) - len(failed)
            return enriched, failed

        with (
            patch("app.assets.seeder.dependencies_available", return_value=True),
            patch("app.assets.seeder.sync_root_safely", return_value=set()),
            patch("app.assets.seeder.collect_paths_for_roots", return_value=[]),
            patch("app.assets.seeder.build_asset_specs", return_value=([], set(), 0)),
            patch("app.assets.seeder.insert_asset_specs", return_value=0),
            patch("app.assets.seeder.get_unenriched_assets_for_roots", side_effect=fake_get_unenriched),
            patch("app.assets.seeder.enrich_assets_batch", side_effect=fake_enrich),
        ):
            fresh_seeder.start(roots=("models",), phase=ScanPhase.ENRICH)
            fresh_seeder.wait(timeout=5.0)

        # First batch: both refs attempted
        assert "r1" in enriched_refs[0]
        assert "r2" in enriched_refs[0]
        # Second batch: r1 filtered out
        assert "r1" not in enriched_refs[1]
        assert "r2" in enriched_refs[1]

    def test_stops_after_consecutive_empty_batches(
        self, fresh_seeder: _AssetSeeder,
    ):
        """Enrich phase terminates after 3 consecutive batches with zero progress."""
        row = _make_row("r1")
        batch_count = 0

        def fake_get_unenriched(*args, **kwargs):
            nonlocal batch_count
            batch_count += 1
            # Always return the same row (simulating a permanently failing ref)
            return [row]

        def fake_enrich(rows, **kwargs):
            # Always fail — zero enriched, all failed
            return 0, [r.reference_id for r in rows]

        with (
            patch("app.assets.seeder.dependencies_available", return_value=True),
            patch("app.assets.seeder.sync_root_safely", return_value=set()),
            patch("app.assets.seeder.collect_paths_for_roots", return_value=[]),
            patch("app.assets.seeder.build_asset_specs", return_value=([], set(), 0)),
            patch("app.assets.seeder.insert_asset_specs", return_value=0),
            patch("app.assets.seeder.get_unenriched_assets_for_roots", side_effect=fake_get_unenriched),
            patch("app.assets.seeder.enrich_assets_batch", side_effect=fake_enrich),
        ):
            fresh_seeder.start(roots=("models",), phase=ScanPhase.ENRICH)
            fresh_seeder.wait(timeout=5.0)

        # Should stop after exactly 3 consecutive empty batches
        # Batch 1: returns row, enrich fails → filtered out in batch 2+
        # But get_unenriched keeps returning it, filter removes it → empty → break
        # Actually: batch 1 has row, fails. Batch 2 get_unenriched returns [row],
        # skip_ids filters it → empty list → breaks via `if not unenriched: break`
        # So it terminates in 2 calls to get_unenriched.
        assert batch_count == 2

    def test_consecutive_empty_counter_resets_on_success(
        self, fresh_seeder: _AssetSeeder,
    ):
        """A successful batch resets the consecutive empty counter."""
        call_count = 0

        def fake_get_unenriched(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 6:
                return [_make_row(f"r{call_count}", f"a{call_count}")]
            return []

        def fake_enrich(rows, **kwargs):
            ref_id = rows[0].reference_id
            # Fail batches 1-2, succeed batch 3, fail batches 4-5, succeed batch 6
            if ref_id in ("r1", "r2", "r4", "r5"):
                return 0, [ref_id]
            return 1, []

        with (
            patch("app.assets.seeder.dependencies_available", return_value=True),
            patch("app.assets.seeder.sync_root_safely", return_value=set()),
            patch("app.assets.seeder.collect_paths_for_roots", return_value=[]),
            patch("app.assets.seeder.build_asset_specs", return_value=([], set(), 0)),
            patch("app.assets.seeder.insert_asset_specs", return_value=0),
            patch("app.assets.seeder.get_unenriched_assets_for_roots", side_effect=fake_get_unenriched),
            patch("app.assets.seeder.enrich_assets_batch", side_effect=fake_enrich),
        ):
            fresh_seeder.start(roots=("models",), phase=ScanPhase.ENRICH)
            fresh_seeder.wait(timeout=5.0)

        # All 6 batches should run + 1 final call returning empty
        assert call_count == 7
        status = fresh_seeder.get_status()
        assert status.state == State.IDLE
