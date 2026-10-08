"""The startup prune waits until the node list has been built once, so a model folder a
custom node registers in INPUT_TYPES counts as not yet scanned rather than missing."""

import threading
import time
from unittest.mock import MagicMock

import pytest

from app.assets import seeder as seeder_module
from app.assets.seeder import PruneCancelledError, ScanPhase, State, _AssetSeeder, _ScanState


@pytest.fixture
def scan_seeder(monkeypatch: pytest.MonkeyPatch) -> _AssetSeeder:
    instance = _AssetSeeder()
    instance._state = State.RUNNING
    instance._scan_state = _ScanState()
    instance._roots = ("models", "input", "output")
    instance._phase = ScanPhase.FAST
    instance._prune_first = instance._prune_pending = True
    monkeypatch.setattr(seeder_module, "dependencies_available", lambda: True)
    monkeypatch.setattr(seeder_module, "get_owned_prefixes", lambda: [])
    monkeypatch.setattr(instance, "_log_scan_config", lambda roots: None)
    monkeypatch.setattr(instance, "_run_fast_phase", lambda roots: (0, 0, 0))
    return instance


@pytest.fixture
def prunes(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def prune(prefixes, _should_stop=None):
        calls.append(prefixes)
        return 0

    monkeypatch.setattr(seeder_module, "mark_missing_outside_prefixes_safely", prune)
    return calls


@pytest.fixture
def temp_syncs(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    calls: list[object] = []
    monkeypatch.setattr(
        seeder_module, "sync_temp_references_safely", lambda progress, _should_stop=None: calls.append(progress)
    )
    return calls


def test_the_startup_scan_does_not_prune_before_the_node_list(scan_seeder, prunes, temp_syncs):
    scan_seeder._run_scan()

    assert prunes == []
    assert len(temp_syncs) == 1
    assert scan_seeder.prune_pending()


def test_the_first_scan_after_the_node_list_prunes_once(scan_seeder, prunes, temp_syncs):
    scan_seeder.node_list_served()
    scan_seeder._run_scan()
    scan_seeder._state = State.RUNNING
    scan_seeder._scan_state = _ScanState()
    scan_seeder._prune_first = False
    scan_seeder._run_scan()

    assert len(prunes) == 1
    assert not scan_seeder.prune_pending()


@pytest.mark.parametrize("outcome", ["failed", "cancelled"])
def test_a_prune_that_did_not_finish_stays_pending(scan_seeder, monkeypatch, temp_syncs, outcome):
    def prune(_prefixes, _should_stop=None):
        if outcome == "cancelled":
            scan_seeder._cancel_event.set()
            return 3
        return None

    monkeypatch.setattr(seeder_module, "mark_missing_outside_prefixes_safely", prune)
    scan_seeder.node_list_served()

    scan_seeder._run_scan()

    assert scan_seeder.prune_pending()


def test_a_standalone_prune_clears_the_pending_one(scan_seeder, prunes):
    scan_seeder._state = State.IDLE

    assert scan_seeder.mark_missing_outside_prefixes() == 0
    assert not scan_seeder.prune_pending()


def test_a_cancelled_standalone_prune_stays_pending(scan_seeder, monkeypatch):
    scan_seeder._state = State.IDLE

    def prune(_prefixes, should_stop):
        scan_seeder._cancel_event.set()
        should_stop()
        return 2

    monkeypatch.setattr(seeder_module, "mark_missing_outside_prefixes_safely", prune)

    with pytest.raises(PruneCancelledError):
        scan_seeder.mark_missing_outside_prefixes()
    assert scan_seeder.prune_pending()


def test_no_scan_starts_once_shutdown_has_begun(monkeypatch):
    instance = _AssetSeeder()
    instance._shutting_down = True
    thread = MagicMock()
    monkeypatch.setattr(seeder_module.threading, "Thread", thread)

    assert instance.start(roots=("output",)) is False
    thread.assert_not_called()


def test_a_scan_queued_during_startup_does_not_run_into_shutdown(monkeypatch):
    instance = _AssetSeeder()
    instance._state = State.RUNNING
    instance._pending_scan = {"roots": ("output",), "phase": ScanPhase.FULL, "compute_hashes": False}
    instance._shutting_down = True
    thread = MagicMock()
    monkeypatch.setattr(seeder_module.threading, "Thread", thread)

    with instance._lock:
        instance._finish_and_start_pending()

    thread.assert_not_called()
    assert instance._state is State.IDLE


@pytest.fixture
def live_seeder(monkeypatch: pytest.MonkeyPatch, prunes, temp_syncs) -> _AssetSeeder:
    """A seeder whose scans run on real threads through the public calls, with the
    filesystem phases stubbed out."""
    instance = _AssetSeeder()
    monkeypatch.setattr(seeder_module, "dependencies_available", lambda: True)
    monkeypatch.setattr(seeder_module, "get_owned_prefixes", lambda: [])
    monkeypatch.setattr(instance, "_log_scan_config", lambda roots: None)
    monkeypatch.setattr(instance, "_run_fast_phase", lambda roots: (0, 0, 0))
    monkeypatch.setattr(instance, "_run_enrich_phase", lambda roots: (False, 0))
    return instance


def test_the_startup_prune_runs_in_the_first_ordinary_scan_after_the_node_list(live_seeder, prunes):
    assert live_seeder.start(prune_first=True, phase=ScanPhase.FAST)
    assert live_seeder.wait(5)
    assert prunes == []

    live_seeder.node_list_served()
    assert live_seeder.start(phase=ScanPhase.FAST)
    assert live_seeder.wait(5)

    assert len(prunes) == 1
    assert not live_seeder.prune_pending()


def test_a_scan_queued_behind_the_startup_scan_runs_the_prune(live_seeder, prunes, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def held_fast_phase(roots):
        entered.set()
        release.wait(5)
        return 0, 0, 0

    monkeypatch.setattr(live_seeder, "_run_fast_phase", held_fast_phase)
    assert live_seeder.start(prune_first=True, phase=ScanPhase.FAST)
    assert entered.wait(5)  # past its prune check, so only the queued scan can prune
    assert prunes == []
    live_seeder.node_list_served()
    assert not live_seeder.start(phase=ScanPhase.FAST)
    live_seeder.enqueue_scan(roots=("models", "input", "output"), phase=ScanPhase.FAST)

    release.set()
    deadline = time.monotonic() + 5
    while (live_seeder.prune_pending() or live_seeder.get_status().state is not State.IDLE) and time.monotonic() < deadline:
        live_seeder.wait(0.1)

    assert len(prunes) == 1
    assert not live_seeder.prune_pending()
