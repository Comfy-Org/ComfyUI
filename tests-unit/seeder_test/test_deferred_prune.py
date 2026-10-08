"""The startup prune waits until the node list has been built once, so a model folder a
custom node registers in INPUT_TYPES counts as not yet scanned rather than missing."""

import threading
from unittest.mock import MagicMock

import pytest

from app.assets import seeder as seeder_module
from app.assets.seeder import ScanPhase, State, _AssetSeeder, _ScanState

ROOTS = ("models", "input", "output")


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


@pytest.fixture
def scan_seeder(monkeypatch: pytest.MonkeyPatch, prunes, temp_syncs) -> _AssetSeeder:
    """A seeder whose scans run on real threads through the public calls, with the
    filesystem phases stubbed out."""
    instance = _AssetSeeder()
    monkeypatch.setattr(seeder_module, "dependencies_available", lambda: True)
    monkeypatch.setattr(seeder_module, "get_owned_prefixes", lambda: [])
    monkeypatch.setattr(instance, "_log_scan_config", lambda roots: None)
    monkeypatch.setattr(instance, "_run_fast_phase", lambda roots: (0, 0, 0))
    monkeypatch.setattr(instance, "_run_enrich_phase", lambda roots: (False, 0))
    return instance


def _hold_fast_phase(instance: _AssetSeeder, monkeypatch: pytest.MonkeyPatch) -> tuple[threading.Event, threading.Event]:
    """Make the next scans wait in their fast phase until released."""
    entered, release = threading.Event(), threading.Event()

    def held(roots):
        entered.set()
        release.wait(5)
        return 0, 0, 0

    monkeypatch.setattr(instance, "_run_fast_phase", held)
    return entered, release


def _settle(instance: _AssetSeeder) -> None:
    """Wait for the running scan and any scan queued behind it."""
    for _ in range(50):
        if instance.wait(0.1) and instance.get_status().state is State.IDLE and instance._pending_scan is None:
            return
    raise AssertionError("seeder did not settle")


def test_the_startup_scan_does_not_prune_before_the_node_list(scan_seeder, prunes, temp_syncs):
    assert scan_seeder.start(prune_first=True, phase=ScanPhase.FAST)
    _settle(scan_seeder)

    assert prunes == []
    assert len(temp_syncs) == 1
    assert scan_seeder._prune_pending


def test_the_scan_after_the_node_list_prunes_once(scan_seeder, prunes, temp_syncs):
    scan_seeder.start(prune_first=True, phase=ScanPhase.FAST)
    _settle(scan_seeder)

    scan_seeder.start_after_node_list(ROOTS, compute_hashes=False)
    _settle(scan_seeder)
    scan_seeder.start_after_node_list(ROOTS, compute_hashes=False)
    _settle(scan_seeder)

    assert len(prunes) == 1
    assert len(temp_syncs) == 1  # the temp sync belongs to the startup scan only
    assert not scan_seeder._prune_pending


def test_a_scan_queued_behind_the_startup_scan_runs_the_prune(scan_seeder, prunes, monkeypatch):
    entered, release = _hold_fast_phase(scan_seeder, monkeypatch)
    scan_seeder.start(prune_first=True, phase=ScanPhase.FAST)
    assert entered.wait(5)  # past its prune check

    scan_seeder.start_after_node_list(ROOTS, compute_hashes=True)
    assert scan_seeder._pending_scan == {"roots": ("models",), "phase": ScanPhase.FULL, "compute_hashes": True}
    release.set()
    _settle(scan_seeder)

    assert len(prunes) == 1


def test_no_second_scan_is_queued_behind_one_that_will_prune(scan_seeder, monkeypatch):
    scan_seeder.start(prune_first=True, phase=ScanPhase.FAST)
    _settle(scan_seeder)
    pruning, release = threading.Event(), threading.Event()
    prunes: list[list[str]] = []

    def slow_prune(prefixes, _should_stop=None):
        pruning.set()
        release.wait(5)
        prunes.append(prefixes)
        return 0

    monkeypatch.setattr(seeder_module, "mark_missing_outside_prefixes_safely", slow_prune)
    scan_seeder.start_after_node_list(ROOTS, compute_hashes=False)
    assert pruning.wait(5)

    scan_seeder.start_after_node_list(ROOTS, compute_hashes=False)  # a second tab loads the page

    assert scan_seeder._pending_scan is None
    release.set()
    _settle(scan_seeder)
    assert len(prunes) == 1

def test_a_failed_prune_is_not_retried(scan_seeder, monkeypatch):
    monkeypatch.setattr(seeder_module, "mark_missing_outside_prefixes_safely", lambda _prefixes, _should_stop=None: None)
    scan_seeder.start(prune_first=True, phase=ScanPhase.FAST)
    _settle(scan_seeder)

    scan_seeder.start_after_node_list(ROOTS, compute_hashes=False)
    _settle(scan_seeder)

    assert not scan_seeder._prune_pending


def test_a_cancelled_prune_stays_pending(scan_seeder, monkeypatch):
    def cancelled(_prefixes, _should_stop=None):
        scan_seeder._cancel_event.set()
        return 3

    monkeypatch.setattr(seeder_module, "mark_missing_outside_prefixes_safely", cancelled)
    scan_seeder.start(prune_first=True, phase=ScanPhase.FAST)
    _settle(scan_seeder)

    scan_seeder.start_after_node_list(ROOTS, compute_hashes=False)
    _settle(scan_seeder)

    assert scan_seeder._prune_pending


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
    instance._scan_state = _ScanState()
    instance._pending_scan = {"roots": ("output",), "phase": ScanPhase.FULL, "compute_hashes": False}
    instance._shutting_down = True
    thread = MagicMock()
    monkeypatch.setattr(seeder_module.threading, "Thread", thread)

    with instance._lock:
        instance._finish_and_start_pending()

    thread.assert_not_called()
    assert instance._state is State.IDLE
