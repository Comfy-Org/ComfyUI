"""The startup prune waits until the node list has been built once, so a model folder a
custom node registers in INPUT_TYPES counts as not yet scanned rather than missing."""

import logging
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


def test_a_scan_queued_during_startup_does_not_run_into_shutdown(monkeypatch, caplog):
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
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_request_merged_into_a_queued_scan_logs_at_debug(monkeypatch, caplog):
    instance = _AssetSeeder()
    instance._state = State.RUNNING

    with caplog.at_level(logging.DEBUG):
        instance.enqueue_scan(roots=("models",), phase=ScanPhase.FULL)
        instance.enqueue_scan(roots=("models",), phase=ScanPhase.FULL)

    queued = [r.levelno for r in caplog.records if r.getMessage().startswith("Scan queued")]
    assert queued == [logging.INFO, logging.DEBUG]


def _record_phases(instance: _AssetSeeder, monkeypatch: pytest.MonkeyPatch, hold: str) -> tuple[list[tuple], threading.Event, threading.Event]:
    """Record the order of prunes and phases; the first `hold` phase waits until released."""
    events: list[tuple] = []
    entered, release = threading.Event(), threading.Event()

    def phase(name, result):
        def run(roots):
            events.append((name, frozenset(roots)))
            if name == hold and not entered.is_set():
                entered.set()
                release.wait(5)
            return result
        return run

    def prune(_prefixes, _should_stop=None):
        events.append(("prune",))
        return 0

    monkeypatch.setattr(seeder_module, "mark_missing_outside_prefixes_safely", prune)
    monkeypatch.setattr(instance, "_run_fast_phase", phase("fast", (0, 0, 0)))
    monkeypatch.setattr(instance, "_run_enrich_phase", phase("enrich", (False, 0)))
    return events, entered, release


@pytest.mark.parametrize("hashing", [False, True])
def test_a_page_load_during_the_fast_phase_prunes_before_enrichment(scan_seeder, monkeypatch, hashing):
    events, entered, release = _record_phases(scan_seeder, monkeypatch, hold="fast")
    enrich_hashes: list[bool] = []
    enrich = scan_seeder._run_enrich_phase
    monkeypatch.setattr(scan_seeder, "_run_enrich_phase", lambda roots: (enrich_hashes.append(scan_seeder._compute_hashes), enrich(roots))[1])
    scan_seeder.start(roots=ROOTS, prune_first=True, compute_hashes=hashing)
    assert entered.wait(5)

    scan_seeder.start_after_node_list(ROOTS, compute_hashes=hashing)
    release.set()
    _settle(scan_seeder)

    assert events == [("fast", frozenset(ROOTS)), ("prune",), ("fast", frozenset(ROOTS)), ("enrich", frozenset(ROOTS))]
    assert enrich_hashes == [hashing]


def test_a_page_load_during_enrichment_waits_for_it(scan_seeder, monkeypatch):
    events, entered, release = _record_phases(scan_seeder, monkeypatch, hold="enrich")
    scan_seeder.start(roots=ROOTS, prune_first=True)
    assert entered.wait(5)

    scan_seeder.start_after_node_list(ROOTS, compute_hashes=False)
    release.set()
    _settle(scan_seeder)

    models = frozenset({"models"})
    assert events == [("fast", frozenset(ROOTS)), ("enrich", frozenset(ROOTS)), ("prune",), ("fast", models), ("enrich", models)]


def test_an_enrich_only_scan_keeps_its_enrichment(scan_seeder, monkeypatch):
    events, _, _ = _record_phases(scan_seeder, monkeypatch, hold="")
    scan_seeder.start(roots=ROOTS, prune_first=True, phase=ScanPhase.FAST)
    _settle(scan_seeder)
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(scan_seeder, "_log_scan_config", lambda roots: (entered.set(), release.wait(5)))
    scan_seeder.start(roots=("input",), phase=ScanPhase.ENRICH)
    assert entered.wait(5)

    scan_seeder.start_after_node_list(ROOTS, compute_hashes=False)
    release.set()
    _settle(scan_seeder)

    assert ("enrich", frozenset({"input"})) in events


def test_a_scan_queued_before_any_page_load_waits_for_enrichment(scan_seeder, monkeypatch):
    events, entered, release = _record_phases(scan_seeder, monkeypatch, hold="fast")
    scan_seeder.start(roots=ROOTS, prune_first=True)
    assert entered.wait(5)

    scan_seeder.enqueue_scan(roots=("output",), phase=ScanPhase.FULL)  # a prompt's output rescan
    release.set()
    _settle(scan_seeder)

    output = frozenset({"output"})
    assert events == [("fast", frozenset(ROOTS)), ("enrich", frozenset(ROOTS)), ("fast", output), ("enrich", output)]


def test_a_scan_queued_after_the_prune_waits_for_enrichment(scan_seeder, monkeypatch):
    scan_seeder.start(roots=ROOTS, prune_first=True, phase=ScanPhase.FAST)
    _settle(scan_seeder)
    scan_seeder.start_after_node_list(ROOTS, compute_hashes=False)  # runs the prune
    _settle(scan_seeder)
    events, entered, release = _record_phases(scan_seeder, monkeypatch, hold="fast")
    scan_seeder.start(roots=ROOTS)
    assert entered.wait(5)

    scan_seeder.enqueue_scan(roots=("output",), phase=ScanPhase.FULL)
    release.set()
    _settle(scan_seeder)

    assert events[:2] == [("fast", frozenset(ROOTS)), ("enrich", frozenset(ROOTS))]


def test_a_seed_scan_before_the_first_page_load_keeps_its_enrichment(scan_seeder, monkeypatch):
    scan_seeder.start(roots=ROOTS, prune_first=True, phase=ScanPhase.FAST)
    _settle(scan_seeder)
    events, entered, release = _record_phases(scan_seeder, monkeypatch, hold="fast")
    scan_seeder.start(roots=("input",))  # e.g. POST /api/assets/seed, which waits for its enrichment
    assert entered.wait(5)

    scan_seeder.start_after_node_list(ROOTS, compute_hashes=False)
    release.set()
    _settle(scan_seeder)

    inputs, models = frozenset({"input"}), frozenset({"models"})
    assert events == [("fast", inputs), ("enrich", inputs), ("prune",), ("fast", models), ("enrich", models)]


def test_a_startup_scan_cancelled_after_its_fast_phase_hands_nothing_off(scan_seeder, monkeypatch):
    events, entered, release = _record_phases(scan_seeder, monkeypatch, hold="fast")
    cancelled = threading.Event()

    def cancel_once(kind, _data):
        if kind == "assets.seed.fast_complete" and not cancelled.is_set():
            cancelled.set()
            scan_seeder.cancel()

    scan_seeder.set_event_sink(cancel_once)
    scan_seeder.start(roots=ROOTS, prune_first=True)
    assert entered.wait(5)

    scan_seeder.start_after_node_list(ROOTS, compute_hashes=False)
    release.set()
    _settle(scan_seeder)

    models = frozenset({"models"})
    assert events == [("fast", frozenset(ROOTS)), ("prune",), ("fast", models), ("enrich", models)]

