"""Queued (post-prompt) scans wait as long as the last one took before starting again."""

import threading
import time
import types

import pytest

from app.assets import seeder as seeder_module
from app.assets.seeder import ScanPhase, State, _AssetSeeder

SETTLE_TIMEOUT = 5.0


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    perf_counter = monotonic


class FakeTimer:
    armed: list["FakeTimer"] = []

    def __init__(self, interval: float, function) -> None:
        self.interval = interval
        self.function = function
        self.daemon = False
        self.cancelled = False

    def start(self) -> None:
        FakeTimer.armed.append(self)

    def cancel(self) -> None:
        self.cancelled = True


class Harness:
    def __init__(self, seeder: _AssetSeeder, clock: FakeClock) -> None:
        self.seeder = seeder
        self.clock = clock
        self.scans: list[tuple[str, ...]] = []
        # Fake seconds each scan takes; the hook runs mid-scan (a prompt finishing meanwhile).
        self.scan_s = 0.0
        self.paused_s = 0.0
        self.during_scan = None
        self.scan_error: Exception | None = None

    def fast_phase(self, roots):
        self.scans.append(tuple(roots))
        self.clock.now += self.scan_s
        if self.scan_error is not None:
            raise self.scan_error
        assert self.seeder._scan_state is not None
        self.seeder._scan_state.paused_s = self.paused_s
        if self.during_scan is not None:
            hook, self.during_scan = self.during_scan, None
            hook()
        return 0, 0, 0

    def settle(self) -> None:
        deadline = time.monotonic() + SETTLE_TIMEOUT
        while time.monotonic() < deadline:
            with self.seeder._lock:
                thread = self.seeder._thread
                if self.seeder._state is State.IDLE and (thread is None or not thread.is_alive()):
                    assert self.scan_error is not None or self.seeder.get_status().errors == []
                    return
            if thread is not None:
                thread.join(timeout=0.05)
        pytest.fail("seeder did not settle")

    def queue_output_scan(self) -> bool:
        return self.seeder.enqueue_scan(roots=("output",), phase=ScanPhase.FULL)

    def live_timers(self) -> list[FakeTimer]:
        return [t for t in FakeTimer.armed if not t.cancelled and t.function is not None]

    def fire_timer(self) -> None:
        [timer] = self.live_timers()
        self.clock.now += timer.interval
        callback, timer.function = timer.function, None
        callback()
        self.settle()


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> Harness:
    clock = FakeClock()
    fake_time = types.SimpleNamespace(
        monotonic=clock.monotonic,
        perf_counter=clock.perf_counter,
        thread_time=time.thread_time,
    )
    fake_threading = types.SimpleNamespace(**vars(threading))
    fake_threading.Timer = FakeTimer
    FakeTimer.armed = []
    monkeypatch.setattr(seeder_module, "time", fake_time)
    monkeypatch.setattr(seeder_module, "threading", fake_threading)
    monkeypatch.setattr(seeder_module, "dependencies_available", lambda: True)
    monkeypatch.setattr(seeder_module, "emit", lambda *args, **kwargs: None)
    seeder = _AssetSeeder()
    h = Harness(seeder, clock)
    monkeypatch.setattr(seeder, "_log_scan_config", lambda roots: None)
    monkeypatch.setattr(seeder, "_run_fast_phase", h.fast_phase)
    monkeypatch.setattr(seeder, "_run_enrich_phase", lambda roots: (False, 0))
    yield h
    seeder.shutdown()


def test_fast_scans_start_right_after_the_next_prompt(harness: Harness) -> None:
    harness.scan_s = 0.2
    assert harness.queue_output_scan()
    harness.settle()
    harness.clock.now += 10  # main.py queues at most once per 10 s GC tick

    assert harness.queue_output_scan()
    harness.settle()

    assert len(harness.scans) == 2
    assert harness.live_timers() == []


def test_a_queued_scan_waits_as_long_as_the_last_one_took(harness: Harness) -> None:
    harness.scan_s = 30
    harness.queue_output_scan()
    harness.settle()
    harness.clock.now += 10

    assert harness.queue_output_scan() is False
    assert len(harness.scans) == 1
    [timer] = harness.live_timers()
    assert timer.interval == pytest.approx(20)

    harness.fire_timer()
    assert len(harness.scans) == 2

    # The timer slot is freed, so the next deferral arms a fresh one.
    harness.clock.now += 1
    assert harness.queue_output_scan() is False
    [timer] = harness.live_timers()
    assert timer.interval == pytest.approx(29)


def test_the_wait_is_capped(harness: Harness) -> None:
    harness.scan_s = 3600
    harness.queue_output_scan()
    harness.settle()

    harness.queue_output_scan()

    [timer] = harness.live_timers()
    assert timer.interval == pytest.approx(seeder_module._QUEUED_GAP_CAP_S)


def test_time_paused_for_prompts_does_not_count(harness: Harness) -> None:
    harness.scan_s = 30
    harness.paused_s = 25
    harness.queue_output_scan()
    harness.settle()

    harness.queue_output_scan()

    [timer] = harness.live_timers()
    assert timer.interval == pytest.approx(5)


def test_prompts_finishing_during_a_scan_merge_into_one_delayed_scan(harness: Harness) -> None:
    harness.scan_s = 30

    started_mid_scan: list[bool] = []

    def two_more_prompts_finish():
        started_mid_scan.extend([harness.queue_output_scan(), harness.queue_output_scan()])

    harness.during_scan = two_more_prompts_finish
    harness.queue_output_scan()
    harness.settle()
    assert started_mid_scan == [False, False]

    # Back to back on master; now the merged scan waits out the gap.
    assert len(harness.scans) == 1
    [timer] = harness.live_timers()
    assert timer.interval == pytest.approx(30)
    assert len(FakeTimer.armed) == 1

    harness.fire_timer()
    assert len(harness.scans) == 2
    assert harness.seeder._pending_scan is None


def test_a_startup_scan_is_not_held_behind_a_deferred_one(harness: Harness) -> None:
    harness.scan_s = 30
    harness.queue_output_scan()
    harness.settle()
    harness.queue_output_scan()
    assert len(harness.live_timers()) == 1

    not_before = harness.seeder._queued_not_before
    harness.scan_s = 5
    assert harness.seeder.start(roots=("models", "input", "output"))
    harness.settle()

    assert harness.scans[-1] == ("models", "input", "output")
    # Its end neither releases the deferred scan early nor moves the gap.
    assert len(harness.scans) == 2
    assert harness.seeder._queued_not_before == not_before
    harness.fire_timer()
    assert len(harness.scans) == 3


def test_a_timer_firing_mid_prompt_starts_the_scan_paused(harness: Harness) -> None:
    harness.scan_s = 30
    harness.queue_output_scan()
    harness.settle()
    harness.queue_output_scan()

    harness.seeder.pause()  # the next prompt starts while the seeder is idle
    [timer] = harness.live_timers()
    harness.clock.now += timer.interval
    timer.function()

    assert harness.seeder.get_status().state is State.PAUSED
    harness.seeder.resume()
    harness.settle()
    assert len(harness.scans) == 2


def test_a_held_scan_does_not_start_after_shutdown(harness: Harness) -> None:
    harness.scan_s = 30
    harness.queue_output_scan()
    harness.settle()
    harness.queue_output_scan()
    [timer] = harness.live_timers()

    harness.seeder.shutdown()
    harness.fire_timer()

    assert len(harness.scans) == 1
    # Not cancelled at shutdown, so it must not hold the interpreter open.
    assert timer.daemon is True


def test_a_prompt_after_the_gap_joins_the_waiting_scan(harness: Harness) -> None:
    harness.scan_s = 30
    harness.queue_output_scan()
    harness.settle()
    harness.queue_output_scan()
    [timer] = harness.live_timers()
    harness.clock.now += timer.interval + 1  # due, but the timer has not run yet

    harness.queue_output_scan()
    harness.settle()
    assert len(harness.scans) == 1
    timer.function()
    harness.settle()

    # One scan for both prompts, not a fresh one now and the old one a gap later.
    assert len(harness.scans) == 2
    assert harness.seeder._pending_scan is None


def test_a_scan_held_during_a_prune_starts_when_the_prune_ends(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness.scan_s = 30
    harness.queue_output_scan()
    harness.settle()
    harness.queue_output_scan()
    [timer] = harness.live_timers()

    threads_in_prune: list[object] = []

    def prune_while_the_timer_fires(_prefixes, _should_stop):
        thread_before = harness.seeder._thread
        harness.clock.now += timer.interval
        timer.function()
        threads_in_prune.append(harness.seeder._thread is thread_before)
        return 0

    monkeypatch.setattr(seeder_module, "get_owned_prefixes", lambda: [])
    monkeypatch.setattr(seeder_module, "mark_missing_outside_prefixes_safely", prune_while_the_timer_fires)
    harness.seeder.mark_missing_outside_prefixes()
    harness.settle()

    assert threads_in_prune == [True]  # nothing started while the prune held the seeder
    assert len(harness.scans) == 2


def test_a_page_load_scan_during_a_prompt_is_not_held(harness: Harness) -> None:
    harness.seeder.pause()  # a prompt is running and the seeder is idle

    assert harness.seeder.start(roots=("models", "input", "output"))

    assert harness.seeder.get_status().state is State.RUNNING
    harness.settle()
    assert len(harness.scans) == 1


def test_a_failed_scan_sets_no_wait(harness: Harness) -> None:
    harness.scan_s = 30
    harness.scan_error = RuntimeError("share went away")
    harness.queue_output_scan()
    harness.settle()

    harness.scan_error = None
    assert harness.queue_output_scan()
    harness.settle()
    assert len(harness.scans) == 2


def test_a_scan_left_queued_by_a_cancelled_prune_does_not_block_the_next(harness: Harness) -> None:
    # A cancelled prune resets to idle and keeps the scan queued during it, with no timer armed.
    harness.seeder._pending_scan = {"roots": ("output",), "phase": ScanPhase.FULL, "compute_hashes": False}

    assert harness.queue_output_scan()
    harness.settle()

    assert harness.scans[0] == ("output",)


def test_a_timer_firing_between_prompts_starts_the_scan_running(harness: Harness) -> None:
    harness.scan_s = 30
    harness.queue_output_scan()
    harness.settle()
    harness.queue_output_scan()

    harness.seeder.pause()
    harness.seeder.resume()  # that prompt ended before the gap did
    harness.during_scan = lambda: running.append(harness.seeder.get_status().state)
    running: list[State] = []
    harness.fire_timer()

    assert running == [State.RUNNING]


def test_a_timer_firing_during_a_startup_scan_leaves_it_running(harness: Harness) -> None:
    harness.scan_s = 30
    harness.queue_output_scan()
    harness.settle()
    harness.queue_output_scan()
    [timer] = harness.live_timers()
    seen: list[tuple[State, tuple[str, ...]]] = []

    def timer_fires():
        harness.clock.now += timer.interval
        timer.function()
        seen.append((harness.seeder.get_status().state, harness.seeder._roots))

    harness.during_scan = timer_fires
    harness.scan_s = 0
    harness.seeder.start(roots=("models", "input", "output"))
    harness.settle()

    assert seen == [(State.RUNNING, ("models", "input", "output"))]
    # The startup scan's end starts the held one, the gap being over.
    assert len(harness.scans) == 3
