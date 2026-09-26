import queue
import threading

import pytest

from app.assets.services import gil


SHARE = gil._SLEEP / (gil._SLEEP + gil._RUN)


class FakeClock:
    """Stands in for gil._clock / gil._sleep; a sleep advances the clock by `sleep_cost`."""

    def __init__(self, sleep_cost: float) -> None:
        self.now = 100.0
        self.sleep_cost = sleep_cost
        self.slept = 0.0
        self.sleeps = 0

    def clock(self) -> float:
        return self.now

    def sleep(self, _seconds: float) -> None:
        self.now += self.sleep_cost
        self.slept += self.sleep_cost
        self.sleeps += 1


@pytest.fixture
def fake(monkeypatch):
    def install(sleep_cost: float) -> FakeClock:
        clock = FakeClock(sleep_cost)
        monkeypatch.setattr(gil, "_clock", clock.clock)
        monkeypatch.setattr(gil, "_sleep", clock.sleep)
        monkeypatch.setattr(gil, "_state", threading.local())
        return clock

    return install


def started(fake, sleep_cost: float, yield_gil) -> FakeClock:
    clock = fake(sleep_cost)
    yield_gil()  # starts this thread's first run window
    return clock


def test_sleeps_only_once_the_run_window_has_passed(fake):
    clock = started(fake, gil._SLEEP, gil._yield_fixed)
    clock.now += gil._RUN / 2
    gil._yield_fixed()
    assert clock.sleeps == 0

    clock.now += gil._RUN
    gil._yield_fixed()
    assert clock.sleeps == 1


def test_run_window_restarts_after_each_sleep(fake):
    clock = started(fake, gil._SLEEP, gil._yield_fixed)
    clock.now += gil._RUN * 1.5
    gil._yield_fixed()
    gil._yield_fixed()
    assert clock.sleeps == 1


def run_hot_loop(clock: FakeClock, yield_gil, seconds: float, work_per_item: float = 0.0001) -> float:
    """Simulate a loop doing `work_per_item` per call; return the fraction spent asleep."""
    start = clock.now
    while clock.now - start < seconds:
        clock.now += work_per_item
        yield_gil()
    return clock.slept / (clock.now - start)


def test_accurate_sleep_keeps_the_measured_duty_cycle(fake):
    clock = started(fake, gil._SLEEP, gil._yield_fixed)
    assert run_hot_loop(clock, gil._yield_fixed, 10.0) == pytest.approx(SHARE, abs=0.02)


def test_coarse_sleep_widens_the_run_window_to_keep_the_duty_cycle(fake):
    # A 1ms sleep that really takes a 15ms timer tick, as on Windows before Python 3.11.
    clock = started(fake, 0.015, gil._yield_scaled)
    assert run_hot_loop(clock, gil._yield_scaled, 30.0) == pytest.approx(SHARE, abs=0.02)


def test_slightly_slow_sleep_scales_the_run_window_continuously(fake):
    clock = started(fake, 0.0015, gil._yield_scaled)
    assert run_hot_loop(clock, gil._yield_scaled, 10.0) == pytest.approx(SHARE, abs=0.02)


def test_very_coarse_sleep_still_yields_at_the_same_duty_cycle(fake):
    clock = started(fake, 0.040, gil._yield_scaled)
    assert run_hot_loop(clock, gil._yield_scaled, 60.0) == pytest.approx(SHARE, abs=0.02)


def test_threads_do_not_consume_each_others_run_window(fake):
    clock = started(fake, gil._SLEEP, gil._yield_fixed)
    sleeps_seen: list[int] = []

    def worker(inbox: queue.Queue, done: queue.Queue) -> None:
        while inbox.get():
            before = clock.sleeps
            gil._yield_fixed()
            sleeps_seen.append(clock.sleeps - before)
            done.put(True)

    threads = []
    for _ in range(2):
        inbox, done = queue.Queue(), queue.Queue()
        t = threading.Thread(target=worker, args=(inbox, done))
        t.start()
        threads.append((t, inbox, done))

    def call(i: int) -> int:
        _, inbox, done = threads[i]
        inbox.put(True)
        done.get(timeout=5)
        return sleeps_seen[-1]

    try:
        assert call(0) == 0 and call(1) == 0  # each thread starts its own run window
        clock.now += gil._RUN * 1.5
        assert call(0) == 1  # thread 0 yields and restarts only its own run window
        assert call(1) == 1  # thread 1's run window is untouched, so it yields too
    finally:
        for t, inbox, _ in threads:
            inbox.put(False)
            t.join(timeout=5)


def test_fixed_window_on_a_coarse_timer_sleeps_far_more_than_a_sixth(fake):
    # Why the scaled version exists: a 1ms sleep that really takes a 15ms tick.
    clock = started(fake, 0.015, gil._yield_fixed)
    assert run_hot_loop(clock, gil._yield_fixed, 30.0) > 2 * SHARE


def test_scaled_version_is_used_only_for_the_coarse_windows_timer():
    coarse = gil.sys.platform == "win32" and gil.sys.version_info < (3, 11)
    assert gil._COARSE_SLEEP is coarse
    assert gil.yield_gil is (gil._yield_scaled if coarse else gil._yield_fixed)
