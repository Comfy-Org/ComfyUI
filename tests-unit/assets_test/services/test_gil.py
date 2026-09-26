import queue
import threading

import pytest

from app.assets.services import gil


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


def started(fake, sleep_cost: float) -> FakeClock:
    clock = fake(sleep_cost)
    gil.yield_gil()  # starts this thread's first run window
    return clock


def test_sleeps_only_once_the_run_window_has_passed(fake):
    clock = started(fake, gil._SLEEP)
    clock.now += gil._RUN / 2
    gil.yield_gil()
    assert clock.sleeps == 0

    clock.now += gil._RUN
    gil.yield_gil()
    assert clock.sleeps == 1


def test_run_window_restarts_after_each_sleep(fake):
    clock = started(fake, gil._SLEEP)
    clock.now += gil._RUN * 1.5
    gil.yield_gil()
    gil.yield_gil()
    assert clock.sleeps == 1


def run_hot_loop(clock: FakeClock, seconds: float, work_per_item: float = 0.0001) -> float:
    """Simulate a loop doing `work_per_item` per call; return the fraction spent asleep."""
    start = clock.now
    while clock.now - start < seconds:
        clock.now += work_per_item
        gil.yield_gil()
    return clock.slept / (clock.now - start)


def test_accurate_sleep_keeps_the_measured_duty_cycle(fake):
    clock = started(fake, gil._SLEEP)
    assert run_hot_loop(clock, 10.0) == pytest.approx(1 / 6, abs=0.02)


def test_coarse_sleep_widens_the_run_window_to_keep_the_duty_cycle(fake):
    # A 1ms sleep that really takes a 15ms timer tick, as on Windows before Python 3.11.
    clock = started(fake, 0.015)
    assert run_hot_loop(clock, 30.0) == pytest.approx(1 / 6, abs=0.02)


def test_slightly_slow_sleep_scales_the_run_window_continuously(fake):
    clock = started(fake, 0.0015)
    assert run_hot_loop(clock, 10.0) == pytest.approx(1 / 6, abs=0.02)


def test_very_coarse_sleep_still_yields_at_the_same_duty_cycle(fake):
    clock = started(fake, 0.040)
    assert run_hot_loop(clock, 60.0) == pytest.approx(1 / 6, abs=0.02)


def test_threads_do_not_consume_each_others_run_window(fake):
    clock = started(fake, gil._SLEEP)
    sleeps_seen: list[int] = []

    def worker(inbox: queue.Queue, done: queue.Queue) -> None:
        while inbox.get():
            before = clock.sleeps
            gil.yield_gil()
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
