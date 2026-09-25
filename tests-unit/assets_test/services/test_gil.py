import threading

import pytest

from app.assets.services import gil


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    sleeps: list[float] = []
    monkeypatch.setattr(gil.time, "perf_counter", lambda: now[0])
    monkeypatch.setattr(gil.time, "sleep", sleeps.append)
    monkeypatch.setattr(gil, "_last", threading.local())
    return now, sleeps


def test_sleeps_only_once_the_interval_has_passed(clock):
    now, sleeps = clock
    gil.yield_gil()
    now[0] += gil._INTERVAL / 2
    gil.yield_gil()
    assert sleeps == []

    now[0] += gil._INTERVAL * 1.5
    gil.yield_gil()
    assert sleeps == [gil._SLEEP]


def test_interval_restarts_after_each_sleep(clock):
    now, sleeps = clock
    gil.yield_gil()
    now[0] += gil._INTERVAL * 1.5
    gil.yield_gil()
    gil.yield_gil()
    assert sleeps == [gil._SLEEP]

