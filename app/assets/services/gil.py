"""Let the event loop run while a scan thread is in a long pure-Python loop.

Without this, the event loop must win the GIL back from the scan after every socket
syscall, and a page load's requests queue behind the scan for seconds. time.sleep(0)
is not enough: the scan thread usually retakes the GIL before the loop wakes.

The duty cycle is one _SLEEP per _INTERVAL of work. Where a short sleep overruns (on
Windows before Python 3.11 it waits a whole system timer tick, ~15.6ms by default),
the interval is widened to keep the same duty cycle, and yielding is switched off if
the sleep is too coarse to be worth it. The real sleep cost is measured once, on first
use, so no platform check is needed.
"""

import statistics
import threading
import time

_INTERVAL = 0.005
_SLEEP = 0.001
_MAX_SLEEP = 0.020
_CALIBRATION_SAMPLES = 5

# Indirection so tests can drive the clock without patching the time module.
_clock = time.perf_counter
_sleep = time.sleep

_UNCALIBRATED = -1.0
_interval: float | None = _UNCALIBRATED  # None: yielding disabled
_calibration_lock = threading.Lock()
_last = threading.local()


def _calibrate() -> float | None:
    samples = []
    for _ in range(_CALIBRATION_SAMPLES):
        start = _clock()
        _sleep(_SLEEP)
        samples.append(_clock() - start)
    actual = statistics.median(samples)
    if actual > _MAX_SLEEP:
        return None
    if actual <= 2 * _SLEEP:
        return _INTERVAL
    return _INTERVAL * actual / _SLEEP


def _yield_interval() -> float | None:
    global _interval
    if _interval == _UNCALIBRATED:
        with _calibration_lock:
            if _interval == _UNCALIBRATED:
                _interval = _calibrate()
    return _interval


def yield_gil() -> None:
    """Call once per item in a hot loop on a background thread; sleeps every interval."""
    interval = _yield_interval()
    if interval is None:
        return
    now = _clock()
    last = getattr(_last, "t", None)
    if last is None:
        _last.t = now
    elif now - last >= interval:
        _sleep(_SLEEP)
        _last.t = _clock()
