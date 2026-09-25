"""Let the event loop run while a scan thread is in a long pure-Python loop.

Without this, the event loop must win the GIL back from the scan after every socket
syscall, and a page load's requests queue behind the scan for seconds. time.sleep(0)
is not enough: the scan thread usually retakes the GIL before the loop wakes.

The interval scales with what one _SLEEP really costs, measured once on first use, so
a thread sleeps one sixth of the time whatever the platform's sleep resolution. Only
the window size changes: 1ms every 6ms here, but a ~15.8ms timer tick every ~95ms on
Windows before Python 3.11. No platform check is needed.
"""

import statistics
import threading
import time

_INTERVAL = 0.005
_SLEEP = 0.001
_CALIBRATION_SAMPLES = 5

# Indirection so tests can drive the clock without patching the time module.
_clock = time.perf_counter
_sleep = time.sleep

_UNCALIBRATED = -1.0
_interval = _UNCALIBRATED
_calibration_lock = threading.Lock()
_last = threading.local()


def _calibrate() -> float:
    samples = []
    for _ in range(_CALIBRATION_SAMPLES):
        start = _clock()
        _sleep(_SLEEP)
        samples.append(_clock() - start)
    return _INTERVAL * max(1.0, statistics.median(samples) / _SLEEP)


def _yield_interval() -> float:
    global _interval
    if _interval == _UNCALIBRATED:
        with _calibration_lock:
            if _interval == _UNCALIBRATED:
                _interval = _calibrate()
    return _interval


def yield_gil() -> None:
    """Call once per item in a hot loop on a background thread; sleeps every interval."""
    interval = _yield_interval()
    now = _clock()
    last = getattr(_last, "t", None)
    if last is None:
        _last.t = now
    elif now - last >= interval:
        _sleep(_SLEEP)
        _last.t = _clock()
