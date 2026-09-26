"""Let the event loop run while a scan thread is in a long pure-Python loop.

Without this, the event loop must win the GIL back from the scan after every socket
syscall, and a page load's requests queue behind the scan for seconds. time.sleep(0)
is not enough: the scan thread usually retakes the GIL before the loop wakes.

A thread sleeps _SLEEP after running _RUN, so it spends about a third of its time
asleep. Measured on a 41k-file first scan, that is where page loads stopped getting
faster (a shorter run bought nothing), for about 2% more scan time. All state is per
thread; nothing is shared.
"""

import sys
import threading
import time

_RUN = 0.002
_SLEEP = 0.001
# The coarse timer tick is ~15.6ms, so a 1ms sleep takes at most ~16x as long there.
_MAX_SCALE = 16.0

# Indirection so tests can drive the clock without patching the time module.
_clock = time.perf_counter
_sleep = time.sleep

_state = threading.local()


def _yield_fixed() -> None:
    """Call once per item in a hot loop on a background thread; sleeps every run window."""
    now = _clock()
    next_at = getattr(_state, "next_at", None)
    if next_at is None:
        _state.next_at = now + _RUN
        return
    if now < next_at:
        return
    _sleep(_SLEEP)
    _state.next_at = _clock() + _RUN


def _yield_scaled() -> None:
    """For a coarse sleep timer: time.sleep(_SLEEP) really takes a whole timer tick, so
    scale the next run by how long the last sleep took, keeping the same share asleep."""
    now = _clock()
    next_at = getattr(_state, "next_at", None)
    if next_at is None:
        _state.next_at = now + _RUN
        return
    if now < next_at:
        return
    _sleep(_SLEEP)
    after = _clock()
    # Scale up to the timer tick this exists for, but no further: a sleep that overshoots
    # because the machine is busy must not buy the scan a longer run.
    scale = min(max(1.0, (after - now) / _SLEEP), _MAX_SCALE)
    _state.next_at = after + _RUN * scale


# Before Python 3.11, time.sleep on Windows rounds up to the ~15.6ms system timer tick.
_COARSE_SLEEP = sys.platform == "win32" and sys.version_info < (3, 11)



def _no_yield() -> None:
    """Free-threaded Python: there is no GIL to hand back, so sleeping would only slow the scan."""


# sys._is_gil_enabled exists from Python 3.13; earlier versions always have the GIL.
_GIL_ENABLED = getattr(sys, "_is_gil_enabled", lambda: True)()

if not _GIL_ENABLED:
    yield_gil = _no_yield
elif _COARSE_SLEEP:
    yield_gil = _yield_scaled
else:
    yield_gil = _yield_fixed
