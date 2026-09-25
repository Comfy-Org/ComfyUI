"""Let the event loop run while a scan thread is in a long pure-Python loop.

Without this, the event loop must win the GIL back from the scan after every socket
syscall, and a page load's requests queue behind the scan for seconds. time.sleep(0)
is not enough: the scan thread usually retakes the GIL before the loop wakes.
"""

import threading
import time

_INTERVAL = 0.005
_SLEEP = 0.001

_last = threading.local()


def yield_gil() -> None:
    """Call once per item in a hot loop on a background thread; sleeps every _INTERVAL."""
    now = time.perf_counter()
    last = getattr(_last, "t", None)
    if last is None:
        _last.t = now
    elif now - last >= _INTERVAL:
        time.sleep(_SLEEP)
        _last.t = time.perf_counter()
