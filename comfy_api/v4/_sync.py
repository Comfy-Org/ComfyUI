"""The sync form of the V4 API.

`comfy_api.internal.async_to_sync` builds a sync API by running each call on a
pool thread with its own event loop. V4 cannot use that. Its calls are engine
work, and PyTorch's inference, grad and autocast modes belong to the thread, so a
call moved to another thread would lose the node's.

This is not a copy of that converter: it has no thread, no event loop and no stub
generation. A V4 call does not wait on the event loop, so its own body is run
where it is called, and a call that would wait is refused.
"""

import functools
import inspect


def _finished(method):
    @functools.wraps(method)
    def call(*args, **kwargs):
        coroutine = method(*args, **kwargs)
        try:
            coroutine.send(None)
        except StopIteration as finished:
            return finished.value
        finally:
            coroutine.close()
        raise RuntimeError(f"{method.__qualname__} needs an event loop; await it")

    return call


class SyncGroup:
    """An API group's declared calls, each finished on the calling thread."""

    def __init__(self, declared: type, group: object):
        for name, _ in inspect.getmembers(declared, inspect.iscoroutinefunction):
            setattr(self, name, _finished(getattr(group, name)))
