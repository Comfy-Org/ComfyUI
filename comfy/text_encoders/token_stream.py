"""The tokens a text model has generated so far, for a caller that shows the text while it is written.

A caller wraps a generate() call in `listen(fn)`; the generate loops `report` their token ids after each step, and
`fn` gets them. With no listener, `report` does nothing.
"""

import contextlib
import contextvars
from typing import Callable, Optional, Sequence

_listener: contextvars.ContextVar[Optional[Callable[[Sequence[int]], None]]] = contextvars.ContextVar(
    "generated_tokens_listener", default=None)


@contextlib.contextmanager
def listen(fn: Callable[[Sequence[int]], None]):
    """Pass the token ids generated so far to `fn` after each generation step, inside this block."""
    token = _listener.set(fn)
    try:
        yield
    finally:
        _listener.reset(token)


def report(token_ids: Sequence[int]) -> None:
    """The generated token ids so far, for the listener, if there is one."""
    fn = _listener.get()
    if fn is not None:
        fn(token_ids)
