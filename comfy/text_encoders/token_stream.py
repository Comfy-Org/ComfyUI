from contextlib import contextmanager
from contextvars import ContextVar


_token_callback = ContextVar("generated_token_callback", default=None)


@contextmanager
def stream_generated_tokens(callback):
    """Observe the growing sequence of committed token IDs during generation."""
    token = _token_callback.set(callback)
    try:
        yield
    finally:
        _token_callback.reset(token)


def report_generated_tokens(ids):
    callback = _token_callback.get()
    if callback is not None:
        callback(ids)
