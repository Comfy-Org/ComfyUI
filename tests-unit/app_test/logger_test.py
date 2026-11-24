import io
from collections import deque

from app import logger


def test_log_interceptor_keeps_stream_error_handler(monkeypatch):
    monkeypatch.setattr(logger, "logs", deque())
    stream = io.TextIOWrapper(io.BytesIO(), encoding="gbk", errors="backslashreplace")
    interceptor = logger.LogInterceptor(stream)

    interceptor.write("☁ BizyAir")
    interceptor.flush()

    assert stream.buffer.getvalue() == b"\\u2601 BizyAir"


def test_log_interceptor_keeps_explicit_error_handler(monkeypatch):
    monkeypatch.setattr(logger, "logs", deque())
    stream = io.TextIOWrapper(io.BytesIO(), encoding="gbk", errors="backslashreplace")
    interceptor = logger.LogInterceptor(stream, errors="replace")

    interceptor.write("☁ BizyAir")
    interceptor.flush()

    assert stream.buffer.getvalue() == b"? BizyAir"
