from collections import deque
import io

import app.logger


def test_carriage_return_can_be_first_log_entry(monkeypatch):
    monkeypatch.setattr(app.logger, "logs", deque())
    stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    interceptor = app.logger.LogInterceptor(stream)

    interceptor.write("\rprogress")

    assert len(app.logger.logs) == 1
    assert app.logger.logs[0]["m"] == "\rprogress"
