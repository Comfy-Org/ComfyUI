import io
from collections import deque

from app import logger


def test_flush_delivers_the_same_batch_to_each_listener(monkeypatch):
    monkeypatch.setattr(logger, "logs", deque(maxlen=300))
    with io.TextIOWrapper(io.BytesIO()) as output:
        stream = logger.LogInterceptor(output)
        first, second = [], []

        def on_flush(entries):
            first.extend(entry["m"] for entry in entries)
            if entries and entries[0]["m"] == "one":
                stream.write("two")

        stream.on_flush(on_flush)
        stream.on_flush(lambda entries: second.extend(entry["m"] for entry in entries))
        stream.write("one")
        stream.flush()
        assert first == second == ["one"]
        stream.flush()
        assert first == second == ["one", "two"]
