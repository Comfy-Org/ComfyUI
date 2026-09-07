import io

from app.logger import LogInterceptor


class FlushErrorBuffer(io.BytesIO):
    def flush(self):
        raise OSError(22, "Invalid argument")


class FlushErrorStream:
    buffer = FlushErrorBuffer()
    encoding = "utf-8"
    line_buffering = False


def test_log_interceptor_flush_ignores_oserror_and_runs_callbacks():
    interceptor = LogInterceptor(FlushErrorStream())
    interceptor._logs_since_flush = [{"m": "message"}]
    flushed_logs = []

    interceptor.on_flush(lambda logs: flushed_logs.append(list(logs)))
    interceptor.flush()

    assert flushed_logs == [[{"m": "message"}]]
    assert interceptor._logs_since_flush == []
