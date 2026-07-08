from collections import deque
from datetime import datetime
import io
import logging
import sys
import threading

ANSI_NAMED_COLORS = {
    'black':   '\033[30m',
    'red':     '\033[31m',
    'green':   '\033[32m',
    'yellow':  '\033[33m',
    'blue':    '\033[34m',
    'magenta': '\033[35m',
    'cyan':    '\033[36m',
    'white':   '\033[37m',
}

ANSI_LEVEL_COLORS = {
    'DEBUG':    ANSI_NAMED_COLORS['cyan'],
    'INFO':     ANSI_NAMED_COLORS['green'],
    'WARNING':  ANSI_NAMED_COLORS['yellow'],
    'ERROR':    ANSI_NAMED_COLORS['red'],
    'CRITICAL': ANSI_NAMED_COLORS['magenta'],
}

ANSI_RESET = '\033[0m'
ANSI_BOLD  = '\033[1m'


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        color = ANSI_LEVEL_COLORS.get(record.levelname, '')
        bold  = ANSI_BOLD if record.levelno >= logging.WARNING else ''
        level_tag = f"{bold}{color}[{record.levelname}]{ANSI_RESET} "
        message = super().format(record)
        line_color = ANSI_NAMED_COLORS.get(getattr(record, 'color', ''), '')
        if line_color:
            return f"{level_tag}{line_color}{message}{ANSI_RESET}"
        return level_tag + message

logs = None
stdout_interceptor = None
stderr_interceptor = None


class LogInterceptor(io.TextIOWrapper):
    def __init__(self, stream,  *args, **kwargs):
        buffer = stream.buffer
        encoding = stream.encoding
        super().__init__(buffer, *args, **kwargs, encoding=encoding, line_buffering=stream.line_buffering)
        self._lock = threading.Lock()
        self._flush_callbacks = []
        self._logs_since_flush = []

    def write(self, data):
        entry = {"t": datetime.now().isoformat(), "m": data}
        with self._lock:
            self._logs_since_flush.append(entry)

            # Simple handling for cr to overwrite the last output if it isnt a full line
            # else logs just get full of progress messages
            if isinstance(data, str) and data.startswith("\r") and not logs[-1]["m"].endswith("\n"):
                logs.pop()
            logs.append(entry)
        super().write(data)

    def flush(self):
        super().flush()
        for cb in self._flush_callbacks:
            cb(self._logs_since_flush)
            self._logs_since_flush = []

    def on_flush(self, callback):
        self._flush_callbacks.append(callback)


def get_logs():
    return logs


def on_flush(callback):
    if stdout_interceptor is not None:
        stdout_interceptor.on_flush(callback)
    if stderr_interceptor is not None:
        stderr_interceptor.on_flush(callback)

def setup_logger(log_level: str = 'INFO', capacity: int = 300, use_stdout: bool = False):
    global logs
    if logs:
        return

    # Override output streams and log to buffer
    logs = deque(maxlen=capacity)

    global stdout_interceptor
    global stderr_interceptor
    stdout_interceptor = sys.stdout = LogInterceptor(sys.stdout)
    stderr_interceptor = sys.stderr = LogInterceptor(sys.stderr)

    # Setup default global logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    formatter = ColoredFormatter("%(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    if use_stdout:
        # Only errors and critical to stderr
        stream_handler.addFilter(lambda record: not record.levelno < logging.ERROR)

        # Lesser to stdout
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        stdout_handler.addFilter(lambda record: record.levelno < logging.ERROR)
        logger.addHandler(stdout_handler)

    logger.addHandler(stream_handler)


STARTUP_WARNINGS = []


def log_startup_warning(msg):
    logging.warning(msg)
    STARTUP_WARNINGS.append(msg)


def print_startup_warnings():
    for s in STARTUP_WARNINGS:
        logging.warning(s)
    STARTUP_WARNINGS.clear()
