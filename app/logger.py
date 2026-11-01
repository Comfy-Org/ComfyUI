from collections import deque
from datetime import datetime
import io
import logging
import os
import sys
import threading

import comfy.internal_logging

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
    'DETAIL':   ANSI_NAMED_COLORS['blue'],
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
            if isinstance(data, str) and data.startswith("\r") and logs and not logs[-1]["m"].endswith("\n"):
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


def get_log_level(level):
    return comfy.internal_logging.DETAIL if level == "DETAIL" else logging.getLevelName(level)


def setup_logger(log_level: str = 'INFO', file_outputs=None, capacity: int = 300, use_stdout: bool = False):
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
    if file_outputs is None:
        file_outputs = [('DETAIL', 'comfyui_detail.log')]
    logger = logging.getLogger()
    console_level = get_log_level(log_level)
    file_levels = [get_log_level(level) for level, _ in file_outputs]
    logger.setLevel(min([console_level, *file_levels]))

    formatter = ColoredFormatter("%(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(console_level)

    if use_stdout:
        # Only errors and critical to stderr
        stream_handler.addFilter(lambda record: not record.levelno < logging.ERROR)

        # Lesser to stdout
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        stdout_handler.setLevel(console_level)
        stdout_handler.addFilter(lambda record: record.levelno < logging.ERROR)
        logger.addHandler(stdout_handler)

    logger.addHandler(stream_handler)

    for output_level, output_path in file_outputs:
        output_path = os.path.abspath(output_path)
        try:
            output_handler = logging.FileHandler(output_path, encoding="utf-8")
        except OSError as e:
            logging.warning("Could not open %s log %s: %s", output_level, output_path, e)
            continue
        output_handler.setLevel(get_log_level(output_level))
        output_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        logger.addHandler(output_handler)
        logging.info("%s log: %s", output_level.title(), output_path)


STARTUP_WARNINGS = []


def log_startup_warning(msg):
    logging.warning(msg)
    STARTUP_WARNINGS.append(msg)


def print_startup_warnings():
    for s in STARTUP_WARNINGS:
        logging.warning(s)
    STARTUP_WARNINGS.clear()
