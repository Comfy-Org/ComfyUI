import subprocess
import sys
from pathlib import Path


def _run_prompt_worker(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        check=False,
    )


def test_prompt_worker_resumes_background_scan_when_execute_raises() -> None:
    script = """
import sys

sys.argv = ["main.py", "--cpu"]

import main

class Queue:
    def get(self, timeout=None):
        return (0, "prompt-id", {}, {}, [], {}), 1

class Server:
    last_prompt_id = None
    client_id = None

class BackgroundScan:
    paused = False

    def pause_background_scan(self):
        self.paused = True

    def resume_background_scan(self):
        self.paused = False

class Executor:
    def __init__(self, *args, **kwargs):
        self.history_result = {}
        self.success = True
        self.status_messages = []

    def execute(self, *args, **kwargs):
        raise RuntimeError("forced execute failure")

main.args.cache_classic = True
main.execution.PromptExecutor = Executor
asset_manager = BackgroundScan()

try:
    main.prompt_worker(Queue(), Server(), asset_manager)
except RuntimeError as error:
    assert str(error) == "forced execute failure"

assert asset_manager.paused is False
"""

    result = _run_prompt_worker(script)

    assert result.returncode == 0, result.stderr


def test_prompt_worker_resumes_background_scan_when_completion_raises() -> None:
    script = """
import sys

sys.argv = ["main.py", "--cpu"]

import main

class Queue:
    def get(self, timeout=None):
        return (0, "prompt-id", {}, {}, [], {}), 1

    def task_done(self, *args, **kwargs):
        raise RuntimeError("forced completion failure")

class Server:
    last_prompt_id = None
    client_id = None

class BackgroundScan:
    paused = False

    def pause_background_scan(self):
        self.paused = True

    def resume_background_scan(self):
        self.paused = False

class Executor:
    def __init__(self, *args, **kwargs):
        self.history_result = {}
        self.success = True
        self.status_messages = []

    def execute(self, *args, **kwargs):
        return None

main.args.cache_classic = True
main.execution.PromptExecutor = Executor
asset_manager = BackgroundScan()

try:
    main.prompt_worker(Queue(), Server(), asset_manager)
except RuntimeError as error:
    assert str(error) == "forced completion failure"

assert asset_manager.paused is False
"""

    result = _run_prompt_worker(script)

    assert result.returncode == 0, result.stderr


def test_prompt_worker_preserves_execute_error_when_resume_raises() -> None:
    script = """
import sys

sys.argv = ["main.py", "--cpu"]

import main

class Queue:
    def get(self, timeout=None):
        return (0, "prompt-id", {}, {}, [], {}), 1

class Server:
    last_prompt_id = None
    client_id = None

class BackgroundScan:
    paused = False

    def pause_background_scan(self):
        self.paused = True

    def resume_background_scan(self):
        self.paused = False
        raise RuntimeError("forced resume failure")

class Executor:
    def __init__(self, *args, **kwargs):
        pass

    def execute(self, *args, **kwargs):
        raise RuntimeError("forced execute failure")

main.args.cache_classic = True
main.execution.PromptExecutor = Executor
asset_manager = BackgroundScan()

try:
    main.prompt_worker(Queue(), Server(), asset_manager)
except RuntimeError as error:
    assert str(error) == "forced execute failure"
else:
    raise AssertionError("prompt_worker should propagate the execute failure")

assert asset_manager.paused is False
"""

    result = _run_prompt_worker(script)

    assert result.returncode == 0, result.stderr
