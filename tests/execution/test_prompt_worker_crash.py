import subprocess
import sys
import os

def test_prompt_worker_survives_exception():
    """Verify prompt_worker recovers after an unhandled exception and processes subsequent prompts."""
    script = """
import sys
sys.argv = ['main.py', '--disable-all-custom-nodes', '--cpu']
import main
from execution import PromptQueue

class DummyServer:
    def __init__(self):
        self.last_prompt_id = None
        self.client_id = None
    def send_sync(self, *args, **kwargs):
        pass
    def queue_updated(self, *args, **kwargs):
        pass

_executor_instance = None

class DummyPromptExecutor:
    def __init__(self, server_instance, cache_type, cache_args, asset_manager=None):
        global _executor_instance
        self.history_result = {}
        self.success = True
        self.status_messages = []
        self.execute_called = 0
        _executor_instance = self

    def execute(self, *args, **kwargs):
        'Raise only on the first call to simulate a one-time crash; succeed afterwards.'
        self.execute_called += 1
        if self.execute_called == 1:
            raise Exception("Simulated unhandled exception!")
        else:
            self.success = True
            self.history_result = {"mock_key": "mock_value_2"}

    def reset(self):
        pass

import execution
execution.PromptExecutor = DummyPromptExecutor
main.gc.collect = lambda: None
main.comfy.model_management.soft_empty_cache = lambda: None

q = PromptQueue(DummyServer())
q.put((1, "prompt_id_crash", {}, {}, [], {}))
q.put((2, "prompt_id_ok",    {}, {}, [], {}))

task_done_calls = []
original_task_done = q.task_done
def fake_task_done(item_id, output, status, process_item=None):
    task_done_calls.append((item_id, output, status))
    original_task_done(item_id, output, status=status, process_item=process_item)
q.task_done = fake_task_done

class StopWorkerException(Exception): pass
original_get = q.get
def fake_get(timeout=None):
    if len(q.queue) == 0 and len(q.currently_running) == 0:
        raise StopWorkerException("Stop")
    return original_get(timeout=timeout)
q.get = fake_get

class DummyAssetManager:
    def pause_background_scan(self):
        pass
    def resume_background_scan(self):
        pass
    def queue_output_scan(self):
        pass

try:
    main.prompt_worker(q, DummyServer(), DummyAssetManager())
except StopWorkerException:
    pass

assert len(q.currently_running) == 0, (
    f"Queue items not completed; currently_running={q.currently_running}"
)
assert _executor_instance is not None, "PromptExecutor was never instantiated"
assert _executor_instance.execute_called == 2, (
    f"Expected execute to be called twice (crash + recovery), got {_executor_instance.execute_called}"
)

assert len(task_done_calls) == 2, f"Expected 2 task_done calls, got {len(task_done_calls)}"

id1, out1, status1 = task_done_calls[0]
assert status1.status_str == 'error', f"First prompt should have error status, got {status1.status_str}"
assert not status1.completed, "First prompt should not be completed"

id2, out2, status2 = task_done_calls[1]
assert status2.status_str == 'success', f"Second prompt should have success status, got {status2.status_str}"
assert status2.completed, "Second prompt should be completed"
assert out2 == {"mock_key": "mock_value_2"}, "Second prompt should have distinct history_result"

print("SUCCESS")
"""
    env = os.environ.copy()
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env["PYTHONPATH"] = repo_root
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
    assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}\n{result.stdout}"
    assert "SUCCESS" in result.stdout
