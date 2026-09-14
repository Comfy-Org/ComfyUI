import subprocess
import sys
import os

def test_prompt_worker_survives_exception():
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

class DummyPromptExecutor:
    def __init__(self, server_instance, cache_type, cache_args, asset_manager=None):
        self.history_result = {}
        self.success = True
        self.status_messages = []
        self.execute_called = 0

    def execute(self, *args, **kwargs):
        self.execute_called += 1
        raise Exception("Simulated unhandled exception!")

    def reset(self):
        pass

import execution
execution.PromptExecutor = DummyPromptExecutor
main.gc.collect = lambda: None
main.comfy.model_management.soft_empty_cache = lambda: None

q = PromptQueue(DummyServer())
prompt_tuple = (1, "prompt_id_123", {}, {}, [], {})
q.put(prompt_tuple)

class StopWorkerException(Exception): pass
original_get = q.get
def fake_get(timeout=None):
    if len(q.queue) == 0:
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

assert len(q.currently_running) == 0, "Queue item not completed"
print("SUCCESS")
"""
    env = os.environ.copy()
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env["PYTHONPATH"] = repo_root
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
    assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}\n{result.stdout}"
    assert "SUCCESS" in result.stdout

