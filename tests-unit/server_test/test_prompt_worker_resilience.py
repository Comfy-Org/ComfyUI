import sys
import importlib.util
from unittest.mock import MagicMock, patch
import pytest

# On some Windows environments, pyav C-extension can fail to initialize during test import.
# Ensure 'av' submodules resolve to mocks if native av is not present.
class _MockFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("av"):
            m = MagicMock()
            m.__path__ = []
            return importlib.util.spec_from_loader(fullname, loader=_MockLoader(m))
        return None

class _MockLoader:
    def __init__(self, m):
        self.m = m
    def create_module(self, spec):
        return self.m
    def exec_module(self, module):
        pass

if "av" not in sys.modules:
    sys.meta_path.insert(0, _MockFinder())

import comfy.cli_args
comfy.cli_args.args.cpu = True

from main import prompt_worker


class StopWorkerLoop(BaseException):
    """Custom exception used to cleanly exit the infinite while loop in prompt_worker."""
    pass


def test_prompt_worker_survives_execute_exception():
    """Verify that unhandled exceptions escaping PromptExecutor.execute do not kill prompt_worker.

    When an exception escapes execute(), prompt_worker must record the prompt as failed,
    complete the queue task, and remain running to process subsequent prompts in the queue.
    """
    queue_calls = []

    class MockQueue:
        def __init__(self):
            self.tasks_done = []
            # item tuple: (number, prompt_id, prompt, extra_data, outputs_to_execute, sensitive)
            self._items = [
                ((1, "prompt-fail-id", {}, {}, [], {}), 1),
                ((2, "prompt-success-id", {}, {}, [], {}), 2),
            ]

        def get(self, timeout=None):
            if self._items:
                return self._items.pop(0)
            raise StopWorkerLoop()

        def task_done(self, item_id, history_result, status=None, process_item=None):
            self.tasks_done.append({
                "item_id": item_id,
                "history_result": history_result,
                "status": status,
            })

        def get_flags(self):
            return {}

    mock_queue = MockQueue()
    mock_server = MagicMock()
    mock_server.client_id = "test-client"
    mock_asset_manager = MagicMock()

    class MockExecutor:
        def __init__(self, *args, **kwargs):
            self.reset()

        def reset(self):
            self.success = True
            self.status_messages = []
            self.history_result = {}

        def execute(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
            queue_calls.append(prompt_id)
            if prompt_id == "prompt-fail-id":
                raise RuntimeError("Simulated framework exception escaping execute_async")
            self.success = True
            self.history_result = {"outputs": {"node1": "out"}}

    with patch("execution.PromptExecutor", MockExecutor):
        with pytest.raises(StopWorkerLoop):
            prompt_worker(mock_queue, mock_server, mock_asset_manager)

    # 1. Verify both prompts were attempted
    assert queue_calls == ["prompt-fail-id", "prompt-success-id"]

    # 2. Verify both tasks were marked done
    assert len(mock_queue.tasks_done) == 2

    # 3. Verify the failed prompt was recorded with error status
    failed_task = mock_queue.tasks_done[0]
    assert failed_task["item_id"] == 1
    assert failed_task["status"].status_str == "error"
    assert failed_task["status"].completed is False

    # 4. Verify the successful prompt was processed normally after the failure
    success_task = mock_queue.tasks_done[1]
    assert success_task["item_id"] == 2
    assert success_task["status"].status_str == "success"
    assert success_task["status"].completed is True
    assert success_task["history_result"] == {"outputs": {"node1": "out"}}

    # 5. Verify server notifications were sent for both prompts
    executing_notifications = [
        call for call in mock_server.send_sync.call_args_list
        if call.args and call.args[0] == "executing"
    ]
    assert len(executing_notifications) >= 2
    prompt_ids_notified = [call.args[1]["prompt_id"] for call in executing_notifications]
    assert "prompt-fail-id" in prompt_ids_notified
    assert "prompt-success-id" in prompt_ids_notified
