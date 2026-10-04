"""Tests for the workflow_metadata key/values added to outgoing websocket messages"""

import pytest

import server
from comfy_api.feature_flags import SERVER_FEATURE_FLAGS  # noqa: F401


class FakeLoop:
    def call_soon_threadsafe(self, callback, *args):
        callback(*args)


@pytest.fixture
def prompt_server():
    instance = server.PromptServer.__new__(server.PromptServer)
    instance.loop = FakeLoop()
    instance.workflow_metadata = {}
    instance.messages = type("Queue", (), {"sent": [], "put_nowait": lambda self, msg: self.sent.append(msg)})()
    return instance


def sent_data(prompt_server):
    return [data for _, data, _ in prompt_server.messages.sent]


def test_metadata_is_added_to_messages(prompt_server):
    prompt_server.workflow_metadata = {"workflow_id": "abc"}
    prompt_server.send_sync("executing", {"node": "1", "prompt_id": "p1"})
    assert sent_data(prompt_server) == [{"workflow_id": "abc", "node": "1", "prompt_id": "p1"}]


def test_no_metadata_leaves_messages_unchanged(prompt_server):
    prompt_server.send_sync("executing", {"node": "1"})
    assert sent_data(prompt_server) == [{"node": "1"}]
    assert "workflow_id" not in sent_data(prompt_server)[0]


def test_message_fields_win_over_metadata(prompt_server):
    prompt_server.workflow_metadata = {"node": "spoofed", "workflow_id": "abc"}
    prompt_server.send_sync("executed", {"node": "1"})
    assert sent_data(prompt_server)[0]["node"] == "1"


def test_status_messages_are_not_touched(prompt_server):
    prompt_server.workflow_metadata = {"workflow_id": "abc"}
    prompt_server.send_sync("status", {"status": {"exec_info": {}}})
    assert sent_data(prompt_server) == [{"status": {"exec_info": {}}}]


def test_binary_messages_are_not_touched(prompt_server):
    prompt_server.workflow_metadata = {"workflow_id": "abc"}
    prompt_server.send_sync(server.BinaryEventTypes.TEXT, "some text")
    assert sent_data(prompt_server) == ["some text"]


def test_metadata_is_captured_when_the_message_is_queued(prompt_server):
    prompt_server.workflow_metadata = {"workflow_id": "first"}
    prompt_server.send_sync("execution_success", {"prompt_id": "p1"})
    prompt_server.workflow_metadata = {"workflow_id": "second"}
    assert sent_data(prompt_server)[0]["workflow_id"] == "first"
