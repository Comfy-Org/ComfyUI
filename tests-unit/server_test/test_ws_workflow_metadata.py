"""Tests for the workflow_metadata key/values added to outgoing websocket messages"""

import json

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


class TestValidWorkflowMetadata:
    """The dict is merged into outgoing messages, so it is validated at the
    request boundary. A client can also put a value straight into extra_data,
    which the route strips before applying the validated one."""

    def test_dict_within_the_limit_is_accepted(self):
        metadata = {"workflow_id": "abc"}
        assert server.valid_workflow_metadata(
            {"workflow_metadata": metadata}
        ) == metadata

    def test_absent_field(self):
        assert server.valid_workflow_metadata({}) is None

    @pytest.mark.parametrize("value", ["abc", 7, ["abc"], None, True])
    def test_non_dict_is_rejected(self, value):
        assert server.valid_workflow_metadata({"workflow_metadata": value}) is None

    def test_oversized_dict_is_rejected(self):
        assert (
            server.valid_workflow_metadata({"workflow_metadata": {"k": "v" * 300}})
            is None
        )

    def test_dict_at_the_limit_is_accepted(self):
        metadata = {"k": "v" * (256 - len('{"k": ""}'))}
        assert len(json.dumps(metadata)) == 256
        assert server.valid_workflow_metadata({"workflow_metadata": metadata}) == metadata

    def test_empty_dict_is_accepted_and_stamps_nothing(self, prompt_server):
        assert server.valid_workflow_metadata({"workflow_metadata": {}}) == {}
        prompt_server.workflow_metadata = {}
        prompt_server.send_sync("executing", {"prompt_id": "p1"})
        assert sent_data(prompt_server) == [{"prompt_id": "p1"}]
