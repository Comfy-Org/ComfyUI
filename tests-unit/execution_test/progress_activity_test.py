from unittest.mock import Mock

import comfy.utils
from comfy_execution.progress import (
    NodeState,
    ProgressRegistry,
    WebUIProgressHandler,
)


def make_registry():
    dynprompt = Mock()
    dynprompt.get_display_node_id.return_value = "1"
    dynprompt.get_parent_node_id.return_value = None
    dynprompt.get_real_node_id.return_value = "1"
    return ProgressRegistry(prompt_id="p", dynprompt=dynprompt)


def test_activity_is_set_and_cleared():
    registry = make_registry()
    registry.start_progress("1")

    registry.set_activity("1", "loading")
    assert registry.nodes["1"]["activity"] == "loading"

    registry.set_activity("1", None)
    assert "activity" not in registry.nodes["1"]


def test_unchanged_activity_does_not_notify():
    registry = make_registry()
    handler = Mock(enabled=True)
    registry.handlers["test"] = handler
    registry.start_progress("1")

    registry.set_activity("1", "loading")
    assert handler.update_handler.call_count == 1

    registry.set_activity("1", "loading")
    assert handler.update_handler.call_count == 1

    registry.set_activity("1", None)
    assert handler.update_handler.call_count == 2


def test_activity_is_only_sent_when_set():
    server = Mock(client_id="c")
    registry = make_registry()
    handler = WebUIProgressHandler(server)
    handler.set_registry(registry)
    registry.register_handler(handler)

    registry.start_progress("1")
    nodes = server.send_sync.call_args[0][1]["nodes"]
    assert "activity" not in nodes["1"]

    registry.set_activity("1", "loading")
    nodes = server.send_sync.call_args[0][1]["nodes"]
    assert nodes["1"]["activity"] == "loading"
    assert nodes["1"]["state"] == NodeState.Running.value


def test_progress_activity_reports_and_clears():
    seen = []
    comfy.utils.set_progress_activity_global_hook(seen.append)
    try:
        with comfy.utils.progress_activity("loading"):
            assert seen == ["loading"]
    finally:
        comfy.utils.set_progress_activity_global_hook(None)
    assert seen == ["loading", None]


def test_progress_activity_clears_on_error():
    seen = []
    comfy.utils.set_progress_activity_global_hook(seen.append)
    try:
        with comfy.utils.progress_activity("loading"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    finally:
        comfy.utils.set_progress_activity_global_hook(None)
    assert seen == ["loading", None]


def test_progress_activity_without_hook():
    comfy.utils.set_progress_activity_global_hook(None)
    with comfy.utils.progress_activity("loading"):
        pass
