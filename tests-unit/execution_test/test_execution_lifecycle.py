import asyncio
import threading

import pytest

from comfy.cli_args import args
from comfy_api.latest import ExecutionLifecycle
from comfy_api.latest._execution_lifecycle import (
    CancelledEvent,
    FailedEvent,
    FailureSource,
    InterruptedEvent,
    LifecycleHandler,
    QueuedEvent,
    StartedEvent,
    SucceededEvent,
)
from comfy_execution import lifecycle


args.cpu = True


@pytest.fixture(autouse=True)
def reset_lifecycle():
    lifecycle._reset_for_tests()
    yield
    lifecycle._reset_for_tests()


def queued_event(prompt_id="prompt-1", metadata=None):
    return QueuedEvent(
        prompt_id=prompt_id,
        queue_number=1.0,
        output_node_ids=("2",),
        metadata=metadata or lifecycle._EMPTY_MAPPING,
        queued_at_ms=100,
    )


def all_events():
    metadata = lifecycle._EMPTY_MAPPING
    return [
        queued_event(metadata=metadata),
        StartedEvent("prompt-1", metadata, 100, 110),
        SucceededEvent("prompt-1", metadata, {}, 100, 110, 120, 10),
        FailedEvent(
            "prompt-1", metadata, FailureSource.NODE, "1", "Node", "ValueError",
            "failed", (), (), None, True, 100, 110, 115, 120, 10,
        ),
        InterruptedEvent("prompt-1", metadata, "1", "Node", (), None, 100, 110, 115, 120, 10),
        CancelledEvent("prompt-1", metadata, 100, 120),
    ]


def test_default_handler_methods_are_noops():
    handler = LifecycleHandler()
    events = all_events()

    assert handler.accepts(events[0]) is True
    handler.on_queued(events[0])
    handler.on_started(events[1])
    handler.on_succeeded(events[2])
    handler.on_failed(events[3])
    handler.on_interrupted(events[4])
    handler.on_cancelled(events[5])


def test_public_namespace_registers_handler():
    class Handler(ExecutionLifecycle.Handler):
        def on_failed(self, event):
            pass

    handler = Handler()
    asyncio.run(ExecutionLifecycle().register(handler))

    assert lifecycle._pending_registrations[0].handler is handler


def test_each_event_routes_to_matching_method_in_fifo_order():
    calls = []

    class Handler(LifecycleHandler):
        def on_queued(self, event):
            calls.append("queued")

        def on_started(self, event):
            calls.append("started")

        def on_succeeded(self, event):
            calls.append("succeeded")

        def on_failed(self, event):
            calls.append("failed")

        def on_interrupted(self, event):
            calls.append("interrupted")

        def on_cancelled(self, event):
            calls.append("cancelled")

    lifecycle._register_handler(Handler())
    lifecycle._freeze_handler_routes()
    for event in all_events():
        lifecycle._publish(event)

    assert lifecycle._wait_for_dispatch_for_tests()
    assert calls == ["queued", "started", "succeeded", "failed", "interrupted", "cancelled"]


def test_registration_order_filtering_and_freeze():
    calls = []

    class Handler(LifecycleHandler):
        def __init__(self, name, accepted=True):
            self.name = name
            self.accepted = accepted

        def accepts(self, event):
            return self.accepted

        def on_queued(self, event):
            calls.append(self.name)

    first = Handler("first")
    rejected = Handler("rejected", False)
    second = Handler("second")
    lifecycle._register_handler(first)
    lifecycle._register_handler(rejected)
    lifecycle._register_handler(second)
    lifecycle._register_handler(first)
    lifecycle._freeze_handler_routes()
    lifecycle._freeze_handler_routes()
    lifecycle._publish(queued_event())

    assert lifecycle._wait_for_dispatch_for_tests()
    assert calls == ["first", "second"]
    assert isinstance(lifecycle._frozen_routes[QueuedEvent], tuple)
    with pytest.raises(RuntimeError):
        lifecycle._register_handler(Handler("late"))


@pytest.mark.parametrize("phase", ["accepts", "callback"])
@pytest.mark.parametrize("exception_type", [RuntimeError, BaseException])
def test_handler_failure_does_not_stop_other_handlers_or_events(phase, exception_type):
    calls = []

    class FailingHandler(LifecycleHandler):
        def accepts(self, event):
            if phase == "accepts":
                raise exception_type("accepts failed")
            return True

        def on_queued(self, event):
            raise exception_type("callback failed")

    class AcceptedHandler(LifecycleHandler):
        def on_queued(self, event):
            calls.append((event.prompt_id, threading.current_thread().name))

    lifecycle._register_handler(FailingHandler())
    lifecycle._register_handler(AcceptedHandler())
    lifecycle._freeze_handler_routes()
    lifecycle._publish(queued_event())
    lifecycle._publish(queued_event("prompt-2"))

    assert lifecycle._wait_for_dispatch_for_tests()
    assert calls == [
        ("prompt-1", "execution-lifecycle"),
        ("prompt-2", "execution-lifecycle"),
    ]


def test_handler_registration_validation_and_no_handler_fast_path():
    class EmptyHandler(LifecycleHandler):
        pass

    class AsyncHandler(LifecycleHandler):
        async def on_queued(self, event):
            pass

    class AsyncFilterHandler(LifecycleHandler):
        async def accepts(self, event):
            return True

        def on_queued(self, event):
            pass

    with pytest.raises(TypeError):
        lifecycle._register_handler(lambda event: None)
    with pytest.raises(TypeError):
        lifecycle._register_handler(EmptyHandler())
    with pytest.raises(TypeError):
        lifecycle._register_handler(AsyncHandler())
    with pytest.raises(TypeError):
        lifecycle._register_handler(AsyncFilterHandler())

    lifecycle._freeze_handler_routes()
    assert lifecycle._has_any_handlers() is False
    assert lifecycle._dispatch_thread is None


def test_metadata_is_recursively_frozen_and_limited_to_reserved_namespace():
    extra_data = {
        "execution_lifecycle": {
            "business_task_id": "task-1",
            "listeners": ["business-status"],
            "nested": {"values": [1, 2]},
        },
        "client_id": "secret",
    }

    metadata = lifecycle._freeze_metadata(extra_data)
    extra_data["execution_lifecycle"]["business_task_id"] = "changed"
    extra_data["execution_lifecycle"]["nested"]["values"].append(3)

    assert metadata["business_task_id"] == "task-1"
    assert metadata["listeners"] == ("business-status",)
    assert metadata["nested"]["values"] == (1, 2)
    assert "client_id" not in metadata
    with pytest.raises(TypeError):
        metadata["business_task_id"] = "mutated"


def test_invalid_and_empty_snapshots_are_distinct():
    assert lifecycle._freeze_metadata({"execution_lifecycle": {"value": object()}}) == {}
    assert lifecycle._freeze_history_result({}) == {}
    assert lifecycle._freeze_history_result({"unsupported": object()}) is None


def test_history_snapshot_is_frozen_and_detached():
    history_result = {
        "outputs": {
            "2": {"lifecycle_test": [{"nested": {"values": [1, 2]}}]},
        },
    }

    snapshot = lifecycle._freeze_history_result(history_result)
    history_result["outputs"]["2"]["lifecycle_test"][0]["nested"]["values"].append(3)

    assert snapshot["outputs"]["2"]["lifecycle_test"][0]["nested"]["values"] == (1, 2)
    with pytest.raises(TypeError):
        snapshot["outputs"]["2"] = {}


def test_execution_context_publishes_started_then_succeeded():
    events = []

    class Handler(LifecycleHandler):
        def on_started(self, event):
            events.append(event)

        def on_succeeded(self, event):
            events.append(event)

    metadata = lifecycle._freeze_metadata({"execution_lifecycle": {"business_task_id": "task-1"}})
    context = lifecycle._QueuedLifecycleContext(100, metadata)
    lifecycle._register_handler(Handler())
    lifecycle._freeze_handler_routes()

    with lifecycle._PromptExecutionLifecycle("prompt-1", context) as execution_lifecycle:
        execution_lifecycle.record_executor_result(
            [("execution_success", {"prompt_id": "prompt-1", "timestamp": 200})],
            {"outputs": {"2": {"images": [{"filename": "result.png"}]}}},
        )
        execution_lifecycle.mark_task_done_succeeded()

    assert lifecycle._wait_for_dispatch_for_tests()
    assert [type(event) for event in events] == [StartedEvent, SucceededEvent]
    assert events[0].metadata is metadata
    assert events[1].metadata is metadata
    assert events[1].history_result["outputs"]["2"]["images"][0]["filename"] == "result.png"


def test_terminal_snapshot_is_not_built_without_matching_handler(monkeypatch):
    class Handler(LifecycleHandler):
        def on_started(self, event):
            pass

    lifecycle._register_handler(Handler())
    lifecycle._freeze_handler_routes()
    monkeypatch.setattr(
        lifecycle,
        "_freeze_history_result",
        lambda result: pytest.fail("history snapshot built"),
    )
    context = lifecycle._QueuedLifecycleContext(100, lifecycle._EMPTY_MAPPING)

    with lifecycle._PromptExecutionLifecycle("prompt-1", context) as execution_lifecycle:
        execution_lifecycle.record_executor_result(
            [("execution_success", {"timestamp": 200})],
            {"outputs": {"unsupported": object()}},
        )
        execution_lifecycle.mark_task_done_succeeded()

    assert lifecycle._wait_for_dispatch_for_tests()


def test_node_failure_remains_primary_when_task_done_raises():
    received = []

    class Handler(LifecycleHandler):
        def on_failed(self, event):
            received.append(event)

    context = lifecycle._QueuedLifecycleContext(100, lifecycle._EMPTY_MAPPING)
    lifecycle._register_handler(Handler())
    lifecycle._freeze_handler_routes()

    with pytest.raises(RuntimeError, match="task_done failed"):
        with lifecycle._PromptExecutionLifecycle("prompt-1", context) as execution_lifecycle:
            execution_lifecycle.record_executor_result(
                [("execution_error", {
                    "node_id": "10",
                    "node_type": "FailingNode",
                    "executed": ["3", "1"],
                    "exception_type": "ValueError",
                    "exception_message": "bad input",
                    "traceback": ["trace"],
                    "timestamp": 200,
                    "current_inputs": {"secret": "do-not-copy"},
                    "current_outputs": ["do-not-copy"],
                })],
                {"outputs": {}},
            )
            raise RuntimeError("task_done failed")

    assert lifecycle._wait_for_dispatch_for_tests()
    event = received[0]
    assert event.source is FailureSource.NODE
    assert event.exception_type == "ValueError"
    assert event.executed_nodes == ("1", "3")
    assert event.history_task_done_succeeded is False
    assert not hasattr(event, "current_inputs")
    assert not hasattr(event, "current_outputs")


def test_missing_terminal_status_is_internal_failure():
    received = []

    class Handler(LifecycleHandler):
        def on_failed(self, event):
            received.append(event)

    lifecycle._register_handler(Handler())
    lifecycle._freeze_handler_routes()
    context = lifecycle._QueuedLifecycleContext(100, lifecycle._EMPTY_MAPPING)

    with lifecycle._PromptExecutionLifecycle("prompt-1", context) as execution_lifecycle:
        execution_lifecycle.record_executor_result([], {"outputs": {}})
        execution_lifecycle.mark_task_done_succeeded()

    assert lifecycle._wait_for_dispatch_for_tests()
    assert received[0].source is FailureSource.INTERNAL
    assert received[0].exception_type == "MissingTerminalExecutionStatus"
    assert received[0].history_task_done_succeeded is True


def test_escaping_exception_does_not_use_stale_history_and_propagates():
    received = []

    class Handler(LifecycleHandler):
        def on_failed(self, event):
            received.append(event)

    lifecycle._register_handler(Handler())
    lifecycle._freeze_handler_routes()
    context = lifecycle._QueuedLifecycleContext(100, lifecycle._EMPTY_MAPPING)

    with pytest.raises(ValueError, match="prepare failed"):
        with lifecycle._PromptExecutionLifecycle("prompt-1", context):
            raise ValueError("prepare failed")

    assert lifecycle._wait_for_dispatch_for_tests()
    assert received[0].source is FailureSource.INTERNAL
    assert received[0].exception_type == "ValueError"
    assert received[0].history_result is None


def test_task_done_failure_replaces_interrupted_terminal_event():
    failures = []
    interruptions = []

    class Handler(LifecycleHandler):
        def on_failed(self, event):
            failures.append(event)

        def on_interrupted(self, event):
            interruptions.append(event)

    lifecycle._register_handler(Handler())
    lifecycle._freeze_handler_routes()
    context = lifecycle._QueuedLifecycleContext(100, lifecycle._EMPTY_MAPPING)

    with pytest.raises(RuntimeError, match="history failed"):
        with lifecycle._PromptExecutionLifecycle("prompt-1", context) as execution_lifecycle:
            execution_lifecycle.record_executor_result(
                [("execution_interrupted", {"timestamp": 200})],
                {"outputs": {}},
            )
            raise RuntimeError("history failed")

    assert lifecycle._wait_for_dispatch_for_tests()
    assert failures[0].source is FailureSource.INTERNAL
    assert failures[0].history_task_done_succeeded is False
    assert interruptions == []


def test_interrupted_event_is_distinct_terminal_event():
    received = []

    class Handler(LifecycleHandler):
        def on_interrupted(self, event):
            received.append(event)

    lifecycle._register_handler(Handler())
    lifecycle._freeze_handler_routes()
    context = lifecycle._QueuedLifecycleContext(100, lifecycle._EMPTY_MAPPING)

    with lifecycle._PromptExecutionLifecycle("prompt-1", context) as execution_lifecycle:
        execution_lifecycle.record_executor_result(
            [("execution_interrupted", {
                "node_id": "10",
                "node_type": "Sampler",
                "executed": ["3", "1"],
                "timestamp": 200,
            })],
            {"outputs": {}},
        )
        execution_lifecycle.mark_task_done_succeeded()

    assert lifecycle._wait_for_dispatch_for_tests()
    assert isinstance(received[0], InterruptedEvent)
    assert received[0].executed_nodes == ("1", "3")
    assert received[0].interruption_reported_at_ms == 200


def test_prompt_queue_uses_same_context_for_queued_and_cancelled_events():
    import execution

    events = []

    class Handler(LifecycleHandler):
        def on_queued(self, event):
            events.append(event)

        def on_cancelled(self, event):
            events.append(event)

    class Server:
        def queue_updated(self):
            pass

    lifecycle._register_handler(Handler())
    lifecycle._freeze_handler_routes()
    prompt_queue = execution.PromptQueue(Server())
    item = (
        1.0,
        "prompt-1",
        {},
        {"execution_lifecycle": {"business_task_id": "task-1"}},
        {"2", "1"},
        {"api_key": "secret"},
    )

    prompt_queue.put(item)
    assert prompt_queue.delete_queue_item(lambda queued: queued[1] == "prompt-1")

    assert lifecycle._wait_for_dispatch_for_tests()
    queued, cancelled = events
    assert queued.output_node_ids == ("1", "2")
    assert queued.metadata is cancelled.metadata
    assert "api_key" not in queued.metadata
    assert queued.queued_at_ms == cancelled.queued_at_ms
    assert prompt_queue._queued_lifecycle_contexts == {}


def test_prompt_queue_get_preserves_contract_and_exposes_private_context():
    import execution

    class Handler(LifecycleHandler):
        def on_started(self, event):
            pass

    class Server:
        def queue_updated(self):
            pass

    lifecycle._register_handler(Handler())
    lifecycle._freeze_handler_routes()
    prompt_queue = execution.PromptQueue(Server())
    item = (
        1.0,
        "prompt-1",
        {},
        {"execution_lifecycle": {"business_task_id": "task-1"}},
        set(),
        {},
    )
    prompt_queue.put(item)

    returned_item, item_id = prompt_queue.get(timeout=0)
    context = prompt_queue._take_lifecycle_context(item_id)

    assert returned_item is item
    assert item_id == 0
    assert context.metadata["business_task_id"] == "task-1"
    assert prompt_queue._queued_lifecycle_contexts == {}
    assert prompt_queue._running_lifecycle_contexts == {}


def test_no_handlers_create_no_queue_context_or_metadata_snapshot(monkeypatch):
    import execution

    class Server:
        def queue_updated(self):
            pass

    lifecycle._freeze_handler_routes()
    monkeypatch.setattr(lifecycle, "_freeze_metadata", lambda extra_data: pytest.fail("metadata copied"))
    prompt_queue = execution.PromptQueue(Server())
    item = (1.0, "prompt-1", {}, {"execution_lifecycle": {"task": "ignored"}}, set(), {})

    prompt_queue.put(item)
    returned_item, item_id = prompt_queue.get(timeout=0)

    assert returned_item is item
    assert item_id == 0
    assert prompt_queue._queued_lifecycle_contexts == {}
    assert prompt_queue._running_lifecycle_contexts == {}
    assert prompt_queue._take_lifecycle_context(item_id) is None


def test_missing_expected_context_returns_none_without_rebuilding(caplog):
    import execution

    class Handler(LifecycleHandler):
        def on_started(self, event):
            pass

    class Server:
        def queue_updated(self):
            pass

    lifecycle._register_handler(Handler())
    lifecycle._freeze_handler_routes()
    prompt_queue = execution.PromptQueue(Server())
    prompt_queue.currently_running[7] = (1.0, "prompt-7", {}, {}, set(), {})

    assert prompt_queue._take_lifecycle_context(7) is None
    assert "Missing execution lifecycle context for prompt prompt-7" in caplog.text


def test_wipe_queue_cancels_duplicate_prompt_ids_with_independent_contexts():
    import execution

    cancelled = []

    class Handler(LifecycleHandler):
        def on_cancelled(self, event):
            cancelled.append(event)

    class Server:
        def queue_updated(self):
            pass

    lifecycle._register_handler(Handler())
    lifecycle._freeze_handler_routes()
    prompt_queue = execution.PromptQueue(Server())
    first = (1.0, "same-prompt", {}, {"execution_lifecycle": {"task": "first"}}, set(), {})
    second = (2.0, "same-prompt", {}, {"execution_lifecycle": {"task": "second"}}, set(), {})
    prompt_queue.put(first)
    prompt_queue.put(second)

    prompt_queue.wipe_queue()

    assert lifecycle._wait_for_dispatch_for_tests()
    assert {event.metadata["task"] for event in cancelled} == {"first", "second"}
    assert prompt_queue._queued_lifecycle_contexts == {}
