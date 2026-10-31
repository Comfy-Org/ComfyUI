import asyncio
import contextvars
import logging

import pytest

import comfy.model_management
import nodes
from comfy_execution.execution_observer import (
    ExecutionObserver,
    get_execution_observers,
    register_execution_observer,
    unregister_execution_observer,
)
from comfy_execution.graph_utils import GraphBuilder
from execution import PromptExecutor

current_node = contextvars.ContextVar("current_node", default=None)


class Constant:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT",)}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, value):
        return (value,)


class Increment(Constant):
    def execute(self, value):
        return (value + 1,)


class AsyncIncrement(Constant):
    async def execute(self, value):
        await asyncio.sleep(0.01)
        return (value + 1,)


class ExpandIncrement(Constant):
    def execute(self, value):
        graph = GraphBuilder()
        increment = graph.node("ObserverIncrement", "increment", value=value)
        return {"result": (increment.out(0),), "expand": graph.finalize()}


class SeeContext(Constant):
    seen = []

    def execute(self, value):
        self.seen.append(current_node.get())
        return (value,)


class AsyncSeeContext(Constant):
    seen = []

    async def execute(self, value):
        await asyncio.sleep(0.01)
        self.seen.append(current_node.get())
        return (value,)


class Fail(Constant):
    def execute(self, value):
        raise ValueError("node failed")


class Interrupt(Constant):
    def execute(self, value):
        raise comfy.model_management.InterruptProcessingException()


class RegisterLate(Constant):
    observer = None

    def execute(self, value):
        register_execution_observer(self.observer)
        return (value,)


class Server:
    client_id = None

    def send_sync(self, *args, **kwargs):
        pass


class Recorder(ExecutionObserver):
    def __init__(self):
        self.events = []
        self.tokens = []

    def on_prompt_start(self, prompt_id):
        self.events.append(("prompt_start", prompt_id))

    def on_nodes_cached(self, prompt_id, node_ids):
        self.events.append(("cached", sorted(node_ids)))

    def on_node_start(self, prompt_id, node_id, class_type, parent_node_id):
        self.events.append(("node_start", node_id, class_type, parent_node_id))
        self.tokens.append(current_node.set(node_id))

    def on_node_end(self, prompt_id, node_id, outcome, exception):
        current_node.reset(self.tokens.pop())
        self.events.append(("node_end", node_id, outcome, type(exception).__name__ if exception else None))

    def on_prompt_end(self, prompt_id, outcome):
        self.events.append(("prompt_end", prompt_id, outcome))


@pytest.fixture
def recorder(monkeypatch):
    for name, node in {
        "ObserverConstant": Constant,
        "ObserverIncrement": Increment,
        "ObserverAsyncIncrement": AsyncIncrement,
        "ObserverExpandIncrement": ExpandIncrement,
        "ObserverSeeContext": SeeContext,
        "ObserverAsyncSeeContext": AsyncSeeContext,
        "ObserverFail": Fail,
        "ObserverInterrupt": Interrupt,
        "ObserverRegisterLate": RegisterLate,
    }.items():
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, name, node)
    observer = Recorder()
    register_execution_observer(observer)
    yield observer
    unregister_execution_observer(observer)


def new_executor():
    return PromptExecutor(Server(), cache_type=False, cache_args={"ram": 0, "ram_inactive": 0})


def node(class_type, value):
    return {"class_type": class_type, "inputs": {"value": value}}


def test_events_follow_execution_including_async_and_expanded_nodes(recorder):
    prompt = {
        "1": node("ObserverConstant", 1),
        "2": node("ObserverAsyncIncrement", ["1", 0]),
        "3": node("ObserverExpandIncrement", ["2", 0]),
    }
    executor = new_executor()
    executor.execute(prompt, "observed", {}, ["3"])
    assert executor.success

    expanded = next(e[1] for e in recorder.events if e[0] == "node_start" and e[2] == "ObserverIncrement")
    assert recorder.events == [
        ("prompt_start", "observed"),
        ("cached", []),
        ("node_start", "1", "ObserverConstant", None),
        ("node_end", "1", "success", None),
        ("node_start", "2", "ObserverAsyncIncrement", None),
        ("node_end", "2", "pending", None),
        ("node_start", "2", "ObserverAsyncIncrement", None),
        ("node_end", "2", "success", None),
        ("node_start", "3", "ObserverExpandIncrement", None),
        ("node_end", "3", "pending", None),
        ("node_start", expanded, "ObserverIncrement", "3"),
        ("node_end", expanded, "success", None),
        ("node_start", "3", "ObserverExpandIncrement", None),
        ("node_end", "3", "success", None),
        ("prompt_end", "observed", "success"),
    ]


def test_context_set_on_node_start_reaches_sync_and_async_node_code(recorder):
    SeeContext.seen, AsyncSeeContext.seen = [], []
    prompt = {
        "1": node("ObserverSeeContext", 1),
        "2": node("ObserverAsyncSeeContext", ["1", 0]),
    }
    executor = new_executor()
    executor.execute(prompt, "context", {}, ["2"])
    assert executor.success
    assert SeeContext.seen == ["1"]
    assert AsyncSeeContext.seen == ["2"]
    assert current_node.get() is None


@pytest.mark.parametrize(("class_type", "outcome", "exception"), [
    ("ObserverFail", "error", "ValueError"),
    ("ObserverInterrupt", "interrupted", "InterruptProcessingException"),
])
def test_failed_and_interrupted_nodes(recorder, class_type, outcome, exception):
    prompt = {"1": node("ObserverConstant", 1), "2": node(class_type, ["1", 0])}
    executor = new_executor()
    executor.execute(prompt, "stopped", {}, ["2"])
    assert not executor.success
    assert recorder.events[-2:] == [("node_end", "2", outcome, exception), ("prompt_end", "stopped", outcome)]


def test_cached_nodes_are_reported_once_and_not_started(recorder):
    prompt = {"1": node("ObserverConstant", 1), "2": node("ObserverIncrement", ["1", 0])}
    executor = new_executor()
    executor.execute(prompt, "first", {}, ["2"])
    recorder.events.clear()
    executor.execute(prompt, "second", {}, ["2"])
    assert recorder.events == [("prompt_start", "second"), ("cached", ["1", "2"]), ("prompt_end", "second", "success")]


def test_a_failing_observer_does_not_break_execution(recorder, caplog):
    class Broken(ExecutionObserver):
        def on_node_start(self, *args):
            raise RuntimeError("observer bug")

    broken = Broken()
    register_execution_observer(broken)
    try:
        executor = new_executor()
        with caplog.at_level(logging.WARNING):
            executor.execute({"1": node("ObserverConstant", 1)}, "broken", {}, ["1"])
    finally:
        unregister_execution_observer(broken)
    assert executor.success
    assert ("node_end", "1", "success", None) in recorder.events
    assert "Execution observer Broken error in on_node_start: observer bug" in caplog.text


def test_observers_changed_mid_prompt_take_effect_from_the_next_prompt(recorder):
    class Leaving(Recorder):
        def on_node_start(self, *args):
            super().on_node_start(*args)
            unregister_execution_observer(self)

    leaving, late = Leaving(), Recorder()
    RegisterLate.observer = late
    register_execution_observer(leaving)
    prompt = {"1": node("ObserverConstant", 1), "2": node("ObserverRegisterLate", ["1", 0])}
    try:
        new_executor().execute(prompt, "first", {}, ["2"])
        assert leaving.events == [
            ("prompt_start", "first"),
            ("cached", []),
            ("node_start", "1", "ObserverConstant", None),
            ("node_end", "1", "success", None),
            ("node_start", "2", "ObserverRegisterLate", None),
            ("node_end", "2", "success", None),
            ("prompt_end", "first", "success"),
        ]
        assert late.events == []
        assert current_node.get() is None

        new_executor().execute(prompt, "second", {}, ["2"])
        assert leaving.events[-1] == ("prompt_end", "first", "success")
        assert late.events[0] == ("prompt_start", "second")
        assert late.events[-1] == ("prompt_end", "second", "success")
        assert len(late.events) == 7
    finally:
        unregister_execution_observer(late)


def test_registration_compares_observers_by_identity():
    class AlwaysEqual(ExecutionObserver):
        def __eq__(self, other):
            return True

        __hash__ = object.__hash__

    first, second = AlwaysEqual(), AlwaysEqual()
    assert get_execution_observers() == ()
    register_execution_observer(first)
    register_execution_observer(first)
    register_execution_observer(second)
    assert [o is first for o in get_execution_observers()] == [True, False]
    assert get_execution_observers()[1] is second
    unregister_execution_observer(second)
    assert len(get_execution_observers()) == 1 and get_execution_observers()[0] is first
    unregister_execution_observer(first)
    assert get_execution_observers() == ()
