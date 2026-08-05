import asyncio
import gc
import weakref

import pytest
import torch

import comfy.model_management
import nodes

import comfy_extras.nodes_loop as nodes_loop
from comfy_execution.graph import DynamicPrompt, ExecutionList
from comfy_execution.graph_utils import GraphBuilder
from comfy_execution.validation import validate_loops
from execution import (
    NODE_FAILURE_POLICY_CONTINUE_INDEPENDENT,
    NODE_FAILURE_POLICY_EXTRA_DATA_KEY,
    CacheType,
    PromptExecutor,
)


STATE = {}


class Payload:
    pass


class Constant:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT",)}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, value):
        return (value,)


class MakePayload:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT",)}}

    RETURN_TYPES = ("PAYLOAD",)
    FUNCTION = "execute"

    def execute(self, value):
        payload = Payload()
        STATE["payload"] = weakref.ref(payload)
        return (payload,)


class Fail:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("*",)}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, value):
        STATE["fail_calls"] = STATE.get("fail_calls", 0) + 1
        raise RuntimeError("node failure")


class SlowFail:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("*",)}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    async def execute(self, value):
        await asyncio.sleep(0.05)
        STATE["fail_calls"] = STATE.get("fail_calls", 0) + 1
        raise RuntimeError("node failure")


class Capture:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("*",)}}

    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True

    def execute(self, value):
        STATE.setdefault("captured", []).append(value)
        return ()


class WaitForFailure:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT",)}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    async def execute(self, value):
        while not STATE.get("fail_calls"):
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        return (value,)


class PayloadProbe:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT",)}}

    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True

    def execute(self, value):
        STATE["executor"].caches.outputs.ram_release(10 ** 30, free_active=True)
        STATE["payload_alive"] = STATE["payload"]() is not None
        return ()


class LazyPick:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"gate": ("INT",), "value": ("INT", {"lazy": True})}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def check_lazy_status(self, gate, value=None):
        return ["value"] if value is None else []

    def execute(self, gate, value):
        return (value,)


class ExpandWithFailingSideBranch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT",)}}

    RETURN_TYPES = ("PAYLOAD",)
    FUNCTION = "execute"

    def execute(self, value):
        STATE["expand_calls"] = STATE.get("expand_calls", 0) + 1
        graph = GraphBuilder()
        payload = graph.node("PartialMakePayload", "payload", value=value)
        failure = graph.node("PartialFail", "failure", value=value)
        graph.node("PartialCapture", "side_output", value=failure.out(0))
        return {"result": (payload.out(0),), "expand": graph.finalize()}


class ExpandWithLateFailingSideBranch(ExpandWithFailingSideBranch):
    def execute(self, value):
        STATE["expand_calls"] = STATE.get("expand_calls", 0) + 1
        graph = GraphBuilder()
        payload = graph.node("PartialMakePayload", "payload", value=value)
        failure = graph.node("PartialSlowFail", "failure", value=value)
        graph.node("PartialCapture", "side_output", value=failure.out(0))
        return {"result": (payload.out(0),), "expand": graph.finalize()}


class UsePayload:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"payload": ("PAYLOAD",)}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, payload):
        STATE["use_calls"] = STATE.get("use_calls", 0) + 1
        return (1,)


class ExpandAroundFailingExpansion:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT",)}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, value):
        STATE["outer_calls"] = STATE.get("outer_calls", 0) + 1
        graph = GraphBuilder()
        inner = graph.node("PartialExpandWithFailingSideBranch", "inner", value=value)
        mid = graph.node("PartialUsePayload", "mid", payload=inner.out(0))
        return {"result": (mid.out(0),), "expand": graph.finalize()}


class LazySwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"select": ("STRING",), "a": ("INT", {"lazy": True}), "b": ("INT", {"lazy": True})}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def check_lazy_status(self, select, a=None, b=None):
        return [select] if {"a": a, "b": b}[select] is None else []

    def execute(self, select, a=None, b=None):
        return (a if select == "a" else b,)


class Increment:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT",)}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, value):
        return (value + 1,)


class CapturePassthrough:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT",)}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"
    OUTPUT_NODE = True

    def execute(self, value):
        STATE.setdefault("captured", []).append(value)
        return (value,)


class FailWhen:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT",), "when": ("INT",)}, "optional": {"error": ("STRING",)}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, value, when, error="runtime"):
        if value == when:
            if error == "memory":
                raise MemoryError("host memory")
            if error == "interrupt":
                raise comfy.model_management.InterruptProcessingException()
            if error == "accelerator":
                raise torch.AcceleratorError("device fault")
            raise RuntimeError("node failure")
        return (value,)


class FailOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("*",)}}

    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True

    def execute(self, value):
        raise RuntimeError("output failure")


class ExpandWithUnknownDisplayId:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT",)}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, value):
        graph = GraphBuilder()
        failure = graph.node("PartialFail", "failure", value=value)
        failure.set_override_display_id("missing")
        return {"result": (failure.out(0),), "expand": graph.finalize()}


class Join:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"first": ("*",), "second": ("*",)}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, first, second):
        return (1,)


class Server:
    client_id = None

    def send_sync(self, *args, **kwargs):
        pass


class Progress:
    def send_progress_text(self, text, node_id):
        pass


@pytest.fixture(autouse=True)
def register_nodes(monkeypatch):
    STATE.clear()
    classes = {
        "StartLoop": nodes_loop.StartLoop,
        "EndLoop": nodes_loop.EndLoop,
        "LoopIteration": nodes_loop.LoopIteration,
        "LoopProgress": nodes_loop.LoopProgress,
        "LoopResult": nodes_loop.LoopResult,
        "PartialConstant": Constant,
        "PartialMakePayload": MakePayload,
        "PartialFail": Fail,
        "PartialCapture": Capture,
        "PartialWaitForFailure": WaitForFailure,
        "PartialPayloadProbe": PayloadProbe,
        "PartialLazyPick": LazyPick,
        "PartialExpandWithFailingSideBranch": ExpandWithFailingSideBranch,
        "PartialSlowFail": SlowFail,
        "PartialExpandWithLateFailingSideBranch": ExpandWithLateFailingSideBranch,
        "PartialUsePayload": UsePayload,
        "PartialExpandAroundFailingExpansion": ExpandAroundFailingExpansion,
        "PartialLazySwitch": LazySwitch,
        "PartialJoin": Join,
        "PartialExpandWithUnknownDisplayId": ExpandWithUnknownDisplayId,
        "PartialIncrement": Increment,
        "PartialCapturePassthrough": CapturePassthrough,
        "PartialFailWhen": FailWhen,
        "PartialFailOutput": FailOutput,
    }
    for name, node in classes.items():
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, name, node)
    monkeypatch.setattr(nodes_loop, "PromptServer", type("PromptServer", (), {"instance": Progress()}))


def run(prompt, outputs, continue_independent=True, cache_type=CacheType.NONE, executor=None):
    starts = {node_id for node_id, node in prompt.items() if node["class_type"] == "StartLoop"}
    ends = {node_id for node_id, node in prompt.items() if node["class_type"] == "EndLoop"}
    if starts:
        validate_loops(prompt, set(outputs), prompt, starts, ends)
    if executor is None:
        executor = PromptExecutor(Server(), cache_type=cache_type, cache_args={"ram": 0, "ram_inactive": 0, "lru": 100})
    STATE["executor"] = executor
    extra_data = {}
    if continue_independent:
        extra_data[NODE_FAILURE_POLICY_EXTRA_DATA_KEY] = NODE_FAILURE_POLICY_CONTINUE_INDEPENDENT
    asyncio.run(asyncio.wait_for(executor.execute_async(prompt, "partial-execution-test", extra_data, outputs), 30))
    return executor


def events(executor):
    return [event for event, _ in executor.status_messages]


def independent_branch_prompt():
    return {
        "seed": {"class_type": "PartialConstant", "inputs": {"value": 1}},
        "fail": {"class_type": "PartialFail", "inputs": {"value": ["seed", 0]}},
        "blocked": {"class_type": "PartialCapture", "inputs": {"value": ["fail", 0]}},
        "independent": {"class_type": "PartialCapture", "inputs": {"value": ["seed", 0]}},
    }


def test_failure_drops_only_dependents():
    executor = run(independent_branch_prompt(), ["blocked", "independent"])

    assert executor.success
    assert STATE["captured"] == [1]
    assert events(executor) == ["execution_start", "execution_cached", "execution_node_error", "execution_success"]
    assert executor.execution_summary == {
        "has_errors": True,
        "execution_error_count": 1,
        "failed_node_ids": ["fail"],
        "blocked_node_ids": ["blocked"],
        "blocked_output_node_ids": ["blocked"],
        "successful_output_node_ids": ["independent"],
        "completion_status": "partial_success",
    }


def test_fail_fast_stays_default():
    executor = run(independent_branch_prompt(), ["blocked", "independent"], continue_independent=False)

    assert not executor.success
    assert executor.execution_summary is None
    assert events(executor)[-1] == "execution_error"


def test_failure_without_surviving_output_is_an_error():
    prompt = independent_branch_prompt()
    del prompt["independent"]

    executor = run(prompt, ["blocked"])

    assert not executor.success
    assert events(executor) == ["execution_start", "execution_cached", "execution_node_error", "execution_error"]


class RecordingServer(Server):
    def __init__(self):
        self.sent = []

    def send_sync(self, event, data, sid=None):
        self.sent.append((event, data))


def test_dropping_many_nodes_sends_one_progress_state():
    prompt = independent_branch_prompt()
    for i in range(20):
        prompt[f"blocked{i}"] = {"class_type": "PartialCapture", "inputs": {"value": ["fail", 0]}}
    outputs = ["independent"] + [f"blocked{i}" for i in range(20)]
    executor = PromptExecutor(RecordingServer(), cache_type=CacheType.NONE, cache_args={"ram": 0, "ram_inactive": 0, "lru": 100})

    run(prompt, outputs, executor=executor)

    blocked_messages = [
        data["nodes"] for event, data in executor.server.sent
        if event == "progress_state" and any(node["state"] == "blocked" for node in data["nodes"].values())
    ]
    assert len(blocked_messages) == 1
    assert {node_id for node_id, node in blocked_messages[0].items() if node["state"] == "blocked"} == {f"blocked{i}" for i in range(20)}
    assert blocked_messages[0]["fail"]["state"] == "error"


def test_loop_body_failure_blocks_the_loop_without_hanging():
    prompt = {
        "loop": {"class_type": "StartLoop", "inputs": {"mode": "simple", "mode.num_iterations": 3}},
        "body": {"class_type": "PartialFail", "inputs": {"value": ["loop", 0]}},
        "close": {"class_type": "EndLoop", "inputs": {"next_iteration_value": ["body", 0], "accumulate": False}},
        "after": {"class_type": "PartialCapture", "inputs": {"value": ["close", 0]}},
        "seed": {"class_type": "PartialConstant", "inputs": {"value": 7}},
        "independent": {"class_type": "PartialCapture", "inputs": {"value": ["seed", 0]}},
    }

    executor = run(prompt, ["after", "independent"])

    assert executor.success
    assert STATE["fail_calls"] == 1
    assert STATE["captured"] == [7]
    assert executor.execution_summary["completion_status"] == "partial_success"
    assert executor.execution_summary["blocked_output_node_ids"] == ["after"]
    assert "close" in executor.execution_summary["blocked_node_ids"]


def test_failure_feeding_the_loop_body_before_the_loop_starts_does_not_hang():
    prompt = {
        "seed": {"class_type": "PartialConstant", "inputs": {"value": 7}},
        "fail": {"class_type": "PartialFail", "inputs": {"value": ["seed", 0]}},
        "gate": {"class_type": "PartialWaitForFailure", "inputs": {"value": ["seed", 0]}},
        "loop": {"class_type": "StartLoop", "inputs": {"mode": "simple", "mode.num_iterations": 3, "initial_iteration_value": ["gate", 0]}},
        "body": {"class_type": "PartialJoin", "inputs": {"first": ["loop", 4], "second": ["fail", 0]}},
        "close": {"class_type": "EndLoop", "inputs": {"output_value": ["body", 0], "next_iteration_value": ["body", 0], "accumulate": False}},
        "after": {"class_type": "PartialCapture", "inputs": {"value": ["close", 0]}},
        "independent": {"class_type": "PartialCapture", "inputs": {"value": ["seed", 0]}},
    }

    executor = run(prompt, ["after", "independent"])

    assert executor.success
    assert STATE["fail_calls"] == 1
    assert STATE["captured"] == [7]
    assert executor.execution_summary["blocked_output_node_ids"] == ["after"]
    assert {"body", "close"} <= set(executor.execution_summary["blocked_node_ids"])


@pytest.mark.parametrize("cache_type", [CacheType.NONE, CacheType.CLASSIC, CacheType.LRU, CacheType.RAM_PRESSURE])
def test_late_lazy_link_never_reruns_failed_node(cache_type):
    prompt = {
        "seed": {"class_type": "PartialConstant", "inputs": {"value": 1}},
        "fail": {"class_type": "PartialFail", "inputs": {"value": ["seed", 0]}},
        "blocked": {"class_type": "PartialCapture", "inputs": {"value": ["fail", 0]}},
        "gate": {"class_type": "PartialWaitForFailure", "inputs": {"value": ["seed", 0]}},
        "pick": {"class_type": "PartialLazyPick", "inputs": {"gate": ["gate", 0], "value": ["fail", 0]}},
        "late": {"class_type": "PartialCapture", "inputs": {"value": ["pick", 0]}},
        "independent": {"class_type": "PartialCapture", "inputs": {"value": ["seed", 0]}},
    }

    executor = run(prompt, ["blocked", "late", "independent"], cache_type=cache_type)

    assert STATE["fail_calls"] == 1
    assert STATE["captured"] == [1]
    assert executor.execution_summary["blocked_node_ids"] == ["blocked", "late", "pick"]
    assert executor.execution_summary["successful_output_node_ids"] == ["independent"]


def test_failed_node_inputs_are_released_while_prompt_continues():
    prompt = {
        "seed": {"class_type": "PartialConstant", "inputs": {"value": 1}},
        "payload": {"class_type": "PartialMakePayload", "inputs": {"value": ["seed", 0]}},
        "fail": {"class_type": "PartialFail", "inputs": {"value": ["payload", 0]}},
        "blocked": {"class_type": "PartialCapture", "inputs": {"value": ["fail", 0]}},
        "gate": {"class_type": "PartialWaitForFailure", "inputs": {"value": ["seed", 0]}},
        "probe": {"class_type": "PartialPayloadProbe", "inputs": {"value": ["gate", 0]}},
    }

    gc.disable()
    try:
        executor = run(prompt, ["blocked", "probe"], cache_type=CacheType.RAM_PRESSURE)
    finally:
        gc.enable()

    assert executor.execution_summary["completion_status"] == "partial_success"
    assert STATE["payload_alive"] is False


def test_outputs_are_only_held_by_the_cache_and_pending_consumers():
    prompt = {
        "seed": {"class_type": "PartialConstant", "inputs": {"value": 1}},
        "expand": {"class_type": "PartialExpandWithFailingSideBranch", "inputs": {"value": ["seed", 0]}},
        "use": {"class_type": "PartialUsePayload", "inputs": {"payload": ["expand", 0]}},
        "probe": {"class_type": "PartialPayloadProbe", "inputs": {"value": ["use", 0]}},
    }

    gc.disable()
    try:
        executor = run(prompt, ["probe"], cache_type=CacheType.RAM_PRESSURE)
    finally:
        gc.enable()

    assert executor.execution_summary["completion_status"] == "partial_success"
    assert executor.execution_summary["failed_node_ids"] == ["expand"]
    assert STATE["payload_alive"] is False


@pytest.mark.parametrize("cache_type", [CacheType.NONE, CacheType.CLASSIC, CacheType.LRU, CacheType.RAM_PRESSURE])
def test_failed_side_branch_does_not_block_the_expansion_result(cache_type):
    prompt = {
        "seed": {"class_type": "PartialConstant", "inputs": {"value": 1}},
        "expand": {"class_type": "PartialExpandWithFailingSideBranch", "inputs": {"value": ["seed", 0]}},
        "output": {"class_type": "PartialCapture", "inputs": {"value": ["expand", 0]}},
    }

    executor = run(prompt, ["output"], cache_type=cache_type)

    assert executor.success
    assert STATE["fail_calls"] == 1
    assert len(STATE["captured"]) == 1
    assert executor.execution_summary["failed_node_ids"] == ["expand"]
    assert executor.execution_summary["successful_output_node_ids"] == ["output"]


@pytest.mark.parametrize("cache_type", [CacheType.CLASSIC, CacheType.LRU, CacheType.RAM_PRESSURE])
@pytest.mark.parametrize("expander", ["PartialExpandWithFailingSideBranch", "PartialExpandWithLateFailingSideBranch"])
def test_expansion_with_a_failed_side_branch_is_not_cached(cache_type, expander):
    prompt = {
        "seed": {"class_type": "PartialConstant", "inputs": {"value": 1}},
        "expand": {"class_type": expander, "inputs": {"value": ["seed", 0]}},
        "output": {"class_type": "PartialCapture", "inputs": {"value": ["expand", 0]}},
    }

    executor = run(prompt, ["output"], cache_type=cache_type)
    run(prompt, ["output"], cache_type=cache_type, executor=executor)

    assert STATE["expand_calls"] == 2
    assert STATE["fail_calls"] == 2
    assert len(STATE["captured"]) == 1
    assert executor.execution_summary["completion_status"] == "partial_success"
    assert executor.execution_summary["failed_node_ids"] == ["expand"]
    assert events(executor).count("execution_cached") == 1
    cached = next(data for event, data in executor.status_messages if event == "execution_cached")
    assert sorted(cached["nodes"]) == ["output", "seed"]


@pytest.mark.parametrize("cache_type", [CacheType.CLASSIC, CacheType.LRU, CacheType.RAM_PRESSURE])
def test_expansion_with_a_failed_side_branch_reruns_behind_cached_consumers(cache_type):
    prompt = {
        "seed": {"class_type": "PartialConstant", "inputs": {"value": 1}},
        "expand": {"class_type": "PartialExpandWithFailingSideBranch", "inputs": {"value": ["seed", 0]}},
        "mid": {"class_type": "PartialUsePayload", "inputs": {"payload": ["expand", 0]}},
        "output": {"class_type": "PartialCapture", "inputs": {"value": ["mid", 0]}},
    }

    executor = run(prompt, ["output"], cache_type=cache_type)
    run(prompt, ["output"], cache_type=cache_type, executor=executor)

    assert STATE["expand_calls"] == 2
    assert STATE["fail_calls"] == 2
    assert STATE["use_calls"] == 1
    assert len(STATE["captured"]) == 1
    cached = next(data for event, data in executor.status_messages if event == "execution_cached")
    assert sorted(cached["nodes"]) == ["mid", "output", "seed"]
    assert executor.execution_summary["failed_node_ids"] == ["expand"]


@pytest.mark.parametrize("cache_type", [CacheType.LRU, CacheType.RAM_PRESSURE])
def test_incomplete_expansion_survives_unrelated_prompts(cache_type):
    prompt = {
        "seed": {"class_type": "PartialConstant", "inputs": {"value": 1}},
        "expand": {"class_type": "PartialExpandWithFailingSideBranch", "inputs": {"value": ["seed", 0]}},
        "mid": {"class_type": "PartialUsePayload", "inputs": {"payload": ["expand", 0]}},
        "output": {"class_type": "PartialCapture", "inputs": {"value": ["mid", 0]}},
    }
    other = {
        "seed": {"class_type": "PartialConstant", "inputs": {"value": 2}},
        "output": {"class_type": "PartialCapture", "inputs": {"value": ["seed", 0]}},
    }

    executor = run(prompt, ["output"], cache_type=cache_type)
    run(other, ["output"], cache_type=cache_type, executor=executor)
    run(prompt, ["output"], cache_type=cache_type, executor=executor)

    assert STATE["expand_calls"] == 2
    assert STATE["fail_calls"] == 2
    assert STATE["use_calls"] == 1
    cached = next(data for event, data in executor.status_messages if event == "execution_cached")
    assert sorted(cached["nodes"]) == ["mid", "output", "seed"]
    assert executor.execution_summary["failed_node_ids"] == ["expand"]


def test_incomplete_expansion_mark_is_kept_under_ram_pressure():
    prompt = {
        "seed": {"class_type": "PartialConstant", "inputs": {"value": 1}},
        "expand": {"class_type": "PartialExpandWithFailingSideBranch", "inputs": {"value": ["seed", 0]}},
        "mid": {"class_type": "PartialUsePayload", "inputs": {"payload": ["expand", 0]}},
        "output": {"class_type": "PartialCapture", "inputs": {"value": ["mid", 0]}},
    }

    executor = run(prompt, ["output"], cache_type=CacheType.RAM_PRESSURE)
    executor.caches.outputs.ram_release(10 ** 30, free_active=True)

    assert executor.caches.outputs.get_local("mid") is None
    assert executor.caches.outputs.is_incomplete("expand")

    run(prompt, ["output"], cache_type=CacheType.RAM_PRESSURE, executor=executor)
    assert STATE["expand_calls"] == 2
    assert STATE["fail_calls"] == 2


@pytest.mark.parametrize("cache_type", [CacheType.CLASSIC, CacheType.LRU, CacheType.RAM_PRESSURE])
def test_nested_incomplete_expansion_reruns_behind_a_cached_intermediate(cache_type):
    prompt = {
        "seed": {"class_type": "PartialConstant", "inputs": {"value": 1}},
        "outer": {"class_type": "PartialExpandAroundFailingExpansion", "inputs": {"value": ["seed", 0]}},
        "output": {"class_type": "PartialCapture", "inputs": {"value": ["outer", 0]}},
    }

    executor = run(prompt, ["output"], cache_type=cache_type)
    assert STATE["fail_calls"] == 1
    assert executor.execution_summary["failed_node_ids"] == ["outer"]

    run(prompt, ["output"], cache_type=cache_type, executor=executor)
    assert STATE["outer_calls"] == 2
    assert STATE["expand_calls"] == 2
    assert STATE["fail_calls"] == 2
    assert STATE["use_calls"] == 1
    assert len(STATE["captured"]) == 1
    assert executor.execution_summary["failed_node_ids"] == ["outer"]


@pytest.mark.parametrize("cache_type", [CacheType.CLASSIC, CacheType.LRU, CacheType.RAM_PRESSURE])
def test_incomplete_expansion_behind_a_lazy_input_follows_the_selection(cache_type):
    def prompt(select, other):
        return {
            "seed": {"class_type": "PartialConstant", "inputs": {"value": 1}},
            "other": {"class_type": "PartialConstant", "inputs": {"value": other}},
            "expand": {"class_type": "PartialExpandWithFailingSideBranch", "inputs": {"value": ["seed", 0]}},
            "mid": {"class_type": "PartialUsePayload", "inputs": {"payload": ["expand", 0]}},
            "switch": {"class_type": "PartialLazySwitch", "inputs": {"select": select, "a": ["mid", 0], "b": ["other", 0]}},
            "output": {"class_type": "PartialCapture", "inputs": {"value": ["switch", 0]}},
        }

    executor = run(prompt("a", 2), ["output"], cache_type=cache_type)
    assert STATE["fail_calls"] == 1
    assert STATE["captured"] == [1]

    run(prompt("b", 2), ["output"], cache_type=cache_type, executor=executor)
    assert STATE["expand_calls"] == 1
    assert STATE["captured"] == [1, 2]
    assert executor.execution_summary is None

    run(prompt("a", 3), ["output"], cache_type=cache_type, executor=executor)
    assert STATE["expand_calls"] == 2
    assert STATE["fail_calls"] == 2
    assert STATE["use_calls"] == 1
    assert STATE["captured"] == [1, 2, 1]
    assert executor.execution_summary["failed_node_ids"] == ["expand"]


def test_incomplete_expansion_outside_the_requested_outputs_stays_idle():
    prompt = {
        "seed": {"class_type": "PartialConstant", "inputs": {"value": 1}},
        "expand": {"class_type": "PartialExpandWithFailingSideBranch", "inputs": {"value": ["seed", 0]}},
        "output": {"class_type": "PartialCapture", "inputs": {"value": ["expand", 0]}},
        "independent": {"class_type": "PartialCapture", "inputs": {"value": ["seed", 0]}},
    }

    executor = run(prompt, ["output", "independent"], cache_type=CacheType.CLASSIC)
    run(prompt, ["independent"], cache_type=CacheType.CLASSIC, executor=executor)
    assert STATE["expand_calls"] == 1
    assert executor.execution_summary is None

    run(prompt, ["output", "independent"], cache_type=CacheType.CLASSIC, executor=executor)
    assert STATE["expand_calls"] == 2
    assert executor.execution_summary["failed_node_ids"] == ["expand"]


def test_outputs_saved_inside_a_loop_count_as_successful_outputs():
    prompt = {
        "seed": {"class_type": "PartialConstant", "inputs": {"value": 0}},
        "loop": {"class_type": "StartLoop", "inputs": {"mode": "simple", "mode.num_iterations": 3, "initial_iteration_value": ["seed", 0]}},
        "increment": {"class_type": "PartialIncrement", "inputs": {"value": ["loop", 4]}},
        "save": {"class_type": "PartialCapturePassthrough", "inputs": {"value": ["increment", 0]}},
        "check": {"class_type": "PartialFailWhen", "inputs": {"value": ["increment", 0], "when": 2}},
        "close": {"class_type": "EndLoop", "inputs": {"output_value": ["check", 0], "next_iteration_value": ["check", 0], "accumulate": False, "termination0": ["save", 0]}},
    }

    executor = run(prompt, ["save"])

    assert executor.success
    assert STATE["captured"] == [1, 2]
    assert executor.execution_summary["completion_status"] == "partial_success"
    assert executor.execution_summary["successful_output_node_ids"] == ["save"]
    assert executor.execution_summary["failed_node_ids"] == ["loop"]
    node_error = next(data for event, data in executor.status_messages if event == "execution_node_error")
    assert node_error["display_node_id"] == "check"
    assert node_error["node_type"] == "PartialFailWhen"
    assert "executed" not in node_error and "current_outputs" not in node_error


def test_unknown_display_id_falls_back_to_the_real_node():
    prompt = {
        "seed": {"class_type": "PartialConstant", "inputs": {"value": 1}},
        "expand": {"class_type": "PartialExpandWithUnknownDisplayId", "inputs": {"value": ["seed", 0]}},
        "blocked": {"class_type": "PartialCapture", "inputs": {"value": ["expand", 0]}},
        "independent": {"class_type": "PartialCapture", "inputs": {"value": ["seed", 0]}},
    }

    executor = run(prompt, ["blocked", "independent"])

    assert executor.success
    assert executor.execution_summary["failed_node_ids"] == ["expand"]
    node_error = next(data for event, data in executor.status_messages if event == "execution_node_error")
    assert node_error["display_node_id"] == "expand"
    assert node_error["node_type"] == "PartialExpandWithUnknownDisplayId"


def test_interrupt_after_a_recoverable_failure_stays_terminal():
    prompt = {
        "seed": {"class_type": "PartialConstant", "inputs": {"value": 1}},
        "fail": {"class_type": "PartialFail", "inputs": {"value": ["seed", 0]}},
        "blocked": {"class_type": "PartialCapture", "inputs": {"value": ["fail", 0]}},
        "wait": {"class_type": "PartialWaitForFailure", "inputs": {"value": ["seed", 0]}},
        "interrupt": {"class_type": "PartialFailWhen", "inputs": {"value": ["wait", 0], "when": 1, "error": "interrupt"}},
        "after": {"class_type": "PartialCapture", "inputs": {"value": ["interrupt", 0]}},
    }

    executor = run(prompt, ["blocked", "after"])

    assert not executor.success
    assert executor.execution_summary is None
    assert events(executor)[-2:] == ["execution_node_error", "execution_interrupted"]


@pytest.mark.skipif(not hasattr(torch, "AcceleratorError"), reason="torch without AcceleratorError")
def test_device_fault_stays_terminal():
    prompt = {
        "seed": {"class_type": "PartialConstant", "inputs": {"value": 1}},
        "fail": {"class_type": "PartialFailWhen", "inputs": {"value": ["seed", 0], "when": 1, "error": "accelerator"}},
        "blocked": {"class_type": "PartialCapture", "inputs": {"value": ["fail", 0]}},
        "independent": {"class_type": "PartialCapture", "inputs": {"value": ["seed", 0]}},
    }

    executor = run(prompt, ["blocked", "independent"])

    assert not executor.success
    assert "execution_node_error" not in events(executor)
    assert events(executor)[-1] == "execution_error"


def test_failed_output_node_is_listed_as_failed_only():
    prompt = {
        "seed": {"class_type": "PartialConstant", "inputs": {"value": 1}},
        "failing": {"class_type": "PartialFailOutput", "inputs": {"value": ["seed", 0]}},
        "ok": {"class_type": "PartialCapture", "inputs": {"value": ["seed", 0]}},
    }

    executor = run(prompt, ["failing", "ok"])

    assert executor.execution_summary == {
        "has_errors": True,
        "execution_error_count": 1,
        "failed_node_ids": ["failing"],
        "blocked_node_ids": [],
        "blocked_output_node_ids": [],
        "successful_output_node_ids": ["ok"],
        "completion_status": "partial_success",
    }


@pytest.mark.parametrize("cache_type,kept", [(CacheType.CLASSIC, False), (CacheType.LRU, True), (CacheType.RAM_PRESSURE, True)])
def test_incomplete_flag_follows_the_marks(cache_type, kept):
    executor = PromptExecutor(Server(), cache_type=cache_type, cache_args={"ram": 0, "ram_inactive": 0, "lru": 100})
    prompt = {
        "seed": {"class_type": "PartialConstant", "inputs": {"value": 1}},
        "expand": {"class_type": "PartialExpandWithFailingSideBranch", "inputs": {"value": ["seed", 0]}},
        "output": {"class_type": "PartialCapture", "inputs": {"value": ["expand", 0]}},
    }
    other = {
        "seed": {"class_type": "PartialConstant", "inputs": {"value": 2}},
        "output": {"class_type": "PartialCapture", "inputs": {"value": ["seed", 0]}},
    }

    run(other, ["output"], executor=executor)
    assert executor.caches.outputs.has_incomplete is False

    run(prompt, ["output"], executor=executor)
    assert executor.caches.outputs.has_incomplete is True

    run(other, ["output"], executor=executor)
    assert executor.caches.outputs.has_incomplete is kept


def test_host_memory_exhaustion_stays_terminal():
    prompt = {
        "seed": {"class_type": "PartialConstant", "inputs": {"value": 1}},
        "fail": {"class_type": "PartialFailWhen", "inputs": {"value": ["seed", 0], "when": 1, "error": "memory"}},
        "blocked": {"class_type": "PartialCapture", "inputs": {"value": ["fail", 0]}},
        "independent": {"class_type": "PartialCapture", "inputs": {"value": ["seed", 0]}},
    }

    executor = run(prompt, ["blocked", "independent"])

    assert not executor.success
    assert "execution_node_error" not in events(executor)
    assert events(executor)[-1] == "execution_error"


def scheduler(blocking, external=None):
    result = ExecutionList(DynamicPrompt({}), None)
    external = external or {}
    for node_id in blocking:
        result.pendingNodes[node_id] = True
        result.blockCount[node_id] = external.get(node_id, 0)
        result.blocking[node_id] = {}
        result.execution_cache[node_id] = {}
    for from_node_id, blocked_nodes in blocking.items():
        for to_node_id in blocked_nodes:
            result.blocking[from_node_id][to_node_id] = {0: True}
            result.blockCount[to_node_id] += 1
    result.externalBlocks = sum(external.values())
    return result


def test_make_unavailable_removes_transitive_dependents_only():
    graph = scheduler({"failed": ["child"], "child": ["grandchild"], "grandchild": [], "other": ["shared"], "shared": []})
    graph.blocking["child"]["shared"] = {0: True}
    graph.blockCount["shared"] += 1
    graph.staged_node_id = "failed"

    removed = graph.make_unavailable("failed")

    assert sorted(removed) == ["child", "failed", "grandchild", "shared"]
    assert list(graph.pendingNodes) == ["other"]
    assert graph.blocking == {"other": {}}
    assert graph.staged_node_id is None
    assert graph.unavailable == set(removed)
    assert set(graph.execution_cache) == {"other"}


def test_make_unavailable_settles_external_blocks():
    graph = scheduler({"failed": ["waiting"], "waiting": []}, external={"waiting": 2})
    unblock = graph.add_external_block("waiting")
    graph.staged_node_id = "failed"

    graph.make_unavailable("failed")
    unblock()

    assert graph.externalBlocks == 0
    assert graph.is_empty()


def test_release_external_block_ignores_dropped_nodes_only():
    graph = scheduler({"failed": ["waiting"], "waiting": [], "other": []}, external={"waiting": 1})
    graph.staged_node_id = "failed"
    graph.make_unavailable("failed")

    graph.release_external_block("waiting")

    assert graph.externalBlocks == 0
    assert list(graph.pendingNodes) == ["other"]
    with pytest.raises(KeyError):
        graph.release_external_block("ghost")


def test_stalled_external_block_is_reported_once_nothing_can_release_it():
    graph = scheduler({"failed": ["releaser"], "releaser": [], "released": ["after"], "after": [], "other": []})
    graph.add_external_block("released")
    graph.staged_node_id = "failed"

    graph.make_unavailable("failed")

    assert not graph.is_stalled()
    graph.staged_node_id = "other"
    graph.complete_node_execution()
    assert graph.is_stalled()
    assert graph.get_externally_blocked_nodes() == ["released"]

    removed = [node_id for stalled in graph.get_externally_blocked_nodes() for node_id in graph.make_unavailable(stalled)]

    assert sorted(removed) == ["after", "released"]
    assert graph.externalBlocks == 0
    assert graph.is_empty()


def test_unavailable_node_cannot_reenter_the_scheduler():
    graph = scheduler({"failed": ["child"], "child": []})
    graph.staged_node_id = "failed"
    graph.make_unavailable("failed")

    graph.add_node("child")
    unblock = graph.add_external_block("child")
    unblock()

    assert graph.is_empty()
    assert graph.externalBlocks == 0
