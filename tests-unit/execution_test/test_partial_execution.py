import asyncio
import gc
import weakref

import pytest

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


class UsePayload:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"payload": ("PAYLOAD",)}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, payload):
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
        "PartialUsePayload": UsePayload,
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


@pytest.mark.parametrize("cache_type", [CacheType.CLASSIC, CacheType.LRU, CacheType.RAM_PRESSURE])
def test_expansion_with_failed_side_branch_is_not_reused(cache_type):
    prompt = {
        "seed": {"class_type": "PartialConstant", "inputs": {"value": 1}},
        "expand": {"class_type": "PartialExpandWithFailingSideBranch", "inputs": {"value": ["seed", 0]}},
        "output": {"class_type": "PartialCapture", "inputs": {"value": ["expand", 0]}},
    }

    executor = run(prompt, ["output"], cache_type=cache_type)
    run(prompt, ["output"], cache_type=cache_type, executor=executor)

    assert STATE["expand_calls"] == 2
    assert STATE["fail_calls"] == 2


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

    removed = graph.make_unavailable("failed", "failed")

    assert sorted(removed) == ["child", "failed", "grandchild", "shared"]
    assert list(graph.pendingNodes) == ["other"]
    assert graph.blocking == {"other": {}}
    assert graph.staged_node_id is None
    assert all(graph.unavailable[node_id] == "failed" for node_id in removed)
    assert set(graph.execution_cache) == {"other"}


def test_make_unavailable_settles_external_blocks():
    graph = scheduler({"failed": ["waiting"], "waiting": []}, external={"waiting": 2})
    unblock = graph.add_external_block("waiting")
    graph.staged_node_id = "failed"

    graph.make_unavailable("failed", "failed")
    unblock()

    assert graph.externalBlocks == 0
    assert graph.is_empty()


def test_dropped_releaser_drops_the_node_it_would_release():
    graph = scheduler({"failed": ["releaser"], "releaser": [], "released": ["after"], "after": []})
    graph.add_external_block("released", released_by="releaser")
    graph.staged_node_id = "failed"

    removed = graph.make_unavailable("failed", "failed")

    assert sorted(removed) == ["after", "failed", "released", "releaser"]
    assert graph.externalBlocks == 0
    assert graph.is_empty()
