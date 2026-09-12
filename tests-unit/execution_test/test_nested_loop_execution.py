import copy

import pytest

import nodes

import comfy_extras.nodes_loop as nodes_loop
from comfy_api.latest import io
from comfy_execution.graph_utils import GraphBuilder
from comfy_execution.validation import validate_loops
from execution import PromptExecutor


class Constant:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT",)}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, value):
        return (value,)


class Increment:
    calls = []

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT",)}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, value):
        value += 1
        self.calls.append(value)
        return (value,)


class ExpandIncrement:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT",)}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, value):
        graph = GraphBuilder()
        increment = graph.node("TestIncrement", "increment", value=value)
        return {"result": (increment.out(0),), "expand": graph.finalize()}


class FalseBranch:
    calls = []

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT",)}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, value):
        self.calls.append(value)
        return (value,)


class TrueBranch(FalseBranch):
    calls = []


class LazySwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "switch": ("BOOLEAN",),
                "on_false": ("INT", {"lazy": True}),
                "on_true": ("INT", {"lazy": True}),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def check_lazy_status(self, switch, on_false=None, on_true=None):
        selected_name = "on_true" if switch else "on_false"
        selected = on_true if switch else on_false
        return [selected_name] if selected is None else []

    def execute(self, switch, on_false=None, on_true=None):
        return (on_true if switch else on_false,)


class Capture:
    values = []

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT",)}}

    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True

    def execute(self, value):
        self.values.append(value)
        return ()


class CapturePassthrough:
    values = []

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT",)}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"
    OUTPUT_NODE = True

    def execute(self, value):
        self.values.append(value)
        return (value,)


class CapturePassthroughFirst(CapturePassthrough):
    values = []


class CapturePassthroughSecond(io.ComfyNode):
    values = []

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TestCapturePassthroughSecond",
            inputs=[io.Int.Input("value"), io.Boolean.Input("last")],
            outputs=[io.Int.Output()],
            hidden=[io.Hidden.dynprompt, io.Hidden.unique_id],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, value, last):
        cls.values.append((value, last))
        return io.NodeOutput(value)

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("NaN")


class Pair:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT",)}}

    RETURN_TYPES = ("*",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "execute"

    def execute(self, value):
        return ([value, str(value)],)


class ListBackedScalar:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT",)}}

    RETURN_TYPES = ("*",)
    FUNCTION = "execute"

    def execute(self, value):
        return ([[value]],)


class AppendIndex:
    values = []

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("*",), "index": ("INT",)}}

    RETURN_TYPES = ("*",)
    FUNCTION = "execute"

    def execute(self, value, index):
        value = value + [index]
        self.values.append(value)
        return (value,)


class EmptyList:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("*",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "execute"

    def execute(self):
        return ([],)


class IntegerList:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("INT",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "execute"

    def execute(self):
        return ([10, 20],)


class CaptureLoopState:
    values = []

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT",),
                "is_first": ("BOOLEAN",),
                "is_last": ("BOOLEAN",),
                "item": ("*",),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"

    def execute(self, index, is_first, is_last, item):
        self.values.append((index, is_first, is_last, item))
        return (index,)


class CaptureLoopResult(nodes_loop.LoopResult):
    output_keys = []

    @classmethod
    def execute(cls, **kwargs):
        cls.output_keys.extend(sorted(name for name in kwargs if name.startswith("output")))
        return super().execute(**kwargs)


class Server:
    client_id = None

    def send_sync(self, *args, **kwargs):
        pass


class Progress:
    messages = []
    body_call_counts = []

    def send_progress_text(self, text, node_id):
        self.messages.append((text, node_id))
        self.body_call_counts.append(len(Increment.calls))


@pytest.fixture(autouse=True)
def register_internal_loop_nodes(monkeypatch):
    classes = {
        "StartLoop": nodes_loop.StartLoop,
        "EndLoop": nodes_loop.EndLoop,
        "LoopIteration": nodes_loop.LoopIteration,
        "LoopProgress": nodes_loop.LoopProgress,
        "LoopResult": nodes_loop.LoopResult,
        "TestConstant": Constant,
        "TestIncrement": Increment,
        "TestExpandIncrement": ExpandIncrement,
        "TestFalseBranch": FalseBranch,
        "TestTrueBranch": TrueBranch,
        "TestLazySwitch": LazySwitch,
        "TestCapture": Capture,
        "TestCapturePassthrough": CapturePassthrough,
        "TestCapturePassthroughFirst": CapturePassthroughFirst,
        "TestCapturePassthroughSecond": CapturePassthroughSecond,
        "TestPair": Pair,
        "TestListBackedScalar": ListBackedScalar,
        "TestAppendIndex": AppendIndex,
        "TestEmptyList": EmptyList,
        "TestIntegerList": IntegerList,
        "TestCaptureLoopState": CaptureLoopState,
    }
    for name, node in classes.items():
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, name, node)
    monkeypatch.setattr(
        nodes_loop,
        "PromptServer",
        type("PromptServer", (), {"instance": Progress()}),
    )
    execute = PromptExecutor.execute

    def execute_validated(executor, prompt, prompt_id, extra_data=None, execute_outputs=None):
        extra_data = extra_data or {}
        execute_outputs = execute_outputs or []
        starts = {node_id for node_id, node in prompt.items() if node["class_type"] == "StartLoop"}
        ends = {node_id for node_id, node in prompt.items() if node["class_type"] == "EndLoop"}
        validate_loops(prompt, set(execute_outputs), prompt, starts, ends)
        execute(executor, prompt, prompt_id, extra_data, execute_outputs)

    monkeypatch.setattr(PromptExecutor, "execute", execute_validated)


def execute_prompt(prompt, prompt_id, outputs):
    executor = PromptExecutor(Server(), cache_type=False, cache_args={"ram": 0, "ram_inactive": 0})
    executor.execute(prompt, prompt_id, execute_outputs=outputs)
    assert executor.success
    return executor


def test_nested_loops_execute_each_body_once_without_final_requeue():
    Increment.calls = []
    Capture.values = []

    prompt = {
        "constant": {
            "class_type": "TestConstant",
            "inputs": {"value": 0},
        },
        "outer": {
            "class_type": "StartLoop",
            "inputs": {
                "mode": "simple",
                "mode.num_iterations": 2,
                "initial_iteration_value": ["constant", 0],
            },
        },
        "inner": {
            "class_type": "StartLoop",
            "inputs": {
                "mode": "simple",
                "mode.num_iterations": 2,
                "parent_iteration": ["outer", 0],
                "initial_iteration_value": ["outer", 4],
            },
        },
        "increment": {
            "class_type": "TestIncrement",
            "inputs": {"value": ["inner", 4]},
        },
        "inner_close": {
            "class_type": "EndLoop",
            "inputs": {
                "output_value": ["increment", 0],
                "next_iteration_value": ["increment", 0],
                "accumulate": False,
            },
        },
        "outer_close": {
            "class_type": "EndLoop",
            "inputs": {
                "output_value": ["inner_close", 0],
                "next_iteration_value": ["inner_close", 0],
                "accumulate": False,
            },
        },
        "capture": {
            "class_type": "TestCapture",
            "inputs": {"value": ["outer_close", 0]},
        },
    }
    execute_prompt(prompt, "nested-loop-test", ["capture"])
    assert Increment.calls == [1, 2, 3, 4]
    assert Capture.values == [4]


def test_loop_executes_termination_without_carried_or_output_value():
    Increment.calls = []
    CapturePassthrough.values = []

    prompt = {
        "loop": {
            "class_type": "StartLoop",
            "inputs": {
                "mode": "simple",
                "mode.num_iterations": 2,
            },
        },
        "increment": {
            "class_type": "TestIncrement",
            "inputs": {"value": ["loop", 0]},
        },
        "preview": {
            "class_type": "TestCapturePassthrough",
            "inputs": {"value": ["increment", 0]},
        },
        "close": {
            "class_type": "EndLoop",
            "inputs": {
                "accumulate": False,
                "termination0": ["preview", 0],
            },
        },
    }
    execute_prompt(prompt, "termination-only-loop-test", ["preview"])
    assert Increment.calls == [1, 2]
    assert CapturePassthrough.values == [1, 2]


def test_loop_executes_multiple_termination_branches_each_iteration():
    Increment.calls = []
    CapturePassthroughFirst.values = []
    CapturePassthroughSecond.values = []
    prompt = {
        "initial": {"class_type": "TestConstant", "inputs": {"value": 0}},
        "loop": {
            "class_type": "StartLoop",
            "inputs": {
                "mode": "simple",
                "mode.num_iterations": 2,
                "initial_iteration_value": ["initial", 0],
            },
        },
        "increment": {"class_type": "TestIncrement", "inputs": {"value": ["loop", 4]}},
        "first": {
            "class_type": "TestCapturePassthroughFirst",
            "inputs": {"value": ["increment", 0]},
        },
        "second": {
            "class_type": "TestCapturePassthroughSecond",
            "inputs": {"value": ["increment", 0], "last": ["loop", 2]},
        },
        "close": {
            "class_type": "EndLoop",
            "inputs": {
                "output_value": ["increment", 0],
                "next_iteration_value": ["increment", 0],
                "accumulate": False,
                "termination0": ["first", 0],
                "termination1": ["second", 0],
            },
        },
    }

    executor = execute_prompt(prompt, "multiple-termination-loop-test", ["first", "second"])

    assert executor.success
    assert Increment.calls == [1, 2]
    assert CapturePassthroughFirst.values == [1, 2]
    assert CapturePassthroughSecond.values == [(1, False), (2, True)]


def test_loop_executes_final_carried_value_without_output():
    Increment.calls = []
    prompt = {
        "loop": {
            "class_type": "StartLoop",
            "inputs": {"mode": "simple", "mode.num_iterations": 2},
        },
        "increment": {"class_type": "TestIncrement", "inputs": {"value": ["loop", 0]}},
        "close": {
            "class_type": "EndLoop",
            "inputs": {"next_iteration_value": ["increment", 0], "accumulate": False},
        },
    }
    execute_prompt(prompt, "carry-only-loop-test", ["close"])
    assert Increment.calls == [1, 2]


def test_loop_preserves_list_backed_carried_value():
    AppendIndex.values = []
    prompt = {
        "initial": {"class_type": "TestListBackedScalar", "inputs": {"value": 7}},
        "loop": {
            "class_type": "StartLoop",
            "inputs": {
                "mode": "simple",
                "mode.num_iterations": 2,
                "initial_iteration_value": ["initial", 0],
            },
        },
        "append": {
            "class_type": "TestAppendIndex",
            "inputs": {"value": ["loop", 4], "index": ["loop", 0]},
        },
        "close": {
            "class_type": "EndLoop",
            "inputs": {
                "output_value": ["append", 0],
                "next_iteration_value": ["append", 0],
                "accumulate": False,
            },
        },
    }
    execute_prompt(prompt, "list-carry-loop-test", ["close"])
    assert AppendIndex.values == [[[7], 0], [[7], 0, 1]]


@pytest.mark.parametrize(
    "mode_inputs",
    [
        {"mode": "simple", "mode.num_iterations": 0},
        {"mode": "List", "mode.list": ["empty_list", 0]},
    ],
)
def test_empty_loop_skips_body(mode_inputs):
    Increment.calls = []
    CapturePassthrough.values = []
    prompt = {
        "empty_list": {
            "class_type": "TestEmptyList",
            "inputs": {},
        },
        "loop": {
            "class_type": "StartLoop",
            "inputs": mode_inputs,
        },
        "increment": {
            "class_type": "TestIncrement",
            "inputs": {"value": ["loop", 0]},
        },
        "preview": {
            "class_type": "TestCapturePassthrough",
            "inputs": {"value": ["increment", 0]},
        },
        "close": {
            "class_type": "EndLoop",
            "inputs": {
                "accumulate": False,
                "termination0": ["preview", 0],
            },
        },
    }
    executor = execute_prompt(prompt, "empty-loop-test", ["preview"])
    assert Increment.calls == []
    assert CapturePassthrough.values == []
    assert executor.caches.outputs.get_local("close") is not None


@pytest.mark.parametrize(
    ("mode_inputs", "expected"),
    [
        (
            {"mode": "For", "mode.start_iteration_index": 2, "mode.max_iteration": 8, "mode.step": 3},
            [(2, True, False, None), (5, False, True, None)],
        ),
        (
            {"mode": "List", "mode.list": ["items", 0]},
            [(0, True, False, 10), (1, False, True, 20)],
        ),
    ],
)
def test_loop_modes_expose_iteration_state(mode_inputs, expected):
    CaptureLoopState.values = []
    prompt = {
        "items": {"class_type": "TestIntegerList", "inputs": {}},
        "loop": {"class_type": "StartLoop", "inputs": mode_inputs},
        "state": {
            "class_type": "TestCaptureLoopState",
            "inputs": {
                "index": ["loop", 0],
                "is_first": ["loop", 1],
                "is_last": ["loop", 2],
                "item": ["loop", 3],
            },
        },
        "close": {
            "class_type": "EndLoop",
            "inputs": {"output_value": ["state", 0], "accumulate": True},
        },
    }
    execute_prompt(prompt, "loop-mode-test", ["close"])
    assert CaptureLoopState.values == expected


@pytest.mark.parametrize(("cache_iterations", "expected_calls"), [(False, [1, 2, 1, 2]), (True, [1, 2])])
def test_iteration_cache_policy_and_end_cache(cache_iterations, expected_calls):
    Increment.calls = []
    Capture.values = []
    Progress.messages = []
    Progress.body_call_counts = []
    prompt = {
        "loop": {
            "class_type": "StartLoop",
            "inputs": {
                "mode": "simple",
                "mode.num_iterations": 2,
                "cache_iterations": cache_iterations,
            },
        },
        "increment": {"class_type": "TestIncrement", "inputs": {"value": ["loop", 0]}},
        "close": {
            "class_type": "EndLoop",
            "inputs": {"output_value": ["increment", 0], "accumulate": True},
        },
        "capture": {"class_type": "TestCapture", "inputs": {"value": ["close", 0]}},
    }
    second_prompt = copy.deepcopy(prompt)
    executor = PromptExecutor(Server(), cache_type=False, cache_args={"ram": 0, "ram_inactive": 0})

    executor.execute(prompt, "loop-cache-first", execute_outputs=["capture"])
    assert executor.success
    assert executor.caches.outputs.get_local("close") is not None
    executor.execute(second_prompt, "loop-cache-second", execute_outputs=["capture"])

    assert executor.success
    assert Increment.calls == expected_calls
    assert Progress.messages == [
        ("Iteration 0 / 2", "loop"),
        ("Iteration 1 / 2", "loop"),
        ("Iteration 2 / 2", "loop"),
    ] * 2
    assert Progress.body_call_counts == [0, 1, 2] + ([2, 2, 2] if cache_iterations else [2, 3, 4])
    assert prompt["close"]["inputs"]["output_value"] == ["increment", 0]
    assert second_prompt["close"]["inputs"]["output_value"] == ["increment", 0]


def test_iteration_cache_still_expands_when_only_termination_is_requested():
    Increment.calls = []
    CapturePassthrough.values = []
    prompt = {
        "loop": {
            "class_type": "StartLoop",
            "inputs": {
                "mode": "simple",
                "mode.num_iterations": 2,
                "cache_iterations": True,
            },
        },
        "increment": {"class_type": "TestIncrement", "inputs": {"value": ["loop", 0]}},
        "preview": {
            "class_type": "TestCapturePassthrough",
            "inputs": {"value": ["increment", 0]},
        },
        "close": {
            "class_type": "EndLoop",
            "inputs": {"accumulate": False, "termination0": ["preview", 0]},
        },
    }
    second_prompt = copy.deepcopy(prompt)
    executor = PromptExecutor(Server(), cache_type=False, cache_args={"ram": 0, "ram_inactive": 0})

    executor.execute(prompt, "termination-cache-first", execute_outputs=["preview"])
    assert executor.success
    executor.execute(second_prompt, "termination-cache-second", execute_outputs=["preview"])

    assert executor.success
    assert Increment.calls == [1, 2]
    assert CapturePassthrough.values == [1, 2]


def test_single_loop_concatenates_list_outputs():
    Capture.values = []
    prompt = {
        "loop": {
            "class_type": "StartLoop",
            "inputs": {"mode": "simple", "mode.num_iterations": 2},
        },
        "pair": {
            "class_type": "TestPair",
            "inputs": {"value": ["loop", 0]},
        },
        "close": {
            "class_type": "EndLoop",
            "inputs": {
                "output_value": ["pair", 0],
                "accumulate": True,
            },
        },
        "capture": {
            "class_type": "TestCapture",
            "inputs": {"value": ["close", 0]},
        },
    }
    execute_prompt(prompt, "single-loop-accumulation-test", ["capture"])
    assert Capture.values == [0, "0", 1, "1"]


def test_loop_final_output_preserves_output_list(monkeypatch):
    Capture.values = []
    CaptureLoopResult.output_keys.clear()
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "LoopResult", CaptureLoopResult)
    prompt = {
        "loop": {"class_type": "StartLoop", "inputs": {"mode": "simple", "mode.num_iterations": 2}},
        "pair": {"class_type": "TestPair", "inputs": {"value": ["loop", 0]}},
        "close": {
            "class_type": "EndLoop",
            "inputs": {"output_value": ["pair", 0], "accumulate": False},
        },
        "capture": {"class_type": "TestCapture", "inputs": {"value": ["close", 0]}},
    }
    execute_prompt(prompt, "final-list-loop-test", ["capture"])
    assert Capture.values == [1, "1"]
    assert CaptureLoopResult.output_keys == ["output0"]


def test_loop_rebuilds_lazy_branch_dependencies_each_iteration():
    FalseBranch.calls = []
    TrueBranch.calls = []
    prompt = {
        "loop": {"class_type": "StartLoop", "inputs": {"mode": "simple", "mode.num_iterations": 2}},
        "false": {"class_type": "TestFalseBranch", "inputs": {"value": ["loop", 0]}},
        "true": {"class_type": "TestTrueBranch", "inputs": {"value": ["loop", 0]}},
        "switch": {
            "class_type": "TestLazySwitch",
            "inputs": {
                "switch": ["loop", 1],
                "on_false": ["false", 0],
                "on_true": ["true", 0],
            },
        },
        "close": {
            "class_type": "EndLoop",
            "inputs": {"output_value": ["switch", 0], "accumulate": True},
        },
    }
    execute_prompt(prompt, "lazy-branch-loop-test", ["close"])
    assert TrueBranch.calls == [0]
    assert FalseBranch.calls == [1]


def test_loop_repeats_runtime_expanded_descendants():
    Increment.calls = []
    prompt = {
        "loop": {"class_type": "StartLoop", "inputs": {"mode": "simple", "mode.num_iterations": 3}},
        "expand": {"class_type": "TestExpandIncrement", "inputs": {"value": ["loop", 0]}},
        "close": {
            "class_type": "EndLoop",
            "inputs": {"output_value": ["expand", 0], "accumulate": True},
        },
    }
    execute_prompt(prompt, "expanded-descendant-loop-test", ["close"])
    assert Increment.calls == [1, 2, 3]


def run_nested_accumulation(producer_name):
    Capture.values = []

    prompt = {
        "outer": {
            "class_type": "StartLoop",
            "inputs": {"mode": "simple", "mode.num_iterations": 2},
        },
        "inner": {
            "class_type": "StartLoop",
            "inputs": {
                "mode": "simple",
                "mode.num_iterations": 2,
                "parent_iteration": ["outer", 0],
            },
        },
        "producer": {
            "class_type": producer_name,
            "inputs": {"value": ["inner", 0]},
        },
        "inner_close": {
            "class_type": "EndLoop",
            "inputs": {
                "output_value": ["producer", 0],
                "accumulate": True,
            },
        },
        "outer_close": {
            "class_type": "EndLoop",
            "inputs": {
                "output_value": ["inner_close", 0],
                "accumulate": True,
            },
        },
        "capture": {
            "class_type": "TestCapture",
            "inputs": {"value": ["outer_close", 0]},
        },
    }
    execute_prompt(prompt, "nested-loop-accumulation-test", ["capture"])
    return Capture.values


def test_nested_loop_concatenates_lists():
    assert run_nested_accumulation("TestPair") == [
        0,
        "0",
        1,
        "1",
        0,
        "0",
        1,
        "1",
    ]


def test_nested_loop_does_not_flatten_list_backed_scalars():
    assert run_nested_accumulation("TestListBackedScalar") == [
        [[0]],
        [[1]],
        [[0]],
        [[1]],
    ]
