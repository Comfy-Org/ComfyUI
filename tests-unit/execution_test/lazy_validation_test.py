from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import execution
import nodes
from comfy_api.latest import io
from comfy_extras.nodes_logic import SwitchNode
from comfy_extras.nodes_primitive import Boolean


pytestmark = pytest.mark.asyncio


@pytest.fixture
def loaded(monkeypatch):
    loaded = []

    class Loader(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id=cls.__name__,
                inputs=[io.Combo.Input("ckpt_name", options=["good.safetensors"])],
                outputs=[io.Float.Output()],
            )

        @classmethod
        def execute(cls, ckpt_name):
            loaded.append(ckpt_name)
            return io.NodeOutput(1.0)

    class Sink(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id=cls.__name__,
                inputs=[io.Float.Input("value"), io.Float.Input("extra", optional=True)],
                outputs=[],
                is_output_node=True,
            )

        @classmethod
        def execute(cls, value, extra=None):
            return io.NodeOutput(ui={"value": [value]})

    for node_class in (Loader, Sink, SwitchNode, Boolean):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, node_class.GET_SCHEMA().node_id, node_class)
    return loaded


def switch_prompt(switch):
    return {
        "1": {"class_type": "Loader", "inputs": {"ckpt_name": "good.safetensors"}},
        "2": {"class_type": "Loader", "inputs": {"ckpt_name": "missing.safetensors"}},
        "3": {"class_type": "ComfySwitchNode", "inputs": {"switch": switch, "on_true": ["1", 0], "on_false": ["2", 0]}},
        "out": {"class_type": "Sink", "inputs": {"value": ["3", 0]}},
    }


async def validate(prompt):
    deferred = {}
    valid, error, outputs, node_errors = await execution.validate_prompt("lazy", prompt, None, deferred_node_errors=deferred)
    return valid, error, outputs, node_errors, deferred


async def run(prompt, outputs, deferred):
    server = SimpleNamespace(client_id=None, send_sync=Mock())
    executor = execution.PromptExecutor(server, cache_type=execution.CacheType.NONE, cache_args={"ram": 0, "ram_inactive": 0})
    await executor.execute_async(prompt, "lazy", extra_data={"deferred_node_errors": deferred}, execute_outputs=outputs)
    return executor


def execution_error(executor):
    return next(data for event, data in executor.status_messages if event == "execution_error")


async def test_unselected_broken_branch_is_deferred(loaded):
    prompt = switch_prompt(True)
    valid, error, outputs, node_errors, deferred = await validate(prompt)

    assert valid, error
    assert outputs == ["out"]
    assert node_errors == {}
    assert list(deferred) == ["2"]
    assert deferred["2"]["dependent_outputs"] == ["out"]
    assert [e["type"] for e in deferred["2"]["errors"]] == ["value_not_in_list"]

    executor = await run(prompt, outputs, deferred)
    assert executor.success, executor.status_messages
    assert loaded == ["good.safetensors"]


async def test_selected_broken_branch_fails_at_execution(loaded):
    prompt = switch_prompt(False)
    valid, error, outputs, node_errors, deferred = await validate(prompt)

    assert valid, error
    assert node_errors == {}
    assert list(deferred) == ["2"]

    executor = await run(prompt, outputs, deferred)
    assert not executor.success
    assert loaded == []
    error = execution_error(executor)
    assert error["node_id"] == "2"
    assert error["exception_type"] == "execution.DeferredValidationError"
    assert "missing.safetensors" in error["exception_message"]


async def test_linked_selector_is_deferred(loaded):
    prompt = switch_prompt(["bool", 0])
    prompt["bool"] = {"class_type": "PrimitiveBoolean", "inputs": {"value": True}}
    valid, error, outputs, node_errors, deferred = await validate(prompt)

    assert valid, error
    assert node_errors == {}
    assert list(deferred) == ["2"]


async def test_broken_node_also_linked_strongly_is_rejected(loaded):
    prompt = switch_prompt(True)
    prompt["out"]["inputs"]["extra"] = ["2", 0]
    valid, error, outputs, node_errors, deferred = await validate(prompt)

    assert not valid
    assert error["type"] == "prompt_outputs_failed_validation"
    assert outputs == []
    assert list(node_errors) == ["2"]
    assert deferred == {}


async def test_strong_and_lazy_consumers_are_reported_per_output(loaded):
    prompt = switch_prompt(True)
    prompt["direct"] = {"class_type": "Sink", "inputs": {"value": ["2", 0]}}
    valid, error, outputs, node_errors, deferred = await validate(prompt)

    assert valid, error
    assert outputs == ["out"]
    assert list(node_errors) == ["2"]
    assert node_errors["2"]["dependent_outputs"] == ["direct"]
    assert deferred["2"]["dependent_outputs"] == ["out"]
