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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "node_output",
    [
        execution.ExecutionBlocker("blocked"),
        io.NodeOutput(block_execution="blocked"),
    ],
    ids=["execution-blocker", "node-output"],
)
async def test_v3_blocker_reaches_downstream_nodes(monkeypatch, node_output):
    received = []

    class Block(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id=cls.__name__, outputs=[io.Int.Output(), io.Int.Output()])

        @classmethod
        def execute(cls):
            return node_output

    class Capture(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id=cls.__name__, inputs=[io.Int.Input("value")], is_output_node=True)

        @classmethod
        def execute(cls, value):
            received.append(value)
            return io.NodeOutput()

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "V3Block", Block)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "V3Capture", Capture)
    prompt = {
        "block": {"class_type": "V3Block", "inputs": {}},
        "first": {"class_type": "V3Capture", "inputs": {"value": ["block", 0]}},
        "second": {"class_type": "V3Capture", "inputs": {"value": ["block", 1]}},
        "independent": {"class_type": "V3Capture", "inputs": {"value": 7}},
    }
    valid, _, outputs, errors = await execution.validate_prompt("blocker", prompt, None)
    assert valid, errors

    server = SimpleNamespace(client_id=None, send_sync=Mock())
    executor = execution.PromptExecutor(server, cache_type=execution.CacheType.NONE, cache_args={"ram": 0, "ram_inactive": 0})
    await executor.execute_async(prompt, "blocker", execute_outputs=outputs)

    assert executor.success, executor.status_messages
    assert received == [7]
    assert {
        (call.args[1]["node_id"], call.args[1]["exception_message"])
        for call in server.send_sync.call_args_list
        if call.args[0] == "execution_error"
    } == {
        ("first", "Execution Blocked: blocked"),
        ("second", "Execution Blocked: blocked"),
    }
