import copy
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


pytestmark = pytest.mark.asyncio


def _register_probe_nodes(monkeypatch):
    """Register a fingerprinted Source and an autogrow Consumer and return the shared execution log."""
    executions = []
    change_counter = [0]

    class Source(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id=cls.__name__, inputs=[], outputs=[io.Float.Output()])

        @classmethod
        def fingerprint_inputs(cls):
            change_counter[0] += 1
            return change_counter[0]

        @classmethod
        def execute(cls):
            executions.append("source")
            return io.NodeOutput(float(change_counter[0]))

    class Consumer(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id=cls.__name__,
                inputs=[io.Autogrow.Input("images", template=io.Autogrow.TemplateNames(
                    io.Float.Input("image"), names=[f"image_{i}" for i in range(1, 17)], min=0))],
                outputs=[],
                is_output_node=True,
            )

        @classmethod
        def execute(cls, images):
            executions.append(("consumer", dict(images or {})))
            return io.NodeOutput()

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Source", Source)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Consumer", Consumer)
    return executions


async def test_autogrow_rejects_nested_list_of_links(monkeypatch):
    # A list of link pairs under the autogrow key itself ("images": [["source", 0]])
    # used to be silently dropped: the entries never reached the node and the
    # producer never ran, while every execution replayed the same output.
    _register_probe_nodes(monkeypatch)
    prompt = {
        "source": {"class_type": "Source", "inputs": {}},
        "consumer": {"class_type": "Consumer", "inputs": {"images": [["source", 0]]}},
    }
    valid, _, _, errors = await execution.validate_prompt("probe", prompt, None)

    assert not valid
    consumer_errors = errors["consumer"]["errors"]
    assert consumer_errors[0]["type"] == "invalid_dynamic_input"
    assert "images" in consumer_errors[0]["details"]
    assert "images.image_1" in consumer_errors[0]["details"]


async def test_autogrow_flat_entries_rerun_when_upstream_reruns(monkeypatch):
    # The flat form must stay content-aware: the consumer reruns when the
    # upstream node reruns, instead of replaying its stale output from cache.
    executions = _register_probe_nodes(monkeypatch)
    prompt = {
        "source": {"class_type": "Source", "inputs": {}},
        "consumer": {"class_type": "Consumer", "inputs": {"images.image_1": ["source", 0]}},
    }
    valid, _, outputs, errors = await execution.validate_prompt("probe", copy.deepcopy(prompt), None)
    assert valid, errors
    server = SimpleNamespace(client_id=None, send_sync=Mock())
    executor = execution.PromptExecutor(server, cache_args={"ram": 0, "ram_inactive": 0})

    await executor.execute_async(copy.deepcopy(prompt), "probe1", execute_outputs=outputs)
    assert executor.success, executor.status_messages
    # a separate submission of the same graph, like a second API queue call
    await executor.execute_async(copy.deepcopy(prompt), "probe2", execute_outputs=outputs)
    assert executor.success, executor.status_messages

    source_runs = executions.count("source")
    consumer_runs = [e[1] for e in executions if isinstance(e, tuple)]
    assert source_runs == 2, "source should rerun every time (is_changed)"
    assert consumer_runs == [{"image_1": 1.0}, {"image_1": 2.0}]


async def test_autogrow_empty_root_list_coexists_with_flat_entries(monkeypatch):
    # An empty list under the autogrow key itself is allowed; only a
    # non-empty list of link pairs is rejected.
    executions = _register_probe_nodes(monkeypatch)
    prompt = {
        "source": {"class_type": "Source", "inputs": {}},
        "consumer": {"class_type": "Consumer", "inputs": {"images": [], "images.image_1": ["source", 0]}},
    }
    valid, _, outputs, errors = await execution.validate_prompt("probe", copy.deepcopy(prompt), None)
    assert valid, errors
    server = SimpleNamespace(client_id=None, send_sync=Mock())
    executor = execution.PromptExecutor(server, cache_args={"ram": 0, "ram_inactive": 0})

    await executor.execute_async(copy.deepcopy(prompt), "probe1", execute_outputs=outputs)
    assert executor.success, executor.status_messages

    consumer_runs = [e[1] for e in executions if isinstance(e, tuple)]
    assert consumer_runs == [{"image_1": 1.0}]
