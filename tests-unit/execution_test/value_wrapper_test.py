import copy

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import execution
import nodes
from comfy_api.latest import io


def test_unwrap_value_takes_the_wrapper_off():
    """A wrapped widget value reaches the node unwrapped."""
    boxes = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
    assert execution.unwrap_value({"__value__": boxes}) == boxes


def test_unwrap_value_leaves_a_link_alone():
    """A link is a plain list and must stay one: only the wrapper is taken off."""
    assert execution.unwrap_value(["5", 0]) == ["5", 0]  # node "5", output 0


def test_unwrap_value_leaves_other_values_alone():
    assert execution.unwrap_value({"a": 1}) == {"a": 1}
    assert execution.unwrap_value("text") == "text"
    assert execution.unwrap_value(7) == 7
    assert execution.unwrap_value(None) is None


@pytest.mark.asyncio
async def test_validation_leaves_the_wrapper_in_the_prompt(monkeypatch):
    """The prompt that is validated is the one stored in /history and in the saved
    image's `prompt` chunk. Unwrapping it in place left an array widget's value
    looking exactly like a node link, so loading the prompt again dropped it and the
    widget came back empty.
    """
    boxes = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]

    class Boxes(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id=cls.__name__, inputs=[
                io.Custom("EDITOR_STATE").Input("editor_state"),
            ], outputs=[], is_output_node=True)

        @classmethod
        def execute(cls, **kwargs):
            return io.NodeOutput()

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Boxes", Boxes)
    prompt = {"probe": {"class_type": "Boxes", "inputs": {"editor_state": {"__value__": boxes}}}}
    before = copy.deepcopy(prompt)

    valid, _, _, _ = await execution.validate_prompt("wrapper", prompt, None)

    assert valid
    assert prompt == before, "validation must not rewrite the prompt it was handed"
    assert prompt["probe"]["inputs"]["editor_state"] == {"__value__": boxes}


def test_the_node_still_receives_the_unwrapped_value(monkeypatch):
    """Taking the unwrap out of validation must not leave the node holding the wrapper."""
    boxes = [{"x": 1, "y": 2}]

    class Boxes(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id=cls.__name__, inputs=[
                io.Custom("EDITOR_STATE").Input("editor_state"),
            ], outputs=[], is_output_node=True)

        @classmethod
        def execute(cls, editor_state):
            return io.NodeOutput()

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Boxes", Boxes)
    inputs = {"editor_state": {"__value__": boxes}}
    input_data_all, _, _ = execution.get_input_data(inputs, Boxes, "probe")

    assert input_data_all["editor_state"] == [boxes]
