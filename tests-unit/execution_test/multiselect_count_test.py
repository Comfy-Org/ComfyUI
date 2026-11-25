import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import execution
import nodes
from comfy_api.latest import io


pytestmark = pytest.mark.asyncio

OPTIONS = ["a.png", "b.png", "c.png"]


def picker_node(**multi_select):
    class Picker(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id="Picker", inputs=[
                io.MultiCombo.Input("image", options=OPTIONS, default=[], upload=io.UploadType.image, **multi_select),
            ], outputs=[], is_output_node=True)

        @classmethod
        def execute(cls, image):
            return io.NodeOutput()

    return Picker


@pytest.mark.parametrize("multi_select,value,expected_errors", [
    pytest.param({"min": 0, "max": 1}, {"__value__": []}, set(), id="empty-allowed-by-min-0"),
    pytest.param({"min": 0, "max": 1}, {"__value__": ["b.png"]}, set(), id="one-within-max"),
    pytest.param({"min": 1}, {"__value__": []}, {"value_smaller_than_min"}, id="empty-below-min"),
    pytest.param({"max": 2}, {"__value__": OPTIONS}, {"value_bigger_than_max"}, id="three-above-max"),
    pytest.param({"max": 1}, "a.png", set(), id="legacy-string-counts-as-one"),
    pytest.param({"min": 2}, "a.png", {"value_smaller_than_min"}, id="legacy-string-below-min"),
    pytest.param({"max": 1}, {"__value__": ["a.png", "x.png"]}, {"value_not_in_list"}, id="unknown-file-reported-once"),
    pytest.param({}, {"__value__": []}, set(), id="no-bounds"),
])
async def test_multiselect_selection_count_is_validated(monkeypatch, multi_select, value, expected_errors):
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Picker", picker_node(**multi_select))
    prompt = {"probe": {"class_type": "Picker", "inputs": {"image": value}}}

    valid, _, _, errors = await execution.validate_prompt("count", prompt, None)

    assert valid is (not expected_errors)
    reported = {error["type"] for error in errors.get("probe", {}).get("errors", [])}
    assert reported == expected_errors
