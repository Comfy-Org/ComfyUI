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


def v3_picker(**bounds):
    class Picker(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id="Picker", inputs=[
                io.MultiCombo.Input("image", options=OPTIONS, default=[], upload=io.UploadType.image, **bounds),
            ], outputs=[], is_output_node=True)

        @classmethod
        def execute(cls, image):
            return io.NodeOutput()

    return Picker


def v1_picker(**bounds):
    class Picker:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"image": (OPTIONS, {"image_upload": True, **bounds})}}

        RETURN_TYPES = ()
        FUNCTION = "run"
        OUTPUT_NODE = True

        def run(self, image):
            return ()

    return Picker


@pytest.mark.parametrize("make_picker", [v3_picker, v1_picker], ids=["v3-multicombo", "v1-combo-without-multiselect"])
@pytest.mark.parametrize("bounds,value,expected_errors", [
    pytest.param({"min": 0, "max": 2}, {"__value__": []}, set(), id="empty-allowed-by-min-0"),
    pytest.param({"min": 0, "max": 2}, {"__value__": ["b.png", "a.png"]}, set(), id="two-within-max"),
    pytest.param({"max": 2}, {"__value__": OPTIONS}, {"value_bigger_than_max"}, id="three-above-max"),
    pytest.param({"min": 0}, {"__value__": ["a.png", "b.png"]}, {"value_bigger_than_max"}, id="min-only-defaults-max-to-1"),
    pytest.param({"max": 3}, {"__value__": []}, {"value_smaller_than_min"}, id="max-only-defaults-min-to-1"),
    pytest.param({"min": 2, "max": 3}, {"__value__": ["a.png"]}, {"value_smaller_than_min"}, id="one-below-min"),
    pytest.param({"min": 0, "max": 1}, "a.png", set(), id="legacy-string-counts-as-one"),
    pytest.param({"min": 2, "max": 3}, "a.png", {"value_smaller_than_min"}, id="legacy-string-below-min"),
    pytest.param({"min": 0, "max": 2}, {"__value__": ["a.png", "x.png"]}, {"value_not_in_list"}, id="unknown-file-reported-once"),
])
async def test_combo_selection_count_is_validated(monkeypatch, make_picker, bounds, value, expected_errors):
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Picker", make_picker(**bounds))
    prompt = {"probe": {"class_type": "Picker", "inputs": {"image": value}}}

    valid, _, _, errors = await execution.validate_prompt("count", prompt, None)

    reported = {error["type"] for error in errors.get("probe", {}).get("errors", [])}
    assert reported == expected_errors
    assert valid is (not expected_errors)


async def test_unbounded_combo_still_rejects_a_list(monkeypatch):
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "Picker", v1_picker())
    prompt = {"probe": {"class_type": "Picker", "inputs": {"image": {"__value__": ["a.png"]}}}}

    valid, _, _, errors = await execution.validate_prompt("count", prompt, None)

    assert valid is False
    assert {error["type"] for error in errors["probe"]["errors"]} == {"value_not_in_list"}
