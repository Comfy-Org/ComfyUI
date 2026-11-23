import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import execution
import nodes
from comfy_api.latest import io
from comfy_extras.nodes_primitive import Combo


pytestmark = pytest.mark.asyncio


class V1Target:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"sampler_name": (["euler", "heun"],)}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "run"


class V3Target(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="V3Target", inputs=[io.Combo.Input("choice", options=["euler", "dpm"])], outputs=[], is_output_node=True)

    @classmethod
    def execute(cls, choice):
        return io.NodeOutput()


class MultiTarget(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MultiTarget", inputs=[io.MultiCombo.Input("choice", options=["euler", "dpm"])], outputs=[], is_output_node=True)

    @classmethod
    def execute(cls, choice):
        return io.NodeOutput()


class OptOutTarget(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="OptOutTarget", inputs=[io.Combo.Input("choice", options=["euler"])], outputs=[], is_output_node=True)

    @classmethod
    def validate_inputs(cls, choice):
        return True

    @classmethod
    def execute(cls, choice):
        return io.NodeOutput()


@pytest.fixture(autouse=True)
def register(monkeypatch):
    for name, cls in [("PrimitiveCombo", Combo), ("V1Target", V1Target), ("V3Target", V3Target), ("MultiTarget", MultiTarget), ("OptOutTarget", OptOutTarget)]:
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, name, cls)


@pytest.mark.parametrize("value,targets,bad_inputs", [
    pytest.param("euler", {"V1Target": "sampler_name", "V3Target": "choice"}, [], id="valid-v1-and-v3"),
    pytest.param("heun", {"V1Target": "sampler_name", "V3Target": "choice"}, ["choice"], id="missing-from-v3-only"),
    pytest.param("bogus", {"V1Target": "sampler_name"}, ["sampler_name"], id="missing-from-v1"),
    pytest.param("bogus", {"OptOutTarget": "choice"}, [], id="target-opts-out"),
    pytest.param({"__value__": ["euler", "bogus"]}, {"MultiTarget": "choice"}, ["choice"], id="multiselect"),
])
async def test_linked_primitive_combo_checked_against_each_target(value, targets, bad_inputs):
    prompt = {"src": {"class_type": "PrimitiveCombo", "inputs": {"value": value}}}
    for class_type, input_name in targets.items():
        prompt[class_type] = {"class_type": class_type, "inputs": {input_name: ["src", 0]}}

    valid, _, _, errors = await execution.validate_prompt("combo", prompt, None)

    assert valid is not bad_inputs
    if not bad_inputs:
        assert errors == {}
        return
    assert list(errors) == ["src"]
    src_errors = errors["src"]["errors"]
    assert [e["type"] for e in src_errors] == ["value_not_in_list"] * len(bad_inputs)
    assert [e["extra_info"]["input_name"] for e in src_errors] == ["value"] * len(bad_inputs)
    assert all(f"input '{name}'" in e["details"] for e, name in zip(src_errors, bad_inputs))
