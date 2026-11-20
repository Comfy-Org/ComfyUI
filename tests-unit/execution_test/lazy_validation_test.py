import asyncio
import dataclasses

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import nodes
from comfy_api.latest import io
from comfy_extras.nodes_logic import BranchNode, SwitchNode
from comfy_extras.nodes_primitive import Boolean
from execution import validate_prompt


def node(class_type, **inputs):
    return {"class_type": class_type, "inputs": inputs}


class Source:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"choice": (["good.safetensors"],)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"


class Sink:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("IMAGE",)}}

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True


class UnmarkedSwitch(SwitchNode):
    @classmethod
    def define_schema(cls):
        return dataclasses.replace(SwitchNode.define_schema(), node_id="UnmarkedSwitch", static_lazy_selection=False)


class HiddenPromptSwitch(SwitchNode):
    @classmethod
    def define_schema(cls):
        return dataclasses.replace(SwitchNode.define_schema(), node_id="HiddenPromptSwitch", hidden=[io.Hidden.prompt])


class AcceptAllSwitch(SwitchNode):
    @classmethod
    def define_schema(cls):
        return dataclasses.replace(SwitchNode.define_schema(), node_id="AcceptAllSwitch", accept_all_inputs=True)


@pytest.fixture(autouse=True)
def test_nodes(monkeypatch):
    for class_type, cls in {
        "Source": Source,
        "Sink": Sink,
        "ComfySwitchNode": SwitchNode,
        "BranchNode": BranchNode,
        "UnmarkedSwitch": UnmarkedSwitch,
        "HiddenPromptSwitch": HiddenPromptSwitch,
        "AcceptAllSwitch": AcceptAllSwitch,
        "PrimitiveBoolean": Boolean,
    }.items():
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, class_type, cls)


def switch_prompt(switch, switch_class="ComfySwitchNode"):
    return {
        "good": node("Source", choice="good.safetensors"),
        "broken": node("Source", choice="missing.safetensors"),
        "switch": node(switch_class, switch=switch, on_true=["good", 0], on_false=["broken", 0]),
        "out": node("Sink", value=["switch", 0]),
    }


def branch_prompt(branch, **links):
    return {
        "good": node("Source", choice="good.safetensors"),
        "broken": node("Source", choice="missing.safetensors"),
        "branch": node(
            "BranchNode",
            branch=branch,
            branch_names={"__value__": ["a", "b"]},
            **{f"autogrow.{key}": value for key, value in links.items()},
        ),
        "out": node("Sink", value=["branch", 0]),
    }


def validate(prompt):
    return asyncio.run(validate_prompt("prompt", prompt, None))


def error_types(node_errors, node_id):
    return [error["type"] for error in node_errors[node_id]["errors"]]


def test_switch_skips_unselected_broken_input():
    valid, error, good_outputs, node_errors = validate(switch_prompt(True))

    assert valid, node_errors
    assert good_outputs == ["out"]
    assert node_errors == {}


def test_switch_validates_selected_broken_input():
    valid, _, _, node_errors = validate(switch_prompt(False))

    assert not valid
    assert error_types(node_errors, "broken") == ["value_not_in_list"]


def test_switch_selector_literal_is_coerced():
    valid, _, _, node_errors = validate(switch_prompt(1))

    assert valid, node_errors


def test_branch_skips_unselected_broken_input():
    valid, _, _, node_errors = validate(branch_prompt("a", branch0=["good", 0], branch1=["broken", 0]))

    assert valid, node_errors


def test_branch_validates_selected_broken_input():
    valid, _, _, node_errors = validate(branch_prompt("b", branch0=["good", 0], branch1=["broken", 0]))

    assert not valid
    assert error_types(node_errors, "broken") == ["value_not_in_list"]


def test_branch_reports_unconnected_selection_as_validation_error():
    valid, _, _, node_errors = validate(branch_prompt("b", branch0=["good", 0]))

    assert not valid
    assert error_types(node_errors, "branch") == ["lazy_selection_failed"]
    assert "not connected" in node_errors["branch"]["errors"][0]["details"]


def test_switch_with_linked_selector_validates_both_inputs():
    prompt = switch_prompt(["selector", 0])
    prompt["selector"] = node("PrimitiveBoolean", value=True)

    valid, _, _, node_errors = validate(prompt)

    assert not valid
    assert error_types(node_errors, "broken") == ["value_not_in_list"]


@pytest.mark.parametrize("switch_class", ["UnmarkedSwitch", "HiddenPromptSwitch", "AcceptAllSwitch"])
def test_lazy_node_without_static_selection_validates_both_inputs(switch_class):
    valid, _, _, node_errors = validate(switch_prompt(True, switch_class=switch_class))

    assert not valid
    assert error_types(node_errors, "broken") == ["value_not_in_list"]
