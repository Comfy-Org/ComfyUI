"""Unit tests for execution.validate_prompt.

Stub node classes are registered via the stub_nodes fixture in conftest.py;
no server, GPU or model files are required.
"""
import pytest

import execution
import nodes


@pytest.mark.asyncio
async def test_missing_class_type():
    """A node without a class_type key fails fast with missing_node_type."""
    prompt = {"1": {"inputs": {}, "_meta": {"title": "My Node"}}}

    valid, error, good_outputs, node_errors = await execution.validate_prompt(
        "test", prompt, None
    )

    assert valid is False
    assert good_outputs == []
    assert node_errors == {}
    assert error["type"] == "missing_node_type"
    assert "My Node" in error["message"]


@pytest.mark.asyncio
async def test_unknown_class_type():
    """A class_type that isn't registered fails with missing_node_type."""
    prompt = {"1": {"class_type": "VPDoesNotExist", "inputs": {}}}

    valid, error, good_outputs, node_errors = await execution.validate_prompt(
        "test", prompt, None
    )

    assert valid is False
    assert good_outputs == []
    assert node_errors == {}
    assert error["type"] == "missing_node_type"
    assert "VPDoesNotExist" in error["message"]


@pytest.mark.asyncio
async def test_prompt_no_outputs(stub_nodes):
    """A prompt with no OUTPUT_NODE nodes fails with prompt_no_outputs."""
    stub_nodes("VPNoOutput", input_types={"required": {}})
    prompt = {"1": {"class_type": "VPNoOutput", "inputs": {}}}

    valid, error, good_outputs, node_errors = await execution.validate_prompt(
        "test", prompt, None
    )

    assert valid is False
    assert error["type"] == "prompt_no_outputs"
    assert good_outputs == []
    assert node_errors == {}


@pytest.mark.asyncio
async def test_partial_execution_list_filters_outputs(stub_nodes):
    """partial_execution_list restricts good_outputs to the listed node(s)."""
    stub_nodes("VPOutputA", input_types={"required": {}}, output_node=True)
    stub_nodes("VPOutputB", input_types={"required": {}}, output_node=True)
    prompt = {
        "1": {"class_type": "VPOutputA", "inputs": {}},
        "2": {"class_type": "VPOutputB", "inputs": {}},
    }

    valid, error, good_outputs, node_errors = await execution.validate_prompt(
        "test", prompt, ["1"]
    )

    assert valid is True
    assert good_outputs == ["1"]


@pytest.mark.asyncio
async def test_partial_execution_list_none_takes_all(stub_nodes):
    """partial_execution_list=None treats every OUTPUT_NODE as an output."""
    stub_nodes("VPOutputA", input_types={"required": {}}, output_node=True)
    stub_nodes("VPOutputB", input_types={"required": {}}, output_node=True)
    prompt = {
        "1": {"class_type": "VPOutputA", "inputs": {}},
        "2": {"class_type": "VPOutputB", "inputs": {}},
    }

    valid, error, good_outputs, node_errors = await execution.validate_prompt(
        "test", prompt, None
    )

    assert valid is True
    assert sorted(good_outputs) == ["1", "2"]


@pytest.mark.asyncio
async def test_all_outputs_invalid(stub_nodes):
    """The sole output fails validation, so the whole prompt is invalid."""
    stub_nodes(
        "VPRequiresInt",
        input_types={"required": {"amount": ("INT", {})}},
        output_node=True,
    )
    prompt = {"1": {"class_type": "VPRequiresInt", "inputs": {}}}

    valid, error, good_outputs, node_errors = await execution.validate_prompt(
        "test", prompt, None
    )

    assert valid is False
    assert error["type"] == "prompt_outputs_failed_validation"
    assert node_errors["1"]["class_type"] == "VPRequiresInt"
    assert node_errors["1"]["dependent_outputs"] == ["1"]
    assert node_errors["1"]["errors"][0]["type"] == "required_input_missing"


@pytest.mark.asyncio
async def test_mixed_valid_and_invalid_outputs(stub_nodes):
    """One output validates fine while a sibling output fails."""
    stub_nodes("VPOutputValid", input_types={"required": {}}, output_node=True)
    stub_nodes(
        "VPOutputInvalid",
        input_types={"required": {"amount": ("INT", {})}},
        output_node=True,
    )
    prompt = {
        "1": {"class_type": "VPOutputValid", "inputs": {}},
        "2": {"class_type": "VPOutputInvalid", "inputs": {}},
    }

    valid, error, good_outputs, node_errors = await execution.validate_prompt(
        "test", prompt, None
    )

    assert valid is True
    assert good_outputs == ["1"]
    assert "2" in node_errors


@pytest.mark.asyncio
async def test_exception_during_validation(monkeypatch):
    """An exception raised inside INPUT_TYPES is caught and reported."""

    class _ExplodingOutput:
        """Stub whose INPUT_TYPES raises during validation recursion."""

        RETURN_TYPES = ()
        FUNCTION = "run"
        CATEGORY = "_test"
        OUTPUT_NODE = True

        @classmethod
        def INPUT_TYPES(cls):
            raise RuntimeError("boom")

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "VPExploding", _ExplodingOutput)
    prompt = {"1": {"class_type": "VPExploding", "inputs": {}}}

    valid, error, good_outputs, node_errors = await execution.validate_prompt(
        "test", prompt, None
    )

    assert valid is False
    assert error["type"] == "prompt_outputs_failed_validation"
    assert node_errors["1"]["errors"][0]["type"] == "exception_during_validation"


@pytest.mark.asyncio
async def test_downstream_nodes_without_reasons_omitted(stub_nodes):
    """A downstream failure with no reasons of its own is filtered out."""
    stub_nodes(
        "VPUpstreamFails",
        input_types={"required": {"amount": ("INT", {})}},
        return_types=("INT",),
    )
    stub_nodes(
        "VPDownstreamOutput",
        input_types={"required": {"amount": ("INT", {})}},
        output_node=True,
    )
    prompt = {
        "1": {"class_type": "VPUpstreamFails", "inputs": {}},
        "2": {"class_type": "VPDownstreamOutput", "inputs": {"amount": ["1", 0]}},
    }

    valid, error, good_outputs, node_errors = await execution.validate_prompt(
        "test", prompt, None
    )

    assert valid is False
    assert "1" in node_errors
    assert "2" not in node_errors


@pytest.mark.asyncio
async def test_fully_valid_prompt(stub_nodes):
    """A single valid output produces a clean, error-free result."""
    stub_nodes("VPValidOutput", input_types={"required": {}}, output_node=True)
    prompt = {"1": {"class_type": "VPValidOutput", "inputs": {}}}

    valid, error, good_outputs, node_errors = await execution.validate_prompt(
        "test", prompt, None
    )

    assert valid is True
    assert error is None
    assert good_outputs == ["1"]
    assert node_errors == {}
