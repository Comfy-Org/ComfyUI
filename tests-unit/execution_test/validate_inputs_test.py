"""Unit tests for execution.validate_inputs.

Stub node classes are registered via the stub_nodes fixture in conftest.py;
no server, GPU or model files are required.
"""
import pytest

import execution
import nodes


@pytest.mark.asyncio
async def test_required_input_missing(stub_nodes):
    """A required input absent from the prompt's inputs dict is reported."""
    stub_nodes(
        "VIRequiredMissing",
        input_types={"required": {"value": ("INT", {"min": 0, "max": 10})}},
    )
    prompt = {"1": {"class_type": "VIRequiredMissing", "inputs": {}}}

    valid, errors, node_id = await execution.validate_inputs("test", prompt, "1", {})

    assert valid is False
    assert [e["type"] for e in errors] == ["required_input_missing"]
    assert node_id == "1"


@pytest.mark.asyncio
async def test_optional_input_absent_is_valid(stub_nodes):
    """An optional input absent from the prompt's inputs dict is not an error."""
    stub_nodes(
        "VIOptionalAbsent",
        input_types={"optional": {"value": ("INT", {})}},
    )
    prompt = {"1": {"class_type": "VIOptionalAbsent", "inputs": {}}}

    valid, errors, node_id = await execution.validate_inputs("test", prompt, "1", {})

    assert valid is True
    assert errors == []


@pytest.mark.asyncio
async def test_bad_linked_input_wrong_length(stub_nodes):
    """A linked input given as a list that isn't length-2 is malformed."""
    stub_nodes(
        "VIBadLink",
        input_types={"required": {"value": ("INT", {})}},
    )
    prompt = {"1": {"class_type": "VIBadLink", "inputs": {"value": ["1"]}}}

    valid, errors, node_id = await execution.validate_inputs("test", prompt, "1", {})

    assert valid is False
    assert [e["type"] for e in errors] == ["bad_linked_input"]


@pytest.mark.asyncio
async def test_return_type_mismatch(stub_nodes):
    """A linked input whose source RETURN_TYPES doesn't match is rejected."""
    stub_nodes("VISource4", input_types={"required": {}}, return_types=("STRING",))
    stub_nodes("VISink4", input_types={"required": {"value": ("INT", {})}})
    prompt = {
        "1": {"class_type": "VISource4", "inputs": {}},
        "2": {"class_type": "VISink4", "inputs": {"value": ["1", 0]}},
    }

    valid, errors, node_id = await execution.validate_inputs("test", prompt, "2", {})

    assert valid is False
    assert [e["type"] for e in errors] == ["return_type_mismatch"]


@pytest.mark.asyncio
async def test_valid_linked_input(stub_nodes):
    """A linked input whose source RETURN_TYPES matches is accepted."""
    stub_nodes("VISource5", input_types={"required": {}}, return_types=("INT",))
    stub_nodes("VISink5", input_types={"required": {"value": ("INT", {})}})
    prompt = {
        "1": {"class_type": "VISource5", "inputs": {}},
        "2": {"class_type": "VISink5", "inputs": {"value": ["1", 0]}},
    }

    valid, errors, node_id = await execution.validate_inputs("test", prompt, "2", {})

    assert valid is True
    assert errors == []


@pytest.mark.asyncio
async def test_dependency_cycle(stub_nodes):
    """Two nodes whose inputs link to each other are flagged as a cycle."""
    stub_nodes(
        "VICycleA",
        input_types={"required": {"value": ("INT", {})}},
        return_types=("INT",),
    )
    stub_nodes(
        "VICycleB",
        input_types={"required": {"value": ("INT", {})}},
        return_types=("INT",),
    )
    prompt = {
        "1": {"class_type": "VICycleA", "inputs": {"value": ["2", 0]}},
        "2": {"class_type": "VICycleB", "inputs": {"value": ["1", 0]}},
    }
    validated = {}

    valid, errors, node_id = await execution.validate_inputs("test", prompt, "1", validated)

    assert valid is False
    cycle_errors = [e for e in errors if e["type"] == "dependency_cycle"]
    assert len(cycle_errors) == 1
    assert set(cycle_errors[0]["extra_info"]["cycle_nodes"]) == {"1", "2"}
    assert "1" in validated
    assert "2" in validated


@pytest.mark.asyncio
async def test_int_coercion(stub_nodes):
    """A string value for an INT input is coerced in place."""
    stub_nodes(
        "VIIntCoerce",
        input_types={"required": {"value": ("INT", {})}},
    )
    prompt = {"1": {"class_type": "VIIntCoerce", "inputs": {"value": "5"}}}

    valid, errors, node_id = await execution.validate_inputs("test", prompt, "1", {})

    assert valid is True
    assert prompt["1"]["inputs"]["value"] == 5


@pytest.mark.asyncio
async def test_float_string_bool_coercion(stub_nodes):
    """FLOAT, STRING and BOOLEAN inputs are each coerced in place."""
    stub_nodes(
        "VIMixedCoerce",
        input_types={
            "required": {
                "f": ("FLOAT", {}),
                "s": ("STRING", {}),
                "b": ("BOOLEAN", {}),
            }
        },
    )
    prompt = {
        "1": {
            "class_type": "VIMixedCoerce",
            "inputs": {"f": "2.5", "s": 42, "b": 0},
        }
    }

    valid, errors, node_id = await execution.validate_inputs("test", prompt, "1", {})

    assert valid is True
    assert prompt["1"]["inputs"]["f"] == 2.5
    assert prompt["1"]["inputs"]["s"] == "42"
    assert prompt["1"]["inputs"]["b"] is False


@pytest.mark.asyncio
async def test_invalid_input_type(stub_nodes):
    """A value that can't be coerced to the declared INT type is an error."""
    stub_nodes(
        "VIInvalidType",
        input_types={"required": {"value": ("INT", {})}},
    )
    prompt = {"1": {"class_type": "VIInvalidType", "inputs": {"value": "abc"}}}

    valid, errors, node_id = await execution.validate_inputs("test", prompt, "1", {})

    assert valid is False
    assert [e["type"] for e in errors] == ["invalid_input_type"]


@pytest.mark.asyncio
async def test_value_smaller_than_min(stub_nodes):
    """A value below the declared min is rejected."""
    stub_nodes(
        "VIBelowMin",
        input_types={"required": {"value": ("INT", {"min": 0, "max": 10})}},
    )
    prompt = {"1": {"class_type": "VIBelowMin", "inputs": {"value": -1}}}

    valid, errors, node_id = await execution.validate_inputs("test", prompt, "1", {})

    assert valid is False
    assert [e["type"] for e in errors] == ["value_smaller_than_min"]


@pytest.mark.asyncio
async def test_value_bigger_than_max(stub_nodes):
    """A value above the declared max is rejected."""
    stub_nodes(
        "VIAboveMax",
        input_types={"required": {"value": ("INT", {"min": 0, "max": 10})}},
    )
    prompt = {"1": {"class_type": "VIAboveMax", "inputs": {"value": 11}}}

    valid, errors, node_id = await execution.validate_inputs("test", prompt, "1", {})

    assert valid is False
    assert [e["type"] for e in errors] == ["value_bigger_than_max"]


@pytest.mark.asyncio
async def test_combo_value_not_in_list(stub_nodes):
    """A combo value outside the declared option list is rejected."""
    stub_nodes(
        "VIComboInvalid",
        input_types={"required": {"value": (["a", "b"],)}},
    )
    prompt = {"1": {"class_type": "VIComboInvalid", "inputs": {"value": "c"}}}

    valid, errors, node_id = await execution.validate_inputs("test", prompt, "1", {})

    assert valid is False
    assert [e["type"] for e in errors] == ["value_not_in_list"]


@pytest.mark.asyncio
async def test_combo_multiselect_invalid_value(stub_nodes):
    """Multiselect combo: invalid entries produce value_not_in_list; the list
    value must be wrapped in __value__ so it is not parsed as a node link."""
    stub_nodes(
        "VIComboMultiselect",
        input_types={
            "required": {"value": (["a", "b"], {"multiselect": True})}
        },
    )
    prompt = {
        "1": {
            "class_type": "VIComboMultiselect",
            "inputs": {"value": {"__value__": ["a", "c"]}},
        }
    }

    valid, errors, node_id = await execution.validate_inputs("test", prompt, "1", {})

    assert valid is False
    assert [e["type"] for e in errors] == ["value_not_in_list"]
    assert "'c'" in errors[0]["details"]


@pytest.mark.asyncio
async def test_combo_long_list_truncates_config(stub_nodes):
    """A combo with more than 20 options truncates input_config in the error."""
    options = [f"o{i}" for i in range(25)]
    stub_nodes(
        "VIComboLong",
        input_types={"required": {"value": (options,)}},
    )
    prompt = {"1": {"class_type": "VIComboLong", "inputs": {"value": "bad"}}}

    valid, errors, node_id = await execution.validate_inputs("test", prompt, "1", {})

    assert valid is False
    assert [e["type"] for e in errors] == ["value_not_in_list"]
    assert errors[0]["extra_info"]["input_config"] is None
    assert "list of length 25" in errors[0]["details"]


@pytest.mark.asyncio
async def test_custom_validate_failure(stub_nodes):
    """A VALIDATE_INPUTS classmethod returning a string fails validation."""

    def _v(s, value):
        return "three is forbidden" if value == 3 else True

    stub_nodes(
        "VICustomValidateFail",
        input_types={"required": {"value": ("INT", {})}},
        validate=_v,
    )
    prompt = {"1": {"class_type": "VICustomValidateFail", "inputs": {"value": 3}}}

    valid, errors, node_id = await execution.validate_inputs("test", prompt, "1", {})

    assert valid is False
    assert [e["type"] for e in errors] == ["custom_validation_failed"]


@pytest.mark.asyncio
async def test_custom_validate_success(stub_nodes):
    """A VALIDATE_INPUTS classmethod returning True passes validation."""

    def _v(s, value):
        return "three is forbidden" if value == 3 else True

    stub_nodes(
        "VICustomValidateOk",
        input_types={"required": {"value": ("INT", {})}},
        validate=_v,
    )
    prompt = {"1": {"class_type": "VICustomValidateOk", "inputs": {"value": 4}}}

    valid, errors, node_id = await execution.validate_inputs("test", prompt, "1", {})

    assert valid is True
    assert errors == []


@pytest.mark.asyncio
async def test_custom_validate_skips_min_max(stub_nodes):
    """An input named in VALIDATE_INPUTS' signature skips the min/max check."""

    def _v(s, value):
        return True

    stub_nodes(
        "VICustomValidateSkipsMinMax",
        input_types={"required": {"value": ("INT", {"min": 0})}},
        validate=_v,
    )
    prompt = {
        "1": {"class_type": "VICustomValidateSkipsMinMax", "inputs": {"value": -5}}
    }

    valid, errors, node_id = await execution.validate_inputs("test", prompt, "1", {})

    assert valid is True
    assert errors == []


@pytest.mark.asyncio
async def test_value_wrapper_dict_unwrapped(stub_nodes):
    """A {"__value__": ...} wrapper is unwrapped before type coercion."""
    stub_nodes(
        "VIValueWrapper",
        input_types={"required": {"value": ("INT", {})}},
    )
    prompt = {
        "1": {"class_type": "VIValueWrapper", "inputs": {"value": {"__value__": 7}}}
    }

    valid, errors, node_id = await execution.validate_inputs("test", prompt, "1", {})

    assert valid is True
    assert prompt["1"]["inputs"]["value"] == 7


@pytest.mark.asyncio
async def test_exception_during_inner_validation(stub_nodes, monkeypatch):
    """An exception raised while validating a linked node is captured."""

    class _ExplodingNode:
        """Stub whose INPUT_TYPES raises during validation recursion."""

        RETURN_TYPES = ("INT",)
        FUNCTION = "run"
        CATEGORY = "_test"
        OUTPUT_NODE = False

        @classmethod
        def INPUT_TYPES(cls):
            raise RuntimeError("boom")

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "VIExploding", _ExplodingNode)
    stub_nodes(
        "VIExplodingSink",
        input_types={"required": {"value": ("INT", {})}},
    )
    prompt = {
        "1": {"class_type": "VIExploding", "inputs": {}},
        "2": {"class_type": "VIExplodingSink", "inputs": {"value": ["1", 0]}},
    }
    validated = {}

    valid, errors, node_id = await execution.validate_inputs("test", prompt, "2", validated)

    assert valid is False
    assert [e["type"] for e in validated["1"][1]] == ["exception_during_inner_validation"]


@pytest.mark.asyncio
async def test_validated_cache_short_circuits(stub_nodes):
    """A pre-populated validated entry short-circuits validation entirely."""
    stub_nodes(
        "VICacheHit",
        input_types={"required": {"value": ("INT", {"min": 0, "max": 10})}},
    )
    prompt = {"1": {"class_type": "VICacheHit", "inputs": {}}}
    cached = (False, [], "1")
    validated = {"1": cached}

    result = await execution.validate_inputs("test", prompt, "1", validated)

    assert result == cached
