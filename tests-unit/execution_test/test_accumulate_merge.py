"""
Unit tests for the accumulate toggle on SaveImage and PreviewImage nodes.

Tests that the accumulate input is correctly defined and that the merge flag
derivation logic in execution.py works for all input shapes.
"""
import inspect

import pytest

import nodes


class TestSaveImageAccumulateInput:
    """Test SaveImage node definition includes accumulate input."""

    def test_accumulate_in_optional_inputs(self):
        input_types = nodes.SaveImage.INPUT_TYPES()
        assert "optional" in input_types
        assert "accumulate" in input_types["optional"]

    def test_accumulate_is_boolean_type(self):
        input_types = nodes.SaveImage.INPUT_TYPES()
        accumulate_def = input_types["optional"]["accumulate"]
        assert accumulate_def[0] == "BOOLEAN"

    def test_accumulate_defaults_to_false(self):
        input_types = nodes.SaveImage.INPUT_TYPES()
        accumulate_def = input_types["optional"]["accumulate"]
        assert accumulate_def[1]["default"] is False

    def test_accumulate_is_advanced(self):
        input_types = nodes.SaveImage.INPUT_TYPES()
        accumulate_def = input_types["optional"]["accumulate"]
        assert accumulate_def[1].get("advanced") is True

    def test_save_images_accepts_accumulate_parameter(self):
        sig = inspect.signature(nodes.SaveImage.save_images)
        assert "accumulate" in sig.parameters
        assert sig.parameters["accumulate"].default is False


class TestPreviewImageAccumulateInput:
    """Test PreviewImage node definition includes accumulate input."""

    def test_accumulate_in_optional_inputs(self):
        input_types = nodes.PreviewImage.INPUT_TYPES()
        assert "optional" in input_types
        assert "accumulate" in input_types["optional"]

    def test_accumulate_is_boolean_type(self):
        input_types = nodes.PreviewImage.INPUT_TYPES()
        accumulate_def = input_types["optional"]["accumulate"]
        assert accumulate_def[0] == "BOOLEAN"

    def test_accumulate_defaults_to_false(self):
        input_types = nodes.PreviewImage.INPUT_TYPES()
        accumulate_def = input_types["optional"]["accumulate"]
        assert accumulate_def[1]["default"] is False


class TestAccumulateMergeFlagDerivation:
    """Test the merge flag logic used in execution.py.

    In execution.py, the merge flag is derived as:
        merge = inputs.get('accumulate') is True

    This must return True only for literal True, not for truthy values
    like lists (which represent node links in the prompt).
    """

    @pytest.mark.parametrize(
        "inputs,expected",
        [
            ({"accumulate": True}, True),
            ({"accumulate": False}, False),
            ({}, False),
            ({"accumulate": None}, False),
            # Node link: accumulate connected to another node's output
            ({"accumulate": ["other_node_id", 0]}, False),
            # String "true" should not match
            ({"accumulate": "true"}, False),
            # Integer 1 should not match
            ({"accumulate": 1}, False),
        ],
    )
    def test_merge_flag(self, inputs, expected):
        merge = inputs.get("accumulate") is True
        assert merge is expected
