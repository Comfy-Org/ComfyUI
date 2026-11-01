"""Regression test for a DynamicCombo whose saved value no longer matches any option.

A node's DynamicCombo option keys can be renamed between versions (e.g.
Comfy-Org/ComfyUI#16154 renamed BlockSparseAttention's "method" options to lower-case
ids). A workflow saved before such a rename still submits the old value. Previously,
DynamicCombo._expand_schema_for_dynamic silently did nothing when the live value
didn't match any current option, which dropped the input out of the resolved schema
entirely. That surfaced much later, deep in node execution, as a confusing
"missing 1 required positional argument" TypeError instead of a clear, catchable
error during prompt validation (see Comfy-Org/ComfyUI#16236).
"""

import pytest

from comfy_api.latest import _io
from comfy_extras.nodes_logic import DCTestNode


def test_dynamic_combo_stale_value_raises_clear_error():
    node_inputs = {"combo": "no_longer_an_option", "string": "hello world"}

    with pytest.raises(ValueError, match="no_longer_an_option"):
        _io.get_finalized_class_inputs(DCTestNode.INPUT_TYPES(), node_inputs)


def test_dynamic_combo_valid_value_still_resolves():
    node_inputs = {"combo": "option1", "string": "hello world"}

    valid_inputs, _hidden, _v3_data = _io.get_finalized_class_inputs(DCTestNode.INPUT_TYPES(), node_inputs)

    # option1's own nested input ("combo.string") must be expanded into the schema
    # too, not just the combo selector itself.
    assert set(valid_inputs["required"]) == {"combo", "combo.string"}
