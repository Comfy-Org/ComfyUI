from comfy_api.latest._io import Combo, MultiCombo


def test_multicombo_serializes_multi_select_as_object():
    multi_combo = MultiCombo.Input(
        id="providers",
        options=["a", "b", "c"],
        default=["a"],
    )

    serialized = multi_combo.as_dict()

    assert serialized["multiselect"] is True
    assert "multi_select" in serialized
    assert serialized["multi_select"] == {}


def test_multicombo_serializes_multi_select_with_placeholder_and_chip():
    multi_combo = MultiCombo.Input(
        id="providers",
        options=["a", "b", "c"],
        default=["a"],
        placeholder="Select providers",
        chip=True,
    )

    serialized = multi_combo.as_dict()

    assert serialized["multiselect"] is True
    assert serialized["multi_select"] == {
        "placeholder": "Select providers",
        "chip": True,
    }


def test_combo_does_not_serialize_multiselect():
    """Regular Combo should not have multiselect in its serialized output."""
    combo = Combo.Input(
        id="choice",
        options=["a", "b", "c"],
    )

    serialized = combo.as_dict()

    # Combo sets multiselect=False, but prune_dict keeps False (not None),
    # so it should be present but False
    assert serialized.get("multiselect") is False
    assert "multi_select" not in serialized


def _validate_combo_values(val, combo_options, is_multiselect):
    """Reproduce the validation logic from execution.py for testing."""
    if is_multiselect and isinstance(val, list):
        return [v for v in val if v not in combo_options]
    else:
        return [val] if val not in combo_options else []


def test_multicombo_validation_accepts_valid_list():
    options = ["a", "b", "c"]
    assert _validate_combo_values(["a", "b"], options, True) == []


def test_multicombo_validation_rejects_invalid_values():
    options = ["a", "b", "c"]
    assert _validate_combo_values(["a", "x"], options, True) == ["x"]


def test_multicombo_validation_accepts_empty_list():
    options = ["a", "b", "c"]
    assert _validate_combo_values([], options, True) == []


def test_combo_validation_rejects_list_even_with_valid_items():
    """A regular Combo should not accept a list value."""
    options = ["a", "b", "c"]
    invalid = _validate_combo_values(["a", "b"], options, False)
    assert len(invalid) > 0
