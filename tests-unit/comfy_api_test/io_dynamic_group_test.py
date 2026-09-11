import pytest

from comfy_api.latest import io
from comfy_api.latest._io import build_nested_inputs, create_input_dict_v1, get_finalized_class_inputs


def _reconstruct(group, values, *, lazy=False):
    _, _, v3_data = get_finalized_class_inputs(create_input_dict_v1([group]), values)
    v3_data["create_dynamic_tuple"] = lazy
    return build_nested_inputs(values, v3_data)


def test_serializes_one_template_with_field_requirements():
    group = io.DynamicGroup.Input(
        "rows",
        template=[io.String.Input("name"), io.Float.Input("weight", default=1.0, optional=True)],
        min=0, max=5, group_name="Item",
    )
    schema = create_input_dict_v1([group])
    assert schema == {"required": {"rows": ("COMFY_DYNAMICGROUP_V3", {
        "template": {
            "required": {"name": ("STRING", {"multiline": False})},
            "optional": {"weight": ("FLOAT", {"default": 1.0})},
        },
        "min": 0, "max": 5, "group_name": "Item",
    })}}


@pytest.mark.parametrize("group_id,template,limits", [
    ("rows", [], {}),
    ("rows", [io.Float.Input("x"), io.Float.Input("x")], {}),
    ("rows.bad", [io.Float.Input("x")], {}),
    ("rows", [io.Float.Input("x.bad")], {}),
    ("rows", [io.Image.Input("image")], {}),
    ("rows", [io.Float.Input("x", force_input=True)], {}),
    ("rows", [io.DynamicGroup.Input("nested", template=[io.Float.Input("x")])], {}),
    ("rows", [io.Float.Input("x")], {"min": -1}),
    ("rows", [io.Float.Input("x")], {"min": 2, "max": 1}),
    ("rows", [io.Float.Input("x")], {"max": 0}),
    ("rows", [io.Float.Input("x")], {"max": 101}),
])
def test_rejects_invalid_template_or_limits(group_id, template, limits):
    with pytest.raises(AssertionError):
        io.DynamicGroup.Input(group_id, template=template, **limits)


@pytest.mark.parametrize("lazy", [False, True])
def test_empty_group_is_an_empty_list(lazy):
    group = io.DynamicGroup.Input("rows", template=[io.Float.Input("x", default=1.0)], min=0)
    assert _reconstruct(group, {}, lazy=lazy) == {"rows": []}


@pytest.mark.parametrize("minimum", [0, 1, 2])
@pytest.mark.parametrize("optional_group", [False, True])
def test_every_submitted_row_keeps_template_requirements(minimum, optional_group):
    group = io.DynamicGroup.Input("rows", template=[
        io.String.Input("name"),
        io.Float.Input("weight", default=1.0),
        io.Boolean.Input("enabled", optional=True),
    ], min=minimum, optional=optional_group)
    values = {"rows.0.name": "A", "rows.0.weight": 0.8, "rows.2.name": "C"}
    schema, _, _ = get_finalized_class_inputs(create_input_dict_v1([group]), values)
    assert set(schema["required"]) == {
        "rows.0.name", "rows.0.weight", "rows.2.name", "rows.2.weight",
    }
    assert set(schema["optional"]) == {"rows.0.enabled", "rows.2.enabled"}


@pytest.mark.parametrize("optional_group", [False, True])
@pytest.mark.parametrize("values", [{}, {"rows.0.x": 1.0}, {"rows.2.x": 1.0}])
def test_min_counts_submitted_rows_without_padding(optional_group, values):
    group = io.DynamicGroup.Input("rows", template=[io.Float.Input("x", optional=True)], min=2, optional=optional_group)
    with pytest.raises(ValueError, match="expected between 2 and"):
        _reconstruct(group, values)


def test_sparse_rows_preserve_positions_without_defaults():
    group = io.DynamicGroup.Input("rows", template=[
        io.String.Input("name"), io.Float.Input("weight", default=1.0, optional=True),
    ], min=2, max=2)
    values = {"rows.2.name": "C", "rows.2.weight": 0.5, "rows.0.name": "A"}
    assert _reconstruct(group, values) == {"rows": [
        {"name": "A", "weight": None},
        {"name": None, "weight": None},
        {"name": "C", "weight": 0.5},
    ]}
    assert values == {"rows.2.name": "C", "rows.2.weight": 0.5, "rows.0.name": "A"}


def test_max_counts_rows_not_fields():
    group = io.DynamicGroup.Input("rows", template=[io.String.Input("name"), io.Float.Input("weight")], max=1)
    assert _reconstruct(group, {"rows.0.name": "A", "rows.0.weight": 0.8}) == {
        "rows": [{"name": "A", "weight": 0.8}],
    }
    with pytest.raises(ValueError, match="received 2 rows; expected between 0 and 1"):
        _reconstruct(group, {"rows.0.name": "A", "rows.2.name": "C"})


@pytest.mark.parametrize("key", [
    "rows.foo.x", "rows.-1.x", "rows.01.x", "rows.+1.x", "rows.١.x",
    "rows.0", "rows..x", "rows.0.unknown", "rows.0.x.extra",
])
def test_rejects_malformed_row_keys(key):
    group = io.DynamicGroup.Input("rows", template=[io.Float.Input("x")])
    with pytest.raises(ValueError) as error:
        _reconstruct(group, {key: 1.0})
    assert key in str(error.value)


def test_largest_supported_index_preserves_position():
    group = io.DynamicGroup.Input("rows", template=[io.Float.Input("x")], max=1)
    rows = _reconstruct(group, {"rows.99.x": 0.5})["rows"]
    assert rows == [{"x": None}] * 99 + [{"x": 0.5}]


@pytest.mark.parametrize("index", [100, 1_000_000])
def test_rejects_out_of_range_index_before_registering_padding(index):
    group = io.DynamicGroup.Input("rows", template=[io.Float.Input("x")], max=1)
    expanded = {"required": {}, "optional": {}, "dynamic_paths": {}, "dynamic_paths_default_value": {}, "list_paths": set()}
    with pytest.raises(ValueError, match="exceeds the index limit of 99"):
        io.DynamicGroup._expand_schema_for_dynamic(
            expanded, {f"rows.{index}.x": 0.5}, (group.io_type, group.as_dict()), "required", ["rows"],
        )
    assert expanded["dynamic_paths"] == {}


def test_lazy_rows_keep_original_field_keys_and_positions():
    group = io.DynamicGroup.Input("rows", template=[io.Float.Input("x")], max=2)
    assert _reconstruct(group, {"rows.2.x": 0.5, "rows.0.x": 0.8}, lazy=True) == {"rows": [
        {"x": (0.8, "rows.0.x")},
        {"x": (None, "rows.1.x")},
        {"x": (0.5, "rows.2.x")},
    ]}


@pytest.mark.parametrize("lazy", [False, True])
def test_group_inside_dynamic_combo_preserves_other_inputs(lazy):
    group = io.DynamicGroup.Input("rows", template=[io.Float.Input("x")])
    combo = io.DynamicCombo.Input("mode", options=[io.DynamicCombo.Option("on", [group])])
    values = {"mode": "on", "mode.rows.0.x": 0.8, "fixed": "untouched"}
    assert _reconstruct(combo, values, lazy=lazy) == {
        "mode": {
            "mode": ("on", "mode") if lazy else "on",
            "rows": [{"x": (0.8, "mode.rows.0.x") if lazy else 0.8}],
        },
        "fixed": "untouched",
    }


@pytest.mark.parametrize("lazy", [False, True])
def test_autogrow_empty_value_is_unchanged(lazy):
    group = io.Autogrow.Input("items", template=io.Autogrow.TemplatePrefix(io.Float.Input("x"), prefix="item", min=0))
    assert _reconstruct(group, {}, lazy=lazy) == {"items": ({}, "items") if lazy else {}}
