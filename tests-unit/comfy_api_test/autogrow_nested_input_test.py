import pytest

from comfy_api.latest._io import Autogrow, Image, create_input_dict_v1, get_finalized_class_inputs


def _autogrow_class_inputs():
    autogrow_input = Autogrow.Input(
        "images",
        template=Autogrow.TemplateNames(
            Image.Input("image"),
            names=["image_1", "image_2"],
            min=0,
        ),
    )
    return create_input_dict_v1([autogrow_input])


def test_autogrow_rejects_nested_dict_on_group_key():
    class_inputs = _autogrow_class_inputs()

    with pytest.raises(ValueError, match=r"Input 'images' is an Autogrow group.*dotted key"):
        get_finalized_class_inputs(class_inputs, {"images": {"image_1": ["14", 0]}})


def test_autogrow_accepts_dotted_sub_slot_key():
    class_inputs = _autogrow_class_inputs()

    _, _, v3_data = get_finalized_class_inputs(class_inputs, {"images.image_1": ["14", 0]})

    assert v3_data["dynamic_paths"] == {"images.image_1": "images.image_1"}
    assert v3_data.get("dynamic_paths_default_value", {}) == {}


def test_autogrow_defaults_to_empty_dict_when_nothing_provided():
    class_inputs = _autogrow_class_inputs()

    _, _, v3_data = get_finalized_class_inputs(class_inputs, {})

    assert v3_data["dynamic_paths"] == {"images": "images"}
    assert v3_data["dynamic_paths_default_value"] == {"images": "empty_dict"}
