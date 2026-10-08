import asyncio

import execution
import folder_paths
import nodes
import pytest


@pytest.fixture
def empty_input_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(tmp_path))
    assert not any(tmp_path.iterdir())
    return tmp_path


def test_load_image_input_is_optional_and_uploadable(empty_input_directory):
    inputs = nodes.LoadImage.INPUT_TYPES()

    assert "image" not in inputs.get("required", {})
    assert "image" in inputs["optional"]
    choices, metadata = inputs["optional"]["image"]
    assert choices == []
    assert metadata["image_upload"] is True


def test_load_image_mask_keeps_image_optional_and_channel_required(
    empty_input_directory,
):
    inputs = nodes.LoadImageMask.INPUT_TYPES()

    assert "image" not in inputs.get("required", {})
    assert "image" in inputs["optional"]
    assert inputs["optional"]["image"][1]["image_upload"] is True
    assert "channel" in inputs["required"]


@pytest.mark.parametrize("inputs", [{}, {"image": ""}])
def test_queue_validation_rejects_missing_image_with_actionable_error(
    empty_input_directory, inputs
):
    prompt = {
        "1": {"class_type": "LoadImage", "inputs": inputs},
        "2": {
            "class_type": "PreviewImage",
            "inputs": {"images": ["1", 0], "filename_prefix": "test"},
        },
    }

    valid, error, good_outputs, node_errors = asyncio.run(
        execution.validate_prompt("load-image-test", prompt, None)
    )

    assert valid is False
    assert good_outputs == []
    custom_errors = [
        item
        for item in node_errors["1"]["errors"]
        if item["type"] == "custom_validation_failed"
    ]
    assert custom_errors
    details = " ".join(item["details"] for item in custom_errors).casefold()
    assert "image" in details
    assert any(
        action in details for action in ("upload", "select", "choose", "provide")
    )


def test_load_image_mask_queue_validation_rejects_missing_image(empty_input_directory):
    prompt = {"1": {"class_type": "LoadImageMask", "inputs": {"channel": "alpha"}}}

    valid, errors, _ = asyncio.run(
        execution.validate_inputs("load-image-mask-test", prompt, "1", {})
    )

    assert valid is False
    custom_errors = [
        error for error in errors if error["type"] == "custom_validation_failed"
    ]
    assert custom_errors
    details = " ".join(error["details"] for error in custom_errors).casefold()
    assert "image" in details
    assert any(
        action in details for action in ("upload", "select", "choose", "provide")
    )


def test_queue_validation_accepts_an_existing_image(monkeypatch, empty_input_directory):
    image_path = empty_input_directory / "image.png"
    image_path.touch()
    monkeypatch.setattr(
        folder_paths, "filter_files_content_types", lambda files, _: files
    )
    monkeypatch.setattr(
        folder_paths, "exists_annotated_filepath", lambda path: path == "image.png"
    )
    prompt = {"1": {"class_type": "LoadImage", "inputs": {"image": "image.png"}}}

    valid, errors, _ = asyncio.run(
        execution.validate_inputs("load-image-test", prompt, "1", {})
    )

    assert valid is True
    assert errors == []
