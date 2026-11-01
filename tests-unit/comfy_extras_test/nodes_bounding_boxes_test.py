import os

import pytest
import torch
from PIL import Image

import folder_paths
from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy_extras.nodes_bounding_boxes import CreateBoundingBoxes  # noqa: E402


def run(background=None):
    return CreateBoundingBoxes.execute(width=64, height=64, background=background).ui


def test_echoes_the_first_background_image_as_a_temp_preview(tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path))
    first = torch.zeros(1, 48, 80, 3)
    second = torch.ones(1, 48, 80, 3)

    ui = run(background=torch.cat([first, second]))

    [entry] = ui["background_images"]
    assert entry["type"] == "temp"
    path = os.path.join(str(tmp_path), entry["subfolder"], entry["filename"])
    with Image.open(path) as saved:
        assert saved.size == (80, 48)
        assert saved.getpixel((0, 0)) == (0, 0, 0)


def test_omits_the_background_preview_without_a_background(tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path))

    ui = run()

    assert "background_images" not in ui
    assert ui["dims"] == [64, 64]


@pytest.fixture
def editor_box():
    return {
        "x": 512,
        "y": 512,
        "width": 512,
        "height": 512,
        "metadata": {"_canvas_size": {"width": 1024, "height": 1024}},
    }


@pytest.mark.parametrize(
    "canvas_size,width,height,expected_box,expected_element",
    [
        ({"width": 1024, "height": 1024}, 768, 1368,
         {"x": 384, "y": 684, "width": 384, "height": 684}, [500, 500, 1000, 1000]),
        ({"width": 1024, "height": 1024}, 1368, 768,
         {"x": 684, "y": 384, "width": 684, "height": 384}, [500, 500, 1000, 1000]),
        ({"width": 1024, "height": 1024}, 1024, 1024,
         {"x": 512, "y": 512, "width": 512, "height": 512}, [500, 500, 1000, 1000]),
        ({"width": 2048, "height": 1024}, 768, 1368,
         {"x": 192, "y": 684, "width": 192, "height": 684}, [500, 250, 1000, 500]),
        ({"width": 1024.0, "height": 1024.0}, 768, 1368,
         {"x": 384, "y": 684, "width": 384, "height": 684}, [500, 500, 1000, 1000]),
    ],
    ids=["portrait", "landscape", "same-size", "non-square-canvas", "integral-json-numbers"],
)
def test_editor_boxes_keep_their_canvas_proportions(
    editor_box, canvas_size, width, height, expected_box, expected_element
):
    editor_box["metadata"]["_canvas_size"] = canvas_size

    result = CreateBoundingBoxes.execute(width=width, height=height, editor_state=[editor_box])

    assert result[1] == [[expected_box]]
    assert result[2] == [{"type": "obj", "bbox": expected_element, "desc": ""}]
    assert result.ui["dims"] == [width, height]


@pytest.mark.parametrize(
    "canvas_size",
    [
        None,
        [],
        {},
        {"width": 1024},
        {"height": 1024},
        {"width": None, "height": 1024},
        {"width": 1024, "height": "1024"},
        {"width": True, "height": 1024},
        {"width": 1024, "height": False},
        {"width": 0, "height": 1024},
        {"width": 1024, "height": -1},
        {"width": 1024.5, "height": 1024},
        {"width": 1024, "height": 1024.5},
        {"width": float("inf"), "height": 1024},
        {"width": 1024, "height": float("nan")},
    ],
    ids=[
        "null", "list", "empty", "missing-height", "missing-width", "null-width",
        "string-height", "bool-width", "bool-height", "zero-width", "negative-height",
        "fractional-width", "fractional-height", "infinite-width", "nan-height",
    ],
)
def test_invalid_canvas_sizes_keep_legacy_pixel_coordinates(editor_box, canvas_size):
    editor_box["metadata"]["_canvas_size"] = canvas_size

    result = CreateBoundingBoxes.execute(width=768, height=1368, editor_state=[editor_box])

    assert result[1] == [[{"x": 512, "y": 512, "width": 512, "height": 512}]]
    assert result[2][0]["bbox"] == [374, 667, 749, 1000]


def test_old_editor_payload_keeps_pixel_coordinates_and_metadata(editor_box):
    editor_box["metadata"] = {"type": "text", "text": "caption", "desc": "region", "palette": ["#ff0000"]}

    result = CreateBoundingBoxes.execute(width=768, height=1368, editor_state=[editor_box])

    assert result[1] == [[{"x": 512, "y": 512, "width": 512, "height": 512}]]
    assert result[2] == [{
        "type": "text", "bbox": [374, 667, 749, 1000], "text": "caption",
        "desc": "region", "color_palette": ["#FF0000"],
    }]


@pytest.mark.parametrize(
    "incoming_changed,expected_box",
    [
        (True, {"x": 100, "y": 125, "width": 200, "height": 250}),
        (False, {"x": 384, "y": 684, "width": 384, "height": 684}),
    ],
    ids=["incoming-ignores-canvas-hint", "unchanged-incoming-keeps-editor"],
)
def test_incoming_boxes_keep_their_existing_priority(editor_box, incoming_changed, expected_box):
    incoming = {
        "x": 100, "y": 125, "width": 200, "height": 250,
        "metadata": {"_canvas_size": {"width": 2048, "height": 2048}},
    }
    last_incoming = [] if incoming_changed else [dict(incoming)]

    result = CreateBoundingBoxes.execute(
        width=768, height=1368, editor_state=[editor_box], bboxes=[incoming], last_incoming=last_incoming
    )

    assert result[1] == [[expected_box]]


def test_canvas_hint_keeps_background_preview_and_runtime_output_dimensions(tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path))
    editor_box = {
        "x": 64, "y": 64, "width": 128, "height": 64,
        "metadata": {"_canvas_size": {"width": 256, "height": 128}},
    }

    result = CreateBoundingBoxes.execute(
        width=768, height=1368, editor_state=[editor_box], background=torch.zeros(1, 128, 256, 3)
    )

    assert result[0].shape == (1, 128, 256, 3)
    assert result[1] == [[{"x": 192, "y": 684, "width": 384, "height": 684}]]
    assert result[2][0]["bbox"] == [500, 250, 1000, 750]
    assert result.ui["dims"] == [768, 1368]
    [entry] = result.ui["background_images"]
    with Image.open(tmp_path / entry["subfolder"] / entry["filename"]) as saved:
        assert saved.size == (256, 128)
