import os

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
