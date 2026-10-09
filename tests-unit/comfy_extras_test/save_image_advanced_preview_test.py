"""SaveImageAdvanced writes a content-addressed preview next to each EXR when assets are on."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from blake3 import blake3
from PIL import Image

import folder_paths
from comfy.cli_args import args
from comfy_execution.preview_generators import linear_to_preview

previous_nodes = sys.modules.get("nodes")
previous_server = sys.modules.get("server")
sys.modules["nodes"] = MagicMock(MAX_RESOLUTION=16384)
sys.modules["server"] = MagicMock()
from comfy_extras import nodes_images  # noqa: E402
from comfy_extras.nodes_images import SaveImageAdvanced  # noqa: E402
for name, previous in (("nodes", previous_nodes), ("server", previous_server)):
    if previous is None:
        sys.modules.pop(name)
    else:
        sys.modules[name] = previous


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    saved = (folder_paths.get_output_directory(), folder_paths.get_previews_directory())
    folder_paths.set_output_directory(str(tmp_path / "output"))
    folder_paths.set_previews_directory(str(tmp_path / "previews"))
    (tmp_path / "output").mkdir()
    monkeypatch.setattr(nodes_images.PromptServer, "instance", SimpleNamespace(asset_manager=SimpleNamespace(enabled=True)), raising=False)
    monkeypatch.setattr(args, "disable_metadata", True)
    monkeypatch.setattr(SaveImageAdvanced, "hidden", SimpleNamespace(prompt=None, extra_pnginfo=None), raising=False)
    yield tmp_path
    folder_paths.set_output_directory(saved[0])
    folder_paths.set_previews_directory(saved[1])


def _save(images: torch.Tensor, colorspace: str = "linear", bit_depth: str = "32-bit float") -> list[dict]:
    fmt = {"format": "exr", "bit_depth": bit_depth, "input_color_space": colorspace}
    return SaveImageAdvanced.execute(images, "t", fmt).ui["images"]


def test_an_rgb_exr_gets_a_jpeg_named_by_its_hash(dirs):
    entry = _save(torch.full((1, 1000, 1200, 3), 0.25))[0]

    ref = entry["asset_preview"]
    data = (dirs / "previews" / ref["filename"]).read_bytes()
    assert ref["filename"] == f"{blake3(data).hexdigest()}.jpg"
    image = Image.open(dirs / "previews" / ref["filename"])
    assert image.format == "JPEG"
    assert (ref["width"], ref["height"]) == image.size
    assert image.width * image.height <= 1_000_000 < 1200 * 1000
    assert (dirs / "output" / entry["filename"]).is_file()


def test_an_rgba_exr_gets_a_webp_with_straight_alpha(dirs):
    rgba = torch.empty((1, 16, 16, 4))
    rgba[..., :3] = 0.1
    rgba[..., 3] = 0.2

    ref = _save(rgba)[0]["asset_preview"]

    image = Image.open(dirs / "previews" / ref["filename"])
    assert ref["filename"].endswith(".webp") and image.mode == "RGBA"
    pixel = np.asarray(image)[8, 8]
    assert abs(int(pixel[0]) - 89) <= 3, "sRGB(0.1), not brightened by un-premultiplying"
    assert abs(int(pixel[3]) - 51) <= 3


@pytest.mark.parametrize("shape", [(1, 8, 8), (1, 8, 8, 1)])
def test_a_gray_exr_gets_an_rgb_preview(dirs, shape):
    ref = _save(torch.full(shape, 0.5))[0]["asset_preview"]

    assert Image.open(dirs / "previews" / ref["filename"]).mode == "RGB"


def test_with_assets_off_no_preview_is_written(dirs, monkeypatch):
    monkeypatch.setattr(nodes_images.PromptServer.instance.asset_manager, "enabled", False)

    entry = _save(torch.full((1, 8, 8, 3), 0.5))[0]

    assert "asset_preview" not in entry
    assert not (dirs / "previews").exists()


def test_a_failing_preview_never_fails_the_save(dirs):
    with patch.object(nodes_images, "linear_to_preview", side_effect=RuntimeError("boom")):
        entry = _save(torch.full((1, 8, 8, 3), 0.5))[0]

    assert "asset_preview" not in entry
    assert (dirs / "output" / entry["filename"]).is_file()


def test_the_previews_directory_is_created_and_identical_saves_share_one_file(dirs):
    first, second = _save(torch.full((2, 8, 8, 3), 0.5))

    assert first["asset_preview"] == second["asset_preview"]
    assert [p.name for p in (dirs / "previews").iterdir()] == [first["asset_preview"]["filename"]]


@pytest.mark.parametrize("colorspace", ["sRGB", "linear", "HDR"])
def test_outputs_and_uploads_tonemap_identically(dirs, colorspace):
    from app.assets.previews import _decode_for_preview

    torch.manual_seed(0)
    image = torch.rand((1, 40, 60, 3)) * 1.2
    with patch.object(nodes_images, "linear_to_preview", wraps=linear_to_preview) as tonemap:
        entry = _save(image, colorspace)[0]

    written = linear_to_preview(tonemap.call_args.args[0])
    uploaded = _decode_for_preview(str(dirs / "output" / entry["filename"]), 1_000_000)
    assert np.array_equal(np.asarray(written), np.asarray(uploaded))


def test_colour_under_transparent_pixels_does_not_reach_the_preview():
    rgba = torch.zeros((1000, 2000, 4))
    rgba[:, ::2, :3] = 0.5
    rgba[:, ::2, 3] = 1.0
    hidden = rgba.clone()
    hidden[:, 1::2, :3] = 1.0

    assert np.array_equal(np.asarray(linear_to_preview(rgba)), np.asarray(linear_to_preview(hidden)))


@pytest.mark.parametrize(("height", "width"), [(2160, 3840), (2, 30000)])
def test_the_downsample_fits_the_pixel_and_side_limits(height, width):
    image = linear_to_preview(torch.full((height, width, 3), 0.5))

    assert image.width * image.height <= 1_000_000
    assert max(image.size) <= 16383


def test_non_finite_values_tonemap_to_black_and_white():
    image = torch.tensor([[[float("nan"), float("inf"), float("-inf")]]])

    assert np.asarray(linear_to_preview(image)).tolist() == [[[0, 255, 0]]]


def test_a_nan_pixel_does_not_blank_its_downsampled_neighbours():
    image = torch.full((4, 4, 3), 0.5)
    image[0, 0] = float("nan")
    zeroed = image.nan_to_num(0.0)

    assert np.array_equal(np.asarray(linear_to_preview(image, 4)), np.asarray(linear_to_preview(zeroed, 4)))
    assert np.asarray(linear_to_preview(image, 4))[0, 0, 0] > 0
