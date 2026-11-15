"""SaveImageAdvanced writes a content-addressed preview next to each EXR when assets are on."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

import folder_paths
from comfy.cli_args import args
from comfy_execution.preview_tonemap import linear_to_preview

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


def test_an_rgba_exr_gets_a_webp_with_straight_alpha_within_one_megapixel(dirs):
    rgba = torch.empty((1, 1000, 1200, 4))
    rgba[..., :3] = 0.1
    rgba[..., 3] = 0.2

    ref = _save(rgba)[0]["asset_preview"]

    image = Image.open(dirs / "previews" / ref["filename"])
    assert ref["filename"].endswith(".webp") and image.mode == "RGBA"
    assert (ref["width"], ref["height"]) == image.size and image.width * image.height <= 1_000_000 < 1200 * 1000
    pixel = np.asarray(image)[image.height // 2, image.width // 2]
    assert abs(int(pixel[0]) - 89) <= 3, "sRGB(0.1), not brightened by un-premultiplying"
    assert abs(int(pixel[3]) - 51) <= 3


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


@pytest.mark.parametrize(
    ("image", "colorspace", "expected"),
    [
        # Over-range clamps to white, linear 0.5 is sRGB 188 (not 128), black stays black.
        (torch.tensor([[[[4.0, 0.5, 0.0], [0.0, 0.0, 0.0]]]]), "linear", [[[255, 188, 0], [0, 0, 0]]]),
        # Channel-less gray 4 wide, sRGB in: every column converted, and half-float darks kept
        # (0.28 sRGB is linear ~0.064, sRGB 71; crushed darks were near 0).
        (torch.full((1, 2, 4), 0.28), "sRGB", [[[71, 71, 71]] * 4] * 2),
    ],
)
def test_the_preview_pixels_match_the_golden_for_outputs_and_uploads(dirs, image, colorspace, expected):
    from app.assets.previews import _decode_for_preview

    with patch.object(nodes_images, "linear_to_preview", wraps=linear_to_preview) as tonemap:
        entry = _save(image, colorspace, bit_depth="16-bit float")[0]

    written = np.asarray(linear_to_preview(tonemap.call_args.args[0]))
    uploaded = np.asarray(_decode_for_preview(str(dirs / "output" / entry["filename"])))
    assert written.tolist() == expected, "the save node's preview"
    assert uploaded.tolist() == expected, "the upload decoder's preview of the saved EXR"
