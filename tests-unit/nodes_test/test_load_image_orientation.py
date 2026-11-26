import numpy as np
import pytest
import torch
from PIL import Image, ImageOps

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import folder_paths
import node_helpers
import nodes


def _write(path, orientation, fmt):
    rng = np.random.default_rng(orientation)
    pixels = (rng.random((48, 80, 3)) * 255).astype(np.uint8)  # non-square: 80 wide, 48 tall
    pixels[:8, :16] = 255  # asymmetric marker so a mirror is not mistaken for a rotation
    exif = Image.Exif()
    exif[0x0112] = orientation
    kwargs = {"exif": exif}
    if fmt == "JPEG":
        kwargs.update(quality=100, subsampling=0)
    elif fmt == "WEBP":
        kwargs.update(lossless=True)
    Image.fromarray(pixels).save(path, fmt, **kwargs)


@pytest.mark.parametrize("fmt,ext", [("PNG", "png"), ("JPEG", "jpg"), ("WEBP", "webp")])
@pytest.mark.parametrize("orientation", range(1, 9))
def test_load_image_applies_exif_orientation(tmp_path, monkeypatch, fmt, ext, orientation):
    path = tmp_path / f"o{orientation}.{ext}"
    _write(path, orientation, fmt)
    assert node_helpers.image_exif_orientation(str(path)) == orientation

    monkeypatch.setattr(folder_paths, "get_annotated_filepath", lambda name: str(path))
    image, mask = nodes.LoadImage().load_image(path.name)

    with Image.open(path) as img:
        expected = np.asarray(ImageOps.exif_transpose(img).convert("RGB"), dtype=np.float32) / 255.0
    got = image[0].float().cpu().numpy()
    assert got.shape == expected.shape
    assert np.abs(got - expected).mean() < 0.01


def test_image_exif_orientation_defaults(tmp_path):
    plain = tmp_path / "plain.png"
    Image.new("RGB", (4, 2)).save(plain)
    assert node_helpers.image_exif_orientation(str(plain)) == 1
    junk = tmp_path / "junk.png"
    junk.write_bytes(b"not an image")
    assert node_helpers.image_exif_orientation(str(junk)) == 1
