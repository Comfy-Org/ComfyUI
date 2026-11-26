import struct
import zlib

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


def _write_png16(path, pixels, orientation):
    """16-bit RGB PNG with an eXIf chunk before IDAT (Pillow cannot write 16-bit RGB)."""
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    h, w, _ = pixels.shape
    exif = Image.Exif()
    exif[0x0112] = orientation
    raw = b"".join(b"\x00" + row.astype(">u2").tobytes() for row in pixels)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0))
                     + chunk(b"eXIf", exif.tobytes()[6:]) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


@pytest.mark.parametrize("orientation", [1, 6])
def test_load_image_keeps_16bit_precision_when_oriented(tmp_path, monkeypatch, orientation):
    rng = np.random.default_rng(16)
    pixels = rng.integers(0, 65536, size=(24, 40, 3), dtype=np.uint16)
    path = tmp_path / f"deep{orientation}.png"
    _write_png16(path, pixels, orientation)
    assert node_helpers.image_exif_orientation(str(path)) == orientation

    monkeypatch.setattr(folder_paths, "get_annotated_filepath", lambda name: str(path))
    image, mask = nodes.LoadImage().load_image(path.name)

    expected = pixels.astype(np.float32) / 65535.0
    if orientation == 6:
        expected = np.rot90(expected, -1, axes=(0, 1))
    got = image[0].float().cpu().numpy()
    assert got.shape == expected.shape
    # 8-bit clipping would leave a mean error of ~1e-3; 16-bit decoding stays well below that
    assert np.abs(got - expected).mean() < 2e-4


def test_load_image_keeps_palette_transparency_when_oriented(tmp_path, monkeypatch):
    path = tmp_path / "palette.png"
    img = Image.new("P", (6, 4), 1)
    img.putpalette([0, 0, 0, 255, 0, 0] + [0] * 762)
    img.putpixel((0, 0), 0)  # index 0 is transparent
    exif = Image.Exif()
    exif[0x0112] = 6
    img.save(path, "PNG", transparency=0, exif=exif)

    monkeypatch.setattr(folder_paths, "get_annotated_filepath", lambda name: str(path))
    image, mask = nodes.LoadImage().load_image(path.name)

    with Image.open(path) as src:
        alpha = np.asarray(ImageOps.exif_transpose(src).convert("RGBA").getchannel("A"), dtype=np.float32) / 255.0
    assert tuple(image.shape[1:3]) == (6, 4)
    assert np.allclose(mask[0].float().cpu().numpy(), 1.0 - alpha)


def test_image_exif_orientation_defaults(tmp_path):
    plain = tmp_path / "plain.png"
    Image.new("RGB", (4, 2)).save(plain)
    assert node_helpers.image_exif_orientation(str(plain)) == 1
    junk = tmp_path / "junk.png"
    junk.write_bytes(b"not an image")
    assert node_helpers.image_exif_orientation(str(junk)) == 1
