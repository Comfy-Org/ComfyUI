import struct
import zlib

import numpy as np
from PIL import Image
import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import nodes


def png_idat(path):
    with path.open("rb") as png:
        assert png.read(8) == b"\x89PNG\r\n\x1a\n"
        idat = bytearray()
        while True:
            length, kind = struct.unpack(">I4s", png.read(8))
            data = png.read(length)
            png.read(4)
            if kind == b"IDAT":
                idat.extend(data)
            elif kind == b"IEND":
                return idat


@pytest.mark.parametrize("shape", [(17, 9, 3), (8, 5, 3)])
def test_save_image_writes_uncompressed_rgb_at_variable_sizes(tmp_path, monkeypatch, shape):
    monkeypatch.setattr(args, "fast_unfiltered_rgb_png", True)
    monkeypatch.setattr(args, "disable_metadata", False)
    image = torch.ones((1, *shape))
    saver = nodes.SaveImage()
    saver.output_dir = str(tmp_path)
    result = saver.save_images(image, prompt={"example": True})
    path = tmp_path / result["ui"]["images"][0]["filename"]

    with Image.open(path) as saved:
        assert saved.size == (shape[1], shape[0])
        assert saved.mode == "RGB"
        assert saved.text["prompt"] == '{"example": true}'
        np.testing.assert_array_equal(np.asarray(saved), np.full(shape, 255, dtype=np.uint8))

    idat = png_idat(path)
    assert idat[2] & 0b110 == 0
    assert zlib.decompress(idat) == (b"\0" + b"\xff" * (shape[1] * 3)) * shape[0]


def test_save_image_keeps_alpha_channel_with_fast_png_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr(args, "fast_unfiltered_rgb_png", True)
    image = torch.tensor([[[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]]]])
    saver = nodes.SaveImage()
    saver.output_dir = str(tmp_path)
    result = saver.save_images(image)
    path = tmp_path / result["ui"]["images"][0]["filename"]
    with Image.open(path) as saved:
        assert saved.mode == "RGBA"
        np.testing.assert_array_equal(np.asarray(saved), [[[255, 0, 0, 0], [0, 255, 0, 255]]])
