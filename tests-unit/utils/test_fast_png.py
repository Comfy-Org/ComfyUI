import os
import stat
import struct
import zlib

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import pytest

from utils.fast_png import save_unfiltered_rgb_png


@pytest.mark.parametrize("include_metadata", [True, False])
def test_unfiltered_rgb_png_roundtrip(tmp_path, include_metadata):
    pixels = np.array(
        [[[0, 13, 255], [73, 0, 14], [0, 128, 3]],
         [[200, 1, 90], [5, 6, 7], [255, 40, 0]]],
        dtype=np.uint8,
    )
    metadata = None
    expected_metadata = {}
    if include_metadata:
        expected_metadata = {"prompt": '{"fixture": true}', "workflow": '{"label": "東京"}'}
        metadata = PngInfo()
        for key, value in expected_metadata.items():
            metadata.add_text(key, value)
    destination = tmp_path / "frame.png"

    save_unfiltered_rgb_png(destination, pixels, metadata)

    with Image.open(destination) as image:
        assert image.format == "PNG"
        assert image.mode == "RGB"
        assert image.size == (3, 2)
        np.testing.assert_array_equal(np.asarray(image), pixels)
        assert image.text == expected_metadata

    data = destination.read_bytes()
    offset = 8
    idat = []
    while offset < len(data):
        length, kind = struct.unpack_from(">I4s", data, offset)
        if kind == b"IDAT":
            idat.append(data[offset + 8 : offset + 8 + length])
        offset += length + 12
    stream = b"".join(idat)
    assert stream[2] & 0b110 == 0  # DEFLATE BTYPE=0 stores the block without compression.
    assert zlib.decompress(stream) == b"".join(b"\0" + row.tobytes() for row in pixels)
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.skipif(os.name != "posix", reason="POSIX output permissions")
@pytest.mark.parametrize("umask", [0o022, 0o077])
def test_fast_png_uses_pillow_output_permissions(tmp_path, umask):
    pixels = np.zeros((2, 3, 3), dtype=np.uint8)
    reference = tmp_path / "pillow.png"
    destination = tmp_path / "unfiltered.png"

    previous_umask = os.umask(umask)
    try:
        Image.fromarray(pixels).save(reference)
        save_unfiltered_rgb_png(destination, pixels, None)
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE(destination.stat().st_mode) == stat.S_IMODE(reference.stat().st_mode)
