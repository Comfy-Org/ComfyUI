import io

import pytest
import torch
from PIL import Image

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import folder_paths
import nodes


def _jpeg_that_probes_as_mpegts() -> bytes:
    """A valid 64x64 red JPEG whose bytes make ffmpeg's probe prefer the mpegts demuxer.

    The MPEG-TS probe scores 0x47 sync bytes found at a 188-byte stride; a COM segment
    carrying a TS-looking header at every stride inside the first probe buffer is enough to
    out-score the JPEG probe, after which av.open() fails with EOFError. Pillow skips the
    COM segment and decodes the image normally.
    """
    image = Image.new("RGB", (64, 64), (255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    data = buffer.getvalue()
    assert data[:4] == b"\xff\xd8\xff\xe0"

    insert_at = 4 + int.from_bytes(data[4:6], "big")  # right after the APP0 segment
    payload = bytearray(2200)
    for offset in range(0, len(payload) - 4, 188):
        payload[offset:offset + 4] = b"\x47\x1f\xff\x10"  # sync byte, PID 0x1FFF, adaptation field
    comment_segment = b"\xff\xfe" + (len(payload) + 2).to_bytes(2, "big") + bytes(payload)
    return data[:insert_at] + comment_segment + data[insert_at:]


@pytest.mark.parametrize("filename", ["mis_probed_image", "mis_probed_image.jpg"])
def test_load_image_falls_back_to_pillow_when_pyav_cannot_open(tmp_path, monkeypatch, filename):
    (tmp_path / filename).write_bytes(_jpeg_that_probes_as_mpegts())
    monkeypatch.setattr(folder_paths, "input_directory", str(tmp_path))

    image, mask = nodes.LoadImage().load_image(filename)

    assert image.shape == (1, 64, 64, 3)
    assert mask.shape == (1, 64, 64)
    assert image[0, :, :, 0].min() > 0.9
    assert image[0, :, :, 1:].max() < 0.1
