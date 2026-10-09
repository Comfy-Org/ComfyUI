"""An EXR writer shared by the preview tests."""

import struct
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


def write_exr(path: Path, width: int, height: int, value=(1.0, 0.5, 0.25), display_window=None) -> Path:
    """An EXR the way SaveImageAdvanced writes one (PyAV, uncompressed half); RGBA given 4 values."""
    fmt = "gbrapf32le" if len(value) == 4 else "gbrpf32le"
    pixels = np.empty((height, width, len(value)), np.float32)
    pixels[...] = value
    codec = av.CodecContext.create("exr", "w")
    codec.width, codec.height, codec.pix_fmt = width, height, fmt
    codec.time_base = Fraction(1, 1)
    codec.options = {"format": "half"}
    frame = av.VideoFrame.from_ndarray(pixels, format=fmt)
    frame.pts = 0
    frame.time_base = codec.time_base
    data = bytearray(b"".join(bytes(p) for p in list(codec.encode(frame)) + list(codec.encode(None))))
    if display_window is not None:
        key = b"displayWindow\x00box2i\x00"
        at = data.find(key) + len(key) + 4
        struct.pack_into("<4i", data, at, *display_window)
    path.write_bytes(bytes(data))
    return path


def write_preview(roots: Path, size=(4, 3), color=(200, 100, 50), alpha=False) -> dict:
    """A preview the way SaveImageAdvanced names one: previews/<blake3>.jpg (.webp with alpha), and its entry ref."""
    import io

    from blake3 import blake3
    from PIL import Image

    buffer = io.BytesIO()
    if alpha:
        Image.new("RGBA", size, (*color, 128)).save(buffer, format="WEBP", quality=80)
    else:
        Image.new("RGB", size, color).save(buffer, format="JPEG", quality=85)
    data = buffer.getvalue()
    filename = f"{blake3(data).hexdigest()}.{'webp' if alpha else 'jpg'}"
    (roots / "previews" / filename).write_bytes(data)
    return {"filename": filename, "width": size[0], "height": size[1]}
