"""An EXR writer shared by the preview tests."""

import struct
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


def write_exr(path: Path, width: int, height: int, value=(1.0, 0.5, 0.25), display_window=None) -> Path:
    """An EXR the way SaveImageAdvanced writes one (PyAV, uncompressed half)."""
    rgb = np.empty((height, width, 3), np.float32)
    rgb[...] = value
    codec = av.CodecContext.create("exr", "w")
    codec.width, codec.height, codec.pix_fmt = width, height, "gbrpf32le"
    codec.time_base = Fraction(1, 1)
    codec.options = {"format": "half"}
    frame = av.VideoFrame.from_ndarray(rgb, format="gbrpf32le")
    frame.pts = 0
    frame.time_base = codec.time_base
    data = bytearray(b"".join(bytes(p) for p in list(codec.encode(frame)) + list(codec.encode(None))))
    if display_window is not None:
        key = b"displayWindow\x00box2i\x00"
        at = data.find(key) + len(key) + 4
        struct.pack_into("<4i", data, at, *display_window)
    path.write_bytes(bytes(data))
    return path
