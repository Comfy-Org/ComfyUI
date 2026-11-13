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


def duplicate_data_window(data: bytes) -> bytes:
    """An EXR header with its dataWindow attribute repeated."""
    key = b"dataWindow\x00box2i\x00"
    at = data.find(key)
    end = at + len(key) + 4 + 16
    return data[:end] + data[at:end] + data[end:]


def write_prores(path: Path, width: int, height: int, frames: int = 2, value=0.5, transfer=None, alpha=False, audio=False) -> Path:
    """A ProRes MOV the way SaveVideo writes one; BT.2020 with ``transfer`` given, PCM audio with ``audio``."""
    from av.video.reformatter import ColorPrimaries

    with av.open(str(path), "w", format="mov") as out:
        stream = out.add_stream("prores_ks", rate=24)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuva444p10le" if alpha else "yuv422p10le"
        if alpha:
            stream.options = {"profile": "4444"}
        if transfer is not None:
            stream.codec_context.color_trc, stream.codec_context.color_primaries, stream.codec_context.colorspace = transfer, ColorPrimaries.BT2020, 9
        sound = out.add_stream("pcm_s16le", rate=48000, layout="stereo") if audio else None
        pixels = np.full((height, width, 4 if alpha else 3), value, np.float32)
        for i in range(frames):
            frame = av.VideoFrame.from_ndarray(pixels, format="gbrapf32le" if alpha else "gbrpf32le")
            frame = frame.reformat(format=stream.pix_fmt, dst_colorspace=9 if transfer is not None else 1)
            if transfer is not None:
                frame.color_trc, frame.color_primaries = transfer, ColorPrimaries.BT2020
            frame.pts = i
            out.mux(stream.encode(frame))
        out.mux(stream.encode(None))
        if sound is not None:
            samples = av.AudioFrame.from_ndarray(np.zeros((1, 2 * 2000 * frames), np.int16), format="s16", layout="stereo")
            samples.sample_rate, samples.pts = 48000, 0
            out.mux(sound.encode(samples))
            out.mux(sound.encode(None))
    return path
