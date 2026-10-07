"""Image dimension extraction for asset ingest.

Reads only the image header via Pillow to capture width/height cheaply,
without a full pixel decode. Returns a metadata dict suitable for merging
into ``AssetReference.system_metadata``.
"""
from __future__ import annotations

import logging
import struct
from typing import Any

logger = logging.getLogger(__name__)

_EXR_MAGIC = b"\x76\x2f\x31\x01"
_EXR_MAX_ATTRIBUTES = 1024


def _read_cstring(f) -> bytes:
    out = bytearray()
    while len(out) <= 255:
        b = f.read(1)
        if not b or b == b"\0":
            return bytes(out)
        out += b
    raise ValueError("EXR header name too long")


def read_exr_windows(file_path: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """(display, data) window sizes as (width, height), read from the header alone.

    Pillow can't open EXR, and opening it with PyAV buffers the whole file.
    """
    windows: dict[bytes, tuple[int, int]] = {}
    try:
        with open(file_path, "rb") as f:
            if f.read(4) != _EXR_MAGIC:
                return None
            f.read(4)  # version and flags
            for _ in range(_EXR_MAX_ATTRIBUTES):
                name = _read_cstring(f)
                if not name:
                    break
                _read_cstring(f)  # attribute type
                (size,) = struct.unpack("<i", f.read(4))
                if size < 0:
                    return None
                if name in (b"displayWindow", b"dataWindow") and size == 16:
                    x_min, y_min, x_max, y_max = struct.unpack("<4i", f.read(16))
                    windows[name] = (x_max - x_min + 1, y_max - y_min + 1)
                else:
                    f.seek(size, 1)
    except (OSError, ValueError, struct.error):
        return None
    display, data = windows.get(b"displayWindow"), windows.get(b"dataWindow")
    if display is None or data is None or min(*display, *data) <= 0:
        return None
    return display, data


def extract_image_dimensions(
    file_path: str, mime_type: str | None = None
) -> dict[str, Any] | None:
    """Extract image dimensions for the file at ``file_path``.

    Args:
        file_path: Absolute path to a file on disk.
        mime_type: Optional MIME type hint. When provided and not prefixed
            with ``image/``, extraction is skipped without touching the file.

    Returns:
        ``{"kind": "image", "width": W, "height": H}`` when the file is a
        recognizable image with positive dimensions, otherwise ``None``.

    The dict shape is intended to be merged into ``system_metadata`` so the
    asset response surfaces ``metadata.kind`` plus dimension fields for image
    assets. Forward-compatible: future media kinds (e.g. ``"video"`` with
    duration/fps) can extend this shape without schema changes.
    """
    if mime_type is not None and not mime_type.startswith("image/"):
        return None
    if mime_type == "image/x-exr":
        windows = read_exr_windows(file_path)
        if windows is None:
            return None
        (width, height), _ = windows
        return {"kind": "image", "width": width, "height": height}

    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError:
        logger.debug(
            "Pillow not available; skipping image dimension extraction for %s",
            file_path,
        )
        return None

    try:
        with Image.open(file_path) as img:
            width, height = img.size
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        logger.debug(
            "Failed to read image dimensions from %s: %s", file_path, exc
        )
        return None

    if (
        not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        return None

    return {"kind": "image", "width": width, "height": height}
