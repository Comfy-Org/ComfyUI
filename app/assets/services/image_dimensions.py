"""Image dimension extraction for asset ingest.

Reads only the image header via Pillow to capture width/height cheaply,
without a full pixel decode. Returns a metadata dict suitable for merging
into ``AssetReference.system_metadata``.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


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
