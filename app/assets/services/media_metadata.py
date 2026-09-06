from __future__ import annotations

from typing import Any

from app.assets.services.image_dimensions import extract_image_dimensions
from app.assets.services.video_metadata import extract_video_metadata

MEDIA_METADATA_KEYS = (
    "kind",
    "width",
    "height",
    "duration",
    "fps",
    "frame_count",
)


def extract_media_metadata(
    file_path: str, mime_type: str | None = None
) -> dict[str, Any] | None:
    """Extract media metadata for the file, dispatched on the MIME prefix.

    Returns ``None`` for non-media MIME types or unreadable files.
    """
    if not mime_type:
        return None
    if mime_type.startswith("image/"):
        return extract_image_dimensions(file_path, mime_type=mime_type)
    if mime_type.startswith("video/"):
        return extract_video_metadata(file_path, mime_type=mime_type)
    return None
