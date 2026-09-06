"""Video metadata extraction for asset ingest.

Reads only the container/stream headers via PyAV to capture dimensions,
duration, fps and frame count cheaply, without decoding frames. Returns a
metadata dict suitable for merging into ``AssetReference.system_metadata``.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def extract_video_metadata(
    file_path: str, mime_type: str | None = None
) -> dict[str, Any] | None:
    """Extract video stream metadata for the file at ``file_path``.

    Args:
        file_path: Absolute path to a file on disk.
        mime_type: Optional MIME type hint. When provided and not prefixed
            with ``video/``, extraction is skipped without touching the file.

    Returns:
        ``{"kind": "video", "width": W, "height": H, ...}`` with optional
        ``duration`` (seconds), ``fps`` and ``frame_count`` keys when the
        file has a recognizable video stream, otherwise ``None``.
    """
    if mime_type is not None and not mime_type.startswith("video/"):
        return None

    try:
        import av
    except ImportError:
        logger.debug(
            "PyAV not available; skipping video metadata extraction for %s",
            file_path,
        )
        return None

    try:
        with av.open(file_path) as container:
            stream = next(
                (s for s in container.streams if s.type == "video"), None
            )
            if stream is None:
                return None

            fps = float(stream.average_rate) if stream.average_rate else None
            duration = None
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = float(container.duration * av.time_base)
            frame_count = stream.frames or None
            if frame_count is None and duration is not None and fps is not None:
                frame_count = round(duration * fps)

            width = stream.codec_context.width
            height = stream.codec_context.height
    except (OSError, ValueError, av.error.FFmpegError) as exc:
        logger.debug("Failed to read video metadata from %s: %s", file_path, exc)
        return None

    if (
        not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        return None

    metadata: dict[str, Any] = {"kind": "video", "width": width, "height": height}
    if duration is not None:
        metadata["duration"] = duration
    if fps is not None:
        metadata["fps"] = fps
    if frame_count is not None:
        metadata["frame_count"] = frame_count
    return metadata
