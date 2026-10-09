"""Core's EXR generator, and making, storing and linking previews: best-effort, since
an asset without a preview is a normal state, and bounded by the caller's deadline."""

from __future__ import annotations

import asyncio
import io
import logging
import mimetypes
import os
import time
import uuid
from dataclasses import dataclass

import av
import numpy as np
from PIL import Image

import folder_paths
from app.assets.database.models import Asset
from app.assets.database.queries.records import create_content, create_record
from app.assets.event_log import emit, error_type
from app.assets.services.image_dimensions import read_exr_windows
from app.database.db import create_write_session
from comfy_execution.preview_generators import (
    PREVIEW_MAX_PIXELS,
    get_preview_generator,
    linear_to_preview,
    set_core_preview_generator,
    submit_preview_job,
)

PREVIEW_DEADLINE_SECONDS = 5.5
# Uploads only: a decode costs ~30 MB per MP, and running out of memory can't be caught.
PREVIEW_MAX_SOURCE_PIXELS = 17_000_000
_ENCODABLE_MODES = frozenset({"RGB", "RGBA", "L", "LA", "P"})


class PreviewSkipped(Exception):
    """A preview Core declines to make; the argument is the event's reason."""


@dataclass(frozen=True)
class _Failed:
    reason: str
    exc: BaseException | None = None


def _decode_for_preview(path: str, max_pixels: int) -> Image.Image:
    """An SDR image of an EXR's display window, at most ``max_pixels``."""
    # Name the demuxer: left to probe, FFmpeg picks one from the file's contents.
    with av.open(path, format="exr_pipe") as container:
        frame = next(container.decode(video=0))
    if frame.format.name.startswith("gray"):
        # Exact; converting half-float gray to gbrpf32le crushes the darks.
        rgb = np.repeat(frame.to_ndarray(format="grayf32le")[..., None], 3, axis=-1)
    else:
        rgb = frame.to_ndarray(format="gbrpf32le")
    import torch  # not at import time: this module loads before main.py configures CUDA's allocator

    return linear_to_preview(torch.from_numpy(rgb), max_pixels)


class ExrPreviewGenerator:
    """Core's EXR generator: scene-linear clamped to sRGB, alpha dropped."""

    mime_types = ("image/x-exr",)

    def generate(self, source_path: str, max_pixels: int) -> Image.Image | None:
        windows = read_exr_windows(source_path)
        if windows is None:
            raise PreviewSkipped("decode_failed")
        if max(w * h for w, h in windows) > PREVIEW_MAX_SOURCE_PIXELS:
            raise PreviewSkipped("too_large")
        return _decode_for_preview(source_path, max_pixels)


set_core_preview_generator(ExrPreviewGenerator())


def preview_mime_type(path: str) -> str | None:
    """The type generators are keyed by: from the path, never an uploader-supplied type."""
    return mimetypes.guess_type(path, strict=False)[0]


def has_preview_generator(path: str) -> bool:
    return get_preview_generator(preview_mime_type(path)) is not None


def _encode(image: object) -> tuple[bytes, int, int]:
    if not isinstance(image, Image.Image) or image.mode not in _ENCODABLE_MODES:
        raise TypeError("preview generators must return an 8-bit PIL image")
    width, height = image.size
    # Downscale before converting, and to WebP's 16383 px side limit as well as the area cap.
    scale = min(1.0, (PREVIEW_MAX_PIXELS / (width * height)) ** 0.5, 16383 / max(width, height))
    if scale < 1.0:
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.BILINEAR)
    image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", quality=80)
    return buffer.getvalue(), image.width, image.height


def _generate(path: str) -> tuple[bytes, int, int] | _Failed | None:
    """Runs on a preview worker: decode and encode only, never disk or database."""
    generator = get_preview_generator(preview_mime_type(path))
    if generator is None:
        return None
    try:
        image = generator.generate(path, PREVIEW_MAX_PIXELS)
    except PreviewSkipped as skipped:
        return _Failed(skipped.args[0])
    except BaseException as exc:  # third-party code on a worker: nothing may escape to the caller
        return _Failed("decode_failed", exc)
    if image is None:
        return None
    try:
        return _encode(image)
    except Exception as exc:
        return _Failed("encode_failed" if isinstance(image, Image.Image) else "decode_failed", exc)


def _store_and_link(parent_id: str, webp: bytes, width: int, height: int) -> str | None:
    """Write the preview, register it and link it. None if the parent is gone or got a preview meanwhile."""
    preview_id = str(uuid.uuid4())
    directory = folder_paths.get_previews_directory()
    path = os.path.join(directory, f"{preview_id}.webp")
    linked = False
    try:
        os.makedirs(directory, exist_ok=True)
        with open(path, "wb") as f:
            f.write(webp)
        mtime_ns = os.stat(path).st_mtime_ns
        # A write session holds the lock from its first statement, so this check and the link are atomic.
        with create_write_session() as session:
            parent = session.get(Asset, parent_id)
            if parent is None or parent.preview_id is not None:
                return None
            content = create_content(session, path, size_bytes=len(webp), mtime_ns=mtime_ns)
            record = create_record(
                session,
                content.id,
                f"{preview_id}.webp",
                mime_type="image/webp",
                tags=["preview"],
                system_metadata={"kind": "image", "width": width, "height": height},
            )
            parent.preview_id = record.id
            record_id = record.id
            session.commit()
            linked = True
        return record_id
    finally:
        if not linked and os.path.exists(path):
            os.remove(path)


def _format(path: str) -> str:
    return "exr" if preview_mime_type(path) == "image/x-exr" else "other"


async def generate_upload_preview(asset_id: str, path: str | None, preview_id: str | None) -> str | None:
    """The preview_id an upload ends with: the one it has, else one generated now if possible.

    Waits at most ``PREVIEW_DEADLINE_SECONDS``. A preview not ready by then is dropped and
    logged as a timeout: it never gets stored, and nothing of it is kept.
    """
    if preview_id is not None or not path or not has_preview_generator(path):
        return preview_id
    started = time.monotonic()
    try:
        future = submit_preview_job(lambda: _generate(path))
        try:
            result = await asyncio.wait_for(asyncio.wrap_future(future), PREVIEW_DEADLINE_SECONDS)
        except asyncio.TimeoutError:
            _emit_failed(path, "timeout")
            return None
        return _store_result(asset_id, path, result, started)
    except Exception:
        logging.warning("Preview generation failed for upload %s", asset_id, exc_info=True)
        return None


def _emit_failed(path: str, reason: str) -> None:
    emit("previews.generation_failed", format=_format(path), reason=reason, source="upload")


def _emit_error(path: str, reason: str, exc: BaseException) -> None:
    emit("previews.generation_failed", format=_format(path), reason=reason, source="upload", error_type=error_type(exc))


def _store_result(parent_id: str, path: str, result, started: float) -> str | None:
    if result is None:
        return None
    if isinstance(result, _Failed):
        if result.exc is None:
            _emit_failed(path, result.reason)
        else:
            _emit_error(path, result.reason, result.exc)
        return None
    webp, width, height = result
    try:
        preview_id = _store_and_link(parent_id, webp, width, height)
    except Exception as exc:
        _emit_error(path, "write_failed", exc)
        return None
    if preview_id is not None:
        emit("previews.generated", format=_format(path), source="upload", elapsed_ms=int((time.monotonic() - started) * 1000))
    return preview_id
