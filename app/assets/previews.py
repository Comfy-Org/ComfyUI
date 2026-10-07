"""Generated previews: Core's EXR generator, and making, storing and linking a preview.

Generation runs on the preview workers; storing and linking run on the caller, and
only for previews that finished before the caller's deadline, so a late or failed
preview never leaves a file or record behind. Everything here is best-effort: an
asset without a preview is a normal state, so failures are logged, never raised.
"""

from __future__ import annotations

import asyncio
import io
import logging
import mimetypes
import os
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from sqlalchemy import update

import folder_paths
from app.assets.database.models import Asset
from app.assets.database.queries.records import create_content, create_record
from app.assets.event_log import emit, error_type
from app.assets.services.image_dimensions import read_exr_windows
from app.database.db import create_write_session
from comfy_execution.preview_generators import (
    get_preview_generator,
    preview_deadline_seconds,
    set_core_preview_generator,
    submit_preview_job,
)

if TYPE_CHECKING:
    from concurrent.futures import Future

PREVIEW_MAX_PIXELS = 1_000_000
# Above this the decode alone needs gigabytes; a crash there can't be caught.
PREVIEW_MAX_SOURCE_PIXELS = 100_000_000
_ENCODABLE_MODES = frozenset({"RGB", "RGBA", "L", "LA", "P"})


class PreviewSkipped(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class _Failed:
    reason: str
    exc: BaseException | None = None


def _decode_for_preview(path: str, max_pixels: int) -> Image.Image:
    """An SDR image of an EXR's display window, at most ``max_pixels``."""
    import av

    # Name the demuxer: left to probe, FFmpeg picks one from the file's contents.
    with av.open(path, format="exr_pipe") as container:
        frame = next(container.decode(video=0))
    rgb = frame.to_ndarray(format="gbrpf32le")
    height, width = rgb.shape[:2]
    scale = min(1.0, (max_pixels / (width * height)) ** 0.5)
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    # Resize in float first, so the per-pixel tonemap below runs on at most max_pixels.
    small = np.stack(
        [np.asarray(Image.fromarray(np.ascontiguousarray(rgb[..., c]), "F").resize(size, Image.BILINEAR)) for c in range(3)],
        axis=-1,
    )
    linear = np.clip(np.nan_to_num(small, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    srgb = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.power(linear, 1 / 2.4) - 0.055)
    return Image.fromarray((srgb * 255 + 0.5).astype(np.uint8), "RGB")


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
    image = image.convert("RGB")
    width, height = image.size
    if width * height > PREVIEW_MAX_PIXELS:
        scale = (PREVIEW_MAX_PIXELS / (width * height)) ** 0.5
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.BILINEAR)
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
        return _Failed(skipped.reason)
    except Exception as exc:
        return _Failed("decode_failed", exc)
    if image is None:
        return None
    try:
        return _encode(image)
    except Exception as exc:
        return _Failed("encode_failed" if isinstance(image, Image.Image) else "decode_failed", exc)


def _store_and_link(parent_id: str, webp: bytes, width: int, height: int) -> str | None:
    """Write the preview, register it and link it to its parent. None if the parent is gone."""
    preview_id = str(uuid.uuid4())
    directory = folder_paths.get_previews_directory()
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{preview_id}.webp")
    with open(path, "wb") as f:
        f.write(webp)
    mtime_ns = os.stat(path).st_mtime_ns
    record_id = None
    try:
        # A write session holds the lock from its first statement, so the parent can't go after this check.
        with create_write_session() as session:
            if session.get(Asset, parent_id) is not None:
                content = create_content(session, path, size_bytes=len(webp), mtime_ns=mtime_ns)
                record = create_record(
                    session,
                    content.id,
                    f"{preview_id}.webp",
                    mime_type="image/webp",
                    tags=["preview"],
                    system_metadata={"kind": "image", "width": width, "height": height},
                )
                session.execute(update(Asset).where(Asset.id == parent_id).values(preview_id=record.id))
                record_id = record.id
                session.commit()
    finally:
        if record_id is None:
            os.remove(path)
    return record_id


def _format(path: str) -> str:
    return "exr" if preview_mime_type(path) == "image/x-exr" else "other"


async def generate_upload_preview(asset_id: str, path: str | None, preview_id: str | None) -> str | None:
    """The preview_id an upload ends with: the one it has, else one generated now if possible."""
    if preview_id is not None or not path or not has_preview_generator(path):
        return preview_id
    try:
        return (await generate_previews([(asset_id, path)], "upload")).get(asset_id)
    except Exception:
        logging.warning("Preview generation failed for upload %s", asset_id, exc_info=True)
        return None


def _emit_failed(path: str, source: str, reason: str) -> None:
    emit("previews.generation_failed", format=_format(path), reason=reason, source=source)


def _emit_error(path: str, source: str, reason: str, exc: BaseException) -> None:
    emit("previews.generation_failed", format=_format(path), reason=reason, source=source, error_type=error_type(exc))


async def generate_previews(parents: list[tuple[str, str]], source: str) -> dict[str, str]:
    """Make, store and link previews for (asset id, path) pairs; returns asset id -> preview id.

    Waits at most ``preview_deadline_seconds(len(parents))``. Anything not stored by then
    is dropped and logged as a timeout: it never gets a preview, and nothing of it is kept.
    """
    if not parents:
        return {}
    started = time.monotonic()
    deadline = started + preview_deadline_seconds(len(parents))
    futures: list[Future] = [submit_preview_job(lambda p=path: _generate(p)) for _, path in parents]
    pending = [asyncio.wrap_future(f) for f in futures]
    await asyncio.wait(pending, timeout=max(0.0, deadline - time.monotonic()))

    linked: dict[str, str] = {}
    for (parent_id, path), future in zip(parents, futures):
        if not future.done() or time.monotonic() > deadline:
            future.cancel()
            _emit_failed(path, source, "timeout")
            continue
        try:
            result = future.result()
            if result is None:
                continue
            if isinstance(result, _Failed):
                if result.exc is None:
                    _emit_failed(path, source, result.reason)
                else:
                    _emit_error(path, source, result.reason, result.exc)
                continue
            webp, width, height = result
            try:
                preview_id = _store_and_link(parent_id, webp, width, height)
            except Exception as exc:
                _emit_error(path, source, "write_failed", exc)
                continue
            if preview_id is not None:
                linked[parent_id] = preview_id
                emit(
                    "previews.generated",
                    format=_format(path),
                    source=source,
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                )
        except Exception:
            logging.warning("Preview generation failed for %s", path, exc_info=True)
    return linked
