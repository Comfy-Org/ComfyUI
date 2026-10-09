"""Previews of uploaded EXRs: decoded, stored and linked on upload. Best-effort, since an
asset without a preview is a normal state."""

from __future__ import annotations

import asyncio
import io
import mimetypes
import os
import threading
import time
import uuid

import av
import numpy as np
import torch
from PIL import Image

import folder_paths
from app.assets.database.models import Asset
from app.assets.database.queries.records import create_content, create_record
from app.assets.event_log import emit, error_type
from app.assets.services.image_dimensions import read_exr_windows
from app.database.db import create_write_session
from comfy_execution.preview_tonemap import linear_to_preview

# A decode costs ~30 MB per MP, and running out of memory can't be caught.
PREVIEW_MAX_SOURCE_PIXELS = 17_000_000
_DECODE_SLOTS = threading.BoundedSemaphore(2)


class PreviewSkipped(Exception):
    """A preview Core declines to make; the argument is the event's reason."""


def _decode_for_preview(path: str) -> Image.Image:
    """An SDR image of an EXR's display window, at most one megapixel; RGBA if the EXR has alpha."""
    windows = read_exr_windows(path)
    if windows is None:
        raise PreviewSkipped("decode_failed")
    if max(w * h for w, h in windows) > PREVIEW_MAX_SOURCE_PIXELS:
        raise PreviewSkipped("too_large")
    # Name the demuxer: left to probe, FFmpeg picks one from the file's contents.
    with av.open(path, format="exr_pipe") as container:
        frame = next(container.decode(video=0))
    if frame.format.name.startswith("gray"):
        # Exact; converting half-float gray to gbrpf32le crushes the darks.
        pixels = np.repeat(frame.to_ndarray(format="grayf32le")[..., None], 3, axis=-1)
    else:
        pixels = frame.to_ndarray(format="gbrapf32le" if len(frame.format.components) == 4 else "gbrpf32le")
    return linear_to_preview(torch.from_numpy(pixels))


def _make_preview(path: str) -> tuple[bytes, int, int]:
    with _DECODE_SLOTS:
        image = _decode_for_preview(path)
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", quality=80)
    return buffer.getvalue(), image.width, image.height


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


def is_exr(path: str) -> bool:
    """From the path, never an uploader-supplied type."""
    return mimetypes.guess_type(path, strict=False)[0] == "image/x-exr"


def _emit_failed(reason: str) -> None:
    emit("previews.generation_failed", reason=reason)


def _emit_error(reason: str, exc: BaseException) -> None:
    emit("previews.generation_failed", reason=reason, error_type=error_type(exc))


async def generate_upload_preview(asset_id: str, path: str | None, preview_id: str | None) -> str | None:
    """The preview_id an upload ends with: the one it has, else one made now for an EXR."""
    if preview_id is not None or not path or not is_exr(path):
        return preview_id
    started = time.monotonic()
    try:
        webp, width, height = await asyncio.to_thread(_make_preview, path)
    except PreviewSkipped as skipped:
        _emit_failed(skipped.args[0])
        return None
    except Exception as exc:
        _emit_error("decode_failed", exc)
        return None
    try:
        linked = _store_and_link(asset_id, webp, width, height)
    except Exception as exc:
        _emit_error("write_failed", exc)
        return None
    if linked is not None:
        emit("previews.generated", elapsed_ms=int((time.monotonic() - started) * 1000))
    return linked
