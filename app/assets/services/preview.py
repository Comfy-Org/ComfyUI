import logging
import os
import uuid

from PIL import Image, ImageOps

import folder_paths
from app.assets.database.queries.asset_reference import (
    get_reference_by_file_path,
    get_reference_by_id,
    set_reference_preview,
)
from app.assets.services.ingest import register_file_in_place
from app.database.db import create_session

PREVIEW_TAG = "preview"


def get_or_create_preview_file(
    source_abs_path: str, max_size: int, quality: int
) -> str:
    source_hash, source_ref_id = _ensure_source_asset(source_abs_path)
    hash_hex = source_hash.split(":")[-1]
    preview_dir = folder_paths.get_system_user_directory("preview_cache")
    preview_path = os.path.join(
        preview_dir, f"{hash_hex}_{max_size}_q{quality}.webp"
    )
    if os.path.isfile(preview_path):
        return preview_path

    os.makedirs(preview_dir, exist_ok=True)
    tmp_path = f"{preview_path}.{uuid.uuid4().hex}.tmp"
    try:
        with Image.open(source_abs_path) as img:
            preview_img = ImageOps.exif_transpose(img)
            if max(preview_img.size) > max_size:
                preview_img = ImageOps.contain(
                    preview_img, (max_size, max_size), Image.Resampling.LANCZOS
                )
            preview_img.save(tmp_path, format="webp", quality=quality)
        os.replace(tmp_path, preview_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise

    _register_preview_asset(preview_path, source_ref_id)
    return preview_path


def _ensure_source_asset(abs_path: str) -> tuple[str, str]:
    abs_path = os.path.abspath(abs_path)
    mtime_ns = os.stat(abs_path).st_mtime_ns

    with create_session() as session:
        ref = get_reference_by_file_path(session, abs_path)
        if (
            ref is not None
            and ref.mtime_ns == mtime_ns
            and ref.asset is not None
            and ref.asset.hash
        ):
            return ref.asset.hash, ref.id

    result = register_file_in_place(
        abs_path=abs_path, name=os.path.basename(abs_path), tags=[]
    )
    if not result.asset.hash:
        raise RuntimeError(f"asset registration produced no hash for {abs_path}")
    return result.asset.hash, result.ref.id


def _register_preview_asset(preview_path: str, source_ref_id: str) -> None:
    try:
        result = register_file_in_place(
            abs_path=preview_path,
            name=os.path.basename(preview_path),
            tags=[PREVIEW_TAG],
            mime_type="image/webp",
        )
        with create_session() as session:
            source_ref = get_reference_by_id(session, source_ref_id)
            if source_ref is not None and source_ref.preview_id is None:
                set_reference_preview(session, source_ref_id, result.ref.id)
            session.commit()
    except Exception:
        logging.warning("Failed to register preview image as asset", exc_info=True)
