import contextlib
import logging
import mimetypes
import os
from typing import Any, Sequence

from sqlalchemy.orm import Session

import app.assets.services.hashing as hashing
from app.assets.database.queries import (
    add_tags_to_reference,
    count_active_siblings,
    create_stub_asset,
    ensure_tags_exist,
    fetch_reference_and_asset,
    get_asset_by_hash,
    get_reference_by_file_path,
    get_reference_tags,
    get_or_create_reference,
    list_references_by_asset_id,
    reference_exists,
    remove_missing_tag_for_asset_id,
    set_reference_metadata,
    set_reference_system_metadata,
    set_reference_tags,
    update_asset_hash_and_mime,
    upsert_asset,
    upsert_reference,
    validate_tags_exist,
)
from app.assets.helpers import get_utc_now, normalize_tags
from app.assets.services.bulk_ingest import batch_insert_seed_assets
from app.assets.services.file_utils import get_size_and_mtime_ns
from app.assets.services.image_dimensions import extract_image_dimensions
from app.assets.services.path_utils import (
    compute_relative_filename,
    get_name_and_tags_from_asset_path,
    resolve_destination_from_tags,
    validate_path_within_base,
)
from app.assets.services.schemas import (
    IngestResult,
    RegisterAssetResult,
    UploadResult,
    UserMetadata,
    extract_asset_data,
    extract_reference_data,
)
from app.database.db import create_session


def _ingest_file_from_path(
    abs_path: str,
    asset_hash: str,
    size_bytes: int,
    mtime_ns: int,
    mime_type: str | None = None,
    info_name: str | None = None,
    owner_id: str = "",
    preview_id: str | None = None,
    user_metadata: UserMetadata = None,
    tags: Sequence[str] = (),
    tag_origin: str = "manual",
    require_existing_tags: bool = False,
) -> IngestResult:
    locator = os.path.abspath(abs_path)
    user_metadata = user_metadata or {}

    asset_created = False
    asset_updated = False
    ref_created = False
    ref_updated = False
    reference_id: str | None = None

    with create_session() as session:
        if preview_id:
            if not reference_exists(session, preview_id):
                preview_id = None

        asset, asset_created, asset_updated = upsert_asset(
            session,
            asset_hash=asset_hash,
            size_bytes=size_bytes,
            mime_type=mime_type,
        )

        ref_created, ref_updated = upsert_reference(
            session,
            asset_id=asset.id,
            file_path=locator,
            name=info_name or os.path.basename(locator),
            mtime_ns=mtime_ns,
            owner_id=owner_id,
        )

        # Get the reference we just created/updated
        ref = get_reference_by_file_path(session, locator)
        if ref:
            reference_id = ref.id

            if preview_id and ref.preview_id != preview_id:
                ref.preview_id = preview_id

            norm = normalize_tags(list(tags))
            if norm:
                if require_existing_tags:
                    validate_tags_exist(session, norm)
                add_tags_to_reference(
                    session,
                    reference_id=reference_id,
                    tags=norm,
                    origin=tag_origin,
                    create_if_missing=not require_existing_tags,
                )

            _update_metadata_with_filename(
                session,
                reference_id=reference_id,
                file_path=ref.file_path,
                current_metadata=ref.user_metadata,
                user_metadata=user_metadata,
            )

            _maybe_store_image_dimensions(
                session,
                reference_id=reference_id,
                file_path=locator,
                mime_type=mime_type,
                current_system_metadata=ref.system_metadata,
            )

        try:
            remove_missing_tag_for_asset_id(session, asset_id=asset.id)
        except Exception:
            logging.exception("Failed to clear 'missing' tag for asset %s", asset.id)

        session.commit()

    return IngestResult(
        asset_created=asset_created,
        asset_updated=asset_updated,
        ref_created=ref_created,
        ref_updated=ref_updated,
        reference_id=reference_id,
    )


def register_output_files(
    file_paths: Sequence[str],
    user_metadata: UserMetadata = None,
    job_id: str | None = None,
) -> int:
    """Register a batch of output file paths as assets.

    Returns the number of files successfully registered.
    """
    registered = 0
    for abs_path in file_paths:
        if not os.path.isfile(abs_path):
            continue
        try:
            if ingest_existing_file(
                abs_path, user_metadata=user_metadata, job_id=job_id
            ):
                registered += 1
        except Exception:
            logging.exception("Failed to register output: %s", abs_path)
    return registered


def ingest_existing_file(
    abs_path: str,
    user_metadata: UserMetadata = None,
    extra_tags: Sequence[str] = (),
    owner_id: str = "",
    job_id: str | None = None,
) -> bool:
    """Register an existing on-disk file as an asset stub.

    If a reference already exists for this path, updates mtime_ns, job_id,
    size_bytes, and resets enrichment so the enricher will re-hash it.

    For brand-new paths, inserts a stub record (hash=NULL) for immediate
    UX visibility.

    Returns True if a row was inserted or updated, False otherwise.
    """
    locator = os.path.abspath(abs_path)
    size_bytes, mtime_ns = get_size_and_mtime_ns(abs_path)
    mime_type = mimetypes.guess_type(abs_path, strict=False)[0]
    name, path_tags = get_name_and_tags_from_asset_path(abs_path)
    tags = list(dict.fromkeys(path_tags + list(extra_tags)))

    with create_session() as session:
        existing_ref = get_reference_by_file_path(session, locator)
        if existing_ref is not None:
            now = get_utc_now()
            existing_ref.mtime_ns = mtime_ns
            existing_ref.job_id = job_id
            existing_ref.is_missing = False
            existing_ref.deleted_at = None
            existing_ref.updated_at = now
            existing_ref.enrichment_level = 0

            asset = existing_ref.asset
            if asset:
                # If other refs share this asset, detach to a new stub
                # instead of mutating the shared row.
                siblings = count_active_siblings(session, asset.id, existing_ref.id)
                if siblings > 0:
                    new_asset = create_stub_asset(
                        session,
                        size_bytes=size_bytes,
                        mime_type=mime_type or asset.mime_type,
                    )
                    existing_ref.asset_id = new_asset.id
                else:
                    asset.hash = None
                    asset.size_bytes = size_bytes
                    if mime_type:
                        asset.mime_type = mime_type
            session.commit()
            return True

        spec = {
            "abs_path": abs_path,
            "size_bytes": size_bytes,
            "mtime_ns": mtime_ns,
            "info_name": name,
            "tags": tags,
            "fname": os.path.basename(abs_path),
            "metadata": None,
            "hash": None,
            "mime_type": mime_type,
            "job_id": job_id,
        }
        if tags:
            ensure_tags_exist(session, tags)
        result = batch_insert_seed_assets(session, [spec], owner_id=owner_id)
        session.commit()
        return result.won_paths > 0


def _register_existing_asset(
    asset_hash: str,
    name: str,
    user_metadata: UserMetadata = None,
    tags: list[str] | None = None,
    tag_origin: str = "manual",
    owner_id: str = "",
    mime_type: str | None = None,
    preview_id: str | None = None,
) -> RegisterAssetResult:
    user_metadata = user_metadata or {}

    with create_session() as session:
        asset = get_asset_by_hash(session, asset_hash=asset_hash)
        if not asset:
            raise ValueError(f"No asset with hash {asset_hash}")

        if mime_type and not asset.mime_type:
            update_asset_hash_and_mime(session, asset_id=asset.id, mime_type=mime_type)

        if preview_id:
            if not reference_exists(session, preview_id):
                preview_id = None

        ref, ref_created = get_or_create_reference(
            session,
            asset_id=asset.id,
            owner_id=owner_id,
            name=name,
            preview_id=preview_id,
        )

        if not ref_created:
            if preview_id and ref.preview_id != preview_id:
                ref.preview_id = preview_id

            tag_names = get_reference_tags(session, reference_id=ref.id)
            result = RegisterAssetResult(
                ref=extract_reference_data(ref),
                asset=extract_asset_data(asset),
                tags=tag_names,
                created=False,
            )
            session.commit()
            return result

        new_meta = dict(user_metadata)
        computed_filename = compute_relative_filename(ref.file_path) if ref.file_path else None
        if computed_filename:
            new_meta["filename"] = computed_filename

        if new_meta:
            set_reference_metadata(
                session,
                reference_id=ref.id,
                user_metadata=new_meta,
            )

        _backfill_image_dimensions_from_siblings(
            session,
            asset_id=asset.id,
            new_reference_id=ref.id,
            current_system_metadata=ref.system_metadata,
        )

        if tags is not None:
            set_reference_tags(
                session,
                reference_id=ref.id,
                tags=tags,
                origin=tag_origin,
            )

        tag_names = get_reference_tags(session, reference_id=ref.id)
        session.refresh(ref)
        result = RegisterAssetResult(
            ref=extract_reference_data(ref),
            asset=extract_asset_data(asset),
            tags=tag_names,
            created=True,
        )
        session.commit()

        return result



def _update_metadata_with_filename(
    session: Session,
    reference_id: str,
    file_path: str | None,
    current_metadata: dict | None,
    user_metadata: dict[str, Any],
) -> None:
    computed_filename = compute_relative_filename(file_path) if file_path else None

    current_meta = current_metadata or {}
    new_meta = dict(current_meta)
    for k, v in user_metadata.items():
        new_meta[k] = v
    if computed_filename:
        new_meta["filename"] = computed_filename

    if new_meta != current_meta:
        set_reference_metadata(
            session,
            reference_id=reference_id,
            user_metadata=new_meta,
        )


_IMAGE_DIMENSION_KEYS = ("kind", "width", "height")


def _maybe_store_image_dimensions(
    session: Session,
    reference_id: str,
    file_path: str,
    mime_type: str | None,
    current_system_metadata: dict | None,
) -> None:
    """Populate ``kind``/``width``/``height`` on system_metadata for image refs.

    Non-image MIME types are a no-op. Pre-existing keys (e.g. enricher-written
    safetensors metadata, download provenance) are preserved by merge.
    """
    if not mime_type or not mime_type.startswith("image/"):
        return

    dims = extract_image_dimensions(file_path, mime_type=mime_type)
    if not dims:
        return

    current = current_system_metadata or {}
    merged = dict(current)
    merged.update(dims)
    if merged != current:
        set_reference_system_metadata(
            session,
            reference_id=reference_id,
            system_metadata=merged,
        )


def _backfill_image_dimensions_from_siblings(
    session: Session,
    asset_id: str,
    new_reference_id: str,
    current_system_metadata: dict | None,
) -> None:
    """Copy image dimension keys from any sibling reference of the same asset.

    The from-hash path doesn't read the file bytes, so dimensions can't be
    extracted there directly. When another reference of the same asset already
    carries image dimensions, copy them onto the new reference so consumers
    see consistent metadata regardless of how the asset was registered.

    Best-effort: missing siblings, non-image siblings, or absent dimension
    keys leave the target reference unchanged.
    """
    current = current_system_metadata or {}
    if current.get("kind") == "image" and "width" in current and "height" in current:
        return

    for sibling in list_references_by_asset_id(session, asset_id):
        if sibling.id == new_reference_id:
            continue
        meta = sibling.system_metadata or {}
        if meta.get("kind") != "image":
            continue
        width = meta.get("width")
        height = meta.get("height")
        if (
            type(width) is not int
            or type(height) is not int
            or width <= 0
            or height <= 0
        ):
            continue
        merged = dict(current)
        merged["kind"] = "image"
        merged["width"] = width
        merged["height"] = height
        if merged != current:
            set_reference_system_metadata(
                session,
                reference_id=new_reference_id,
                system_metadata=merged,
            )
        return


def _sanitize_filename(name: str | None, fallback: str) -> str:
    n = os.path.basename((name or "").strip() or fallback)
    return n if n else fallback


class HashMismatchError(Exception):
    pass


class DependencyMissingError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


def upload_from_temp_path(
    temp_path: str,
    name: str | None = None,
    tags: list[str] | None = None,
    user_metadata: dict | None = None,
    client_filename: str | None = None,
    owner_id: str = "",
    expected_hash: str | None = None,
    mime_type: str | None = None,
    preview_id: str | None = None,
) -> UploadResult:
    try:
        digest, _ = hashing.compute_blake3_hash(temp_path)
    except ImportError as e:
        raise DependencyMissingError(str(e))
    except Exception as e:
        raise RuntimeError(f"failed to hash uploaded file: {e}")
    asset_hash = "blake3:" + digest

    if expected_hash and asset_hash != expected_hash.strip().lower():
        raise HashMismatchError("Uploaded file hash does not match provided hash.")

    with create_session() as session:
        existing = get_asset_by_hash(session, asset_hash=asset_hash)

    if existing is not None:
        with contextlib.suppress(Exception):
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

        display_name = _sanitize_filename(name or client_filename, fallback=digest)
        result = _register_existing_asset(
            asset_hash=asset_hash,
            name=display_name,
            user_metadata=user_metadata or {},
            tags=tags or [],
            tag_origin="manual",
            owner_id=owner_id,
            mime_type=mime_type,
            preview_id=preview_id,
        )
        return UploadResult(
            ref=result.ref,
            asset=result.asset,
            tags=result.tags,
            created_new=False,
        )

    if not tags:
        raise ValueError("tags are required for new asset uploads")
    base_dir, subdirs = resolve_destination_from_tags(tags)
    dest_dir = os.path.join(base_dir, *subdirs) if subdirs else base_dir
    os.makedirs(dest_dir, exist_ok=True)

    src_for_ext = (client_filename or name or "").strip()
    _ext = os.path.splitext(os.path.basename(src_for_ext))[1] if src_for_ext else ""
    ext = _ext if 0 < len(_ext) <= 16 else ""
    hashed_basename = f"{digest}{ext}"
    dest_abs = os.path.abspath(os.path.join(dest_dir, hashed_basename))
    validate_path_within_base(dest_abs, base_dir)

    content_type = mime_type or (
        mimetypes.guess_type(os.path.basename(src_for_ext), strict=False)[0]
        or mimetypes.guess_type(hashed_basename, strict=False)[0]
        or "application/octet-stream"
    )

    try:
        os.replace(temp_path, dest_abs)
    except Exception as e:
        raise RuntimeError(f"failed to move uploaded file into place: {e}")

    try:
        size_bytes, mtime_ns = get_size_and_mtime_ns(dest_abs)
    except OSError as e:
        raise RuntimeError(f"failed to stat destination file: {e}")

    ingest_result = _ingest_file_from_path(
        asset_hash=asset_hash,
        abs_path=dest_abs,
        size_bytes=size_bytes,
        mtime_ns=mtime_ns,
        mime_type=content_type,
        info_name=_sanitize_filename(name or client_filename, fallback=digest),
        owner_id=owner_id,
        preview_id=preview_id,
        user_metadata=user_metadata or {},
        tags=tags,
        tag_origin="manual",
        require_existing_tags=False,
    )
    reference_id = ingest_result.reference_id
    if not reference_id:
        raise RuntimeError("failed to create asset reference")

    with create_session() as session:
        pair = fetch_reference_and_asset(
            session, reference_id=reference_id, owner_id=owner_id
        )
        if not pair:
            raise RuntimeError("inconsistent DB state after ingest")
        ref, asset = pair
        tag_names = get_reference_tags(session, reference_id=ref.id)

    return UploadResult(
        ref=extract_reference_data(ref),
        asset=extract_asset_data(asset),
        tags=tag_names,
        created_new=ingest_result.asset_created,
    )


def register_file_in_place(
    abs_path: str,
    name: str,
    tags: list[str],
    owner_id: str = "",
    mime_type: str | None = None,
) -> UploadResult:
    """Register an already-saved file in the asset database without moving it.

    Tags are derived from the filesystem path (root category + subfolder names),
    merged with any caller-provided tags, matching the behavior of the scanner.
    If the path is not under a known root, only the caller-provided tags are used.
    """
    try:
        _, path_tags = get_name_and_tags_from_asset_path(abs_path)
    except ValueError:
        path_tags = []
    merged_tags = normalize_tags([*path_tags, *tags])

    try:
        digest, _ = hashing.compute_blake3_hash(abs_path)
    except ImportError as e:
        raise DependencyMissingError(str(e))
    except Exception as e:
        raise RuntimeError(f"failed to hash file: {e}")
    asset_hash = "blake3:" + digest

    size_bytes, mtime_ns = get_size_and_mtime_ns(abs_path)
    content_type = mime_type or (
        mimetypes.guess_type(abs_path, strict=False)[0]
        or "application/octet-stream"
    )

    ingest_result = _ingest_file_from_path(
        abs_path=abs_path,
        asset_hash=asset_hash,
        size_bytes=size_bytes,
        mtime_ns=mtime_ns,
        mime_type=content_type,
        info_name=_sanitize_filename(name, fallback=digest),
        owner_id=owner_id,
        tags=merged_tags,
        tag_origin="upload",
        require_existing_tags=False,
    )
    reference_id = ingest_result.reference_id
    if not reference_id:
        raise RuntimeError("failed to create asset reference")

    with create_session() as session:
        pair = fetch_reference_and_asset(
            session, reference_id=reference_id, owner_id=owner_id
        )
        if not pair:
            raise RuntimeError("inconsistent DB state after ingest")
        ref, asset = pair
        tag_names = get_reference_tags(session, reference_id=ref.id)

    return UploadResult(
        ref=extract_reference_data(ref),
        asset=extract_asset_data(asset),
        tags=tag_names,
        created_new=ingest_result.asset_created,
    )


def create_from_hash(
    hash_str: str,
    name: str,
    tags: list[str] | None = None,
    user_metadata: dict | None = None,
    owner_id: str = "",
    mime_type: str | None = None,
    preview_id: str | None = None,
) -> UploadResult | None:
    canonical = hash_str.strip().lower()

    try:
        result = _register_existing_asset(
            asset_hash=canonical,
            name=_sanitize_filename(
                name, fallback=canonical.split(":", 1)[1] if ":" in canonical else canonical
            ),
            user_metadata=user_metadata or {},
            tags=tags or [],
            tag_origin="manual",
            owner_id=owner_id,
            mime_type=mime_type,
            preview_id=preview_id,
        )
    except ValueError:
        logging.warning("create_from_hash: no asset found for hash %s", canonical)
        return None

    return UploadResult(
        ref=result.ref,
        asset=result.asset,
        tags=result.tags,
        created_new=False,
    )
