"""Serves the per-asset operations behind the API: reading an asset's detail,
updating its name, tags, metadata and preview, deleting a record, and resolving
a hash to a servable path. An update moves ``updated_at`` only when it actually
changed something, so a call that requests what is already true is not recorded
as a user edit.
"""

import mimetypes
import os
from typing import Sequence

from sqlalchemy import delete, select, update

from app.assets.database.models import Asset, AssetContent, AssetTag
from app.assets.database.queries import (
    delete_record,
    fetch_record_tags,
    get_record_by_id,
    update_record_access_time,
)
from app.assets.database.queries.records import (
    bump_record_updated_at,
    ensure_tag,
    ensure_tag_link,
    get_preview_file_paths_by_ids,
    rename_record,
)
from app.assets.helpers import get_utc_now, normalize_tags, validate_blake3_hash
from app.assets.services.lookup import lookup_for_view
from app.assets.services.schemas import (
    AssetData,
    AssetDetailResult,
    DownloadResolutionResult,
    ReferenceData,
    UserMetadata,
)
from app.database.db import create_session


def _record_to_detail_result(session, record) -> AssetDetailResult:
    content = session.get(AssetContent, record.content_id)
    tags = fetch_record_tags(session, record.id)
    api_hash = content.hash if content else None
    ref = ReferenceData(
        id=record.id,
        name=record.name,
        file_path=content.path if content else None,
        loader_path=record.loader_path,
        user_metadata=record.user_metadata,
        preview_id=record.preview_id,
        system_metadata=record.system_metadata,
        job_id=record.job_id,
        created_at=record.created_at,
        updated_at=record.updated_at,
        last_access_time=record.last_access_time,
    )
    asset = AssetData(
        hash=api_hash,
        size_bytes=content.size_bytes if content else None,
        mime_type=record.mime_type,
        is_missing=bool(content and content.is_missing),
    )
    return AssetDetailResult(ref=ref, asset=asset, tags=tags)


def get_asset_detail(
    reference_id: str,
) -> AssetDetailResult | None:
    with create_session() as session:
        record = get_record_by_id(session, reference_id)
        if record is None:
            return None
        return _record_to_detail_result(session, record)


def _fetch_manual_tags(session, reference_id: str) -> set[str]:
    return set(
        session.scalars(
            select(AssetTag.tag_name).where(
                AssetTag.asset_id == reference_id,
                AssetTag.origin != "automatic",
            )
        )
    )


def update_asset_metadata(
    reference_id: str,
    name: str | None = None,
    tags: Sequence[str] | None = None,
    user_metadata: UserMetadata = None,
    tag_origin: str = "manual",
    mime_type: str | None = None,
    preview_id: str | None = None,
) -> AssetDetailResult:
    with create_session() as session:
        record = get_record_by_id(session, reference_id)
        if record is None:
            raise ValueError(f"Asset {reference_id} not found")

        if name is not None:
            rename_record(session, reference_id, name)
        if user_metadata is not None:
            session.execute(
                update(Asset)
                .where(Asset.id == reference_id)
                .values(user_metadata=dict(user_metadata), updated_at=get_utc_now())
            )
        manual_tags_before: set[str] = set()
        if tags is not None:
            manual_tags_before = _fetch_manual_tags(session, reference_id)
            session.execute(
                delete(AssetTag).where(
                    AssetTag.asset_id == reference_id,
                    AssetTag.origin != "automatic",
                )
            )
        if mime_type is not None:
            session.execute(
                update(Asset)
                .where(Asset.id == reference_id)
                .values(mime_type=mime_type, updated_at=get_utc_now())
            )
        if preview_id is not None:
            if session.get(Asset, preview_id) is None:
                raise ValueError(
                    f"preview_id {preview_id!r} does not reference an existing asset"
                )
            session.execute(
                update(Asset)
                .where(Asset.id == reference_id)
                .values(preview_id=preview_id, updated_at=get_utc_now())
            )
        if tags is not None:
            for tag_name in normalize_tags(list(tags)):
                ensure_tag(session, tag_name)
                ensure_tag_link(
                    session,
                    asset_id=reference_id,
                    tag_name=tag_name,
                    origin=tag_origin,
                )
            session.flush()
            if _fetch_manual_tags(session, reference_id) != manual_tags_before:
                bump_record_updated_at(session, reference_id)
        session.commit()

    detail = get_asset_detail(reference_id)
    if detail is None:
        raise RuntimeError("Asset deleted during update")
    return detail


def delete_asset_reference(
    reference_id: str,
) -> bool:
    with create_session() as session:
        if get_record_by_id(session, reference_id) is None:
            return False
        delete_record(session, reference_id)
        session.commit()
        return True


def asset_exists(asset_hash: str) -> bool:
    try:
        canonical = validate_blake3_hash(asset_hash)
    except ValueError:
        return False
    with create_session() as session:
        return lookup_for_view(session, canonical) is not None


def resolve_hash_to_path(
    asset_hash: str,
) -> DownloadResolutionResult | None:
    """Resolve a blake3 hash to an on-disk file path via lookup_for_view.

    Uses the first qualified live content row. Temp paths are excluded from all
    hash lookups inside qualified_content_iterator, so a hash resolving only to
    temp content returns None. Updates last_access_time on every record pointing
    at the served content.

    Filename and Content-Type both come from the newest record so they never
    describe different records. Deleting the last record preserves its content,
    so content with zero records stays servable off the content path alone.
    """
    try:
        canonical = validate_blake3_hash(asset_hash)
    except ValueError:
        return None
    with create_session() as session:
        content = lookup_for_view(session, canonical)
        if content is None:
            return None

        records = list(
            session.scalars(
                select(Asset)
                .where(Asset.content_id == content.id)
                .order_by(Asset.created_at, Asset.id)
            )
        )
        display_name = os.path.basename(content.path)
        mime_type = None
        if records:
            latest_record = records[-1]
            display_name = latest_record.name or display_name
            mime_type = latest_record.mime_type
        for record in records:
            update_record_access_time(session, record.id)
        abs_path = content.path
        session.commit()

        ctype = (
            mime_type
            or mimetypes.guess_type(display_name)[0]
            or mimetypes.guess_type(abs_path)[0]
            or "application/octet-stream"
        )
    return DownloadResolutionResult(
        abs_path=abs_path,
        content_type=ctype,
        download_name=display_name,
    )


def get_preview_file_paths(preview_ids: list[str]) -> dict[str, str]:
    """Map preview reference id -> file_path, in one query for the whole page."""
    if not preview_ids:
        return {}
    with create_session() as session:
        return get_preview_file_paths_by_ids(session, preview_ids=preview_ids)


def resolve_asset_for_download(
    reference_id: str,
) -> DownloadResolutionResult:
    with create_session() as session:
        record = get_record_by_id(session, reference_id)
        if record is None:
            raise ValueError(f"AssetReference {reference_id} not found")

        content = session.get(AssetContent, record.content_id)
        if (
            content is None
            or content.is_missing
            or not os.path.isfile(content.path)
        ):
            raise FileNotFoundError(
                f"No live content for AssetReference {reference_id} "
                f"(content id={record.content_id}, name={record.name})"
            )

        ref_name = record.name
        asset_mime = record.mime_type
        abs_path = content.path

        update_record_access_time(session, reference_id)
        session.commit()

        ctype = (
            asset_mime
            or mimetypes.guess_type(ref_name or abs_path)[0]
            or "application/octet-stream"
        )
        download_name = ref_name or os.path.basename(abs_path)
        return DownloadResolutionResult(
            abs_path=abs_path,
            content_type=ctype,
            download_name=download_name,
        )
