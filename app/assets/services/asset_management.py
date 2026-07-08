import contextlib
import mimetypes
import os
from datetime import timezone
from typing import Sequence

from app.assets.services.cursor import (
    CursorPayload,
    InvalidCursorError,
    decode_cursor,
    decode_cursor_int,
    decode_cursor_time,
    encode_cursor,
    encode_cursor_from_time,
)


from app.assets.database.models import Asset
from app.assets.database.queries import (
    asset_exists_by_hash,
    reference_exists_for_asset_id,
    delete_reference_by_id,
    fetch_reference_and_asset,
    soft_delete_reference_by_id,
    fetch_reference_asset_and_tags,
    get_asset_by_hash as queries_get_asset_by_hash,
    get_reference_by_id,
    get_reference_with_owner_check,
    list_references_page,
    list_all_file_paths_by_asset_id,
    list_references_by_asset_id,
    set_reference_metadata,
    set_reference_preview,
    set_reference_tags,
    update_asset_hash_and_mime,
    update_reference_access_time,
    update_reference_name,
    update_reference_updated_at,
)
from app.assets.helpers import select_best_live_path
from app.assets.services.path_utils import compute_relative_filename
from app.assets.services.schemas import (
    AssetData,
    AssetDetailResult,
    AssetSummaryData,
    DownloadResolutionResult,
    ListAssetsResult,
    UserMetadata,
    extract_asset_data,
    extract_reference_data,
)
from app.database.db import create_session


def get_asset_detail(
    reference_id: str,
    owner_id: str = "",
) -> AssetDetailResult | None:
    with create_session() as session:
        result = fetch_reference_asset_and_tags(
            session,
            reference_id=reference_id,
            owner_id=owner_id,
        )
        if not result:
            return None

        ref, asset, tags = result
        return AssetDetailResult(
            ref=extract_reference_data(ref),
            asset=extract_asset_data(asset),
            tags=tags,
        )


def update_asset_metadata(
    reference_id: str,
    name: str | None = None,
    tags: Sequence[str] | None = None,
    user_metadata: UserMetadata = None,
    tag_origin: str = "manual",
    owner_id: str = "",
    mime_type: str | None = None,
    preview_id: str | None = None,
) -> AssetDetailResult:
    with create_session() as session:
        ref = get_reference_with_owner_check(session, reference_id, owner_id)

        touched = False
        if name is not None and name != ref.name:
            update_reference_name(session, reference_id=reference_id, name=name)
            touched = True

        computed_filename = compute_relative_filename(ref.file_path) if ref.file_path else None

        new_meta: dict | None = None
        if user_metadata is not None:
            new_meta = dict(user_metadata)
        elif computed_filename:
            current_meta = ref.user_metadata or {}
            if current_meta.get("filename") != computed_filename:
                new_meta = dict(current_meta)

        if new_meta is not None:
            if computed_filename:
                new_meta["filename"] = computed_filename
            set_reference_metadata(
                session, reference_id=reference_id, user_metadata=new_meta
            )
            touched = True

        if tags is not None:
            set_reference_tags(
                session,
                reference_id=reference_id,
                tags=tags,
                origin=tag_origin,
            )
            touched = True

        if mime_type is not None:
            updated = update_asset_hash_and_mime(
                session, asset_id=ref.asset_id, mime_type=mime_type
            )
            if updated:
                touched = True

        if preview_id is not None:
            set_reference_preview(
                session,
                reference_id=reference_id,
                preview_reference_id=preview_id,
            )
            touched = True

        if touched and user_metadata is None:
            update_reference_updated_at(session, reference_id=reference_id)

        result = fetch_reference_asset_and_tags(
            session,
            reference_id=reference_id,
            owner_id=owner_id,
        )
        if not result:
            raise RuntimeError("State changed during update")

        ref, asset, tag_list = result
        detail = AssetDetailResult(
            ref=extract_reference_data(ref),
            asset=extract_asset_data(asset),
            tags=tag_list,
        )
        session.commit()

        return detail


def delete_asset_reference(
    reference_id: str,
    owner_id: str,
    delete_content_if_orphan: bool = True,
) -> bool:
    """Delete an asset reference.

    With ``delete_content_if_orphan=False`` (a soft delete), the reference is
    hidden and the underlying content is preserved. With ``True``, the content
    is also removed once it becomes orphaned.

    Note: the public DELETE /api/assets/{id} endpoint always soft-deletes
    (passes ``False``); the orphan-reclamation path is intentionally
    internal-only, retained for a future GC/admin caller.
    """
    with create_session() as session:
        if not delete_content_if_orphan:
            # Soft delete: mark the reference as deleted but keep everything
            deleted = soft_delete_reference_by_id(
                session, reference_id=reference_id, owner_id=owner_id
            )
            session.commit()
            return deleted

        ref_row = get_reference_by_id(session, reference_id=reference_id)
        asset_id = ref_row.asset_id if ref_row else None
        file_path = ref_row.file_path if ref_row else None

        deleted = delete_reference_by_id(
            session, reference_id=reference_id, owner_id=owner_id
        )
        if not deleted:
            session.commit()
            return False

        if not asset_id:
            session.commit()
            return True

        still_exists = reference_exists_for_asset_id(session, asset_id=asset_id)
        if still_exists:
            session.commit()
            return True

        # Orphaned asset - gather ALL file paths (including
        # soft-deleted / missing refs) so their on-disk files get cleaned up.
        file_paths = list_all_file_paths_by_asset_id(session, asset_id=asset_id)
        # Also include the just-deleted file path
        if file_path:
            file_paths.append(file_path)

        asset_row = session.get(Asset, asset_id)
        if asset_row is not None:
            session.delete(asset_row)

        session.commit()

        # Delete files after commit
        for p in file_paths:
            with contextlib.suppress(Exception):
                if p and os.path.isfile(p):
                    os.remove(p)

    return True


def set_asset_preview(
    reference_id: str,
    preview_reference_id: str | None = None,
    owner_id: str = "",
) -> AssetDetailResult:
    with create_session() as session:
        get_reference_with_owner_check(session, reference_id, owner_id)

        set_reference_preview(
            session,
            reference_id=reference_id,
            preview_reference_id=preview_reference_id,
        )

        result = fetch_reference_asset_and_tags(
            session, reference_id=reference_id, owner_id=owner_id
        )
        if not result:
            raise RuntimeError("State changed during preview update")

        ref, asset, tags = result
        detail = AssetDetailResult(
            ref=extract_reference_data(ref),
            asset=extract_asset_data(asset),
            tags=tags,
        )
        session.commit()

        return detail


def asset_exists(asset_hash: str) -> bool:
    with create_session() as session:
        return asset_exists_by_hash(session, asset_hash=asset_hash)


def get_asset_by_hash(asset_hash: str) -> AssetData | None:
    with create_session() as session:
        asset = queries_get_asset_by_hash(session, asset_hash=asset_hash)
        return extract_asset_data(asset)


# Sort fields that support cursor pagination. `last_access_time` is not
# in this list — it falls back to offset/limit.
_CURSOR_SORT_FIELDS = ("created_at", "updated_at", "name", "size")


def list_assets_page(
    owner_id: str = "",
    include_tags: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
    name_contains: str | None = None,
    metadata_filter: dict | None = None,
    limit: int = 20,
    offset: int = 0,
    sort: str = "created_at",
    order: str = "desc",
    after: str | None = None,
) -> ListAssetsResult:
    """List assets with optional cursor pagination.

    When ``after`` is supplied it overrides ``offset``. The cursor's sort field
    must match ``sort`` and be in the cursor-supported allowlist; mismatches
    raise InvalidCursorError so the handler can map to 400 INVALID_CURSOR.
    """
    cursor_value: object | None = None
    cursor_id: str | None = None
    # Mint next_cursor on every page where the sort is cursor-supported, not
    # only when the request itself arrived with a cursor. Otherwise a first
    # request (no `after`) returns next_cursor=None and the client can never
    # enter cursor mode.
    mint_cursor = sort in _CURSOR_SORT_FIELDS

    if after is not None:
        if sort not in _CURSOR_SORT_FIELDS:
            raise InvalidCursorError(
                f"cursor pagination is not supported for sort={sort!r}"
            )
        payload = decode_cursor(after, _CURSOR_SORT_FIELDS, expected_order=order)
        if payload.sort_field != sort:
            raise InvalidCursorError(
                f"cursor sort field {payload.sort_field!r} does not match request sort {sort!r}"
            )
        cursor_value, cursor_id = _resolve_cursor_value(payload), payload.id

    # Over-fetch by one row so we can distinguish "exactly `limit` rows total
    # remaining" from "more rows past this page" without a second query. Drop
    # the sentinel before returning.
    fetch_limit = limit + 1 if mint_cursor else limit

    with create_session() as session:
        refs, tag_map, total = list_references_page(
            session,
            owner_id=owner_id,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            name_contains=name_contains,
            metadata_filter=metadata_filter,
            limit=fetch_limit,
            offset=offset,
            sort=sort,
            order=order,
            after_cursor_value=cursor_value,
            after_cursor_id=cursor_id,
        )

        next_cursor: str | None = None
        if mint_cursor and len(refs) > limit:
            # There's at least one more row past this page — mint a cursor from
            # the last row of the page (i.e. index `limit - 1`, since we
            # over-fetched), and drop the sentinel.
            next_cursor = _encode_next_cursor(refs[limit - 1], sort, order)
            refs = refs[:limit]

        items: list[AssetSummaryData] = []
        for ref in refs:
            items.append(
                AssetSummaryData(
                    ref=extract_reference_data(ref),
                    asset=extract_asset_data(ref.asset),
                    tags=tag_map.get(ref.id, []),
                )
            )

        return ListAssetsResult(items=items, total=total, next_cursor=next_cursor)


def _resolve_cursor_value(payload: CursorPayload) -> object:
    """Map a decoded cursor payload to a column-typed Python value."""
    if payload.sort_field in ("created_at", "updated_at"):
        # DB stores naive UTC; strip tzinfo so the comparison binds against a
        # `TIMESTAMP WITHOUT TIME ZONE` column without an offset shift.
        return decode_cursor_time(payload).replace(tzinfo=None)
    if payload.sort_field == "size":
        return decode_cursor_int(payload)
    return payload.value  # name, str-typed


def _encode_next_cursor(ref, sort: str, order: str) -> str | None:
    """Mint a cursor pointing at *ref* for the given sort dimension.

    Returns None when the boundary row carries a NULL sort value (e.g. an asset
    record whose size_bytes hasn't been backfilled). Continuing pagination
    across a NULL boundary is undefined under keyset ordering — better to
    truncate cleanly here than to mint a cursor that mis-positions.
    """
    if sort == "name":
        return encode_cursor("name", ref.name, ref.id, order=order)
    if sort == "size":
        if ref.asset is None or ref.asset.size_bytes is None:
            return None
        return encode_cursor("size", str(ref.asset.size_bytes), ref.id, order=order)
    # created_at / updated_at — DB datetimes are naive UTC; attach tz before encoding.
    value = ref.created_at if sort == "created_at" else ref.updated_at
    if value is None:
        return None
    return encode_cursor_from_time(sort, value.replace(tzinfo=timezone.utc), ref.id, order=order)


def resolve_hash_to_path(
    asset_hash: str,
    owner_id: str = "",
) -> DownloadResolutionResult | None:
    """Resolve a blake3 hash to an on-disk file path.

    Only references visible to *owner_id* are considered (owner-less
    references are always visible).

    Returns a DownloadResolutionResult with abs_path, content_type, and
    download_name, or None if no asset or live path is found.
    """
    with create_session() as session:
        asset = queries_get_asset_by_hash(session, asset_hash)
        if not asset:
            return None
        refs = list_references_by_asset_id(session, asset_id=asset.id)
        visible = [
            r for r in refs
            if r.owner_id == "" or r.owner_id == owner_id
        ]
        abs_path = select_best_live_path(visible)
        if not abs_path:
            return None
        display_name = os.path.basename(abs_path)
        for ref in visible:
            if ref.file_path == abs_path and ref.name:
                display_name = ref.name
                break
        ctype = (
            asset.mime_type
            or mimetypes.guess_type(display_name)[0]
            or "application/octet-stream"
        )
    return DownloadResolutionResult(
        abs_path=abs_path,
        content_type=ctype,
        download_name=display_name,
    )


def resolve_asset_for_download(
    reference_id: str,
    owner_id: str = "",
) -> DownloadResolutionResult:
    with create_session() as session:
        pair = fetch_reference_and_asset(
            session, reference_id=reference_id, owner_id=owner_id
        )
        if not pair:
            raise ValueError(f"AssetReference {reference_id} not found")

        ref, asset = pair

        # For references with file_path, use that directly
        if ref.file_path and os.path.isfile(ref.file_path):
            abs_path = ref.file_path
        else:
            # For API-created refs without file_path, find a path from other refs
            refs = list_references_by_asset_id(session, asset_id=asset.id)
            abs_path = select_best_live_path(refs)
            if not abs_path:
                raise FileNotFoundError(
                    f"No live path for AssetReference {reference_id} "
                    f"(asset id={asset.id}, name={ref.name})"
                )

        # Capture ORM attributes before commit (commit expires loaded objects)
        ref_name = ref.name
        asset_mime = asset.mime_type

        update_reference_access_time(session, reference_id=reference_id)
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
