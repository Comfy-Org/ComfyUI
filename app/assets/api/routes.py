import asyncio
import functools
import json
import logging
import mimetypes
import os
import urllib.parse
import uuid
from datetime import timezone
from typing import Any

from aiohttp import web
from pydantic import ValidationError

import folder_paths
from app import user_manager
from app.assets import mode
from app.assets.api import schemas_in, schemas_out
from app.assets.services import schemas
from app.assets.api.schemas_in import (
    AssetValidationError,
    UploadError,
)
from app.assets.helpers import normalize_tags, validate_blake3_hash
from app.assets.api.upload import (
    delete_temp_file_if_exists,
    parse_multipart_upload,
)
from app.assets.database.models import Asset
from app.assets.database.queries.records import (
    RecordCursorBoundary,
    RecordPageSpec,
    RecordSortField,
    RecordSortOrder,
    get_preview_file_paths_by_ids,
    list_records_page,
)
from app.assets.seeder import ScanInProgressError, asset_seeder
from app.assets.services import (
    DependencyMissingError,
    HashMismatchError,
    UploadUnstableError,
    apply_tags,
    asset_exists,
    create_from_hash,
    delete_asset_reference,
    get_asset_detail,
    get_preview_file_paths,
    list_tags,
    remove_tags,
    resolve_asset_for_download,
    update_asset_metadata,
    upload_from_temp_path,
)
from app.assets.services.path_utils import compute_asset_response_paths
from app.assets.services.cursor import (
    InvalidCursorError,
    decode_cursor,
    decode_cursor_int,
    decode_cursor_time,
    encode_cursor,
    encode_cursor_from_time,
)
from app.assets.services.tagging import list_tag_histogram
from app.database.db import create_session

ROUTES = web.RouteTableDef()
USER_MANAGER: user_manager.UserManager | None = None
_ASSETS_ENABLED = False
SYSTEM_TAGS = frozenset({"missing"})
_CURSOR_SORT_FIELDS: tuple[RecordSortField, ...] = (
    "created_at",
    "updated_at",
    "name",
    "size",
)


def _require_assets_feature_enabled(handler):
    @functools.wraps(handler)
    async def wrapper(request: web.Request) -> web.Response:
        if not _ASSETS_ENABLED:
            return _build_error_response(
                503,
                "SERVICE_DISABLED",
                "Assets system is disabled. Start the server with --enable-assets to use this feature.",
            )
        return await handler(request)

    return wrapper


# UUID regex (canonical hyphenated form, case-insensitive)
UUID_RE = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"


def get_query_dict(request: web.Request) -> dict[str, Any]:
    """Gets a dictionary of query parameters from the request.

    request.query is a MultiMapping[str], needs to be converted to a dict
    to be validated by Pydantic.
    """
    query_dict = {
        key: request.query.getall(key)
        if len(request.query.getall(key)) > 1
        else request.query.get(key)
        for key in request.query.keys()
    }
    return query_dict


# Note to any custom node developers reading this code:
# The assets system is not yet fully implemented,
# do not rely on the code in /app/assets remaining the same.


def register_assets_routes(
    app: web.Application,
    user_manager_instance: user_manager.UserManager | None = None,
) -> None:
    global USER_MANAGER, _ASSETS_ENABLED
    if user_manager_instance is not None:
        USER_MANAGER = user_manager_instance
        _ASSETS_ENABLED = True
    app.add_routes(ROUTES)


def disable_assets_routes() -> None:
    """Disable asset routes at runtime (e.g. after DB init failure)."""
    global _ASSETS_ENABLED
    _ASSETS_ENABLED = False


def _build_error_response(
    status: int, code: str, message: str, details: dict | None = None
) -> web.Response:
    return web.json_response(
        {"error": {"code": code, "message": message, "details": details or {}}},
        status=status,
    )


def _build_validation_error_response(code: str, ve: ValidationError) -> web.Response:
    errors = json.loads(ve.json())
    return _build_error_response(400, code, "Validation failed.", {"errors": errors})


def _reject_system_tags(tags: list[str]) -> None:
    for tag in tags:
        if tag in SYSTEM_TAGS:
            raise web.HTTPBadRequest(
                reason=f"Tag '{tag}' is system-managed and cannot be modified via the API"
            )


class InvalidTagFilterError(Exception):
    """Invalid combination of tag-filter query parameters."""

    def __init__(self, message: str, details: dict):
        super().__init__(message)
        self.details = details


# Caps the per-tag EXISTS fan-out; deliberately covers the legacy spellings too.
MAX_TAG_FILTER_TAGS = 100


def _resolve_tag_filters(
    q: schemas_in.ListAssetsQuery | schemas_in.TagsRefineQuery,
) -> tuple[list[str], list[str], list[str]]:
    """Resolve legacy (include/exclude) and new (all/any/none) tag-filter
    spellings into effective (all, any, none) lists.

    Combination validation applies only when the request uses at least one
    new-name parameter (non-empty after normalisation); requests using only
    the legacy names keep their historical behaviour, including degenerate
    combinations like include_tags=a&exclude_tags=a.
    """
    # model_dump, not attribute access: deprecated fields warn on every attribute read.
    legacy = q.model_dump(include={"include_tags", "exclude_tags"})
    include_tags = normalize_tags(legacy["include_tags"])
    exclude_tags = normalize_tags(legacy["exclude_tags"])
    tags_all = normalize_tags(q.tags_all)
    tags_any = normalize_tags(q.tags_any)
    tags_none = normalize_tags(q.tags_none)

    for param_name, values in (
        ("include_tags", include_tags),
        ("exclude_tags", exclude_tags),
        ("tags_all", tags_all),
        ("tags_any", tags_any),
        ("tags_none", tags_none),
    ):
        if len(values) > MAX_TAG_FILTER_TAGS:
            raise InvalidTagFilterError(
                f"'{param_name}' lists {len(values)} tags; the maximum is "
                f"{MAX_TAG_FILTER_TAGS}.",
                {
                    "parameter": param_name,
                    "count": len(values),
                    "max": MAX_TAG_FILTER_TAGS,
                },
            )

    if not (tags_all or tags_any or tags_none):
        return include_tags, [], exclude_tags

    if include_tags and tags_all:
        raise InvalidTagFilterError(
            "Cannot combine 'include_tags' with 'tags_all'; use 'tags_all'.",
            {"parameters": ["include_tags", "tags_all"]},
        )
    if exclude_tags and tags_none:
        raise InvalidTagFilterError(
            "Cannot combine 'exclude_tags' with 'tags_none'; use 'tags_none'.",
            {"parameters": ["exclude_tags", "tags_none"]},
        )

    all_param, all_list = (
        ("tags_all", tags_all) if tags_all else ("include_tags", include_tags)
    )
    none_param, none_list = (
        ("tags_none", tags_none) if tags_none else ("exclude_tags", exclude_tags)
    )

    conflicting = sorted(set(all_list) & set(none_list))
    if conflicting:
        raise InvalidTagFilterError(
            f"Query can never match: {', '.join(repr(t) for t in conflicting)} "
            f"required by '{all_param}' but rejected by '{none_param}'.",
            {"conflicting_tags": conflicting, "parameters": [all_param, none_param]},
        )

    return all_list, tags_any, none_list


def _validate_sort_field(requested: str | None) -> RecordSortField:
    if not requested:
        return "created_at"
    match requested.lower():
        case "name":
            return "name"
        case "created_at":
            return "created_at"
        case "updated_at":
            return "updated_at"
        case "size":
            return "size"
        case "last_access_time":
            return "last_access_time"
        case _:
            return "created_at"


# What a client can render from the bytes themselves; anything else needs a nominated preview.
PREVIEWABLE_MIME_PREFIXES = ("image/", "video/", "audio/", "text/")

# models is deliberately absent: /api/view has no directory type for it.
VIEWABLE_NAMESPACES = frozenset({"input", "output", "temp"})


def _has_previewable_content(asset: schemas.AssetData | None, file_path: str | None) -> bool:
    if asset is None:
        return False
    # Resolved from the path, not the caller-editable name, so a rename cannot change what previews.
    raw = asset.mime_type or mimetypes.guess_type(file_path or "")[0] or ""
    return raw.split(";", 1)[0].strip().lower().startswith(PREVIEWABLE_MIME_PREFIXES)


def _build_view_url(file_path: str | None) -> str | None:
    # /api/view is a FileResponse: byte-range seeking, no user header, no access write.
    if not file_path:
        return None
    paths = compute_asset_response_paths(file_path)
    if not paths:
        return None
    logical_path, relative_path = paths
    namespace = logical_path.split("/", 1)[0]
    if namespace not in VIEWABLE_NAMESPACES or not relative_path:
        return None

    subfolder, _, filename = relative_path.rpartition("/")
    url = f"/api/view?type={namespace}&filename={urllib.parse.quote(filename, safe='')}"
    if subfolder:
        url += f"&subfolder={urllib.parse.quote(subfolder, safe='')}"
    return url


def _resolve_preview_paths(
    results: "list[schemas.AssetDetailResult] | list[schemas.AssetSummaryData]",
) -> dict[str, str]:
    # A miss means no live preview - that is what keeps a soft-deleted one quiet.
    preview_ids = {r.ref.preview_id for r in results if r.ref.preview_id}
    return get_preview_file_paths(sorted(preview_ids))


def _build_asset_response(
    result: schemas.AssetDetailResult | schemas.UploadResult,
    preview_paths: dict[str, str],
) -> schemas_out.Asset:
    if result.ref.preview_id:
        # A nominated preview is one whatever it holds, so no media check here.
        preview_url = _build_view_url(preview_paths.get(result.ref.preview_id))
    elif result.asset is not None and result.asset.is_missing:
        preview_url = None
    elif _has_previewable_content(result.asset, result.ref.file_path):
        preview_url = _build_view_url(result.ref.file_path)
    else:
        preview_url = None
    if result.ref.file_path:
        paths = compute_asset_response_paths(result.ref.file_path)
        display_name = paths[1] if paths else None
        # In-root loader path (model category dropped): what model loaders consume.
        loader_path = result.ref.loader_path
    else:
        display_name = None
        loader_path = None
    asset_content_hash = result.asset.hash if result.asset else None
    return schemas_out.Asset(
        id=result.ref.id,
        name=result.ref.name,
        hash=asset_content_hash,
        loader_path=loader_path,
        display_name=display_name,
        size=int(result.asset.size_bytes) if result.asset else None,
        mime_type=result.asset.mime_type if result.asset else None,
        tags=result.tags,
        preview_url=preview_url,
        preview_id=result.ref.preview_id,
        user_metadata=result.ref.user_metadata or {},
        metadata=result.ref.system_metadata,
        job_id=result.ref.job_id,
        prompt_id=result.ref.job_id,  # deprecated alias of job_id, kept for compatibility
        created_at=result.ref.created_at,
        updated_at=result.ref.updated_at,
        last_access_time=result.ref.last_access_time,
    )


def _build_record_response(
    record: Asset,
    tags: list[str],
    preview_paths: dict[str, str],
) -> schemas_out.Asset:
    content = record.content
    paths = compute_asset_response_paths(content.path)
    display_name = paths[1] if paths else None
    if record.preview_id:
        preview_url = _build_view_url(preview_paths.get(record.preview_id))
    elif content.is_missing:
        preview_url = None
    else:
        mime_type = record.mime_type or mimetypes.guess_type(content.path)[0] or ""
        if mime_type.split(";", 1)[0].strip().lower().startswith(
            PREVIEWABLE_MIME_PREFIXES
        ):
            preview_url = _build_view_url(content.path)
        else:
            preview_url = None

    return schemas_out.Asset(
        id=record.id,
        name=record.name,
        hash=content.hash,
        loader_path=record.loader_path,
        display_name=display_name,
        size=content.size_bytes,
        mime_type=record.mime_type,
        tags=tags,
        preview_url=preview_url,
        preview_id=record.preview_id,
        user_metadata=record.user_metadata or {},
        metadata=record.system_metadata,
        job_id=record.job_id,
        prompt_id=record.job_id,
        created_at=record.created_at,
        updated_at=record.updated_at,
        last_access_time=record.last_access_time,
    )


def _decode_record_cursor(
    after: str | None,
    sort: RecordSortField,
    order: RecordSortOrder,
) -> RecordCursorBoundary | None:
    if after is None:
        return None
    if sort not in _CURSOR_SORT_FIELDS:
        raise InvalidCursorError(
            f"cursor pagination is not supported for sort={sort!r}"
        )
    payload = decode_cursor(
        after,
        _CURSOR_SORT_FIELDS,
        expected_order=order,
    )
    if payload.sort_field != sort:
        raise InvalidCursorError(
            f"cursor sort field {payload.sort_field!r} does not match request sort {sort!r}"
        )
    match payload.sort_field:
        case "created_at" | "updated_at":
            value = decode_cursor_time(payload).replace(tzinfo=None)
        case "size":
            value = decode_cursor_int(payload)
        case "name":
            value = payload.value
        case unsupported:
            raise InvalidCursorError(f"unsupported sort field {unsupported!r}")
    return RecordCursorBoundary(value=value, id=payload.id)


def _encode_record_cursor(
    record: Asset,
    sort: RecordSortField,
    order: RecordSortOrder,
) -> str:
    match sort:
        case "name":
            return encode_cursor("name", record.name, record.id, order=order)
        case "size":
            return encode_cursor(
                "size",
                str(record.content.size_bytes),
                record.id,
                order=order,
            )
        case "created_at":
            timestamp = record.created_at
        case "updated_at":
            timestamp = record.updated_at
        case "last_access_time":
            raise InvalidCursorError(
                "cursor pagination is not supported for sort='last_access_time'"
            )
    return encode_cursor_from_time(
        sort,
        timestamp.replace(tzinfo=timezone.utc),
        record.id,
        order=order,
    )


@ROUTES.head("/api/assets/hash/{hash}")
@_require_assets_feature_enabled
async def head_asset_by_hash(request: web.Request) -> web.Response:
    hash_str = request.match_info.get("hash", "").strip().lower()
    try:
        hash_str = validate_blake3_hash(hash_str)
    except ValueError:
        return _build_error_response(
            400, "INVALID_HASH", "hash must be like 'blake3:<hex>'"
        )
    exists = asset_exists(hash_str)
    return web.Response(status=200 if exists else 404)


@ROUTES.get("/api/assets")
@_require_assets_feature_enabled
async def list_assets_route(request: web.Request) -> web.Response:
    """
    GET request to list assets.
    """
    if "metadata_filter" in request.query:
        return _build_error_response(
            400, "UNSUPPORTED_PARAM", "metadata_filter is no longer supported"
        )

    query_dict = get_query_dict(request)
    try:
        q = schemas_in.ListAssetsQuery.model_validate(query_dict)
    except ValidationError as ve:
        return _build_validation_error_response("INVALID_QUERY", ve)

    try:
        tags_all, tags_any, tags_none = _resolve_tag_filters(q)
    except InvalidTagFilterError as e:
        return _build_error_response(400, "INVALID_TAG_FILTER", str(e), e.details)

    sort = _validate_sort_field(q.sort)
    order = q.order

    try:
        cursor_boundary = _decode_record_cursor(q.after, sort, order)
        cursor_supported = sort in _CURSOR_SORT_FIELDS
        fetch_limit = q.limit + 1 if cursor_supported else q.limit
        with create_session() as session:
            records, tag_map, total = list_records_page(
                session,
                RecordPageSpec(
                    all_tags=tuple(tags_all),
                    any_tags=tuple(tags_any),
                    none_tags=tuple(tags_none),
                    name_contains=q.name_contains,
                    limit=fetch_limit,
                    offset=q.offset,
                    sort=sort,
                    order=order,
                    after=cursor_boundary,
                ),
            )
            next_cursor = None
            if cursor_supported and len(records) > q.limit:
                records = records[: q.limit]
                next_cursor = _encode_record_cursor(records[-1], sort, order)

            preview_ids = sorted(
                {record.preview_id for record in records if record.preview_id}
            )
            preview_paths = get_preview_file_paths_by_ids(session, preview_ids)
            summaries = [
                _build_record_response(
                    record,
                    tag_map.get(record.id, []),
                    preview_paths,
                )
                for record in records
            ]
    except InvalidCursorError as e:
        return _build_error_response(400, "INVALID_CURSOR", str(e))

    if q.after is not None:
        has_more = next_cursor is not None
    else:
        has_more = q.offset + len(summaries) < total

    payload = schemas_out.AssetsList(
        assets=summaries,
        total=total,
        has_more=has_more,
        next_cursor=next_cursor,
    )
    return web.json_response(payload.model_dump(mode="json", exclude_none=True))


@ROUTES.get(f"/api/assets/{{id:{UUID_RE}}}")
@_require_assets_feature_enabled
async def get_asset_route(request: web.Request) -> web.Response:
    """
    GET request to get an asset's info as JSON.
    """
    reference_id = str(uuid.UUID(request.match_info["id"]))
    try:
        result = get_asset_detail(
            reference_id=reference_id,
        )
        if not result:
            return _build_error_response(
                404,
                "ASSET_NOT_FOUND",
                f"AssetReference {reference_id} not found",
                {"id": reference_id},
            )

        payload = _build_asset_response(result, _resolve_preview_paths([result]))
    except ValueError as e:
        return _build_error_response(
            404, "ASSET_NOT_FOUND", str(e), {"id": reference_id}
        )
    except Exception:
        logging.exception(
            "get_asset failed for reference_id=%s, tenant_id=%s",
            reference_id,
            USER_MANAGER.get_request_user_id(request),
        )
        return _build_error_response(500, "INTERNAL", "Unexpected server error.")
    return web.json_response(payload.model_dump(mode="json", exclude_none=True), status=200)


@ROUTES.get(f"/api/assets/{{id:{UUID_RE}}}/content")
@_require_assets_feature_enabled
async def download_asset_content(request: web.Request) -> web.Response:
    disposition = request.query.get("disposition", "attachment").lower().strip()
    if disposition not in {"inline", "attachment"}:
        disposition = "attachment"

    try:
        result = resolve_asset_for_download(
            reference_id=str(uuid.UUID(request.match_info["id"])),
        )
        abs_path = result.abs_path
        content_type = result.content_type
        filename = result.download_name
    except ValueError as ve:
        return _build_error_response(404, "ASSET_NOT_FOUND", str(ve))
    except NotImplementedError as nie:
        return _build_error_response(501, "BACKEND_UNSUPPORTED", str(nie))
    except FileNotFoundError:
        return _build_error_response(
            404, "FILE_NOT_FOUND", "Underlying file not found on disk."
        )

    # User-controlled asset content must not render inline in the app origin
    # (stored XSS via SVG/HTML/XML). Force dangerous types to download and
    # override any requested inline disposition; SVG loaded into an <img> is
    # exempt, see renders_safely_as_image. Centralised through folder_paths so
    # this can't drift from /view and /userdata (the previous inline set here
    # omitted image/svg+xml and missed the charset/casing/+xml-dialect bypasses).
    extra_headers = {}
    sec_fetch_dest = request.headers.get("Sec-Fetch-Dest")
    if folder_paths.is_dangerous_content_type(content_type):
        # This response now depends on a request header, so it must not be
        # reused across destinations by a browser or intermediary cache: an
        # inline SVG primed by an <img> fetch and replayed to a document
        # navigation of the same URL would re-enable the stored XSS.
        extra_headers["Vary"] = "Sec-Fetch-Dest"
        extra_headers["Cache-Control"] = "no-store"
        if not folder_paths.renders_safely_as_image(content_type, sec_fetch_dest):
            content_type = "application/octet-stream"
            disposition = "attachment"

    # mime_type is uploader-supplied and unvalidated, so it can carry
    # parameters. aiohttp rejects a charset in the content_type argument with
    # ValueError, which would turn a valid inline SVG into a 500.
    content_type = content_type.split(";", 1)[0].strip() or "application/octet-stream"

    safe_name = (filename or "").replace("\r", "").replace("\n", "")
    encoded = urllib.parse.quote(safe_name)
    cd = f"{disposition}; filename*=UTF-8''{encoded}"

    file_size = os.path.getsize(abs_path)
    size_mb = file_size / (1024 * 1024)
    logging.info(
        "download_asset_content: path=%s, size=%d bytes (%.2f MB), type=%s, name=%s",
        abs_path,
        file_size,
        size_mb,
        content_type,
        filename,
    )

    async def stream_file_chunks():
        chunk_size = 64 * 1024
        with open(abs_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    return web.Response(
        body=stream_file_chunks(),
        content_type=content_type,
        headers={
            "Content-Disposition": cd,
            "Content-Length": str(file_size),
            "X-Content-Type-Options": "nosniff",
            **extra_headers,
        },
    )


@ROUTES.post("/api/assets/from-hash")
@_require_assets_feature_enabled
async def create_asset_from_hash_route(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
        body = schemas_in.CreateFromHashBody.model_validate(payload)
    except ValidationError as ve:
        return _build_validation_error_response("INVALID_BODY", ve)
    except Exception:
        return _build_error_response(
            400, "INVALID_JSON", "Request body must be valid JSON."
        )

    # Derive name from hash if not provided
    name = body.name
    if name is None:
        name = body.hash.split(":", 1)[1] if ":" in body.hash else body.hash

    if not mode.hashing_enabled():
        return _build_error_response(
            400, "FEATURE_DISABLED", "Asset hashing is disabled."
        )

    try:
        result = create_from_hash(
            hash_str=body.hash,
            name=name,
            tags=body.tags,
            user_metadata=body.user_metadata,
            mime_type=body.mime_type,
            preview_id=body.preview_id,
        )
    except ValueError as e:
        return _build_error_response(400, "INVALID_BODY", str(e))
    if result is None:
        return _build_error_response(
            404, "ASSET_NOT_FOUND", f"Asset content {body.hash} does not exist"
        )

    asset = _build_asset_response(result, _resolve_preview_paths([result]))
    payload_out = schemas_out.AssetCreated(
        **asset.model_dump(),
        created_new=result.created_new,
    )
    return web.json_response(payload_out.model_dump(mode="json", exclude_none=True), status=201)


@ROUTES.post("/api/assets")
@_require_assets_feature_enabled
async def upload_asset(request: web.Request) -> web.Response:
    """Multipart/form-data endpoint for Asset uploads."""
    try:
        parsed = await parse_multipart_upload(request, check_hash_exists=asset_exists)
    except UploadError as e:
        return _build_error_response(e.status, e.code, e.message)

    tenant_id = USER_MANAGER.get_request_user_id(request)

    try:
        spec = schemas_in.UploadAssetSpec.model_validate(
            {
                "tags": parsed.tags_raw,
                "name": parsed.provided_name,
                "user_metadata": parsed.user_metadata_raw,
                "hash": parsed.provided_hash,
                "mime_type": parsed.provided_mime_type,
                "preview_id": parsed.provided_preview_id,
            }
        )
    except ValidationError as ve:
        delete_temp_file_if_exists(parsed.tmp_path)
        return _build_error_response(
            400, "INVALID_BODY", f"Validation failed: {ve.json()}"
        )

    try:
        if not parsed.file_present and spec.hash:
            result = create_from_hash(
                hash_str=spec.hash,
                name=spec.name or (spec.hash.split(":", 1)[1]),
                tags=spec.tags,
                user_metadata=spec.user_metadata or {},
                mime_type=spec.mime_type,
                preview_id=spec.preview_id,
            )
            if result is None:
                return _build_error_response(
                    404, "ASSET_NOT_FOUND", f"Asset content {spec.hash} does not exist"
                )
        elif parsed.tmp_path and os.path.exists(parsed.tmp_path):
            result = upload_from_temp_path(
                temp_path=parsed.tmp_path,
                name=spec.name,
                tags=spec.tags,
                user_metadata=spec.user_metadata or {},
                client_filename=parsed.file_client_name,
                expected_hash=spec.hash,
                mime_type=spec.mime_type,
                preview_id=spec.preview_id,
            )
        else:
            return _build_error_response(
                400,
                "MISSING_INPUT",
                "Provided hash not found and no file uploaded.",
            )
    except AssetValidationError as e:
        delete_temp_file_if_exists(parsed.tmp_path)
        return _build_error_response(400, e.code, str(e))
    except ValueError as e:
        delete_temp_file_if_exists(parsed.tmp_path)
        return _build_error_response(400, "INVALID_BODY", str(e))
    except HashMismatchError as e:
        delete_temp_file_if_exists(parsed.tmp_path)
        return _build_error_response(400, "HASH_MISMATCH", str(e))
    except UploadUnstableError as e:
        delete_temp_file_if_exists(parsed.tmp_path)
        return _build_error_response(500, "UPLOAD_UNSTABLE", str(e))
    except DependencyMissingError as e:
        delete_temp_file_if_exists(parsed.tmp_path)
        return _build_error_response(503, "DEPENDENCY_MISSING", e.message)
    except Exception:
        delete_temp_file_if_exists(parsed.tmp_path)
        logging.exception("upload_asset failed for tenant_id=%s", tenant_id)
        return _build_error_response(500, "INTERNAL", "Unexpected server error.")

    asset = _build_asset_response(result, _resolve_preview_paths([result]))
    payload_out = schemas_out.AssetCreated(
        **asset.model_dump(),
        created_new=result.created_new,
    )
    status = 201 if result.created_new else 200
    return web.json_response(payload_out.model_dump(mode="json", exclude_none=True), status=status)


@ROUTES.put(f"/api/assets/{{id:{UUID_RE}}}")
@_require_assets_feature_enabled
async def update_asset_route(request: web.Request) -> web.Response:
    reference_id = str(uuid.UUID(request.match_info["id"]))
    try:
        body = schemas_in.UpdateAssetBody.model_validate(await request.json())
    except ValidationError as ve:
        return _build_validation_error_response("INVALID_BODY", ve)
    except Exception:
        return _build_error_response(
            400, "INVALID_JSON", "Request body must be valid JSON."
        )

    try:
        result = update_asset_metadata(
            reference_id=reference_id,
            name=body.name,
            user_metadata=body.user_metadata,
            preview_id=body.preview_id,
        )
        payload = _build_asset_response(result, _resolve_preview_paths([result]))
    except PermissionError as pe:
        return _build_error_response(403, "FORBIDDEN", str(pe), {"id": reference_id})
    except ValueError as ve:
        return _build_error_response(
            404, "ASSET_NOT_FOUND", str(ve), {"id": reference_id}
        )
    except Exception:
        logging.exception(
            "update_asset failed for reference_id=%s, tenant_id=%s",
            reference_id,
            USER_MANAGER.get_request_user_id(request),
        )
        return _build_error_response(500, "INTERNAL", "Unexpected server error.")
    return web.json_response(payload.model_dump(mode="json", exclude_none=True), status=200)


@ROUTES.delete(f"/api/assets/{{id:{UUID_RE}}}")
@_require_assets_feature_enabled
async def delete_asset_route(request: web.Request) -> web.Response:
    reference_id = str(uuid.UUID(request.match_info["id"]))

    try:
        deleted = delete_asset_reference(
            reference_id=reference_id,
        )
    except Exception:
        logging.exception(
            "delete_asset_reference failed for reference_id=%s, tenant_id=%s",
            reference_id,
            USER_MANAGER.get_request_user_id(request),
        )
        return _build_error_response(500, "INTERNAL", "Unexpected server error.")

    if not deleted:
        return _build_error_response(
            404, "ASSET_NOT_FOUND", f"AssetReference {reference_id} not found."
        )
    return web.Response(status=204)


@ROUTES.get("/api/tags")
@_require_assets_feature_enabled
async def get_tags(request: web.Request) -> web.Response:
    """
    GET request to list all tags based on query parameters.
    """
    query_map = dict(request.rel_url.query)

    try:
        query = schemas_in.TagsListQuery.model_validate(query_map)
    except ValidationError as e:
        return _build_error_response(
            400,
            "INVALID_QUERY",
            "Invalid query parameters",
            {"errors": json.loads(e.json())},
        )

    rows, total = list_tags(
        prefix=query.prefix,
        limit=query.limit,
        offset=query.offset,
        order=query.order,
        include_zero=query.include_zero,
    )

    tags = [
        schemas_out.TagUsage(name=name, count=count)
        for (name, count) in rows
    ]
    payload = schemas_out.TagsList(
        tags=tags, total=total, has_more=(query.offset + len(tags)) < total
    )
    return web.json_response(payload.model_dump(mode="json", exclude_none=True))


@ROUTES.post(f"/api/assets/{{id:{UUID_RE}}}/tags")
@_require_assets_feature_enabled
async def add_asset_tags(request: web.Request) -> web.Response:
    reference_id = str(uuid.UUID(request.match_info["id"]))
    try:
        json_payload = await request.json()
        data = schemas_in.TagsAdd.model_validate(json_payload)
    except ValidationError as ve:
        return _build_error_response(
            400,
            "INVALID_BODY",
            "Invalid JSON body for tags add.",
            {"errors": ve.errors()},
        )
    except Exception:
        return _build_error_response(
            400, "INVALID_JSON", "Request body must be valid JSON."
        )

    _reject_system_tags(data.tags)

    try:
        result = apply_tags(
            reference_id=reference_id,
            tags=data.tags,
            origin="manual",
        )
        payload = schemas_out.TagsAdd(
            added=result.added,
            already_present=result.already_present,
            total_tags=result.total_tags,
        )
    except PermissionError as pe:
        return _build_error_response(403, "FORBIDDEN", str(pe), {"id": reference_id})
    except ValueError as ve:
        return _build_error_response(
            404, "ASSET_NOT_FOUND", str(ve), {"id": reference_id}
        )
    except Exception:
        logging.exception(
            "add_tags_to_asset failed for reference_id=%s, tenant_id=%s",
            reference_id,
            USER_MANAGER.get_request_user_id(request),
        )
        return _build_error_response(500, "INTERNAL", "Unexpected server error.")

    return web.json_response(payload.model_dump(mode="json", exclude_none=True), status=200)


@ROUTES.delete(f"/api/assets/{{id:{UUID_RE}}}/tags")
@_require_assets_feature_enabled
async def delete_asset_tags(request: web.Request) -> web.Response:
    reference_id = str(uuid.UUID(request.match_info["id"]))
    try:
        json_payload = await request.json()
        data = schemas_in.TagsRemove.model_validate(json_payload)
    except ValidationError as ve:
        return _build_error_response(
            400,
            "INVALID_BODY",
            "Invalid JSON body for tags remove.",
            {"errors": ve.errors()},
        )
    except Exception:
        return _build_error_response(
            400, "INVALID_JSON", "Request body must be valid JSON."
        )

    _reject_system_tags(data.tags)

    try:
        result = remove_tags(
            reference_id=reference_id,
            tags=data.tags,
        )
        payload = schemas_out.TagsRemove(
            removed=result.removed,
            not_present=result.not_present,
            total_tags=result.total_tags,
            protected=result.protected,
        )
    except PermissionError as pe:
        return _build_error_response(403, "FORBIDDEN", str(pe), {"id": reference_id})
    except ValueError as ve:
        return _build_error_response(
            404, "ASSET_NOT_FOUND", str(ve), {"id": reference_id}
        )
    except Exception:
        logging.exception(
            "remove_tags_from_asset failed for reference_id=%s, tenant_id=%s",
            reference_id,
            USER_MANAGER.get_request_user_id(request),
        )
        return _build_error_response(500, "INTERNAL", "Unexpected server error.")

    return web.json_response(payload.model_dump(mode="json", exclude_none=True), status=200)


@ROUTES.get("/api/assets/tags/refine")
@_require_assets_feature_enabled
async def get_tags_refine(request: web.Request) -> web.Response:
    """GET request to get tag histogram for filtered assets."""
    if "metadata_filter" in request.query:
        return _build_error_response(
            400, "UNSUPPORTED_PARAM", "metadata_filter is no longer supported"
        )

    query_dict = get_query_dict(request)
    try:
        q = schemas_in.TagsRefineQuery.model_validate(query_dict)
    except ValidationError as ve:
        return _build_validation_error_response("INVALID_QUERY", ve)

    try:
        tags_all, tags_any, tags_none = _resolve_tag_filters(q)
    except InvalidTagFilterError as e:
        return _build_error_response(400, "INVALID_TAG_FILTER", str(e), e.details)

    tag_counts = list_tag_histogram(
        include_tags=tags_all,
        exclude_tags=tags_none,
        any_tags=tags_any,
        name_contains=q.name_contains,
        limit=q.limit,
    )
    payload = schemas_out.TagHistogram(tag_counts=tag_counts)
    return web.json_response(payload.model_dump(mode="json", exclude_none=True), status=200)


@ROUTES.post("/api/assets/seed")
@_require_assets_feature_enabled
async def seed_assets(request: web.Request) -> web.Response:
    """Trigger asset seeding for specified roots (models, input, output).

    Query params:
        wait: If "true", block until scan completes (synchronous behavior for tests)

    Returns:
        202 Accepted if scan started
        409 Conflict if scan already running
        200 OK with final stats if wait=true
    """
    try:
        payload = await request.json()
        roots = payload.get("roots", ["models", "input", "output"])
    except Exception:
        roots = ["models", "input", "output"]

    valid_roots = tuple(r for r in roots if r in ("models", "input", "output"))
    if not valid_roots:
        return _build_error_response(400, "INVALID_BODY", "No valid roots specified")

    wait_param = request.query.get("wait", "").lower()
    should_wait = wait_param in ("true", "1", "yes")

    started = asset_seeder.start(
        roots=valid_roots, compute_hashes=mode.hashing_enabled()
    )
    if not started:
        return web.json_response({"status": "already_running"}, status=409)

    if should_wait:
        await asyncio.to_thread(asset_seeder.wait)
        status = asset_seeder.get_status()
        return web.json_response(
            {
                "status": "completed",
                "progress": {
                    "scanned": status.progress.scanned if status.progress else 0,
                    "total": status.progress.total if status.progress else 0,
                    "created": status.progress.created if status.progress else 0,
                    "skipped": status.progress.skipped if status.progress else 0,
                },
                "errors": status.errors,
            },
            status=200,
        )

    return web.json_response({"status": "started"}, status=202)


@ROUTES.get("/api/assets/seed/status")
@_require_assets_feature_enabled
async def get_seed_status(request: web.Request) -> web.Response:
    """Get current scan status and progress."""
    status = asset_seeder.get_status()
    return web.json_response(
        {
            "state": status.state.value,
            "progress": {
                "scanned": status.progress.scanned,
                "total": status.progress.total,
                "created": status.progress.created,
                "skipped": status.progress.skipped,
            }
            if status.progress
            else None,
            "errors": status.errors,
        },
        status=200,
    )


@ROUTES.post("/api/assets/seed/cancel")
@_require_assets_feature_enabled
async def cancel_seed(request: web.Request) -> web.Response:
    """Request cancellation of in-progress scan."""
    cancelled = asset_seeder.cancel()
    if cancelled:
        return web.json_response({"status": "cancelling"}, status=200)
    return web.json_response({"status": "idle"}, status=200)


@ROUTES.post("/api/assets/prune")
@_require_assets_feature_enabled
async def mark_missing_assets(request: web.Request) -> web.Response:
    """Mark assets as missing when outside all known root prefixes.

    This is a non-destructive soft-delete operation. Assets and metadata
    are preserved, but references are flagged as missing. They can be
    restored if the file reappears in a future scan.

    Returns:
        200 OK with count of marked assets
        409 Conflict if a scan is currently running
    """
    try:
        marked = asset_seeder.mark_missing_outside_prefixes()
    except ScanInProgressError:
        return web.json_response(
            {"status": "scan_running", "marked": 0},
            status=409,
        )
    return web.json_response({"status": "completed", "marked": marked}, status=200)
