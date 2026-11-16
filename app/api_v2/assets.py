"""v2 asset endpoints over the asset service, and the mapping of asset records to contract Assets and Outputs.

``upload`` reads the multipart body into a temporary file and hands it to the
service's ``upload_from_temp_path``, which hashes and places it.

Deduplication follows ``--enable-asset-hashing``. With it, an upload reuses stored
identical bytes and the hash lookups (``by-hash``, ``from-hash``) go through
``find_live_reference_by_hash``; a hash resolves through its references, so once the
last one is deleted the hash is a miss. Without it, every upload is written and the
hash lookups always miss.

``delete`` removes the reference only; the bytes stay on disk.

``content`` re-checks on every read that the file's real path lies under a root
(``contained_root``), so a registered file later swapped for an outside symlink
is a 404. It serves with ``UnconditionalFileResponse``: Range works, cache
preconditions are ignored, because the contract declares neither 304 nor 412.
"""

import asyncio
import json
import logging
import os
import re
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import NamedTuple

from aiohttp import BodyPartReader, hdrs, web
from aiohttp.http_exceptions import BadHttpMessage

import folder_paths
from app.assets import mode
from app.assets.api.download import download_headers
from app.database.db import dependencies_available
from comfy.cli_args import args

if dependencies_available():
    from sqlalchemy.exc import SQLAlchemyError

    from app.assets.api.upload import delete_temp_file_if_exists
    from app.assets.services.asset_management import delete_asset_reference, get_asset_detail, resolve_asset_for_download
    from app.assets.services.ingest import HashMismatchError, UploadUnstableError, upload_from_temp_path
    from app.assets.services.lookup import find_live_reference_by_hash
    from app.assets.services.path_utils import compute_loader_path, get_asset_category_and_relative_path, get_comfy_models_folders, resolve_destination_from_tags

URL_TTL = timedelta(hours=24)
FIELD_LIMIT = 64 * 1024
# The multipart stream is not bounded by client_max_size, so all non-file parts share one budget.
FORM_TEXT_LIMIT = 2 * FIELD_LIMIT
MAX_PARTS = 128
MAX_TAGS = 64
MAX_TAG_BYTES = 256
ROLE_TAGS = frozenset(("input", "output", "models"))
TEXT_FIELDS = ("content_type", "file_path", "expected_hash", "expires_in")
HASH_RE = re.compile(r"blake3:[0-9a-f]{64}")
EXPIRES_IN_RE = re.compile(r"[0-9]{1,7}")
# C0, DEL, C1, U+2028/9, and bidi marks, embeddings, overrides and isolates.
CONTROL_RE = re.compile("[\x00-\x1f\x7f-\x9f\u061c\u200e\u200f\u2028-\u202e\u2066-\u2069]")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details: dict | None = None, headers: dict | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.details, self.headers = status, code, message, details, headers


def _invalid(message: str) -> ApiError:
    return ApiError(422, "invalid_request", message)


class UploadForm(NamedTuple):
    fields: dict[str, str]
    tags: list[str]


def iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def asset_json(result, base: str, created_new: bool | None = None, job_metadata: dict | None = None) -> dict:
    """Contract ``Asset`` from anything with ``.ref`` (ReferenceData) and ``.asset`` (AssetData)."""
    ref, data = result.ref, result.asset
    body = {
        "id": ref.id,
        "hash": data.hash,
        "size_bytes": data.size_bytes,
        "content_type": data.mime_type or "application/octet-stream",
        "file_path": ref.loader_path,
        "created_at": iso(ref.created_at),
        "url": f"{base}/api/v2/assets/{ref.id}/content",
        "url_expires_at": iso(datetime.now(timezone.utc) + URL_TTL),
        "expires_at": None,
    }
    if created_new is not None:
        body["created_new"] = created_new
    if ref.job_id:
        body["job_id"] = ref.job_id
    if job_metadata:
        body["job_metadata"] = job_metadata
    return body


def output_json(reference_id: str, base: str, node_id: str, name: str, kind: str, job_id: str) -> dict | None:
    """Contract ``Output`` of a live asset reference, or None when there is none. Blocking."""
    try:
        result = get_asset_detail(str(uuid.UUID(reference_id)))
    except ValueError:
        return None
    if result is None:
        return None
    asset = asset_json(result, base)
    return {
        "node_id": node_id,
        "name": name,
        "type": kind,
        **{k: asset[k] for k in ("content_type", "size_bytes", "id", "hash", "url", "url_expires_at")},
        "job_id": job_id,
    }


def contained_root(path: str) -> str | None:
    """Real path of the input, output, temp or model root that ``realpath(path)`` lies under, else None."""
    roots = [folder_paths.get_input_directory(), folder_paths.get_output_directory(), folder_paths.get_temp_directory()]
    roots += [b for _name, bases, _exts in get_comfy_models_folders() for b in bases]
    return next((os.path.realpath(r) for r in roots if folder_paths.is_within_directory(r, path)), None)


def _parse_tags(raw: str) -> list[str]:
    if not raw.startswith("["):
        return [raw.strip()]
    try:
        tags = json.loads(raw)
    except ValueError:
        tags = None
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise _invalid("a tags part must be a JSON array of strings or a single tag")
    return tags


def _asset_name(file_path: str) -> str:
    """Last segment of ``file_path``, the asset name; the rest of the path is not used."""
    name = re.split(r"[/\\]", file_path)[-1]
    if not name or CONTROL_RE.search(file_path):
        raise _invalid("file_path must end in a file name and contain no control characters")
    return name


def _path_roles(file_path: str) -> list[str]:
    """Routing tags named by the leading folder of ``file_path``: ``input/``, ``output/``,
    or a model folder as ``<folder>/`` or ``models/<folder>/``; [] for a bare name or ``models/<name>``.
    Any other folder, an absolute path, or an empty, ``.`` or ``..`` segment is ApiError 422."""
    segments = re.split(r"[/\\]", file_path)
    if any(s in ("", ".", "..") for s in segments):
        raise _invalid("file_path must be relative, without empty, '.' or '..' segments")
    folders = segments[:-1]
    if folders[:1] in (["input"], ["output"]):
        return folders[:1]
    if folders[:1] == ["models"]:
        folders = folders[1:]
    if not folders:
        return []
    if folders[0] in {name for name, _bases, _exts in get_comfy_models_folders()}:
        return ["models", f"model_type:{folders[0]}"]
    raise _invalid(f"file_path folder '{folders[0]}' is not input, output or a model folder")


def _upload_tags(file_path: str, tags: list[str]) -> list[str]:
    """``tags`` plus the routing tags of ``file_path``, else plus ``input`` when no tag routes."""
    roles = _path_roles(file_path)
    if not roles:
        return tags if ROLE_TAGS.intersection(tags) else [*tags, "input"]
    stripped = [t.strip() for t in tags]
    if any((t in ROLE_TAGS or t.startswith("model_type:")) and t not in roles for t in stripped):
        raise _invalid(f"tags name a different destination than file_path ({', '.join(roles)})")
    return [*tags, *(r for r in roles if r not in tags)]


def _validate(fields: dict[str, str]) -> None:
    content_type = fields.get("content_type")
    if content_type is None or not content_type.strip() or CONTROL_RE.search(content_type):
        raise _invalid("content_type is required, non-blank and without control characters")
    if "file_path" not in fields:
        raise _invalid("file_path is required")
    _asset_name(fields["file_path"])
    if "expected_hash" in fields and not HASH_RE.fullmatch(fields["expected_hash"]):
        raise _invalid("expected_hash must be blake3:<64 lowercase hex digits>")
    expires_in = fields.get("expires_in")
    if expires_in is not None and not (EXPIRES_IN_RE.fullmatch(expires_in) and 60 <= int(expires_in) <= 604800):
        raise _invalid("expires_in must be an integer from 60 to 604800")


def _validate_tags(tags: list[str]) -> None:
    if len(tags) > MAX_TAGS:
        raise _invalid(f"at most {MAX_TAGS} tags are allowed")
    if any(not 1 <= len(t.encode("utf-8")) <= MAX_TAG_BYTES or "\x00" in t for t in tags):
        raise _invalid(f"each tag must be 1-{MAX_TAG_BYTES} UTF-8 bytes without NUL")


async def _read_text(part: BodyPartReader, budget: int) -> str:
    data = bytearray()
    while chunk := await part.read_chunk():
        data += chunk
        if len(data) > min(FIELD_LIMIT, budget):
            raise _invalid(f"field '{part.name}' exceeds {FIELD_LIMIT} bytes or the text fields together exceed {FORM_TEXT_LIMIT}")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise _invalid(f"field '{part.name}' is not valid UTF-8")


async def _stream_file(part: BodyPartReader, out) -> None:
    limit = round(args.max_upload_size * 1024 * 1024)
    size = 0
    while chunk := await part.read_chunk(1 << 16):
        size += len(chunk)
        if size > limit:
            raise ApiError(413, "payload_too_large", f"upload exceeds {limit} bytes")
        out.write(chunk)


async def _read_form(request: web.Request, staging: str) -> UploadForm:
    """Read every part in any order; ``file`` streams into the new file ``staging``."""
    fields: dict[str, str] = {}
    tags: list[str] = []
    has_file = False
    budget, count = FORM_TEXT_LIMIT, 0
    with open(staging, "xb") as out:
        try:
            async for part in await request.multipart():
                count += 1
                if count > MAX_PARTS:
                    raise _invalid(f"at most {MAX_PARTS} multipart parts are allowed")
                if not isinstance(part, BodyPartReader) or part.name is None:
                    continue
                if part.name == "file":
                    if has_file:
                        raise _invalid("only one 'file' part is allowed")
                    await _stream_file(part, out)
                    has_file = True
                    continue
                if part.name in fields:
                    raise _invalid(f"field '{part.name}' is given more than once")
                text = await _read_text(part, budget)
                budget -= len(text.encode("utf-8"))
                if part.name == "tags":
                    tags += _parse_tags(text)
                    _validate_tags(tags)
                elif part.name in TEXT_FIELDS:
                    fields[part.name] = text
        except (ValueError, BadHttpMessage) as e:
            raise _invalid(f"malformed multipart body: {e}")
    if not has_file:
        raise _invalid("missing 'file' part")
    _validate(fields)
    return UploadForm(fields, tags)


async def upload(request: web.Request, base: str) -> tuple[int, dict]:
    """``POST /api/v2/assets``: returns (status, Asset body) or raises ``ApiError``.

    The asset service hashes and places the bytes, and reuses stored identical bytes
    only with ``--enable-asset-hashing``. The leading folder of ``file_path`` routes
    them (see ``_path_roles``), else a tag naming an upload root (``input``,
    ``output``, or ``models`` with ``model_type:<folder>``), else the input root.
    """
    if request.content_type != "multipart/form-data":
        raise _invalid("expected multipart/form-data")
    staging = os.path.join(folder_paths.get_temp_directory(), "uploads", uuid.uuid4().hex, ".upload.part")
    os.makedirs(os.path.dirname(staging))
    try:
        form = await _read_form(request, staging)
        name = _asset_name(form.fields["file_path"])
        tags = _upload_tags(form.fields["file_path"], form.tags)
        # Validated up front: on tags with no destination the service would reuse identical bytes from any folder.
        resolve_destination_from_tags(tags)
        result = await asyncio.to_thread(
            upload_from_temp_path,
            temp_path=staging,
            name=name,
            client_filename=name,
            tags=tags,
            mime_type=form.fields["content_type"],
            expected_hash=form.fields.get("expected_hash"),
            reuse_content=mode.hashing_enabled(),
        )
    except HashMismatchError:
        raise ApiError(409, "hash_mismatch", "expected_hash does not match the received bytes")
    except ValueError as e:
        raise _invalid(str(e))
    except (OSError, RuntimeError, SQLAlchemyError, UploadUnstableError) as e:
        logging.exception("v2 upload: storing %s failed", staging)
        raise ApiError(500, "upstream_error", f"storing the upload failed: {type(e).__name__}")
    finally:
        delete_temp_file_if_exists(staging)
    created_new = not result.content_reused
    return 201 if created_new else 200, asset_json(result, base, created_new=created_new)


def _not_found(what: str = "asset") -> ApiError:
    return ApiError(404, "not_found", f"{what} not found")


def parse_id(raw: str) -> str:
    """Canonical reference id, or ApiError 404 ``not_found`` when ``raw`` is not a UUID."""
    try:
        return str(uuid.UUID(raw))
    except ValueError:
        raise _not_found()


def live_reference(hash: str) -> str | None:
    """Reference id serving ``hash``; a malformed hash is a miss."""
    return find_live_reference_by_hash(hash) if HASH_RE.fullmatch(hash) else None


def mintable_reference(hash: str) -> str | None:
    """``live_reference`` for the dedup fast-path (HEAD by-hash, from-hash); always a miss
    without ``--enable-asset-hashing``, so clients upload the bytes instead."""
    return live_reference(hash) if mode.hashing_enabled() else None


def mintable_detail(hash: str):
    """``get_asset_detail`` of ``mintable_reference(hash)``, or None (also when it vanished meanwhile)."""
    ref = mintable_reference(hash)
    return get_asset_detail(ref) if ref else None


def _asset_ref_value(info: dict) -> str | None:
    """The widget value ComfyUI loads a ``core/ASSET`` reference by, or None when it does not resolve.

    ``info.id`` is a reference id and never falls back to ``info.hash``; ``hash`` alone is
    the comfy-api-proxy extension. Values are what ``folder_paths.annotated_filepath`` and
    the model loaders read: ``rel``, ``rel [output]``, ``rel [temp]`` or the model loader path.
    """
    ref_id = info.get("id")
    if ref_id is None and isinstance(info.get("hash"), str):
        ref_id = live_reference(info["hash"])
    if not isinstance(ref_id, str):
        return None
    try:
        path = resolve_asset_for_download(str(uuid.UUID(ref_id)), record_access=False).abs_path
    except (ValueError, FileNotFoundError):
        return None
    if contained_root(path) is None:
        return None
    try:
        category, rel = get_asset_category_and_relative_path(path)
    except ValueError:
        return None
    if category == "models":
        return compute_loader_path(path)
    return rel + {"output": " [output]", "temp": " [temp]"}.get(category, "")


def resolve_asset_refs(node, missing: list[str]):
    """Copy of a workflow with every ``core/ASSET`` object replaced by its widget value.

    Unresolvable or malformed references stay as they are and are appended to ``missing``.
    """
    if isinstance(node, dict) and node.get("__type") == "core/ASSET":
        info = node.get("info")
        value = _asset_ref_value(info) if isinstance(info, dict) else None
        if value is None:
            label = (info.get("id") if info.get("id") is not None else info.get("hash")) if isinstance(info, dict) else None
            missing.append(str(label) if label is not None else "<malformed core/ASSET>")
            return node
        return value
    if isinstance(node, dict):
        return {k: resolve_asset_refs(v, missing) for k, v in node.items()}
    if isinstance(node, list):
        return [resolve_asset_refs(v, missing) for v in node]
    return node


async def from_hash(request: web.Request, base: str) -> dict:
    """``POST /api/v2/assets/from-hash``: the existing Asset (``created_new`` false) or ApiError.

    ``file_path`` and ``tags`` are validated but never applied, as the contract allows self-hosted.
    """
    try:
        payload = json.loads(await request.read())
    except (ValueError, web.HTTPRequestEntityTooLarge):
        raise _invalid("body must be a JSON object")
    if not isinstance(payload, dict) or not isinstance(payload.get("hash"), str):
        raise _invalid("body must be a JSON object with a string 'hash'")
    file_path = payload.get("file_path", "")
    if not isinstance(file_path, str):
        raise _invalid("file_path must be a string")
    if file_path:
        _asset_name(file_path)
    tags = payload.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise _invalid("tags must be an array of strings")
    _validate_tags(tags)
    expires_in = payload.get("expires_in", 60)
    if type(expires_in) is not int or not 60 <= expires_in <= 604800:
        raise _invalid("expires_in must be an integer from 60 to 604800")
    detail = await asyncio.to_thread(mintable_detail, payload["hash"])
    if detail is None:
        raise ApiError(404, "blob_not_found", "no stored blob has that hash")
    return asset_json(detail, base, created_new=False)


async def detail(raw_id: str, base: str, job_metadata: Callable[[str], dict | None] | None = None) -> dict:
    """``GET /api/v2/assets/{id}``. ``job_metadata(job_id)`` returns the producing job's metadata
    when the caller owns that job's record, else None; without it no ``job_metadata`` is emitted.
    """
    reference_id = parse_id(raw_id)
    try:
        result = await asyncio.to_thread(get_asset_detail, reference_id)
    except ValueError:
        result = None
    if result is None:
        raise _not_found()
    meta = job_metadata(result.ref.job_id) if job_metadata and result.ref.job_id else None
    return asset_json(result, base, job_metadata=meta)


async def delete(raw_id: str) -> None:
    """``DELETE /api/v2/assets/{id}``: removes the reference only, the bytes stay on disk."""
    reference_id = parse_id(raw_id)
    try:
        deleted = await asyncio.to_thread(delete_asset_reference, reference_id=reference_id)
    except (OSError, SQLAlchemyError) as e:
        logging.exception("v2 delete: deleting %s failed", reference_id)
        raise ApiError(500, "upstream_error", f"deleting the asset failed: {type(e).__name__}")
    if not deleted:
        raise _not_found()


PRECONDITION_HEADERS = (
    hdrs.IF_NONE_MATCH,
    hdrs.IF_MODIFIED_SINCE,
    hdrs.IF_MATCH,
    hdrs.IF_UNMODIFIED_SINCE,
    hdrs.ACCEPT_ENCODING,  # FileResponse would otherwise serve a sibling <path>.gz/.br instead of the asset bytes
)


class UnconditionalFileResponse(web.FileResponse):
    """``FileResponse`` that never answers 304 or 412: cache preconditions are dropped, Range and If-Range kept.

    Return it unprepared; aiohttp prepares it after the handler.
    """

    async def prepare(self, request):
        headers = request.headers.copy()
        for name in PRECONDITION_HEADERS:
            headers.popall(name, None)
        return await super().prepare(request.clone(headers=headers))


def _resolve_contained(reference_id: str):
    """(real path, content type, download name) of a live file under a root; ApiError 404 otherwise."""
    try:
        found = resolve_asset_for_download(reference_id, record_access=False)
    except (ValueError, FileNotFoundError):
        raise _not_found()
    path = os.path.realpath(found.abs_path)
    if contained_root(path) is None or not os.path.isfile(path):
        raise _not_found()
    return path, found.content_type, found.download_name


async def content(request: web.Request) -> web.StreamResponse:
    """``GET /api/v2/assets/{id}/content``: an unprepared ``UnconditionalFileResponse`` or ApiError."""
    path, content_type, name = await asyncio.to_thread(_resolve_contained, parse_id(request.match_info["id"]))
    content_type, headers = download_headers(content_type, name, "attachment", request.headers.get("Sec-Fetch-Dest"))
    return UnconditionalFileResponse(path, headers={hdrs.CONTENT_TYPE: content_type, **headers})
