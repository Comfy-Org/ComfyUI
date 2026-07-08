import logging
import os
import uuid
from typing import Callable

from aiohttp import web

import folder_paths
from app.assets.api.schemas_in import ParsedUpload, UploadError
from app.assets.helpers import validate_blake3_hash


def normalize_and_validate_hash(s: str) -> str:
    """Validate and normalize a hash string.

    Returns canonical 'blake3:<hex>' or raises UploadError.
    """
    try:
        return validate_blake3_hash(s)
    except ValueError:
        raise UploadError(400, "INVALID_HASH", "hash must be like 'blake3:<hex>'")


async def parse_multipart_upload(
    request: web.Request,
    check_hash_exists: Callable[[str], bool],
) -> ParsedUpload:
    """
    Parse a multipart/form-data upload request.

    Args:
        request: The aiohttp request
        check_hash_exists: Callable(hash_str) -> bool to check if a hash exists

    Returns:
        ParsedUpload with parsed fields and temp file path

    Raises:
        UploadError: On validation or I/O errors
    """
    if not (request.content_type or "").lower().startswith("multipart/"):
        raise UploadError(
            415, "UNSUPPORTED_MEDIA_TYPE", "Use multipart/form-data for uploads."
        )

    reader = await request.multipart()

    file_present = False
    file_client_name: str | None = None
    tags_raw: list[str] = []
    provided_name: str | None = None
    user_metadata_raw: str | None = None
    provided_hash: str | None = None
    provided_hash_exists: bool | None = None
    provided_mime_type: str | None = None
    provided_preview_id: str | None = None

    file_written = 0
    tmp_path: str | None = None

    while True:
        field = await reader.next()
        if field is None:
            break

        fname = getattr(field, "name", "") or ""

        if fname == "hash":
            try:
                s = ((await field.text()) or "").strip().lower()
            except Exception:
                raise UploadError(
                    400, "INVALID_HASH", "hash must be like 'blake3:<hex>'"
                )

            if s:
                provided_hash = normalize_and_validate_hash(s)
                try:
                    provided_hash_exists = check_hash_exists(provided_hash)
                except Exception as e:
                    logging.exception(
                        "check_hash_exists failed for hash=%s: %s", provided_hash, e
                    )
                    raise UploadError(
                        500,
                        "HASH_CHECK_FAILED",
                        "Backend error while checking asset hash.",
                    )

        elif fname == "file":
            file_present = True
            file_client_name = (field.filename or "").strip()

            if provided_hash and provided_hash_exists is True:
                # Hash exists - drain file but don't write to disk
                try:
                    while True:
                        chunk = await field.read_chunk(8 * 1024 * 1024)
                        if not chunk:
                            break
                        file_written += len(chunk)
                except Exception:
                    raise UploadError(
                        500, "UPLOAD_IO_ERROR", "Failed to receive uploaded file."
                    )
                continue

            uploads_root = os.path.join(folder_paths.get_temp_directory(), "uploads")
            unique_dir = os.path.join(uploads_root, uuid.uuid4().hex)
            os.makedirs(unique_dir, exist_ok=True)
            tmp_path = os.path.join(unique_dir, ".upload.part")

            try:
                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = await field.read_chunk(8 * 1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        file_written += len(chunk)
            except Exception:
                delete_temp_file_if_exists(tmp_path)
                raise UploadError(
                    500, "UPLOAD_IO_ERROR", "Failed to receive and store uploaded file."
                )

        elif fname == "tags":
            tags_raw.append((await field.text()) or "")
        elif fname == "name":
            provided_name = (await field.text()) or None
        elif fname == "user_metadata":
            user_metadata_raw = (await field.text()) or None
        elif fname == "id":
            raise UploadError(
                400,
                "UNSUPPORTED_FIELD",
                "Client-provided 'id' is not supported. Asset IDs are assigned by the server.",
            )
        elif fname == "mime_type":
            provided_mime_type = ((await field.text()) or "").strip() or None
        elif fname == "preview_id":
            provided_preview_id = ((await field.text()) or "").strip() or None

    if not file_present and not (provided_hash and provided_hash_exists):
        raise UploadError(
            400, "MISSING_FILE", "Form must include a 'file' part or a known 'hash'."
        )

    if (
        file_present
        and file_written == 0
        and not (provided_hash and provided_hash_exists)
    ):
        delete_temp_file_if_exists(tmp_path)
        raise UploadError(400, "EMPTY_UPLOAD", "Uploaded file is empty.")

    return ParsedUpload(
        file_present=file_present,
        file_written=file_written,
        file_client_name=file_client_name,
        tmp_path=tmp_path,
        tags_raw=tags_raw,
        provided_name=provided_name,
        user_metadata_raw=user_metadata_raw,
        provided_hash=provided_hash,
        provided_hash_exists=provided_hash_exists,
        provided_mime_type=provided_mime_type,
        provided_preview_id=provided_preview_id,
    )


def delete_temp_file_if_exists(tmp_path: str | None) -> None:
    """Safely remove a temp file and its parent directory if empty."""
    if tmp_path:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError as e:
            logging.debug("Failed to delete temp file %s: %s", tmp_path, e)
        try:
            parent = os.path.dirname(tmp_path)
            if parent and os.path.isdir(parent):
                os.rmdir(parent)  # only succeeds if empty
        except OSError:
            pass
