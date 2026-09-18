import asyncio
import uuid
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import folder_paths
from app.assets.api.schemas_in import UploadError
from app.assets.api.upload import parse_multipart_upload
from app.assets.services.ingest import upload_from_temp_path


@pytest.mark.asyncio
async def test_multipart_id_after_file_removes_temp_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path))

    file_field = AsyncMock()
    file_field.name = "file"
    file_field.filename = "model.safetensors"
    file_field.read_chunk.side_effect = [b"uploaded bytes", b""]

    id_field = AsyncMock()
    id_field.name = "id"

    reader = AsyncMock()
    reader.next.side_effect = [file_field, id_field, None]

    request = AsyncMock()
    request.content_type = "multipart/form-data"
    request.multipart.return_value = reader

    with pytest.raises(UploadError, match="Client-provided 'id' is not supported"):
        await parse_multipart_upload(request, lambda _hash: False)

    assert list((tmp_path / "uploads").iterdir()) == []


def _file_field() -> AsyncMock:
    field = AsyncMock()
    field.name = "file"
    field.filename = "model.safetensors"
    field.read_chunk.side_effect = [b"uploaded bytes", b""]
    return field


def _multipart_request(*fields: AsyncMock) -> AsyncMock:
    reader = AsyncMock()
    reader.next.side_effect = [*fields, None]
    request = AsyncMock()
    request.content_type = "multipart/form-data"
    request.multipart.return_value = reader
    return request


@pytest.mark.asyncio
async def test_duplicate_file_parts_are_rejected_without_leaking_temp_uploads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path))
    request = _multipart_request(_file_field(), _file_field())

    error: UploadError | None = None
    try:
        await parse_multipart_upload(request, lambda _hash: False)
    except UploadError as exc:
        error = exc

    remaining = list((tmp_path / "uploads").iterdir())
    assert error is not None, f"duplicate file parts accepted; temp paths remain: {remaining}"
    assert error.status == 400
    assert error.code == "UNSUPPORTED_FIELD"
    assert error.message == "Multiple 'file' parts are not supported."
    assert remaining == []


@pytest.mark.asyncio
async def test_invalid_utf8_after_file_removes_temp_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path))
    tags_field = AsyncMock()
    tags_field.name = "tags"
    tags_field.text.side_effect = UnicodeDecodeError(
        "utf-8", b"\xff", 0, 1, "invalid start byte"
    )
    request = _multipart_request(_file_field(), tags_field)

    with pytest.raises(UnicodeDecodeError):
        await parse_multipart_upload(request, lambda _hash: False)

    assert list((tmp_path / "uploads").iterdir()) == []


@pytest.mark.asyncio
async def test_malformed_framing_after_file_removes_temp_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path))
    request = _multipart_request(_file_field())
    request.multipart.return_value.next.side_effect = [
        _file_field(),
        ValueError("Reading after EOF"),
    ]

    with pytest.raises(ValueError, match="Reading after EOF"):
        await parse_multipart_upload(request, lambda _hash: False)

    assert list((tmp_path / "uploads").iterdir()) == []


@pytest.mark.asyncio
async def test_cancelled_upload_removes_temp_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path))
    request = _multipart_request(_file_field())
    request.multipart.return_value.next.side_effect = [
        _file_field(),
        asyncio.CancelledError(),
    ]

    with pytest.raises(asyncio.CancelledError):
        await parse_multipart_upload(request, lambda _hash: False)

    assert list((tmp_path / "uploads").iterdir()) == []


@pytest.mark.asyncio
async def test_successful_parse_returns_consumable_temp_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path))
    request = _multipart_request(_file_field())

    parsed = await parse_multipart_upload(request, lambda _hash: False)

    assert parsed.tmp_path is not None
    assert Path(parsed.tmp_path).read_bytes() == b"uploaded bytes"


def test_destination_resolution_failure_removes_temp_upload(
    mock_create_session, tmp_path: Path
) -> None:
    upload_dir = tmp_path / "uploads" / uuid.uuid4().hex
    upload_dir.mkdir(parents=True)
    temp_path = upload_dir / ".upload.part"
    temp_path.write_bytes(b"uploaded bytes")

    with pytest.raises(ValueError, match="exactly one destination role"):
        upload_from_temp_path(
            temp_path=str(temp_path),
            name="model.safetensors",
            tags=["not-a-destination"],
            client_filename="model.safetensors",
        )

    assert not temp_path.exists()
    assert not upload_dir.exists()


def test_successful_upload_reaps_empty_per_upload_directory(
    mock_create_session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temp_root = tmp_path / "temp"
    output_root = tmp_path / "output"
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(temp_root))
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output_root))

    upload_dir = temp_root / "uploads" / uuid.uuid4().hex
    upload_dir.mkdir(parents=True)
    temp_path = upload_dir / ".upload.part"
    temp_path.write_bytes(b"uploaded bytes")

    result = upload_from_temp_path(
        temp_path=str(temp_path),
        name="model.safetensors",
        tags=["output"],
        client_filename="model.safetensors",
    )

    assert result.created_new is True
    assert not upload_dir.exists()
