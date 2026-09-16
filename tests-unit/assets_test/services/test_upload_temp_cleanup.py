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
