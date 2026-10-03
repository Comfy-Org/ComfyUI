"""Uploads whose destination is on a different volume than the temp upload, where
the rename fails with EXDEV (WinError 17 on Windows) and the bytes are copied."""

import errno
import os
import uuid
from pathlib import Path

import pytest

import app.assets.mode as mode_module
import app.assets.services.ingest as ingest_module
import folder_paths
from app.assets.services.ingest import upload_from_temp_path
from app.assets.services.snapshot_hash import snapshot_hash

_CONTENT = b"cross-device upload bytes"


@pytest.fixture
def cross_device_upload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class FakeArgs:
        enable_asset_hashing = True

    mode_module.init(FakeArgs())
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path / "temp"))
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(tmp_path / "input"))

    def exdev(src, dst):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(ingest_module.os, "replace", exdev)
    temp = tmp_path / "temp" / "uploads" / uuid.uuid4().hex / ".upload.part"
    temp.parent.mkdir(parents=True)
    temp.write_bytes(_CONTENT)
    dest = tmp_path / "input" / f"{snapshot_hash(str(temp))[0]}.png"
    yield temp, dest
    mode_module.init(None)


def _upload(temp: Path):
    return upload_from_temp_path(
        temp_path=str(temp), name="photo.png", tags=["input"], client_filename="photo.png"
    )


def test_cross_device_upload_is_copied_into_place(mock_create_session, cross_device_upload):
    temp, dest = cross_device_upload

    result = _upload(temp)

    assert result.created_new is True
    assert dest.read_bytes() == _CONTENT
    assert not temp.exists()


def test_cross_device_upload_keeps_a_same_size_file_already_there(
    mock_create_session, cross_device_upload, monkeypatch
):
    temp, dest = cross_device_upload
    dest.parent.mkdir(parents=True)
    dest.write_bytes(_CONTENT)
    os.utime(dest, ns=(1_600_000_000_000_000_000,) * 2)
    copies: list = []
    monkeypatch.setattr(ingest_module.shutil, "copyfile", lambda *a: copies.append(a))

    _upload(temp)

    assert copies == []
    assert dest.stat().st_mtime_ns == 1_600_000_000_000_000_000
    assert not temp.exists()
