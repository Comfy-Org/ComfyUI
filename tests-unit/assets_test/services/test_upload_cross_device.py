"""Uploads whose destination is on a different volume than the temp upload, where
the rename fails with EXDEV (WinError 17 on Windows) and the bytes are copied."""

import errno
import os
import uuid
from pathlib import Path

import pytest
from sqlalchemy import select

import app.assets.mode as mode_module
import app.assets.services.ingest as ingest_module
import folder_paths
from app.assets.database.models import AssetContent
from app.assets.services.ingest import upload_from_temp_path
from app.assets.services.snapshot_hash import snapshot_hash

_CONTENT = b"cross-device upload bytes"
_OLD_MTIME_NS = 1_600_000_000_000_000_000


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
    os.utime(temp, ns=(_OLD_MTIME_NS, _OLD_MTIME_NS))
    dest = tmp_path / "input" / f"{snapshot_hash(str(temp))[0]}.png"
    yield temp, dest
    mode_module.init(None)


def _upload(temp: Path):
    return upload_from_temp_path(
        temp_path=str(temp), name="photo.png", tags=["input"], client_filename="photo.png"
    )


def _recorded_mtime_ns(session_factory, dest: Path) -> int:
    with session_factory() as session:
        return session.scalars(
            select(AssetContent.mtime_ns).where(AssetContent.path == str(dest))
        ).one()


def test_cross_device_upload_is_copied_into_place(mock_create_session, cross_device_upload):
    temp, dest = cross_device_upload

    result = _upload(temp)

    assert result.created_new is True
    assert dest.read_bytes() == _CONTENT
    assert not temp.exists()
    # The copy has its own mtime, and that is what must be recorded.
    assert dest.stat().st_mtime_ns != _OLD_MTIME_NS
    assert _recorded_mtime_ns(mock_create_session, dest) == dest.stat().st_mtime_ns


def test_cross_device_upload_keeps_identical_bytes_already_there(
    mock_create_session, cross_device_upload, monkeypatch
):
    temp, dest = cross_device_upload
    dest.parent.mkdir(parents=True)
    dest.write_bytes(_CONTENT)
    existing_mtime_ns = _OLD_MTIME_NS - 1_000_000_000
    os.utime(dest, ns=(existing_mtime_ns, existing_mtime_ns))
    copies: list = []
    monkeypatch.setattr(ingest_module.shutil, "copyfile", lambda *a: copies.append(a))

    _upload(temp)

    assert copies == []
    assert dest.stat().st_mtime_ns == existing_mtime_ns
    assert _recorded_mtime_ns(mock_create_session, dest) == existing_mtime_ns
    assert not temp.exists()


@pytest.mark.parametrize(
    "existing", [_CONTENT[::-1], _CONTENT[:5]], ids=["same-size-other-bytes", "truncated"]
)
def test_cross_device_upload_replaces_other_bytes_at_the_destination(
    mock_create_session, cross_device_upload, existing
):
    temp, dest = cross_device_upload
    dest.parent.mkdir(parents=True)
    dest.write_bytes(existing)

    _upload(temp)

    assert dest.read_bytes() == _CONTENT


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
@pytest.mark.parametrize(
    "target", [b"someone else's file", _CONTENT, None], ids=["other-bytes", "same-bytes", "dangling"]
)
def test_cross_device_upload_replaces_a_symlink(
    mock_create_session, cross_device_upload, tmp_path, target
):
    temp, dest = cross_device_upload
    dest.parent.mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere.png"
    if target is not None:
        elsewhere.write_bytes(target)
    dest.symlink_to(elsewhere)

    _upload(temp)

    if target is None:
        assert not elsewhere.exists()
    else:
        assert elsewhere.read_bytes() == target
    assert not dest.is_symlink()
    assert dest.read_bytes() == _CONTENT


@pytest.mark.parametrize("twin", [b"someone else's file!!", _CONTENT], ids=["other-bytes", "same-bytes"])
def test_cross_device_upload_replaces_a_hardlink(
    mock_create_session, cross_device_upload, tmp_path, twin
):
    temp, dest = cross_device_upload
    dest.parent.mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere.png"
    elsewhere.write_bytes(twin)
    os.link(elsewhere, dest)

    _upload(temp)

    assert elsewhere.read_bytes() == twin
    assert not os.path.samefile(elsewhere, dest)
    assert dest.read_bytes() == _CONTENT


def test_failed_cross_device_copy_leaves_no_partial_file(
    mock_create_session, cross_device_upload, monkeypatch
):
    temp, dest = cross_device_upload

    def disk_full(src, dst):
        Path(dst).write_bytes(_CONTENT[:5])
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(ingest_module.shutil, "copyfile", disk_full)

    with pytest.raises(RuntimeError, match="failed to move uploaded file into place"):
        _upload(temp)

    assert not dest.exists()
    assert not temp.exists()


def test_other_move_errors_raise_without_copying(
    mock_create_session, cross_device_upload, monkeypatch
):
    temp, _dest = cross_device_upload

    def denied(src, dst):
        raise PermissionError(errno.EACCES, "Access is denied")

    monkeypatch.setattr(ingest_module.os, "replace", denied)
    copies: list = []
    monkeypatch.setattr(ingest_module.shutil, "copyfile", lambda *a: copies.append(a))

    with pytest.raises(RuntimeError, match="failed to move uploaded file into place"):
        _upload(temp)

    assert copies == []
