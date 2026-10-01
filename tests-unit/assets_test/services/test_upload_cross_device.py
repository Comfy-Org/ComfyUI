"""Uploads whose destination sits on a different volume than the temp upload
(another drive letter on Windows, another mount on POSIX), where a rename fails
with EXDEV and the bytes have to be copied instead."""

import errno
import os
import shutil
import sys
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
_SOURCE_MTIME_NS = 1_600_000_000_123_456_789


@pytest.fixture
def hashing_on():
    class FakeArgs:
        enable_asset_hashing = True

    mode_module.init(FakeArgs())
    yield
    mode_module.init(None)


@pytest.fixture
def dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    temp_root = tmp_path / "temp"
    input_root = tmp_path / "input"
    temp_root.mkdir()
    input_root.mkdir()
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(temp_root))
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(input_root))
    return temp_root, input_root


def _write_temp(temp_root: Path) -> Path:
    unique_dir = temp_root / "uploads" / uuid.uuid4().hex
    unique_dir.mkdir(parents=True)
    path = unique_dir / ".upload.part"
    path.write_bytes(_CONTENT)
    os.utime(path, ns=(_SOURCE_MTIME_NS, _SOURCE_MTIME_NS))
    return path


def _dest_for(input_root: Path, temp: Path) -> Path:
    snapshot = snapshot_hash(str(temp))
    assert snapshot is not None
    return input_root / f"{snapshot[0]}.png"


def _upload(temp: Path):
    return upload_from_temp_path(
        temp_path=str(temp), name="photo.png", tags=["input"], client_filename="photo.png"
    )


def _fail_first_replace(monkeypatch: pytest.MonkeyPatch, exc: OSError, then=None) -> list:
    """Make the first os.replace (temp -> destination) raise ``exc``; later calls
    run ``then`` if given, else the real os.replace."""
    real_replace = os.replace
    calls: list = []

    def fake_replace(src, dst):
        calls.append((src, dst))
        if len(calls) == 1:
            raise exc
        if then is not None:
            return then(src, dst)
        return real_replace(src, dst)

    monkeypatch.setattr(ingest_module.os, "replace", fake_replace)
    return calls


def _exdev() -> OSError:
    return OSError(errno.EXDEV, "Invalid cross-device link")


def _assert_nothing_left(temp: Path, input_root: Path, dest: Path | None) -> None:
    assert not temp.exists()
    assert not temp.parent.exists()
    leftovers = sorted(p.name for p in input_root.iterdir())
    assert leftovers == ([dest.name] if dest is not None else [])


def test_cross_device_upload_copies_into_place_and_records_the_copys_stat(
    mock_create_session, hashing_on, dirs, monkeypatch
):
    temp_root, input_root = dirs
    temp = _write_temp(temp_root)
    dest = _dest_for(input_root, temp)
    copied_mtime_ns = _SOURCE_MTIME_NS + 7_000_000_000
    real_copy2 = shutil.copy2

    def copy2_on_coarser_clock(src, dst):
        real_copy2(src, dst)
        os.utime(dst, ns=(copied_mtime_ns, copied_mtime_ns))

    monkeypatch.setattr(ingest_module.shutil, "copy2", copy2_on_coarser_clock)
    calls = _fail_first_replace(monkeypatch, _exdev())

    result = _upload(temp)

    assert result.created_new is True
    assert dest.read_bytes() == _CONTENT
    staging_src, staging_dst = calls[1]
    assert Path(staging_dst) == dest
    assert Path(staging_src).parent == input_root
    assert Path(staging_src).name.startswith(".")
    assert Path(staging_src).name.endswith(".tmp")
    _assert_nothing_left(temp, input_root, dest)
    with mock_create_session() as session:
        content = session.scalars(select(AssetContent)).one()
        assert content.path == str(dest)
        assert content.size_bytes == len(_CONTENT)
        assert content.mtime_ns == dest.stat().st_mtime_ns == copied_mtime_ns


def test_copy_failure_leaves_no_destination_and_no_staging_file(
    mock_create_session, hashing_on, dirs, monkeypatch
):
    temp_root, input_root = dirs
    temp = _write_temp(temp_root)

    def partial_copy(src, dst):
        Path(dst).write_bytes(_CONTENT[:5])
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(ingest_module.shutil, "copy2", partial_copy)
    _fail_first_replace(monkeypatch, _exdev())

    with pytest.raises(RuntimeError, match="failed to move uploaded file into place"):
        _upload(temp)

    _assert_nothing_left(temp, input_root, None)


def test_short_copy_is_rejected_before_it_takes_the_final_name(
    mock_create_session, hashing_on, dirs, monkeypatch
):
    temp_root, input_root = dirs
    temp = _write_temp(temp_root)
    monkeypatch.setattr(
        ingest_module.shutil, "copy2", lambda src, dst: Path(dst).write_bytes(_CONTENT[:5])
    )
    _fail_first_replace(monkeypatch, _exdev())

    with pytest.raises(RuntimeError, match="failed to move uploaded file into place"):
        _upload(temp)

    _assert_nothing_left(temp, input_root, None)


def test_final_rename_failure_removes_the_staging_file(
    mock_create_session, hashing_on, dirs, monkeypatch
):
    temp_root, input_root = dirs
    temp = _write_temp(temp_root)

    def locked(src, dst):
        raise PermissionError(errno.EACCES, "The process cannot access the file")

    _fail_first_replace(monkeypatch, _exdev(), then=locked)

    with pytest.raises(RuntimeError, match="failed to move uploaded file into place"):
        _upload(temp)

    _assert_nothing_left(temp, input_root, None)


def test_other_move_errors_do_not_fall_back_to_a_copy(
    mock_create_session, hashing_on, dirs, monkeypatch
):
    temp_root, input_root = dirs
    temp = _write_temp(temp_root)
    copies: list = []
    monkeypatch.setattr(ingest_module.shutil, "copy2", lambda *a: copies.append(a))
    _fail_first_replace(monkeypatch, PermissionError(errno.EACCES, "Access is denied"))

    with pytest.raises(RuntimeError, match="failed to move uploaded file into place"):
        _upload(temp)

    assert copies == []
    _assert_nothing_left(temp, input_root, None)


def test_same_device_upload_is_a_plain_rename(
    mock_create_session, hashing_on, dirs, monkeypatch
):
    temp_root, input_root = dirs
    temp = _write_temp(temp_root)
    dest = _dest_for(input_root, temp)
    copies: list = []
    monkeypatch.setattr(ingest_module.shutil, "copy2", lambda *a: copies.append(a))

    _upload(temp)

    assert copies == []
    assert dest.read_bytes() == _CONTENT
    _assert_nothing_left(temp, input_root, dest)
    with mock_create_session() as session:
        assert session.scalars(select(AssetContent)).one().mtime_ns == _SOURCE_MTIME_NS


@pytest.mark.skipif(sys.platform != "win32", reason="Windows error mapping")
def test_windows_not_same_device_error_maps_to_exdev():
    # ERROR_NOT_SAME_DEVICE (17) is what os.replace raises across drive letters.
    assert OSError(None, "not same device", None, 17).errno == errno.EXDEV


def test_real_cross_filesystem_upload(mock_create_session, hashing_on, tmp_path, monkeypatch):
    shm = Path("/dev/shm")
    if not shm.is_dir() or not os.access(shm, os.W_OK):
        pytest.skip("no writable /dev/shm")
    if shm.stat().st_dev == tmp_path.stat().st_dev:
        pytest.skip("/dev/shm and tmp_path share a filesystem")
    temp_root = shm / f"comfy-cross-device-{uuid.uuid4().hex}"
    input_root = tmp_path / "input"
    temp_root.mkdir()
    input_root.mkdir()
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(temp_root))
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(input_root))
    try:
        temp = _write_temp(temp_root)
        dest = _dest_for(input_root, temp)

        _upload(temp)

        assert dest.read_bytes() == _CONTENT
        _assert_nothing_left(temp, input_root, dest)
        with mock_create_session() as session:
            assert session.scalars(select(AssetContent)).one().mtime_ns == (
                dest.stat().st_mtime_ns
            )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
