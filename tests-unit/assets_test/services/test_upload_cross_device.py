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
_SOURCE_MTIME_NS = 1_600_000_000_000_000_000  # whole seconds: NTFS keeps 100 ns


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


def _cross_device(monkeypatch: pytest.MonkeyPatch, exc: OSError | None = None) -> None:
    """Make renames between directories fail as they do across volumes: os.replace
    raises ``exc`` (EXDEV by default), and so does the os.rename shutil.move tries
    first. Renames within one directory (staging file to final name) still work."""
    exc = exc or OSError(errno.EXDEV, "Invalid cross-device link")
    real_replace = os.replace

    def fail(src, dst):
        if os.path.dirname(src) == os.path.dirname(dst):
            return real_replace(src, dst)
        raise exc

    monkeypatch.setattr(ingest_module.os, "replace", fail)
    monkeypatch.setattr(ingest_module.os, "rename", fail)


def _assert_nothing_left(temp: Path, input_root: Path, dest: Path | None) -> None:
    assert not temp.exists()
    assert not temp.parent.exists()
    leftovers = sorted(p.name for p in input_root.iterdir())
    assert leftovers == ([dest.name] if dest is not None else [])


def test_cross_device_upload_is_copied_into_place(
    mock_create_session, hashing_on, dirs, monkeypatch
):
    temp_root, input_root = dirs
    temp = _write_temp(temp_root)
    dest = _dest_for(input_root, temp)
    _cross_device(monkeypatch)
    real_move = shutil.move
    copied_mtime_ns = _SOURCE_MTIME_NS + 2_000_000_000

    def move_onto_coarser_clock(src, dst):
        real_move(src, dst)
        os.utime(dst, ns=(copied_mtime_ns, copied_mtime_ns))

    monkeypatch.setattr(ingest_module.shutil, "move", move_onto_coarser_clock)

    result = _upload(temp)

    assert result.created_new is True
    assert dest.read_bytes() == _CONTENT
    _assert_nothing_left(temp, input_root, dest)
    with mock_create_session() as session:
        content = session.scalars(select(AssetContent)).one()
        assert content.path == str(dest)
        assert content.size_bytes == len(_CONTENT)
        assert content.mtime_ns == dest.stat().st_mtime_ns == copied_mtime_ns


def test_failed_cross_device_copy_leaves_no_truncated_file(
    mock_create_session, hashing_on, dirs, monkeypatch
):
    temp_root, input_root = dirs
    temp = _write_temp(temp_root)
    _cross_device(monkeypatch)

    def partial_copy(src, dst):
        Path(dst).write_bytes(_CONTENT[:5])
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(ingest_module.shutil, "move", partial_copy)

    with pytest.raises(RuntimeError, match="failed to move uploaded file into place"):
        _upload(temp)

    _assert_nothing_left(temp, input_root, None)


def test_failed_cross_device_copy_leaves_an_existing_file_untouched(
    mock_create_session, hashing_on, dirs, monkeypatch
):
    temp_root, input_root = dirs
    temp = _write_temp(temp_root)
    dest = _dest_for(input_root, temp)
    dest.write_bytes(_CONTENT)
    _cross_device(monkeypatch)

    def disk_full_mid_copy(src, dst):
        with open(dst, "wb") as partial:
            partial.write(_CONTENT[:5])
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(ingest_module.shutil, "move", disk_full_mid_copy)

    with pytest.raises(RuntimeError, match="failed to move uploaded file into place"):
        _upload(temp)

    assert dest.read_bytes() == _CONTENT
    _assert_nothing_left(temp, input_root, dest)


def test_other_move_errors_do_not_fall_back_to_a_copy(
    mock_create_session, hashing_on, dirs, monkeypatch
):
    temp_root, input_root = dirs
    temp = _write_temp(temp_root)
    moves: list = []
    monkeypatch.setattr(ingest_module.shutil, "move", lambda *a: moves.append(a))
    _cross_device(monkeypatch, PermissionError(errno.EACCES, "Access is denied"))

    with pytest.raises(RuntimeError, match="failed to move uploaded file into place"):
        _upload(temp)

    assert moves == []
    _assert_nothing_left(temp, input_root, None)


def test_same_device_upload_is_a_plain_rename(
    mock_create_session, hashing_on, dirs, monkeypatch
):
    temp_root, input_root = dirs
    temp = _write_temp(temp_root)
    dest = _dest_for(input_root, temp)
    moves: list = []
    monkeypatch.setattr(ingest_module.shutil, "move", lambda *a: moves.append(a))

    _upload(temp)

    assert moves == []
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
