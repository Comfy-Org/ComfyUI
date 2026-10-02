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
from app.assets.services.ingest import UploadUnstableError, upload_from_temp_path
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
    temp.chmod(0o640)
    dest = _dest_for(input_root, temp)
    calls = _fail_first_replace(monkeypatch, _exdev())

    result = _upload(temp)

    assert result.created_new is True
    assert dest.read_bytes() == _CONTENT
    staging_src, staging_dst = calls[1]
    assert Path(staging_dst) == dest
    assert Path(staging_src).parent == input_root
    assert Path(staging_src).name.startswith(".")
    assert Path(staging_src).name.endswith(".tmp")
    if sys.platform != "win32":
        assert dest.stat().st_mode & 0o777 == 0o640
    _assert_nothing_left(temp, input_root, dest)
    with mock_create_session() as session:
        content = session.scalars(select(AssetContent)).one()
        assert content.path == str(dest)
        assert content.size_bytes == len(_CONTENT)
        assert content.mtime_ns == dest.stat().st_mtime_ns != _SOURCE_MTIME_NS


def test_destination_that_rejects_mode_bits_still_accepts_the_upload(
    mock_create_session, hashing_on, dirs, monkeypatch
):
    temp_root, input_root = dirs
    temp = _write_temp(temp_root)
    dest = _dest_for(input_root, temp)

    def no_chmod(src, dst):
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(ingest_module.shutil, "copymode", no_chmod)
    _fail_first_replace(monkeypatch, _exdev())

    _upload(temp)

    assert dest.read_bytes() == _CONTENT
    _assert_nothing_left(temp, input_root, dest)


def test_copy_failure_leaves_no_destination_and_no_staging_file(
    mock_create_session, hashing_on, dirs, monkeypatch
):
    temp_root, input_root = dirs
    temp = _write_temp(temp_root)

    def partial_copy(src, dst):
        dst.write(src.read(5))
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(ingest_module.shutil, "copyfileobj", partial_copy)
    _fail_first_replace(monkeypatch, _exdev())

    with pytest.raises(RuntimeError, match="failed to move uploaded file into place"):
        _upload(temp)

    _assert_nothing_left(temp, input_root, None)


def _rewrite_same_size(temp: Path) -> None:
    temp.write_bytes(_CONTENT[::-1])
    os.utime(temp, ns=(_SOURCE_MTIME_NS + 2_000_000_000,) * 2)


def test_source_changed_after_hashing_is_not_copied(
    mock_create_session, hashing_on, dirs, monkeypatch
):
    temp_root, input_root = dirs
    temp = _write_temp(temp_root)

    def rewrite_then_exdev(src, dst):
        _rewrite_same_size(temp)
        raise _exdev()

    monkeypatch.setattr(ingest_module.os, "replace", rewrite_then_exdev)

    with pytest.raises(UploadUnstableError, match="changed after hashing"):
        _upload(temp)

    _assert_nothing_left(temp, input_root, None)


def test_source_changed_during_the_copy_is_not_published(
    mock_create_session, hashing_on, dirs, monkeypatch
):
    temp_root, input_root = dirs
    temp = _write_temp(temp_root)
    real_copyfileobj = shutil.copyfileobj

    def copy_then_rewrite(src, dst):
        real_copyfileobj(src, dst)
        _rewrite_same_size(temp)

    monkeypatch.setattr(ingest_module.shutil, "copyfileobj", copy_then_rewrite)
    _fail_first_replace(monkeypatch, _exdev())

    with pytest.raises(UploadUnstableError, match="changed after hashing"):
        _upload(temp)

    _assert_nothing_left(temp, input_root, None)


def _deny_staging_writes(monkeypatch: pytest.MonkeyPatch, input_root: Path) -> list:
    """An ACL that denies writes in the destination: creating a file there raises
    PermissionError while os.access still reports the directory writable. The
    first os.replace raises EXDEV and switches os.name to "nt", where
    tempfile.mkstemp retries that PermissionError up to TMP_MAX times."""
    real_open = os.open
    attempts: list = []

    def denying_open(path, flags, mode=0o777, *, dir_fd=None):
        if os.path.dirname(path) == str(input_root):
            attempts.append(path)
            if len(attempts) > 50:
                raise RuntimeError("staging creation kept retrying")
            raise PermissionError(errno.EACCES, "Access is denied", str(path))
        return real_open(path, flags, mode, dir_fd=dir_fd)

    def exdev_as_windows(src, dst):
        monkeypatch.setattr(os, "name", "nt")
        raise _exdev()

    monkeypatch.setattr(ingest_module.os, "open", denying_open)
    monkeypatch.setattr(ingest_module.os, "replace", exdev_as_windows)
    return attempts


def test_write_denied_destination_fails_fast(
    mock_create_session, hashing_on, dirs, monkeypatch
):
    temp_root, input_root = dirs
    temp = _write_temp(temp_root)
    attempts = _deny_staging_writes(monkeypatch, input_root)

    with pytest.raises(RuntimeError, match="failed to move uploaded file into place: .*denied"):
        _upload(temp)

    monkeypatch.undo()
    assert len(attempts) == 1
    _assert_nothing_left(temp, input_root, None)


def test_staging_name_collision_takes_another_name(
    mock_create_session, hashing_on, dirs, monkeypatch
):
    temp_root, input_root = dirs
    temp = _write_temp(temp_root)
    dest = _dest_for(input_root, temp)
    taken = iter(["0" * 16, "0" * 16, "1" * 16])
    (input_root / f".{'0' * 16}.upload.tmp").write_bytes(b"someone else's")
    monkeypatch.setattr(ingest_module.secrets, "token_hex", lambda _n: next(taken))
    _fail_first_replace(monkeypatch, _exdev())

    _upload(temp)

    assert dest.read_bytes() == _CONTENT
    assert sorted(p.name for p in input_root.iterdir()) == sorted(
        [dest.name, f".{'0' * 16}.upload.tmp"]
    )


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
    monkeypatch.setattr(ingest_module.shutil, "copyfileobj", lambda *a: copies.append(a))
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
    monkeypatch.setattr(ingest_module.shutil, "copyfileobj", lambda *a: copies.append(a))

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
