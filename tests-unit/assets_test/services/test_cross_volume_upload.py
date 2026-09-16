import errno
from pathlib import Path

import pytest

import app.assets.services.ingest as ingest


def test_move_temp_to_dest_copies_across_filesystems(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "upload.part"
    destination = tmp_path / "output" / "upload.bin"
    source.write_bytes(b"upload")

    real_replace = ingest.os.replace

    def fail_cross_volume_move(source_path: str, destination_path: str) -> None:
        if source_path == str(source) and destination_path == str(destination):
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        real_replace(source_path, destination_path)

    monkeypatch.setattr(ingest.os, "replace", fail_cross_volume_move)

    ingest._move_temp_to_dest(str(source), str(destination))

    assert destination.read_bytes() == b"upload"
    assert not source.exists()


def test_move_temp_to_dest_keeps_final_destination_intact_when_cross_volume_copy_fails(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "upload.part"
    destination = tmp_path / "output" / "upload.bin"
    source.write_bytes(b"upload")
    destination.parent.mkdir()
    destination.write_bytes(b"existing")

    real_replace = ingest.os.replace

    def replace_across_filesystems(source_path: str, destination_path: str) -> None:
        if source_path == str(source) and destination_path == str(destination):
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        real_replace(source_path, destination_path)

    def fail_copy(source_path: str, destination_path: str) -> None:
        Path(destination_path).write_bytes(Path(source_path).read_bytes()[:2])
        raise OSError("copy failed")

    monkeypatch.setattr(ingest.os, "replace", replace_across_filesystems)
    monkeypatch.setattr(ingest.shutil, "copy2", fail_copy)

    with pytest.raises(OSError, match="copy failed"):
        ingest._move_temp_to_dest(str(source), str(destination))

    assert destination.read_bytes() == b"existing"
    assert source.read_bytes() == b"upload"
    assert not (destination.parent / ".upload.bin.tmp").exists()
