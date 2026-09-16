import errno
from pathlib import Path

import app.assets.services.ingest as ingest


def test_move_temp_to_dest_copies_across_filesystems(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "upload.part"
    destination = tmp_path / "output" / "upload.bin"
    source.write_bytes(b"upload")

    def fail_cross_volume_move(*_args: object) -> None:
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(ingest.os, "replace", fail_cross_volume_move)

    ingest._move_temp_to_dest(str(source), str(destination))

    assert destination.read_bytes() == b"upload"
    assert not source.exists()
