import errno
import threading
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


def test_move_temp_to_dest_uses_isolated_cross_volume_copy_temps(tmp_path: Path, monkeypatch) -> None:
    first_source = tmp_path / "first-upload.part"
    second_source = tmp_path / "second-upload.part"
    destination = tmp_path / "output" / "upload.bin"
    first_source.write_bytes(b"AAAAAAAA")
    second_source.write_bytes(b"BBBBBBBB")
    complete_contents = {first_source.read_bytes(), second_source.read_bytes()}

    real_replace = ingest.os.replace
    copy_order_lock = threading.Lock()
    first_copy_started = threading.Event()
    second_copy_started = threading.Event()
    second_copy_published = threading.Event()
    first_copy = True
    errors = []

    def replace_across_filesystems(source_path: str, destination_path: str) -> None:
        if destination_path == str(destination) and source_path in {
            str(first_source),
            str(second_source),
        }:
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        real_replace(source_path, destination_path)
        if destination_path == str(destination) and second_copy_started.is_set():
            second_copy_published.set()

    def copy_with_overlap(source_path: str, destination_path: str) -> None:
        nonlocal first_copy
        with copy_order_lock:
            is_first_copy = first_copy
            first_copy = False

        source_contents = Path(source_path).read_bytes()
        if is_first_copy:
            with open(destination_path, "wb") as destination_file:
                midpoint = len(source_contents) // 2
                destination_file.write(source_contents[:midpoint])
                destination_file.flush()
                first_copy_started.set()
                assert second_copy_started.wait(timeout=5)
                assert second_copy_published.wait(timeout=5)
                destination_file.write(source_contents[midpoint:])
                destination_file.flush()
            return

        assert first_copy_started.wait(timeout=5)
        Path(destination_path).write_bytes(source_contents)
        second_copy_started.set()

    def move(source: Path) -> None:
        try:
            ingest._move_temp_to_dest(str(source), str(destination))
        except Exception as error:
            errors.append(error)

    monkeypatch.setattr(ingest.os, "replace", replace_across_filesystems)
    monkeypatch.setattr(ingest.shutil, "copy2", copy_with_overlap)

    first_worker = threading.Thread(target=move, args=(first_source,))
    second_worker = threading.Thread(target=move, args=(second_source,))
    first_worker.start()
    assert first_copy_started.wait(timeout=5)
    second_worker.start()
    first_worker.join(timeout=5)
    second_worker.join(timeout=5)

    assert not first_worker.is_alive()
    assert not second_worker.is_alive()
    assert destination.read_bytes() in complete_contents
    assert not errors
    assert not list(destination.parent.glob(f".{destination.name}*.tmp"))


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
    assert not list(destination.parent.glob(f".{destination.name}*.tmp"))
