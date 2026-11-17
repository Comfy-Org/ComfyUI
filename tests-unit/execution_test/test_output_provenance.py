import errno
import json
import os
from pathlib import Path

import pytest

import folder_paths
from comfy_execution import output_provenance
from comfy_execution.utils import CurrentNodeContext


@pytest.fixture
def storage(tmp_path, monkeypatch):
    outputs = tmp_path / "output"
    outputs.mkdir()
    monkeypatch.setattr(folder_paths, "output_directory", str(outputs))
    monkeypatch.setenv("COMFYUI_OUTPUT_JOURNAL_DIR", str(tmp_path / "journal"))
    monkeypatch.setenv("COMFYUI_OUTPUT_PRODUCER_ID", "build-a")
    return outputs, tmp_path / "journal"


def records(directory):
    return [
        json.loads(line)
        for file in directory.glob("*.jsonl")
        for line in file.read_text().splitlines()
    ]


def save(path, data, **options):
    with CurrentNodeContext("prompt-id", "save"):
        with output_provenance.output_file(str(path), **options) as output:
            Path(output.path).write_bytes(data)
    return path.with_name(output.filename)


def matches_file(record, path):
    info = path.stat()
    return (
        int(record["inode"]) == info.st_ino
        and int(record["size"]) == info.st_size
        and int(record["mtime_ns"]) == info.st_mtime_ns
    )


def test_records_the_file_version_under_its_producer_before_publishing(storage, monkeypatch):
    outputs, journal = storage
    target = outputs / "test.png"
    original_link = os.link

    def assert_record_exists(staged, destination):
        record, = records(journal)
        assert not target.exists()
        assert record["path"] == str(target.resolve())
        assert matches_file(record, staged)
        original_link(staged, destination)

    monkeypatch.setattr(os, "link", assert_record_exists)
    assert save(target, b"photo") == target
    record, = records(journal)
    assert record["version"] == 1
    assert record["producer_id"] == "build-a"
    assert record["job_id"] == "prompt-id"
    assert matches_file(record, target)
    assert target.read_bytes() == b"photo"


def test_disambiguates_concurrent_names_and_preserves_both_files(storage):
    outputs, journal = storage
    target = outputs / "test.png"
    with CurrentNodeContext("overlapping-prompt", "save"):
        with output_provenance.output_file(str(target)) as first:
            Path(first.path).write_bytes(b"first")
            assert save(target, b"second") == target
    other = target.with_name(first.filename)

    assert other != target
    assert target.read_bytes() == b"second"
    assert other.read_bytes() == b"first"
    assert sum(
        matches_file(record, target)
        for record in records(journal)
        if record["path"] == str(target.resolve())
    ) == 1


def test_explicit_overwrite_retires_the_earlier_file_version(storage, monkeypatch):
    outputs, journal = storage
    target = outputs / "test.png"
    save(target, b"first", overwrite=True)
    original, = records(journal)

    monkeypatch.setenv("COMFYUI_OUTPUT_PRODUCER_ID", "build-b")
    save(target, b"other", overwrite=True)

    assert target.read_bytes() == b"other"
    assert not matches_file(original, target)
    replacement = records(journal)[-1]
    assert replacement["producer_id"] == "build-b"
    assert matches_file(replacement, target)


def test_file_is_saved_on_volumes_without_hard_links(storage, monkeypatch):
    outputs, journal = storage

    def unsupported(*args):
        raise OSError(errno.ENOTSUP, "hard links unavailable")

    monkeypatch.setattr(os, "link", unsupported)
    target = save(outputs / "test.png", b"file")
    assert target.read_bytes() == b"file"
    record, = records(journal)
    assert matches_file(record, target)


def test_encoder_failure_keeps_partial_bytes_out_of_outputs(storage):
    outputs, journal = storage
    with pytest.raises(RuntimeError, match="encode failed"):
        with CurrentNodeContext("prompt-id", "save"):
            with output_provenance.output_file(str(outputs / "test.png")) as output:
                Path(output.path).write_bytes(b"partial")
                raise RuntimeError("encode failed")

    assert list(outputs.iterdir()) == []
    assert records(journal) == []


def test_journal_failure_preserves_the_saved_file(storage, caplog):
    outputs, journal = storage
    journal.touch()

    file = save(outputs / "test.png", b"file")
    assert file.read_bytes() == b"file"
    assert "Could not record output provenance" in caplog.text

    journal.unlink()
    save(outputs / "again.png", b"file")
    assert len(records(journal)) == 1


def test_preserves_untracked_saves(storage, tmp_path, monkeypatch):
    outputs, journal = storage
    outside = tmp_path / "outside.png"
    assert save(outside, b"outside") == outside
    temporary = save(outputs / "preview.png", b"preview", track=False)
    assert temporary.read_bytes() == b"preview"

    outside_prompt = outputs / "outside-prompt.png"
    with output_provenance.output_file(str(outside_prompt)) as output:
        Path(output.path).write_bytes(b"outside a prompt")
    assert outside_prompt.read_bytes() == b"outside a prompt"

    monkeypatch.delenv("COMFYUI_OUTPUT_JOURNAL_DIR")
    unmanaged = save(outputs / "regular.png", b"regular")
    assert unmanaged.read_bytes() == b"regular"
    assert records(journal) == []
