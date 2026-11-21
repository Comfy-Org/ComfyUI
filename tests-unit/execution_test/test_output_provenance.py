import json
import os

import pytest

import folder_paths
from comfy_execution.asset_enrichment import register_cached_outputs, register_executed_outputs


class _CatalogDisabled:
    enabled = False


@pytest.fixture
def storage(tmp_path, monkeypatch):
    outputs = tmp_path / "output"
    temp = tmp_path / "temp"
    outputs.mkdir()
    temp.mkdir()
    monkeypatch.setattr(folder_paths, "output_directory", str(outputs))
    monkeypatch.setattr(folder_paths, "temp_directory", str(temp))
    monkeypatch.setenv("COMFYUI_OUTPUT_JOURNAL_DIR", str(tmp_path / "journal"))
    monkeypatch.setenv("COMFYUI_OUTPUT_PRODUCER_ID", "build-a")
    return outputs, tmp_path / "journal"


def records(directory):
    return [
        json.loads(line)
        for file in sorted(directory.glob("*.jsonl"))
        for line in file.read_text().splitlines()
    ]


def entry(filename, subfolder="", folder_type="output"):
    return {"filename": filename, "subfolder": subfolder, "type": folder_type}


def emit(*entries, job_id="prompt-id"):
    return register_executed_outputs({"images": list(entries)}, job_id, _CatalogDisabled())


def matches_file(record, path):
    info = path.stat()
    return (
        record["path"] == str(path.resolve())
        and int(record["device"]) == info.st_dev & ((1 << 32) - 1 if os.name == "nt" else (1 << 64) - 1)
        and int(record["inode"]) == info.st_ino
        and int(record["size"]) == info.st_size
        and int(record["mtime_ns"]) == info.st_mtime_ns
    )


def test_records_every_emitted_output_file_with_the_catalog_disabled(storage):
    outputs, journal = storage
    first = outputs / "ComfyUI_00001_.png"
    nested = outputs / "videos" / "clip.mp4"
    nested.parent.mkdir()
    first.write_bytes(b"photo")
    nested.write_bytes(b"video")
    output_ui = {"images": [entry(first.name)], "gifs": [entry(nested.name, "videos")], "text": ["done"]}

    emitted = register_executed_outputs(output_ui, "prompt-id", _CatalogDisabled())

    assert emitted == output_ui
    photo, video = records(journal)
    assert photo["version"] == video["version"] == 1
    assert photo["producer_id"] == video["producer_id"] == "build-a"
    assert photo["job_id"] == video["job_id"] == "prompt-id"
    assert matches_file(photo, first)
    assert matches_file(video, nested)


def test_replacing_a_file_retires_the_earlier_record(storage, monkeypatch):
    outputs, journal = storage
    target = outputs / "test.png"
    target.write_bytes(b"first")
    emit(entry(target.name))
    original, = records(journal)

    monkeypatch.setenv("COMFYUI_OUTPUT_PRODUCER_ID", "build-b")
    target.write_bytes(b"another build")
    emit(entry(target.name))

    assert not matches_file(original, target)
    replacement = records(journal)[-1]
    assert replacement["producer_id"] == "build-b"
    assert matches_file(replacement, target)


def test_only_files_in_the_output_directory_are_recorded(storage, tmp_path):
    outputs, journal = storage
    (tmp_path / "temp" / "preview.png").write_bytes(b"preview")
    (tmp_path / "outside.png").write_bytes(b"outside")

    emit(
        entry("preview.png", folder_type="temp"),
        entry("outside.png", subfolder=".."),
        entry("missing.png"),
    )

    assert records(journal) == []


@pytest.mark.skipif(os.name == "nt", reason="creating a symlink needs a privilege on Windows")
def test_a_symlink_in_the_output_directory_is_not_recorded(storage, tmp_path):
    outputs, journal = storage
    elsewhere = tmp_path / "elsewhere.png"
    elsewhere.write_bytes(b"elsewhere")
    (outputs / "link.png").symlink_to(elsewhere)

    emit(entry("link.png"))

    assert records(journal) == []


def test_a_cached_replay_writes_no_record(storage):
    outputs, journal = storage
    (outputs / "test.png").write_bytes(b"photo")
    emit(entry("test.png"))

    register_cached_outputs({"meta": {}, "output": {"images": [entry("test.png")]}}, "replay", _CatalogDisabled())

    assert [record["job_id"] for record in records(journal)] == ["prompt-id"]


def test_a_journal_failure_leaves_the_result_intact_and_later_saves_recorded(storage, caplog):
    outputs, journal = storage
    (outputs / "test.png").write_bytes(b"photo")
    (outputs / "again.png").write_bytes(b"photo")
    journal.touch()

    assert emit(entry("test.png")) == {"images": [entry("test.png")]}
    assert "Could not record output provenance" in caplog.text

    journal.unlink()
    emit(entry("again.png"))
    record, = records(journal)
    assert matches_file(record, outputs / "again.png")


@pytest.mark.parametrize("variable", ["COMFYUI_OUTPUT_JOURNAL_DIR", "COMFYUI_OUTPUT_PRODUCER_ID"])
def test_nothing_is_written_unless_the_launcher_sets_both_variables(storage, monkeypatch, variable):
    outputs, journal = storage
    (outputs / "test.png").write_bytes(b"photo")
    monkeypatch.delenv(variable)

    emit(entry("test.png"))

    assert not journal.exists()


def test_a_relative_journal_directory_is_ignored(storage, monkeypatch, tmp_path):
    outputs, journal = storage
    (outputs / "test.png").write_bytes(b"photo")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COMFYUI_OUTPUT_JOURNAL_DIR", "relative-journal")

    emit(entry("test.png"))

    assert not (tmp_path / "relative-journal").exists()
