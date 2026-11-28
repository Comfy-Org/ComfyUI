import errno
import os
from pathlib import Path

import pytest

from comfy_execution.output_publish import output_file


def save(path, data, **options):
    with output_file(str(path), **options) as output:
        Path(output.path).write_bytes(data)
    return path.with_name(output.filename)


def test_publishes_at_the_requested_name_and_leaves_nothing_else_behind(tmp_path):
    target = tmp_path / "ComfyUI_00001_.png"

    assert save(target, b"photo") == target

    assert target.read_bytes() == b"photo"
    assert os.listdir(tmp_path) == [target.name]


def test_the_encoder_writes_to_a_hidden_sibling_with_the_same_extension(tmp_path):
    target = tmp_path / "clip.webm"

    with output_file(str(target)) as output:
        staged = Path(output.path)
        assert staged.parent == tmp_path
        assert staged.name.startswith(".")
        assert staged.suffix == ".webm"
        assert not target.exists()
        staged.write_bytes(b"video")


def test_two_writers_that_chose_one_name_both_keep_their_file(tmp_path):
    target = tmp_path / "ComfyUI_00001_.png"

    with output_file(str(target)) as first:
        Path(first.path).write_bytes(b"first")
        assert save(target, b"second") == target
    other = target.with_name(first.filename)

    assert other != target
    assert other.name.startswith("ComfyUI_00001__") and other.suffix == ".png"
    assert target.read_bytes() == b"second"
    assert other.read_bytes() == b"first"
    assert sorted(os.listdir(tmp_path)) == sorted([target.name, other.name])


def test_an_explicit_overwrite_replaces_the_existing_file(tmp_path):
    target = tmp_path / "test.png"
    save(target, b"first")

    assert save(target, b"replacement", overwrite=True) == target

    assert target.read_bytes() == b"replacement"
    assert os.listdir(tmp_path) == [target.name]


def test_both_files_survive_on_a_volume_without_hard_links(tmp_path, monkeypatch):
    def unsupported(*args):
        raise OSError(errno.ENOTSUP, "hard links unavailable")

    monkeypatch.setattr(os, "link", unsupported)
    target = tmp_path / "test.png"

    assert save(target, b"first") == target
    other = save(target, b"second")

    assert other != target
    assert target.read_bytes() == b"first"
    assert other.read_bytes() == b"second"
    assert sorted(os.listdir(tmp_path)) == sorted([target.name, other.name])


def test_a_failed_encode_leaves_nothing_in_the_folder(tmp_path):
    with pytest.raises(RuntimeError, match="encode failed"):
        with output_file(str(tmp_path / "test.png")) as output:
            Path(output.path).write_bytes(b"partial")
            raise RuntimeError("encode failed")

    assert os.listdir(tmp_path) == []


def test_a_non_atomic_save_writes_straight_to_the_path(tmp_path):
    target = tmp_path / "preview.png"

    with output_file(str(target), atomic=False) as output:
        assert output.path == str(target)
        Path(output.path).write_bytes(b"preview")

    assert output.filename == target.name
    assert target.read_bytes() == b"preview"
