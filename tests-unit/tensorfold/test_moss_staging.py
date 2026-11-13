import errno
import json
import os

import pytest

from comfy.tensorfold.moss_checkpoint import CONFIG, Checkpoint, _identity


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "moss.safetensors"
    path.write_bytes(b"checkpoint staging fixture")
    return Checkpoint(path, _identity(path.stat()), "unused")


@pytest.mark.parametrize("code", [errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP, errno.EMLINK])
def test_readable_checkpoint_stages_when_hard_links_are_unavailable(checkpoint, tmp_path, monkeypatch, code):
    directory = tmp_path / "transformer"
    target = directory / "diffusion_pytorch_model.safetensors"

    def refused(source, destination):
        assert source == checkpoint.path and destination == target
        raise OSError(code, "Hard link unavailable")

    monkeypatch.setattr(os, "link", refused)
    checkpoint.stage(directory)
    assert target.is_symlink()
    assert target.read_bytes() == checkpoint.path.read_bytes()
    assert json.loads((directory / "config.json").read_text()) == CONFIG
    checkpoint.verify()


@pytest.mark.parametrize("code", [errno.EACCES, errno.EIO, errno.ENOSPC])
def test_staging_preserves_other_filesystem_errors(checkpoint, tmp_path, monkeypatch, code):
    error = OSError(code, "Filesystem failure")

    def refused(*args):
        raise error

    monkeypatch.setattr(os, "link", refused)
    with pytest.raises(OSError) as caught:
        checkpoint.stage(tmp_path / "transformer")
    assert caught.value is error
