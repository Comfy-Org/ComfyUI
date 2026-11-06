import json
import os
import struct
import tempfile

import pytest
import safetensors.torch
import torch

import comfy.utils


@pytest.fixture
def safetensors_file():
    tensors = {"weight": torch.arange(12, dtype=torch.float32).reshape(3, 4)}
    with tempfile.TemporaryDirectory() as tmpdirname:
        path = os.path.join(tmpdirname, "model.safetensors")
        safetensors.torch.save_file(tensors, path, metadata={"format": "pt"})
        yield path, tensors


def test_disable_mmap_does_not_use_safe_open(safetensors_file, monkeypatch):
    path, tensors = safetensors_file
    monkeypatch.setattr(comfy.utils, "DISABLE_MMAP", True)

    def boom(*args, **kwargs):
        raise AssertionError("safetensors.safe_open should not be used when DISABLE_MMAP is set")

    monkeypatch.setattr(comfy.utils.safetensors, "safe_open", boom)

    sd, metadata = comfy.utils.load_torch_file(path, device=torch.device("cpu"), return_metadata=True)

    assert torch.equal(sd["weight"], tensors["weight"])
    assert metadata == {"format": "pt"}


def test_load_safetensors_no_mmap_rejects_corrupt_data_offsets(tmp_path):
    # A corrupt header claiming a non-empty shape with start == end must be
    # rejected instead of silently producing an uninitialized tensor.
    header = {
        "weight": {"dtype": "F32", "shape": [3, 4], "data_offsets": [0, 0]},
    }
    header_bytes = json.dumps(header).encode("utf-8")
    path = tmp_path / "corrupt.safetensors"
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)

    with pytest.raises(ValueError):
        comfy.utils.load_safetensors_no_mmap(str(path), torch.device("cpu"))


def _write_safetensors_with_raw_offsets(path, header, data):
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(data)


def test_load_safetensors_no_mmap_rejects_gap_between_tensors(tmp_path):
    # weight2 starts one byte after weight ends, leaving an unindexed gap.
    header = {
        "weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "weight2": {"dtype": "F32", "shape": [1], "data_offsets": [5, 9]},
    }
    path = tmp_path / "gap.safetensors"
    _write_safetensors_with_raw_offsets(path, header, b"\x00" * 9)

    with pytest.raises(ValueError):
        comfy.utils.load_safetensors_no_mmap(str(path), torch.device("cpu"))


def test_load_safetensors_no_mmap_rejects_overlapping_tensors(tmp_path):
    # weight2 starts before weight ends, so the ranges overlap.
    header = {
        "weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "weight2": {"dtype": "F32", "shape": [1], "data_offsets": [2, 6]},
    }
    path = tmp_path / "overlap.safetensors"
    _write_safetensors_with_raw_offsets(path, header, b"\x00" * 6)

    with pytest.raises(ValueError):
        comfy.utils.load_safetensors_no_mmap(str(path), torch.device("cpu"))


def test_load_safetensors_no_mmap_rejects_trailing_bytes(tmp_path):
    # The declared data region ends before the end of the file.
    header = {
        "weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
    }
    path = tmp_path / "trailing.safetensors"
    _write_safetensors_with_raw_offsets(path, header, b"\x00" * 8)

    with pytest.raises(ValueError):
        comfy.utils.load_safetensors_no_mmap(str(path), torch.device("cpu"))
