import copy
import errno
import json
import math
import os
from pathlib import Path
import struct
import sys

import pytest

from comfy.tensorfold.moss_checkpoint import CONFIG, read_checkpoint, tensor_schema


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="The native MOSS checkpoint fixtures use Linux sparse files")


def checkpoint_header():
    header = {}
    offset = 0
    for name, (dtype, shape) in tensor_schema().items():
        length = math.prod(shape) * (4 if dtype == "F32" else 2)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + length]}
        offset += length
    return header, offset


def write_checkpoint(path, header=None):
    defaults, data_size = checkpoint_header()
    data = json.dumps(defaults if header is None else header, separators=(",", ":")).encode()
    data += b" " * (-len(data) % 8)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(data)))
        stream.write(data)
        stream.truncate(8 + len(data) + data_size)
    return path


@pytest.fixture
def checkpoint(tmp_path):
    # Sparse payloads exercise complete metadata without allocating or reading the model's weights.
    return write_checkpoint(tmp_path / "moss.safetensors")


def test_complete_native_schema_and_zero_copy_stage(checkpoint, tmp_path):
    schema = tensor_schema()
    assert len(schema) == 463
    assert schema["layers.41.proj.ffn_up_gate.weight"] == ("BF16", [16384, 3072])
    assert schema["text_embedder.blocks.1.qkv.weight"] == ("BF16", [9216, 3072])
    assert schema["rope.bands"] == ("F32", [16])
    before = checkpoint.stat()
    model = read_checkpoint(checkpoint)
    folder = tmp_path / "transformer"
    model.stage(folder)
    linked = folder / "diffusion_pytorch_model.safetensors"
    assert linked.stat().st_ino == before.st_ino
    assert linked.stat().st_size == before.st_size
    assert json.loads((folder / "config.json").read_text()) == CONFIG
    linked.unlink()
    model.verify()
    assert checkpoint.is_file()


def test_cross_filesystem_stage_uses_owned_symlink(checkpoint, tmp_path, monkeypatch):
    def different_filesystem(*args):
        raise OSError(errno.EXDEV, "cross-device link")
    monkeypatch.setattr(os, "link", different_filesystem)
    model = read_checkpoint(checkpoint)
    folder = tmp_path / "transformer"
    model.stage(folder)
    assert (folder / "diffusion_pytorch_model.safetensors").resolve() == checkpoint
    assert (folder / "diffusion_pytorch_model.safetensors").is_symlink()


@pytest.mark.parametrize("change,match", [
    ("missing", "original MOSS"), ("extra", "original MOSS"),
    ("dtype", "must be BF16"), ("control", "must be BF16"),
    ("offset", "overlaps"), ("extent", "extent"),
    ("metadata", "metadata"), ("rope", "must be F32"),
])
def test_checkpoint_admission_catches_native_boundary_failures(tmp_path, change, match):
    header, _ = checkpoint_header()
    header = copy.deepcopy(header)
    if change == "missing":
        header.pop("layers.41.proj.q.weight")
    elif change == "extra":
        header["control.weight"] = header["rope.bands"]
    elif change == "dtype":
        header["layers.0.proj.q.weight"]["dtype"] = "F16"
    elif change == "control":
        header["video_embedder.weight"]["shape"] = [3072, 256]
    elif change == "offset":
        header["layers.0.proj.k.weight"]["data_offsets"] = header["layers.0.proj.q.weight"]["data_offsets"]
    elif change == "extent":
        header["rope.bands"]["data_offsets"][1] -= 4
    elif change == "metadata":
        header["__metadata__"] = {"source": False}
    elif change == "rope":
        header["rope.bands"]["dtype"] = "BF16"
    with pytest.raises(ValueError, match=match):
        read_checkpoint(write_checkpoint(tmp_path / "bad.safetensors", header))


def test_missing_payload_is_rejected(checkpoint):
    with checkpoint.open("r+b") as stream:
        stream.truncate(checkpoint.stat().st_size - 2)
    with pytest.raises(ValueError, match="truncated"):
        read_checkpoint(checkpoint)


def test_replaced_checkpoint_is_refused(checkpoint, tmp_path):
    model = read_checkpoint(checkpoint)
    replacement = write_checkpoint(tmp_path / "replacement.safetensors")
    replacement.replace(checkpoint)
    with pytest.raises(RuntimeError, match="changed"):
        model.verify()


@pytest.mark.parametrize("data", [b"", struct.pack("<Q", 99), struct.pack("<Q", 2) + b"[]"])
def test_invalid_or_truncated_header(tmp_path, data):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        read_checkpoint(path)


def test_duplicate_json_fields_are_rejected(tmp_path):
    data = b'{"rope.bands":{},"rope.bands":{}}'
    path = tmp_path / "duplicate.safetensors"
    path.write_bytes(struct.pack("<Q", len(data)) + data)
    with pytest.raises(ValueError, match="Duplicate"):
        read_checkpoint(path)


def test_directory_is_rejected_before_opening(tmp_path):
    with pytest.raises(ValueError, match="regular"):
        read_checkpoint(Path(tmp_path))


def test_native_bf16_host_views_require_even_payload_alignment(tmp_path):
    header, data_size = checkpoint_header()
    data = json.dumps(header, separators=(",", ":")).encode()
    data += b" " * (1 - len(data) % 2)
    path = tmp_path / "unaligned.safetensors"
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(data)) + data)
        stream.truncate(8 + len(data) + data_size)
    with pytest.raises(ValueError, match="align to two bytes"):
        read_checkpoint(path)


def test_fused_comfy_checkpoint_explains_offline_preparation(tmp_path):
    header, _ = checkpoint_header()
    parts = [header.pop(f"layers.0.proj.{part}.weight") for part in ("q", "k", "v")]
    header["layers.0.proj.qkv.weight"] = {"dtype": "BF16", "shape": [9216, 3072],
                                           "data_offsets": [parts[0]["data_offsets"][0], parts[-1]["data_offsets"][1]]}
    with pytest.raises(ValueError, match="prepare_moss.py"):
        read_checkpoint(write_checkpoint(tmp_path / "fused.safetensors", header))
