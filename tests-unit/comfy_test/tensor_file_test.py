import os
import sys

import pytest
import safetensors.torch
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.model_state  # noqa: E402
from comfy.memory_management import read_tensor_file_slice_into, write_tensor_file_slice  # noqa: E402
from comfy.utils import create_safetensors, open_safetensors, rewrite_safetensors_header  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.name != "nt" and sys.platform != "linux",
    reason="anonymous files require Linux or Windows",
)


def test_sparse_safetensors_and_write_validation():
    geometry = {"key": torch.empty((3, 5), device="meta"), "tail": torch.empty((7,), dtype=torch.uint8, device="meta")}
    state, file = create_safetensors(geometry)

    try:
        expected = torch.zeros(3, 5)
        restored = torch.empty_like(expected)
        assert read_tensor_file_slice_into(state["key"], restored)
        assert torch.equal(restored, expected)
        file.seek(0)
        saved = safetensors.torch.load(file.read())
        assert torch.equal(saved["key"], expected)
        assert torch.equal(saved["tail"], torch.zeros(7, dtype=torch.uint8))

        with pytest.raises(ValueError):
            write_tensor_file_slice(state["key"], expected[1])
        with pytest.raises(ValueError):
            write_tensor_file_slice(expected, expected)
        with pytest.raises(ValueError):
            write_tensor_file_slice(state["key"], torch.empty_like(expected, device="meta"))
    finally:
        file.close()

    large, file = create_safetensors({"kv": torch.empty((1 << 30,), dtype=torch.uint8, device="meta")})
    try:
        stat = os.fstat(file.fileno())
        assert stat.st_size > 1 << 30
        if hasattr(stat, "st_blocks"):
            assert stat.st_blocks * 512 < 65536
        file.seek(-32, os.SEEK_END)
        assert file.read() == bytes(32)
    finally:
        file.close()


def test_anonymous_safetensors():
    state, file = create_safetensors({"kv": torch.empty((3, 5), dtype=torch.int32, device="meta")})
    info = state["kv"].untyped_storage()._comfy_tensor_file_slice
    try:
        result = torch.empty(3, 5, dtype=torch.int32)
        assert read_tensor_file_slice_into(state["kv"], result)
        assert torch.equal(result, torch.zeros_like(result))
        expected = torch.arange(15, dtype=torch.int32).reshape(3, 5)
        file.seek(info.offset)
        file.write(expected.numpy().tobytes())
        assert read_tensor_file_slice_into(state["kv"], result)
        assert torch.equal(result, expected)
        file.seek(0)
        assert torch.equal(safetensors.torch.load(file.read())["kv"], expected)
    finally:
        file.close()


class _KV(torch.nn.Module):
    def __init__(self, rows=4):
        super().__init__()
        self.register_buffer("k", torch.empty(rows, 8))
        self.register_buffer("v", torch.empty(rows, 8, dtype=torch.float16))


def test_model_state_persists_values_and_metadata(tmp_path):
    cpu = torch.device("cpu")
    path = str(tmp_path / "state.safetensors")
    state = comfy.model_state.ModelState(path)
    metadata = state.metadata  # handed to the generation before the first load creates the file
    model = state.load(_KV(), cpu)
    assert state.metadata is metadata
    assert not model.k.count_nonzero() and not model.v.count_nonzero()
    model.k.fill_(3.0)
    model.v.fill_(5.0)
    metadata["position"] = 7
    state.save()
    assert state.load(_KV(), cpu) is model
    model.k[0, 0] = 9.0
    copied = state.copy(str(tmp_path / "copy.safetensors"))
    state.release()
    assert state.patcher is None

    assert os.path.getsize(path) > comfy.model_state.HEADER_RESERVE
    saved = safetensors.torch.load_file(path)
    assert saved["k"][0, 0] == 9.0 and torch.equal(saved["v"], torch.full((4, 8), 5.0, dtype=torch.float16))

    reopened = comfy.model_state.ModelState(path, create="never")
    assert reopened.metadata == {"position": 7}
    model = reopened.load(_KV(), cpu)
    assert torch.equal(model.k, saved["k"]) and torch.equal(model.v, saved["v"])
    assert copied.metadata == {"position": 7}
    assert torch.equal(copied.load(_KV(), cpu).k, saved["k"])

    with pytest.raises(FileNotFoundError):
        comfy.model_state.ModelState(str(tmp_path / "missing.safetensors"), create="never")
    with pytest.raises(RuntimeError):
        comfy.model_state.ModelState(path).load(_KV(rows=1), cpu)

    recreated = comfy.model_state.ModelState(path, create="always")
    assert recreated.metadata == {}
    assert not recreated.load(_KV(), cpu).k.count_nonzero()
    assert safetensors.torch.load_file(path)["k"].count_nonzero() == 0


def test_model_state_rejects_plain_safetensors(tmp_path):
    path = str(tmp_path / "plain.safetensors")
    safetensors.torch.save_file({"k": torch.zeros(4, 8)}, path)
    with pytest.raises(ValueError):
        comfy.model_state.ModelState(path)


def test_anonymous_model_state_survives_release_and_copies(tmp_path):
    cpu = torch.device("cpu")
    state = comfy.model_state.ModelState()
    model = state.load(_KV(), cpu)
    model.k.fill_(2.0)
    state.metadata["position"] = 3
    state.flush()
    state.release()
    assert torch.equal(state.load(_KV(), cpu).k, torch.full((4, 8), 2.0))
    copied = state.copy(str(tmp_path / "copy.safetensors"))
    state.release()
    assert copied.metadata == {"position": 3}
    assert torch.equal(copied.load(_KV(), cpu).k, torch.full((4, 8), 2.0))


def test_rewrite_safetensors_header_keeps_reserved_size():
    geometry = {"k": torch.empty((2,), device="meta")}
    _, file = create_safetensors(geometry, {"a": "1"}, header_size=256)
    try:
        _, header = open_safetensors(file)
        header["__metadata__"] = {"a": "1", "b": "x" * 100}
        rewrite_safetensors_header(file, header)
        assert open_safetensors(file)[1]["__metadata__"]["b"] == "x" * 100
        header["__metadata__"]["b"] = "x" * 300
        with pytest.raises(ValueError):
            rewrite_safetensors_header(file, header)
    finally:
        file.close()
