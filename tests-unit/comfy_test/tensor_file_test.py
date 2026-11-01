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
from comfy.utils import create_safetensors  # noqa: E402

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
    state = comfy.model_state.ModelState.open(path)
    assert state.tensors == {} and state.metadata == {}
    model = state.load(_KV(), cpu)
    assert not model.k.count_nonzero() and not model.v.count_nonzero()
    model.k.fill_(3.0)
    model.v.fill_(5.0)
    state.metadata["position"] = 7
    state.flush()
    assert state.patcher is None
    state.save(path)

    saved = safetensors.torch.load_file(path)
    assert torch.equal(saved["k"], torch.full((4, 8), 3.0)) and torch.equal(saved["v"], torch.full((4, 8), 5.0, dtype=torch.float16))

    reopened = comfy.model_state.ModelState.open(path, create="never")
    assert reopened.metadata == {"position": 7}

    # The next turn continues on a copy; the opened state only reaches the file when saved.
    turn = reopened.copy()
    model = turn.load(_KV(), cpu)
    assert torch.equal(model.k, saved["k"]) and torch.equal(model.v, saved["v"])
    model.k[0, 0] = 9.0
    turn.metadata["position"] = 8
    turn.flush()
    reopened.save(path)
    assert safetensors.torch.load_file(path)["k"][0, 0] == 3.0
    turn.save(path)
    assert safetensors.torch.load_file(path)["k"][0, 0] == 9.0
    assert comfy.model_state.ModelState.open(path).metadata == {"position": 8}

    with pytest.raises(FileNotFoundError):
        comfy.model_state.ModelState.open(str(tmp_path / "missing.safetensors"), create="never")
    with pytest.raises(RuntimeError):
        comfy.model_state.ModelState.open(path).load(_KV(rows=1), cpu)

    recreated = comfy.model_state.ModelState.open(path, create="always")
    assert recreated.metadata == {} and recreated.tensors == {}
    assert not recreated.load(_KV(), cpu).k.count_nonzero()
    assert safetensors.torch.load_file(path)["k"][0, 0] == 9.0


def test_model_state_rejects_plain_safetensors(tmp_path):
    path = str(tmp_path / "plain.safetensors")
    safetensors.torch.save_file({"k": torch.zeros(4, 8)}, path)
    with pytest.raises(ValueError):
        comfy.model_state.ModelState.open(path)


def test_model_state_copies_are_independent(tmp_path):
    cpu = torch.device("cpu")
    first = comfy.model_state.ModelState()
    model = first.load(_KV(), cpu)
    model.k.fill_(2.0)
    first.metadata["position"] = 3
    first.flush()

    second = first.copy()
    assert second.metadata == {"position": 3} and second.metadata is not first.metadata
    model = second.load(_KV(), cpu)
    model.k[0, 0] = 1.0
    second.metadata["position"] = 4
    second.flush()

    third = second.copy()
    assert third.metadata == {"position": 4}
    assert torch.equal(third.load(_KV(), cpu).k, model.k) and model.k[0, 0] == 1.0
    assert torch.equal(first.load(_KV(), cpu).k, torch.full((4, 8), 2.0)) and first.metadata == {"position": 3}

    # Releasing a loaded state drops values its metadata may already describe, so it cannot reach another node.
    first.patcher.model.k.fill_(5.0)
    first.release()
    with pytest.raises(RuntimeError):
        first.copy()
    with pytest.raises(RuntimeError):
        first.save(str(tmp_path / "released.safetensors"))
    # A written back state stays usable.
    third.flush()
    third.release()
    assert torch.equal(third.copy().load(_KV(), cpu).k, model.k)
