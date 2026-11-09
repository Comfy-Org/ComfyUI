import os
import sys

import pytest
import safetensors.torch
import torch

from comfy.memory_management import read_tensor_file_slice_into, write_tensor_file_slice
from comfy.utils import create_safetensors

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
