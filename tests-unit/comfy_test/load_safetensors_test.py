"""
Regression test for https://github.com/comfyanonymous/ComfyUI/issues/14784

VAELoader (and any other model loader going through comfy.utils.load_torch_file)
raised a low-level, confusing:

    ValueError: buffer length (N bytes) after offset (0 bytes) must be a
    multiple of element size (4)

instead of a clear "corrupt/incomplete file" message, whenever the aimdo
mmap-based safetensors loader (comfy.utils.load_safetensors) was handed a
truncated/incomplete .safetensors file (e.g. one that was still downloading,
or whose download was interrupted). The root cause: load_safetensors() never
validated that the mapped file actually contained as many bytes as the
header declared before slicing them and handing the (silently truncated)
slice to torch.frombuffer().

comfy_aimdo.model_mmap.ModelMMAP requires a compiled, GPU-platform-specific
native backend that is unavailable on this (and any non-GPU CI) machine, so
it is monkeypatched here with a fake that maps a real, small, synthetic
safetensors-formatted buffer we fully control. This lets the test exercise
comfy.utils.load_safetensors() itself, deterministically and without any
real checkpoint or GPU.
"""
import ctypes
import json
import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from comfy.cli_args import args  # noqa: E402

if not __import__("torch").cuda.is_available():
    args.cpu = True

import comfy.utils  # noqa: E402
import comfy_aimdo.model_mmap  # noqa: E402


def _write_fake_safetensors(path, data_size, declared_size):
    """Write a minimal safetensors-formatted file with a single F32 tensor
    whose header declares `declared_size` bytes of data, but only
    `data_size` bytes are actually written after the header (simulating a
    truncated/incomplete download when data_size < declared_size)."""
    header = {"test.weight": {"dtype": "F32", "shape": [declared_size // 4], "data_offsets": [0, declared_size]}}
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(b"\x00" * data_size)
    return header


class _FakeModelMMAP:
    """Stand-in for comfy_aimdo.model_mmap.ModelMMAP that maps a real file
    into a real, addressable in-process buffer without requiring the
    compiled aimdo native backend."""

    def __init__(self, filepath):
        with open(filepath, "rb") as f:
            content = f.read()
        self._buf = bytearray(content)
        self._carray = (ctypes.c_uint8 * len(self._buf)).from_buffer(self._buf)
        self._address = ctypes.addressof(self._carray)

    def get(self):
        return self._address

    def get_file_handle(self):
        return 0


@pytest.fixture
def fake_model_mmap(monkeypatch):
    monkeypatch.setattr(comfy_aimdo.model_mmap, "ModelMMAP", _FakeModelMMAP)


def test_load_safetensors_truncated_file_raises_clear_error(tmp_path, fake_model_mmap):
    """A safetensors file whose data section is shorter than the header
    declares (e.g. an interrupted/partial download) must fail with a clear,
    actionable ValueError -- not the confusing low-level
    'buffer length ... must be a multiple of element size' error raised
    deep inside torch.frombuffer."""
    path = str(tmp_path / "truncated.safetensors")
    # Header declares 40 bytes (10 x float32) of tensor data, but only 37 bytes
    # are actually present on disk -- this reproduces the reported bug's
    # "buffer length (N) ... must be a multiple of element size (4)" failure,
    # since 37 is not a multiple of 4.
    _write_fake_safetensors(path, data_size=37, declared_size=40)

    with pytest.raises(ValueError) as excinfo:
        comfy.utils.load_safetensors(path)

    message = str(excinfo.value)
    assert "corrupt/incomplete" in message
    assert path in message
    # The old, confusing torch-internal wording must not be what the user sees.
    assert "must be a multiple of element size" not in message


def test_load_safetensors_complete_file_loads_successfully(tmp_path, fake_model_mmap):
    """Sanity check: a well-formed, complete file (the non-truncated case)
    still loads correctly through the same code path."""
    path = str(tmp_path / "complete.safetensors")
    _write_fake_safetensors(path, data_size=40, declared_size=40)

    sd, metadata = comfy.utils.load_safetensors(path)

    assert set(sd.keys()) == {"test.weight"}
    assert sd["test.weight"].shape == (10,)
    assert sd["test.weight"].dtype.__str__() == "torch.float32"
