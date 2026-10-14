from collections import OrderedDict
import hashlib

import pytest
import torch

from comfy_api.latest._sdk import _InProcessModels


class UnsupportedWeightObject:
    pass


@pytest.mark.parametrize("use_zip", [False, True], ids=["legacy", "zip"])
def test_original_restricted_weight_formats_are_accepted_without_rewriting(tmp_path, use_zip):
    path = tmp_path / "original.pth"
    state = OrderedDict(weight=torch.arange(12, dtype=torch.float32).reshape(3, 4))
    torch.save(state, path, _use_new_zipfile_serialization=use_zip)
    original = path.read_bytes()

    _InProcessModels._verify_weight_file(str(path), ".pth")

    assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(original).digest()
    restored = torch.load(path, map_location="cpu", weights_only=True)
    assert torch.equal(restored["weight"], state["weight"])


@pytest.mark.parametrize("use_zip", [False, True], ids=["legacy", "zip"])
def test_custom_pickle_globals_remain_rejected_for_both_formats(tmp_path, use_zip):
    path = tmp_path / "custom.pth"
    torch.save({"weight": torch.ones(2), "custom": UnsupportedWeightObject()}, path,
               _use_new_zipfile_serialization=use_zip)

    with pytest.raises(ValueError, match="not a restricted PyTorch weight archive"):
        _InProcessModels._verify_weight_file(str(path), ".pth")


@pytest.mark.parametrize("use_zip", [False, True], ids=["legacy", "zip"])
def test_tensor_free_archives_remain_rejected(tmp_path, use_zip):
    path = tmp_path / "no_weights.pth"
    torch.save({"version": 1, "labels": ["a", "b"]}, path,
               _use_new_zipfile_serialization=use_zip)

    with pytest.raises(ValueError, match="contains no tensors"):
        _InProcessModels._verify_weight_file(str(path), ".pth")


@pytest.mark.parametrize("use_zip", [False, True], ids=["legacy", "zip"])
def test_truncated_weight_archives_remain_rejected(tmp_path, use_zip):
    path = tmp_path / "truncated.pth"
    torch.save({"weight": torch.ones(32)}, path,
               _use_new_zipfile_serialization=use_zip)
    raw = path.read_bytes()
    path.write_bytes(raw[:len(raw) // 2])

    with pytest.raises(ValueError, match="not a restricted PyTorch weight archive"):
        _InProcessModels._verify_weight_file(str(path), ".pth")
