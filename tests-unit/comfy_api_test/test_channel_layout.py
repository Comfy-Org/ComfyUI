"""Tensors keep their memory layout across the pack-isolation channel (shared memory)."""
import pytest
import torch

from comfy_api._impl.isolation import channel


def _channel():
    side = channel.Channel.__new__(channel.Channel)
    side._written = []
    return side


def _round_trip(tensor):
    handle = _channel()._export_tensor(tensor)
    return channel._import(handle), handle


def test_a_permuted_layout_crosses_as_it_is():
    # ComfyUI's VAEDecode returns images as a movedim view: channels-last strides, not contiguous
    image = torch.rand(1, 3, 64, 48).movedim(1, -1)
    assert not image.is_contiguous()
    back, handle = _round_trip(image)
    assert handle.get("permute") is not None
    assert back.shape == image.shape and back.stride() == image.stride() and not back.is_contiguous()
    assert torch.equal(back, image)


def test_bfloat16_keeps_its_layout_too():
    image = torch.rand(2, 4, 32, 32).to(torch.bfloat16).movedim(1, -1)
    back, _handle = _round_trip(image)
    assert back.dtype == torch.bfloat16 and back.stride() == image.stride() and torch.equal(back, image)


def test_contiguous_and_other_layouts_cross_contiguous():
    plain = torch.rand(4, 64, 64)
    back, handle = _round_trip(plain)
    assert "permute" not in handle and back.is_contiguous() and torch.equal(back, plain)
    sliced = torch.rand(4, 64, 128)[:, :, ::2]  # gaps: not a permutation, so it crosses as a contiguous copy
    back, handle = _round_trip(sliced)
    assert "permute" not in handle and back.is_contiguous() and torch.equal(back, sliced)


def test_a_bad_layout_from_the_other_side_is_refused():
    handle = _channel()._export_tensor(torch.rand(1, 3, 64, 48).movedim(1, -1))
    handle["permute"] = [0, 0, 1, 2]
    with pytest.raises(ValueError):
        channel._import(handle)
