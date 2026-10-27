"""Regression test: RGBA (or other non-RGB) images must not inflate the Qwen VL patch count."""

import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy.text_encoders.qwen_vl import process_qwen2vl_images  # noqa: E402


def test_extra_channels_are_dropped_before_patching():
    images_rgb = torch.rand(1, 224, 224, 3)
    images_rgba = torch.rand(1, 224, 224, 4)

    patches_rgb, grid_rgb = process_qwen2vl_images(images_rgb, patch_size=16, image_mean=[0.5] * 3, image_std=[0.5] * 3)
    patches_rgba, grid_rgba = process_qwen2vl_images(images_rgba, patch_size=16, image_mean=[0.5] * 3, image_std=[0.5] * 3)

    assert patches_rgba.shape == patches_rgb.shape
    assert torch.equal(grid_rgba, grid_rgb)

    pos_rows = int(grid_rgba[:, 1:].prod(dim=1).sum())
    patch_embed_rows = patches_rgba.numel() // (3 * 2 * 16 * 16)
    assert patch_embed_rows == pos_rows
