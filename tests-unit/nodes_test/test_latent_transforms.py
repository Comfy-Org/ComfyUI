import math

import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import nodes  # noqa: E402


def _samples(*shape):
    return torch.arange(math.prod(shape)).reshape(*shape).float()


def test_flip_vertical_on_4d_latent_flips_height():
    samples = _samples(1, 2, 2, 3)

    s, = nodes.LatentFlip().flip({"samples": samples}, "x-axis: vertically")

    assert torch.equal(s["samples"], torch.flip(samples, dims=[2]))


def test_flip_horizontal_on_4d_latent_flips_width():
    samples = _samples(1, 2, 2, 3)

    s, = nodes.LatentFlip().flip({"samples": samples}, "y-axis: horizontally")

    assert torch.equal(s["samples"], torch.flip(samples, dims=[3]))


def test_flip_vertical_on_5d_latent_flips_height_not_frames():
    samples = _samples(2, 1, 2, 3, 4)

    s, = nodes.LatentFlip().flip({"samples": samples}, "x-axis: vertically")

    assert torch.equal(s["samples"], torch.flip(samples, dims=[3]))
    assert not torch.equal(s["samples"], torch.flip(samples, dims=[2]))


def test_flip_horizontal_on_5d_latent_flips_width_not_height():
    samples = _samples(2, 1, 2, 3, 4)

    s, = nodes.LatentFlip().flip({"samples": samples}, "y-axis: horizontally")

    assert torch.equal(s["samples"], torch.flip(samples, dims=[4]))
    assert not torch.equal(s["samples"], torch.flip(samples, dims=[3]))


def test_flip_on_single_frame_5d_latent_flips_spatial_axes():
    samples = _samples(1, 1, 1, 3, 4)

    s_vertical, = nodes.LatentFlip().flip({"samples": samples}, "x-axis: vertically")
    s_horizontal, = nodes.LatentFlip().flip({"samples": samples}, "y-axis: horizontally")

    assert torch.equal(s_vertical["samples"], torch.flip(samples, dims=[3]))
    assert torch.equal(s_horizontal["samples"], torch.flip(samples, dims=[4]))


def test_rotate_90_on_4d_latent_is_unchanged():
    samples = _samples(2, 2, 3, 4)

    s, = nodes.LatentRotate().rotate({"samples": samples}, "90 degrees")

    assert s["samples"].shape == (2, 2, 4, 3)
    assert torch.equal(s["samples"], torch.rot90(samples, k=1, dims=[3, 2]))


def test_rotate_90_on_5d_latent_swaps_height_and_width_only():
    samples = _samples(1, 1, 2, 3, 4)

    s, = nodes.LatentRotate().rotate({"samples": samples}, "90 degrees")

    assert s["samples"].shape == (1, 1, 2, 4, 3)
    assert torch.equal(s["samples"], torch.rot90(samples, k=1, dims=[4, 3]))


def test_rotate_180_on_5d_latent_keeps_frame_axis():
    samples = _samples(1, 1, 2, 3, 4)

    s, = nodes.LatentRotate().rotate({"samples": samples}, "180 degrees")

    assert s["samples"].shape == samples.shape
    assert torch.equal(s["samples"], torch.rot90(samples, k=2, dims=[4, 3]))


def test_rotate_270_on_5d_latent_swaps_height_and_width_only():
    samples = _samples(1, 1, 2, 3, 4)

    s, = nodes.LatentRotate().rotate({"samples": samples}, "270 degrees")

    assert s["samples"].shape == (1, 1, 2, 4, 3)
    assert torch.equal(s["samples"], torch.rot90(samples, k=3, dims=[4, 3]))


def test_rotate_none_returns_input():
    samples = _samples(1, 1, 2, 3, 4)

    s, = nodes.LatentRotate().rotate({"samples": samples}, "none")

    assert torch.equal(s["samples"], samples)


def test_crop_on_4d_latent_is_unchanged():
    samples = _samples(2, 1, 16, 16)

    s, = nodes.LatentCrop().crop({"samples": samples}, width=16, height=8, x=16, y=8)

    assert s["samples"].shape == (2, 1, 1, 2)
    assert torch.equal(s["samples"], samples[:, :, 1:2, 2:4])


def test_crop_on_5d_latent_crops_height_and_width():
    samples = _samples(1, 1, 2, 16, 16)

    s, = nodes.LatentCrop().crop({"samples": samples}, width=16, height=8, x=16, y=8)

    assert s["samples"].shape == (1, 1, 2, 1, 2)
    assert torch.equal(s["samples"], samples[..., 1:2, 2:4])
