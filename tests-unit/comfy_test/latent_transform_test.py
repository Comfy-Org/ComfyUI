import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import nodes  # noqa: E402

ROTATIONS = ["none", "90 degrees", "180 degrees", "270 degrees"]
FLIPS = ["x-axis: vertically", "y-axis: horizontally"]
CROPS = [(128, 64, 64, 64), (64, 64, 0, 0), (512, 512, 400, 400)]


def rotate(latent, rotation):
    return nodes.LatentRotate().rotate({"samples": latent}, rotation)[0]["samples"]


def flip(latent, flip_method):
    return nodes.LatentFlip().flip({"samples": latent}, flip_method)[0]["samples"]


def crop(latent, width, height, x, y):
    return nodes.LatentCrop().crop({"samples": latent}, width, height, x, y)[0]["samples"]


def per_frame(fn, latent, *args):
    return torch.stack([fn(latent[:, :, t], *args) for t in range(latent.shape[2])], dim=2)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_rotate_4d_latent(rotation):
    latent = torch.randn(2, 4, 3, 5)
    k = ROTATIONS.index(rotation)

    assert torch.equal(rotate(latent, rotation), torch.rot90(latent, k=k, dims=[3, 2]))


@pytest.mark.parametrize("flip_method,dim", [(FLIPS[0], 2), (FLIPS[1], 3)])
def test_flip_4d_latent(flip_method, dim):
    latent = torch.randn(2, 4, 3, 5)

    assert torch.equal(flip(latent, flip_method), torch.flip(latent, dims=[dim]))


@pytest.mark.parametrize("frames", [1, 3])
@pytest.mark.parametrize("rotation", ROTATIONS)
def test_rotate_5d_latent_rotates_each_frame(rotation, frames):
    latent = torch.randn(2, 16, frames, 3, 5)

    assert torch.equal(rotate(latent, rotation), per_frame(rotate, latent, rotation))


@pytest.mark.parametrize("frames", [1, 3])
@pytest.mark.parametrize("flip_method", FLIPS)
def test_flip_5d_latent_flips_each_frame(flip_method, frames):
    latent = torch.randn(2, 16, frames, 3, 5)

    assert torch.equal(flip(latent, flip_method), per_frame(flip, latent, flip_method))


def test_crop_4d_latent():
    latent = torch.randn(2, 4, 32, 32)

    assert torch.equal(crop(latent, 128, 64, 64, 64), latent[:, :, 8:16, 8:24])
    # x/y are clamped so at least 8 latent pixels remain
    assert torch.equal(crop(latent, 512, 512, 400, 400), latent[:, :, 24:, 24:])


@pytest.mark.parametrize("frames", [1, 3])
@pytest.mark.parametrize("crop_args", CROPS)
def test_crop_5d_latent_crops_each_frame(crop_args, frames):
    latent = torch.randn(2, 16, frames, 32, 32)

    assert torch.equal(crop(latent, *crop_args), per_frame(crop, latent, *crop_args))
