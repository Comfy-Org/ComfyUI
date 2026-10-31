import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy_extras.nodes_images import ImageScaleToMaxDimension  # noqa: E402


def run(image, largest_size=512, method="bilinear"):
    return ImageScaleToMaxDimension.execute(image, method, largest_size).result[0]


def test_thin_image_keeps_one_pixel_short_side():
    wide = run(torch.full((1, 1, 2048, 3), 0.25))
    tall = run(torch.full((1, 2048, 1, 3), 0.25))

    assert wide.shape == (1, 1, 512, 3)
    assert tall.shape == (1, 512, 1, 3)


def test_normal_image_is_unchanged_by_the_clamp():
    out = run(torch.rand(1, 64, 128, 3))

    assert out.shape == (1, 256, 512, 3)


def test_square_image_uses_largest_size_for_both_sides():
    out = run(torch.rand(1, 40, 40, 3))

    assert out.shape == (1, 512, 512, 3)
