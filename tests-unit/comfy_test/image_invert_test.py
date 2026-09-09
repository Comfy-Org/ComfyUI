import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import nodes  # noqa: E402


def test_invert_rgb_inverts_every_channel():
    image = torch.rand(1, 4, 4, 3)

    s, = nodes.ImageInvert().invert(image)

    assert s.shape == image.shape
    assert torch.allclose(s, 1.0 - image)


def test_invert_rgba_leaves_alpha_untouched():
    image = torch.rand(2, 4, 4, 4)

    s, = nodes.ImageInvert().invert(image)

    assert s.shape == image.shape
    assert torch.allclose(s[..., :3], 1.0 - image[..., :3])
    assert torch.equal(s[..., 3], image[..., 3])


def test_invert_rgba_keeps_transparent_pixels_transparent():
    image = torch.zeros(1, 1, 2, 4)
    image[..., 3] = torch.tensor([1.0, 0.0])

    s, = nodes.ImageInvert().invert(image)

    assert s[0, 0, 0, 3] == 1.0
    assert s[0, 0, 1, 3] == 0.0


def test_invert_does_not_mutate_input():
    image = torch.rand(1, 4, 4, 4)
    original = image.clone()

    nodes.ImageInvert().invert(image)

    assert torch.equal(image, original)
