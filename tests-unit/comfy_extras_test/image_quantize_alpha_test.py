import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy_extras.nodes_post_processing import Quantize  # noqa: E402

DITHERS = ["none", "floyd-steinberg", "bayer-2", "bayer-4", "bayer-8", "bayer-16"]


def image(channels, alpha=0.8, size=8):
    torch.manual_seed(0)
    t = torch.rand(1, size, size, channels)
    if channels == 4:
        t[..., 3] = alpha
    return t


@pytest.mark.parametrize("dither", DITHERS)
def test_rgb_still_quantizes(dither):
    src = image(3)

    out = Quantize.execute(src, 4, dither).result[0]

    assert out.shape == src.shape
    assert len(torch.unique(out.reshape(-1, 3), dim=0)) <= 4


@pytest.mark.parametrize("dither", DITHERS)
def test_rgba_does_not_raise_and_keeps_alpha(dither):
    src = image(4)

    out = Quantize.execute(src, 4, dither).result[0]

    assert out.shape == src.shape
    assert torch.equal(out[..., 3], src[..., 3])


def test_rgba_colour_channels_are_quantized():
    src = image(4)

    out = Quantize.execute(src, 4, "none").result[0]

    assert len(torch.unique(out[..., :3].reshape(-1, 3), dim=0)) <= 4


def test_varying_alpha_is_preserved_per_pixel():
    src = image(4)
    src[0, :, :, 3] = torch.linspace(0.0, 1.0, src.shape[2])

    out = Quantize.execute(src, 8, "none").result[0]

    assert torch.equal(out[..., 3], src[..., 3])


def test_does_not_mutate_input():
    src = image(4)
    before = src.clone()

    Quantize.execute(src, 4, "none")

    assert torch.equal(src, before)
