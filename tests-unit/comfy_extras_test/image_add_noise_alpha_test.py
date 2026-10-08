import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy_extras.nodes_images import ImageAddNoise  # noqa: E402


def image(channels, value=0.5, alpha=0.5, size=8):
    t = torch.full((1, size, size, channels), value)
    if channels == 4:
        t[..., 3] = alpha
    return t


def test_rgb_gets_noise_on_every_channel():
    src = image(3)

    s = ImageAddNoise.execute(src, 0, 0.5).result[0]

    assert s.shape == src.shape
    assert not torch.equal(s, src)


def test_rgba_keeps_alpha_untouched():
    src = image(4)

    s = ImageAddNoise.execute(src, 0, 0.5).result[0]

    assert s.shape[-1] == 4
    assert torch.equal(s[..., 3], src[..., 3])
    assert not torch.equal(s[..., :3], src[..., :3])


def test_fully_transparent_pixels_stay_transparent():
    src = image(4, alpha=0.0)

    s = ImageAddNoise.execute(src, 0, 1.0).result[0]

    assert torch.all(s[..., 3] == 0.0)


def test_only_the_alpha_channel_is_guarded():
    """Colour channels must still get exactly the unguarded noise result."""
    src = image(4)
    generator = torch.manual_seed(0)
    unguarded = torch.clip(src + 0.5 * torch.randn(src.size(), generator=generator), min=0.0, max=1.0)

    s = ImageAddNoise.execute(src, 0, 0.5).result[0]

    assert torch.equal(s[..., :3], unguarded[..., :3])
    assert torch.equal(s[..., 3], src[..., 3])


def test_does_not_mutate_input():
    src = image(4)
    before = src.clone()

    ImageAddNoise.execute(src, 0, 0.5)

    assert torch.equal(src, before)
