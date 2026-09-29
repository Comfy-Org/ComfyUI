import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy_extras.nodes_morphology import ImageRGBToYUV, ImageYUVToRGB  # noqa: E402


def image(channels, alpha=0.8, size=4):
    torch.manual_seed(0)
    t = torch.rand(1, size, size, channels)
    if channels == 4:
        t[..., 3] = alpha
    return t


def test_rgb_converts():
    y, u, v = ImageRGBToYUV.execute(image(3)).result

    assert y.shape[-1] == 3 and u.shape[-1] == 3 and v.shape[-1] == 3


def test_rgba_does_not_raise():
    y, u, v = ImageRGBToYUV.execute(image(4)).result

    assert y.shape[-1] == 3 and u.shape[-1] == 3 and v.shape[-1] == 3


def test_rgba_matches_rgb_conversion():
    """Alpha must not leak into the luma/chroma values."""
    rgba = image(4)
    rgb = rgba[..., :3].clone()

    from_rgba = ImageRGBToYUV.execute(rgba).result
    from_rgb = ImageRGBToYUV.execute(rgb).result

    for a, b in zip(from_rgba, from_rgb):
        assert torch.equal(a, b)


def test_alpha_value_does_not_change_the_conversion():
    opaque = image(4, alpha=1.0)
    transparent = image(4, alpha=0.0)

    for a, b in zip(ImageRGBToYUV.execute(opaque).result, ImageRGBToYUV.execute(transparent).result):
        assert torch.equal(a, b)


def test_round_trip_recovers_rgb():
    rgb = image(3)

    y, u, v = ImageRGBToYUV.execute(rgb).result
    out = ImageYUVToRGB.execute(y, u, v).result[0]

    # float32 through the YCbCr matrix loses ~2e-4 either way; that is pre-existing.
    assert torch.allclose(out, rgb, atol=1e-3)


def test_yuv_to_rgb_ignores_alpha_on_its_inputs():
    """A 4 channel Y/U/V input must not drag alpha into the channel mean."""
    rgb = image(3)
    y, u, v = ImageRGBToYUV.execute(rgb).result

    def with_alpha(t):
        return torch.cat((t, torch.full_like(t[..., :1], 0.3)), dim=-1)

    expected = ImageYUVToRGB.execute(y, u, v).result[0]
    out = ImageYUVToRGB.execute(with_alpha(y), with_alpha(u), with_alpha(v)).result[0]

    assert torch.equal(out, expected)


def test_does_not_mutate_input():
    src = image(4)
    before = src.clone()

    ImageRGBToYUV.execute(src)

    assert torch.equal(src, before)
