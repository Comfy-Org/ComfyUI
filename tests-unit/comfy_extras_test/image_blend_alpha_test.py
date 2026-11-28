import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy_extras.nodes_post_processing import Blend  # noqa: E402

MODES = ["normal", "multiply", "screen", "overlay", "soft_light", "difference"]


def image(value, alpha=None, size=4):
    channels = 3 if alpha is None else 4
    t = torch.full((1, size, size, channels), value)
    if alpha is not None:
        t[..., 3] = alpha
    return t


@pytest.mark.parametrize("mode", MODES)
def test_rgb_blend_is_unchanged(mode):
    """3 channel images must keep going through the untouched code path."""
    image1, image2 = image(0.8), image(0.3)

    out = Blend.execute(image1, image2, 0.5, mode).result[0]

    assert out.shape == image1.shape
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


@pytest.mark.parametrize("mode", MODES)
def test_rgba_keeps_image1_alpha(mode):
    image1, image2 = image(0.8, alpha=0.6), image(0.3, alpha=0.1)

    out = Blend.execute(image1, image2, 0.5, mode).result[0]

    assert out.shape[-1] == 4
    assert torch.equal(out[..., 3], image1[..., 3])


@pytest.mark.parametrize("mode", MODES)
def test_two_opaque_images_stay_opaque(mode):
    """Regression: difference mode used to compute 1.0 - 1.0 and erase the image."""
    image1, image2 = image(0.8, alpha=1.0), image(0.3, alpha=1.0)

    out = Blend.execute(image1, image2, 1.0, mode).result[0]

    assert torch.all(out[..., 3] == 1.0)


@pytest.mark.parametrize("mode", MODES)
def test_rgb_channels_still_blend_on_rgba(mode):
    """Preserving alpha must not stop the colour channels from blending.

    Mid-tones on purpose: pure white over pure black is a fixed point for
    several of the modes, so it would not prove anything.
    """
    image1, image2 = image(0.6, alpha=1.0), image(0.25, alpha=1.0)

    out = Blend.execute(image1, image2, 1.0, mode).result[0]

    assert not torch.equal(out[..., :3], image1[..., :3])


def test_does_not_mutate_inputs():
    image1, image2 = image(0.8, alpha=0.6), image(0.3, alpha=0.1)
    before1, before2 = image1.clone(), image2.clone()

    Blend.execute(image1, image2, 0.5, "difference")

    assert torch.equal(image1, before1)
    assert torch.equal(image2, before2)


def test_mismatched_channel_counts_still_supported():
    """image_alpha_fix (CORE-103) pads the RGB input; that must keep working."""
    rgba, rgb = image(0.8, alpha=0.5), image(0.3)

    out = Blend.execute(rgba, rgb, 0.5, "normal").result[0]

    assert out.shape[-1] == 4
    assert torch.equal(out[..., 3], rgba[..., 3])


def _rgb_image(rgb, alpha=None, size=4):
    t = torch.tensor(rgb, dtype=torch.float32).view(1, 1, 1, 3).expand(1, size, size, 3).clone()
    if alpha is not None:
        a = torch.full((1, size, size, 1), alpha)
        t = torch.cat([t, a], dim=-1)
    return t


@pytest.mark.parametrize("mode", ["luminosity", "color"])
def test_hls_blend_modes_rgba_preserve_alpha(mode):
    image1 = _rgb_image((0.5, 0.5, 0.5), alpha=0.6)
    image2 = _rgb_image((0.8, 0.2, 0.2), alpha=0.1)

    out = Blend.execute(image1, image2, 1.0, mode).result[0]

    assert out.shape == (1, 4, 4, 4)
    assert torch.equal(out[..., 3], image1[..., 3])
    assert not torch.isnan(out).any()


def test_color_and_luminosity_blend_semantics():
    gray = _rgb_image((0.4, 0.4, 0.4))
    red = _rgb_image((0.8, 0.2, 0.2))

    colorized = Blend.execute(gray, red, 1.0, "color").result[0]
    lum_swapped = Blend.execute(red, gray, 1.0, "luminosity").result[0]

    # Color mode transfers hue/saturation from image2 onto image1's lightness,
    # which is symmetric to luminosity mode with the inputs swapped.
    assert torch.allclose(colorized, lum_swapped, atol=1e-5)
    # Result should stay reddish (R > G == B) and reflect gray's lightness (0.4).
    assert (colorized[..., 0] > colorized[..., 1]).all()
    assert torch.allclose(colorized[..., 1], colorized[..., 2], atol=1e-5)
    lightness = (colorized.max(dim=-1).values + colorized.min(dim=-1).values) / 2
    assert torch.allclose(lightness, torch.full_like(lightness, 0.4), atol=1e-5)


@pytest.mark.parametrize("mode", ["luminosity", "color"])
@pytest.mark.parametrize(
    "rgb1,rgb2",
    [
        ((0.0, 0.0, 0.0), (0.8, 0.2, 0.2)),
        ((0.8, 0.2, 0.2), (0.0, 0.0, 0.0)),
        ((1.0, 1.0, 1.0), (0.2, 0.7, 0.4)),
        ((0.2, 0.7, 0.4), (1.0, 1.0, 1.0)),
    ],
)
def test_hls_blend_modes_extreme_luminance_no_nans(mode, rgb1, rgb2):
    out = Blend.execute(_rgb_image(rgb1), _rgb_image(rgb2), 1.0, mode).result[0]
    assert torch.isfinite(out).all()
    assert (out >= 0.0).all() and (out <= 1.0).all()
