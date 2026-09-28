import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy_extras.nodes_text_overlay import TextOverlay  # noqa: E402

TEXT = "Hello"
SIZE = 64


def image(channels, value=0.25, alpha=1.0):
    t = torch.full((1, SIZE, SIZE, channels), value)
    if channels == 4:
        t[..., 3] = alpha
    return t


def overlay(images, text=TEXT, outline=True):
    return TextOverlay.execute(images, text, 20.0, "#ffffff", "top", "left", outline).result[0]


def rendered_overlay(text=TEXT, outline=True):
    return TextOverlay.render_overlay_text(
        SIZE, SIZE, text, "top", "left", 20.0, (255, 255, 255, 255), (0, 0, 0, 255) if outline else (0, 0, 0, 0)
    )


def test_rgb_draws_text():
    src = image(3)

    out = overlay(src)

    assert out.shape == src.shape
    assert not torch.equal(out, src)


def test_rgba_does_not_raise():
    src = image(4)

    out = overlay(src)

    assert out.shape == src.shape


def test_colour_result_matches_the_rgb_case():
    from_rgb = overlay(image(3))
    from_rgba = overlay(image(4))

    assert torch.allclose(from_rgba[..., :3], from_rgb)


def test_opaque_image_stays_opaque():
    out = overlay(image(4, alpha=1.0))

    assert torch.all(out[..., 3] == 1.0)


def test_text_is_visible_on_a_transparent_image():
    """Coverage has to be added where the glyphs land, or the text is invisible."""
    out = overlay(image(4, alpha=0.0))

    assert out[..., 3].max() > 0.0


def test_alpha_matches_source_over():
    src = image(4, alpha=0.5)
    _, overlay_alpha = rendered_overlay()
    expected = overlay_alpha + (1.0 - overlay_alpha) * src[..., 3:]

    out = overlay(src)

    assert torch.allclose(out[..., 3:], expected, atol=1e-6)


def test_semi_transparent_destination_gains_coverage():
    """max(dst, src) would return 0.5 here; source over must exceed it."""
    src = image(4, alpha=0.5)

    out = overlay(src)

    assert out[..., 3].max() > 0.5


def test_transparent_background_does_not_darken_text():
    """Undefined RGB under a fully transparent pixel must not bleed into glyphs."""
    src = image(4, value=0.0, alpha=0.0)

    out = overlay(src, outline=False)

    covered = out[..., 3] > 0.01
    assert covered.any()
    assert torch.allclose(out[..., :3][covered], torch.ones_like(out[..., :3][covered]), atol=1e-5)


def test_untouched_pixels_keep_their_alpha():
    src = image(4, alpha=0.0)

    out = overlay(src, text="i")

    assert torch.any(out[..., 3] == 0.0)


def test_empty_text_passes_the_image_through():
    src = image(4, alpha=0.4)

    out = overlay(src, text="   ")

    assert torch.equal(out, src)


def test_does_not_mutate_input():
    src = image(4, alpha=0.6)
    before = src.clone()

    overlay(src)

    assert torch.equal(src, before)
