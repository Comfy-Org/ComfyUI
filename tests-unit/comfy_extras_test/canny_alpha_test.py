import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy_extras.nodes_canny import Canny  # noqa: E402


def edged_image(channels, alpha=0.8, size=16):
    """Half black, half white, so there is a real edge down the middle."""
    t = torch.zeros(1, size, size, channels)
    t[:, :, size // 2:, :3] = 1.0
    if channels == 4:
        t[..., 3] = alpha
    return t


def test_rgb_detects_the_edge():
    out = Canny.execute(edged_image(3), 0.4, 0.8).result[0]

    assert out.shape[-1] == 3
    assert out.max() > 0.0


def test_rgba_does_not_raise():
    out = Canny.execute(edged_image(4), 0.4, 0.8).result[0]

    assert out.shape[-1] == 3
    assert out.max() > 0.0


def test_rgba_and_rgb_give_the_same_edges():
    """Alpha must not influence edge detection."""
    from_rgb = Canny.execute(edged_image(3), 0.4, 0.8).result[0]
    from_rgba = Canny.execute(edged_image(4), 0.4, 0.8).result[0]

    assert torch.equal(from_rgb, from_rgba)


def test_alpha_pattern_does_not_change_the_result():
    opaque = edged_image(4, alpha=1.0)
    transparent = edged_image(4, alpha=0.0)

    from_opaque = Canny.execute(opaque, 0.4, 0.8).result[0]
    from_transparent = Canny.execute(transparent, 0.4, 0.8).result[0]

    assert torch.equal(from_opaque, from_transparent)


def test_does_not_mutate_input():
    src = edged_image(4)
    before = src.clone()

    Canny.execute(src, 0.4, 0.8)

    assert torch.equal(src, before)
