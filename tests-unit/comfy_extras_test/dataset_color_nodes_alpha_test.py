import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy_extras.nodes_dataset import (  # noqa: E402
    AdjustBrightnessNode,
    AdjustContrastNode,
    NormalizeImagesNode,
)

# (node, kwargs) for each colour adjustment that must leave alpha alone.
CASES = [
    (NormalizeImagesNode, {"mean": 0.5, "std": 0.5}),
    (AdjustBrightnessNode, {"factor": 1.5}),
    (AdjustContrastNode, {"factor": 1.5}),
]
IDS = [node.__name__ for node, _ in CASES]


def image(channels, value=0.25, alpha=0.8, size=4):
    # alpha 0.8 on purpose: 0.0, 0.5 and 1.0 are fixed points of one or more of
    # these transforms, so they would hide the bug.
    t = torch.full((1, size, size, channels), value)
    if channels == 4:
        t[..., 3] = alpha
    return t


@pytest.mark.parametrize(("node", "kwargs"), CASES, ids=IDS)
def test_rgb_is_adjusted_on_every_channel(node, kwargs):
    src = image(3)

    out = node._process(src, **kwargs)

    assert out.shape == src.shape
    assert not torch.equal(out, src)


@pytest.mark.parametrize(("node", "kwargs"), CASES, ids=IDS)
def test_rgba_keeps_alpha_untouched(node, kwargs):
    src = image(4)

    out = node._process(src, **kwargs)

    assert out.shape[-1] == 4
    assert torch.equal(out[..., 3], src[..., 3])


@pytest.mark.parametrize(("node", "kwargs"), CASES, ids=IDS)
def test_colour_channels_still_change(node, kwargs):
    src = image(4)

    out = node._process(src, **kwargs)

    assert not torch.equal(out[..., :3], src[..., :3])


@pytest.mark.parametrize(("node", "kwargs"), CASES, ids=IDS)
def test_transparent_pixels_stay_transparent(node, kwargs):
    src = image(4, alpha=0.0)

    out = node._process(src, **kwargs)

    assert torch.all(out[..., 3] == 0.0)


@pytest.mark.parametrize(("node", "kwargs"), CASES, ids=IDS)
def test_does_not_mutate_input(node, kwargs):
    src = image(4)
    before = src.clone()

    node._process(src, **kwargs)

    assert torch.equal(src, before)
