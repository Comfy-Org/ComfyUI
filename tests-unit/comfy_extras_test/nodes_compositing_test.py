from itertools import cycle, islice

import pytest
from PIL import Image
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy_extras.nodes_compositing import PorterDuffImageComposite, PorterDuffMode, porter_duff_composite


SOURCE = torch.tensor([0.8, 0.2, 0.6])
BACKDROP = torch.tensor([0.3, 0.7, 0.4])


def reference_source_over(mode, source_alpha, backdrop_alpha):
    if mode == PorterDuffMode.DARKEN:
        mixed = torch.minimum(BACKDROP, SOURCE)
    elif mode == PorterDuffMode.LIGHTEN:
        mixed = torch.maximum(BACKDROP, SOURCE)
    elif mode == PorterDuffMode.MULTIPLY:
        mixed = BACKDROP * SOURCE
    elif mode == PorterDuffMode.OVERLAY:
        mixed = torch.where(
            2 * BACKDROP <= 1,
            2 * BACKDROP * SOURCE,
            1 - 2 * (1 - BACKDROP) * (1 - SOURCE),
        )

    output_alpha = source_alpha + backdrop_alpha * (1 - source_alpha)
    premultiplied = (
        source_alpha * (1 - backdrop_alpha) * SOURCE
        + source_alpha * backdrop_alpha * mixed
        + (1 - source_alpha) * backdrop_alpha * BACKDROP
    )
    output = premultiplied / output_alpha if output_alpha else torch.zeros_like(SOURCE)
    return output, 1 - output_alpha


@pytest.mark.parametrize(
    "mode",
    [
        PorterDuffMode.DARKEN,
        PorterDuffMode.LIGHTEN,
        PorterDuffMode.MULTIPLY,
        PorterDuffMode.OVERLAY,
    ],
)
@pytest.mark.parametrize(
    ("source_alpha", "backdrop_alpha"),
    [(1.0, 0.0), (0.0, 1.0), (0.35, 0.65), (1.0, 1.0)],
)
def test_blend_modes_use_source_over_alpha(mode, source_alpha, backdrop_alpha):
    output, output_mask = porter_duff_composite(
        SOURCE.reshape(1, 1, 3),
        torch.tensor(1 - source_alpha).reshape(1, 1, 1),
        BACKDROP.reshape(1, 1, 3),
        torch.tensor(1 - backdrop_alpha).reshape(1, 1, 1),
        mode,
    )
    expected, expected_mask = reference_source_over(mode, source_alpha, backdrop_alpha)

    torch.testing.assert_close(output.flatten(), expected)
    torch.testing.assert_close(output_mask.flatten(), torch.tensor([expected_mask]))


@pytest.mark.parametrize(
    "batch_sizes", [(3, 1, 3, 1), (1, 3, 3, 3), (3, 3, 1, 3), (3, 3, 3, 1), (2, 1, 3, 1), (1, 3, 1, 1), (1, 1, 1, 3), (3, 3, 3, 3)]
)
def test_composite_broadcasts_shorter_batches(batch_sizes):
    """All frames survive a shared mask or image, matching Pillow's source-over."""
    source = torch.tensor([[204, 51, 153], [51, 153, 102], [153, 76, 25]], dtype=torch.float32).reshape(3, 1, 1, 3) / 255
    destination = torch.tensor([[76, 178, 102], [25, 51, 204], [153, 204, 51]], dtype=torch.float32).reshape(3, 1, 1, 3) / 255
    source_mask = torch.tensor([0, 64, 128], dtype=torch.float32).reshape(3, 1, 1) / 255
    destination_mask = torch.tensor([32, 96, 192], dtype=torch.float32).reshape(3, 1, 1) / 255
    inputs = [value[:size] for value, size in zip((source, source_mask, destination, destination_mask), batch_sizes)]
    originals = [value.clone() for value in inputs]

    images, masks = PorterDuffImageComposite.execute(*inputs, "SRC_OVER").result

    expected = []
    for src, src_mask, dst, dst_mask in islice(zip(*(cycle(value) for value in inputs)), 3):
        src_rgba = tuple((src.flatten() * 255).round().int().tolist()) + (round((1 - src_mask.item()) * 255),)
        dst_rgba = tuple((dst.flatten() * 255).round().int().tolist()) + (round((1 - dst_mask.item()) * 255),)
        result = Image.alpha_composite(Image.new("RGBA", (1, 1), dst_rgba), Image.new("RGBA", (1, 1), src_rgba))
        expected.append(result.getpixel((0, 0)))
    reference = torch.tensor(expected, dtype=torch.float32) / 255
    assert images.shape == (3, 1, 1, 3)
    assert masks.shape == (3, 1, 1)
    torch.testing.assert_close(images[:, 0, 0], reference[:, :3], atol=1 / 255, rtol=0)
    torch.testing.assert_close(masks[:, 0, 0], 1 - reference[:, 3], atol=1 / 255, rtol=0)
    for value, original in zip(inputs, originals):
        torch.testing.assert_close(value, original)
