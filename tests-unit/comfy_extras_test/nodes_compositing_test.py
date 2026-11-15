import pytest
import torch

from comfy_extras.nodes_compositing import PorterDuffMode, porter_duff_composite


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
