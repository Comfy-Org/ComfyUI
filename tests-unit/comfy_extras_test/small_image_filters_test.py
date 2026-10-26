import math

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy_extras.nodes_post_processing import Blur, Sharpen


def _apply_filter(node, image, radius):
    if node is Sharpen:
        return node.execute(image, radius, sigma=1.0, alpha=0.1)[0]
    return node.execute(image, radius, sigma=1.0)[0]


@pytest.mark.parametrize("node", [Blur, Sharpen])
@pytest.mark.parametrize("shape,radius,dtype", [
    ((1, 1, 1, 3), 31, torch.float32),
    ((2, 2, 3, 4), 3, torch.float32),
    ((1, 4, 1, 3), 3, torch.float64),
    ((1, 3, 3, 3), 3, torch.float32),
])
def test_small_filters_preserve_constant_images(node, shape, radius, dtype):
    batch, height, width, channels = shape
    image = torch.linspace(0.2, 0.8, batch * channels, dtype=dtype).reshape(batch, 1, 1, channels)
    image = image.expand(batch, height, width, channels).clone()
    before = image.clone()
    result = _apply_filter(node, image, radius)

    assert result.shape == image.shape
    assert result.dtype == image.dtype
    torch.testing.assert_close(result, image.to(result.device), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(image, before)


@pytest.mark.parametrize("node", [Blur, Sharpen])
def test_filters_keep_reflected_edges_for_normal_images(node):
    image = torch.tensor([0.3, 0.4, 0.5, 0.6, 0.7]).reshape(1, 1, 5, 1).repeat(1, 3, 1, 3)
    edge_change = 0.2 * math.exp(-0.5) / (1 + 2 * math.exp(-0.5))
    if node is Sharpen:
        edge_change = -edge_change
    expected = torch.tensor([0.3 + edge_change, 0.4, 0.5, 0.6, 0.7 - edge_change])
    expected = expected.reshape(1, 1, 5, 1).expand_as(image)

    result = _apply_filter(node, image, 1)
    torch.testing.assert_close(result, expected.to(result.device))


@pytest.mark.parametrize("node", [Blur, Sharpen])
def test_filters_keep_zero_radius_passthrough(node):
    image = torch.rand(1, 1, 1, 3)
    assert _apply_filter(node, image, 0) is image
