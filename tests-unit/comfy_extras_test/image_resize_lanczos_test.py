import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy.utils import common_upscale, lanczos
from comfy_extras.nodes_mask import MaskToImage
from comfy_extras.nodes_post_processing import ResizeImageMaskNode, ResizeType


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("height,width", [(1, 9), (5, 1), (5, 9)])
def test_lanczos_preserves_channels_and_batch(channels, height, width):
    samples = torch.zeros(2, channels, 8, 16)
    samples[1] = 1

    result = common_upscale(samples, width, height, "lanczos", "disabled")

    assert result.shape == (2, channels, height, width)
    assert torch.equal(result[0], torch.zeros_like(result[0]))
    assert torch.equal(result[1], torch.ones_like(result[1]))
    assert result.dtype == samples.dtype
    assert result.device == samples.device


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("height,width", [(1, 9), (5, 1), (5, 9)])
def test_lanczos_resized_mask_converts_to_image(batch, height, width):
    mask = torch.zeros(batch, 8, 16)
    mask[-1] = 1

    resized = ResizeImageMaskNode.execute(mask, "lanczos", {
        "resize_type": ResizeType.SCALE_DIMENSIONS,
        "width": width,
        "height": height,
        "crop": "disabled",
    }).result[0]

    assert resized.shape == (batch, height, width)
    image = MaskToImage.execute(resized).result[0]
    assert image.shape == (batch, height, width, 3)
    assert torch.equal(image, mask[:, :1, :1, None].expand_as(image))


def test_lanczos_preserves_three_dimensional_input():
    samples = torch.ones(2, 8, 16, dtype=torch.float16)

    result = lanczos(samples, 9, 1)

    assert result.shape == (2, 1, 9)
    assert result.dtype == samples.dtype
    assert torch.equal(result, torch.ones_like(result))
