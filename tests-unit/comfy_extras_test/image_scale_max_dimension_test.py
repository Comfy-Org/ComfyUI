import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy_extras.nodes_images import ImageScaleToMaxDimension


@pytest.mark.parametrize("upscale_method", ["area", "lanczos", "bilinear", "nearest-exact", "bicubic"])
@pytest.mark.parametrize(
    "height,width,largest_size,expected_size",
    [
        (1, 2048, 512, (1, 512)),
        (2048, 1, 512, (512, 1)),
        (1, 2, 1, (1, 1)),
        (2, 1, 1, (1, 1)),
        (3, 4, 8, (6, 8)),
        (4, 3, 8, (8, 6)),
        (4, 4, 8, (8, 8)),
    ],
)
def test_scale_to_max_dimension(height, width, largest_size, expected_size, upscale_method):
    image = torch.zeros((2, height, width, 3))
    image[0, ..., 0] = 1
    image[1, ..., 1] = 1
    original = image.clone()

    output = ImageScaleToMaxDimension.execute(image, upscale_method, largest_size)[0]

    assert output.shape == (2, *expected_size, 3)
    assert output.dtype == image.dtype
    assert output.device == image.device
    torch.testing.assert_close(output, image[:, :1, :1, :].expand_as(output))
    assert torch.equal(image, original)
