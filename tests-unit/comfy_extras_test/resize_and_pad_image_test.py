import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy_extras.nodes_images import ResizeAndPadImage


@pytest.mark.parametrize("interpolation", ["area", "bicubic", "nearest-exact", "bilinear", "lanczos"])
@pytest.mark.parametrize("padding_color", ["black", "white"])
@pytest.mark.parametrize("height,width", [(1, 1000), (1000, 1)])
def test_thin_image_keeps_one_pixel_after_downscaling(height, width, padding_color, interpolation):
    image = torch.full((2, height, width, 3), 0.25)

    output = ResizeAndPadImage.execute(image, 64, 64, padding_color, interpolation)[0]

    expected = torch.full((2, 64, 64, 3), 0.0 if padding_color == "black" else 1.0)
    if height == 1:
        expected[:, 31:32, :, :] = 0.25
    else:
        expected[:, :, 31:32, :] = 0.25
    assert output.dtype == image.dtype
    assert output.device == image.device
    torch.testing.assert_close(output, expected, atol=1 / 255, rtol=0)


@pytest.mark.parametrize("height,width", [(1, 1000), (1000, 1)])
def test_minimum_target_size(height, width):
    image = torch.full((1, height, width, 3), 0.5)

    output = ResizeAndPadImage.execute(image, 1, 1, "black", "area")[0]

    torch.testing.assert_close(output, torch.full((1, 1, 1, 3), 0.5))


def test_regular_image_keeps_existing_resize_and_padding():
    image = torch.arange(24.0).reshape(1, 2, 4, 3) / 24

    output = ResizeAndPadImage.execute(image, 8, 8, "white", "nearest-exact")[0]

    expected = torch.ones(1, 8, 8, 3)
    expected[:, 2:6, :, :] = image.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
    torch.testing.assert_close(output, expected)
