import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from nodes import ImageScale, ImageScaleBy


@pytest.fixture
def image():
    return torch.linspace(0.11, 0.89, steps=72).reshape(2, 3, 4, 3)


@pytest.mark.parametrize("width,height,crop", [(4, 3, "center"), (0, 3, "disabled"), (4, 0, "disabled")])
def test_image_scale_preserves_pixels_at_same_size(image, width, height, crop):
    (result,) = ImageScale().upscale(image, "lanczos", width, height, crop)
    torch.testing.assert_close(result, image, rtol=0, atol=0)


@pytest.mark.parametrize("scale_by", [1.0, 1.1])
def test_image_scale_by_preserves_pixels_at_same_size(image, scale_by):
    (result,) = ImageScaleBy().upscale(image, "lanczos", scale_by)
    torch.testing.assert_close(result, image, rtol=0, atol=0)


def test_image_scale_can_still_crop_and_resize():
    image = torch.arange(18, dtype=torch.float32).reshape(1, 2, 3, 3) / 18
    (result,) = ImageScale().upscale(image, "nearest-exact", 2, 4, "center")
    expected = image[:, :, 1:2].repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_image_scale_by_can_still_resize(image):
    (result,) = ImageScaleBy().upscale(image, "nearest-exact", 2.0)
    expected = image.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
