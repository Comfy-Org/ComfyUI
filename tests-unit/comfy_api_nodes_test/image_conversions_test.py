from io import BytesIO

import pytest
import torch
from PIL import Image

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy_api_nodes.util.conversions import (  # noqa: E402
    downscale_image_tensor_by_max_sides,
    bytesio_to_image_tensor,
    downscale_image_tensor,
    pad_images_to_common_channels,
)


def encode(image: Image.Image, image_format: str = "PNG") -> BytesIO:
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    buffer.seek(0)
    return buffer


def test_rgb_png_stays_three_channels():
    tensor = bytesio_to_image_tensor(encode(Image.new("RGB", (4, 4), (10, 20, 30))))
    assert tensor.shape == (1, 4, 4, 3)


def test_jpeg_stays_three_channels():
    tensor = bytesio_to_image_tensor(encode(Image.new("RGB", (4, 4), (10, 20, 30)), "JPEG"))
    assert tensor.shape == (1, 4, 4, 3)


def test_grayscale_is_expanded_to_rgb():
    tensor = bytesio_to_image_tensor(encode(Image.new("L", (4, 4), 128)))
    assert tensor.shape == (1, 4, 4, 3)


def test_rgba_png_keeps_its_alpha():
    tensor = bytesio_to_image_tensor(encode(Image.new("RGBA", (4, 4), (10, 20, 30, 0))))
    assert tensor.shape == (1, 4, 4, 4)
    assert tensor[..., 3].max() == 0.0


def test_palette_png_with_transparency_keeps_its_alpha():
    image = Image.new("P", (4, 4), 1)
    image.putpalette([0, 0, 0, 255, 255, 255])
    image.info["transparency"] = 0
    image.putpixel((0, 0), 0)
    tensor = bytesio_to_image_tensor(encode(image))
    assert tensor.shape == (1, 4, 4, 4)
    assert tensor[0, 0, 0, 3] == 0.0
    assert tensor[0, 1, 1, 3] == 1.0


@pytest.mark.parametrize("mode,channels", [("RGB", 3), ("RGBA", 4)])
def test_explicit_mode_is_respected(mode, channels):
    tensor = bytesio_to_image_tensor(encode(Image.new("RGBA", (4, 4), (10, 20, 30, 128))), mode=mode)
    assert tensor.shape == (1, 4, 4, channels)


def test_pad_mixed_channels_concatenates():
    rgb = torch.rand(1, 4, 4, 3)
    rgba = torch.rand(2, 4, 4, 4)
    padded = pad_images_to_common_channels([rgb, rgba])
    result = torch.cat(padded, dim=0)
    assert result.shape == (3, 4, 4, 4)


def test_pad_adds_opaque_alpha_and_keeps_rgb_values():
    rgb = torch.rand(1, 4, 4, 3)
    rgba = torch.rand(1, 4, 4, 4)
    padded_rgb, padded_rgba = pad_images_to_common_channels([rgb, rgba])
    assert torch.equal(padded_rgb[..., :3], rgb)
    assert padded_rgb[..., 3].min() == 1.0
    assert padded_rgba is rgba


def test_pad_leaves_homogeneous_channels_unchanged():
    images = [torch.rand(1, 4, 4, 3), torch.rand(2, 4, 4, 3)]
    padded = pad_images_to_common_channels(images)
    assert all(p is i for p, i in zip(padded, images))


DOWNSCALE_CASES = [
    (5000, 2000, 2048 * 2048),
    (2000, 5000, 2048 * 2048),
    (4096, 1638, 2048 * 2048),
    (1000, 400, 128 * 128),
    (400, 1000, 128 * 128),
    (999, 333, 100 * 100),
    (333, 999, 100 * 100),
    (3000, 3000, 256 * 256),
]


def downscaled_size(width, height, total_pixels):
    out = downscale_image_tensor(torch.zeros(1, height, width, 3), total_pixels=total_pixels)
    return out.shape[2], out.shape[1]


@pytest.mark.parametrize("width, height, total_pixels", DOWNSCALE_CASES)
def test_downscale_dims_are_even(width, height, total_pixels):
    new_w, new_h = downscaled_size(width, height, total_pixels)
    assert new_w % 2 == 0 and new_h % 2 == 0


@pytest.mark.parametrize("width, height, total_pixels", DOWNSCALE_CASES)
def test_downscale_fits_total_pixels(width, height, total_pixels):
    new_w, new_h = downscaled_size(width, height, total_pixels)
    assert new_w * new_h <= total_pixels


@pytest.mark.parametrize("width, height, total_pixels", DOWNSCALE_CASES)
def test_downscale_never_makes_aspect_more_elongated(width, height, total_pixels):
    new_w, new_h = downscaled_size(width, height, total_pixels)
    src_ratio, new_ratio = width / height, new_w / new_h
    if src_ratio >= 1:
        assert 1 <= new_ratio <= src_ratio
    else:
        assert src_ratio <= new_ratio <= 1


def test_downscale_leaves_fitting_images_untouched():
    image = torch.zeros(1, 300, 700, 3)
    assert downscale_image_tensor(image, total_pixels=700 * 300) is image


@pytest.mark.parametrize(
    "src, expected",
    [
        ((5120, 2048), (2048, 820)),
        ((2048, 5120), (820, 2048)),
        ((4096, 4096), (1024, 1024)),
        ((2048, 1024), (2048, 1024)),
        ((768, 432), (768, 432)),
        ((1000, 3000), (683, 2048)),
    ],
)
def test_downscale_by_max_sides(src, expected):
    w, h = src
    out = downscale_image_tensor_by_max_sides(torch.zeros(1, h, w, 3), max_long_side=2048, max_short_side=1024)
    assert (out.shape[2], out.shape[1]) == expected
    src_ar = max(w, h) / min(w, h)
    out_ar = max(out.shape[1], out.shape[2]) / min(out.shape[1], out.shape[2])
    assert out_ar <= src_ar + 1e-9
