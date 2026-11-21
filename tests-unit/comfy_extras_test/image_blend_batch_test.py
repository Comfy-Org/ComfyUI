import pytest
import torch
from unittest import mock

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.utils  # noqa: E402
from comfy_extras.nodes_post_processing import Blend  # noqa: E402

MODES = ["normal", "multiply", "screen", "overlay", "soft_light", "difference"]


def images(batch, value, size=4):
    """A batch of `batch` images, each filled with a distinct value."""
    return torch.stack([torch.full((size, size, 3), value + i / 10) for i in range(batch)])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("image1_batch,image2_batch", [(1, 3), (3, 1), (2, 3), (3, 2), (3, 3)])
def test_blend_matches_batches_cyclically(mode, image1_batch, image2_batch):
    """Unequal batches must blend pairwise instead of failing to broadcast.

    The shorter batch repeats cyclically, matching the batch matching the
    other compositing nodes use.
    """
    image1 = images(image1_batch, 0.1)
    image2 = images(image2_batch, 0.5)
    original1, original2 = image1.clone(), image2.clone()

    out = Blend.execute(image1, image2, 0.5, mode).result[0]

    batch_size = max(image1_batch, image2_batch)
    assert out.shape[0] == batch_size
    for i in range(batch_size):
        expected = Blend.execute(
            image1[i % image1_batch : i % image1_batch + 1],
            image2[i % image2_batch : i % image2_batch + 1],
            0.5,
            mode,
        ).result[0]
        torch.testing.assert_close(out[i : i + 1], expected)

    torch.testing.assert_close(image1, original1)
    torch.testing.assert_close(image2, original2)


def test_equal_batches_blend_each_pair():
    """Two equally sized batches blend pairwise, pixel for pixel."""
    image1 = images(2, 0.1)
    image2 = images(2, 0.5)

    out = Blend.execute(image1, image2, 0.5, "normal").result[0]

    expected = (image1 + image2) * 0.5
    torch.testing.assert_close(out, expected)
    assert out.shape == image1.shape


def test_blend_keeps_spatial_resize_for_unequal_batches():
    """Matching the batch must not stop image2 from being resized to image1."""
    image1 = images(1, 0.1, size=4)
    image2 = images(3, 0.5, size=8)
    # A gradient, so a wrong resize method cannot pass on the pixel values.
    image2 = image2 + torch.arange(8 * 8, dtype=image2.dtype).view(1, 8, 8, 1) / 100

    out = Blend.execute(image1, image2, 0.5, "normal").result[0]

    assert out.shape == (3, 4, 4, 3)
    resized_image2 = torch.nn.functional.interpolate(
        image2.permute(0, 3, 1, 2), size=(4, 4), mode="bicubic"
    ).permute(0, 2, 3, 1)
    torch.testing.assert_close(out, (image1 + resized_image2) * 0.5)


def test_image2_is_resized_before_it_is_repeated():
    """Resizing after repeating would copy a large image2 across the whole batch."""
    image1 = images(16, 0.1, size=4)
    image2 = images(2, 0.5, size=64)
    resized_batches = []

    real_upscale = comfy.utils.common_upscale

    def spy(samples, width, height, upscale_method, crop):
        resized_batches.append(samples.shape[0])
        return real_upscale(samples, width, height, upscale_method, crop)

    with mock.patch.object(comfy.utils, "common_upscale", spy):
        out = Blend.execute(image1, image2, 0.5, "normal").result[0]

    assert resized_batches == [2]
    assert out.shape == (16, 4, 4, 3)


def test_a_single_image_is_not_repeated():
    """A lone image broadcasts against the batch, so it never needs copying."""
    image1 = images(1, 0.1)
    image2 = images(3, 0.5)
    repeated = []

    real_repeat = comfy.utils.repeat_to_batch_size

    def spy(tensor, batch_size, dim=0):
        repeated.append(tensor.shape[0])
        return real_repeat(tensor, batch_size, dim)

    with mock.patch.object(comfy.utils, "repeat_to_batch_size", spy):
        out = Blend.execute(image1, image2, 0.5, "normal").result[0]

    assert repeated == []
    assert out.shape == (3, 4, 4, 3)
