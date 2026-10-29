import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

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


def test_blend_keeps_spatial_resize_for_unequal_batches():
    """Matching the batch must not stop image2 from being resized to image1."""
    image1 = images(1, 0.1, size=4)
    image2 = images(3, 0.5, size=8)

    out = Blend.execute(image1, image2, 0.5, "normal").result[0]

    assert out.shape == (3, 4, 4, 3)


def test_equal_batches_are_unchanged():
    """The common case of two equally sized batches must not be altered."""
    image1 = images(2, 0.1)
    image2 = images(2, 0.5)

    out = Blend.execute(image1, image2, 0.5, "normal").result[0]

    expected = Blend.execute(image1, image2, 0.5, "normal").result[0]
    torch.testing.assert_close(out, expected)
    assert out.shape == image1.shape
