"""Tests for comfy.ldm.hidream_o1.utils.resize_tensor."""

import pytest
import torch

from comfy.ldm.hidream_o1.utils import resize_tensor


def test_normal_square_image_fits_area():
    out = resize_tensor(torch.rand(1, 3, 512, 512), 384, 32)
    assert out.shape == (1, 3, 384, 384)
    assert out.shape[-1] % 32 == 0 and out.shape[-2] % 32 == 0


def test_normal_rectangle_keeps_existing_behavior():
    out = resize_tensor(torch.rand(1, 3, 384, 768), 384, 32)
    assert out.shape == (1, 3, 256, 512)


@pytest.mark.parametrize("thin", [16, 20, 100])
def test_thin_panorama_does_not_divide_by_zero(thin):
    out = resize_tensor(torch.rand(1, 3, 4000, thin), 384, 32)
    assert out.shape[-1] % 32 == 0 and out.shape[-2] % 32 == 0
    # The long edge must stay bounded: 16:1 aspect crop keeps the
    # intermediate from exploding for extreme panoramas.
    assert out.shape[-1] <= 16 * out.shape[-2] * 2


def test_extreme_aspect_is_cropped_not_stretched():
    # 4000x20 used to land on a 5408-wide target before the aspect crop.
    out = resize_tensor(torch.rand(1, 3, 4000, 20), 384, 32)
    assert out.shape[-1] < 2000


def test_wide_orientation_crops_width():
    out = resize_tensor(torch.rand(1, 3, 20, 4000), 384, 32)
    assert out.shape[-2] < 2000
    assert out.shape[-1] % 32 == 0 and out.shape[-2] % 32 == 0
