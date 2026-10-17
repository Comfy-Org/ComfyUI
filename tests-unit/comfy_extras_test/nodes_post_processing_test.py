import sys

import torch

import comfy.options

comfy.options.enable_args_parsing()
sys.argv = [sys.argv[0], "--cpu"]

from comfy_extras.nodes_post_processing import Blend  # noqa: E402


def test_image_blend_difference_uses_absolute_difference():
    image1 = torch.tensor([[[[0.2, 0.8, 0.4]]]])
    image2 = torch.tensor([[[[0.7, 0.3, 0.4]]]])

    result = Blend.blend_mode(image1, image2, "difference")

    expected = torch.tensor([[[[0.5, 0.5, 0.0]]]])
    assert torch.allclose(result, expected)
