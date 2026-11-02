import pytest
import torch

from comfy.cli_args import args as cli_args

cli_args.cpu = True

from comfy_extras.nodes_post_processing import Blur, Sharpen  # noqa: E402


@pytest.mark.parametrize("node, kwargs", [
    (Blur, {"blur_radius": 3, "sigma": 1.0}),
    (Sharpen, {"sharpen_radius": 3, "sigma": 1.0, "alpha": 0.1}),
])
@pytest.mark.parametrize("height, width", [(1, 1), (1, 4), (4, 1), (3, 3)])
def test_filter_keeps_small_constant_image(node, kwargs, height, width):
    image = torch.full((1, height, width, 3), 0.5)

    result = node.execute(image, **kwargs).result[0]

    assert result.shape == image.shape
    assert result.dtype == image.dtype
    assert torch.allclose(result, image, atol=1e-6)


@pytest.mark.parametrize("node, kwargs, expected", [
    (Blur, {"blur_radius": 3, "sigma": 1.0}, [0.4147475, 0.5, 0.5852525]),
    (Sharpen, {"sharpen_radius": 3, "sigma": 1.0, "alpha": 0.1}, [0.0852527, 0.5, 0.9147478]),
])
def test_filter_applies_kernel_to_small_image(node, kwargs, expected):
    image = torch.tensor([0.25, 0.5, 0.75]).reshape(1, 1, 3, 1)

    result = node.execute(image, **kwargs).result[0]

    assert torch.allclose(result.flatten(), torch.tensor(expected), atol=1e-6)


@pytest.mark.parametrize("node, kwargs, expected_corner", [
    (Blur, {"blur_radius": 1, "sigma": 1.0}, 0.3356464),
    (Sharpen, {"sharpen_radius": 1, "sigma": 1.0, "alpha": 0.1}, 0.1643536),
])
def test_filter_keeps_reflection_for_large_enough_image(node, kwargs, expected_corner):
    image = 0.25 + torch.arange(16, dtype=torch.float32).reshape(1, 4, 4, 1) / 32

    result = node.execute(image, **kwargs).result[0]

    assert result[0, 0, 0, 0] == pytest.approx(expected_corner, abs=1e-6)
