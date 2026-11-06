import numpy as np
import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy_extras.nodes_mesh_postprocess import _sample_voxel_attrs_per_texel, bake_texture_from_voxel_fn


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("channels", [3, 6])
def test_voxel_texture_interpolation_and_nearest_fallback(dtype, channels):
    coords = torch.tensor([[0, 0, 0], [1, 0, 0]])
    colors = torch.tensor([
        [0.0, 0.25, 0.5, 0.75, 1.0, 0.0],
        [1.0, 0.75, 0.5, 0.25, 0.0, 1.0],
    ], dtype=dtype)[:, :channels].requires_grad_()
    original = colors.detach().clone()
    positions = np.array([[
        [0.0, -0.25, -0.25],
        [10.0, -0.25, -0.25],
        [-0.25, -0.25, -0.25],
    ]], dtype=np.float32)
    mask = np.array([[True, True, False]])

    result = _sample_voxel_attrs_per_texel(positions, mask, coords, colors, 2)

    assert result.dtype == np.float32
    np.testing.assert_allclose(result[0, 0], np.full(channels, 0.5))
    np.testing.assert_allclose(result[0, 1], original[1].float().numpy())
    np.testing.assert_array_equal(result[0, 2], np.zeros(channels))
    assert colors.dtype == dtype
    assert torch.equal(colors.detach(), original)


def test_bfloat16_voxel_texture_with_empty_mask():
    result = _sample_voxel_attrs_per_texel(
        np.zeros((2, 2, 3), dtype=np.float32),
        np.zeros((2, 2), dtype=bool),
        torch.zeros((1, 3), dtype=torch.int64),
        torch.ones((1, 6), dtype=torch.bfloat16),
        2,
    )

    assert result.shape == (2, 2, 6)
    assert result.dtype == np.float32
    assert not result.any()


@pytest.mark.parametrize("channels", [3, 6])
def test_bake_bfloat16_voxel_texture(channels):
    """Bake bfloat16 RGB/PBR attributes through rasterization and seam filling."""
    vertices = torch.tensor([[-0.25, -0.25, 0.0], [0.25, -0.25, 0.0], [-0.25, 0.25, 0.0]])
    faces = torch.tensor([[0, 1, 2]])
    uvs = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    coords = torch.tensor([[0, 0, 0]])
    colors = torch.tensor([[0.25, 0.5, 0.75, 0.5, 0.25, 1.0]], dtype=torch.bfloat16)[:, :channels]

    _, _, _, texture, mr = bake_texture_from_voxel_fn(vertices, faces, coords, colors, 2, 8, uvs)

    assert texture.shape == (8, 8, 3)
    assert texture.dtype == torch.float32
    torch.testing.assert_close(texture, torch.tensor([0.25, 0.5, 0.75]).expand_as(texture))
    if channels == 6:
        torch.testing.assert_close(mr, torch.tensor([0.0, 0.25, 0.5]).expand_as(texture))
    else:
        assert mr is None
