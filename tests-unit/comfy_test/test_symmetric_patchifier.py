import torch

from comfy.ldm.lightricks.symmetric_patchifier import (
    SymmetricPatchifier,
    latent_to_pixel_coords,
)


def _guard_no_tensor_from_python_data(monkeypatch):
    real_tensor = torch.tensor

    def guarded(data, *args, **kwargs):
        raise AssertionError(
            "must not build a tensor from python scalars on every call; this is an "
            "H2D copy that breaks CUDA graph capture"
        )

    monkeypatch.setattr(torch, "tensor", guarded)
    return real_tensor


def test_get_latent_coords_end_matches_patch_size_without_host_tensor(monkeypatch):
    patchifier = SymmetricPatchifier(patch_size=2, start_end=True)
    _guard_no_tensor_from_python_data(monkeypatch)

    coords = patchifier.get_latent_coords(
        latent_num_frames=2, latent_height=4, latent_width=4, batch_size=1,
        device=torch.device("cpu"),
    )

    start = coords[..., 0]
    end = coords[..., 1]
    for axis, patch in enumerate(patchifier.patch_size):
        assert torch.equal(end[:, axis], start[:, axis] + patch)


def test_latent_to_pixel_coords_scales_each_axis_without_host_tensor(monkeypatch):
    latent_coords = torch.tensor([[[0, 1], [0, 2], [0, 3]]])
    real_tensor = _guard_no_tensor_from_python_data(monkeypatch)

    pixel_coords = latent_to_pixel_coords(latent_coords, scale_factors=(8, 32, 32))

    expected = real_tensor([[[0, 8], [0, 64], [0, 96]]])
    assert torch.equal(pixel_coords, expected)
