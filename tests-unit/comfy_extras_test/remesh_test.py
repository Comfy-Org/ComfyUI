import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy_extras.mesh3d.postprocess.remesh import remesh_narrow_band_dc


def _cube():
    v = torch.tensor([[x, y, z] for x in (-0.25, 0.25) for y in (-0.25, 0.25) for z in (-0.25, 0.25)])
    f = torch.tensor([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return v, f


def test_remesh_ignores_non_finite_vertices():
    v, f = _cube()
    clean_v, clean_f, _ = remesh_narrow_band_dc(v, f, resolution=32, scale=0.9, center=torch.zeros(3))

    # extra faces on NaN/inf vertices, as a bad decode can produce
    bad_v = torch.cat([v, torch.tensor([[float("nan"), 0.0, 0.0], [float("inf"), 0.0, 0.0]])])
    bad_f = torch.cat([f, torch.tensor([[0, 1, 8], [2, 3, 9]])])
    rv, rf, _ = remesh_narrow_band_dc(bad_v, bad_f, resolution=32, scale=0.9, center=torch.zeros(3))

    assert rf.shape[0] > 0
    assert torch.isfinite(rv).all()
    assert torch.equal(rv, clean_v) and torch.equal(rf, clean_f)
