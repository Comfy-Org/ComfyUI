"""Torch UV packer (the path used without numba): the batched placement search must pick
exactly the spot the exhaustive search would."""

import numpy as np
import pytest
import torch

from comfy.cli_args import args as cli_args

# Model management probes the device on import, before pytest fixtures run.
_original_cpu = cli_args.cpu
try:
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        cli_args.cpu = True
    from comfy_extras.mesh3d.uv_unwrap import pack
finally:
    cli_args.cpu = _original_cpu


def _charts(seed, n):
    """Fans of random triangles: mostly small, some large (strided prescreen), a few
    collinear slivers (empty bitmaps)."""
    rng = np.random.default_rng(seed)
    uvs, faces, uv_off, f_off, a3, auv = [], [], [0], [0], [], []
    for _ in range(n):
        kind = rng.random()
        nv = int(rng.integers(3, 10)) if kind < 0.8 else int(rng.integers(30, 120))
        pts = rng.random((nv, 2)) * (rng.random() * 3 + 0.05)
        if kind > 0.95:
            pts[:, 1] = pts[:, 0] * 0.5
        nf = nv - 2
        tri = np.stack([np.zeros(nf, np.int64), np.arange(1, nf + 1), np.arange(2, nf + 2)], 1)
        e1, e2 = pts[tri[:, 1]] - pts[tri[:, 0]], pts[tri[:, 2]] - pts[tri[:, 0]]
        area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum()
        uvs.append(pts)
        faces.append(tri)
        uv_off.append(uv_off[-1] + nv)
        f_off.append(f_off[-1] + nf)
        auv.append(area)
        a3.append(area * (rng.random() * 2 + 0.5))
    return (np.concatenate(uvs), np.array(uv_off), np.concatenate(faces), np.array(f_off),
            np.array(a3), np.array(auv))


def _exhaustive(atlas, lin, dims, cxy, n_sky, cur_w, cur_h, ar_p, exact_args):
    pix0, dim0, dim1 = exact_args
    best = pack._best_placement_exact(atlas, pix0, dim0, dim1, cxy.permute(1, 2, 0), n_sky,
                                      cur_w, cur_h, cxy.device)
    return tuple(best.tolist())


@pytest.mark.parametrize("fast_exact", [256, 1])
@pytest.mark.parametrize("padding,texels_per_unit", [(0, 24.0), (1, 40.0), (2, 90.0)])
def test_torch_packer_matches_exhaustive_placement_search(monkeypatch, padding, texels_per_unit,
                                                          fast_exact):
    monkeypatch.setattr(pack, "_HAVE_NUMBA_PACK", False)
    monkeypatch.setattr(pack, "_FAST_EXACT", fast_exact)    # 1 forces the in-search fallback
    charts = _charts(padding, 60)

    fast = pack.pack_bitmap_concat(*charts, texels_per_unit=texels_per_unit, padding_texels=padding)
    monkeypatch.setattr(pack, "_best_placement_torch", _exhaustive)
    ref = pack.pack_bitmap_concat(*charts, texels_per_unit=texels_per_unit, padding_texels=padding)

    for got, want in zip(fast, ref):
        np.testing.assert_array_equal(got, want)
