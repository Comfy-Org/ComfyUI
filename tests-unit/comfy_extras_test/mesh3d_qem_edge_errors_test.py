"""QEM edge cost in qem_decimate._edge_errors: the quadric v^T Q v plus the edge-length term."""

import pytest
import torch

from comfy.cli_args import args as cli_args

# Model management probes the device on import, before pytest fixtures run.
_original_cpu = cli_args.cpu
try:
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        cli_args.cpu = True
    from comfy_extras.mesh3d.postprocess import qem_decimate
finally:
    cli_args.cpu = _original_cpu


@pytest.mark.parametrize("placement_mode", ["midpoint", "qem"])
def test_edge_cost_is_the_quadric_error_plus_the_length_term(placement_mode):
    g = torch.Generator().manual_seed(0)
    verts = torch.rand(200, 3, generator=g, dtype=torch.float64)
    planes = torch.randn(200, 3, 4, generator=g, dtype=torch.float64)
    Q = planes.transpose(1, 2) @ planes                  # sums of plane quadrics, symmetric PSD
    edges = torch.randint(0, 200, (500, 2), generator=g)
    edges = edges[edges[:, 0] != edges[:, 1]]
    cfg = qem_decimate.QEMConfig(placement_mode=placement_mode)
    big = torch.tensor(1e30, dtype=torch.float64)

    opt, cost, _ = qem_decimate._edge_errors(verts, Q, edges, torch.tensor(1e-6, dtype=torch.float64),
                                             big, torch.tensor(1.0, dtype=torch.float64), cfg)

    v4 = torch.cat([opt, torch.ones(len(opt), 1, dtype=torch.float64)], dim=1)
    Qe = Q[edges].sum(dim=1)
    el = torch.norm(verts[edges[:, 1]] - verts[edges[:, 0]], dim=-1)
    expected = torch.einsum("ei,eij,ej->e", v4, Qe, v4).abs() + cfg.lambda_edge_length * el * el
    torch.testing.assert_close(cost, expected)
