import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy.ldm.trellis2.flexgemm import sparse_pool3d_mean  # noqa: E402


def _pool_inputs(n, c, extent, factor, device, dtype=torch.float32):
    torch.manual_seed(0)
    coords = torch.cat(
        [torch.zeros(n, 1, dtype=torch.int32), torch.randint(0, extent, (n, 3), dtype=torch.int32)],
        dim=1,
    ).to(device)
    feats = torch.randn(n, c, device=device, dtype=dtype)
    return feats, coords, (extent,) * 3, factor


def test_sparse_pool3d_mean_pools_to_reference_values():
    feats, coords, shape, factor = _pool_inputs(512, 8, 16, 4, "cpu")

    pooled, _out_coords, out_shape, index = sparse_pool3d_mean(feats, coords, shape, factor)

    assert out_shape == (4, 4, 4)

    flat = ((coords[:, 0].long() * 4 + coords[:, 1].long() // factor) * 4
            + coords[:, 2].long() // factor) * 4 + coords[:, 3].long() // factor
    unique_flat, index_ref = torch.unique(flat, return_inverse=True)
    assert torch.equal(index, index_ref)

    reference = torch.zeros((unique_flat.shape[0], feats.shape[-1]))
    reference.index_add_(0, index_ref, feats)
    counts = torch.zeros((unique_flat.shape[0],))
    counts.index_add_(0, index_ref, torch.ones(index_ref.shape[0]))
    reference /= counts.unsqueeze(-1)

    assert torch.allclose(pooled, reference, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sparse_pool3d_mean_is_deterministic_on_cuda():
    feats, coords, shape, factor = _pool_inputs(200_000, 64, 64, 4, "cuda")

    runs = [sparse_pool3d_mean(feats, coords, shape, factor)[0] for _ in range(4)]

    assert all(torch.equal(runs[0], r) for r in runs[1:])
