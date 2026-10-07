"""Optional Prism top-k/top-p kernels, imported only for sparse mode."""
import torch
import torch.nn.functional as F


def sparse_attention(q, k, v, heads, grid, sparsity, cdf_threshold):
    from .kernels.bsa_interface import flash_attn_bsa_3d
    temporal, height, width = grid
    padding = tuple((-value) % 4 for value in grid)
    padded_grid = tuple(a + b for a, b in zip(grid, padding))
    batch, _, dim = q.shape
    depth = dim // heads
    streams = [x.reshape(batch, -1, heads, depth).transpose(1, 2) for x in (q, k, v)]
    valid = None
    if any(padding):
        streams = [F.pad(x.reshape(batch, heads, temporal, height, width, depth),
            (0, 0, 0, padding[2], 0, padding[1], 0, padding[0])).reshape(batch, heads, -1, depth).contiguous() for x in streams]
        t = torch.arange(padded_grid[0], device=q.device)
        h = torch.arange(padded_grid[1], device=q.device)
        w = torch.arange(padded_grid[2], device=q.device)
        valid = ((t[:, None, None] < temporal) & (h[None, :, None] < height)
                 & (w[None, None, :] < width)).reshape(-1).contiguous()
    out = flash_attn_bsa_3d(*streams, padded_grid, padded_grid, sparsity=sparsity,
        cdf_threshold=cdf_threshold, chunk_3d_shape_q=(4,4,4), chunk_3d_shape_k=(4,4,4), valid_mask=valid)
    if valid is not None:
        out = out.reshape(batch, heads, *padded_grid, depth)[:, :, :temporal, :height, :width]
        out = out.contiguous().reshape(batch, heads, -1, depth)
    return out.transpose(1, 2).flatten(2)
