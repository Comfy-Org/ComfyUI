from typing import Optional, Tuple
import torch

import comfy.model_management


def compute_kernel_offsets(Kw, Kh, Kd, Dw, Dh, Dd, device):
    """Kernel spatial offsets in the same order as the CUDA/Triton kernels."""
    offsets = []
    for vx in range(Kw):
        for vy in range(Kh):
            for vz in range(Kd):
                offsets.append((vx * Dw, vy * Dh, vz * Dd))
    return torch.tensor(offsets, device=device, dtype=torch.int32)


class TorchHashMap:
    """Sorted-array hashmap backed by torch.searchsorted."""

    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        self.sorted_keys, order = torch.sort(keys.to(torch.long))
        self.sorted_vals = values[order]
        self._n = self.sorted_keys.numel()

    # Chunk size for lookup_flat, caps each transient to ~CHUNK rows.
    _LOOKUP_CHUNK = 1 << 23   # 8M rows ≈ 64 MB per int64 temp

    def lookup_flat(self, flat_keys: torch.Tensor) -> torch.Tensor:
        N = flat_keys.shape[0]
        out = torch.full((N,), -1, device=flat_keys.device, dtype=torch.int32)
        if self._n == 0 or N == 0:
            return out
        for s in range(0, N, self._LOOKUP_CHUNK):
            e = min(s + self._LOOKUP_CHUNK, N)
            flat_chunk = flat_keys[s:e].to(torch.long)
            idx = torch.searchsorted(self.sorted_keys, flat_chunk)
            in_range = idx < self._n
            idx.clamp_(max=self._n - 1)  # reuse idx as the "safe" index
            found = in_range & (self.sorted_keys[idx] == flat_chunk)
            if found.any():
                found_idx = found.nonzero(as_tuple=True)[0]
                out[s + found_idx] = self.sorted_vals[idx[found_idx]].to(torch.int32)
        return out


def build_submanifold_neighbor_map(
    hashmap,
    coords: torch.Tensor,
    W, H, D,
    Kw, Kh, Kd,
    Dw, Dh, Dd,
):
    # neighbor[i, v] = index of the voxel at voxel i's coord + kernel-offset v, or -1.
    # Chunked over voxels so the [chunk, V, 3] candidate transient stays bounded.
    device = coords.device
    M = coords.shape[0]
    offsets = compute_kernel_offsets(Kw, Kh, Kd, Dw, Dh, Dd, device)             # [V, 3]
    V = offsets.shape[0]
    center = torch.tensor([(Kw // 2) * Dw, (Kh // 2) * Dh, (Kd // 2) * Dd], dtype=torch.int32, device=device)
    WHD, HD = W * H * D, H * D

    neighbor = torch.empty((M, V), dtype=torch.int32, device=device)
    # ~V*40 bytes/voxel of transient (int64 cand + flat + masks); cap at ~0.5 GB.
    chunk = max(1, min(M, int(0.5 * (1024 ** 3) / (V * 40))))

    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        b = coords[s:e, 0].long()
        cand = coords[s:e, 1:4][:, None, :] + offsets[None, :, :] - center  # [c, V, 3]
        x, y, z = cand[..., 0], cand[..., 1], cand[..., 2]
        in_bounds = (x >= 0) & (x < W) & (y >= 0) & (y < H) & (z >= 0) & (z < D)   # [c, V]
        flat = x.long().mul_(HD)
        flat.add_(y.long().mul_(D)).add_(z).add_(b[:, None] * WHD)
        flat.masked_fill_(~in_bounds, -1)                                         # OOB -> guaranteed miss
        neighbor[s:e] = hashmap.lookup_flat(flat.reshape(-1)).view(e - s, V)
    return neighbor

def get_recommended_chunk_mem(
    device=None,
    safety_fraction: float = 0.2,
    min_gb: float = 0.25,
    max_gb: float = 0.5,
):
    """Pick a chunk-memory budget (in GB) for sparse conv batching."""
    free_gb = comfy.model_management.get_free_memory(device) / (1024 ** 3)
    return max(min_gb, min(free_gb * safety_fraction, max_gb))

def sparse_submanifold_conv3d(
    feats: torch.Tensor,
    coords: torch.Tensor,
    shape: tuple,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    neighbor_cache: Optional[torch.Tensor],
    dilation: tuple,
    cache_neighbor_map: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if feats.shape[0] == 0:
        Co = weight.shape[0]
        return torch.empty((0, Co), device=feats.device, dtype=feats.dtype), None

    W, H, D = shape

    Co, Kw, Kh, Kd, Ci = weight.shape
    V = Kw * Kh * Kd
    device = feats.device

    hashmap = None
    if neighbor_cache is None:
        b_stride = W * H * D
        x_stride = H * D
        y_stride = D
        z_stride = 1

        flat_keys = coords[:, 0].long() * b_stride
        flat_keys.add_(coords[:, 1], alpha=x_stride)
        flat_keys.add_(coords[:, 2], alpha=y_stride)
        flat_keys.add_(coords[:, 3], alpha=z_stride)
        vals = torch.arange(coords.shape[0], dtype=torch.int32, device=device)
        hashmap = TorchHashMap(flat_keys, vals)

        if cache_neighbor_map:
            neighbor = build_submanifold_neighbor_map(
                hashmap, coords, W, H, D, Kw, Kh, Kd,
                dilation[0], dilation[1], dilation[2]
            )
        else:
            neighbor = None
    else:
        neighbor = neighbor_cache

    N_pts = feats.shape[0]

    weight_T = weight.view(Co, V * Ci).T

    output = torch.empty(N_pts, Co, device=device, dtype=feats.dtype)

    # Chunk over voxels to bound the (chunk, V, Ci) gather.
    max_chunk_mem_gb = get_recommended_chunk_mem(device)
    mem_per_row = V * Ci * feats.element_size()
    max_chunk_mem = max_chunk_mem_gb * (1024 ** 3)
    chunk_size = max(1, int(max_chunk_mem / mem_per_row))
    chunk_size = min(chunk_size, N_pts)

    for start in range(0, N_pts, chunk_size):
        end = min(start + chunk_size, N_pts)
        actual_chunk = end - start

        if neighbor is None:
            neighbor_chunk = build_submanifold_neighbor_map(
                hashmap, coords[start:end], W, H, D, Kw, Kh, Kd,
                dilation[0], dilation[1], dilation[2]
            )
        else:
            neighbor_chunk = neighbor[start:end]

        chunk_idx = neighbor_chunk.clamp_min(0)
        gathered = feats[chunk_idx]                         # (chunk, V, Ci)
        gathered.masked_fill_(neighbor_chunk[:, :, None] < 0, 0)
        gathered_flat = gathered.view(actual_chunk, V * Ci)
        output[start:end] = torch.matmul(gathered_flat, weight_T)  # (chunk, V*Ci) @ (V*Ci, Co)

    if bias is not None:
        output += bias.unsqueeze(0).to(output.dtype)

    return output, neighbor


def sparse_pool3d_mean(
    feats: torch.Tensor,
    coords: torch.Tensor,
    shape: tuple,
    factor: int,
) -> Tuple[torch.Tensor, torch.Tensor, tuple, torch.Tensor]:
    """Average-pool a sparse volume by `factor` along the three spatial axes.

    coords are (batch, x, y, z) and shape is the spatial extent (X, Y, Z), as for
    sparse_submanifold_conv3d. Returns the pooled (feats, coords, shape) plus the
    fine->coarse index, which is what sparse_upsample3d_nearest needs to invert the pooling.
    """
    batch = coords[:, 0].long()
    fine = coords[:, 1:4].long()
    out_x, out_y, out_z = ((int(s) + factor - 1) // factor for s in shape)

    coarse = fine.div(factor, rounding_mode="floor")
    flat = ((batch * out_x + coarse[:, 0]) * out_y + coarse[:, 1]) * out_z + coarse[:, 2]
    unique_flat, index = torch.unique(flat, return_inverse=True)

    pooled = torch.zeros((unique_flat.shape[0], feats.shape[-1]), device=feats.device, dtype=feats.dtype)
    pooled.index_add_(0, index, feats)
    counts = torch.zeros((unique_flat.shape[0],), device=feats.device, dtype=feats.dtype)
    counts.index_add_(0, index, torch.ones_like(index, dtype=feats.dtype))
    pooled /= counts.unsqueeze(-1)

    z = unique_flat % out_z
    rest = unique_flat.div(out_z, rounding_mode="floor")
    y = rest % out_y
    rest = rest.div(out_y, rounding_mode="floor")
    x = rest % out_x
    b = rest.div(out_x, rounding_mode="floor")
    out_coords = torch.stack([b, x, y, z], dim=-1).to(torch.int32)

    return pooled, out_coords, (out_x, out_y, out_z), index


def sparse_upsample3d_nearest(feats: torch.Tensor, pool_index: torch.Tensor) -> torch.Tensor:
    """Invert sparse_pool3d_mean's spatial mapping: each fine voxel takes its parent's feature.

    `pool_index` is the index returned by the matching sparse_pool3d_mean call, so the
    output is aligned with that call's input coords (the UNet's skip coords).
    """
    return feats[pool_index]
