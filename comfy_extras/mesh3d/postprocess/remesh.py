"""Narrow-band Dual Contouring remeshing.

Re-extracts a mesh from a sparse narrow-band voxel grid around the input
surface (pure-PyTorch approximation of CuMesh's remesh_narrow_band_dc).
Coarse-to-fine voxelise the band, sample SDF/UDF at voxel corners, dual
contour (optionally QEF / Manifold DC), then optionally project back,
filter components, fix poles, smooth, and interpolate vertex colors.
"""
from __future__ import annotations

import functools
import math
from typing import Optional, Tuple

import numpy as np
import torch
import scipy.spatial
import comfy.utils
from tqdm import tqdm as _tqdm
from comfy.model_management import throw_exception_if_processing_interrupted

from .qem_decimate import _sorted_edge_halfedges


# Point-to-triangle distance (exact, vectorised)

def _point_tri_closest(points: torch.Tensor, tris: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact closest point + squared distance per (point, triangle) pair; points (N,3), tris (N,3,3)."""
    a = tris[:, 0]
    b = tris[:, 1]
    c = tris[:, 2]
    ab = b - a
    ac = c - a
    ap = points - a

    d1 = (ab * ap).sum(-1)
    d2 = (ac * ap).sum(-1)

    region_A = (d1 <= 0) & (d2 <= 0)

    bp = points - b
    d3 = (ab * bp).sum(-1)
    d4 = (ac * bp).sum(-1)
    region_B = (d3 >= 0) & (d4 <= d3)

    cp = points - c
    d5 = (ab * cp).sum(-1)
    d6 = (ac * cp).sum(-1)
    region_C = (d6 >= 0) & (d5 <= d6)

    # Edge AB
    vc = d1 * d4 - d3 * d2
    region_AB = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    v_ab = d1 / (d1 - d3 + 1e-20)
    closest_AB = a + v_ab.unsqueeze(-1) * ab

    # Edge AC
    vb = d5 * d2 - d1 * d6
    region_AC = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    v_ac = d2 / (d2 - d6 + 1e-20)
    closest_AC = a + v_ac.unsqueeze(-1) * ac

    # Edge BC
    va = d3 * d6 - d5 * d4
    region_BC = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    v_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6) + 1e-20)
    closest_BC = b + v_bc.unsqueeze(-1) * (c - b)

    # Face interior (barycentric)
    denom = va + vb + vc + 1e-20
    v_face = vb / denom
    w_face = vc / denom
    closest_face = a + v_face.unsqueeze(-1) * ab + w_face.unsqueeze(-1) * ac

    # Combine by mask via in-place where (out= aliases input, no per-step alloc)
    closest = closest_face                                         # fresh; safe to mutate
    torch.where(region_BC.unsqueeze(-1), closest_BC, closest, out=closest)
    torch.where(region_AC.unsqueeze(-1), closest_AC, closest, out=closest)
    torch.where(region_AB.unsqueeze(-1), closest_AB, closest, out=closest)
    torch.where(region_C .unsqueeze(-1), c,          closest, out=closest)
    torch.where(region_B .unsqueeze(-1), b,          closest, out=closest)
    torch.where(region_A .unsqueeze(-1), a,          closest, out=closest)

    diff = points - closest
    return closest, (diff * diff).sum(-1)


def _build_centroid_tree(tri_verts: torch.Tensor):
    """scipy cKDTree over triangle centroids; build once and reuse across _udf_exact calls.
    balanced_tree/compact_nodes off: ~2.4x faster build (and faster queries on near-uniform
    centroid clouds) with identical exact-kNN results."""
    return scipy.spatial.cKDTree(tri_verts.mean(dim=1).detach().cpu().numpy(),
                                 balanced_tree=False, compact_nodes=False)


def _udf_exact(query_points: torch.Tensor, tri_verts: torch.Tensor,
               k: int = 8, chunk: int = 262144, tree=None):
    """Exact UDF (no max_dist cap) via centroid kNN; returns (dist [N], closest [N,3], tri_idx [N]). Pass prebuilt `tree` to skip rebuild.

    k=8 nearest centroids before the exact point-triangle test: on dense meshes the true
    closest triangle is essentially always within the first few neighbours. Measured vs k=16:
    bit-identical topology, ~0.003-voxel RMS sub-voxel drift, ~15% faster overall."""
    device = query_points.device
    F = tri_verts.shape[0]
    kq = int(min(k, F))
    if tree is None:
        tree = _build_centroid_tree(tri_verts)
    _, cand = tree.query(query_points.detach().cpu().numpy(), k=kq, workers=-1)
    if cand.ndim == 1:
        cand = cand[:, None]
    cand = np.ascontiguousarray(cand)

    N = query_points.shape[0]
    out_d = torch.empty(N, device=device, dtype=query_points.dtype)
    out_c = torch.empty(N, 3, device=device, dtype=query_points.dtype)
    out_t = torch.empty(N, dtype=torch.long, device=device)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        n = e - s
        ci = torch.from_numpy(cand[s:e]).to(device).long()
        tri = tri_verts[ci].reshape(n * kq, 3, 3)
        P = query_points[s:e][:, None, :].expand(-1, kq, -1).reshape(n * kq, 3)
        closest, d2 = _point_tri_closest(P, tri)
        d2 = d2.reshape(n, kq)
        closest = closest.reshape(n, kq, 3)
        best = d2.argmin(dim=1)
        ar = torch.arange(n, device=device)
        out_d[s:e] = d2[ar, best].sqrt()
        out_c[s:e] = closest[ar, best]
        out_t[s:e] = ci[ar, best]
    return out_d, out_c, out_t


# UDF query via spatial hash on triangle AABBs

def _build_tri_spatial_hash(centroids: torch.Tensor, tri_radii: torch.Tensor,
                            cell_size: torch.Tensor):
    """Bucket triangles into `cell_size` cells (each tri into every cell its AABB touches); returns hash tuple."""
    device = centroids.device
    aabb_lo = (centroids - tri_radii.unsqueeze(-1))
    aabb_hi = (centroids + tri_radii.unsqueeze(-1))
    origin = aabb_lo.min(0)[0]
    extent = aabb_hi.max(0)[0] - origin
    dims = (extent / cell_size).long() + 2

    cell_lo = ((aabb_lo - origin) / cell_size).long().clamp(min=0)
    cell_hi = ((aabb_hi - origin) / cell_size).long()
    cell_hi = torch.minimum(cell_hi, dims - 1)

    # Cap span at 3 cells/axis to bound memory
    spans = (cell_hi - cell_lo + 1).clamp(max=3)
    n_per_tri = spans.prod(dim=-1)
    total = int(n_per_tri.sum().item())

    # Per-insertion local offset within each tri's cell box
    rep = torch.repeat_interleave(torch.arange(centroids.shape[0], device=device), n_per_tri)
    cum = torch.cat([torch.zeros(1, device=device, dtype=n_per_tri.dtype),
                     n_per_tri.cumsum(0)[:-1]])
    local = torch.arange(total, device=device) - cum[rep]
    sx = spans[rep, 0]
    sy = spans[rep, 1]

    lx = local % sx
    ly = (local // sx) % sy
    lz = local // (sx * sy)
    cx = cell_lo[rep, 0] + lx
    cy = cell_lo[rep, 1] + ly
    cz = cell_lo[rep, 2] + lz
    keys = (cx * dims[1] + cy) * dims[2] + cz

    sort_idx = keys.argsort()
    sorted_keys = keys[sort_idx]
    tri_per_cell = rep[sort_idx]

    unique_keys, counts = torch.unique_consecutive(sorted_keys, return_counts=True)
    cell_starts = torch.cat([torch.zeros(1, dtype=counts.dtype, device=device),
                             counts.cumsum(0)])
    return origin, dims, unique_keys, tri_per_cell, cell_starts, centroids, tri_radii


def _udf_query(query_points: torch.Tensor,
               tri_verts: torch.Tensor,
               hash_data,
               cell_size: torch.Tensor,
               max_dist: float,
               chunk_max: int = 4096,
               return_closest: bool = False,
               return_tri_idx: bool = False):
    """Capped UDF to nearest triangle (<= max_dist), optionally with closest point and/or tri index; chunk size is adaptive to hash density."""
    origin, dims, unique_keys, tri_per_cell, cell_starts, tri_centroids, tri_radii = hash_data
    device = query_points.device
    Q = query_points.shape[0]
    # Adaptive chunk: bound per-chunk candidate-gather memory by hash density
    avg_per_cell = tri_per_cell.numel() / max(1, unique_keys.numel())
    est_cands_per_query = max(1.0, avg_per_cell * 27)
    chunk = max(256, min(chunk_max, int(50_000_000 / est_cands_per_query)))
    out_d2 = torch.full((Q,), float(max_dist) ** 2, dtype=query_points.dtype, device=device)
    # Default closest_pt = query_pt itself, so a missed query's lerp is a no-op
    out_closest = (query_points.clone() if return_closest else None)
    out_tri = (torch.full((Q,), -1, dtype=torch.long, device=device)
               if return_tri_idx else None)

    rng = torch.tensor([-1, 0, 1], device=device, dtype=torch.long)
    offs = torch.stack(torch.meshgrid(rng, rng, rng, indexing="ij"), dim=-1).reshape(-1, 3)  # (27, 3)

    for cs in range(0, Q, chunk):
        ce = min(cs + chunk, Q)
        qp = query_points[cs:ce]
        q_cell = ((qp - origin) / cell_size).long()
        # Look up 27 neighbour cells per query
        n_cell = q_cell.unsqueeze(1) + offs.unsqueeze(0)             # (q, 27, 3)
        n_valid = ((n_cell >= 0) & (n_cell < dims)).all(-1)
        n_key = (n_cell[..., 0] * dims[1] + n_cell[..., 1]) * dims[2] + n_cell[..., 2]
        flat_key = n_key.reshape(-1).contiguous()
        ins = torch.searchsorted(unique_keys, flat_key)
        ins_c = ins.clamp(max=unique_keys.numel() - 1)
        found = (ins < unique_keys.numel()) & (unique_keys[ins_c] == flat_key) & n_valid.reshape(-1)
        cell_idx = torch.where(found, ins_c, torch.zeros_like(ins_c))
        c_starts = cell_starts[cell_idx]
        c_ends = cell_starts[cell_idx + 1]
        c_counts = (c_ends - c_starts) * found.long()
        rep_q = torch.repeat_interleave(
            torch.arange(qp.shape[0] * 27, device=device) // 27, c_counts)
        if rep_q.numel() == 0:
            continue
        total = rep_q.numel()
        slot_starts_per_pair = torch.cumsum(c_counts, dim=0) - c_counts
        per_pair_start = torch.repeat_interleave(c_starts, c_counts)
        slot_within = torch.arange(total, device=device) - torch.repeat_interleave(slot_starts_per_pair, c_counts)
        tri_indices = tri_per_cell[per_pair_start + slot_within]

        pts = qp[rep_q]
        # Centroid pre-cull (squared): drop where ||pts-centroid||-radius > max_dist
        diff = pts - tri_centroids[tri_indices]
        d2_cand = (diff * diff).sum(-1)
        thresh = max_dist + tri_radii[tri_indices]
        cull_keep = d2_cand < thresh * thresh
        rep_q = rep_q[cull_keep]
        pts = pts[cull_keep]
        tri_indices = tri_indices[cull_keep]
        if rep_q.numel() == 0:
            continue
        tri = tri_verts[tri_indices]
        closest, d2 = _point_tri_closest(pts, tri)

        # Min per query for this chunk.
        local_min = torch.full((qp.shape[0],), float(max_dist) ** 2,
                               dtype=query_points.dtype, device=device)
        local_min.scatter_reduce_(0, rep_q, d2, reduce="amin", include_self=True)
        # Only update where this chunk improved; ties may overwrite (any is valid)
        better = local_min < out_d2[cs:ce]
        out_d2[cs:ce] = torch.where(better, local_min, out_d2[cs:ce])
        if return_closest or return_tri_idx:
            ties = (d2 == local_min[rep_q]) & better[rep_q]
            if return_closest:
                out_closest[cs + rep_q[ties]] = closest[ties]
            if return_tri_idx:
                out_tri[cs + rep_q[ties]] = tri_indices[ties]

    out_d = out_d2.sqrt()
    extras = []
    if return_closest:
        extras.append(out_closest)
    if return_tri_idx:
        extras.append(out_tri)
    if extras:
        return (out_d, *extras)
    return out_d


# Sparse coarse-to-fine voxel grid in narrow band

def _build_narrow_band_voxels(verts: torch.Tensor, faces: torch.Tensor,
                              center: torch.Tensor, scale: float,
                              resolution: int, eps: float,
                              progress_callback=None) -> torch.Tensor:
    """Voxel coords (Nv,3) in 0..resolution-1 whose centre is within ~0.87 cell_size of the surface; also returns the kept cKDTree."""
    device = verts.device
    tri_verts = verts[faces.long()]
    # Exact UDF; build the centroid cKDTree once and reuse across refinement levels
    tree = _build_centroid_tree(tri_verts)

    base_resolution = resolution
    while base_resolution > 32 and base_resolution % 2 == 0:
        base_resolution //= 2

    rng = torch.arange(base_resolution, device=device, dtype=torch.long)
    coords = torch.stack(torch.meshgrid(rng, rng, rng, indexing="ij"), dim=-1).reshape(-1, 3)

    OFFSETS = torch.tensor([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ], dtype=torch.long, device=device)

    current_res = base_resolution
    while True:
        throw_exception_if_processing_interrupted()
        cell_size = scale / current_res
        pts = ((coords.float() + 0.5) / current_res - 0.5) * scale + center
        dists, _, _ = _udf_exact(pts, tri_verts, tree=tree)
        keep = dists < 0.87 * cell_size + eps
        coords = coords[keep]
        if progress_callback is not None:
            progress_callback()
        if current_res >= resolution:
            break
        current_res *= 2
        coords = coords * 2
        coords = (coords.unsqueeze(1) + OFFSETS.unsqueeze(0)).reshape(-1, 3)

    return coords, tree


# Dual Contouring

def _dual_contour(voxel_coords: torch.Tensor, corner_udf: torch.Tensor,
                  corner_keys: torch.Tensor,
                  resolution: int, scale: float, center: torch.Tensor,
                  tri_face_normals: Optional[torch.Tensor] = None,
                  qef_query=None,
                  corner_valid: Optional[torch.Tensor] = None,
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dual contour active voxels; returns (Nv,3) dual verts and (M,3) faces into them. QEF placement when tri_face_normals+qef_query given, else centroid of crossings."""
    device = voxel_coords.device
    Nv = voxel_coords.shape[0]
    # 8 corners per voxel, packed into a 1d key
    CORNER_OFFS = torch.tensor([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ], dtype=torch.long, device=device)

    corner_pos_per_voxel = voxel_coords.unsqueeze(1) + CORNER_OFFS.unsqueeze(0)  # (Nv, 8, 3)
    R1 = resolution + 1
    keys_per_voxel = (corner_pos_per_voxel[..., 0] * R1
                      + corner_pos_per_voxel[..., 1]) * R1 + corner_pos_per_voxel[..., 2]
    # Look up SDF index per corner; missing corners default to +1 (outside)
    idx_per_voxel = torch.searchsorted(corner_keys, keys_per_voxel.reshape(-1))
    idx_clamped = idx_per_voxel.clamp(max=corner_keys.numel() - 1)
    found = (idx_per_voxel < corner_keys.numel()) & (corner_keys[idx_clamped] == keys_per_voxel.reshape(-1))
    sd = torch.where(found, corner_udf[idx_clamped], torch.full_like(corner_udf[idx_clamped], 1.0))
    sd = sd.reshape(Nv, 8)         # surface at sign = 0

    # 12 voxel edges as (corner_a, corner_b) pairs (indices into the 8 corners above)
    EDGES = torch.tensor([
        [0, 1], [2, 3], [4, 5], [6, 7],   # x-axis edges
        [0, 2], [1, 3], [4, 6], [5, 7],   # y-axis
        [0, 4], [1, 5], [2, 6], [3, 7],   # z-axis
    ], dtype=torch.long, device=device)

    a_sd = sd[:, EDGES[:, 0]]              # (Nv, 12)
    b_sd = sd[:, EDGES[:, 1]]
    crosses = (a_sd * b_sd) < 0            # (Nv, 12) bool
    # Skip crossings touching an invalid corner (avoids fake faces at band edge)
    if corner_valid is not None:
        cv_per_voxel = torch.where(found, corner_valid[idx_clamped],
                                   torch.zeros_like(found)).reshape(Nv, 8)
        edge_valid = cv_per_voxel[:, EDGES[:, 0]] & cv_per_voxel[:, EDGES[:, 1]]
        crosses = crosses & edge_valid
        del cv_per_voxel, edge_valid
    del keys_per_voxel, idx_per_voxel, idx_clamped, found
    # Zero-crossing interp factor per edge
    t = a_sd / (a_sd - b_sd + 1e-20)
    t = t.clamp(0.0, 1.0).unsqueeze(-1)

    corner_world = (corner_pos_per_voxel.float() / resolution - 0.5) * scale + center.unsqueeze(0).unsqueeze(0)  # (Nv, 8, 3)
    a_pos = corner_world[:, EDGES[:, 0]]   # (Nv, 12, 3)
    b_pos = corner_world[:, EDGES[:, 1]]
    crossing_pts = torch.lerp(a_pos, b_pos, t)  # (Nv, 12, 3)
    del corner_pos_per_voxel, a_pos, b_pos, t

    # Default dual vert: centroid of crossings (also QEF/no-crossing fallback)
    crosses_f = crosses.float().unsqueeze(-1)
    crossing_sum = (crossing_pts * crosses_f).sum(dim=1)
    n_cross = crosses.float().sum(dim=1, keepdim=True).clamp_min(1.0)
    centroid_verts = crossing_sum / n_cross
    centre_world = ((voxel_coords.float() + 0.5) / resolution - 0.5) * scale + center.unsqueeze(0)
    has_cross = crosses.any(dim=1, keepdim=True)
    dual_verts = torch.where(has_cross, centroid_verts, centre_world)

    # QEF placement: minimise sum_i (n_i·(x-p_i))² via Tikhonov-regularised
    # normal equations (A+reg I)x=b; clamp to voxel bbox, else fall back to centroid.
    if tri_face_normals is not None and qef_query is not None:
        Nv = voxel_coords.shape[0]
        flat_pts = crossing_pts.reshape(-1, 3)
        flat_mask = crosses.reshape(-1)
        if flat_mask.any():
            query_pts = flat_pts[flat_mask]
            _, _, qef_tri_idx = qef_query(query_pts)
            # Missed queries get a zero normal (null constraint, ignored by solver)
            valid_q = qef_tri_idx >= 0
            normals_at_q = torch.zeros_like(query_pts)
            normals_at_q[valid_q] = tri_face_normals[qef_tri_idx[valid_q]]
            full_normals = torch.zeros((Nv * 12, 3), dtype=query_pts.dtype, device=device)
            full_normals[flat_mask] = normals_at_q
            n_per_edge = full_normals.reshape(Nv, 12, 3)

            # einsum sums into the 3x3 directly, skipping a big intermediate
            A = torch.einsum('vec,ved->vcd', n_per_edge, n_per_edge)        # (Nv, 3, 3)
            n_dot_p = (n_per_edge * crossing_pts).sum(dim=-1)               # (Nv, 12)
            b = torch.einsum('ve,vec->vc', n_dot_p, n_per_edge)             # (Nv, 3)

            # Tikhonov regularisation in-place (A, b are fresh einsum outputs)
            reg = 1e-2
            A.diagonal(dim1=-2, dim2=-1).add_(reg)
            b.add_(centroid_verts, alpha=reg)
            try:
                qef_solution = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
            except torch.linalg.LinAlgError:
                qef_solution = centroid_verts

            # Clamp QEF output to the voxel bbox
            lo = corner_world[:, 0]                                          # (Nv, 3) min corner
            hi = corner_world[:, 7]                                          # (Nv, 3) max corner
            in_box = (qef_solution >= lo).all(dim=-1) & (qef_solution <= hi).all(dim=-1)
            qef_solution = torch.where(in_box.unsqueeze(-1), qef_solution, centroid_verts)

            dual_verts = torch.where(has_cross, qef_solution, centre_world)
            del query_pts, qef_tri_idx, valid_q, normals_at_q, full_normals, n_per_edge
            del A, n_dot_p, b, qef_solution, lo, hi, in_box
        del flat_pts, flat_mask

    del crossing_pts, corner_world, crosses_f, crossing_sum, n_cross
    del centroid_verts, centre_world, has_cross

    # Topology: each crossing grid edge is shared by 4 voxels -> quad -> 2 tris.
    # NEIGHBOUR_OFFS lays out the 4 sharing voxels per axis; y-axis order is
    # reversed vs x/z to keep manifold winding around each shared edge.
    NEIGHBOUR_OFFS = torch.tensor([
        [[0, 0, 0], [0, -1, 0], [0, -1, -1], [0, 0, -1]],
        [[0, 0, 0], [0, 0, -1], [-1, 0, -1], [-1, 0, 0]],
        [[0, 0, 0], [-1, 0, 0], [-1, -1, 0], [0, -1, 0]],
    ], dtype=torch.long, device=device)

    # Min-corner +axis edge index per axis (slots 0/4/8 in EDGES)
    EDGE_OF_AXIS = torch.tensor([0, 4, 8], dtype=torch.long, device=device)

    # Sorted voxel-coord keys for neighbour lookup
    vox_dims = voxel_coords.max(dim=0)[0] + 2
    vox_key = (voxel_coords[:, 0] * vox_dims[1] + voxel_coords[:, 1]) * vox_dims[2] + voxel_coords[:, 2]
    sort_v = vox_key.argsort()
    sorted_vox_key = vox_key[sort_v]

    tris = []
    for axis in range(3):
        edge_idx = EDGE_OF_AXIS[axis]
        owner_mask = crosses[:, edge_idx]                 # (Nv,) bool
        if not owner_mask.any():
            continue
        owner_voxels = voxel_coords[owner_mask]           # (No, 3)
        a_sign = a_sd[owner_mask, edge_idx]               # (No,) sign at corner a
        nbrs = owner_voxels.unsqueeze(1) + NEIGHBOUR_OFFS[axis].unsqueeze(0)   # (No, 4, 3)
        nbr_keys = (nbrs[..., 0] * vox_dims[1] + nbrs[..., 1]) * vox_dims[2] + nbrs[..., 2]
        flat = nbr_keys.reshape(-1).contiguous()
        ins = torch.searchsorted(sorted_vox_key, flat)
        ins_c = ins.clamp(max=sorted_vox_key.numel() - 1)
        valid = (ins < sorted_vox_key.numel()) & (sorted_vox_key[ins_c] == flat)
        valid = valid.reshape(-1, 4).all(dim=1)
        if not valid.any():
            continue
        dual_indices = sort_v[ins_c].reshape(-1, 4)[valid]    # (Mv, 4)
        sign_a = a_sign[valid]
        # Winding: flip when corner a is outside (sign_a > 0) so normal points out
        d0 = dual_indices[:, 0]
        d1 = dual_indices[:, 1]
        d2 = dual_indices[:, 2]
        d3 = dual_indices[:, 3]
        flip = sign_a > 0
        t1a = torch.stack([d0, d1, d2], dim=1)
        t2a = torch.stack([d0, d2, d3], dim=1)
        t1b = torch.stack([d0, d2, d1], dim=1)
        t2b = torch.stack([d0, d3, d2], dim=1)
        t1 = torch.where(flip.unsqueeze(-1), t1b, t1a)
        t2 = torch.where(flip.unsqueeze(-1), t2b, t2a)
        tris.append(t1)
        tris.append(t2)

    if not tris:
        return dual_verts, torch.empty((0, 3), dtype=torch.long, device=device)
    new_faces = torch.cat(tris, dim=0)
    return dual_verts, new_faces


# Manifold Dual Contouring (Schaefer, Ju, Warren 2007)

@functools.lru_cache(maxsize=None)
def _build_mdc_lut() -> Tuple[torch.Tensor, torch.Tensor]:
    """Per 8-corner sign pattern: K (256,) patch count and group (256,12) patch id per edge (-1 if non-crossing)."""
    EDGE_PAIRS = [
        (0, 1), (2, 3), (4, 5), (6, 7),         # x-axis edges
        (0, 2), (1, 3), (4, 6), (5, 7),         # y-axis edges
        (0, 4), (1, 5), (2, 6), (3, 7),         # z-axis edges
    ]
    K = torch.zeros(256, dtype=torch.int64)
    group = torch.full((256, 12), -1, dtype=torch.int64)

    for pat in range(256):
        signs = [(pat >> i) & 1 for i in range(8)]    # 1=outside, 0=inside

        parent = list(range(8))

        def find(x: int) -> int:
            r = x
            while parent[r] != r:
                r = parent[r]
            while parent[x] != r:
                nxt = parent[x]
                parent[x] = r
                x = nxt
            return r

        # Union same-sign corners (not separated by the surface)
        for a, b in EDGE_PAIRS:
            if signs[a] == signs[b]:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

        # Distinct (interior_root, exterior_root) pairs are distinct patches
        group_map: dict[tuple[int, int], int] = {}
        for ei, (a, b) in enumerate(EDGE_PAIRS):
            if signs[a] == signs[b]:
                continue
            in_c = a if signs[a] == 0 else b
            ex_c = b if signs[a] == 0 else a
            key = (find(in_c), find(ex_c))
            if key not in group_map:
                group_map[key] = len(group_map)
            group[pat, ei] = group_map[key]
        K[pat] = len(group_map)

    return K, group


def _mdc_lut(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    K, g = _build_mdc_lut()
    return K.to(device), g.to(device)


def _dual_contour_manifold(voxel_coords: torch.Tensor, corner_udf: torch.Tensor,
                           corner_keys: torch.Tensor,
                           resolution: int, scale: float, center: torch.Tensor,
                           corner_valid: Optional[torch.Tensor] = None,
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Manifold DC: like _dual_contour but emits 1-4 dual verts per voxel via the patch LUT (centroid placement only)."""
    device = voxel_coords.device
    Nv = voxel_coords.shape[0]

    CORNER_OFFS = torch.tensor([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ], dtype=torch.long, device=device)
    corner_pos = voxel_coords.unsqueeze(1) + CORNER_OFFS.unsqueeze(0)        # (Nv, 8, 3)
    R1 = resolution + 1
    keys = (corner_pos[..., 0] * R1 + corner_pos[..., 1]) * R1 + corner_pos[..., 2]
    flat_keys = keys.reshape(-1)
    idx = torch.searchsorted(corner_keys, flat_keys)
    idx_c = idx.clamp(max=corner_keys.numel() - 1)
    found = (idx < corner_keys.numel()) & (corner_keys[idx_c] == flat_keys)
    sd = torch.where(found, corner_udf[idx_c],
                     torch.full_like(corner_udf[idx_c], 1.0)).reshape(Nv, 8)

    # Sign pattern: bit i = (sd[i] > 0), matching the LUT convention
    sign_bits = (sd > 0).to(torch.int64)                                     # (Nv, 8)
    weights = (1 << torch.arange(8, device=device, dtype=torch.int64))
    pat_per_voxel = (sign_bits * weights).sum(dim=-1)                        # (Nv,) in 0..255

    K_lut, group_lut = _mdc_lut(device)
    K_per_voxel = K_lut[pat_per_voxel]                                       # (Nv,)
    total_verts = int(K_per_voxel.sum().item())
    if total_verts == 0:
        return (torch.empty((0, 3), dtype=voxel_coords.dtype, device=device),
                torch.empty((0, 3), dtype=torch.long, device=device))

    vert_offset = (torch.cumsum(K_per_voxel, dim=0) - K_per_voxel)           # (Nv,)
    voxel_per_subvol = torch.repeat_interleave(
        torch.arange(Nv, device=device), K_per_voxel)                        # (total_verts,)

    EDGES = torch.tensor([
        [0, 1], [2, 3], [4, 5], [6, 7],
        [0, 2], [1, 3], [4, 6], [5, 7],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ], dtype=torch.long, device=device)
    sb_a = sign_bits[:, EDGES[:, 0]]                                         # (Nv, 12)
    sb_b = sign_bits[:, EDGES[:, 1]]
    crosses = sb_a != sb_b                                                   # (Nv, 12)

    if corner_valid is not None:
        cv = torch.where(found, corner_valid[idx_c],
                         torch.zeros_like(found)).reshape(Nv, 8)
        edge_valid = cv[:, EDGES[:, 0]] & cv[:, EDGES[:, 1]]
        crosses = crosses & edge_valid

    edge_group_per_voxel = group_lut[pat_per_voxel]                          # (Nv, 12), -1 if not crossing
    # LUT already gives -1 for non-crossing edges; only re-mask for corner_valid
    if corner_valid is not None:
        edge_group_per_voxel = torch.where(crosses, edge_group_per_voxel,
                                           torch.full_like(edge_group_per_voxel, -1))

    a_sd = sd[:, EDGES[:, 0]]                                                # (Nv, 12)
    b_sd = sd[:, EDGES[:, 1]]
    denom = a_sd - b_sd
    t = torch.where(denom.abs() > 1e-20, a_sd / denom, torch.zeros_like(a_sd))
    t = t.clamp(0.0, 1.0).unsqueeze(-1)
    corner_world = (corner_pos.float() / resolution - 0.5) * scale + center.unsqueeze(0).unsqueeze(0)
    a_pos = corner_world[:, EDGES[:, 0]]
    b_pos = corner_world[:, EDGES[:, 1]]
    crossing_pts = torch.lerp(a_pos, b_pos, t)                               # (Nv, 12, 3)

    # Aggregate crossing positions per (voxel, subvolume) into global dual verts
    flat_group = edge_group_per_voxel.reshape(-1)
    valid_mask = flat_group >= 0
    flat_voxel = torch.arange(Nv, device=device).unsqueeze(-1).expand(Nv, 12).reshape(-1)
    flat_pos = crossing_pts.reshape(-1, 3)
    v_idx = flat_voxel[valid_mask]
    g_idx = flat_group[valid_mask]
    pos = flat_pos[valid_mask]
    global_idx = vert_offset[v_idx] + g_idx                                  # (Nvalid,)

    pos_dtype = crossing_pts.dtype
    sums = torch.zeros((total_verts, 3), dtype=pos_dtype, device=device)
    counts = torch.zeros(total_verts, dtype=pos_dtype, device=device)
    sums.scatter_add_(0, global_idx.unsqueeze(-1).expand(-1, 3), pos)
    counts.scatter_add_(0, global_idx, torch.ones_like(g_idx, dtype=pos_dtype))
    # Fully-masked subvolumes default to the voxel centre (unreferenced)
    voxel_centre = ((voxel_coords.float() + 0.5) / resolution - 0.5) * scale + center.unsqueeze(0)
    dual_verts = torch.where(
        counts.unsqueeze(-1) > 0,
        sums / counts.clamp_min(1.0).unsqueeze(-1),
        voxel_centre[voxel_per_subvol].to(pos_dtype),
    )

    # Face emission. SHARED_LOCAL_EDGE[axis,k] = the k-th neighbour's local edge
    # slot corresponding to the shared grid edge (owner's slot = EDGE_OF_AXIS[axis]).
    NEIGHBOUR_OFFS = torch.tensor([
        [[0, 0, 0], [0, -1, 0], [0, -1, -1], [0, 0, -1]],
        [[0, 0, 0], [0, 0, -1], [-1, 0, -1], [-1, 0, 0]],
        [[0, 0, 0], [-1, 0, 0], [-1, -1, 0], [0, -1, 0]],
    ], dtype=torch.long, device=device)
    SHARED_LOCAL_EDGE = torch.tensor([
        [0, 1, 3, 2],     # x-axis
        [4, 6, 7, 5],     # y-axis
        [8, 9, 11, 10],   # z-axis
    ], dtype=torch.long, device=device)
    EDGE_OF_AXIS = torch.tensor([0, 4, 8], dtype=torch.long, device=device)

    vox_dims = voxel_coords.max(dim=0)[0] + 2
    vox_key = (voxel_coords[:, 0] * vox_dims[1] + voxel_coords[:, 1]) * vox_dims[2] + voxel_coords[:, 2]
    sort_v = vox_key.argsort()
    sorted_vox_key = vox_key[sort_v]

    tris_out = []
    for axis in range(3):
        edge_idx = EDGE_OF_AXIS[axis]
        owner_mask = crosses[:, edge_idx]
        if not owner_mask.any():
            continue
        owner_voxels = voxel_coords[owner_mask]
        sign_a_at_owner = sb_a[owner_mask, edge_idx]                         # (No,) — 0 inside, 1 outside

        nbrs = owner_voxels.unsqueeze(1) + NEIGHBOUR_OFFS[axis].unsqueeze(0)  # (No, 4, 3)
        nbr_keys = (nbrs[..., 0] * vox_dims[1] + nbrs[..., 1]) * vox_dims[2] + nbrs[..., 2]
        flat = nbr_keys.reshape(-1).contiguous()
        ins = torch.searchsorted(sorted_vox_key, flat)
        ins_c = ins.clamp(max=sorted_vox_key.numel() - 1)
        valid_nbr = (ins < sorted_vox_key.numel()) & (sorted_vox_key[ins_c] == flat)
        valid_quad = valid_nbr.reshape(-1, 4).all(dim=1)
        if not valid_quad.any():
            continue

        nbr_orig = sort_v[ins_c].reshape(-1, 4)[valid_quad]                  # (Mv, 4) voxel idx
        nbr_pat = pat_per_voxel[nbr_orig]                                    # (Mv, 4)
        local_e = SHARED_LOCAL_EDGE[axis].unsqueeze(0).expand_as(nbr_pat)
        nbr_subvol = group_lut[nbr_pat, local_e]                             # (Mv, 4)
        # Every neighbour must agree the shared edge is crossing
        ok = (nbr_subvol >= 0).all(dim=1)
        if not ok.any():
            continue
        nbr_subvol = nbr_subvol[ok]
        nbr_orig = nbr_orig[ok]
        dual_indices = vert_offset[nbr_orig] + nbr_subvol                    # (Mv', 4)
        sign_a = sign_a_at_owner[valid_quad][ok]                             # 0 = inside, 1 = outside

        # Winding: flip when corner a is outside (same as _dual_contour)
        flip = sign_a > 0
        d0, d1, d2, d3 = dual_indices.unbind(dim=1)
        t1a = torch.stack([d0, d1, d2], dim=1)
        t2a = torch.stack([d0, d2, d3], dim=1)
        t1b = torch.stack([d0, d2, d1], dim=1)
        t2b = torch.stack([d0, d3, d2], dim=1)
        tris_out.append(torch.where(flip.unsqueeze(-1), t1b, t1a))
        tris_out.append(torch.where(flip.unsqueeze(-1), t2b, t2a))

    if not tris_out:
        return dual_verts, torch.empty((0, 3), dtype=torch.long, device=device)
    return dual_verts, torch.cat(tris_out, dim=0)


# Main entry

def _filter_components(verts: torch.Tensor, faces: torch.Tensor,
                       min_fraction: float = 0.01,
                       drop_inverted: bool = True,
                       drop_enclosed: bool = True) -> torch.Tensor:
    """Drop tiny / inverted-volume / bbox-enclosed connected components; returns filtered faces."""
    device = faces.device
    V = verts.shape[0]

    # Connected components via min-label propagation across faces (200-iter max)
    label = torch.arange(V, dtype=torch.long, device=device)
    for _ in range(200):
        f_min = torch.minimum(torch.minimum(label[faces[:, 0]], label[faces[:, 1]]),
                              label[faces[:, 2]])
        new_label = label.clone()
        new_label.scatter_reduce_(0, faces[:, 0], f_min, reduce="amin", include_self=True)
        new_label.scatter_reduce_(0, faces[:, 1], f_min, reduce="amin", include_self=True)
        new_label.scatter_reduce_(0, faces[:, 2], f_min, reduce="amin", include_self=True)
        new_label = new_label[new_label]   # path compression
        if torch.equal(new_label, label):
            break
        label = new_label

    face_label = label[faces[:, 0]]                                    # (F,)
    unique_labels, inv = torch.unique(face_label, return_inverse=True)
    C = unique_labels.shape[0]
    counts = torch.bincount(inv, minlength=C)
    max_count = int(counts.max().item())
    keep = torch.ones(C, dtype=torch.bool, device=device)

    if min_fraction > 0:
        threshold = max(1, int(max_count * min_fraction))
        keep = keep & (counts >= threshold)

    if drop_inverted:
        # Drop components with negative signed volume, but always keep the largest
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        face_vol = (v0 * torch.cross(v1, v2, dim=-1)).sum(dim=-1)        # (F,)
        comp_vol = torch.zeros(C, dtype=face_vol.dtype, device=device)
        comp_vol.scatter_add_(0, inv, face_vol)
        if C > 1:
            large = counts.argmax()
            vol_ok = (comp_vol >= 0)
            vol_ok[large] = True
            keep = keep & vol_ok

    if drop_enclosed and C > 1:
        # Two-pass: (1) bbox-inside-largest test, then (2) +X raycast point-in-mesh
        large = counts.argmax()
        face_v = verts[faces]
        face_min = face_v.min(dim=1).values
        face_max = face_v.max(dim=1).values
        comp_min = torch.full((C, 3), float("inf"), dtype=verts.dtype, device=device)
        comp_max = torch.full((C, 3), float("-inf"), dtype=verts.dtype, device=device)
        comp_min.scatter_reduce_(0, inv[:, None].expand(-1, 3), face_min,
                                 reduce="amin", include_self=True)
        comp_max.scatter_reduce_(0, inv[:, None].expand(-1, 3), face_max,
                                 reduce="amax", include_self=True)
        big_min = comp_min[large]
        big_max = comp_max[large]
        enclosed = ((comp_min >= big_min).all(dim=-1)
                    & (comp_max <= big_max).all(dim=-1))
        enclosed[large] = False

        # Per-component centroid for the raycast test
        face_centroid = face_v.mean(dim=1)                               # (F, 3)
        comp_centroid = torch.zeros((C, 3), dtype=verts.dtype, device=device)
        comp_centroid.scatter_add_(0, inv[:, None].expand(-1, 3), face_centroid)
        comp_centroid = comp_centroid / counts.to(verts.dtype).unsqueeze(-1).clamp_min(1.0)

        # Raycast surviving non-largest candidates (small loop)
        big_faces = faces[inv == large]
        bv0 = verts[big_faces[:, 0]]
        bv1 = verts[big_faces[:, 1]]
        bv2 = verts[big_faces[:, 2]]
        candidates = torch.nonzero((keep & ~enclosed)
                                   & (torch.arange(C, device=device) != large),
                                   as_tuple=True)[0]
        for ci in candidates.tolist():
            origin = comp_centroid[ci]
            # 2D point-in-triangle in YZ for the ray origin's (y, z)
            oy, oz = origin[1], origin[2]
            s12 = (bv1[:, 1] - oy) * (bv2[:, 2] - oz) - (bv1[:, 2] - oz) * (bv2[:, 1] - oy)
            s20 = (bv2[:, 1] - oy) * (bv0[:, 2] - oz) - (bv2[:, 2] - oz) * (bv0[:, 1] - oy)
            s01 = (bv0[:, 1] - oy) * (bv1[:, 2] - oz) - (bv0[:, 2] - oz) * (bv1[:, 1] - oy)
            total = s12 + s20 + s01
            inside_yz = (((s12 >= 0) & (s20 >= 0) & (s01 >= 0))
                         | ((s12 <= 0) & (s20 <= 0) & (s01 <= 0)))
            inside_yz = inside_yz & (total.abs() > 1e-20)
            inv_t = 1.0 / total.where(total.abs() > 1e-20, torch.ones_like(total))
            hit_x = (s12 * bv0[:, 0] + s20 * bv1[:, 0] + s01 * bv2[:, 0]) * inv_t
            crossings = int((inside_yz & (hit_x > origin[0])).sum().item())
            if crossings % 2 == 1:
                enclosed[ci] = True
        keep = keep & ~enclosed

    if keep.all():
        return faces
    face_keep = keep[inv]
    return faces[face_keep]


def _taubin_smooth(verts: torch.Tensor, faces: torch.Tensor,
                   iters: int, lam: float = 0.5, mu: float = -0.53,
                   progress_callback=None) -> torch.Tensor:
    """Taubin lambda|mu low-pass smoothing (volume-preserving); boundary verts are no-ops."""
    if iters <= 0 or verts.numel() == 0 or faces.numel() == 0:
        return verts
    device = verts.device
    V = verts.shape[0]
    sorted_keys, _, _ = _sorted_edge_halfedges(faces, V)
    uniq_keys, _ = torch.unique_consecutive(sorted_keys, return_counts=True)
    P = V + 1
    a = uniq_keys // P
    b = uniq_keys % P
    ones = torch.ones_like(a, dtype=verts.dtype)
    counts = torch.zeros(V, dtype=verts.dtype, device=device)
    counts.scatter_add_(0, a, ones)
    counts.scatter_add_(0, b, ones)
    counts_safe = counts.clamp_min(1.0).unsqueeze(-1)
    has_nb = (counts > 0).unsqueeze(-1)
    a_exp = a.unsqueeze(-1).expand(-1, 3)
    b_exp = b.unsqueeze(-1).expand(-1, 3)

    out = verts
    for _ in range(iters):
        throw_exception_if_processing_interrupted()
        for w in (lam, mu):
            sums = torch.zeros_like(out)
            sums.scatter_add_(0, a_exp, out[b])
            sums.scatter_add_(0, b_exp, out[a])
            delta = (sums / counts_safe - out) * has_nb
            out = out + w * delta
        if progress_callback is not None:
            progress_callback()
    return out


def _fix_poles(verts: torch.Tensor, faces: torch.Tensor,
               colors: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Midpoint-collapse edge-sharing valence-3 vertex pairs (DC T-junction poles); boundary verts excluded."""
    device = verts.device
    V = verts.shape[0]
    if V == 0 or faces.numel() == 0:
        return verts, faces, colors

    sorted_keys, _, _ = _sorted_edge_halfedges(faces, V)
    uniq_keys, key_counts = torch.unique_consecutive(sorted_keys, return_counts=True)
    P = V + 1
    a = uniq_keys // P
    b = uniq_keys % P
    # Boundary verts (endpoints of single-face edges) are excluded from poles
    boundary_v = torch.zeros(V, dtype=torch.bool, device=device)
    bnd_mask = key_counts == 1
    if bnd_mask.any():
        boundary_v[a[bnd_mask]] = True
        boundary_v[b[bnd_mask]] = True
    ones = torch.ones_like(a)
    valence = torch.zeros(V, dtype=torch.long, device=device)
    valence.scatter_add_(0, a, ones)
    valence.scatter_add_(0, b, ones)
    is_pole = (valence == 3) & ~boundary_v
    if int(is_pole.sum().item()) < 2:
        return verts, faces, colors

    pp_edge = is_pole[a] & is_pole[b]
    if not pp_edge.any():
        return verts, faces, colors
    cand_a = a[pp_edge]
    cand_b = b[pp_edge]

    # Greedy maximal matching: accept candidates whose endpoints are still free
    used = torch.zeros(V, dtype=torch.bool, device="cpu")
    cand_a_cpu = cand_a.cpu().tolist()
    cand_b_cpu = cand_b.cpu().tolist()
    pairs: list[tuple[int, int]] = []
    for ai, bi in zip(cand_a_cpu, cand_b_cpu):
        if not used[ai] and not used[bi]:
            pairs.append((ai, bi))
            used[ai] = True
            used[bi] = True
    if not pairs:
        return verts, faces, colors

    pairs_t = torch.tensor(pairs, dtype=torch.long, device=device)         # (P, 2)
    keep_i = torch.minimum(pairs_t[:, 0], pairs_t[:, 1])
    drop_i = torch.maximum(pairs_t[:, 0], pairs_t[:, 1])

    new_verts = verts.clone()
    new_verts[keep_i] = 0.5 * (verts[pairs_t[:, 0]] + verts[pairs_t[:, 1]])
    new_colors = None
    if colors is not None:
        new_colors = colors.clone()
        new_colors[keep_i] = 0.5 * (colors[pairs_t[:, 0]] + colors[pairs_t[:, 1]])

    remap = torch.arange(V, dtype=torch.long, device=device)
    remap[drop_i] = keep_i
    new_faces = remap[faces.long()]
    degen = ((new_faces[:, 0] == new_faces[:, 1])
             | (new_faces[:, 1] == new_faces[:, 2])
             | (new_faces[:, 0] == new_faces[:, 2]))
    new_faces = new_faces[~degen]

    used_mask = torch.zeros(V, dtype=torch.bool, device=device)
    used_mask[new_faces.reshape(-1)] = True
    if not used_mask.all():
        compact = used_mask.long().cumsum(0) - 1
        new_verts = new_verts[used_mask]
        if new_colors is not None:
            new_colors = new_colors[used_mask]
        new_faces = compact[new_faces]
    return new_verts, new_faces.to(faces.dtype), new_colors


def remesh_narrow_band_dc(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    resolution: int = 256,
    target_faces: int = 0,                  # 0 = use `resolution`; >0 = auto-derive resolution
    band: float = 1.0,
    project_back: float = 0.0,
    qef: bool = True,
    sign_mode: str = "udf",                 # "sdf" | "udf"
    drop_small_components: float = 0.01,    # drop components below this fraction of max
    drop_inverted_components: bool = True,  # drop closed components with negative signed volume
    drop_enclosed_components: bool = True,  # drop components whose bbox is inside the largest's bbox
    fix_poles: bool = False,                # collapse 3-3 valence vertex pairs (DC T-junction artifact)
    smooth_iters: int = 0,                  # Taubin smoothing iterations (low-pass, volume-preserving)
    smooth_lambda: float = 0.5,
    smooth_mu: float = -0.53,
    manifold: bool = False,                 # Manifold DC: emit 1-4 dual verts per voxel for multi-sheet cases
    colors: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    center: Optional[torch.Tensor] = None,
):
    """Narrow-band Dual Contouring re-extraction; returns (new_vertices, new_faces, new_colors), new_colors None unless `colors` given.

    Key params: target_faces>0 auto-derives resolution; sign_mode sdf/udf
    (UDF disables qef and may need component filters); project_back lerps verts
    toward the closest surface point; scale/center default to bbox.
    """
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    device = vertices.device

    if center is None:
        center = 0.5 * (vertices.max(dim=0)[0] + vertices.min(dim=0)[0])
    else:
        center = center.to(device=device, dtype=vertices.dtype)
    if scale is None:
        bbox = vertices.max(dim=0)[0] - vertices.min(dim=0)[0]
        scale = float(bbox.max().item()) * 1.1

    # Auto-derive resolution from target_faces (~3 tris/crossing-voxel; +-30%)
    if target_faces > 0:
        tv = vertices[faces.long()]
        cross_v = torch.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0], dim=-1)
        surface_area = 0.5 * cross_v.norm(dim=-1).sum().item()
        relative_area = max(surface_area / (scale * scale), 1e-6)
        derived = int(math.sqrt(target_faces / (3.0 * relative_area)))
        # Round to a multiple of 32 (builder doubles from a <=32 base)
        derived = ((derived + 31) // 32) * 32
        derived = max(32, min(1024, derived))
        resolution = derived

    eps = band * scale / resolution

    # progress: one tick per narrow-band level + 3 stages (SDF/DC/post) + each smoothing iter
    n_levels, _b = 1, resolution
    while _b > 32 and _b % 2 == 0:
        _b //= 2
    while _b < resolution:
        _b *= 2
        n_levels += 1
    _total_ticks = n_levels + 3 + int(smooth_iters)
    _pbar = comfy.utils.ProgressBar(_total_ticks)
    _tq = _tqdm(total=_total_ticks, desc="Remesh DC", leave=False)

    def tick():
        _pbar.update(1)
        _tq.update(1)

    # Step 1: sparse narrow-band voxel grid (coarse-to-fine)
    voxel_coords, _band_tree = _build_narrow_band_voxels(
        vertices, faces, center, scale, resolution, eps,
        progress_callback=tick)
    if voxel_coords.numel() == 0:
        return (torch.empty((0, 3), dtype=vertices.dtype, device=device),
                torch.empty((0, 3), dtype=faces.dtype, device=device),
                None if colors is None else torch.empty((0, colors.shape[1]),
                                                        dtype=colors.dtype, device=device))

    # Step 2: collect unique corner positions of all active voxels
    CORNER_OFFS = torch.tensor([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ], dtype=torch.long, device=device)
    corners = (voxel_coords.unsqueeze(1) + CORNER_OFFS.unsqueeze(0)).reshape(-1, 3)
    R1 = resolution + 1
    corner_keys = (corners[:, 0] * R1 + corners[:, 1]) * R1 + corners[:, 2]
    unique_corner_keys, corner_inv = torch.unique(corner_keys, return_inverse=True)
    unique_corners = torch.zeros((unique_corner_keys.shape[0], 3), dtype=torch.long, device=device)
    unique_corners[corner_inv] = corners
    del corners, corner_keys, corner_inv

    if sign_mode == "sdf":
        use_sdf = True
    elif sign_mode == "udf":
        use_sdf = False
    else:
        raise ValueError(f"sign_mode must be 'sdf'|'udf', got {sign_mode!r}")

    # Step 3: distance field at every unique corner.
    tri_verts_g = vertices[faces.long()]
    # face normals: needed for the SDF sign AND for QEF placement (QEF is sign-agnostic,
    # so it works in UDF mode too — (n·(x-p))² is unchanged by normal orientation)
    if use_sdf or qef:
        tri_face_normals_all = torch.nn.functional.normalize(
            torch.cross(tri_verts_g[:, 1] - tri_verts_g[:, 0],
                        tri_verts_g[:, 2] - tri_verts_g[:, 0], dim=-1),
            p=2, dim=-1, eps=1e-12)
    cell_size = scale / resolution
    corner_world = (unique_corners.float() / resolution - 0.5) * scale + center.unsqueeze(0)
    # Exact corner UDF (no max_dist cap) so DC crossings keep fine detail
    udf, corner_closest, corner_tri = _udf_exact(corner_world, tri_verts_g, tree=_band_tree)
    corner_valid = corner_tri >= 0
    if use_sdf:
        sign = torch.ones_like(udf)
        n_for_corner = tri_face_normals_all[corner_tri.clamp(min=0)]
        offset = corner_world - corner_closest
        sign_dot = (offset * n_for_corner).sum(-1)
        sign = torch.where(corner_valid & (sign_dot < 0), -sign, sign)
        sdf = sign * udf
    else:
        # UDF mode: iso at UDF=eps; double surface on closed meshes, weld after
        sdf = udf - eps
    del unique_corners, corner_world, udf, corner_closest, corner_tri
    if use_sdf:
        del sign, n_for_corner, offset, sign_dot
    tick()  # SDF done

    # Step 4 + 5: dual contouring + topology. QEF works in both modes (sign-agnostic);
    # in UDF it pulls the ±eps crossing back onto the triangle planes → sharper edges.
    if qef:
        tri_face_normals = tri_face_normals_all
        # QEF needs the nearest triangle per crossing point. The centroid cKDTree
        # (_band_tree) is already built, and its exact k-NN query is markedly faster
        # here than a spatial-hash gather (which builds ~100-triangle candidate lists
        # per query on a dense input) — and it's exact. So reuse it directly.
        def _qef_query(pts):
            return _udf_exact(pts, tri_verts_g, tree=_band_tree)
    else:
        tri_face_normals = None
        _qef_query = None

    if manifold and use_sdf:
        # MDC ignores qef / tri_face_normals — centroid placement only.
        dual_verts, new_faces = _dual_contour_manifold(
            voxel_coords, sdf, unique_corner_keys,
            resolution, scale, center,
            corner_valid=corner_valid)
    else:
        dual_verts, new_faces = _dual_contour(
            voxel_coords, sdf, unique_corner_keys,
            resolution, scale, center,
            tri_face_normals=tri_face_normals, qef_query=_qef_query,
            # corner_valid filter only matters in SDF mode
            corner_valid=corner_valid if use_sdf else None)
    del voxel_coords, sdf, unique_corner_keys, corner_valid
    del tri_face_normals, _qef_query
    if use_sdf or qef:
        del tri_face_normals_all
    tick()  # DC done

    # Step 6: project_back and / or color sampling share one closest-point query
    need_query = (project_back > 0 or colors is not None) and dual_verts.numel() > 0
    out_colors = None
    if need_query:
        centroids = tri_verts_g.mean(dim=1)
        tri_radii = (tri_verts_g - centroids.unsqueeze(1)).norm(dim=-1).max(dim=-1).values
        short_hash_cell_t = torch.tensor(2.0 * cell_size, dtype=vertices.dtype, device=device)
        short_hash = _build_tri_spatial_hash(centroids, tri_radii, short_hash_cell_t)
        result = _udf_query(
            dual_verts, tri_verts_g, short_hash, short_hash_cell_t,
            max_dist=4.0 * cell_size,
            return_closest=True,
            return_tri_idx=(colors is not None))
        if colors is not None:
            _, closest_pts, closest_tri = result
        else:
            _, closest_pts = result

        if project_back > 0:
            dual_verts = torch.lerp(dual_verts, closest_pts, float(project_back))

        if colors is not None:
            # Barycentric-interpolate input colors at the closest point
            safe_tri = closest_tri.clamp(min=0)
            tri_v_idx = faces[safe_tri].long()                 # (N, 3)
            tri_v = vertices[tri_v_idx]                        # (N, 3, 3)
            v0 = tri_v[:, 0]
            v1 = tri_v[:, 1]
            v2 = tri_v[:, 2]
            e0 = v1 - v0
            e1 = v2 - v0
            e2 = closest_pts - v0
            d00 = (e0 * e0).sum(-1)
            d01 = (e0 * e1).sum(-1)
            d11 = (e1 * e1).sum(-1)
            d20 = (e2 * e0).sum(-1)
            d21 = (e2 * e1).sum(-1)
            denom = d00 * d11 - d01 * d01 + 1e-20
            bv = ((d11 * d20 - d01 * d21) / denom).clamp(0.0, 1.0)
            bw = ((d00 * d21 - d01 * d20) / denom).clamp(0.0, 1.0)
            bu = (1.0 - bv - bw).clamp(0.0, 1.0)
            tri_c = colors[tri_v_idx]                          # (N, 3, C)
            out_colors = (bu.unsqueeze(-1) * tri_c[:, 0]
                          + bv.unsqueeze(-1) * tri_c[:, 1]
                          + bw.unsqueeze(-1) * tri_c[:, 2])
            # Zero out failed-query rows (their barycentric used bogus triangle 0)
            invalid = closest_tri < 0
            if invalid.any():
                out_colors[invalid] = 0

    # Filter spurious components (tiny pieces, inverted inner shells)
    if (new_faces.numel() > 0
            and (drop_small_components > 0 or drop_inverted_components
                 or drop_enclosed_components)):
        new_faces = _filter_components(
            dual_verts, new_faces,
            min_fraction=drop_small_components if drop_small_components > 0 else 0.0,
            drop_inverted=drop_inverted_components,
            drop_enclosed=drop_enclosed_components)

    if fix_poles and new_faces.numel() > 0:
        dual_verts, new_faces, out_colors = _fix_poles(
            dual_verts, new_faces, out_colors)
    tick()  # post-process done

    if smooth_iters > 0 and dual_verts.numel() > 0 and new_faces.numel() > 0:
        dual_verts = _taubin_smooth(dual_verts, new_faces,
                                    iters=int(smooth_iters),
                                    lam=float(smooth_lambda),
                                    mu=float(smooth_mu),
                                    progress_callback=tick)

    # Drop unused verts (non-crossing voxels' dual verts) and compact faces
    if dual_verts.numel() > 0 and new_faces.numel() > 0:
        used = torch.zeros(dual_verts.shape[0], dtype=torch.bool, device=device)
        used[new_faces[:, 0]] = True
        used[new_faces[:, 1]] = True
        used[new_faces[:, 2]] = True
        remap = used.long().cumsum(0) - 1
        dual_verts = dual_verts[used]
        new_faces = remap[new_faces.long()]
        if out_colors is not None:
            out_colors = out_colors[used]

    return (dual_verts.to(vertices.dtype),
            new_faces.to(faces.dtype),
            out_colors.to(colors.dtype) if (out_colors is not None and colors is not None) else None)
