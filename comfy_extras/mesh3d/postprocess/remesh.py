"""Narrow-band dual contouring remesher, a PyTorch approximation of CuMesh's remesh_narrow_band_dc."""
from __future__ import annotations

import functools
import math
from typing import Optional, Tuple

import numpy as np
import torch
import scipy.spatial
import comfy.quant_ops
import comfy.utils
from tqdm import tqdm as _tqdm
from comfy.model_management import throw_exception_if_processing_interrupted

from .qem_decimate import _connected_labels, _kitchen_op, _sorted_edge_halfedges, _split_touching_sheets


def _point_tri_closest(points: torch.Tensor, tris: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Closest point and squared distance per (point, triangle) pair (Ericson's region test); points (N, 3), tris (N, 3, 3)."""
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

    vc = d1 * d4 - d3 * d2
    region_AB = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    v_ab = d1 / (d1 - d3 + 1e-20)
    closest_AB = a + v_ab.unsqueeze(-1) * ab

    vb = d5 * d2 - d1 * d6
    region_AC = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    v_ac = d2 / (d2 - d6 + 1e-20)
    closest_AC = a + v_ac.unsqueeze(-1) * ac

    va = d3 * d6 - d5 * d4
    region_BC = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    v_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6) + 1e-20)
    closest_BC = b + v_bc.unsqueeze(-1) * (c - b)

    denom = va + vb + vc + 1e-20
    v_face = vb / denom
    w_face = vc / denom
    closest_face = a + v_face.unsqueeze(-1) * ab + w_face.unsqueeze(-1) * ac

    # in-place where into the fresh closest_face; later regions take precedence
    closest = closest_face
    torch.where(region_BC.unsqueeze(-1), closest_BC, closest, out=closest)
    torch.where(region_AC.unsqueeze(-1), closest_AC, closest, out=closest)
    torch.where(region_AB.unsqueeze(-1), closest_AB, closest, out=closest)
    torch.where(region_C .unsqueeze(-1), c,          closest, out=closest)
    torch.where(region_B .unsqueeze(-1), b,          closest, out=closest)
    torch.where(region_A .unsqueeze(-1), a,          closest, out=closest)

    diff = points - closest
    return closest, (diff * diff).sum(-1)


def _split_long_triangles(tri_verts: torch.Tensor, max_edge: float):
    """Bisect triangles on their longest edge until no edge exceeds `max_edge`; returns (pieces (M, 3, 3), source index (M,)).

    Keeps the centroid kNN in _udf_exact from missing the closest triangle on low-poly inputs; pieces keep their source's winding."""
    src = torch.arange(tri_verts.shape[0], device=tri_verts.device)
    while True:
        e = (tri_verts.roll(-1, 1) - tri_verts).norm(dim=-1)  # edge k runs from vertex k to k + 1
        split = e.max(1).values > max_edge
        if not bool(split.any()):
            return tri_verts, src
        k = e[split].argmax(1)
        rot = (torch.arange(3, device=k.device)[None] + k[:, None]) % 3  # longest edge first
        t = torch.gather(tri_verts[split], 1, rot[..., None].expand(-1, -1, 3))
        m = 0.5 * (t[:, 0] + t[:, 1])
        tri_verts = torch.cat([tri_verts[~split], torch.stack([t[:, 0], m, t[:, 2]], 1), torch.stack([m, t[:, 1], t[:, 2]], 1)])
        src = torch.cat([src[~split], src[split], src[split]])


def _build_udf_tree(tri_verts: torch.Tensor):
    """Closest-triangle index for _udf_exact: comfy-kitchen's mesh BVH when available, else a cKDTree over triangle centroids."""
    mesh_bvh = _kitchen_op("mesh_bvh", tri_verts)
    if mesh_bvh is not None and tri_verts.dtype == torch.float32:
        return mesh_bvh(tri_verts)
    return scipy.spatial.cKDTree(tri_verts.mean(dim=1).detach().cpu().numpy(),
                                 balanced_tree=False, compact_nodes=False)


def _udf_exact(query_points: torch.Tensor, tri_verts: torch.Tensor,
               k: int = 8, chunk: int = 262144, tree=None):
    """Unsigned distance to the triangles; returns (dist (N,), closest point (N, 3), triangle index (N,)).

    The cKDTree path only tests the `k` nearest centroids, so long triangles must be split first (_split_long_triangles)."""
    if tree is None:
        tree = _build_udf_tree(tri_verts)
    if not isinstance(tree, scipy.spatial.cKDTree):
        return comfy.quant_ops.ck.closest_point_on_mesh(tree, query_points)
    device = query_points.device
    F = tri_verts.shape[0]
    kq = int(min(k, F))
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


def _build_narrow_band_voxels(tri_verts: torch.Tensor,
                              center: torch.Tensor, scale: float,
                              resolution: int, eps: float,
                              progress_callback=None) -> torch.Tensor:
    """Returns voxel coords (Nv, 3) in [0, resolution) whose centre is within half a cell diagonal + eps of the surface, and the distance tree."""
    device = tri_verts.device
    tree = _build_udf_tree(tri_verts)

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


# _CUBE_EDGES: the 12 voxel edges as corner pairs, x-, y-, then z-aligned. _EDGE_VOXELS: per axis, the 4 voxels
# around a grid edge; the y order is reversed vs x/z so quads wind consistently around every edge.
_VOXEL_CHUNK = 1 << 21  # voxels per pass of the per-corner lookups, the remesh's largest transient
_POINT_CHUNK = 1 << 21  # points per field evaluation, tet walk and sorted-merge pass; each walk step gathers 4 float64 planes per point
_CUBE_CORNERS = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1))
_CUBE_EDGES = ((0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7))
_EDGE_VOXELS = (((0, 0, 0), (0, -1, 0), (0, -1, -1), (0, 0, -1)),
                ((0, 0, 0), (0, 0, -1), (-1, 0, -1), (-1, 0, 0)),
                ((0, 0, 0), (-1, 0, 0), (-1, -1, 0), (0, -1, 0)))


def _voxel_key(coords, resolution):
    """Row-major key of voxel or corner coords on the (resolution + 1)³ corner grid; a band is its sorted voxel keys."""
    r1 = resolution + 1
    return (coords[..., 0] * r1 + coords[..., 1]) * r1 + coords[..., 2]


def _decode_key(keys, resolution):
    r1 = resolution + 1
    return torch.stack([keys // (r1 * r1), keys // r1 % r1, keys % r1], dim=1)


def _dual_contour(vox_keys: torch.Tensor, corner_udf: torch.Tensor,
                  corner_keys: torch.Tensor,
                  resolution: int, scale: float, center: torch.Tensor,
                  tri_face_normals: Optional[torch.Tensor] = None,
                  qef_query=None,
                  corner_valid: Optional[torch.Tensor] = None,
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dual contour the active voxels, given as sorted keys; returns dual verts (Nv, 3) and faces (M, 3), surface where
    corner values change sign.

    Verts go to the QEF minimiser when tri_face_normals and qef_query are given, else to the centroid of the edge crossings."""
    device = vox_keys.device
    CORNER_OFFS = torch.tensor(_CUBE_CORNERS, dtype=torch.long, device=device)
    EDGES = torch.tensor(_CUBE_EDGES, dtype=torch.long, device=device)
    corner_key_offs = _voxel_key(CORNER_OFFS, resolution)

    def edge_values(vk):
        """Per voxel: crossing position along each of the 12 edges, whether corner a of the edges the voxel owns
        (slots 0/4/8) is outside, and which edges cross."""
        keys = vk[:, None] + corner_key_offs
        idx = torch.searchsorted(corner_keys, keys.reshape(-1))
        idx_c = idx.clamp(max=corner_keys.numel() - 1)
        found = (idx < corner_keys.numel()) & (corner_keys[idx_c] == keys.reshape(-1))
        # corners missing from corner_keys count as outside (+1)
        sd = torch.where(found, corner_udf[idx_c], 1.0).reshape(-1, 8)
        a_sd, b_sd = sd[:, EDGES[:, 0]], sd[:, EDGES[:, 1]]              # (n, 12)
        crosses = (a_sd * b_sd) < 0
        # crossings at an invalid corner would emit fake faces at the band edge
        if corner_valid is not None:
            cv = torch.where(found, corner_valid[idx_c], False).reshape(-1, 8)
            crosses &= cv[:, EDGES[:, 0]] & cv[:, EDGES[:, 1]]
        return (a_sd / (a_sd - b_sd + 1e-20)).clamp(0.0, 1.0), a_sd[:, 0::4] > 0, crosses

    # only voxels with a crossing emit anything, and all 4 voxels around a crossing edge share it; the per-corner
    # lookups run in chunks, since over the whole band at once they would be the remesh's largest allocation
    keep = torch.empty(vox_keys.numel(), dtype=torch.bool, device=device)
    for s in range(0, vox_keys.numel(), _VOXEL_CHUNK):
        keep[s:s + _VOXEL_CHUNK] = edge_values(vox_keys[s:s + _VOXEL_CHUNK])[2].any(dim=1)
    vox_keys = vox_keys[keep]
    Nv = vox_keys.numel()
    if Nv == 0:
        return torch.empty((0, 3), device=device), torch.empty((0, 3), dtype=torch.long, device=device)
    EDGE_OF_AXIS = torch.tensor([0, 4, 8], dtype=torch.long, device=device)  # a voxel owns the +axis edges at its min corner

    def corner_world(vc, c):
        return ((vc + CORNER_OFFS[c]).float() / resolution - 0.5) * scale + center  # (n, 3)

    # per voxel, a chunk at a time: the crossing points of all 12 edges, and the QEF normals, would otherwise be the
    # largest arrays of the pass. What survives is the dual vertex and the crossing edges the voxel owns, for the quads.
    dual_verts = torch.empty((Nv, 3), dtype=torch.float32, device=device)
    own_voxel, own_axis, own_flip = [], [], []
    for s in range(0, Nv, _VOXEL_CHUNK):
        vk = vox_keys[s:s + _VOXEL_CHUNK]
        vc = _decode_key(vk, resolution)
        t, a_outside, crosses = edge_values(vk)
        t = t.unsqueeze(-1)
        crossing_pts = torch.empty((vc.shape[0], 12, 3), dtype=torch.float32, device=device)
        for e in range(12):
            crossing_pts[:, e] = torch.lerp(corner_world(vc, EDGES[e, 0]), corner_world(vc, EDGES[e, 1]), t[:, e])
        del t

        # centroid of the crossings, also the QEF fallback; the entries of edges without a crossing are zeroed, nothing reads them
        crosses_f = crosses.float().unsqueeze(-1)
        crossing_pts.mul_(crosses_f)
        verts = centroid = crossing_pts.sum(dim=1) / crosses_f.sum(dim=1)

        # QEF: minimise sum_i (n_i·(x - p_i))², regularised towards the centroid; solutions outside the voxel fall back to it
        if tri_face_normals is not None and qef_query is not None:
            flat_mask = crosses.reshape(-1)
            if flat_mask.any():
                query_pts = crossing_pts.reshape(-1, 3)[flat_mask]
                _, _, qef_tri_idx = qef_query(query_pts)
                # missed queries (tri -1) get a zero normal, i.e. no constraint
                valid_q = qef_tri_idx >= 0
                normals_at_q = torch.zeros_like(query_pts)
                normals_at_q[valid_q] = tri_face_normals[qef_tri_idx[valid_q]]
                n_per_edge = torch.zeros((vc.shape[0] * 12, 3), dtype=query_pts.dtype, device=device)
                n_per_edge[flat_mask] = normals_at_q
                n_per_edge = n_per_edge.reshape(-1, 12, 3)

                A = torch.einsum('vec,ved->vcd', n_per_edge, n_per_edge)        # (n, 3, 3)
                n_dot_p = (n_per_edge * crossing_pts).sum(dim=-1)
                b = torch.einsum('ve,vec->vc', n_dot_p, n_per_edge)             # (n, 3)

                # Tikhonov regularisation, in place on the fresh einsum outputs
                reg = 1e-2
                A.diagonal(dim1=-2, dim2=-1).add_(reg)
                b.add_(centroid, alpha=reg)
                qef_solution = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)

                in_box = (qef_solution >= corner_world(vc, 0)).all(dim=-1) & (qef_solution <= corner_world(vc, 7)).all(dim=-1)
                verts = torch.where(in_box.unsqueeze(-1), qef_solution, centroid)
                del query_pts, qef_tri_idx, valid_q, normals_at_q, n_per_edge, A, n_dot_p, b, qef_solution, in_box
        dual_verts[s:s + vc.shape[0]] = verts
        v, axis = crosses[:, EDGE_OF_AXIS].nonzero(as_tuple=True)
        own_voxel.append(v + s)
        own_axis.append(axis.to(torch.int8))
        own_flip.append(a_outside[v, axis])
        del vk, vc, a_outside, crosses, crossing_pts, crosses_f, verts, centroid
    own_voxel, own_axis, own_flip = torch.cat(own_voxel), torch.cat(own_axis), torch.cat(own_flip)

    # each crossing grid edge gives a quad over the 4 voxels around it, split into 2 triangles; the keys are sorted,
    # so a neighbour's position is its dual vertex
    neighbour_key_offs = _voxel_key(torch.tensor(_EDGE_VOXELS, dtype=torch.long, device=device), resolution)   # (3, 4)

    tris = []
    for axis in range(3):
        on_axis = own_axis == axis
        owners, flips = own_voxel[on_axis], own_flip[on_axis]
        t1s, t2s = [], []
        for s in range(0, owners.shape[0], _VOXEL_CHUNK):
            flat = (vox_keys[owners[s:s + _VOXEL_CHUNK], None] + neighbour_key_offs[axis]).reshape(-1)
            ins = torch.searchsorted(vox_keys, flat)
            ins_c = ins.clamp(max=Nv - 1)
            valid = ((ins < Nv) & (vox_keys[ins_c] == flat)).reshape(-1, 4).all(dim=1)
            if not valid.any():
                continue
            dual_indices = ins_c.reshape(-1, 4)[valid]    # (Mv, 4)
            # split along the shorter diagonal; a fixed one folds curved surfaces into grid-aligned terraces
            p = dual_verts[dual_indices]
            rot = (p[:, 1] - p[:, 3]).norm(dim=-1) < (p[:, 0] - p[:, 2]).norm(dim=-1)
            dual_indices = torch.where(rot.unsqueeze(-1), dual_indices.roll(-1, dims=1), dual_indices)
            del p, rot
            # flip when corner a is outside so the normal points out
            d0 = dual_indices[:, 0]
            d1 = dual_indices[:, 1]
            d2 = dual_indices[:, 2]
            d3 = dual_indices[:, 3]
            flip = flips[s:s + _VOXEL_CHUNK][valid]
            t1a = torch.stack([d0, d1, d2], dim=1)
            t2a = torch.stack([d0, d2, d3], dim=1)
            t1b = torch.stack([d0, d2, d1], dim=1)
            t2b = torch.stack([d0, d3, d2], dim=1)
            t1s.append(torch.where(flip.unsqueeze(-1), t1b, t1a))
            t2s.append(torch.where(flip.unsqueeze(-1), t2b, t2a))
        if t1s:
            tris.append(torch.cat(t1s))
            tris.append(torch.cat(t2s))

    if not tris:
        return dual_verts, torch.empty((0, 3), dtype=torch.long, device=device)
    new_faces = torch.cat(tris, dim=0)
    return dual_verts, new_faces


# Manifold Dual Contouring (Schaefer, Ju, Warren 2007)

@functools.lru_cache(maxsize=None)
def _build_mdc_lut() -> Tuple[torch.Tensor, torch.Tensor]:
    """Per 8-corner sign pattern: patch count K (256,) and per-edge patch id `group` (256, 12), -1 if not crossing."""
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

        # corners joined by a same-sign edge are on the same side of one patch
        for a, b in _CUBE_EDGES:
            if signs[a] == signs[b]:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

        # each distinct (interior root, exterior root) pair is one patch
        group_map: dict[tuple[int, int], int] = {}
        for ei, (a, b) in enumerate(_CUBE_EDGES):
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
    """Manifold DC: like _dual_contour but with one dual vert per patch (1-4 per voxel), centroid placement only."""
    device = voxel_coords.device
    Nv = voxel_coords.shape[0]

    CORNER_OFFS = torch.tensor(_CUBE_CORNERS, dtype=torch.long, device=device)
    corner_pos = voxel_coords.unsqueeze(1) + CORNER_OFFS.unsqueeze(0)        # (Nv, 8, 3)
    R1 = resolution + 1
    keys = (corner_pos[..., 0] * R1 + corner_pos[..., 1]) * R1 + corner_pos[..., 2]
    flat_keys = keys.reshape(-1)
    idx = torch.searchsorted(corner_keys, flat_keys)
    idx_c = idx.clamp(max=corner_keys.numel() - 1)
    found = (idx < corner_keys.numel()) & (corner_keys[idx_c] == flat_keys)
    sd = torch.where(found, corner_udf[idx_c],
                     torch.full_like(corner_udf[idx_c], 1.0)).reshape(Nv, 8)

    # bit i set when corner i is outside, as in the LUT
    sign_bits = (sd > 0).to(torch.int64)
    weights = (1 << torch.arange(8, device=device, dtype=torch.int64))
    pat_per_voxel = (sign_bits * weights).sum(dim=-1)                        # (Nv,) in 0..255

    K_lut, group_lut = _mdc_lut(device)
    K_per_voxel = K_lut[pat_per_voxel]
    total_verts = int(K_per_voxel.sum().item())
    if total_verts == 0:
        return (torch.empty((0, 3), dtype=voxel_coords.dtype, device=device),
                torch.empty((0, 3), dtype=torch.long, device=device))

    vert_offset = (torch.cumsum(K_per_voxel, dim=0) - K_per_voxel)
    voxel_per_subvol = torch.repeat_interleave(
        torch.arange(Nv, device=device), K_per_voxel)

    EDGES = torch.tensor(_CUBE_EDGES, dtype=torch.long, device=device)
    sb_a = sign_bits[:, EDGES[:, 0]]
    sb_b = sign_bits[:, EDGES[:, 1]]
    crosses = sb_a != sb_b

    if corner_valid is not None:
        cv = torch.where(found, corner_valid[idx_c],
                         torch.zeros_like(found)).reshape(Nv, 8)
        edge_valid = cv[:, EDGES[:, 0]] & cv[:, EDGES[:, 1]]
        crosses = crosses & edge_valid

    edge_group_per_voxel = group_lut[pat_per_voxel]                          # (Nv, 12)
    # the LUT already marks non-crossing edges -1; mask the ones at invalid corners too
    if corner_valid is not None:
        edge_group_per_voxel = torch.where(crosses, edge_group_per_voxel,
                                           torch.full_like(edge_group_per_voxel, -1))

    a_sd = sd[:, EDGES[:, 0]]
    b_sd = sd[:, EDGES[:, 1]]
    denom = a_sd - b_sd
    t = torch.where(denom.abs() > 1e-20, a_sd / denom, torch.zeros_like(a_sd))
    t = t.clamp(0.0, 1.0).unsqueeze(-1)
    corner_world = (corner_pos.float() / resolution - 0.5) * scale + center.unsqueeze(0).unsqueeze(0)
    a_pos = corner_world[:, EDGES[:, 0]]
    b_pos = corner_world[:, EDGES[:, 1]]
    crossing_pts = torch.lerp(a_pos, b_pos, t)                               # (Nv, 12, 3)

    # each dual vert is the centroid of its patch's crossings
    flat_group = edge_group_per_voxel.reshape(-1)
    valid_mask = flat_group >= 0
    flat_voxel = torch.arange(Nv, device=device).unsqueeze(-1).expand(Nv, 12).reshape(-1)
    flat_pos = crossing_pts.reshape(-1, 3)
    v_idx = flat_voxel[valid_mask]
    g_idx = flat_group[valid_mask]
    pos = flat_pos[valid_mask]
    global_idx = vert_offset[v_idx] + g_idx

    pos_dtype = crossing_pts.dtype
    sums = torch.zeros((total_verts, 3), dtype=pos_dtype, device=device)
    counts = torch.zeros(total_verts, dtype=pos_dtype, device=device)
    sums.scatter_add_(0, global_idx.unsqueeze(-1).expand(-1, 3), pos)
    counts.scatter_add_(0, global_idx, torch.ones_like(g_idx, dtype=pos_dtype))
    # fully masked patches get the voxel centre; no face uses them
    voxel_centre = ((voxel_coords.float() + 0.5) / resolution - 0.5) * scale + center.unsqueeze(0)
    dual_verts = torch.where(
        counts.unsqueeze(-1) > 0,
        sums / counts.clamp_min(1.0).unsqueeze(-1),
        voxel_centre[voxel_per_subvol].to(pos_dtype),
    )

    # SHARED_LOCAL_EDGE[axis, k]: the slot of the shared grid edge in the k-th neighbour around it
    NEIGHBOUR_OFFS = torch.tensor(_EDGE_VOXELS, dtype=torch.long, device=device)
    SHARED_LOCAL_EDGE = torch.tensor([
        [0, 1, 3, 2],
        [4, 6, 7, 5],
        [8, 9, 11, 10],
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
        sign_a_at_owner = sb_a[owner_mask, edge_idx]                         # 0 inside, 1 outside

        nbrs = owner_voxels.unsqueeze(1) + NEIGHBOUR_OFFS[axis].unsqueeze(0)  # (No, 4, 3)
        nbr_keys = (nbrs[..., 0] * vox_dims[1] + nbrs[..., 1]) * vox_dims[2] + nbrs[..., 2]
        flat = nbr_keys.reshape(-1).contiguous()
        ins = torch.searchsorted(sorted_vox_key, flat)
        ins_c = ins.clamp(max=sorted_vox_key.numel() - 1)
        valid_nbr = (ins < sorted_vox_key.numel()) & (sorted_vox_key[ins_c] == flat)
        valid_quad = valid_nbr.reshape(-1, 4).all(dim=1)
        if not valid_quad.any():
            continue

        nbr_orig = sort_v[ins_c].reshape(-1, 4)[valid_quad]                  # (Mv, 4)
        nbr_pat = pat_per_voxel[nbr_orig]
        local_e = SHARED_LOCAL_EDGE[axis].unsqueeze(0).expand_as(nbr_pat)
        nbr_subvol = group_lut[nbr_pat, local_e]
        # every neighbour must agree the shared edge is crossing
        ok = (nbr_subvol >= 0).all(dim=1)
        if not ok.any():
            continue
        nbr_subvol = nbr_subvol[ok]
        nbr_orig = nbr_orig[ok]
        dual_indices = vert_offset[nbr_orig] + nbr_subvol
        sign_a = sign_a_at_owner[valid_quad][ok]
        p = dual_verts[dual_indices]
        rot = (p[:, 1] - p[:, 3]).norm(dim=-1) < (p[:, 0] - p[:, 2]).norm(dim=-1)
        dual_indices = torch.where(rot.unsqueeze(-1), dual_indices.roll(-1, dims=1), dual_indices)
        del p, rot

        # same split and winding as _dual_contour
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


def _filter_components(verts: torch.Tensor, faces: torch.Tensor,
                       min_fraction: float = 0.01,
                       drop_inverted: bool = True,
                       drop_enclosed: bool = True) -> torch.Tensor:
    """Drop tiny, inside-out and enclosed connected components; returns the kept faces."""
    device = faces.device
    V = verts.shape[0]

    label = _connected_labels(torch.cat([faces[:, :2], faces[:, 1:]]), V)
    face_label = label[faces[:, 0]]
    unique_labels, inv = torch.unique(face_label, return_inverse=True)
    C = unique_labels.shape[0]
    counts = torch.bincount(inv, minlength=C)
    max_count = int(counts.max().item())
    keep = torch.ones(C, dtype=torch.bool, device=device)

    if min_fraction > 0:
        threshold = max(1, int(max_count * min_fraction))
        keep = keep & (counts >= threshold)

    if drop_inverted:
        # negative signed volume means inside out; the largest component is always kept
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        face_vol = (v0 * torch.cross(v1, v2, dim=-1)).sum(dim=-1)
        comp_vol = torch.zeros(C, dtype=face_vol.dtype, device=device)
        comp_vol.scatter_add_(0, inv, face_vol)
        if C > 1:
            large = counts.argmax()
            vol_ok = (comp_vol >= 0)
            vol_ok[large] = True
            keep = keep & vol_ok

    if drop_enclosed and C > 1:
        # enclosed: bbox inside the largest's bbox, or centroid inside the largest by +X ray parity
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

        face_centroid = face_v.mean(dim=1)
        comp_centroid = torch.zeros((C, 3), dtype=verts.dtype, device=device)
        comp_centroid.scatter_add_(0, inv[:, None].expand(-1, 3), face_centroid)
        comp_centroid = comp_centroid / counts.to(verts.dtype).unsqueeze(-1).clamp_min(1.0)

        big_faces = faces[inv == large]
        bv0 = verts[big_faces[:, 0]]
        bv1 = verts[big_faces[:, 1]]
        bv2 = verts[big_faces[:, 2]]
        candidates = torch.nonzero((keep & ~enclosed)
                                   & (torch.arange(C, device=device) != large),
                                   as_tuple=True)[0]
        for ci in candidates.tolist():
            origin = comp_centroid[ci]
            # the +X ray hits the triangles whose YZ projection contains the origin
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
    """Taubin lambda/mu smoothing with uniform weights, a low-pass filter that limits shrinkage."""
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
    """Collapse adjacent interior valence-3 vertex pairs (DC pole artifacts) to their midpoint; returns (verts, faces, colors)."""
    device = verts.device
    V = verts.shape[0]
    if V == 0 or faces.numel() == 0:
        return verts, faces, colors

    sorted_keys, _, _ = _sorted_edge_halfedges(faces, V)
    uniq_keys, key_counts = torch.unique_consecutive(sorted_keys, return_counts=True)
    P = V + 1
    a = uniq_keys // P
    b = uniq_keys % P
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

    # greedy matching, so each vertex collapses at most once
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

    pairs_t = torch.tensor(pairs, dtype=torch.long, device=device)
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


def _merge_sorted(keys: torch.Tensor, new_keys: torch.Tensor, *columns):
    """Merge sorted, disjoint `new_keys` into sorted `keys` without re-sorting; each (old, new) column in `columns`
    follows its keys. A key's slot is its own index plus the count of the other array's keys below it."""
    device = keys.device
    columns = ((keys, new_keys),) + columns
    merged = [torch.empty((keys.numel() + new_keys.numel(),) + old.shape[1:], dtype=old.dtype, device=device) for old, _ in columns]
    at_new = torch.searchsorted(keys, new_keys) + torch.arange(new_keys.numel(), device=device)
    for out, (_, new) in zip(merged, columns):
        out[at_new] = new
    for s in range(0, keys.numel(), _POINT_CHUNK):
        at_old = torch.searchsorted(new_keys, keys[s:s + _POINT_CHUNK]) + torch.arange(s, min(s + _POINT_CHUNK, keys.numel()), device=device)
        for out, (old, _) in zip(merged, columns):
            out[at_old] = old[s:s + _POINT_CHUNK]
    return merged


def _close_band(vox_keys: torch.Tensor, corner_keys: torch.Tensor, corner_sdf: torch.Tensor,
                field, resolution: int, scale: float, center: torch.Tensor):
    """Add the missing voxels around every sign-changing grid edge, so each one emits a quad and the surface closes.

    `vox_keys` are the band's sorted voxel keys and `field(points)` gives the SDF at new corners. Returns the grown,
    still sorted (vox_keys, corner_keys, corner_sdf)."""
    device = vox_keys.device
    corner_offs = torch.tensor(_CUBE_CORNERS, dtype=torch.long, device=device)
    edges = torch.tensor(_CUBE_EDGES, dtype=torch.long, device=device)
    axis_of_edge = torch.tensor([0] * 4 + [1] * 4 + [2] * 4, device=device)
    around = torch.tensor(_EDGE_VOXELS, dtype=torch.long, device=device)
    corner_key_offs = _voxel_key(corner_offs, resolution)
    while True:
        new = []
        for s in range(0, vox_keys.numel(), _VOXEL_CHUNK):
            vk = vox_keys[s:s + _VOXEL_CHUNK]
            val = corner_sdf[torch.searchsorted(corner_keys, vk[:, None] + corner_key_offs)]   # (n, 8)
            v, e = ((val[:, edges[:, 0]] < 0) != (val[:, edges[:, 1]] < 0)).nonzero(as_tuple=True)
            # the 4 voxels around the crossing edge, from its min corner; nearly all are in the band already
            vox = ((_decode_key(vk, resolution)[v] + corner_offs[edges[e, 0]])[:, None, :] + around[axis_of_edge[e]]).reshape(-1, 3)
            need = torch.unique(_voxel_key(vox[((vox >= 0) & (vox < resolution)).all(1)], resolution))
            pos = torch.searchsorted(vox_keys, need).clamp(max=vox_keys.numel() - 1)
            new.append(need[vox_keys[pos] != need])
        new = torch.unique(torch.cat(new))
        if new.numel() == 0:
            return vox_keys, corner_keys, corner_sdf
        vox_keys = _merge_sorted(vox_keys, new)[0]
        new_corners = torch.unique(new[:, None] + corner_key_offs)
        pos = torch.searchsorted(corner_keys, new_corners).clamp(max=corner_keys.numel() - 1)
        new_corners = new_corners[corner_keys[pos] != new_corners]
        if new_corners.numel():
            new_sdf = torch.empty(new_corners.numel(), dtype=corner_sdf.dtype, device=device)
            for s in range(0, new_corners.numel(), _POINT_CHUNK):
                new_sdf[s:s + _POINT_CHUNK] = field((_decode_key(new_corners[s:s + _POINT_CHUNK], resolution).to(corner_sdf.dtype) / resolution - 0.5) * scale + center)
            corner_keys, corner_sdf = _merge_sorted(corner_keys, new_corners, (corner_sdf, new_sdf))


# Solid sign mode (CelloCut, arXiv 2605.17853): inside/outside from a graph cut over a tetrahedralization of the UDF shell.

_TET_FACES = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))  # face k is opposite vertex k


def _min_cut(nbr: torch.Tensor, cap: torch.Tensor, s_cap: torch.Tensor, t_cap: torch.Tensor,
             relabel_every: int = 64) -> torch.Tensor:
    """Data-parallel push-relabel min s-t cut; returns the source side as a bool mask (N,).

    nbr (N, 4) holds neighbour ids (-1 = none) with symmetric capacities `cap` (N, 4); s_cap / t_cap are per-node terminal capacities."""
    min_cut = _kitchen_op("min_cut", nbr)
    if min_cut is not None:
        return min_cut(nbr, cap.float(), s_cap.float(), t_cap.float())
    n = nbr.shape[0]
    device = nbr.device
    big = n + 2
    valid = nbr >= 0
    j = nbr.clamp(min=0)
    rev = (nbr[j] == torch.arange(n, device=device)[:, None, None]).int().argmax(-1)  # nbr[j, rev] == i
    tol = 1e-12 * float(cap.max()) if cap.numel() else 0.0
    r = torch.where(valid, cap, torch.zeros_like(cap))
    rt = t_cap.clone()
    e = s_cap.clone()

    def global_relabel():
        h = torch.full((n,), big, dtype=torch.long, device=device)
        frontier = rt > tol
        h[frontier] = 1
        d = 1
        while bool(frontier.any()):
            d += 1
            frontier = ((r > tol) & frontier[j]).any(1) & (h == big)
            h[frontier] = d
        return h

    h = global_relabel()
    it = 0
    while True:
        throw_exception_if_processing_interrupted()
        if not bool(((e > tol) & (h < big)).any()):
            break
        amt = torch.where((e > tol) & (h == 1) & (rt > tol), torch.minimum(e, rt), torch.zeros_like(e))
        e = e - amt
        rt = rt - amt
        for k in range(4):  # one slot at a time, so a node never pushes more than its excess
            jk = j[:, k]
            adm = (e > tol) & (h < big) & (r[:, k] > tol) & (h == h[jk] + 1)
            idx = adm.nonzero().squeeze(1)
            amt = torch.minimum(e[idx], r[idx, k])
            e[idx] -= amt
            r[idx, k] -= amt
            r[jk[idx], rev[idx, k]] += amt
            e.index_add_(0, jk[idx], amt)
        stuck = (e > tol) & (h < big)
        if bool(stuck.any()):
            hmin = torch.where(r > tol, h[j], big).amin(1)
            hmin = torch.where(rt > tol, torch.zeros_like(hmin), hmin)
            h = torch.where(stuck, torch.maximum(h, (hmin + 1).clamp(max=big)), h)
        it += 1
        if it % relabel_every == 0:
            h = global_relabel()
    return global_relabel() >= big


def _tet_locator(tet_pts: torch.Tensor, tets: torch.Tensor, nbr: torch.Tensor, max_steps: int = 1024):
    """Returns locate(points) -> containing tet per point (-1 outside the hull), by stochastic visibility walks from the
    nearest triangulated vertex. The per-tet setup is done once here, the points come in batches.

    Tests run in the float64 of `tet_pts`: in float32 the faces of near-flat hull tets enclose points far outside them."""
    device = tet_pts.device
    N = tets.shape[0]
    tet_faces = torch.tensor(_TET_FACES, device=device)
    # planes from sorted vertex ids, oriented by the better-conditioned tet, so both tets of a face agree on sides;
    # built a chunk of tets at a time, since the gathered face vertices are several times the planes' size
    n = torch.empty((N, 4, 3), dtype=tet_pts.dtype, device=device)
    d = torch.empty((N, 4), dtype=tet_pts.dtype, device=device)
    opp = torch.empty((N, 4), dtype=tet_pts.dtype, device=device)
    for s in range(0, N, _VOXEL_CHUNK):
        chunk = tets[s:s + _VOXEL_CHUNK]
        fv = chunk[:, tet_faces].sort(-1).values
        a = tet_pts[fv[..., 0]]
        nc = torch.cross(tet_pts[fv[..., 1]] - a, tet_pts[fv[..., 2]] - a, dim=-1)
        n[s:s + _VOXEL_CHUNK] = nc
        d[s:s + _VOXEL_CHUNK] = (nc * a).sum(-1)
        opp[s:s + _VOXEL_CHUNK] = (nc * tet_pts[chunk]).sum(-1) - d[s:s + _VOXEL_CHUNK]  # opposite vertex goes on the inner side
        del chunk, fv, a, nc
    for s in range(0, N, _VOXEL_CHUNK):
        nb = nbr[s:s + _VOXEL_CHUNK]
        j = nb.clamp(min=0)
        rev = (nbr[j] == torch.arange(s, s + nb.shape[0], device=device)[:, None, None]).int().argmax(-1)
        opp_j = opp[j, rev]
        o = opp[s:s + _VOXEL_CHUNK]
        side = torch.where((nb >= 0) & (opp_j.abs() > o.abs()), torch.where(opp_j > 0, -1.0, 1.0), torch.where(o >= 0, 1.0, -1.0))
        n[s:s + _VOXEL_CHUNK] *= side[..., None]
        d[s:s + _VOXEL_CHUNK] *= side
        del nb, j, rev, opp_j, o, side
    del opp
    vert_tet = torch.full((tet_pts.shape[0],), -1, dtype=torch.long, device=device)
    vert_tet[tets.reshape(-1)] = torch.arange(tets.shape[0], device=device).repeat_interleave(4)
    used = torch.nonzero(vert_tet >= 0).squeeze(1)  # qhull leaves out near-duplicate points
    # nearest triangulated vertex, as the closest of zero-area triangles
    start = tet_pts[used].float()[:, None].expand(-1, 3, -1)
    tree = _build_udf_tree(start)
    gen = torch.Generator(device=device).manual_seed(0)

    def locate(points):
        out = torch.full((points.shape[0],), -1, dtype=torch.long, device=device)
        for s in range(0, points.shape[0], _POINT_CHUNK):
            q = points[s:s + _POINT_CHUNK].to(tet_pts.dtype)
            cur = vert_tet[used[_udf_exact(q.float(), start, k=1, tree=tree)[2]]]
            found = torch.full((q.shape[0],), -1, dtype=torch.long, device=device)
            todo = torch.arange(q.shape[0], device=device)
            for _ in range(max_steps):
                if todo.numel() == 0:
                    break
                t = cur[todo]
                outside = (n[t] * q[todo][:, None, :]).sum(-1) < d[t]
                inside = ~outside.any(1)
                found[todo[inside]] = t[inside]
                # leave through a random face the point is outside of: no cycles under rounding
                k = torch.where(outside, torch.rand(outside.shape, device=device, generator=gen), -1.0).argmax(1)
                step = nbr[t, k]
                moving = ~inside & (step >= 0)
                cur[todo[moving]] = step[moving]
                todo = todo[moving]
            found[todo] = cur[todo]
            out[s:s + q.shape[0]] = found
        return out

    return locate


def _solid_partition(shell_v: torch.Tensor, tri_verts: torch.Tensor, tree, eps: float, fill: float):
    """Label the tets of a Delaunay tetrahedralization of shell sites solid or outside by a min cut.

    Returns (tet points, tets, neighbours, solid per tet, outward boundary triangles (M, 3, 3))."""
    device = shell_v.device
    # sites: one shell vertex per block of 4 cells and offset direction, so both sides of a thin sheet keep theirs and
    # a corner gets one per facing; about the 5% a decimation kept, without its edges and quadrics
    away = shell_v - _udf_exact(shell_v, tri_verts, tree=tree)[1]
    axis = away.abs().argmax(1)
    facing = axis * 2 + (away.gather(1, axis[:, None])[:, 0] < 0).long()
    block = torch.floor(shell_v / (4 * eps)).long()
    block -= block.amin(0)
    span = block.amax(0) + 1
    cluster = torch.unique(((block[:, 0] * span[1] + block[:, 1]) * span[2] + block[:, 2]) * 6 + facing, return_inverse=True)[1]
    first = torch.full((int(cluster.max()) + 1,), shell_v.shape[0], dtype=torch.long, device=device)
    dv = shell_v[first.scatter_reduce(0, cluster, torch.arange(shell_v.shape[0], device=device), "amin")]
    del away, axis, facing, block, cluster, first
    # jitter: grid-aligned flat regions give cocircular points that qhull is very slow to merge
    gen = torch.Generator(device=device).manual_seed(0)
    dv = dv + (torch.rand(dv.shape, generator=gen, device=device, dtype=dv.dtype) * 2 - 1) * (1e-3 * eps)
    delaunay3d = _kitchen_op("delaunay3d", dv)
    if delaunay3d is not None:
        # tets on the kitchen's bounding-tet corners cover the space outside the points' hull
        tet_pts, tets, nbr = delaunay3d(dv)
        tets, nbr = tets.long(), nbr.long()
        outside = (tets >= dv.shape[0]).any(1)
    else:
        dt = scipy.spatial.Delaunay(dv.double().cpu().numpy(), qhull_options="Qbb Qc Qz Q12 Qt Q5")  # Q5: skip the final outer-plane check
        tet_pts = torch.from_numpy(dt.points).to(device)
        tets = torch.from_numpy(dt.simplices).to(device).long()
        nbr = torch.from_numpy(dt.neighbors).to(device).long()
        outside = (nbr < 0).any(1)

    # per chunk: the four face areas and whether the centroid lies in the shell layer; all faces of all tets at
    # once would be the partition's largest array
    tet_faces = torch.tensor(_TET_FACES, device=device)
    area = torch.empty((tets.shape[0], 4), dtype=tet_pts.dtype, device=device)
    shell = torch.empty(tets.shape[0], dtype=torch.bool, device=device)
    for s in range(0, tets.shape[0], _VOXEL_CHUNK):
        chunk = tets[s:s + _VOXEL_CHUNK]
        fv = tet_pts[chunk[:, tet_faces]]                                     # (n, 4, 3, 3)
        area[s:s + _VOXEL_CHUNK] = 0.5 * torch.cross(fv[:, :, 1] - fv[:, :, 0], fv[:, :, 2] - fv[:, :, 0], dim=-1).norm(dim=-1)
        shell[s:s + _VOXEL_CHUNK] = _udf_exact(tet_pts[chunk].mean(1).to(tri_verts.dtype), tri_verts, tree=tree)[0] < 0.95 * eps
        del chunk, fv
    hull = outside & ~shell

    # shell tets are solid, hull tets outside; the cut runs over face areas, (1 + fill)x higher between
    # two shell or two non-shell tets, so it follows the shell and only crosses open space to close small openings
    j = nbr.clamp(min=0)
    cap = torch.where(nbr >= 0, area * torch.where(shell[:, None] == shell[j], 1.0 + fill, 1.0), 0.0)
    free = ~shell & ~hull
    free_idx = torch.full((tets.shape[0],), -1, dtype=torch.long, device=device)
    free_idx[free] = torch.arange(int(free.sum()), device=device)
    fn, fcap = nbr[free], cap[free]
    fj = fn.clamp(min=0)
    to_free = (fn >= 0) & free[fj]
    solid = shell.clone()
    solid[free] = _min_cut(torch.where(to_free, free_idx[fj], -1), torch.where(to_free, fcap, 0.0),
                           torch.where((fn >= 0) & shell[fj], fcap, 0.0).sum(1),
                           torch.where((fn >= 0) & hull[fj], fcap, 0.0).sum(1))

    # boundary faces of solid tets, wound to face away from their tet
    bnd = solid[:, None] & ((nbr < 0) | ~solid[j])
    t, k = bnd.nonzero(as_tuple=True)
    tri = tet_pts[tets[t].gather(1, tet_faces[k])]
    opp = tet_pts[tets[t, k]]
    inward = (torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1) * (opp - tri[:, 0])).sum(-1) > 0
    tri = torch.where(inward[:, None, None], tri[:, [0, 2, 1]], tri)
    return tet_pts, tets, nbr, solid, tri.to(tri_verts.dtype)


def remesh_narrow_band_dc(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    resolution: int = 256,
    target_faces: int = 0,                  # 0 = use `resolution`; >0 = auto-derive resolution
    band: float = 1.0,
    project_back: float = 0.0,
    qef: bool = True,
    sign_mode: str = "udf",                 # "sdf" | "udf" | "solid"
    drop_small_components: float = 0.01,    # drop components below this fraction of the largest
    drop_inverted_components: bool = True,  # drop inside-out components
    drop_enclosed_components: bool = True,  # drop components inside the largest one
    fix_poles: bool = False,                # collapse adjacent valence-3 vertex pairs
    smooth_iters: int = 0,                  # Taubin smoothing iterations
    smooth_lambda: float = 0.5,
    smooth_mu: float = -0.53,
    manifold: bool = False,                 # Manifold DC, sdf mode only
    fill: float = 20.0,                     # solid: cost factor for closing openings through open space
    colors: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    center: Optional[torch.Tensor] = None,
):
    """Narrow-band dual contouring remesh; returns (vertices, faces, colors), colors None unless `colors` is given.

    sign_mode "sdf" signs corners by the nearest face normal, "udf" contours the unsigned distance at eps (closed inputs
    also give an inner shell for the component filters to drop), "solid" signs them by a graph cut that closes small openings
    and erodes the eps offset back to half a cell: a closing, so creases are filleted and gaps bridged at eps while the
    surface sits half a cell outside the input.
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

    # resolution from target_faces, at ~3 triangles per surface voxel
    if target_faces > 0:
        tv = vertices[faces.long()]
        cross_v = torch.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0], dim=-1)
        surface_area = 0.5 * cross_v.norm(dim=-1).sum().item()
        relative_area = max(surface_area / (scale * scale), 1e-6)
        derived = int(math.sqrt(target_faces / (3.0 * relative_area)))
        # multiple of 32: the band builder doubles up from a base <= 32
        derived = ((derived + 31) // 32) * 32
        derived = max(32, min(1024, derived))
        resolution = derived

    eps = band * scale / resolution

    # ticks: one per narrow-band level, SDF, DC, post-process, each smoothing iter, and 2 more in solid mode
    n_levels, _b = 1, resolution
    while _b > 32 and _b % 2 == 0:
        _b //= 2
    while _b < resolution:
        _b *= 2
        n_levels += 1
    _total_ticks = n_levels + 3 + int(smooth_iters) + (2 if sign_mode == "solid" else 0)
    _pbar = comfy.utils.ProgressBar(_total_ticks)
    _tq = _tqdm(total=_total_ticks, desc="Remesh DC", leave=False)

    def tick():
        _pbar.update(1)
        _tq.update(1)

    # Distance queries run on triangles no longer than a few cells; tri_src maps back to `faces`
    tri_verts_g, tri_src = _split_long_triangles(vertices[faces.long()], 4.0 * scale / resolution)

    # Step 1: narrow-band voxels
    voxel_coords, _band_tree = _build_narrow_band_voxels(
        tri_verts_g, center, scale, resolution, eps,
        progress_callback=tick)
    if voxel_coords.numel() == 0:
        return (torch.empty((0, 3), dtype=vertices.dtype, device=device),
                torch.empty((0, 3), dtype=faces.dtype, device=device),
                None if colors is None else torch.empty((0, colors.shape[1]),
                                                        dtype=colors.dtype, device=device))

    if sign_mode not in ("sdf", "udf", "solid"):
        raise ValueError(f"sign_mode must be 'sdf'|'udf'|'solid', got {sign_mode!r}")
    use_sdf = sign_mode == "sdf"

    # Step 2: the band as sorted voxel keys, and its unique corners
    CORNER_OFFS = torch.tensor(_CUBE_CORNERS, dtype=torch.long, device=device)
    corner_key_offs = _voxel_key(CORNER_OFFS, resolution)
    vox_keys = _voxel_key(voxel_coords, resolution).sort().values
    del voxel_coords

    def corner_world_of(keys):
        return (_decode_key(keys, resolution).float() / resolution - 0.5) * scale + center

    # a voxel's 8 corner keys are mostly its neighbours' too: dedupe per chunk before the unique over all of them
    unique_corner_keys = torch.unique(torch.cat([torch.unique(vox_keys[s:s + _VOXEL_CHUNK, None] + corner_key_offs)
                                                 for s in range(0, vox_keys.numel(), _VOXEL_CHUNK)]))
    corner_world = corner_world_of(unique_corner_keys)

    # Step 3: signed field at the corners
    # normals feed the SDF sign and QEF; QEF ignores their orientation, so it works in every sign mode
    if use_sdf or qef:
        tri_face_normals_all = torch.nn.functional.normalize(
            torch.cross(tri_verts_g[:, 1] - tri_verts_g[:, 0],
                        tri_verts_g[:, 2] - tri_verts_g[:, 0], dim=-1),
            p=2, dim=-1, eps=1e-12)
    cell_size = scale / resolution
    udf, corner_closest, corner_tri = _udf_exact(corner_world, tri_verts_g, tree=_band_tree)
    corner_valid = corner_tri >= 0
    if use_sdf:
        sign = torch.ones_like(udf)
        n_for_corner = tri_face_normals_all[corner_tri.clamp(min=0)]
        offset = corner_world - corner_closest
        sign_dot = (offset * n_for_corner).sum(-1)
        sign = torch.where(corner_valid & (sign_dot < 0), -sign, sign)
        sdf = sign * udf
        del sign, n_for_corner, offset, sign_dot, corner_world, udf, corner_closest, corner_tri
    elif sign_mode == "udf":
        sdf = udf - eps
        del corner_world, udf, corner_closest, corner_tri
    else:
        del corner_world, corner_closest, corner_tri   # the sign pass re-queries distances per chunk; the shell needs udf
        # graph-cut a tetrahedralization of the UDF shell into one solid and sign the corners by it
        shell_v = _dual_contour(vox_keys, udf - eps, unique_corner_keys, resolution, scale, center)[0]
        del udf
        tet_pts, tets, nbr, solid, cut_tri = _solid_partition(shell_v, tri_verts_g, _band_tree, eps, fill)
        locate = _tet_locator(tet_pts, tets, nbr)
        del shell_v, tet_pts, tets, nbr
        tick()
        # where the solid closes an opening its boundary crosses open space: widen the band to it
        cut_tri = _split_long_triangles(cut_tri, 4.0 * cell_size)[0]
        cut_coords, cut_tree = _build_narrow_band_voxels(cut_tri, center, scale, resolution, eps)
        cut_keys = torch.unique(_voxel_key(cut_coords, resolution))
        del cut_coords
        pos = torch.searchsorted(vox_keys, cut_keys).clamp(max=vox_keys.numel() - 1)
        cut_keys = cut_keys[vox_keys[pos] != cut_keys]
        vox_keys = _merge_sorted(vox_keys, cut_keys)[0]

        def solid_field(points, d, closest):
            # inside the shell is solid, elsewhere the cut tets decide; they only approximate the shell, so probe one eps further out
            inside = d < eps
            out = ~inside
            p, dd = points[out], d[out]
            probe = torch.where((dd < 2 * eps)[:, None], p + eps * torch.nn.functional.normalize(p - closest[out], dim=-1), p)
            tet = locate(probe)
            inside[out] = (tet >= 0) & solid[tet.clamp(min=0)]
            # no voxels exist past the grid edge, so the solid has to close before it
            g = ((points - center) / scale + 0.5) * resolution
            inside &= ((g > 0.5) & (g < resolution - 0.5)).all(1)
            # magnitude: offset distance near the input, distance to the cut across closed openings
            mag = (d - eps).abs()
            far = d >= 2 * eps
            mag[far] = _udf_exact(points[far], cut_tri, tree=cut_tree)[0]
            return torch.where(inside, -1.0, 1.0) * mag.clamp(min=1e-6 * cell_size)

        # the corners of the cut voxels join the band's; the field is evaluated a chunk at a time, distances included
        new = torch.unique(cut_keys[:, None] + corner_key_offs)
        del cut_keys
        pos = torch.searchsorted(unique_corner_keys, new).clamp(max=unique_corner_keys.numel() - 1)
        unique_corner_keys = _merge_sorted(unique_corner_keys, new[unique_corner_keys[pos] != new])[0]
        del pos, new
        field = lambda p: solid_field(p, *_udf_exact(p, tri_verts_g, tree=_band_tree)[:2])
        sdf = torch.cat([field(corner_world_of(unique_corner_keys[s:s + _POINT_CHUNK]))
                         for s in range(0, unique_corner_keys.numel(), _POINT_CHUNK)])
        vox_keys, unique_corner_keys, sdf = _close_band(vox_keys, unique_corner_keys, sdf, field, resolution, scale, center)
        # that surface is the input dilated by eps, which fills creases and gaps; erode it back to half a cell (a closing)
        erode = eps - 0.5 * cell_size
        if erode > 0:
            offset_v, offset_f = _dual_contour(vox_keys, sdf, unique_corner_keys, resolution, scale, center)
            offset_tri = offset_v[offset_f]
            offset_tree = _build_udf_tree(offset_tri)

            def eroded(points, sign):
                val = torch.where(sign < 0, -1.0, 1.0) * _udf_exact(points, offset_tri, tree=offset_tree)[0] + erode
                return val.masked_fill(val == 0, 1e-6 * cell_size)  # a zero crosses for _close_band but not for _dual_contour

            # _close_band left every corner of every voxel in it
            corner_sdf = torch.cat([eroded(corner_world_of(unique_corner_keys[s:s + _POINT_CHUNK]), sdf[s:s + _POINT_CHUNK])
                                    for s in range(0, unique_corner_keys.numel(), _POINT_CHUNK)])
            vox_keys, unique_corner_keys, sdf = _close_band(
                vox_keys, unique_corner_keys, corner_sdf,
                lambda p: eroded(p, solid_field(p, *_udf_exact(p, tri_verts_g, tree=_band_tree)[:2])), resolution, scale, center)
            del offset_v, offset_f, offset_tri, offset_tree, corner_sdf
        del locate, solid, cut_tri, cut_tree
        tick()
    tick()  # SDF done

    # Step 4: dual contouring; QEF fits planes through the offset crossings with the input's normals, keeping creases sharp
    if qef:
        tri_face_normals = tri_face_normals_all
        def _qef_query(pts):
            d, closest, tri = _udf_exact(pts, tri_verts_g, tree=_band_tree)
            if sign_mode == "solid":  # crossings on closed openings have no input plane to snap to
                tri = torch.where(d < 2 * eps, tri, -1)
            return d, closest, tri
    else:
        tri_face_normals = None
        _qef_query = None

    if manifold and use_sdf:
        # MDC places verts at centroids, no QEF
        dual_verts, new_faces = _dual_contour_manifold(
            _decode_key(vox_keys, resolution), sdf, unique_corner_keys,
            resolution, scale, center,
            corner_valid=corner_valid)
    else:
        dual_verts, new_faces = _dual_contour(
            vox_keys, sdf, unique_corner_keys,
            resolution, scale, center,
            tri_face_normals=tri_face_normals, qef_query=_qef_query,
            # corner_valid only matters for sdf
            corner_valid=corner_valid if use_sdf else None)
    if sign_mode == "solid" and new_faces.numel() > 0:
        # the closed contour is two-manifold only where one dual vertex carries one sheet
        new_faces, src_vert = _split_touching_sheets(dual_verts, new_faces)
        dual_verts = dual_verts[src_vert]
    del vox_keys, sdf, unique_corner_keys, corner_valid
    del tri_face_normals, _qef_query
    if use_sdf or qef:
        del tri_face_normals_all
    tick()  # DC done

    # Step 5: project_back and color sampling share one closest-point query
    need_query = (project_back > 0 or colors is not None) and dual_verts.numel() > 0
    out_colors = None
    if need_query:
        d, closest_pts, closest_tri = _udf_exact(dual_verts, tri_verts_g, tree=_band_tree)

        if project_back > 0:
            # only verts near the input, not lids over openings closed in solid mode
            back = (d <= 4.0 * cell_size)[:, None]
            dual_verts = torch.where(back, torch.lerp(dual_verts, closest_pts, float(project_back)), dual_verts)

        if colors is not None:
            # barycentric-interpolate the input colors at the closest point
            tri_v_idx = faces[tri_src[closest_tri]].long()
            tri_v = vertices[tri_v_idx]
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

    # Step 6: post-process
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

    # drop dual verts no face uses (voxels without crossings)
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
