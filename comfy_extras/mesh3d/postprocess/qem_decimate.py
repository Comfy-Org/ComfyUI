"""
Pure-PyTorch GPU-parallel QEM mesh simplification.

  - Parallel greedy edge-matching collapse loop
  - Plane/line/feature-edge/boundary quadrics, memoryless accumulation
  - Normal-flip prevention, link-condition, skinny penalties
  - Non-manifold/sliver handling without dropping faces
  - Pre/post-clean pipeline (weld, degenerates, small components)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math

import numpy as _np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm as _tqdm
import comfy.utils as _comfy_utils


@dataclass
class QEMConfig:
    # Precision
    dtype: torch.dtype = torch.float32  # float64 much slower on consumer GPUs

    # Numerical conditioning
    stabilizer_scale: float = 1e-3  # Tikhonov reg: stabilizer = mesh_scale^2 * this
    wander_threshold: float = 2.0  # fall back to midpoint if v* lands > N×edge_length from an endpoint
    clamp_v_to_edge: bool = True  # project v* onto the edge segment (qem mode only)

    # Placement mode (also selects collapse driver):
    # "midpoint" = threshold-schedule driver, most stable (defaults below match it);
    # "qem" = sharpest, QEM-optimum placement + ratio driver.
    placement_mode: str = "midpoint"

    flip_reject_hard: bool = True  # hard-reject (err=+inf) top-K collapses that flip any 1-ring normal

    # Per-iteration batch sizing
    sampling_cap: int = 10_000_000  # max edges processed per outer iter
    max_collapses_fraction: float = 0.25  # of remaining faces-to-remove
    max_collapses_floor: int = 10_000
    max_collapses_ceiling: int = 1_000_000
    max_collapses_relative_cap: float = 0.10  # cap per-iter collapses as fraction of current faces; 0 disables

    # Loop control
    max_iterations: int = 5_000
    compaction_period: int = 5
    compaction_threshold: float = 0.85  # compact when alive_frac < this

    # Quality knobs
    boundary_quadrics: bool = True
    boundary_weight: float = 1000.0
    recompute_normals_post: bool = True
    line_quadric_weight: float = 0.0  # penalise deviation ⟂ to edge dir → more uniform verts; 0 disables
    line_quadric_skip_opposite_normals_cos: float = 0.0  # skip line quadrics on edges with endpoint cos < this

    # Feature-edge quadrics on sharp interior edges (dihedral > min); 0 disables.
    feature_edge_quadric_weight: float = 0.0
    feature_edge_min_dihedral_deg: float = 30.0

    # Flip check (FA-QEM §3.3)
    quality_topk_multiplier: int = 4  # quality-check band size = this * max_collapses_per_iter
    flip_cos_threshold: float = 0.0  # 0 = count any sign reversal (dihedral > 90°)
    flip_check_max_degree: int = 16  # cap on vertex degree for the flip-check table

    # Triangle shape penalty
    skinny_weight: float = 1e-3  # penalise top-K collapses producing needle/sliver tris; 0 disables

    #  Topology preservation
    enforce_link_condition: bool = True  # reject collapses that violate the link condition

    # Quadric area weighting
    area_weighted_quadrics: bool = False  # True: Garland-Heckbert area-weighted; False: un-weighted

    # edge-length cost regularizer
    lambda_edge_length: float = 1e-2  # add λ*len² to bias toward short edges; 0 disables
    lambda_edge_length_absolute: bool = True  # apply λ absolutely vs relative-to-QEM-median

    # Threshold-schedule driver (placement_mode == "midpoint"):
    # each round collapses a disjoint set with cost <= thresh, ×10 when < 1% removed.
    threshold_start: float = 1e-8
    memoryless_qem: bool = True  # rebuild quadrics each round vs accumulate
    repair_nonmanifold: bool = True  # final repair_non_manifold_edges pass

    # Pre-clean (input mesh)
    preclean: bool = True  # weld coincident verts, drop degenerate/duplicate/unused

    # Post-clean (output mesh)
    postclean: bool = True  # remove slivers, tiny components, unused verts left by collapse
    postclean_min_angle_deg: float = 0.5
    postclean_max_aspect_ratio: float = 100.0
    postclean_min_component_faces: int = 8  # drop components with fewer faces than this

    # Preclean tuning
    preclean_weld_epsilon_rel: float = 1e-5  # weld tolerance as fraction of bbox diagonal
    preclean_min_component_faces: int = 0  # 0 = keep all components


    @property
    def threshold_driver(self) -> bool:
        """The cost-threshold collapse driver is used by the midpoint placement mode."""
        return self.placement_mode == "midpoint"


def _sorted_edge_halfedges(
    faces: torch.Tensor, num_verts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """3F half-edges sorted by key min(a,b)*(V+1)+max(a,b); returns (sorted_keys, face_ids, slot_ids)."""
    device = faces.device
    F = faces.shape[0]
    e_all = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)
    e_sorted, _ = torch.sort(e_all, dim=1)
    P = num_verts + 1
    key = e_sorted[:, 0].long() * P + e_sorted[:, 1].long()
    face_per_he = torch.arange(F, device=device, dtype=torch.long).repeat(3)
    slot_per_he = torch.arange(3, device=device, dtype=torch.long).repeat_interleave(F)
    sort_idx = torch.argsort(key)
    return key[sort_idx], face_per_he[sort_idx], slot_per_he[sort_idx]


def _vert_is_boundary_mask(faces: torch.Tensor, num_verts: int) -> torch.Tensor:
    """(V,) bool mask: True for verts incident to any boundary edge."""
    device = faces.device
    out = torch.zeros(num_verts, dtype=torch.bool, device=device)
    bedges = _detect_boundary_edges(faces, num_verts)
    if bedges.numel() == 0:
        return out
    out[bedges[:, 0]] = True
    out[bedges[:, 1]] = True
    return out


def _detect_boundary_edges(faces: torch.Tensor, num_verts: int) -> torch.Tensor:
    """Boundary edges as [N, 2] of vertex indices (each appearing in exactly one face)."""
    if faces.numel() == 0:
        return torch.empty((0, 2), dtype=torch.int64, device=faces.device)
    sorted_keys, _, _ = _sorted_edge_halfedges(faces, num_verts)
    unique_key, counts = torch.unique(sorted_keys, return_counts=True)
    boundary_key = unique_key[counts == 1]
    if boundary_key.numel() == 0:
        return torch.empty((0, 2), dtype=torch.int64, device=faces.device)
    P = num_verts + 1
    bv0 = boundary_key // P
    bv1 = boundary_key % P
    return torch.stack([bv0, bv1], dim=1)


def _manifold_edge_pairs(
    sorted_keys: torch.Tensor, sorted_faces: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Edges shared by exactly 2 faces (filters >2-incident groups); returns (pair_keys, fa, fb)."""
    if sorted_keys.shape[0] < 2:
        empty = sorted_keys.new_empty(0)
        return empty, empty, empty
    pair_mask = sorted_keys[:-1] == sorted_keys[1:]
    if not pair_mask.any():
        empty = sorted_keys.new_empty(0)
        return empty, empty, empty
    pair_starts = torch.nonzero(pair_mask, as_tuple=True)[0]
    # manifold iff neither neighbour half-edge shares the key
    cur = sorted_keys[pair_starts]
    prev_ok = (pair_starts == 0) | (sorted_keys[(pair_starts - 1).clamp_min(0)] != cur)
    nxt_idx = (pair_starts + 2).clamp(max=sorted_keys.shape[0] - 1)
    nxt_ok = (pair_starts + 2 >= sorted_keys.shape[0]) | (sorted_keys[nxt_idx] != cur)
    pair_starts = pair_starts[prev_ok & nxt_ok]
    return (sorted_keys[pair_starts],
            sorted_faces[pair_starts],
            sorted_faces[pair_starts + 1])


def _line_quadric_planes(
    pa: torch.Tensor, pb: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Two plane equations (E,4) per edge whose squared-dist sum = squared ⟂ distance to the edge line."""
    e = pb - pa                                                # (E, 3)
    elen = torch.norm(e, dim=-1, keepdim=True).clamp_min(1e-12)
    e_unit = e / elen                                          # (E, 3)
    m = 0.5 * (pa + pb)                                        # (E, 3)
    # helper axis not parallel to e_unit, then Gram-Schmidt against e_unit
    helper = torch.zeros_like(e_unit)
    helper.scatter_(-1, e_unit.abs().argmin(dim=-1, keepdim=True), 1.0)
    u = helper - (helper * e_unit).sum(-1, keepdim=True) * e_unit
    u = u / torch.norm(u, dim=-1, keepdim=True).clamp_min(1e-12)
    w = torch.cross(e_unit, u, dim=-1)
    d_u = -(u * m).sum(-1, keepdim=True)
    d_w = -(w * m).sum(-1, keepdim=True)
    p_u = torch.cat([u, d_u], dim=-1)                          # (E, 4)
    p_w = torch.cat([w, d_w], dim=-1)
    return p_u, p_w, elen.squeeze(-1)


def _add_line_quadrics(
    verts: torch.Tensor,
    faces: torch.Tensor,
    face_areas: torch.Tensor,
    Q_flat: torch.Tensor,
    weight: float,
    skip_he_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Add line quadrics for all 3F half-edges, weighted by face_area*weight; skip_he_mask zeroes True positions."""
    a_all = torch.cat([faces[:, 0], faces[:, 1], faces[:, 2]], dim=0).long()
    b_all = torch.cat([faces[:, 1], faces[:, 2], faces[:, 0]], dim=0).long()
    pa = verts[a_all]
    pb = verts[b_all]
    p_u, p_w, _ = _line_quadric_planes(pa, pb)
    area_per_edge = face_areas.repeat(3)
    w_per_edge = area_per_edge * weight
    if skip_he_mask is not None:
        w_per_edge = torch.where(skip_he_mask, torch.zeros_like(w_per_edge), w_per_edge)
    w_per_edge = w_per_edge.unsqueeze(-1).unsqueeze(-1)
    K_line = (
        p_u.unsqueeze(-1) * p_u.unsqueeze(-2)
        + p_w.unsqueeze(-1) * p_w.unsqueeze(-2)
    ) * w_per_edge
    K_flat = K_line.reshape(-1, 16)
    Q_flat.scatter_add_(0, a_all.unsqueeze(1).expand(-1, 16), K_flat)  # scatter to both endpoints
    Q_flat.scatter_add_(0, b_all.unsqueeze(1).expand(-1, 16), K_flat)
    return Q_flat


def _build_quadrics(
    verts: torch.Tensor,
    faces: torch.Tensor,
    cfg: QEMConfig,
) -> torch.Tensor:
    """Per-vertex area-weighted quadric (V, 4, 4)."""
    V = verts.shape[0]
    dtype = verts.dtype
    device = verts.device

    Q_flat = torch.zeros((V, 16), dtype=dtype, device=device)

    if faces.numel() > 0:
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        e1 = v1 - v0
        e2 = v2 - v0
        n = torch.cross(e1, e2, dim=-1)
        area = torch.norm(n, dim=-1)
        mask = area > 1e-12
        # where() avoids boolean-index gather+scatter (fewer index kernels)
        n_norm = torch.where(mask.unsqueeze(-1),
                             n / area.unsqueeze(-1).clamp_min(1e-12),
                             n.new_zeros(()))
        d = -(n_norm * v0).sum(dim=-1, keepdim=True)
        p = torch.cat([n_norm, d], dim=-1)           # (F, 4)
        K = torch.einsum("fi,fj->fij", p, p)         # (F, 4, 4)

        if cfg.area_weighted_quadrics:
            K.mul_(area[:, None, None])
        K_flat = K.reshape(-1, 16)
        for corner in range(3):
            idx = faces[:, corner].unsqueeze(1).expand(-1, 16)
            Q_flat.scatter_add_(0, idx, K_flat)

    # Line quadrics: squared ⟂ distance from v to the edge-midpoint line, all 3F half-edges in one pass.
    if cfg.line_quadric_weight > 0 and faces.numel() > 0:
        # skip thin-shell rim edges (endpoint normals oppose)
        skip_he_sharp = None
        if cfg.line_quadric_skip_opposite_normals_cos < 1.0:
            v_norm = torch.zeros((V, 3), dtype=dtype, device=device)
            n_weighted = n_norm * area.unsqueeze(-1)  # normal * 2× area
            for corner in range(3):
                v_norm.scatter_add_(0, faces[:, corner].unsqueeze(-1).expand(-1, 3),
                                     n_weighted)
            v_norm = torch.nn.functional.normalize(v_norm, p=2, dim=-1, eps=1e-12)
            a_he = torch.cat([faces[:, 0], faces[:, 1], faces[:, 2]], dim=0).long()
            b_he = torch.cat([faces[:, 1], faces[:, 2], faces[:, 0]], dim=0).long()
            cos_endpoints = (v_norm[a_he] * v_norm[b_he]).sum(dim=-1)
            skip_he_sharp = cos_endpoints < cfg.line_quadric_skip_opposite_normals_cos
            if not skip_he_sharp.any():
                skip_he_sharp = None
        Q_flat = _add_line_quadrics(verts, faces, area, Q_flat,
                                     cfg.line_quadric_weight,
                                     skip_he_mask=skip_he_sharp)

    # Boundary line quadrics: pin boundary-edge endpoints to the boundary line.
    if cfg.boundary_quadrics and faces.numel() > 0:
        b_edges = _detect_boundary_edges(faces, V)
        if b_edges.shape[0] > 0:
            ba = b_edges[:, 0]
            bb = b_edges[:, 1]
            pa = verts[ba]
            pb = verts[bb]
            p_u, p_w, _ = _line_quadric_planes(pa, pb)
            K_b = (torch.einsum("ei,ej->eij", p_u, p_u)
                   + torch.einsum("ei,ej->eij", p_w, p_w)) * cfg.boundary_weight
            K_b_flat = K_b.reshape(-1, 16)
            Q_flat.scatter_add_(0, ba.unsqueeze(1).expand(-1, 16), K_b_flat)
            Q_flat.scatter_add_(0, bb.unsqueeze(1).expand(-1, 16), K_b_flat)

    # Feature-edge quadrics: line quadric on sharp interior edges weighted by (1 - cos(dihedral)).
    if cfg.feature_edge_quadric_weight > 0 and faces.numel() > 0:
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        fn = torch.cross(v1 - v0, v2 - v0, dim=-1)
        fn = torch.nn.functional.normalize(fn, p=2, dim=-1, eps=1e-12)
        sorted_keys_fe, sorted_faces_fe, _ = _sorted_edge_halfedges(faces, V)
        pair_keys, f1_idx, f2_idx = _manifold_edge_pairs(sorted_keys_fe, sorted_faces_fe)
        if pair_keys.numel() > 0:
            P = V + 1
            edge_a = pair_keys // P
            edge_b = pair_keys % P
            cos_dihedral = (fn[f1_idx] * fn[f2_idx]).sum(dim=-1)
            cos_thresh = math.cos(math.radians(cfg.feature_edge_min_dihedral_deg))
            sharp = cos_dihedral < cos_thresh
            if sharp.any():
                fa = edge_a[sharp]
                fb = edge_b[sharp]
                p_u, p_w, _ = _line_quadric_planes(verts[fa], verts[fb])
                sharpness = (1.0 - cos_dihedral[sharp]).clamp_min(0.0)
                avg_area = 0.5 * (area[f1_idx[sharp]] + area[f2_idx[sharp]])
                w = (avg_area * sharpness * cfg.feature_edge_quadric_weight) \
                    .unsqueeze(-1).unsqueeze(-1)
                K_feat = (
                    p_u.unsqueeze(-1) * p_u.unsqueeze(-2)
                    + p_w.unsqueeze(-1) * p_w.unsqueeze(-2)
                ) * w
                K_flat = K_feat.reshape(-1, 16)
                Q_flat.scatter_add_(0, fa.unsqueeze(1).expand(-1, 16), K_flat)
                Q_flat.scatter_add_(0, fb.unsqueeze(1).expand(-1, 16), K_flat)

    return Q_flat.reshape(V, 4, 4)


def _edge_errors(
    verts: torch.Tensor,
    Q: torch.Tensor,
    edges: torch.Tensor,
    stabilizer: float,
    max_edge_length_sq: float,
    mesh_scale_sq: float,
    cfg: QEMConfig,
    vert_is_boundary: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (optimal_pos, error, valid_mask); vert_is_boundary enables boundary-aware midpoint."""
    n_edges = edges.shape[0]
    dtype = verts.dtype
    device = verts.device

    if n_edges == 0:
        return (
            torch.empty((0, 3), dtype=dtype, device=device),
            torch.empty((0,), dtype=dtype, device=device),
            torch.zeros((0,), dtype=torch.bool, device=device),
        )

    verts_pair = verts[edges]           # (E, 2, 3)
    pa = verts_pair[:, 0]
    pb = verts_pair[:, 1]
    edge_vec = pb - pa
    el = torch.norm(edge_vec, dim=-1)

    # boundary-aware midpoint: snap to the boundary endpoint when exactly one is boundary
    if vert_is_boundary is not None:
        ba = vert_is_boundary[edges[:, 0]]
        bb = vert_is_boundary[edges[:, 1]]
        w_a = torch.where(ba & ~bb, torch.ones_like(el),
              torch.where(~ba & bb, torch.zeros_like(el),
              torch.full_like(el, 0.5)))
        midpoint = pa * w_a.unsqueeze(-1) + pb * (1.0 - w_a).unsqueeze(-1)
    else:
        midpoint = torch.lerp(pa, pb, 0.5)

    Qe = Q[edges].sum(dim=1)            # (E, 4, 4) — sum of Q[va] and Q[vb]

    if cfg.placement_mode == "midpoint":
        opt = midpoint
    else:
        A = Qe[:, :3, :3] + torch.eye(3, device=device, dtype=dtype) * stabilizer
        b = -Qe[:, :3, 3].unsqueeze(-1)

        # stabilizer keeps A invertible; full-batch solve, midpoint fallback via where (no sync)
        sol = torch.linalg.solve(A, b)
        dets = torch.det(A)
        good = (dets.abs() > 1e-12).unsqueeze(-1)
        opt = torch.where(good, sol.squeeze(-1), midpoint)

        if cfg.clamp_v_to_edge:
            # project v* onto the edge segment (subsumes the wander check)
            edge_len_sq = (edge_vec * edge_vec).sum(dim=-1) + 1e-20
            t = ((opt - pa) * edge_vec).sum(dim=-1) / edge_len_sq
            t = t.clamp(0.0, 1.0).unsqueeze(-1)
            opt = torch.lerp(pa, pb, t)
        else:
            # fall back to midpoint when v* wanders from both endpoints
            dist_a = torch.norm(opt - pa, dim=-1)
            dist_b = torch.norm(opt - pb, dim=-1)
            wander_bad = ((dist_a > cfg.wander_threshold * el) |
                          (dist_b > cfg.wander_threshold * el)).unsqueeze(-1)
            opt = torch.where(wander_bad, midpoint, opt)

    v4 = torch.cat([opt, torch.ones((n_edges, 1), device=device, dtype=dtype)], dim=1)
    err = torch.abs(torch.einsum("ei,eij,ej->e", v4, Qe, v4))

    # mesh_scale_sq: Python float or 0-d tensor
    if torch.is_tensor(mesh_scale_sq):
        length_ok = el * el > mesh_scale_sq * 1e-10
    else:
        length_ok = el > math.sqrt(mesh_scale_sq) * 1e-5
    error_ok = err < max_edge_length_sq
    nan_ok = ~torch.isnan(opt).any(dim=-1) & ~torch.isnan(err)
    valid = length_ok & error_ok & nan_ok

    # edge-length regularizer: bias collapse order toward short edges (uniform sizing)
    if cfg.lambda_edge_length > 0.0 and valid.any():
        el2 = el * el
        if cfg.lambda_edge_length_absolute:
            err = err + cfg.lambda_edge_length * el2
        else:
            qem_med = err[valid].median()
            len_med = el2[valid].median().clamp_min(1e-30)
            err = err + cfg.lambda_edge_length * el2 * (qem_med / len_med)
    return opt, err, valid


def _greedy_matching(
    edges: torch.Tensor,
    err: torch.Tensor,
    v_alive: torch.Tensor,
    max_select: int,
) -> torch.Tensor:
    """Vectorised independent edge-set selection: an edge wins iff it is the min-key edge at both endpoints."""
    device = edges.device
    n_edges = edges.shape[0]
    if n_edges == 0:
        return torch.empty(0, dtype=torch.int64, device=device)

    va = edges[:, 0]
    vb = edges[:, 1]
    num_verts = v_alive.shape[0]

    err32 = err.to(torch.float32).clamp(min=0).contiguous()
    err_bits = err32.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    edge_idx = torch.arange(n_edges, device=device, dtype=torch.int64)
    key = (err_bits << 32) | edge_idx

    INT64_MAX = torch.iinfo(torch.int64).max
    best_key = torch.full((num_verts,), INT64_MAX, dtype=torch.int64, device=device)
    best_key.scatter_reduce_(0, va, key, reduce="amin", include_self=True)
    best_key.scatter_reduce_(0, vb, key, reduce="amin", include_self=True)

    is_winner = (key == best_key[va]) & (key == best_key[vb]) & v_alive[va] & v_alive[vb]
    sel = torch.nonzero(is_winner, as_tuple=True)[0]

    if sel.numel() > max_select:
        sel_err = err[sel]
        top = torch.topk(sel_err, max_select, largest=False).indices
        sel = sel[top]
    return sel


def _build_vert_to_faces_pad(
    faces: torch.Tensor,
    num_verts: int,
    max_deg: int,
) -> torch.Tensor:
    """Pad-CSR vertex-to-incident-faces table (V, max_deg) of face indices, -1 padded, degree truncated."""
    device = faces.device
    F = faces.shape[0]
    if F == 0:
        return torch.full((num_verts, max_deg), -1, dtype=torch.int64, device=device)
    v_rep = faces.flatten().long()
    f_rep = torch.arange(F, device=device, dtype=torch.int64).repeat_interleave(3)
    sort_idx = v_rep.argsort()
    sorted_v = v_rep[sort_idx]
    sorted_f = f_rep[sort_idx]
    offsets = torch.searchsorted(
        sorted_v, torch.arange(num_verts + 1, device=device, dtype=sorted_v.dtype)
    )
    slot = torch.arange(sorted_v.shape[0], device=device, dtype=torch.int64) - offsets[sorted_v]
    keep = slot < max_deg
    table = torch.full((num_verts, max_deg), -1, dtype=torch.int64, device=device)
    table[sorted_v[keep], slot[keep]] = sorted_f[keep]
    return table


def _normal_flip_mask(
    verts: torch.Tensor,        # (V, 3)
    faces: torch.Tensor,        # (F, 3) — must be alive faces only
    edges: torch.Tensor,        # (E, 2) candidate collapse edges
    opt: torch.Tensor,          # (E, 3) proposed collapse positions
    vert_to_faces: torch.Tensor,  # (V, max_deg) face indices or -1
    cos_threshold: float = 0.0,
    chunk_size: int = 100_000,
    return_count: bool = False,
) -> torch.Tensor:
    """(E,) bool mask (no adjacent-face flip), or int count of would-flip faces per edge if return_count."""
    E = edges.shape[0]
    device = verts.device
    if return_count:
        out = torch.zeros(E, dtype=torch.int32, device=device)
    else:
        out = torch.ones(E, dtype=torch.bool, device=device)
    if E == 0:
        return out

    max_deg = vert_to_faces.shape[1]
    a_all = edges[:, 0]
    b_all = edges[:, 1]

    for start in range(0, E, chunk_size):
        stop = min(start + chunk_size, E)
        Ec = stop - start
        a = a_all[start:stop]
        b = b_all[start:stop]
        oc = opt[start:stop]

        fa = vert_to_faces[a]                                 # (Ec, max_deg)
        fb = vert_to_faces[b]
        all_f = torch.cat([fa, fb], dim=1)                    # (Ec, 2*max_deg)
        valid_f = all_f >= 0
        all_f_safe = all_f.clamp(min=0)
        fv = faces[all_f_safe]                                # (Ec, 2*max_deg, 3)

        a_b = a.view(Ec, 1)
        b_b = b.view(Ec, 1)
        s0_a = fv[..., 0] == a_b
        s0_b = fv[..., 0] == b_b
        s1_a = fv[..., 1] == a_b
        s1_b = fv[..., 1] == b_b
        s2_a = fv[..., 2] == a_b
        s2_b = fv[..., 2] == b_b
        contains_a = s0_a | s1_a | s2_a
        contains_b = s0_b | s1_b | s2_b
        # affected: face contains exactly one of {a, b} and slot is non-pad
        affected = (contains_a ^ contains_b) & valid_f
        if not affected.any():
            continue

        p0 = verts[fv[..., 0]]                                # (Ec, 2*max_deg, 3)
        p1 = verts[fv[..., 1]]
        p2 = verts[fv[..., 2]]
        n_old = torch.cross(p1 - p0, p2 - p0, dim=-1)

        opt_b = oc.view(Ec, 1, 3).expand(-1, 2 * max_deg, -1)
        rep0 = (s0_a | s0_b).unsqueeze(-1)
        rep1 = (s1_a | s1_b).unsqueeze(-1)
        rep2 = (s2_a | s2_b).unsqueeze(-1)
        p0n = torch.where(rep0, opt_b, p0)
        p1n = torch.where(rep1, opt_b, p1)
        p2n = torch.where(rep2, opt_b, p2)
        n_new = torch.cross(p1n - p0n, p2n - p0n, dim=-1)

        nlen_old = torch.norm(n_old, dim=-1)
        nlen_new = torch.norm(n_new, dim=-1)
        # degenerate-before faces can't flip; treat as OK
        denom = nlen_old * nlen_new
        safe = denom > 1e-20
        cos = torch.where(safe, (n_old * n_new).sum(dim=-1) / denom.clamp_min(1e-20),
                          torch.ones_like(denom))
        flip = (cos < cos_threshold) & affected & safe
        if return_count:
            out[start:stop] = flip.sum(dim=-1).to(torch.int32)
        else:
            out[start:stop] = ~flip.any(dim=-1)

    return out


def _link_condition_mask(
    faces: torch.Tensor,         # (F, 3) alive faces only
    edges: torch.Tensor,         # (E, 2) candidate collapse edges
    vert_to_faces: torch.Tensor, # (V, max_deg) face idx or -1
    chunk_size: int = 100_000,
) -> torch.Tensor:
    """(E,) bool mask — True where the collapse is topology-safe (link condition: common neighbours <= edge faces)."""
    E = edges.shape[0]
    device = faces.device
    out = torch.ones(E, dtype=torch.bool, device=device)
    if E == 0:
        return out
    D = vert_to_faces.shape[1]
    a_all = edges[:, 0]
    b_all = edges[:, 1]

    for s in range(0, E, chunk_size):
        e = min(s + chunk_size, E)
        a = a_all[s:e]
        b = b_all[s:e]
        Ec = a.shape[0]

        fa = vert_to_faces[a]                       # (Ec, D)
        fb = vert_to_faces[b]
        fa_ok = fa >= 0
        fb_ok = fb >= 0
        fav = faces[fa.clamp(min=0)]                # (Ec, D, 3)
        fbv = faces[fb.clamp(min=0)]

        # neighbour verts of a/b: take the 2 non-anchor verts per incident face → (Ec, 2D)
        a_b = a[:, None]
        b_b = b[:, None]
        an1 = torch.where(fav[..., 0] == a_b, fav[..., 1], fav[..., 0])
        an2 = torch.where(fav[..., 2] == a_b, fav[..., 1], fav[..., 2])
        bn1 = torch.where(fbv[..., 0] == b_b, fbv[..., 1], fbv[..., 0])
        bn2 = torch.where(fbv[..., 2] == b_b, fbv[..., 1], fbv[..., 2])
        na = torch.stack([an1, an2], dim=-1).reshape(Ec, 2 * D)
        nb = torch.stack([bn1, bn2], dim=-1).reshape(Ec, 2 * D)
        fa_okx = fa_ok.repeat_interleave(2, dim=1)
        fb_okx = fb_ok.repeat_interleave(2, dim=1)
        na[(na == a_b) | (na == b_b) | ~fa_okx] = -1
        nb[(nb == a_b) | (nb == b_b) | ~fb_okx] = -1

        # common neighbours: na entries also appearing in nb
        in_b = (na[:, :, None] == nb[:, None, :]) & (na[:, :, None] >= 0)
        na_common = torch.where(in_b.any(dim=2), na, torch.full_like(na, -1))
        # distinct count of common neighbours per edge (sort + count transitions)
        cs, _ = na_common.sort(dim=1)
        count_common = ((cs[:, 1:] != cs[:, :-1]) & (cs[:, 1:] >= 0)).sum(dim=1) \
                       + (cs[:, :1] >= 0).sum(dim=1)

        # faces on the edge = a's faces also containing b
        count_faces = ((fav == b[:, None, None]).any(dim=2) & fa_ok).sum(dim=1)

        out[s:e] = count_common <= count_faces

    return out


def _skinny_penalty(
    verts: torch.Tensor,        # (V, 3)
    faces: torch.Tensor,        # (F, 3) — alive faces only
    edges: torch.Tensor,        # (E, 2) candidate collapse edges
    opt: torch.Tensor,          # (E, 3) proposed collapse positions
    vert_to_faces: torch.Tensor,  # (V, max_deg)
    chunk_size: int = 100_000,
) -> torch.Tensor:
    """Per-edge post-collapse triangle-shape penalty (lambda_skinny); mean of 1 - clamp(shape,0,1) over the 1-ring."""
    E = edges.shape[0]
    device = verts.device
    out = torch.zeros(E, dtype=verts.dtype, device=device)
    if E == 0:
        return out

    max_deg = vert_to_faces.shape[1]
    a_all = edges[:, 0]
    b_all = edges[:, 1]
    sqrt3_4 = 4.0 * math.sqrt(3.0)

    for start in range(0, E, chunk_size):
        stop = min(start + chunk_size, E)
        Ec = stop - start
        a = a_all[start:stop]
        b = b_all[start:stop]
        oc = opt[start:stop]

        fa = vert_to_faces[a]
        fb = vert_to_faces[b]
        all_f = torch.cat([fa, fb], dim=1)
        valid_f = all_f >= 0
        all_f_safe = all_f.clamp(min=0)
        fv = faces[all_f_safe]

        a_b = a.view(Ec, 1)
        b_b = b.view(Ec, 1)
        s0_a = fv[..., 0] == a_b
        s0_b = fv[..., 0] == b_b
        s1_a = fv[..., 1] == a_b
        s1_b = fv[..., 1] == b_b
        s2_a = fv[..., 2] == a_b
        s2_b = fv[..., 2] == b_b
        contains_a = s0_a | s1_a | s2_a
        contains_b = s0_b | s1_b | s2_b
        # affected: face contains exactly one of {a, b} and slot is non-pad
        affected = (contains_a ^ contains_b) & valid_f
        if not affected.any():
            continue

        p0 = verts[fv[..., 0]]
        p1 = verts[fv[..., 1]]
        p2 = verts[fv[..., 2]]
        opt_b = oc.view(Ec, 1, 3).expand(-1, 2 * max_deg, -1)
        rep0 = (s0_a | s0_b).unsqueeze(-1)
        rep1 = (s1_a | s1_b).unsqueeze(-1)
        rep2 = (s2_a | s2_b).unsqueeze(-1)
        p0n = torch.where(rep0, opt_b, p0)
        p1n = torch.where(rep1, opt_b, p1)
        p2n = torch.where(rep2, opt_b, p2)

        e01 = p1n - p0n
        e02 = p2n - p0n
        e12 = p2n - p1n
        two_area = torch.cross(e01, e02, dim=-1).norm(dim=-1)
        edge_sum_sq = ((e01 * e01).sum(-1)
                       + (e02 * e02).sum(-1)
                       + (e12 * e12).sum(-1))
        shape = (sqrt3_4 * 0.5 * two_area) / edge_sum_sq.clamp_min(1e-20)
        term = 1.0 - shape.clamp(0.0, 1.0)
        term = torch.where(affected, term, torch.zeros_like(term))
        n_affected = affected.sum(dim=-1).clamp_min(1).to(term.dtype)
        out[start:stop] = term.sum(dim=-1) / n_affected

    return out


def _quality_checks_fused(
    verts: torch.Tensor,
    faces: torch.Tensor,
    edges: torch.Tensor,
    opt: torch.Tensor,
    vert_to_faces: torch.Tensor,
    cos_threshold: float = 0.0,
    want_flip: bool = True,
    want_skinny: bool = True,
    want_link: bool = False,
    chunk_size: int = 100_000,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Fused 1-ring checks (flip count / skinny / link) sharing one faces gather.
    Returns (flip_count|None, skinny|None, link_safe|None)."""
    E = edges.shape[0]
    device = verts.device
    flip_out = torch.zeros(E, dtype=torch.int32, device=device) if want_flip else None
    skinny_out = torch.zeros(E, dtype=verts.dtype, device=device) if want_skinny else None
    link_out = torch.ones(E, dtype=torch.bool, device=device) if want_link else None
    if E == 0:
        return flip_out, skinny_out, link_out

    D = vert_to_faces.shape[1]
    a_all = edges[:, 0]
    b_all = edges[:, 1]
    sqrt3_4 = 4.0 * math.sqrt(3.0)
    need_geom = want_flip or want_skinny

    for start in range(0, E, chunk_size):
        stop = min(start + chunk_size, E)
        Ec = stop - start
        a = a_all[start:stop]
        b = b_all[start:stop]

        # shared gather of a's and b's incident faces (the expensive part)
        fa = vert_to_faces[a]
        fb = vert_to_faces[b]
        all_f = torch.cat([fa, fb], dim=1)                   # (Ec, 2D)
        valid_f = all_f >= 0
        fv = faces[all_f.clamp(min=0)]                       # (Ec, 2D, 3)
        a_b = a.view(Ec, 1)
        b_b = b.view(Ec, 1)

        if need_geom:
            oc = opt[start:stop]
            s0_a = fv[..., 0] == a_b
            s0_b = fv[..., 0] == b_b
            s1_a = fv[..., 1] == a_b
            s1_b = fv[..., 1] == b_b
            s2_a = fv[..., 2] == a_b
            s2_b = fv[..., 2] == b_b
            contains_a = s0_a | s1_a | s2_a
            contains_b = s0_b | s1_b | s2_b
            affected = (contains_a ^ contains_b) & valid_f
            if affected.any():
                p0 = verts[fv[..., 0]]
                p1 = verts[fv[..., 1]]
                p2 = verts[fv[..., 2]]
                opt_b = oc.view(Ec, 1, 3).expand(-1, 2 * D, -1)
                rep0 = (s0_a | s0_b).unsqueeze(-1)
                rep1 = (s1_a | s1_b).unsqueeze(-1)
                rep2 = (s2_a | s2_b).unsqueeze(-1)
                p0n = torch.where(rep0, opt_b, p0)
                p1n = torch.where(rep1, opt_b, p1)
                p2n = torch.where(rep2, opt_b, p2)

                # post-collapse normal (skinny's two_area == flip's ‖n_new‖)
                e01 = p1n - p0n
                e02 = p2n - p0n
                n_new = torch.cross(e01, e02, dim=-1)
                nlen_new = torch.norm(n_new, dim=-1)

                if want_flip:
                    n_old = torch.cross(p1 - p0, p2 - p0, dim=-1)
                    nlen_old = torch.norm(n_old, dim=-1)
                    denom = nlen_old * nlen_new
                    safe = denom > 1e-20
                    cos = torch.where(safe, (n_old * n_new).sum(dim=-1) / denom.clamp_min(1e-20),
                                      torch.ones_like(denom))
                    flip = (cos < cos_threshold) & affected & safe
                    flip_out[start:stop] = flip.sum(dim=-1).to(torch.int32)

                if want_skinny:
                    e12 = p2n - p1n
                    edge_sum_sq = ((e01 * e01).sum(-1) + (e02 * e02).sum(-1) + (e12 * e12).sum(-1))
                    shape = (sqrt3_4 * 0.5 * nlen_new) / edge_sum_sq.clamp_min(1e-20)
                    term = 1.0 - shape.clamp(0.0, 1.0)
                    term = torch.where(affected, term, torch.zeros_like(term))
                    n_affected = affected.sum(dim=-1).clamp_min(1).to(term.dtype)
                    skinny_out[start:stop] = term.sum(dim=-1) / n_affected

        if want_link:
            # reuses fv / valid_f; matches _link_condition_mask
            fa_ok = valid_f[:, :D]
            fb_ok = valid_f[:, D:]
            fav = fv[:, :D]
            fbv = fv[:, D:]
            an1 = torch.where(fav[..., 0] == a_b, fav[..., 1], fav[..., 0])
            an2 = torch.where(fav[..., 2] == a_b, fav[..., 1], fav[..., 2])
            bn1 = torch.where(fbv[..., 0] == b_b, fbv[..., 1], fbv[..., 0])
            bn2 = torch.where(fbv[..., 2] == b_b, fbv[..., 1], fbv[..., 2])
            na = torch.stack([an1, an2], dim=-1).reshape(Ec, 2 * D)
            nb = torch.stack([bn1, bn2], dim=-1).reshape(Ec, 2 * D)
            fa_okx = fa_ok.repeat_interleave(2, dim=1)
            fb_okx = fb_ok.repeat_interleave(2, dim=1)
            na[(na == a_b) | (na == b_b) | ~fa_okx] = -1
            nb[(nb == a_b) | (nb == b_b) | ~fb_okx] = -1
            in_b = (na[:, :, None] == nb[:, None, :]) & (na[:, :, None] >= 0)
            na_common = torch.where(in_b.any(dim=2), na, torch.full_like(na, -1))
            cs, _ = na_common.sort(dim=1)
            count_common = ((cs[:, 1:] != cs[:, :-1]) & (cs[:, 1:] >= 0)).sum(dim=1) \
                           + (cs[:, :1] >= 0).sum(dim=1)
            count_faces = ((fav == b[:, None, None]).any(dim=2) & fa_ok).sum(dim=1)
            link_out[start:stop] = count_common <= count_faces

    return flip_out, skinny_out, link_out


def _compute_vertex_normals(verts: torch.Tensor, faces: torch.Tensor, weld: bool = True) -> torch.Tensor:
    """Area-weighted smooth vertex normals. `weld` averages face normals across vertices that
    share a position (UV-seam duplicates from unwrapping) so both sides of a seam get one
    identical normal — otherwise a visible shading seam appears in the exported GLB."""
    if faces.numel() == 0:
        return torch.zeros_like(verts)
    faces_long = faces.to(torch.int64)
    i0, i1, i2 = faces_long[:, 0], faces_long[:, 1], faces_long[:, 2]
    v0, v1, v2 = verts[i0], verts[i1], verts[i2]
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)
    if weld and verts.shape[0]:
        # Group coincident positions (quantized to ~1e-5 of the bbox) into one shared normal.
        lo = verts.min(0).values
        inv_tol = 1.0 / (float((verts.max(0).values - lo).max().clamp_min(1e-9)) * 1e-5)
        q = ((verts - lo) * inv_tol).round().to(torch.int64)
        _, group = torch.unique(q, dim=0, return_inverse=True)
        acc = torch.zeros((int(group.max()) + 1, 3), dtype=verts.dtype, device=verts.device)
        acc.scatter_add_(0, group[i0].unsqueeze(-1).expand_as(fn), fn)
        acc.scatter_add_(0, group[i1].unsqueeze(-1).expand_as(fn), fn)
        acc.scatter_add_(0, group[i2].unsqueeze(-1).expand_as(fn), fn)
        vn = acc[group]
    else:
        vn = torch.zeros_like(verts)
        vn.scatter_add_(0, i0.unsqueeze(-1).expand_as(fn), fn)
        vn.scatter_add_(0, i1.unsqueeze(-1).expand_as(fn), fn)
        vn.scatter_add_(0, i2.unsqueeze(-1).expand_as(fn), fn)
    return torch.nn.functional.normalize(vn, p=2, dim=-1, eps=1e-6)


# Public API

@dataclass
class CleanStats:
    in_verts: int = 0
    in_faces: int = 0
    out_verts: int = 0
    out_faces: int = 0
    welded_verts: int = 0          # how many vertex IDs collapsed during welding
    degenerate_faces: int = 0      # zero-area or repeated-index faces removed
    duplicate_faces: int = 0       # same vertex-set removed
    unused_verts: int = 0          # verts not in any face removed
    components_dropped: int = 0    # disconnected components below threshold

    def __str__(self):
        return (f"clean: in={self.in_verts}v/{self.in_faces}f -> "
                f"out={self.out_verts}v/{self.out_faces}f "
                f"(welded {self.welded_verts}v, degen {self.degenerate_faces}f, "
                f"dup {self.duplicate_faces}f, unused {self.unused_verts}v, "
                f"comps {self.components_dropped})")


def _weld_vertices(
    verts: torch.Tensor, faces: torch.Tensor, epsilon,
    colors: Optional[torch.Tensor] = None,
    normals: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], int]:
    """Merge vertices closer than epsilon (L_inf grid), cluster-averaging attributes; returns (v, f, colors, normals, n_welded)."""
    if verts.shape[0] == 0:
        return verts, faces, colors, normals, 0
    device = verts.device
    scale = 1.0 / epsilon
    bbox_min = verts.min(dim=0)[0]
    q = ((verts - bbox_min) * scale).round().to(torch.int64)
    bbox = (verts.max(dim=0)[0] - bbox_min)
    extent = (bbox * scale).round().to(torch.int64) + 2
    key = (q[:, 0] * extent[1] + q[:, 1]) * extent[2] + q[:, 2]  # pack 3D quantized pos to 1D key
    unique_key, inv = torch.unique(key, return_inverse=True)
    n_unique = unique_key.shape[0]
    if n_unique == verts.shape[0]:
        return verts, faces, colors, normals, 0
    counts = torch.zeros(n_unique, dtype=verts.dtype, device=device)
    counts.scatter_add_(0, inv, torch.ones(verts.shape[0], dtype=verts.dtype, device=device))
    counts_div = counts.unsqueeze(-1).clamp_min(1.0)

    new_verts = torch.zeros((n_unique, 3), dtype=verts.dtype, device=device)
    new_verts.scatter_add_(0, inv.unsqueeze(-1).expand_as(verts), verts)
    new_verts = new_verts / counts_div

    new_colors = None
    if colors is not None:
        new_colors = torch.zeros((n_unique, colors.shape[1]), dtype=colors.dtype, device=device)
        new_colors.scatter_add_(0, inv.unsqueeze(-1).expand_as(colors), colors)
        new_colors = new_colors / counts_div.to(colors.dtype)

    new_normals = None
    if normals is not None:
        new_normals = torch.zeros((n_unique, normals.shape[1]), dtype=normals.dtype, device=device)
        new_normals.scatter_add_(0, inv.unsqueeze(-1).expand_as(normals), normals)
        new_normals = torch.nn.functional.normalize(new_normals, p=2, dim=-1, eps=1e-6)

    new_faces = inv[faces.long()] if faces.numel() > 0 else faces
    return new_verts, new_faces, new_colors, new_normals, int(verts.shape[0] - n_unique)


def _drop_degenerate_faces(
    verts: torch.Tensor, faces: torch.Tensor,
    min_area: float = 1e-14,
) -> Tuple[torch.Tensor, int]:
    """Drop degenerate-by-construction faces (repeated indices or zero-area); slivers go to _collapse_slivers."""
    if faces.numel() == 0:
        return faces, 0
    idx_bad = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 0] == faces[:, 2])
    f_good = faces[~idx_bad]
    v0 = verts[f_good[:, 0]]
    v1 = verts[f_good[:, 1]]
    v2 = verts[f_good[:, 2]]
    e0 = v1 - v0
    e2 = v0 - v2
    area = 0.5 * torch.norm(torch.cross(e0, -e2, dim=-1), dim=-1)
    bad = area < min_area
    kept = f_good[~bad]
    n_dropped = idx_bad.sum() + bad.sum()  # tensor-scalar; caller .item()s once
    return kept, n_dropped


def _collapse_slivers(
    verts: torch.Tensor, faces: torch.Tensor,
    min_angle_deg: float = 0.0,
    max_aspect_ratio: float = 0.0,
) -> Tuple[torch.Tensor, int]:
    """Resolve sliver triangles by collapsing each sliver's shortest edge (no holes); returns (faces, n_collapsed)."""
    if faces.numel() == 0 or (min_angle_deg <= 0 and max_aspect_ratio <= 0):
        return faces, 0

    fl = faces.long()
    v0 = verts[fl[:, 0]]
    v1 = verts[fl[:, 1]]
    v2 = verts[fl[:, 2]]
    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v0 - v2
    l0 = torch.norm(e0, dim=-1)
    l1 = torch.norm(e1, dim=-1)
    l2 = torch.norm(e2, dim=-1)
    area = 0.5 * torch.norm(torch.cross(e0, -e2, dim=-1), dim=-1)

    bad = torch.zeros(faces.shape[0], dtype=torch.bool, device=verts.device)
    if max_aspect_ratio > 0:
        max_edge = torch.maximum(torch.maximum(l0, l1), l2)
        aspect = max_edge * max_edge / (2.0 * area + 1e-12)
        bad = bad | (aspect > max_aspect_ratio)
    if min_angle_deg > 0:
        cos_a = (l1 * l1 + l2 * l2 - l0 * l0) / (2 * l1 * l2 + 1e-12)
        cos_b = (l0 * l0 + l2 * l2 - l1 * l1) / (2 * l0 * l2 + 1e-12)
        cos_c = (l0 * l0 + l1 * l1 - l2 * l2) / (2 * l0 * l1 + 1e-12)
        cos_all = torch.stack([cos_a, cos_b, cos_c], dim=-1)
        angles_deg = torch.acos(torch.clamp(cos_all, -1, 1)) * (180.0 / math.pi)
        bad = bad | (angles_deg.min(dim=-1).values < min_angle_deg)

    if not bad.any():
        return faces, 0

    # per sliver pick its shortest edge to collapse
    edge_lens = torch.stack([l0, l1, l2], dim=-1)  # (F, 3)
    shortest_slot = edge_lens.argmin(dim=-1)        # (F,) ∈ {0,1,2}

    V = verts.shape[0]
    # collapse higher-index endpoint into lower (min/max ordering avoids cycles)
    merge_map = torch.arange(V, device=verts.device, dtype=torch.int64)
    bad_idx = torch.nonzero(bad, as_tuple=True)[0]
    for slot in range(3):
        sel = bad_idx[shortest_slot[bad_idx] == slot]
        if sel.numel() == 0:
            continue
        a = fl[sel, slot]
        b = fl[sel, (slot + 1) % 3]
        lo = torch.minimum(a, b)
        hi = torch.maximum(a, b)
        merge_map[hi] = lo  # last-write-wins on conflict

    # path-compress until stable
    for _ in range(10):
        new_map = merge_map[merge_map]
        if torch.equal(new_map, merge_map):
            break
        merge_map = new_map

    new_faces = merge_map[fl]
    nondeg = ((new_faces[:, 0] != new_faces[:, 1]) &
              (new_faces[:, 1] != new_faces[:, 2]) &
              (new_faces[:, 0] != new_faces[:, 2]))
    new_faces = new_faces[nondeg].to(dtype=faces.dtype)
    return new_faces, bad.sum()


def _drop_duplicate_faces(faces: torch.Tensor, num_verts: int) -> Tuple[torch.Tensor, int]:
    """Remove duplicate faces (same vertex set), keeping the first occurrence (winding-preserving)."""
    if faces.shape[0] <= 1:
        return faces, 0
    key_sorted = torch.sort(faces, dim=1)[0]
    P = num_verts + 1
    packed = (key_sorted[:, 0].long() * P + key_sorted[:, 1].long()) * P + key_sorted[:, 2].long()
    unique_packed, inv = torch.unique(packed, return_inverse=True)
    if unique_packed.shape[0] == faces.shape[0]:
        return faces, 0
    # first-occurrence index per unique key
    arange = torch.arange(packed.shape[0], device=packed.device)
    first = torch.full((unique_packed.shape[0],), packed.shape[0],
                       dtype=torch.int64, device=packed.device)
    first.scatter_reduce_(0, inv, arange, reduce="amin", include_self=True)
    kept = faces[first]
    return kept, int(faces.shape[0] - kept.shape[0])


def _drop_unused_verts(
    verts: torch.Tensor, faces: torch.Tensor,
    colors: Optional[torch.Tensor] = None,
    normals: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], int]:
    """Remove vertices not referenced by any face; remap faces and filter attributes."""
    if verts.shape[0] == 0 or faces.numel() == 0:
        return verts, faces, colors, normals, 0
    used = torch.zeros(verts.shape[0], dtype=torch.bool, device=verts.device)
    used[faces[:, 0]] = True
    used[faces[:, 1]] = True
    used[faces[:, 2]] = True
    # cumsum compact remap: 0..N-1 to used verts in order
    remap = used.long().cumsum(0) - 1
    new_verts = verts[used]
    new_faces = remap[faces.long()]
    new_colors = colors[used] if colors is not None else None
    new_normals = normals[used] if normals is not None else None
    n_dropped = verts.shape[0] - used.sum()
    return new_verts, new_faces, new_colors, new_normals, n_dropped


def _repair_nonmanifold_edges(
    verts: torch.Tensor, faces: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """repair_non_manifold_edges: explode corners, re-merge only across manifold edges; returns (verts, faces, src)."""
    if faces.numel() == 0:
        return verts, faces
    dev, vdt, fdt = verts.device, verts.dtype, faces.dtype
    F = faces.detach().cpu().numpy().astype(_np.int64)
    V = verts.detach().cpu().numpy()
    nf = F.shape[0]
    nv = V.shape[0]
    corner_vert = F.reshape(-1)                      # (3F,) original vertex per corner

    # per-face edges keyed by (vmin,vmax)
    keys_l, ca_l, cb_l = [], [], []
    for (i, j) in ((0, 1), (1, 2), (2, 0)):
        va, vb = F[:, i], F[:, j]
        ci = 3 * _np.arange(nf) + i
        cj = 3 * _np.arange(nf) + j
        amin = _np.where(va <= vb, ci, cj)           # corner of the smaller-id endpoint
        amax = _np.where(va <= vb, cj, ci)
        vmin = _np.minimum(va, vb).astype(_np.int64)
        vmax = _np.maximum(va, vb).astype(_np.int64)
        keys_l.append(vmin * (nv + 1) + vmax)
        ca_l.append(amin)
        cb_l.append(amax)
    keys = _np.concatenate(keys_l)
    ca = _np.concatenate(ca_l)
    cb = _np.concatenate(cb_l)
    order = _np.argsort(keys, kind="stable")
    keys = keys[order]
    ca = ca[order]
    cb = cb[order]
    uniq, start, cnt = _np.unique(keys, return_index=True, return_counts=True)
    man = start[cnt == 2]                            # manifold edges (exactly 2 incident faces)
    # union both endpoints' corners across each manifold edge
    rows = _np.concatenate([ca[man], cb[man]])
    cols = _np.concatenate([ca[man + 1], cb[man + 1]])

    n = 3 * nf
    g = coo_matrix((_np.ones(rows.shape[0], dtype=_np.int8), (rows, cols)), shape=(n, n))
    _ncomp, labels = connected_components(g, directed=False)

    new_faces = labels[3 * _np.arange(nf)[:, None] + _np.array([0, 1, 2])[None, :]]
    nnv = int(labels.max()) + 1
    # source original-vertex index per new vertex
    src = _np.zeros(nnv, dtype=_np.int64)
    src[labels] = corner_vert
    new_verts = V[src]
    src_t = torch.from_numpy(src).to(device=dev)
    return (torch.from_numpy(new_verts).to(device=dev, dtype=vdt),
            torch.from_numpy(new_faces.astype(_np.int64)).to(device=dev, dtype=fdt),
            src_t)


def _drop_small_components(
    verts: torch.Tensor, faces: torch.Tensor, min_faces: int,
    max_propagation_iters: int = 200,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Label-propagation connected components; drop components below min_faces."""
    if faces.numel() == 0 or min_faces <= 1:
        return verts, faces, 0
    device = verts.device
    V = verts.shape[0]
    labels = torch.arange(V, device=device, dtype=torch.int64)
    for _ in range(max_propagation_iters):
        v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]
        face_min = torch.minimum(torch.minimum(labels[v0], labels[v1]), labels[v2])
        new_labels = labels.clone()
        new_labels.scatter_reduce_(0, v0, face_min, reduce="amin", include_self=True)
        new_labels.scatter_reduce_(0, v1, face_min, reduce="amin", include_self=True)
        new_labels.scatter_reduce_(0, v2, face_min, reduce="amin", include_self=True)
        new_labels = new_labels[new_labels]  # path-compress
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
    face_label = labels[faces[:, 0]]
    unique_labels, counts = torch.unique(face_label, return_counts=True)
    big_labels = unique_labels[counts >= min_faces]
    if big_labels.shape[0] == unique_labels.shape[0]:
        return verts, faces, 0
    # safety: never drop every component (return the small mesh, not an empty one)
    if big_labels.shape[0] == 0:
        return verts, faces, 0
    keep_face = torch.isin(face_label, big_labels)
    kept_faces = faces[keep_face]
    n_dropped = int(unique_labels.shape[0] - big_labels.shape[0])
    return verts, kept_faces, n_dropped


def clean_mesh(
    verts: torch.Tensor, faces: torch.Tensor,
    colors: Optional[torch.Tensor] = None,
    normals: Optional[torch.Tensor] = None,
    weld_epsilon: float = 0.0,
    weld_epsilon_rel: float = 1e-6,
    drop_degenerate: bool = True,
    drop_duplicates: bool = True,
    drop_unused: bool = True,
    min_component_faces: int = 0,
    min_angle_deg: float = 0.0,
    max_aspect_ratio: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], CleanStats]:
    """Mesh hygiene pipeline; preserves per-vertex attributes through welding. Returns (v, f, colors, normals, stats)."""
    stats = CleanStats(in_verts=verts.shape[0], in_faces=faces.shape[0])
    v = verts
    f = faces.long() if faces.numel() > 0 else faces
    c = colors
    n = normals

    if weld_epsilon != 0.0 or weld_epsilon_rel > 0:
        # eps stays a 0-d tensor (no sync)
        if weld_epsilon > 0:
            eps = torch.as_tensor(weld_epsilon, dtype=v.dtype, device=v.device)
        else:
            eps = torch.norm(v.max(dim=0)[0] - v.min(dim=0)[0]) * weld_epsilon_rel
        v, f, c, n, n_welded = _weld_vertices(v, f, eps, c, n)
        stats.welded_verts = n_welded

    if drop_degenerate:
        f_new, n_drop = _drop_degenerate_faces(v, f)
        stats.degenerate_faces = n_drop
        f = f_new
        # slivers get collapse-merged instead of dropped (preserves topology)
        if min_angle_deg > 0 or max_aspect_ratio > 0:
            f_new, n_sliv = _collapse_slivers(
                v, f, min_angle_deg=min_angle_deg, max_aspect_ratio=max_aspect_ratio,
            )
            stats.degenerate_faces += n_sliv
            f = f_new

    if drop_duplicates:
        f_new, n_dup = _drop_duplicate_faces(f, v.shape[0])
        stats.duplicate_faces = n_dup
        f = f_new

    if min_component_faces > 1:
        v, f, n_comp = _drop_small_components(v, f, min_component_faces)
        stats.components_dropped = n_comp

    if drop_unused:
        v, f, c, n, n_unused = _drop_unused_verts(v, f, c, n)
        stats.unused_verts = n_unused

    stats.out_verts = v.shape[0]
    stats.out_faces = f.shape[0]
    # materialize tensor-scalar counts to plain ints once at exit
    for field in ("welded_verts", "degenerate_faces", "duplicate_faces",
                  "unused_verts", "components_dropped"):
        val = getattr(stats, field)
        if torch.is_tensor(val):
            setattr(stats, field, int(val.item()))
    return v, f, c, n, stats


@dataclass
class SimplifyStats:
    input_verts: int = 0
    input_faces: int = 0
    output_verts: int = 0
    output_faces: int = 0
    iterations: int = 0
    total_collapses: int = 0


def qem_simplify(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    target_faces: int,
    colors: Optional[torch.Tensor] = None,
    normals: Optional[torch.Tensor] = None,
    max_edge_length: Optional[float] = None,
    config: Optional[QEMConfig] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], SimplifyStats]:
    """Single-mesh QEM simplification. Returns (v, f, colors, normals, stats)."""
    cfg = config or QEMConfig()

    device = vertices.device
    in_v_dtype = vertices.dtype
    in_f_dtype = faces.dtype
    in_c_dtype = colors.dtype if colors is not None else None
    in_n_dtype = normals.dtype if normals is not None else None

    verts = vertices.to(device=device, dtype=cfg.dtype, copy=True)
    faces = faces.to(device=device, dtype=torch.int64).clone()
    colors_w = colors.to(device=device, dtype=cfg.dtype, copy=True) if colors is not None else None
    normals_w = normals.to(device=device, dtype=cfg.dtype, copy=True) if normals is not None else None

    # preclean: weld + drop degenerate/duplicate, attributes cluster-averaged
    if cfg.preclean:
        verts, faces, colors_w, normals_w, _cs = clean_mesh(
            verts, faces, colors_w, normals_w,
            weld_epsilon_rel=cfg.preclean_weld_epsilon_rel,
            min_component_faces=cfg.preclean_min_component_faces,
        )

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    stats = SimplifyStats(input_verts=num_verts, input_faces=num_faces)

    if num_faces <= target_faces or num_verts < 4:
        stats.output_verts = num_verts
        stats.output_faces = num_faces
        return verts.to(in_v_dtype), faces.to(in_f_dtype), \
               (colors_w.to(in_c_dtype) if colors_w is not None else None), \
               (normals_w.to(in_n_dtype) if normals_w is not None else None), \
               stats

    v_alive = torch.ones(num_verts, dtype=torch.bool, device=device)
    f_alive = torch.ones(num_faces, dtype=torch.bool, device=device)

    Q = _build_quadrics(verts, faces, cfg)

    bbox = verts.max(dim=0)[0] - verts.min(dim=0)[0]
    mesh_scale = torch.norm(bbox)                  # 0-d tensor; never .item()'d
    if max_edge_length is None or max_edge_length <= 0:
        max_edge_length = mesh_scale * 2.0
    else:
        max_edge_length = torch.as_tensor(max_edge_length, dtype=cfg.dtype, device=device)
    # tiny-bbox guard (tensor-side, no sync)
    max_edge_length = torch.where(
        max_edge_length < 1e-6,
        torch.ones((), dtype=max_edge_length.dtype, device=device),
        max_edge_length,
    )

    stabilizer = mesh_scale * mesh_scale * cfg.stabilizer_scale
    max_edge_length_sq = max_edge_length * max_edge_length
    mesh_scale_sq = mesh_scale * mesh_scale

    # threshold scaled by mesh_scale² so the 1e-8 start is scale-robust
    thresh = float(cfg.threshold_start) * float(mesh_scale_sq) if cfg.threshold_driver else 0.0

    # pre-allocated merge_map, reused each iter
    merge_map = torch.arange(num_verts, device=device)

    # py_n_faces: Python-int face count (no host sync in hot loop), re-synced at compaction
    py_n_faces = num_faces

    iteration = 0
    total_collapses = 0

    # progress bars (tqdm + comfy ProgressBar)
    _start_faces = num_faces
    _prog_total = max(1, _start_faces - int(target_faces))
    _qtq = _tqdm(total=100, desc="QEM simplify", leave=False)
    _qpbar = _comfy_utils.ProgressBar(100)

    def _qreport():
        pct = min(100, max(0, int(100 * (_start_faces - py_n_faces) / _prog_total)))
        _qtq.n = pct
        _qtq.refresh()
        _qpbar.update_absolute(pct, 100)

    while True:
        if py_n_faces <= target_faces:
            break
        _qreport()

        alive_f = torch.nonzero(f_alive, as_tuple=True)[0]
        if alive_f.numel() == 0:
            break

        active_faces = faces[alive_f]

        # memoryless QEM: rebuild Q from current geometry each iter
        if cfg.threshold_driver and cfg.memoryless_qem and iteration > 0:
            Q = _build_quadrics(verts, active_faces, cfg)

        Q_for_iter = Q
        # edge extraction: pack (min*V + max) so unique dedups in one pass
        af_roll = torch.roll(active_faces, shifts=-1, dims=1)
        mn = torch.minimum(active_faces, af_roll)
        mx = torch.maximum(active_faces, af_roll)
        packed = torch.add(mx, mn, alpha=num_verts).flatten()
        packed = torch.unique(packed)
        edges_orig = torch.stack([packed // num_verts, packed % num_verts], dim=1)

        # filter by edge length
        pab = verts[edges_orig]                              # (E, 2, 3)
        el = torch.norm(pab[:, 1] - pab[:, 0], dim=-1)
        edges_orig = edges_orig[el < max_edge_length]
        if edges_orig.shape[0] == 0:
            break

        # sampling cap
        n_edges_total = edges_orig.shape[0]
        if n_edges_total > cfg.sampling_cap:
            perm = torch.randperm(n_edges_total, device=device)[: cfg.sampling_cap]
            edges_orig = edges_orig[perm]

        # boundary mask only needed for non-qem placement
        if cfg.placement_mode != "qem":
            vib = _vert_is_boundary_mask(active_faces, num_verts)
        else:
            vib = None
        optimal, err, valid = _edge_errors(
            verts, Q_for_iter, edges_orig, stabilizer, max_edge_length_sq,
            mesh_scale_sq, cfg, vert_is_boundary=vib,
        )
        valid_idx = torch.nonzero(valid, as_tuple=True)[0]
        edges_orig = edges_orig[valid_idx]
        optimal = optimal[valid_idx]
        err = err[valid_idx]

        faces_to_remove = py_n_faces - target_faces
        n_faces_round_start = py_n_faces
        # ~2 faces removed per collapse, so cap the round at faces_to_remove//2
        cap_to_target = max(1, faces_to_remove // 2)

        if cfg.threshold_driver:
            # band = cost <= thresh (×10 until non-empty), quality-check, then collapse a disjoint set
            cand = err <= thresh
            esc = 0
            while not bool(cand.any()) and esc < 50:
                thresh *= 10.0
                cand = err <= thresh
                esc += 1
            cand_idx = torch.nonzero(cand, as_tuple=True)[0]
            ce = edges_orig[cand_idx]
            copt = optimal[cand_idx]
            cerr = err[cand_idx].clone()
            need_flip = cfg.flip_reject_hard
            if ((need_flip or cfg.skinny_weight > 0 or cfg.enforce_link_condition)
                    and ce.shape[0] > 0):
                afq = faces[alive_f]
                v_to_f = _build_vert_to_faces_pad(afq, num_verts, cfg.flip_check_max_degree)
                # link + flip + skinny share one fused 1-ring pass
                fc, sk, link_safe = _quality_checks_fused(
                    verts, afq, ce, copt, v_to_f, cos_threshold=cfg.flip_cos_threshold,
                    want_flip=need_flip, want_skinny=(cfg.skinny_weight > 0),
                    want_link=cfg.enforce_link_condition)
                if link_safe is not None:
                    cerr[~link_safe] = float("inf")
                if fc is not None:
                    cerr = torch.where(fc > 0, torch.full_like(cerr, float("inf")), cerr)
                if sk is not None:
                    el_sq = (verts[ce[:, 1]] - verts[ce[:, 0]]).pow(2).sum(dim=-1)
                    cerr = cerr + cfg.skinny_weight * sk * el_sq
                del v_to_f, afq
                # penalties may push edges above thresh — re-gate the band
                keep = cerr <= thresh
                ce = ce[keep]
                copt = copt[keep]
                cerr = cerr[keep]
            edges_orig = ce
            optimal = copt
            sel = _greedy_matching(ce, cerr, v_alive, cap_to_target)
            if sel.numel() == 0:
                # band fully rejected → raise thresh and retry
                thresh *= 10.0
                iteration += 1
                if iteration >= cfg.max_iterations:
                    break
                continue
        else:
            max_collapses = min(
                cfg.max_collapses_ceiling,
                max(cfg.max_collapses_floor, int(faces_to_remove * cfg.max_collapses_fraction)),
            )
            if cfg.max_collapses_relative_cap > 0:
                # cap to a fraction of current mesh size (anti cascade-overshoot)
                rel_cap = max(1, int(py_n_faces * cfg.max_collapses_relative_cap))
                max_collapses = min(max_collapses, rel_cap)
            max_collapses = min(max_collapses, cap_to_target)

            # soft quality penalties on top-K: flip + skinny, sharing one v_to_f build
            need_flip = cfg.flip_reject_hard
            need_quality = ((need_flip or cfg.skinny_weight > 0 or cfg.enforce_link_condition)
                            and edges_orig.shape[0] > 0)
            if need_quality:
                n_check = min(edges_orig.shape[0],
                              max(1, cfg.quality_topk_multiplier * max_collapses))
                if n_check < edges_orig.shape[0]:
                    topk = torch.topk(err, n_check, largest=False).indices
                else:
                    topk = torch.arange(edges_orig.shape[0], device=device)
                active_for_quality = faces[alive_f]
                v_to_f = _build_vert_to_faces_pad(active_for_quality, num_verts,
                                                   cfg.flip_check_max_degree)
                err = err.clone()
                if cfg.enforce_link_condition:
                    # reject link-condition violations on ALL candidate edges, not just top-K
                    link_safe = _link_condition_mask(active_for_quality, edges_orig, v_to_f)
                    err[~link_safe] = float("inf")
                e_tk = edges_orig[topk]
                o_tk = optimal[topk]
                _do_flip = need_flip
                _do_skinny = cfg.skinny_weight > 0
                if _do_flip and _do_skinny:
                    flip_count, skinny, _ = _quality_checks_fused(
                        verts, active_for_quality, e_tk, o_tk, v_to_f,
                        cos_threshold=cfg.flip_cos_threshold, want_link=False)
                elif _do_flip:
                    flip_count = _normal_flip_mask(
                        verts, active_for_quality, e_tk, o_tk, v_to_f,
                        cos_threshold=cfg.flip_cos_threshold, return_count=True)
                    skinny = None
                else:
                    skinny = _skinny_penalty(verts, active_for_quality, e_tk, o_tk, v_to_f)
                    flip_count = None
                if _do_flip:
                    # hard reject: any flipping top-K edge → +inf
                    flips = flip_count > 0
                    if flips.any():
                        err[topk] = torch.where(
                            flips, torch.full_like(err[topk], float("inf")),
                            err[topk],
                        )
                if _do_skinny:
                    # skinny_cost * len² (match QEM's length² scaling)
                    elen_sq = (verts[e_tk[:, 1]] - verts[e_tk[:, 0]]).pow(2).sum(dim=-1)
                    err[topk] = torch.add(err[topk], skinny * elen_sq,
                                          alpha=cfg.skinny_weight)
                del v_to_f, active_for_quality

            sel = _greedy_matching(edges_orig, err, v_alive, max_collapses)

            if sel.numel() == 0:
                break

        ed_sel = edges_orig[sel]
        v_a = ed_sel[:, 0]
        v_b = ed_sel[:, 1]
        new_pos = optimal[sel]

        # interpolate attributes by new_pos's position along [pa, pb]
        if colors_w is not None or normals_w is not None:
            pa_sel = verts[v_a]
            pb_sel = verts[v_b]
            edge_vec = pb_sel - pa_sel
            edge_len_sq = (edge_vec * edge_vec).sum(dim=-1) + 1e-20
            t = ((new_pos - pa_sel) * edge_vec).sum(dim=-1) / edge_len_sq
            t = t.clamp(0.0, 1.0).unsqueeze(-1)
            if colors_w is not None:
                colors_w[v_a] = torch.lerp(colors_w[v_a], colors_w[v_b], t)
            if normals_w is not None:
                normals_w[v_a] = torch.lerp(normals_w[v_a], normals_w[v_b], t)

        # apply collapse
        verts[v_a] = new_pos
        v_alive[v_b] = False
        if not (cfg.threshold_driver and cfg.memoryless_qem):
            Q[v_a] += Q[v_b]

        merge_map[v_b] = v_a
        faces = merge_map[faces]
        merge_map[v_b] = v_b  # restore identity for next iter

        bad = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 2] == faces[:, 0])
        f_alive.masked_fill_(bad, False)
        py_n_faces -= 2 * v_a.numel()  # ~2 faces/collapse estimate; re-synced at compaction

        # schedule: round removed < 1% → raise thresh ×10
        if cfg.threshold_driver:
            removed = n_faces_round_start - py_n_faces
            if removed < 0.01 * n_faces_round_start:
                thresh *= 10.0

        total_collapses += int(v_a.numel())
        iteration += 1

        # periodic compaction (resyncs py_n_faces exactly)
        if iteration % cfg.compaction_period == 0:
            alive_frac = py_n_faces / max(1, num_faces)
            if alive_frac < cfg.compaction_threshold:
                faces = faces[f_alive]
                num_faces = faces.shape[0]
                f_alive = torch.ones(num_faces, dtype=torch.bool, device=device)
                py_n_faces = num_faces

        if iteration >= cfg.max_iterations:
            break

    _qreport()
    _qtq.close()

    # finalize: compact verts and faces
    final_v = verts[v_alive]
    final_c = colors_w[v_alive] if colors_w is not None else None
    final_n = normals_w[v_alive] if normals_w is not None else None

    remap = torch.full((num_verts,), -1, dtype=torch.int64, device=device)
    remap[v_alive] = v_alive.long().cumsum(0)[v_alive] - 1  # compact remap, no sync

    final_f_raw = faces[f_alive]
    alive_mask = v_alive[final_f_raw].all(dim=1)
    final_f_raw = final_f_raw[alive_mask]
    final_f = remap[final_f_raw]
    valid_faces = (final_f >= 0).all(dim=1)
    final_f = final_f[valid_faces]

    # drop degenerate faces (two indices equal)
    if final_f.numel() > 0:
        nondeg = (final_f[:, 0] != final_f[:, 1]) & (final_f[:, 1] != final_f[:, 2]) & (final_f[:, 0] != final_f[:, 2])
        final_f = final_f[nondeg]

    # dedup duplicate faces, winding-preserving
    if final_f.numel() > 0:
        key = torch.sort(final_f, dim=1)[0]
        packed = (key[:, 0].long() * (final_v.shape[0] + 1) + key[:, 1].long()) \
                 * (final_v.shape[0] + 1) + key[:, 2].long()
        unique_packed, inv = torch.unique(packed, return_inverse=True)
        arange = torch.arange(packed.shape[0], device=packed.device)
        first = torch.full((unique_packed.shape[0],), packed.shape[0],
                            dtype=torch.int64, device=packed.device)
        first.scatter_reduce_(0, inv, arange, reduce="amin", include_self=True)
        final_f = final_f[first]

    # split back fused surface sheets (after dedup, before pruning)
    if cfg.repair_nonmanifold and final_f.numel() > 0:
        final_v, final_f, _src = _repair_nonmanifold_edges(final_v, final_f)
        if final_c is not None:
            final_c = final_c[_src]
        if final_n is not None:
            final_n = final_n[_src]

    # post-clean: drop slivers, tiny components, unused verts
    if cfg.postclean and final_f.numel() > 0:
        comp_threshold = cfg.postclean_min_component_faces
        final_v, final_f, final_c, final_n, _ps = clean_mesh(
            final_v, final_f, final_c, final_n,
            weld_epsilon=0.0, weld_epsilon_rel=0.0,  # already welded
            drop_degenerate=True,
            drop_duplicates=False,                   # already done above
            drop_unused=True,
            min_component_faces=comp_threshold,
            min_angle_deg=cfg.postclean_min_angle_deg,
            max_aspect_ratio=cfg.postclean_max_aspect_ratio,
        )

    # post-simplify normals
    if cfg.recompute_normals_post and final_f.numel() > 0:
        final_n = _compute_vertex_normals(final_v, final_f)
    elif final_n is not None and final_f.numel() > 0:
        # keep supplied normals; flip face winding where it disagrees
        v0 = final_v[final_f[:, 0]]
        v1 = final_v[final_f[:, 1]]
        v2 = final_v[final_f[:, 2]]
        fn = torch.cross(v1 - v0, v2 - v0, dim=-1)
        ref = (final_n[final_f[:, 0]] + final_n[final_f[:, 1]]
               + final_n[final_f[:, 2]]) / 3.0
        wrong = (fn * ref).sum(dim=-1) < 0
        final_f[wrong] = final_f[wrong][:, [0, 2, 1]]

    stats.iterations = iteration
    stats.total_collapses = total_collapses
    stats.output_verts = final_v.shape[0]
    stats.output_faces = final_f.shape[0]

    return (
        final_v.to(in_v_dtype),
        final_f.to(in_f_dtype),
        final_c.to(in_c_dtype) if final_c is not None else None,
        final_n.to(in_n_dtype) if (final_n is not None and in_n_dtype is not None) else final_n,
        stats,
    )


def qem_decimate_simplify(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    target: int,
    colors: Optional[torch.Tensor] = None,
    normals: Optional[torch.Tensor] = None,
    max_edge_length: Optional[float] = None,
    config: Optional[QEMConfig] = None,
):
    """Batched wrapper. Accepts (V,3)/(F,3) or (B,V,3)/(B,F,3)."""
    if vertices.ndim == 3:
        out_v, out_f, out_c, out_n, out_s = [], [], [], [], []
        for i in range(vertices.shape[0]):
            c_in = colors[i] if colors is not None else None
            n_in = normals[i] if normals is not None else None
            v, f, c, n, s = qem_simplify(vertices[i], faces[i], target, c_in, n_in, max_edge_length, config)
            out_v.append(v)
            out_f.append(f)
            out_s.append(s)
            if c is not None:
                out_c.append(c)
            if n is not None:
                out_n.append(n)
        return (out_v, out_f,
                out_c if out_c else None,
                out_n if out_n else None,
                out_s)
    return qem_simplify(vertices, faces, target, colors, normals, max_edge_length, config)


def qem_cluster_decimate(
    vertices: torch.Tensor, faces: torch.Tensor,
    target_verts: int = 1_000_000,
    colors: Optional[torch.Tensor] = None,
    face_chunk: int = 4_000_000,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Vertex-cluster decimation (Rossignac-Borrel): grid-bin/average verts, remap faces,
    drop degenerate/duplicate. Fast O(V+F) prepass for huge meshes. Returns (verts, faces, colors)."""
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        return vertices, faces, colors

    device = vertices.device
    bbox = vertices.max(dim=0)[0] - vertices.min(dim=0)[0]
    bbox_min = vertices.min(dim=0)[0]
    # cell size so the bbox holds ~3× target_verts cells (surface occupancy ~1/3)
    cell_count_target = max(target_verts * 3, 1000)
    extent_max = float(bbox.max().item())
    cells_per_axis = (cell_count_target ** (1 / 3))
    cell_size = extent_max / max(1.0, cells_per_axis)
    scale = 1.0 / max(cell_size, 1e-20)

    q = ((vertices - bbox_min) * scale).floor().to(torch.int64)
    extent = (bbox * scale).floor().to(torch.int64) + 2
    Wy = extent[1]
    Wz = extent[2]
    key = (q[:, 0] * Wy + q[:, 1]) * Wz + q[:, 2]

    unique_key, inv = torch.unique(key, return_inverse=True)
    n_unique = unique_key.shape[0]
    counts = torch.zeros(n_unique, dtype=vertices.dtype, device=device)
    counts.scatter_add_(0, inv, torch.ones(vertices.shape[0], dtype=vertices.dtype, device=device))
    counts_div = counts.unsqueeze(-1).clamp_min(1.0)

    new_verts = torch.zeros((n_unique, 3), dtype=vertices.dtype, device=device)
    new_verts.scatter_add_(0, inv.unsqueeze(-1).expand_as(vertices), vertices)
    new_verts = new_verts / counts_div

    new_colors = None
    if colors is not None:
        new_colors = torch.zeros((n_unique, colors.shape[1]), dtype=colors.dtype, device=device)
        new_colors.scatter_add_(0, inv.unsqueeze(-1).expand_as(colors), colors)
        new_colors = new_colors / counts_div.to(colors.dtype)

    # remap faces in chunks (face tensor can be huge), drop degenerates per chunk
    out_chunks = []
    F = faces.shape[0]
    for fs in range(0, F, face_chunk):
        fe = min(fs + face_chunk, F)
        cf = inv[faces[fs:fe].long()]
        nondeg = ((cf[:, 0] != cf[:, 1]) & (cf[:, 1] != cf[:, 2]) & (cf[:, 0] != cf[:, 2]))
        if nondeg.any():
            out_chunks.append(cf[nondeg])
    if out_chunks:
        new_faces = torch.cat(out_chunks, dim=0)
    else:
        new_faces = torch.empty((0, 3), dtype=faces.dtype, device=device)

    # drop duplicate faces (same vertex set after clustering)
    if new_faces.numel() > 0:
        key_sorted = torch.sort(new_faces, dim=1)[0]
        P = n_unique + 1
        packed = (key_sorted[:, 0].long() * P + key_sorted[:, 1].long()) * P + key_sorted[:, 2].long()
        _, first = torch.unique(packed, return_inverse=True)
        arange = torch.arange(packed.shape[0], device=device, dtype=torch.int64)
        first_idx = torch.full((int(first.max().item()) + 1,), packed.shape[0],
                               dtype=torch.int64, device=device)
        first_idx.scatter_reduce_(0, first, arange, reduce="amin", include_self=True)
        new_faces = new_faces[first_idx]

    return new_verts.to(vertices.dtype), new_faces.to(faces.dtype), new_colors
