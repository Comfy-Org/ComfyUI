"""Adaptive cost-grow chart segmentation (vectorized torch, CPU or GPU)."""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor
from tqdm import tqdm

from .mesh import MeshData, face_edge_lengths


DEFAULT_W_NORMAL_DEVIATION = 2.0
DEFAULT_W_ROUNDNESS = 0.01
DEFAULT_W_STRAIGHTNESS = 6.0
DEFAULT_MAX_COST = 2.0
NORMAL_DEVIATION_HARD_CUTOFF = 0.707  # ~75°


def _grow_iter(face_chart, frontier, ff, fn, fa, fel, basis, nsum, area, perim, K,
               nd_cutoff, tau, w_nd, w_round, w_straight):
    """One grow pass: each frontier face joins its lowest-cost adjacent chart if cost <= tau;
    returns the number of faces assigned."""
    u = frontier.nonzero(as_tuple=True)[0]
    if u.numel() == 0:
        return 0
    nb = ff[u]                                       # (U,3) neighbor face ids
    nbc = torch.where(nb >= 0, face_chart[nb.clamp_min(0)], nb.new_full((), -1))
    valid = nbc >= 0
    d = (fn[u][:, None, :] * basis[nbc.clamp_min(0)]).sum(-1)
    nd = (1.0 - d).clamp(0.0, 1.0)
    valid &= nd < nd_cutoff
    el = fel[u]                                      # (U,3)
    # l_in per candidate chart j: edge k counts if its (assigned) neighbor is in chart j
    inm = (nbc[:, :, None] == nbc[:, None, :]) & valid[:, None, :]
    l_in = (el[:, None, :] * inm).sum(-1)            # (U,3)
    tot = el.sum(-1, keepdim=True)
    l_out = tot - l_in
    ca = area[nbc.clamp_min(0)]
    cp = perim[nbc.clamp_min(0)]
    new_perim = cp - l_in + l_out
    new_r = new_perim * new_perim / (ca + fa[u][:, None]).clamp_min(1e-20)
    round_cost = torch.where((cp <= 1e-20) | (ca <= 1e-20) | (new_r <= 1e-20),
                             torch.zeros_like(new_r),
                             1.0 - (cp * cp / ca.clamp_min(1e-20)) / new_r.clamp_min(1e-20))
    straight_cost = ((l_out - l_in) / tot.clamp_min(1e-20)).clamp(max=0.0)
    cost = w_nd * nd + w_round * round_cost + w_straight * straight_cost
    cost = torch.where(valid, cost, cost.new_full((), float("inf")))
    best_cost, best_j = cost.min(1)
    acc = best_cost <= tau
    n_acc = int(acc.sum())
    if n_acc == 0:
        return 0

    f_acc = u[acc]
    c_acc = nbc.gather(1, best_j[:, None]).squeeze(1)[acc]
    nbc_old = nbc[acc]                               # neighbor charts before this commit
    face_chart[f_acc] = c_acc
    nb_acc = nb[acc]
    nbs_acc = nb_acc.clamp_min(0)
    nbc_post = torch.where(nb_acc >= 0, face_chart[nbs_acc], nb_acc.new_full((), -1))
    # frontier update: committed faces leave; their still-unassigned neighbors enter
    frontier[f_acc] = False
    grow_nb = nbs_acc[(nb_acc >= 0) & (nbc_post < 0)]
    frontier[grow_nb] = True
    el_acc = el[acc]
    cx = c_acc[:, None]
    dper = torch.where(nbc_old == cx, -el_acc,       # was member: edge turns interior
                       torch.where(nbc_post == cx, torch.zeros_like(el_acc),  # co-committer
                                   el_acc)).sum(1)   # boundary / other chart
    perim.scatter_add_(0, c_acc, dper)
    area.scatter_add_(0, c_acc, fa[f_acc])
    nsum.index_add_(0, c_acc, fn[f_acc] * fa[f_acc, None])
    nl = nsum[:K].norm(dim=1, keepdim=True)
    basis[:K] = torch.where(nl > 1e-20, nsum[:K] / nl.clamp_min(1e-20), basis[:K])
    return n_acc


def segment_charts(
    mesh: MeshData,
    max_cost: float = DEFAULT_MAX_COST,
    w_normal_deviation: float = DEFAULT_W_NORMAL_DEVIATION,
    w_roundness: float = DEFAULT_W_ROUNDNESS,
    w_straightness: float = DEFAULT_W_STRAIGHTNESS,
    progress_callback=None,
) -> Tensor:
    """Segment mesh into charts (parallel batch cost-grow). Returns face -> chart_id."""
    F = mesh.faces.shape[0]
    device = mesh.faces.device
    if F == 0:
        return torch.zeros(0, dtype=torch.long, device=device)

    fn = mesh.face_normal.detach().to(torch.float32)
    fa = mesh.face_area.detach().to(torch.float32)
    fc = mesh.face_centroid.detach().to(torch.float32)
    ff = mesh.face_face.detach().long()
    fel = face_edge_lengths(mesh.vertices, mesh.faces).detach().to(torch.float32)
    nd_cutoff = NORMAL_DEVIATION_HARD_CUTOFF

    # one seed per connected component (first face of each)
    comp = mesh.component.detach().long().to(device)
    ncomp = int(comp.max()) + 1 if comp.numel() else 0
    if ncomp:
        seeds = torch.full((ncomp,), F, dtype=torch.long, device=device)
        seeds.scatter_reduce_(0, comp, torch.arange(F, device=device), reduce="amin")
    else:
        seeds = torch.zeros(1, dtype=torch.long, device=device)
    K = seeds.shape[0]

    max_total_charts = max(F, 8000)
    cap = K + F + 1                                  # every re-seed assigns a face, so K < K0 + F
    face_chart = torch.full((F,), -1, dtype=torch.long, device=device)
    basis = torch.zeros(cap, 3, dtype=torch.float32, device=device)
    nsum = torch.zeros(cap, 3, dtype=torch.float32, device=device)
    area = torch.zeros(cap, dtype=torch.float32, device=device)
    perim = torch.zeros(cap, dtype=torch.float32, device=device)
    face_chart[seeds] = torch.arange(K, device=device)
    basis[:K] = fn[seeds]
    nsum[:K] = fn[seeds] * fa[seeds, None]
    area[:K] = fa[seeds]
    perim[:K] = fel[seeds].sum(1)
    frontier = torch.zeros(F, dtype=torch.bool, device=device)
    seed_nb = ff[seeds]
    seed_nb = seed_nb[seed_nb >= 0]
    frontier[seed_nb] = True
    frontier &= face_chart < 0

    min_d2 = torch.full((F,), float("inf"), dtype=torch.float32, device=device)
    for i in range(0, K, 32):                        # chunked: (F, <=32, 3) stays small
        d2 = ((fc[:, None, :] - fc[seeds[i:i + 32]][None, :, :]) ** 2).sum(-1)
        min_d2 = torch.minimum(min_d2, d2.amin(1))

    # Multi-pass threshold schedule (low-cost first); tau cap 0.5 keeps cones ~30deg.
    tau_final = min(max_cost * 0.25, 0.5)
    thresholds = [t for t in (0.05, 0.1, 0.25) if t < tau_final] + [tau_final]
    max_inner = max(64, int(F ** 0.5) * 2)
    outer_iter = 0
    assigned = 0
    tq = tqdm(total=F, desc="unwrap: segment (adaptive)", unit="face", leave=False)
    while True:
        outer_iter += 1
        if outer_iter > F + 16:
            break
        for tau in thresholds:
            for _ in range(max_inner):
                n_added = _grow_iter(face_chart, frontier, ff, fn, fa, fel, basis, nsum,
                                     area, perim, K, nd_cutoff, tau, w_normal_deviation,
                                     w_roundness, w_straightness)
                if n_added == 0:
                    break
                tq.update(n_added)
                assigned += n_added
                if progress_callback is not None:
                    progress_callback(assigned, F)
        unassigned = face_chart < 0
        if int(unassigned.sum()) == 0:
            break
        if K >= max_total_charts:
            break
        # re-seed at the unassigned face farthest from every existing seed
        new_seed = int(torch.where(unassigned, min_d2,
                                   min_d2.new_full((), float("-inf"))).argmax())
        face_chart[new_seed] = K
        basis[K] = fn[new_seed]
        nsum[K] = fn[new_seed] * fa[new_seed]
        area[K] = fa[new_seed]
        perim[K] = fel[new_seed].sum()
        K += 1
        min_d2 = torch.minimum(min_d2, ((fc - fc[new_seed]) ** 2).sum(-1))
        tq.update(1)
        frontier[new_seed] = False
        ns_nb = ff[new_seed]
        ns_nb = ns_nb[ns_nb >= 0]
        frontier[ns_nb[face_chart[ns_nb] < 0]] = True

    tq.close()

    # Orphan cleanup: leftover faces join their best-matching neighbor's chart.
    while True:
        orphans = (face_chart < 0).nonzero(as_tuple=True)[0]
        if orphans.numel() == 0:
            break
        nb = ff[orphans]
        nbc = torch.where(nb >= 0, face_chart[nb.clamp_min(0)], nb.new_full((), -1))
        valid = nbc >= 0
        assignable = valid.any(1)
        if not bool(assignable.any()):
            break
        d = (fn[orphans][:, None, :] * basis[nbc.clamp_min(0)]).sum(-1)
        ndv = torch.where(valid, 1.0 - d, d.new_full((), float("inf")))
        best_c = nbc.gather(1, ndv.argmin(1, keepdim=True)).squeeze(1)
        face_chart[orphans[assignable]] = best_c[assignable]
    leftover = (face_chart < 0).nonzero(as_tuple=True)[0]
    if leftover.numel():                             # isolated faces become singleton charts
        face_chart[leftover] = K + torch.arange(leftover.numel(), device=device)

    _, inverse = torch.unique(face_chart, sorted=True, return_inverse=True)
    return inverse


# Parallel edge-collapse (PEC) chart clustering (GPU)
def _combine_normal_cones(
    axis_a: Tensor, half_a: Tensor,
    axis_b: Tensor, half_b: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Merge two normal cones along the great circle from axis_a; returns (combined_axis, combined_half_angle, axis_angle)."""
    cos_angle = (axis_a * axis_b).sum(dim=-1).clamp(-1.0, 1.0)
    axis_angle = torch.acos(cos_angle)
    new_low = torch.minimum(-half_a, axis_angle - half_b)
    new_high = torch.maximum(half_a, axis_angle + half_b)
    new_half = (new_high - new_low) * 0.5
    rot_angle = (new_high + new_low) * 0.5
    b_perp = axis_b - axis_a * cos_angle.unsqueeze(-1)
    b_perp_norm = b_perp.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    b_perp_unit = b_perp / b_perp_norm
    new_axis = (
        axis_a * torch.cos(rot_angle).unsqueeze(-1)
        + b_perp_unit * torch.sin(rot_angle).unsqueeze(-1)
    )
    new_axis_norm = new_axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    new_axis = new_axis / new_axis_norm
    return new_axis, new_half, axis_angle


def _build_chart_edges(
    face_face: Tensor,
    chart_id: Tensor,
    face_edge_len: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Build chart-edge list (chart_pairs[E,2] with a<b, edge_length[E]); same-chart edges dropped, duplicates summed."""
    F = face_face.shape[0]
    device = face_face.device
    f_idx = torch.arange(F, device=device).repeat_interleave(3)
    nb = face_face.flatten()
    valid = nb >= 0
    f_idx = f_idx[valid]
    nb = nb[valid]
    el = face_edge_len.flatten()[valid]

    ca = chart_id[f_idx]
    cb = chart_id[nb]
    diff = ca != cb
    ca = ca[diff]
    cb = cb[diff]
    el = el[diff]
    if ca.numel() == 0:
        return (
            torch.empty((0, 2), dtype=torch.long, device=device),
            torch.empty(0, device=device),
        )

    lo = torch.minimum(ca, cb)
    hi = torch.maximum(ca, cb)
    V = int(chart_id.max().item()) + 1
    key = lo * V + hi
    sort_idx = torch.argsort(key)
    sorted_key = key[sort_idx]
    sorted_lo = lo[sort_idx]
    sorted_hi = hi[sort_idx]
    sorted_el = el[sort_idx]
    unique_key, inverse, counts = torch.unique(
        sorted_key, return_inverse=True, return_counts=True
    )
    n_unique = unique_key.shape[0]
    reduced_el = torch.zeros(n_unique, device=device, dtype=el.dtype)
    reduced_el.scatter_add_(0, inverse, sorted_el)
    first_idx = torch.cat([
        torch.zeros(1, dtype=torch.long, device=device),
        counts.cumsum(0)[:-1],
    ])
    pair_lo = sorted_lo[first_idx]
    pair_hi = sorted_hi[first_idx]
    chart_pairs = torch.stack([pair_lo, pair_hi], dim=1)
    return chart_pairs, reduced_el


def _merge_small_charts(
    chart_id: Tensor, face_normal: Tensor, face_area: Tensor,
    face_face: Tensor, face_edge_len: Tensor,
    min_faces: int, cost_cap: float,
) -> Tensor:
    """Absorb charts under min_faces faces into their lowest-cone-cost neighbor (capped at cost_cap)."""
    if chart_id.numel() == 0:
        return chart_id
    device = chart_id.device
    for _ in range(16):
        N = int(chart_id.max().item()) + 1
        sizes = torch.bincount(chart_id, minlength=N)
        # recompute cones from scratch: area-weighted mean axis, max deviation as half-angle
        axis = torch.zeros(N, 3, dtype=torch.float32, device=device)
        axis.index_add_(0, chart_id, face_normal * face_area[:, None])
        axis = axis / axis.norm(dim=1, keepdim=True).clamp_min(1e-12)
        dev = torch.acos((face_normal * axis[chart_id]).sum(1).clamp(-1.0, 1.0))
        half = torch.zeros(N, dtype=torch.float32, device=device)
        half.scatter_reduce_(0, chart_id, dev, reduce="amax")

        edges, _ = _build_chart_edges(face_face, chart_id, face_edge_len)
        if edges.shape[0] == 0:
            break
        a, b = edges[:, 0], edges[:, 1]
        _, new_half, _ = _combine_normal_cones(axis[a], half[a], axis[b], half[b])
        ok = new_half <= cost_cap
        E = edges.shape[0]
        key = (torch.clamp(new_half * 1e6, max=2e9).to(torch.int64) << 32) \
            | torch.arange(E, dtype=torch.long, device=device)
        best = torch.full((N,), 1 << 62, dtype=torch.long, device=device)
        va = (sizes[a] < min_faces) & ok
        vb = (sizes[b] < min_faces) & ok
        best.scatter_reduce_(0, a[va], key[va], reduce="amin")
        best.scatter_reduce_(0, b[vb], key[vb], reduce="amin")
        src = (best < (1 << 62)).nonzero(as_tuple=True)[0]
        if src.numel() == 0:
            break
        eid = best[src] & 0xFFFFFFFF
        ea, eb = a[eid], b[eid]
        tgt = torch.where(ea == src, eb, ea)
        # cycle break: keep src->tgt only if tgt merges nowhere itself or src > tgt;
        # the kept graph is then a DAG, so the pointer-doubling below terminates
        prop = torch.arange(N, dtype=torch.long, device=device)
        prop[src] = tgt
        keepm = (prop[tgt] == tgt) | (src > tgt)
        remap = torch.arange(N, dtype=torch.long, device=device)
        remap[src[keepm]] = tgt[keepm]
        for _ in range(32):
            nr = remap[remap]
            if torch.equal(nr, remap):
                break
            remap = nr
        chart_id = remap[chart_id]
        _, chart_id = torch.unique(chart_id, return_inverse=True)
    return chart_id


def cluster_charts_pec(
    mesh: MeshData,
    max_cost: float = 0.7,
    max_iters: int = 1024,
    min_faces: int = 8,
    progress_callback=None,
) -> Tensor:
    """Parallel edge-collapse clustering; returns face_chart [F]. max_cost is the per-merge
    cutoff (~0.7 rad ~ 40deg); charts under min_faces are then absorbed at a relaxed 2x cutoff."""
    device = mesh.faces.device
    F = mesh.faces.shape[0]
    faces = mesh.faces.to(torch.long)
    vertices = mesh.vertices.to(torch.float32)
    face_normal = mesh.face_normal.to(torch.float32)
    face_face = mesh.face_face.to(torch.long)

    face_edge_len = face_edge_lengths(vertices, faces)

    chart_id = torch.arange(F, dtype=torch.long, device=device)
    chart_axis = face_normal.clone()
    chart_half = torch.zeros(F, dtype=torch.float32, device=device)

    for it in range(max_iters):
        edges, _ = _build_chart_edges(face_face, chart_id, face_edge_len)
        if edges.shape[0] == 0:
            break

        a = edges[:, 0]
        b = edges[:, 1]
        axis_a = chart_axis[a]
        axis_b = chart_axis[b]
        half_a = chart_half[a]
        half_b = chart_half[b]
        _, new_half, _ = _combine_normal_cones(axis_a, half_a, axis_b, half_b)
        cost = new_half.clone()

        # Pack (cost, edge_id) so scatter_reduce amin picks the right edge.
        E = edges.shape[0]
        N = int(chart_id.max().item()) + 1
        edge_ids = torch.arange(E, dtype=torch.long, device=device)
        cost_i32 = torch.clamp(cost * 1e6, max=2e9).to(torch.int64)
        key = (cost_i32 << 32) | edge_ids
        chart_min = torch.full((N,), (2**62), dtype=torch.long, device=device)
        chart_min.scatter_reduce_(0, a, key, reduce="amin", include_self=True)
        chart_min.scatter_reduce_(0, b, key, reduce="amin", include_self=True)

        # Mutual-min collapse: each chart in at most one merge per iter (winners are disjoint pairs).
        is_a_min = chart_min[a] == key
        is_b_min = chart_min[b] == key
        mutual = is_a_min & is_b_min
        within = cost <= max_cost
        winners = mutual & within

        n_merge = int(winners.sum().item())
        if n_merge == 0:
            break
        if progress_callback is not None:
            progress_callback(F - N + n_merge, F)      # saturating: charts remaining vs faces

        win_a = a[winners]
        win_b = b[winners]

        axis_a_w = chart_axis[win_a]
        half_a_w = chart_half[win_a]
        axis_b_w = chart_axis[win_b]
        half_b_w = chart_half[win_b]
        new_axis, new_half_w, _ = _combine_normal_cones(
            axis_a_w, half_a_w, axis_b_w, half_b_w,
        )
        chart_axis[win_a] = new_axis
        chart_half[win_a] = new_half_w

        remap = torch.arange(N, dtype=torch.long, device=device)
        remap[win_b] = win_a
        chart_id = remap[chart_id]

    if min_faces > 1:
        chart_id = _merge_small_charts(chart_id, face_normal, mesh.face_area.to(torch.float32),
                                       face_face, face_edge_len, min_faces, 2.0 * max_cost)
    _, inverse = torch.unique(chart_id, sorted=True, return_inverse=True)
    return inverse
