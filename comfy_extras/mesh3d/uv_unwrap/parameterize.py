"""Chart parameterization: ortho PCA projection, falling back to ABF/LSCM."""
from __future__ import annotations

import warnings
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from torch import Tensor

import comfy.model_management
from . import mesh as _mesh

LSCM_BATCH_MAX_VERTS = 256      # charts above this solve per-chart sparse (lscm_chart)


def _lscm_solve_on_gpu(device: "torch.device | None") -> bool:
    """True if the dense batched normal-equations solve should run on device rather
    than CPU. Excludes ROCm: hipBLAS's batched getrf (torch.linalg.solve's backing
    call) raises HIPBLAS_STATUS_ALLOC_FAILED on these chart batches even with plenty
    of free VRAM, and PyTorch reports a ROCm device's .type as "cuda", so the type
    check alone can't tell AMD and NVIDIA apart."""
    return device is not None and device.type == "cuda" and not comfy.model_management.is_amd()


def solve_least_squares(A: sp.csr_matrix, b: np.ndarray) -> np.ndarray:
    """Solve ||Ax - b||^2 by factorizing AtA."""
    At = A.T.tocsr()
    AtA = (At @ A).tocsc()
    Atb = At @ b
    return spla.spsolve(AtA, Atb)


def _triangle_local_2d(verts_3d: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-triangle 2D coords [F, 3, 2] with v0 at origin, v1 along +x."""
    v0 = verts_3d[faces[:, 0]]
    v1 = verts_3d[faces[:, 1]]
    v2 = verts_3d[faces[:, 2]]
    e01 = v1 - v0
    e02 = v2 - v0
    L01 = np.linalg.norm(e01, axis=1).clip(min=1e-20)
    x_axis = e01 / L01[:, None]
    n = np.cross(e01, e02)
    n /= np.linalg.norm(n, axis=1, keepdims=True).clip(min=1e-20)
    y_axis = np.cross(n, x_axis)

    out = np.zeros((faces.shape[0], 3, 2), dtype=np.float64)
    out[:, 1, 0] = L01
    out[:, 2, 0] = (e02 * x_axis).sum(axis=1)
    out[:, 2, 1] = (e02 * y_axis).sum(axis=1)
    return out


def _pick_pins(loops: List[List[int]], verts_3d: np.ndarray) -> Tuple[int, int]:
    """Pick the longest-diameter axis-extremal boundary vertex pair across all boundary verts."""
    if not loops:
        # Closed surface: two far verts via two-pass farthest.
        d2 = np.sum((verts_3d - verts_3d[0]) ** 2, axis=1)
        a = int(np.argmax(d2))
        d2 = np.sum((verts_3d - verts_3d[a]) ** 2, axis=1)
        b = int(np.argmax(d2))
        return a, b
    boundary_verts: List[int] = []
    for loop in loops:
        boundary_verts.extend(loop)
    seen = set()
    uniq = []
    for v in boundary_verts:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    bv = np.asarray(uniq, dtype=np.int64)
    pts = verts_3d[bv]
    pin_pairs = []
    for axis in range(3):
        i_min = int(bv[int(np.argmin(pts[:, axis]))])
        i_max = int(bv[int(np.argmax(pts[:, axis]))])
        d = float(np.linalg.norm(verts_3d[i_min] - verts_3d[i_max]))
        pin_pairs.append((d, i_min, i_max))
    d0, _, _ = pin_pairs[0]
    d1, _, _ = pin_pairs[1]
    d2, _, _ = pin_pairs[2]
    if d0 > d1 and d0 > d2:
        _, a, b = pin_pairs[0]
    elif d1 > d2:
        _, a, b = pin_pairs[1]
    else:
        _, a, b = pin_pairs[2]
    return a, b


def _ortho_project(verts_3d: np.ndarray) -> np.ndarray:
    """PCA-fit plane normal, axis-aligned tangent, project verts to 2D."""
    centroid = verts_3d.mean(axis=0)
    pts = verts_3d - centroid
    cov = pts.T @ pts
    _w, ev = np.linalg.eigh(cov)
    normal = ev[:, 0]
    a = np.abs(normal)
    if a[0] < a[1] and a[0] < a[2]:
        t = np.array([1.0, 0.0, 0.0])
    elif a[1] < a[2]:
        t = np.array([0.0, 1.0, 0.0])
    else:
        t = np.array([0.0, 0.0, 1.0])
    t = t - normal * float(np.dot(normal, t))
    t /= max(float(np.linalg.norm(t)), 1e-20)
    b = np.cross(normal, t)
    return np.stack([verts_3d @ t, verts_3d @ b], axis=1)


def ortho_project_concat(verts: np.ndarray, chart_of_vert: np.ndarray, n_charts: int) -> np.ndarray:
    """_ortho_project for every chart at once over concatenated per-chart vertices."""
    cnt = np.bincount(chart_of_vert, minlength=n_charts).clip(min=1).astype(np.float64)
    cen = np.stack([np.bincount(chart_of_vert, weights=verts[:, i], minlength=n_charts)
                    for i in range(3)], axis=1) / cnt[:, None]
    d = verts - cen[chart_of_vert]
    cov = np.zeros((n_charts, 3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(i, 3):
            s = np.bincount(chart_of_vert, weights=d[:, i] * d[:, j], minlength=n_charts)
            cov[:, i, j] = s
            cov[:, j, i] = s
    _w, ev = np.linalg.eigh(cov)
    normal = ev[:, :, 0]
    t = np.eye(3, dtype=np.float64)[np.argmin(np.abs(normal), axis=1)]
    t = t - normal * (normal * t).sum(axis=1, keepdims=True)
    t /= np.linalg.norm(t, axis=1, keepdims=True).clip(min=1e-20)
    b = np.cross(normal, t)
    tt, bb = t[chart_of_vert], b[chart_of_vert]
    return np.stack([(verts * tt).sum(1), (verts * bb).sum(1)], axis=1)


def stretch_metrics_concat(
    verts: np.ndarray, uvs: np.ndarray, faces: np.ndarray,
    chart_of_face: np.ndarray, n_charts: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-chart Sander stretch metrics (rms, max, n_flipped, n_zero_area); rms/max inf where undefined."""
    p = verts[faces]
    t = uvs[faces]
    pa_signed = 0.5 * (
        (t[:, 1, 1] - t[:, 0, 1]) * (t[:, 2, 0] - t[:, 0, 0])
        - (t[:, 2, 1] - t[:, 0, 1]) * (t[:, 1, 0] - t[:, 0, 0]))
    n_flip = np.bincount(chart_of_face[pa_signed < -1e-12], minlength=n_charts)
    n_zero = np.bincount(chart_of_face[np.abs(pa_signed) < 1e-12], minlength=n_charts)
    pa = np.abs(pa_signed).clip(min=1e-20)
    ga = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    keep = (ga > 1e-12) & (np.abs(pa_signed) > 1e-12)
    t1, s1 = t[:, 0, 0], t[:, 0, 1]
    t2, s2 = t[:, 1, 0], t[:, 1, 1]
    t3, s3 = t[:, 2, 0], t[:, 2, 1]
    inv_2pa = 1.0 / (2.0 * pa)
    Ss = (p[:, 0] * (t2 - t3)[:, None] + p[:, 1] * (t3 - t1)[:, None]
          + p[:, 2] * (t1 - t2)[:, None]) * inv_2pa[:, None]
    St = (p[:, 0] * (s3 - s2)[:, None] + p[:, 1] * (s1 - s3)[:, None]
          + p[:, 2] * (s2 - s1)[:, None]) * inv_2pa[:, None]
    a = (Ss * Ss).sum(axis=1)
    bb = (Ss * St).sum(axis=1)
    c = (St * St).sum(axis=1)
    sigma2_sq = 0.5 * (a + c + np.sqrt(np.maximum(0.0, (a - c) ** 2 + 4 * bb ** 2)))
    rms_sq = (a + c) * 0.5
    cf = chart_of_face[keep]
    tg = np.bincount(cf, weights=ga[keep], minlength=n_charts)
    tp = np.bincount(cf, weights=pa[keep], minlength=n_charts)
    rs = np.bincount(cf, weights=(rms_sq * ga)[keep], minlength=n_charts)
    smax = np.zeros(n_charts, dtype=np.float64)
    np.maximum.at(smax, cf, sigma2_sq[keep])
    ok = tg > 0.0
    tg_safe = np.where(ok, tg, 1.0)
    norm = np.sqrt(tp / tg_safe)
    rms = np.where(ok, np.sqrt(rs / tg_safe) * norm, np.inf)
    mx = np.where(ok, np.sqrt(smax) * norm, np.inf)
    return rms, mx, n_flip, n_zero


def _segment_argmax(vals: np.ndarray, seg: np.ndarray, n: int) -> np.ndarray:
    """Index of the (first) max element per segment; -1 for empty segments."""
    amax = np.full(n, -np.inf)
    np.maximum.at(amax, seg, vals)
    hit = vals == amax[seg]
    out = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(out, seg[hit], np.nonzero(hit)[0])
    return np.where(out == np.iinfo(np.int64).max, -1, out)


def lscm_charts_batch(
    verts: np.ndarray,            # (sumV, 3) float64, per-chart concatenated
    uv_pins: np.ndarray,          # (sumV, 2) float64, ortho UVs (pin values + fallback)
    faces_gl: np.ndarray,         # (sumF, 3) global-local ids into verts
    face_pos: np.ndarray,         # (sumF,) row index of each face within its chart
    chart_of_face: np.ndarray,    # (sumF,)
    chart_of_vert: np.ndarray,    # (sumV,)
    vert_offsets: np.ndarray,     # (n_charts+1,)
    chart_ids: np.ndarray,        # charts to solve (each with >=3 verts, >=1 face)
    n_charts: int,
    max_bucket_verts: int = LSCM_BATCH_MAX_VERTS,
    device: "torch.device | None" = None,
) -> dict:
    """Batched dense ABF/LSCM; returns {chart_id: (Vc, 2) float32}. Charts larger than
    max_bucket_verts are left out (the caller solves those sparse)."""
    out: dict = {}
    if chart_ids.size == 0:
        return out
    sel = np.zeros(n_charts, dtype=bool)
    sel[chart_ids] = True
    vcounts = np.diff(vert_offsets)

    # ABF coefficients for all selected faces in one shot
    fmask = sel[chart_of_face]
    f_ids = np.nonzero(fmask)[0]
    abf_ids, abf_cos, abf_sin, abf_valid = _abf_face_coefficients(verts, faces_gl[f_ids])

    # farthest-point pin pair per chart (two passes)
    vmask = sel[chart_of_vert]
    v_ids = np.nonzero(vmask)[0]
    cv = chart_of_vert[v_ids]
    first = vert_offsets[:-1]
    d0 = ((verts[v_ids] - verts[first[cv]]) ** 2).sum(1)
    pin_a = _segment_argmax(d0, cv, n_charts)          # global vert index (into v_ids space)
    pin_a = np.where(pin_a >= 0, v_ids[pin_a.clip(min=0)], -1)
    d1 = ((verts[v_ids] - verts[pin_a.clip(min=0)[cv]]) ** 2).sum(1)
    pin_b = _segment_argmax(d1, cv, n_charts)
    pin_b = np.where(pin_b >= 0, v_ids[pin_b.clip(min=0)], -1)
    # degenerate (all verts coincide): any distinct vert within the chart (Vc >= 3 guaranteed)
    alt = np.where(pin_a == first, first + 1, first)
    pin_b = np.where(pin_a == pin_b, alt, pin_b)

    fcounts = np.bincount(chart_of_face[f_ids], minlength=n_charts)
    # size-sorted chunks padded to their own max, bounded by an element budget so one
    # face-heavy chart can't inflate a whole chunk
    small = chart_ids[vcounts[chart_ids] <= max_bucket_verts]
    sorted_ids = small[np.argsort(vcounts[small], kind="stable")]
    budget = (96 << 20) // 8                            # float64 elements in a chunk's A
    chunks = []
    cs = 0
    fmax_r = vmax_r = 0
    for idx in range(sorted_ids.size):
        c2 = sorted_ids[idx]
        fm2 = max(fmax_r, int(fcounts[c2]))
        vm2 = max(vmax_r, int(vcounts[c2]))
        nb = idx - cs + 1
        if nb > 1 and (nb > 128 or nb * 4 * fm2 * vm2 > budget):
            chunks.append((cs, idx))
            cs = idx
            fmax_r, vmax_r = int(fcounts[c2]), int(vcounts[c2])
        else:
            fmax_r, vmax_r = fm2, vm2
    if sorted_ids.size:
        chunks.append((cs, sorted_ids.size))
    for s, e in chunks:
        cids = sorted_ids[s:e]
        B = cids.size
        Vmax = int(vcounts[cids].max())
        Fmax = int(fcounts[cids].max())
        N = 2 * Vmax
        R = 2 * Fmax
        compact = np.full(n_charts, -1, dtype=np.int64)
        compact[cids] = np.arange(B)
        fm = compact[chart_of_face[f_ids]] >= 0
        fi = f_ids[fm]                               # face rows for this chunk
        bi = compact[chart_of_face[fi]]              # chart slot per face
        frow = face_pos[fi]
        v0 = vert_offsets[chart_of_face[fi]]         # local id = global-local - v0
        am = fm.nonzero()[0]                         # index into abf_* arrays

        pieces_i: list = []
        pieces_v: list = []

        def scatter(rows, cols, vals, bsel):
            pieces_i.append((bsel * R + rows) * N + cols)
            pieces_v.append(vals)

        val = abf_valid[am]
        ii = am[val]
        ids = abf_ids[ii] - v0[val, None]            # local vert ids, reordered
        cosf, sinf = abf_cos[ii], abf_sin[ii]
        rr, bsel = frow[val] * 2, bi[val]
        ones = np.ones(ii.size)
        for cc2, vv in ((ids[:, 0], cosf - 1.0), (ids[:, 0] + Vmax, -sinf),
                        (ids[:, 1], -cosf), (ids[:, 1] + Vmax, sinf), (ids[:, 2], ones)):
            scatter(rr, cc2, vv, bsel)
        for cc2, vv in ((ids[:, 0], sinf), (ids[:, 0] + Vmax, cosf - 1.0),
                        (ids[:, 1], -sinf), (ids[:, 1] + Vmax, -cosf), (ids[:, 2] + Vmax, ones)):
            scatter(rr + 1, cc2, vv, bsel)

        inv = ~val
        if inv.any():
            jj = fi[inv]
            tri2d = _triangle_local_2d(verts, faces_gl[jj])
            twice = tri2d[:, 1, 0] * tri2d[:, 2, 1] - tri2d[:, 1, 1] * tri2d[:, 2, 0]
            w = 1.0 / np.sqrt(2.0 * np.abs(twice).clip(min=1e-20))
            rr2, bs2 = frow[inv] * 2, bi[inv]
            lids = faces_gl[jj] - v0[inv, None]
            for j in range(3):
                jp1, jp2 = (j + 1) % 3, (j + 2) % 3
                aj = (tri2d[:, jp1, 0] - tri2d[:, jp2, 0]) * w
                bj = (tri2d[:, jp1, 1] - tri2d[:, jp2, 1]) * w
                vc2 = lids[:, j]
                scatter(rr2, vc2, aj, bs2)
                scatter(rr2, vc2 + Vmax, -bj, bs2)
                scatter(rr2 + 1, vc2, bj, bs2)
                scatter(rr2 + 1, vc2 + Vmax, aj, bs2)

        flat = np.concatenate(pieces_i)
        A = np.bincount(flat, weights=np.concatenate(pieces_v),
                        minlength=B * R * N).reshape(B, R, N)

        # pins: move their columns to the RHS, then constrain via identity rows
        voff = vert_offsets[cids]
        pa_l = pin_a[cids] - voff
        pb_l = pin_b[cids] - voff
        pin_cols = np.stack([pa_l, pb_l, pa_l + Vmax, pb_l + Vmax], 1)       # (B,4)
        pin_vals = np.stack([uv_pins[pin_a[cids], 0], uv_pins[pin_b[cids], 0],
                             uv_pins[pin_a[cids], 1], uv_pins[pin_b[cids], 1]], 1)
        rhs = np.zeros((B, R), dtype=np.float64)
        barange = np.arange(B)
        for k in range(4):
            rhs -= A[barange, :, pin_cols[:, k]] * pin_vals[:, k, None]
            A[barange, :, pin_cols[:, k]] = 0.0

        # constrained columns: the 4 pins + padding beyond each chart's vert count
        vcs = vcounts[cids]
        padm = np.arange(Vmax)[None, :] >= vcs[:, None]
        con = np.concatenate([padm, padm], axis=1)                            # (B,N)
        np.put_along_axis(con, pin_cols, True, axis=1)
        cval = np.zeros((B, N), dtype=np.float64)
        np.put_along_axis(cval, pin_cols, pin_vals, axis=1)

        # normal equations + batched solve; the fp64 dense algebra goes to the GPU when available
        use_gpu = _lscm_solve_on_gpu(device)
        if use_gpu:
            A_t = torch.from_numpy(A).to(device)
            At = A_t.transpose(1, 2)
            AtA = At @ A_t
            Atb = (At @ torch.from_numpy(rhs).to(device).unsqueeze(2)).squeeze(2)
            con_t = torch.from_numpy(con).to(device)
            free2 = (~con_t[:, :, None]) & (~con_t[:, None, :])
            AtA = AtA * free2
            diag = torch.diagonal(AtA, dim1=1, dim2=2)
            # median (not max) positive diagonal: a degenerate face's ~1e19 squared row
            # weight would blow a max-scaled eps past the unit ABF rows
            dpos = torch.where(diag > 0, diag, torch.full_like(diag, float("nan")))
            dsc = 1e-12 * torch.nan_to_num(dpos.nanmedian(dim=1).values, nan=1e-8).clamp_min(1e-20)
            diag += torch.where(con_t, torch.ones_like(diag), dsc[:, None].expand_as(diag))
            Atb = torch.where(con_t, torch.from_numpy(cval).to(device), Atb)
            x = torch.linalg.solve(AtA, Atb).cpu().numpy()
        else:
            At = A.transpose(0, 2, 1)
            AtA = At @ A                             # batched BLAS dgemm
            Atb = (At @ rhs[:, :, None])[:, :, 0]
            AtA *= (~con[:, :, None]) & (~con[:, None, :])
            dg = AtA.reshape(B, -1)[:, ::N + 1]
            # median positive diagonal (see GPU branch): robust to degenerate-face weights
            dpos = np.where(dg > 0, dg, np.nan)
            with np.errstate(all="ignore"):
                dsc = 1e-12 * np.nan_to_num(np.nanmedian(dpos, axis=1), nan=1e-8).clip(min=1e-20)
            dg += np.where(con, 1.0, dsc[:, None])
            Atb2 = np.where(con, cval, Atb)
            x = np.linalg.solve(AtA, Atb2[..., None])[..., 0]
        for i2, c2 in enumerate(cids):
            vc3 = int(vcs[i2])
            out[int(c2)] = np.stack([x[i2, :vc3], x[i2, Vmax:Vmax + vc3]], 1).astype(np.float32)
    return out


def _uv_boundary_self_intersects(
    uvs: np.ndarray, faces: np.ndarray, face_face: np.ndarray, eps: float = 1e-9
) -> bool:
    """True if any chart-boundary edge pair crosses in 2D (ortho folded the chart)."""
    fi, ei = np.nonzero(face_face < 0)
    n = fi.size
    if n < 2:
        return False
    a = uvs[faces[fi, ei]].astype(np.float64)
    b = uvs[faces[fi, (ei + 1) % 3]].astype(np.float64)
    d = b - a
    # Pairwise segment crossings, row-chunked to bound memory at chunk*n.
    chunk = max(1, min(n, 1_000_000 // max(n, 1)))
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d1 = d[s:e, None, :]
        denom = d1[:, :, 0] * d[None, :, 1] - d1[:, :, 1] * d[None, :, 0]
        rx = a[None, :, 0] - a[s:e, None, 0]
        ry = a[None, :, 1] - a[s:e, None, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (rx * d[None, :, 1] - ry * d[None, :, 0]) / denom
            u = (rx * d1[:, :, 1] - ry * d1[:, :, 0]) / denom
        cross = (
            (np.abs(denom) >= eps)
            & (t > eps) & (t < 1.0 - eps)
            & (u > eps) & (u < 1.0 - eps)
        )
        if bool(cross.any()):
            return True
    return False


def _abf_face_coefficients(
    verts_3d: np.ndarray, faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-face ABF constraint (largest-sine vertex at local index 2); returns (faces_reordered, cosine, sine, valid_mask) with valid_mask False for degenerate tris."""
    p0 = verts_3d[faces[:, 0]]
    p1 = verts_3d[faces[:, 1]]
    p2 = verts_3d[faces[:, 2]]
    e01 = p1 - p0
    e12 = p2 - p1
    e20 = p0 - p2
    L01 = np.linalg.norm(e01, axis=1).clip(min=1e-20)
    L12 = np.linalg.norm(e12, axis=1).clip(min=1e-20)
    L20 = np.linalg.norm(e20, axis=1).clip(min=1e-20)
    cos_a0 = ((-e20) * e01).sum(axis=1) / (L20 * L01)
    cos_a1 = ((-e01) * e12).sum(axis=1) / (L01 * L12)
    cos_a2 = ((-e12) * e20).sum(axis=1) / (L12 * L20)
    cos_a0 = cos_a0.clip(-1.0, 1.0)
    cos_a1 = cos_a1.clip(-1.0, 1.0)
    cos_a2 = cos_a2.clip(-1.0, 1.0)
    a = np.arccos(cos_a0)
    b_ang = np.arccos(cos_a1)
    c_ang = np.arccos(cos_a2)
    angles = np.stack([a, b_ang, c_ang], axis=1)
    sines = np.stack([np.sin(a), np.sin(b_ang), np.sin(c_ang)], axis=1)
    valid = (angles > 1e-12).all(axis=1)
    ids = faces.astype(np.int64).copy()

    s0, s1, s2 = sines[:, 0], sines[:, 1], sines[:, 2]
    pattA = (s1 > s0) & (s1 > s2)
    pattB = (~pattA) & (s0 > s1) & (s0 > s2)

    if pattA.any():
        old_a = angles[pattA].copy()
        old_s = sines[pattA].copy()
        old_id = ids[pattA].copy()
        angles[pattA] = old_a[:, [2, 0, 1]]
        sines[pattA] = old_s[:, [2, 0, 1]]
        ids[pattA] = old_id[:, [2, 0, 1]]
    if pattB.any():
        old_a = angles[pattB].copy()
        old_s = sines[pattB].copy()
        old_id = ids[pattB].copy()
        angles[pattB] = old_a[:, [1, 2, 0]]
        sines[pattB] = old_s[:, [1, 2, 0]]
        ids[pattB] = old_id[:, [1, 2, 0]]

    a0 = angles[:, 0]
    s0 = sines[:, 0]
    s1 = sines[:, 1]
    s2 = sines[:, 2]
    c0 = np.cos(a0)
    ratio = np.where(s2 > 0.0, s1 / s2.clip(min=1e-20), 1.0)
    cosine = c0 * ratio
    sine = s0 * ratio
    return ids, cosine, sine, valid


def lscm_chart(
    local_verts: Tensor,
    local_faces: Tensor,
    local_face_face: Tensor,
    pin_positions: "np.ndarray | None" = None,
) -> Tensor:
    """ABF parameterization on one chart (degenerate faces use plain LSCM rows; two pins fix gauge at pin_positions)."""
    verts_np = local_verts.detach().cpu().numpy().astype(np.float64)
    faces_np = local_faces.detach().cpu().numpy().astype(np.int64)
    Vc = verts_np.shape[0]
    Fc = faces_np.shape[0]

    if Vc < 3 or Fc == 0:
        return torch.zeros((Vc, 2), dtype=torch.float32, device=local_verts.device)

    loops = _mesh.chart_boundary_loops(local_faces, local_face_face)
    pin_a, pin_b = _pick_pins(loops, verts_np)

    if pin_positions is not None and pin_positions.shape == (Vc, 2):
        pa = pin_positions[pin_a]
        pb = pin_positions[pin_b]
        u_a, v_a = float(pa[0]), float(pa[1])
        u_b, v_b = float(pb[0]), float(pb[1])
    else:
        u_a, v_a = 0.0, 0.0
        u_b, v_b = 1.0, 0.0

    abf_ids, abf_cos, abf_sin, abf_valid = _abf_face_coefficients(verts_np, faces_np)

    rows_list: List[np.ndarray] = []
    cols_list: List[np.ndarray] = []
    vals_list: List[np.ndarray] = []

    # ABF rows for valid faces.
    valid_idx = np.nonzero(abf_valid)[0]
    if valid_idx.size:
        Nv = valid_idx.size
        id0 = abf_ids[valid_idx, 0]
        id1 = abf_ids[valid_idx, 1]
        id2 = abf_ids[valid_idx, 2]
        cosf = abf_cos[valid_idx]
        sinf = abf_sin[valid_idx]
        r_real = valid_idx * 2
        r_imag = valid_idx * 2 + 1
        ones = np.ones(Nv, dtype=np.float64)
        rows_list.extend([r_real] * 5)
        cols_list.extend([id0, id0 + Vc, id1, id1 + Vc, id2])
        vals_list.extend([cosf - 1.0, -sinf, -cosf, sinf, ones])
        rows_list.extend([r_imag] * 5)
        cols_list.extend([id0, id0 + Vc, id1, id1 + Vc, id2 + Vc])
        vals_list.extend([sinf, cosf - 1.0, -sinf, -cosf, ones])

    # Plain-LSCM rows for invalid (degenerate) faces.
    invalid_idx = np.nonzero(~abf_valid)[0]
    if invalid_idx.size:
        tri2d_inv = _triangle_local_2d(verts_np, faces_np[invalid_idx])
        twice_area_inv = (
            tri2d_inv[:, 1, 0] * tri2d_inv[:, 2, 1]
            - tri2d_inv[:, 1, 1] * tri2d_inv[:, 2, 0]
        )
        weight_inv = 1.0 / np.sqrt(2.0 * np.abs(twice_area_inv).clip(min=1e-20))
        r_real_inv = invalid_idx * 2
        r_imag_inv = invalid_idx * 2 + 1
        for j in range(3):
            jp1 = (j + 1) % 3
            jp2 = (j + 2) % 3
            a_j = (tri2d_inv[:, jp1, 0] - tri2d_inv[:, jp2, 0]) * weight_inv
            b_j = (tri2d_inv[:, jp1, 1] - tri2d_inv[:, jp2, 1]) * weight_inv
            v_idx = faces_np[invalid_idx, j]
            rows_list.extend([r_real_inv, r_real_inv, r_imag_inv, r_imag_inv])
            cols_list.extend([v_idx, v_idx + Vc, v_idx, v_idx + Vc])
            vals_list.extend([a_j, -b_j, b_j, a_j])

    rows = np.concatenate(rows_list) if rows_list else np.empty(0, dtype=np.int64)
    cols = np.concatenate(cols_list) if cols_list else np.empty(0, dtype=np.int64)
    vals = np.concatenate(vals_list) if vals_list else np.empty(0, dtype=np.float64)

    A_full = sp.csr_matrix((vals, (rows, cols)), shape=(2 * Fc, 2 * Vc))

    pin_cols = np.array([pin_a, pin_b, pin_a + Vc, pin_b + Vc], dtype=np.int64)
    pin_vals = np.array([u_a, u_b, v_a, v_b], dtype=np.float64)

    free_mask = np.ones(2 * Vc, dtype=bool)
    free_mask[pin_cols] = False
    free_cols = np.nonzero(free_mask)[0]

    A_pinned = A_full[:, pin_cols]
    A_free = A_full[:, free_cols]
    b = -(A_pinned @ pin_vals)

    # Singular system (under-constrained chart) falls back to ortho.
    fallback_to_ortho = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=sp.linalg.MatrixRankWarning)
            x_free = solve_least_squares(A_free, b)
        if not np.all(np.isfinite(x_free)):
            fallback_to_ortho = True
    except (sp.linalg.MatrixRankWarning, RuntimeError):
        fallback_to_ortho = True  # singular / under-constrained system

    if fallback_to_ortho:
        if pin_positions is not None and pin_positions.shape == (Vc, 2):
            uvs = pin_positions.astype(np.float32)
        else:
            uvs = _ortho_project(verts_np).astype(np.float32)
        return torch.from_numpy(uvs).to(local_verts.device)

    full = np.zeros(2 * Vc, dtype=np.float64)
    full[free_cols] = x_free
    full[pin_cols] = pin_vals
    uvs = np.stack([full[:Vc], full[Vc:]], axis=1).astype(np.float32)
    if not np.all(np.isfinite(uvs)):
        if pin_positions is not None and pin_positions.shape == (Vc, 2):
            uvs = pin_positions.astype(np.float32)
        else:
            uvs = _ortho_project(verts_np).astype(np.float32)

    return torch.from_numpy(uvs).to(local_verts.device)
