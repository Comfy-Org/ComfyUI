"""Atlas packing via bitmap rasterize-and-place."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import max_pool1d

import comfy.model_management

# Numba is optional, but ~5x faster than torch on these operations, potential TODO: comfy-kitchen cuda/triton kernels as even faster alternative
try:
    from numba import njit as _njit, prange as _prange, get_num_threads as _nb_threads
    _HAVE_NUMBA_PACK = True
except ImportError:
    _HAVE_NUMBA_PACK = False
    _prange = range
    def _nb_threads(): return 1
    def _njit(*args, **kwargs):
        def deco(fn): return fn
        return deco if not args else args[0]


# Cap on deterministic sweep density: tiny charts on a large atlas would otherwise enumerate every texel column.
_SWEEP_CAP = 1024
_TORCH_PREP_MAX_ANGLE_ELEMENTS = 1 << 23
_TORCH_RASTER_FACE_BATCH = 1 << 17


@dataclass
class ChartPlacement:
    chart_id: int
    offset: Tuple[float, float]       # in texels
    scale: float                      # texels per UV unit
    rotation: float = 0.0             # radians
    swap_xy: bool = False             # extra 90° bitmap rotation chosen at place time
    chart_h: float = 0.0              # unswapped bitmap height in texels (rotation pivot)


@_njit(cache=True, boundscheck=False, parallel=True)
def _prepare_dims_jit(uvs, uv_off, a3, auv, tpu, padding, theta, scale, bw, bh, rot_uv):
    """Pass 1: per-chart best rotation, texel scale, rotated/scaled UVs, padded bitmap dims."""
    n = uv_off.shape[0] - 1
    half_pi = math.pi * 0.5
    for c in _prange(n):
        v0, v1 = uv_off[c], uv_off[c + 1]
        best_area = 1e30
        best_t = 0.0
        for k in range(36):
            th = half_pi * k / 36.0
            co = math.cos(th)
            si = math.sin(th)
            xmin = 1e30
            xmax = -1e30
            ymin = 1e30
            ymax = -1e30
            for i in range(v0, v1):
                xr = uvs[i, 0] * co - uvs[i, 1] * si
                yr = uvs[i, 0] * si + uvs[i, 1] * co
                if xr < xmin:
                    xmin = xr
                if xr > xmax:
                    xmax = xr
                if yr < ymin:
                    ymin = yr
                if yr > ymax:
                    ymax = yr
            area = (xmax - xmin) * (ymax - ymin)
            if area < best_area:
                best_area = area
                best_t = th
        theta[c] = best_t
        co = math.cos(best_t)
        si = math.sin(best_t)
        xmin = 1e30
        xmax = -1e30
        ymin = 1e30
        ymax = -1e30
        for i in range(v0, v1):
            xr = uvs[i, 0] * co - uvs[i, 1] * si
            yr = uvs[i, 0] * si + uvs[i, 1] * co
            rot_uv[i, 0] = xr
            rot_uv[i, 1] = yr
            if xr < xmin:
                xmin = xr
            if xr > xmax:
                xmax = xr
            if yr < ymin:
                ymin = yr
            if yr > ymax:
                ymax = yr
        if v1 == v0:
            xmin = 0.0
            xmax = 0.0
            ymin = 0.0
            ymax = 0.0
        s = math.sqrt(max(a3[c], 1e-12) / max(auv[c], 1e-12)) * tpu
        nominal = math.sqrt(max(a3[c], 1e-12)) * tpu
        max_bbox = max(8.0, 4.0 * nominal)
        bbox_max = max(max(xmax - xmin, ymax - ymin), 1e-12)
        if s * bbox_max > max_bbox:
            s = max_bbox / bbox_max
        scale[c] = s
        wmax = 0.0
        hmax = 0.0
        for i in range(v0, v1):
            rot_uv[i, 0] = (rot_uv[i, 0] - xmin) * s
            rot_uv[i, 1] = (rot_uv[i, 1] - ymin) * s
            if rot_uv[i, 0] > wmax:
                wmax = rot_uv[i, 0]
            if rot_uv[i, 1] > hmax:
                hmax = rot_uv[i, 1]
        bw[c] = int(math.ceil(wmax)) + padding + 1
        bh[c] = int(math.ceil(hmax)) + padding + 1


@_njit(cache=True, boundscheck=False, parallel=True)
def _raster_all_jit(rot_uv, uv_off, faces, f_off, bw, bh, boff, buf, padding,
                    tw, th_out, perim):
    """Pass 2: rasterize + dilate each chart into the flat buffer; records trimmed dims
    (origin kept) and the perimeter used for placement ordering."""
    n = uv_off.shape[0] - 1
    eps = 1e-7
    for c in _prange(n):
        f0, f1 = f_off[c], f_off[c + 1]
        v0 = uv_off[c]
        V = uv_off[c + 1] - v0
        w = bw[c]
        h = bh[c]
        o = boff[c]
        for fi in range(f0, f1):
            i0 = faces[fi, 0] + v0
            i1 = faces[fi, 1] + v0
            i2 = faces[fi, 2] + v0
            x0 = rot_uv[i0, 0]
            y0 = rot_uv[i0, 1]
            x1 = rot_uv[i1, 0]
            y1 = rot_uv[i1, 1]
            x2 = rot_uv[i2, 0]
            y2 = rot_uv[i2, 1]
            xmin_f = min(x0, min(x1, x2))
            xmax_f = max(x0, max(x1, x2))
            ymin_f = min(y0, min(y1, y2))
            ymax_f = max(y0, max(y1, y2))
            xmin = max(int(math.floor(xmin_f)), 0)
            xmax = min(int(math.ceil(xmax_f)), w - 1)
            ymin = max(int(math.floor(ymin_f)), 0)
            ymax = min(int(math.ceil(ymax_f)), h - 1)
            if xmax < xmin or ymax < ymin:
                continue
            denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denom) < 1e-20:
                continue
            inv_denom = 1.0 / denom
            for py in range(ymin, ymax + 1):
                yc = py + 0.5
                for px in range(xmin, xmax + 1):
                    xc = px + 0.5
                    aa = ((y1 - y2) * (xc - x2) + (x2 - x1) * (yc - y2)) * inv_denom
                    bb = ((y2 - y0) * (xc - x2) + (x0 - x2) * (yc - y2)) * inv_denom
                    cc = 1.0 - aa - bb
                    if aa >= -eps and bb >= -eps and cc >= -eps:
                        buf[o + py * w + px] = True
        # Manhattan dilation by `padding` steps (ping-pong on a scratch copy)
        if padding > 0 and f1 > f0:
            tmp = np.empty(h * w, dtype=np.bool_)
            for _ in range(padding):
                for j in range(h * w):
                    tmp[j] = buf[o + j]
                for py in range(h):
                    for px in range(w):
                        if tmp[py * w + px]:
                            continue
                        hit = False
                        if py > 0 and tmp[(py - 1) * w + px]:
                            hit = True
                        elif py < h - 1 and tmp[(py + 1) * w + px]:
                            hit = True
                        elif px > 0 and tmp[py * w + px - 1]:
                            hit = True
                        elif px < w - 1 and tmp[py * w + px + 1]:
                            hit = True
                        if hit:
                            buf[o + py * w + px] = True
        # trimmed dims (keep origin; 1x1 empty bitmap when nothing was rasterized)
        rmax = -1
        cmax = -1
        for py in range(h):
            for px in range(w):
                if buf[o + py * w + px]:
                    if py > rmax:
                        rmax = py
                    if px > cmax:
                        cmax = px
        if rmax < 0:
            for j in range(h * w):
                buf[o + j] = False
            tw[c] = 1
            th_out[c] = 1
        else:
            tw[c] = cmax + 1
            th_out[c] = rmax + 1
        # unique-edge perimeter via sorted int64 keys
        Fc = f1 - f0
        if Fc > 0 and V > 0:
            keys = np.empty(Fc * 3, dtype=np.int64)
            for fi in range(f0, f1):
                for j in range(3):
                    a = faces[fi, j]
                    b = faces[fi, (j + 1) % 3]
                    if a < b:
                        keys[(fi - f0) * 3 + j] = a * V + b
                    else:
                        keys[(fi - f0) * 3 + j] = b * V + a
            keys = np.sort(keys)
            p = 0.0
            for i in range(keys.shape[0]):
                if i > 0 and keys[i] == keys[i - 1]:
                    continue
                a = keys[i] // V + v0
                b = keys[i] % V + v0
                dx = rot_uv[a, 0] - rot_uv[b, 0]
                dy = rot_uv[a, 1] - rot_uv[b, 1]
                p += math.sqrt(dx * dx + dy * dy)
            perim[c] = p


@_njit(cache=True, boundscheck=False, parallel=True)
def _place_all_jit(buf, boff, stride_w, tw, th, order, start, stop,
                   atlas, skyline, pool, attempts, sweep_cap, margin,
                   n_threads, cur_wh, out_x, out_y, out_sw):
    """Place charts order[start:stop]; returns the first index NOT processed (== stop when
    done, earlier when the atlas must grow — the caller resizes and resumes). The candidate
    scan is striped with a (score, index) min-reduction: deterministic for any thread count,
    and no thread intrinsics (dynamic globals would defeat cache=True)."""
    aw = atlas.shape[1]
    ah = atlas.shape[0]
    cur_w = cur_wh[0]
    cur_h = cur_wh[1]
    n_pool = pool.shape[0]
    big = np.int64(1) << 62
    nt = n_threads
    t_score = np.empty(nt, dtype=np.int64)
    t_k = np.empty(nt, dtype=np.int64)
    t_x = np.empty(nt, dtype=np.int64)
    t_y = np.empty(nt, dtype=np.int64)
    t_sw = np.empty(nt, dtype=np.int64)
    for oi in range(start, stop):
        ci = order[oi]
        if cur_h + margin > ah or cur_w + margin > aw:
            cur_wh[0] = cur_w
            cur_wh[1] = cur_h
            return oi
        w0 = tw[ci]                                   # unswapped trimmed dims
        h0 = th[ci]
        W = stride_w[ci]                              # row stride of the untrimmed block
        o = boff[ci]
        step = min(w0, h0) // 8
        if step < 1:
            step = 1
        cap_step = max(cur_w, cur_h) // sweep_cap
        if cap_step > step:
            step = cap_step

        poff = (oi * attempts) % (n_pool - attempts + 1)
        x_range = cur_w + 1 if cur_w > 0 else 1
        y_range = cur_h + 1 if cur_h > 0 else 1
        # candidate groups per orientation: skyline-flush sweep, y=0 / y=cur_h sweeps,
        # x=0 / x=cur_w sweeps; then the shared random pool
        nx = max(cur_w, 1) // step + 2
        ny = max(cur_h, 1) // step + 2
        n_det = nx * 3 + ny * 2
        total = n_det * 2 + attempts
        for t in range(nt):
            t_score[t] = big
            t_k[t] = big
        for t2 in _prange(nt):
            for k in range(t2, total, nt):
                x = 0                                      # int inits and no body-level continue:
                y = 0                                      # parfor lowering types undef-path
                swap = 0                                   # variables as f64
                valid = True
                if k < 2 * n_det:
                    if k >= n_det:
                        swap = 1
                    kk = k - n_det if swap == 1 else k
                    cw = w0 if swap == 0 else h0
                    if kk < nx:                            # skyline-flush sweep
                        x = kk * step
                        if x > cur_w:
                            valid = False
                        else:
                            x_end = x + cw
                            if x_end > skyline.shape[0]:
                                x_end = skyline.shape[0]
                            for xs in range(x, x_end):
                                if skyline[xs] > y:
                                    y = int(skyline[xs])
                    elif kk < 3 * nx:                      # y=0 and y=cur_h sweeps
                        kk2 = kk - nx
                        x = (kk2 % nx) * step
                        if x > cur_w:
                            valid = False
                        elif kk2 >= nx:
                            y = cur_h
                    else:                                  # x=0 and x=cur_w sweeps
                        kk2 = kk - 3 * nx
                        if kk2 >= 2 * ny:
                            valid = False
                        else:
                            y = (kk2 % ny) * step
                            if y > cur_h:
                                valid = False
                            elif kk2 >= ny:
                                x = cur_w
                else:
                    r = k - 2 * n_det
                    x = int(pool[poff + r, 0] % x_range)
                    y = int(pool[poff + r, 1] % y_range)
                    swap = int(r & 1)

                if valid:
                    ch = h0 if swap == 0 else w0
                    cw = w0 if swap == 0 else h0
                    nw = cur_w if cur_w > x + cw else x + cw
                    nh = cur_h if cur_h > y + ch else y + ch
                    ext = nw if nw > nh else nh
                    score = ext * ext + nw * nh
                    if score < t_score[t2] or (score == t_score[t2] and k < t_k[t2]):
                        ok = True
                        for j in range(ch):
                            yy = int(y + j)
                            if yy >= ah:
                                continue
                            for i in range(cw):
                                if swap == 0:
                                    bit = buf[o + j * W + i]
                                else:
                                    # 90deg rotation: bm_rot[j, i] = bm[h0-1-i, j]
                                    bit = buf[o + (h0 - 1 - i) * W + j]
                                if not bit:
                                    continue
                                xx = int(x + i)
                                if xx >= aw:
                                    continue
                                if atlas[yy, xx]:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            t_score[t2] = score
                            t_k[t2] = k
                            t_x[t2] = x
                            t_y[t2] = y
                            t_sw[t2] = swap

        best_x = -1
        best_y = -1
        best_swap = 0
        bs = big
        bk = big
        for t in range(nt):
            if t_score[t] < bs or (t_score[t] == bs and t_k[t] < bk):
                bs = t_score[t]
                bk = t_k[t]
                best_x = t_x[t]
                best_y = t_y[t]
                best_swap = t_sw[t]

        if best_x < 0:                                 # fallback: extension corner
            best_x = cur_w
            best_y = 0
            best_swap = 0
        bh_ = h0 if best_swap == 0 else w0
        bw_ = w0 if best_swap == 0 else h0
        # blit + extents + skyline lift
        for j in range(bh_):
            for i in range(bw_):
                if best_swap == 0:
                    bit = buf[o + j * W + i]
                else:
                    bit = buf[o + (h0 - 1 - i) * W + j]
                if bit:
                    atlas[best_y + j, best_x + i] = True
        if best_x + bw_ > cur_w:
            cur_w = best_x + bw_
        if best_y + bh_ > cur_h:
            cur_h = best_y + bh_
        for i in range(bw_):
            col_x = best_x + i
            if col_x >= skyline.shape[0]:
                continue
            col_top = -1
            for j in range(bh_ - 1, -1, -1):
                if best_swap == 0:
                    bit = buf[o + j * W + i]
                else:
                    bit = buf[o + (h0 - 1 - i) * W + j]
                if bit:
                    col_top = j
                    break
            if col_top >= 0:
                nh2 = best_y + col_top + 1
                if nh2 > skyline[col_x]:
                    skyline[col_x] = nh2
        out_x[ci] = best_x
        out_y[ci] = best_y
        out_sw[ci] = best_swap
    cur_wh[0] = cur_w
    cur_wh[1] = cur_h
    return stop


# Torch fallback (used when numba is unavailable; runs on GPU if present)

def _dilate_local(x: Tensor, p: int) -> Tensor:
    """4-connectivity dilation by p over a batch of (cnt,g,g) bitmaps. Dilation distributes
    over union, so dilating per-triangle then OR-scattering equals dilating the chart."""
    for _ in range(p):
        y = x.clone()
        y[:, 1:, :] |= x[:, :-1, :]
        y[:, :-1, :] |= x[:, 1:, :]
        y[:, :, 1:] |= x[:, :, :-1]
        y[:, :, :-1] |= x[:, :, 1:]
        x = y
    return x


def _raster_all_torch(uvs_tex, uv_offsets, faces_cat, face_offsets, bw_t, bh_t, padding, device):
    """Rasterize every chart into one flat bool buffer; buf[cbase[i]:cbase[i+1]].view(bh,bw)
    is chart i's bitmap. Triangles are bucketed by next-pow2 bbox size to bound memory."""
    n = len(face_offsets) - 1
    bwL, bhL = bw_t.long(), bh_t.long()
    cbase = torch.zeros(n + 1, dtype=torch.long, device=device)
    torch.cumsum(bwL * bhL, 0, out=cbase[1:])
    buf = torch.zeros(int(cbase[-1].item()), dtype=torch.bool, device=device)

    total_faces = int(faces_cat.shape[0])
    if total_faces == 0:
        return buf, cbase
    faces = torch.from_numpy(np.ascontiguousarray(faces_cat)).to(device=device, dtype=torch.long)
    face_counts = torch.from_numpy(np.diff(face_offsets)).to(device=device, dtype=torch.long)
    face_charts = torch.arange(n, device=device).repeat_interleave(face_counts)
    uv_offsets_t = torch.as_tensor(uv_offsets, dtype=torch.long, device=device)
    free = comfy.model_management.get_free_memory(device)
    budget = int(min(1 << 23, max(1 << 20, (free * 0.25) / 56)))
    for fs in range(0, total_faces, _TORCH_RASTER_FACE_BATCH):
        fe = min(fs + _TORCH_RASTER_FACE_BATCH, total_faces)
        cid = face_charts[fs:fe]
        tri_f = uvs_tex[faces[fs:fe] + uv_offsets_t[cid, None]]

        # per-triangle pixel bbox, inflated by padding (origin >= 0); bucket by next-pow2 max-dim
        tmin = tri_f.amin(1)
        tmax = tri_f.amax(1)
        x0 = (tmin[:, 0].floor().long() - padding).clamp_min(0)
        y0 = (tmin[:, 1].floor().long() - padding).clamp_min(0)
        bbw = (tmax[:, 0].ceil().long() + padding) - x0 + 1
        bbh = (tmax[:, 1].ceil().long() + padding) - y0 + 1
        mxd = torch.maximum(bbw, bbh).clamp_min(1)
        bsz = (2 ** torch.ceil(torch.log2(mxd.float())).long()).long()

        a = tri_f[:, 0]
        b = tri_f[:, 1]
        c = tri_f[:, 2]
        v0 = b - a
        v1 = c - a
        d00 = (v0 * v0).sum(-1)
        d01 = (v0 * v1).sum(-1)
        d11 = (v1 * v1).sum(-1)
        den = (d00 * d11 - d01 * d01).clamp(min=1e-20)

        for g in torch.unique(bsz).tolist():                             # one batch per pow2 grid
            sel_g = (bsz == g).nonzero(as_tuple=True)[0]
            per = max(1, budget // (g * g))
            for cs in range(0, sel_g.shape[0], per):
                sel = sel_g[cs:cs + per]
                m = sel.shape[0]
                xs0 = x0[sel].view(m, 1, 1)
                ys0 = y0[sel].view(m, 1, 1)
                cc = cid[sel]
                bwp = bwL[cc].view(m, 1, 1)
                bhp = bhL[cc].view(m, 1, 1)
                gi = torch.arange(g, device=device)
                px = xs0 + gi.view(1, 1, g)
                py = ys0 + gi.view(1, g, 1)                                    # (m,g,g) int
                pxf = px.float() + 0.5
                pyf = py.float() + 0.5
                v2x = pxf - a[sel, 0].view(m, 1, 1)
                v2y = pyf - a[sel, 1].view(m, 1, 1)
                d20 = v2x * v0[sel, 0].view(m, 1, 1) + v2y * v0[sel, 1].view(m, 1, 1)
                d21 = v2x * v1[sel, 0].view(m, 1, 1) + v2y * v1[sel, 1].view(m, 1, 1)
                idn = den[sel].view(m, 1, 1).reciprocal()
                vv = torch.addcmul(d11[sel].view(m, 1, 1) * d20, d01[sel].view(m, 1, 1), d21, value=-1) * idn
                ww = torch.addcmul(d00[sel].view(m, 1, 1) * d21, d01[sel].view(m, 1, 1), d20, value=-1) * idn
                uu = 1.0 - vv - ww
                inside = (uu >= -1e-6) & (vv >= -1e-6) & (ww >= -1e-6)
                if padding > 0:
                    inside = _dilate_local(inside, padding)
                valid = inside & (px < bwp) & (py < bhp)
                flat = (cbase[cc].view(m, 1, 1) + py * bwp + px)[valid]
                buf[flat] = True
                del sel, xs0, ys0, cc, bwp, bhp, gi, px, py, pxf, pyf
                del v2x, v2y, d20, d21, idn, vv, ww, uu, inside, valid, flat
        del cid, tri_f, tmin, tmax, x0, y0, bbw, bbh, mxd, bsz
        del a, b, c, v0, v1, d00, d01, d11, den
    return buf, cbase


def _build_candidates_gpu(sky_t, ar, cur_w, cur_h, bw0, bw1, step, rand01, device):
    """Candidate (x, y) positions as a (2, M, 2) tensor (dim 0 = orientation). The first
    n_sky rows per orientation are skyline-flush and collision-free by construction.
    rand01 is (2, rand_n, 2) pre-drawn uniforms; ar a preallocated arange."""
    hi_x = max(cur_w, 1) + 1
    hi_y = max(cur_h, 1) + 1
    xs = ar[0:hi_x:step]
    ys = ar[0:hi_y:step]
    n_sky = (hi_x + step - 1) // step
    zx = torch.zeros_like(xs)
    zy = torch.zeros_like(ys)
    common = torch.cat([
        torch.stack([xs, zx], 1), torch.stack([xs, zx + cur_h], 1),
        torch.stack([zy, ys], 1), torch.stack([zy + cur_w, ys], 1)])
    wm = []
    for cw in (bw0, bw1):
        span = (n_sky - 1) * step + cw
        wm.append(max_pool1d(sky_t[:span].view(1, 1, -1).float(), kernel_size=cw,
                             stride=step).view(-1))
    sky = torch.stack([torch.stack([xs, wm[0].long()], 1),
                       torch.stack([xs, wm[1].long()], 1)])
    lim = torch.tensor([hi_x, hi_y], dtype=rand01.dtype, device=device)
    rnd = (rand01 * lim).long()
    return torch.cat([sky, common.expand(2, -1, -1), rnd], 1), n_sky


def _best_placement_torch(atlas, pix0, dim0, dim1, cands, n_sky, cur_w, cur_h, device):
    """Lowest-score non-colliding placement as a (3,) int tensor [x, y, swap]. The best
    skyline candidate bounds the score; only strictly better candidates are pixel-tested."""
    m = cands.shape[1]
    chw = torch.tensor([[dim0[0], dim0[1]], [dim1[0], dim1[1]]], device=device)
    nw = torch.clamp(cands[..., 0] + chw[:, 1:], min=cur_w)             # (2,M)
    nh = torch.clamp(cands[..., 1] + chw[:, :1], min=cur_h)
    ext = torch.maximum(nw, nh)
    sc = ext * ext + nw * nh
    js = sc[:, :n_sky].reshape(-1).argmin()                             # best skyline candidate
    sky_o = js // n_sky
    s_star = sc[:, :n_sky].reshape(-1)[js]
    sky = torch.cat([cands[sky_o, js % n_sky], sky_o.reshape(1)])
    cflat = cands.reshape(-1, 2)
    surv = (sc.reshape(-1) < s_star).nonzero(as_tuple=True)[0]          # compact once
    total = surv.shape[0]
    if total == 0:
        return sky

    k = pix0.shape[0]
    if k == 0:                                                          # empty chart: anywhere free
        j = surv[sc.reshape(-1)[surv].argmin()]
        return torch.cat([cflat[j], (j // m).reshape(1)])
    ordr = surv[torch.argsort(sc.reshape(-1)[surv], stable=True)]

    # flattened-index collision test: one int32 gather index instead of two int64 rows/cols
    aw = atlas.shape[1]
    idt = torch.int32 if atlas.numel() < (1 << 31) else torch.long
    lin0 = (pix0[:, 0] * aw + pix0[:, 1]).to(idt)                       # (y, x)
    lin1 = (pix0[:, 1] * aw + (dim0[0] - 1 - pix0[:, 0])).to(idt)       # rotated: (x, h-1-y)
    linp = torch.stack([lin0, lin1])
    aflat = atlas.view(-1)
    og = (ordr >= m).long()
    base = (cflat[ordr, 1] * aw + cflat[ordr, 0]).to(idt)

    # prescreen survivors on ~128 strided pixels: a sampled hit proves collision, so only
    # subsample-clean candidates need the exact test
    stride = (k + 127) // 128
    linp_sub = linp[:, ::stride].contiguous()
    maybe = ~aflat[base[:, None] + linp_sub[og]].any(1)
    passers = maybe.nonzero(as_tuple=True)[0]                           # ascending = score-sorted
    npass = passers.shape[0]
    if npass == 0:
        return sky
    if stride == 1:                                                     # prescreen was already exact
        j = ordr[passers[0]]
        return torch.cat([cflat[j], (j // m).reshape(1)])

    budget = 1 << 22                                                    # pixel-tests per chunk
    start = 0
    while start < npass:
        take = max(1, budget // k)
        pi = passers[start:start + take]
        free = ~aflat[base[pi][:, None] + linp[og[pi]]].any(1)          # (t,k) True-pixel gather
        # single host read per chunk: whether a free hit exists and where
        has, first = torch.stack([free.any().long(), free.long().argmax()]).tolist()
        if has:
            j = ordr[pi[first]]                                         # lowest score: sorted order
            return torch.cat([cflat[j], (j // m).reshape(1)])
        start += take
        budget = min(budget * 4, 1 << 25)
    return sky


def _pack_bitmap_torch(uvs_cat, uv_offsets, chart_3d_areas, chart_uv_areas, faces_cat, face_offsets,
                       texels_per_unit, padding_texels, attempts=4096, rng_seed=0,
                       progress_callback=None):
    """Torch rasterize-and-place packer (numba-free fallback). Returns (placements, atlas_w, atlas_h)."""
    n = len(uv_offsets) - 1
    if n == 0:
        return [], 1, 1
    device = comfy.model_management.get_torch_device()
    ang = torch.linspace(0.0, math.pi / 2.0, 37, device=device)[:-1]
    cos_a, sin_a = ang.cos(), ang.sin()

    # ---- Prepare pass 1: best-rotation + scale + bbox in bounded batches ----
    vcount = np.diff(uv_offsets).astype(np.int64, copy=False)
    uvs_src = torch.from_numpy(np.ascontiguousarray(uvs_cat)).to(device=device, dtype=torch.float32)
    uvs_tex = torch.empty_like(uvs_src)
    a3 = torch.tensor([max(a, 1e-12) for a in chart_3d_areas], device=device)
    au = torch.tensor([max(a, 1e-12) for a in chart_uv_areas], device=device)
    ti_all = torch.empty(n, dtype=torch.long, device=device)
    scale_all = torch.empty(n, device=device)
    bw_t = torch.empty(n, dtype=torch.int, device=device)
    bh_t = torch.empty(n, dtype=torch.int, device=device)
    start = 0
    while start < n:
        end = start
        vmax = 0
        while end < n:
            next_vmax = max(vmax, int(vcount[end]))
            if end > start and (end - start + 1) * next_vmax * 36 > _TORCH_PREP_MAX_ANGLE_ELEMENTS:
                break
            vmax = next_vmax
            end += 1
        m = end - start
        uvs_pad = torch.zeros(m, vmax, 2, device=device)
        vmask = torch.zeros(m, vmax, dtype=torch.bool, device=device)
        for j, i in enumerate(range(start, end)):
            count = int(vcount[i])
            uvs_pad[j, :count] = uvs_src[int(uv_offsets[i]):int(uv_offsets[i + 1])]
            vmask[j, :count] = True
        u0, u1 = uvs_pad[..., 0], uvs_pad[..., 1]
        xsp = torch.empty(m, 36, device=device)
        ysp = torch.empty(m, 36, device=device)
        angle_batch = max(1, min(36, _TORCH_PREP_MAX_ANGLE_ELEMENTS // max(m * vmax, 1)))
        mask3 = vmask[:, :, None]
        for a0 in range(0, 36, angle_batch):
            a1 = min(a0 + angle_batch, 36)
            cc = cos_a[a0:a1]
            ss = sin_a[a0:a1]
            xr = torch.addcmul(u0[:, :, None] * cc, u1[:, :, None], ss, value=-1)
            xr.masked_fill_(~mask3, -1e30)
            xmax = xr.amax(1)
            xr.masked_fill_(~mask3, 1e30)
            xsp[:, a0:a1] = xmax - xr.amin(1)
            del xr
            yr = torch.addcmul(u0[:, :, None] * ss, u1[:, :, None], cc)
            yr.masked_fill_(~mask3, -1e30)
            ymax = yr.amax(1)
            yr.masked_fill_(~mask3, 1e30)
            ysp[:, a0:a1] = ymax - yr.amin(1)
            del yr
        ti = (xsp * ysp).argmin(1)
        cc, ss = cos_a[ti][:, None], sin_a[ti][:, None]
        rx = torch.addcmul(u0 * cc, u1, ss, value=-1)
        ry = torch.addcmul(u0 * ss, u1, cc)
        rx.masked_fill_(~vmask, -1e30)
        rxmax = rx.amax(1)
        rx.masked_fill_(~vmask, 1e30)
        rxmin = rx.amin(1)
        ry.masked_fill_(~vmask, -1e30)
        rymax = ry.amax(1)
        ry.masked_fill_(~vmask, 1e30)
        rymin = ry.amin(1)
        base = (a3[start:end] / au[start:end]).sqrt() * texels_per_unit
        maxb = (4.0 * a3[start:end].sqrt() * texels_per_unit).clamp_min(8.0)
        bbm = torch.maximum(rxmax - rxmin, rymax - rymin).clamp_min(1e-12)
        scale = torch.minimum(base, maxb / bbm)
        uvs_batch = torch.stack([(rx - rxmin[:, None]) * scale[:, None],
                                 (ry - rymin[:, None]) * scale[:, None]], dim=-1)
        uvs_tex[int(uv_offsets[start]):int(uv_offsets[end])] = uvs_batch[vmask]
        ti_all[start:end] = ti
        scale_all[start:end] = scale
        bw_t[start:end] = ((rxmax - rxmin) * scale).ceil().int() + padding_texels + 1
        bh_t[start:end] = ((rymax - rymin) * scale).ceil().int() + padding_texels + 1
        del uvs_pad, vmask, u0, u1, xsp, ysp, mask3, rx, ry, uvs_batch
        start = end

    # one sync: pull all per-chart scalars
    thetas = ang[ti_all].cpu().tolist()
    scales = scale_all.cpu().tolist()
    del uvs_src, a3, au, ti_all, scale_all

    # ---- Prepare pass 2: rasterize ALL charts at once, then derive per-chart sparse data ----
    buf, cbase = _raster_all_torch(
        uvs_tex, uv_offsets, faces_cat, face_offsets, bw_t, bh_t, padding_texels, device)

    # nonzero over the flat buffer is ascending, so pixels come out grouped by chart
    nz = buf.nonzero(as_tuple=True)[0]
    del buf
    cid = torch.searchsorted(cbase, nz, right=True) - 1
    bwl = bw_t.long()
    local = nz - cbase[cid]
    py = local // bwl[cid]
    px = local - py * bwl[cid]
    del nz, local
    counts = torch.bincount(cid, minlength=n)
    rmax = torch.full((n,), -1, dtype=torch.long, device=device)
    cmax = torch.full((n,), -1, dtype=torch.long, device=device)
    rmax.scatter_reduce_(0, cid, py, reduce="amax")
    cmax.scatter_reduce_(0, cid, px, reduce="amax")
    ht = (rmax + 1).clamp_min(1)                    # trimmed bitmap dims (1x1 when empty)
    wt = (cmax + 1).clamp_min(1)
    pix_all = torch.stack([py, px], 1)              # True-pixel (row, col) offsets, sparse
    pixr_all = torch.stack([px, rmax[cid] - py], 1)  # 90deg rotation: (y, x) -> (x, h-1-y)
    meta = torch.stack([ht, wt, counts.cumsum(0)], 1).cpu().tolist()     # one sync for all charts
    dim_l = [(m[0], m[1]) for m in meta]
    dimr_l = [(w, h) for (h, w) in dim_l]
    offs = [0] + [m[2] for m in meta]
    pix_l = [pix_all[offs[i]:offs[i + 1]] for i in range(n)]
    pixr_l = [pixr_all[offs[i]:offs[i + 1]] for i in range(n)]

    del cid, py, px, rmax, cmax

    # ---- Placement: skyline bin-pack on GPU ----
    order = sorted(range(n), key=lambda i: -(dim_l[i][0] * dim_l[i][1]))   # biggest bitmap first
    max_b = max(max(d) for d in dim_l)
    margin = max_b + 8
    side_guess = int(math.sqrt(sum(d[0] * d[1] for d in dim_l)) * 2) + 16
    cap = side_guess + margin
    atlas = torch.zeros((cap, cap), dtype=torch.bool, device=device)
    sky_t = torch.zeros(cap, dtype=torch.long, device=device)
    ar = torch.arange(cap + 1, device=device)
    cur_w = cur_h = 0
    placements = [None] * n
    gen = torch.Generator(device=device).manual_seed(rng_seed)
    rand_n = min(512, attempts)                     # random samples per orientation
    # no _SWEEP_CAP here: the skyline-bound pruning depends on the dense sweep
    rand01 = torch.rand(n, 2, rand_n, 2, generator=gen, device=device)   # all draws upfront

    for t_i, ci in enumerate(order):
        if progress_callback is not None and (t_i & 255) == 0:
            progress_callback(n + t_i, 2 * n)
        if cur_h + margin > atlas.shape[0] or cur_w + margin > atlas.shape[1]:
            ns = max(atlas.shape[0], cur_h + margin, cur_w + margin)
            na = torch.zeros((ns, ns), dtype=torch.bool, device=device)
            na[:atlas.shape[0], :atlas.shape[1]] = atlas
            atlas = na
            nsk = torch.zeros(ns, dtype=torch.long, device=device)
            nsk[:sky_t.shape[0]] = sky_t
            sky_t = nsk
            ar = torch.arange(ns + 1, device=device)
        dim, dimr = dim_l[ci], dimr_l[ci]
        step = max(1, min(dim[0], dim[1]) // 8)
        cands, n_sky = _build_candidates_gpu(
            sky_t, ar, cur_w, cur_h, dim[1], dimr[1], step, rand01[t_i], device)
        res = _best_placement_torch(atlas, pix_l[ci], dim, dimr,
                                    cands, n_sky, cur_w, cur_h, device)
        bx, by, swap = (int(v) for v in res.tolist())
        if bx < 0:
            bx, by, swap = cur_w, 0, 0
        pix = pixr_l[ci] if swap else pix_l[ci]
        bh_, bw_ = (dimr if swap else dim)
        atlas[by + pix[:, 0], bx + pix[:, 1]] = True                    # sparse blit
        cur_w = max(cur_w, bx + bw_)
        cur_h = max(cur_h, by + bh_)
        ct = torch.full((bw_,), -1, dtype=torch.long, device=device)
        ct.scatter_reduce_(0, pix[:, 1], pix[:, 0], reduce="amax")
        ix = ar[bx:bx + bw_]
        sky_t[ix] = torch.where(ct >= 0, torch.maximum(sky_t[ix], by + ct + 1), sky_t[ix])
        placements[ci] = ChartPlacement(chart_id=ci, offset=(float(bx), float(by)),
                                        scale=scales[ci], rotation=thetas[ci], swap_xy=bool(swap),
                                        chart_h=float(dim_l[ci][0]))
    return placements, cur_w, cur_h


def pack_bitmap_concat(
    uvs_cat: np.ndarray,          # (sumV, 2) per-chart concatenated UVs
    uv_offsets: np.ndarray,       # (n+1,)
    faces_cat: np.ndarray,        # (sumF, 3) local vert ids per chart
    face_offsets: np.ndarray,     # (n+1,)
    chart_3d_areas: np.ndarray,
    chart_uv_areas: np.ndarray,
    texels_per_unit: float = 256.0,
    padding_texels: int = 2,
    attempts: int = 4096,
    rng_seed: int = 0,
    progress_callback=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Rasterize-and-place packer over concatenated chart arrays (no per-chart python).
    Returns (x, y, swap, rotation, scale, chart_h, atlas_w, atlas_h) with one entry per chart.
    progress_callback(done, total) is invoked periodically; total is 2*n_charts."""
    n = int(uv_offsets.shape[0]) - 1
    empty = np.zeros(n, dtype=np.int64)
    if n == 0:
        return empty, empty, empty, empty.astype(np.float64), empty.astype(np.float64), empty, 1, 1
    if not _HAVE_NUMBA_PACK:
        placements, w, h = _pack_bitmap_torch(
            uvs_cat, uv_offsets, [float(a) for a in chart_3d_areas], [float(a) for a in chart_uv_areas],
            faces_cat, face_offsets, texels_per_unit, padding_texels, attempts=attempts,
            rng_seed=rng_seed, progress_callback=progress_callback)
        px = np.array([p.offset[0] for p in placements], dtype=np.int64)
        py = np.array([p.offset[1] for p in placements], dtype=np.int64)
        sw = np.array([1 if p.swap_xy else 0 for p in placements], dtype=np.int64)
        th = np.array([p.rotation for p in placements], dtype=np.float64)
        sc = np.array([p.scale for p in placements], dtype=np.float64)
        chh = np.array([p.chart_h for p in placements], dtype=np.int64)
        return px, py, sw, th, sc, chh, w, h

    uvs64 = np.ascontiguousarray(uvs_cat, dtype=np.float64)
    faces64 = np.ascontiguousarray(faces_cat, dtype=np.int64)
    uv_off = np.ascontiguousarray(uv_offsets, dtype=np.int64)
    f_off = np.ascontiguousarray(face_offsets, dtype=np.int64)
    a3 = np.ascontiguousarray(chart_3d_areas, dtype=np.float64)
    auv = np.ascontiguousarray(chart_uv_areas, dtype=np.float64)

    theta = np.zeros(n, dtype=np.float64)
    scale = np.zeros(n, dtype=np.float64)
    bw = np.zeros(n, dtype=np.int64)
    bh = np.zeros(n, dtype=np.int64)
    rot_uv = np.empty_like(uvs64)
    _prepare_dims_jit(uvs64, uv_off, a3, auv, float(texels_per_unit), int(padding_texels),
                      theta, scale, bw, bh, rot_uv)
    boff = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(bw * bh, out=boff[1:])
    buf = np.zeros(int(boff[-1]), dtype=np.bool_)
    tw = np.zeros(n, dtype=np.int64)
    th_arr = np.zeros(n, dtype=np.int64)
    perim = np.zeros(n, dtype=np.float64)
    _raster_all_jit(rot_uv, uv_off, faces64, f_off, bw, bh, boff, buf,
                    int(padding_texels), tw, th_arr, perim)
    if progress_callback is not None:
        progress_callback(n, 2 * n)

    order = np.argsort(-perim, kind="stable")
    max_b = int(max(int(tw.max()), int(th_arr.max())))
    margin = max_b + 8
    side_guess = int(math.sqrt(float((tw * th_arr).sum()))) * 2 + 16
    cap = side_guess + margin
    atlas = np.zeros((cap, cap), dtype=np.bool_)
    skyline = np.zeros(cap, dtype=np.int64)
    rng = np.random.default_rng(rng_seed)
    # shared random pool, sliced at a rotating offset per chart
    pool = rng.integers(0, 1 << 31, size=(attempts * 8, 2)).astype(np.int64)
    out_x = np.full(n, -1, dtype=np.int64)
    out_y = np.full(n, -1, dtype=np.int64)
    out_sw = np.zeros(n, dtype=np.int64)
    cur_wh = np.zeros(2, dtype=np.int64)
    start = 0
    while start < n:
        stop = min(n, start + 1024)
        nxt = _place_all_jit(buf, boff, bw, tw, th_arr, order, start, stop,
                             atlas, skyline, pool, int(attempts), int(_SWEEP_CAP),
                             int(margin), int(_nb_threads()), cur_wh, out_x, out_y, out_sw)
        if nxt < stop:                               # atlas must grow before this chart fits
            ns = max(atlas.shape[0], int(cur_wh[1]) + margin, int(cur_wh[0]) + margin)
            na = np.zeros((ns, ns), dtype=np.bool_)
            na[:atlas.shape[0], :atlas.shape[1]] = atlas
            atlas = na
            nsk = np.zeros(ns, dtype=np.int64)
            nsk[:skyline.shape[0]] = skyline
            skyline = nsk
        start = nxt
        if progress_callback is not None:
            progress_callback(n + start, 2 * n)
    return out_x, out_y, out_sw, theta, scale, th_arr, int(cur_wh[0]), int(cur_wh[1])


def apply_placements_concat(
    uvs_cat: np.ndarray, uv_offsets: np.ndarray,
    px: np.ndarray, py: np.ndarray, sw: np.ndarray,
    theta: np.ndarray, scale: np.ndarray, chart_h: np.ndarray,
    atlas_w: int, atlas_h: int,
) -> np.ndarray:
    """apply_placements over concatenated charts, fully vectorized. Returns (sumV, 2) float32."""
    n = int(uv_offsets.shape[0]) - 1
    side = float(max(atlas_w, atlas_h, 1))
    cov = np.repeat(np.arange(n), np.diff(uv_offsets))
    u_in = uvs_cat[:, 0].astype(np.float64)
    v_in = uvs_cat[:, 1].astype(np.float64)
    c = np.cos(theta)[cov]
    s = np.sin(theta)[cov]
    u = u_in * c - v_in * s
    v = u_in * s + v_in * c
    umin = np.full(n, np.inf)
    vmin = np.full(n, np.inf)
    np.minimum.at(umin, cov, u)
    np.minimum.at(vmin, cov, v)
    u = (u - umin[cov]) * scale[cov]
    v = (v - vmin[cov]) * scale[cov]
    swv = sw[cov].astype(bool)
    # 90 deg rotation matching the rotated-bitmap access: (u, v) -> (chart_h - v, u)
    u2 = np.where(swv, chart_h[cov] - v, u) + px[cov]
    v2 = np.where(swv, u, v) + py[cov]
    out = np.stack([u2, v2], axis=1) / side
    np.clip(out, 0.0, 1.0, out=out)                  # slivers can stick sub-texel past extents
    return out.astype(np.float32)
