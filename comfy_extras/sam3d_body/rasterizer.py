"""Pure-PyTorch rasterizer for SAM 3D Body meshes.

Algorithm: forward triangle rasterizer with hard z-buffer. Per-face screen
bbox cull → faces sorted by bbox size and chunked under a fixed pixel
budget → inside-test via edge functions, barycentric interpolation, depth
test via `scatter_reduce_(amin)`.
"""

from typing import Sequence

import numpy as np
import torch

import comfy.model_management
from .utils import jet_colormap

_CANONICAL_PRESETS = {"rainbow", "rainbow_face_normal", "rainbow_face_semantic"}

def _faces_to_device(faces, device, cache: dict) -> torch.Tensor:
    """Device-side face indices. Topology is static per model, but the render
    loop calls this once per frame, so keep the H2D copy out of the loop."""
    key = ("faces", id(faces), device)
    hit = cache.get(key)
    if hit is not None:
        return hit[1]
    t = torch.as_tensor(np.asarray(faces, dtype=np.int64), device=device)
    # Hold the source array too: id() alone could be recycled after a GC.
    cache[key] = (faces, t)
    return t


def rainbow_colors_from_canonical(
    positions: np.ndarray,
    tilt_x_deg: float = 0.0,
    tilt_z_deg: float = 0.0,
) -> np.ndarray:
    """Compute per-vertex jet-colormap RGB from canonical (T-pose, Y-up) vertices.

    Args:
        positions: (N_v, 3) canonical vertex positions, Y-up (head at max Y).
        tilt_x_deg: rotation of the jet axis around X (in degrees). Positive
            biases the ramp toward +Z (front).
        tilt_z_deg: rotation of the jet axis around Z (in degrees). Positive
            biases the ramp toward +X (right, in body frame).

    Returns:
        (N_v, 3) float32 RGB in [0, 1].
    """
    theta_x = np.deg2rad(tilt_x_deg)
    theta_z = np.deg2rad(tilt_z_deg)
    axis = np.array([
        np.sin(theta_z),
        np.cos(theta_z) * np.cos(theta_x),
        np.cos(theta_z) * np.sin(theta_x),
    ], dtype=np.float32)

    s = positions @ axis
    s = (s - s.min()) / max(float(s.max() - s.min()), 1e-8)
    s = np.clip(s * 0.98, 0.0, 1.0).astype(np.float32)
    return jet_colormap(s)


def _vertex_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Area-weighted per-vertex normals; matches `_compute_vertex_normals`."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)
    vn = torch.zeros_like(verts)
    vn.index_add_(0, faces[:, 0], fn)
    vn.index_add_(0, faces[:, 1], fn)
    vn.index_add_(0, faces[:, 2], fn)
    return vn / vn.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def _build_vcolor(canonical_colors, shader_preset, tilt_x, tilt_z):
    """Mirrors the canonical_colors -> per-vertex RGB pipeline in
    `rasterizer.render_pose_data`. Returns a numpy float32 (V, 3) table."""
    positions = np.asarray(canonical_colors.get("positions"), dtype=np.float32)
    vcolor = rainbow_colors_from_canonical(positions, tilt_x_deg=tilt_x, tilt_z_deg=tilt_z).copy()

    if shader_preset in ("rainbow_face_normal", "rainbow_face_semantic"):
        face_mask = canonical_colors.get("face_mask")
        if face_mask is not None and face_mask.any():
            if shader_preset == "rainbow_face_normal":
                norm = np.asarray(canonical_colors["norm"], dtype=np.float32)
                vcolor[face_mask] = norm[face_mask]
            else:  # rainbow_face_semantic
                sem = np.asarray(canonical_colors["face_region_rgb"], dtype=np.float32)
                assigned = sem.sum(axis=1) > 0
                vcolor[assigned] = sem[assigned]
    return vcolor


def _rasterize_chunk(
    fv_pix: torch.Tensor,    # (Fc, 3, 2) — pixel coords (sub-pixel float)
    fv_z: torch.Tensor,      # (Fc, 3)    — image-frame z (smaller=closer)
    bb_min_x: torch.Tensor, bb_max_x: torch.Tensor,  # (Fc,) clamped int bboxes
    bb_min_y: torch.Tensor, bb_max_y: torch.Tensor,
    max_sx: int, max_sy: int,
    W: int,
):
    """Rasterize a chunk of faces at pixel centers. Returns flat tensors of
    inside fragments: (pixel_idx, depth, face_local, bary).
    """
    device = fv_pix.device
    if max_sx == 0 or max_sy == 0:
        return None

    sx = bb_max_x - bb_min_x
    sy = bb_max_y - bb_min_y

    px_off = torch.arange(max_sx, device=device)
    py_off = torch.arange(max_sy, device=device)

    # Pixel-center sample positions, broadcast to (Fc, max_sy, max_sx).
    P_x = (bb_min_x[:, None, None] + px_off[None, None, :]).float() + 0.5
    P_y = (bb_min_y[:, None, None] + py_off[None, :, None]).float() + 0.5

    in_bb = (px_off[None, None, :] < sx[:, None, None]) & \
            (py_off[None, :, None] < sy[:, None, None])

    Ax = fv_pix[:, 0, 0][:, None, None]
    Ay = fv_pix[:, 0, 1][:, None, None]
    Bx = fv_pix[:, 1, 0][:, None, None]
    By = fv_pix[:, 1, 1][:, None, None]
    Cx = fv_pix[:, 2, 0][:, None, None]
    Cy = fv_pix[:, 2, 1][:, None, None]

    area2 = (Bx - Ax) * (Cy - Ay) - (By - Ay) * (Cx - Ax)  # (Fc, 1, 1)
    e_a = (Bx - P_x) * (Cy - P_y) - (By - P_y) * (Cx - P_x)
    e_b = (Cx - P_x) * (Ay - P_y) - (Cy - P_y) * (Ax - P_x)
    e_c = (Ax - P_x) * (By - P_y) - (Ay - P_y) * (Bx - P_x)

    # Same-sign-as-area2 inside test (no back-face culling — match either winding).
    nondegen = area2.abs() > 1e-6 # threshold rejects near-degenerate triangles
    inside = (e_a * area2 >= 0) & (e_b * area2 >= 0) & (e_c * area2 >= 0)
    inside = inside & in_bb & nondegen

    inv_a2 = torch.where(nondegen, 1.0 / area2, torch.zeros_like(area2))
    w_a = e_a * inv_a2
    w_b = e_b * inv_a2
    w_c = e_c * inv_a2

    z_a = fv_z[:, 0, None, None]
    z_b = fv_z[:, 1, None, None]
    z_c = fv_z[:, 2, None, None]
    z_grid = w_a * z_a + w_b * z_b + w_c * z_c

    fi, yi, xi = inside.nonzero(as_tuple=True)
    px_pixel = bb_min_x[fi] + xi
    py_pixel = bb_min_y[fi] + yi
    pixel_idx = (py_pixel * W + px_pixel).long()
    z_flat = z_grid[fi, yi, xi]
    bary_flat = torch.stack([w_a[fi, yi, xi], w_b[fi, yi, xi], w_c[fi, yi, xi]], dim=-1)
    return pixel_idx, z_flat, fi.long(), bary_flat


def _rasterize_person(
    verts_world: torch.Tensor, faces: torch.Tensor,
    focal: float, W: int, H: int,
    z_buf: torch.Tensor, color_buf: torch.Tensor, mask_buf: torch.Tensor,
    shade_fn,
):
    # Project image-frame verts to pixel coords. Skip verts at/behind camera.
    z_min_ok = 0.05
    valid_v = verts_world[:, 2] > z_min_ok
    safe_z = verts_world[:, 2].clamp(min=z_min_ok)
    px = 0.5 * W + focal * verts_world[:, 0] / safe_z
    py = 0.5 * H + focal * verts_world[:, 1] / safe_z

    Fv_pix = torch.stack([px, py], dim=-1)[faces]   # (F, 3, 2)
    Fv_z = verts_world[faces][..., 2]               # (F, 3)
    Fv_valid = valid_v[faces].all(dim=-1)

    sx_face = Fv_pix[..., 0]
    sy_face = Fv_pix[..., 1]
    bb_min_x = sx_face.amin(dim=-1).floor().long().clamp(min=0, max=W)
    bb_max_x = (sx_face.amax(dim=-1).ceil().long() + 1).clamp(min=0, max=W)
    bb_min_y = sy_face.amin(dim=-1).floor().long().clamp(min=0, max=H)
    bb_max_y = (sy_face.amax(dim=-1).ceil().long() + 1).clamp(min=0, max=H)
    sx_all = bb_max_x - bb_min_x
    sy_all = bb_max_y - bb_min_y
    valid_face = Fv_valid & (sx_all > 0) & (sy_all > 0)

    keep = torch.where(valid_face)[0]
    if keep.numel() == 0:
        return

    bbsize = torch.maximum(sx_all, sy_all)[keep]
    PIXEL_BUDGET = 4_000_000
    MAX_CHUNK = 8192
    lower = 0
    limit = 1
    max_extent = max(W, H)
    while lower < max_extent:
        bin_keep = keep[(bbsize > lower) & (bbsize <= limit)]
        chunk_size = min(MAX_CHUNK, max(1, PIXEL_BUDGET // (limit * limit)))
        for i in range(0, bin_keep.shape[0], chunk_size):
            chunk = bin_keep[i : i + chunk_size]
            pixel_idx, z_chunk, face_local, bary = _rasterize_chunk(
                Fv_pix[chunk], Fv_z[chunk],
                bb_min_x[chunk], bb_max_x[chunk],
                bb_min_y[chunk], bb_max_y[chunk],
                limit, limit, W,
            )
            face_global = chunk[face_local]

            old_at = z_buf[pixel_idx].clone()
            z_buf.scatter_reduce_(0, pixel_idx, z_chunk, reduce='amin', include_self=True)
            new_at = z_buf[pixel_idx]
            is_min = (z_chunk == new_at) & (new_at < old_at)

            surv_pixel = pixel_idx[is_min]
            surv_face = face_global[is_min]
            surv_bary = bary[is_min]
            sort_perm = torch.argsort(surv_pixel, stable=True)
            sp = surv_pixel[sort_perm]
            first = torch.ones_like(sp, dtype=torch.bool)
            first[1:] = sp[1:] != sp[:-1]
            selected = sort_perm[first]

            wp_idx = surv_pixel[selected]
            color_buf[wp_idx] = shade_fn(surv_face[selected], surv_bary[selected])
            mask_buf[wp_idx] = True
        lower = limit
        limit = min(limit * 2, max_extent)


def _make_shade_fn(
    shader_preset, composite,
    view_normals_v, view_pos_v, vcolor_v, faces,
    base_color, light_dir, pastel_mix,
):
    device = view_normals_v.device
    base_color_t = torch.as_tensor(base_color, dtype=torch.float32, device=device)
    light_dir_t = torch.as_tensor(light_dir, dtype=torch.float32, device=device)
    # Light-vector constants — normalized once per render call.
    l_unit = -light_dir_t
    l_unit = l_unit / l_unit.norm().clamp(min=1e-8)

    if pastel_mix <= 0.0:
        apply_pastel = lambda rgb: rgb
    else:
        pm = float(pastel_mix)
        apply_pastel = lambda rgb: rgb * (1.0 - pm) + pm

    def gather_n(face_idx, bary):
        n = (view_normals_v[faces[face_idx]] * bary.unsqueeze(-1)).sum(dim=1)
        return n / n.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    def gather_pos(face_idx, bary):
        return (view_pos_v[faces[face_idx]] * bary.unsqueeze(-1)).sum(dim=1)

    def gather_color(face_idx, bary):
        return (vcolor_v[faces[face_idx]] * bary.unsqueeze(-1)).sum(dim=1)

    if composite == "silhouette":
        return lambda fi, ba: torch.ones((fi.shape[0], 3), device=device)

    if shader_preset == "normals":
        # View-space surface normal encoded as RGB (OpenGL Y+ convention).
        # +X right → R; +Y up → G; +Z toward viewer → B. Each face shows mostly
        # one channel, matching standard normal-map visualization.
        def shade(face_idx, bary):
            n = gather_n(face_idx, bary)
            return apply_pastel(((n + 1.0) * 0.5).clamp(0.0, 1.0))
        return shade

    use_vcolor = (shader_preset in _CANONICAL_PRESETS) and (vcolor_v is not None)

    if not use_vcolor:
        # default.frag: ambient + diffuse + rim
        def shade(face_idx, bary):
            n = gather_n(face_idx, bary)
            v = -gather_pos(face_idx, bary)
            v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            ndotl = (n * l_unit).sum(dim=-1).clamp(min=0)
            ndotv = (n * v).sum(dim=-1).clamp(min=0)
            rim = (1.0 - ndotv).pow(3.0)
            lit = 0.25 * base_color_t \
                + 0.75 * base_color_t * ndotl.unsqueeze(-1) \
                + 0.35 * rim.unsqueeze(-1)
            return apply_pastel(lit)
        return shade

    if shader_preset == "rainbow":
        def shade(face_idx, bary):
            base = gather_color(face_idx, bary)
            n = gather_n(face_idx, bary)
            ndotl = (n * l_unit).sum(dim=-1).clamp(min=0)
            return apply_pastel(base * (0.65 + 0.35 * ndotl).unsqueeze(-1))
        return shade

    # rainbow_face_*  →  rainbow_lit.frag. All light-direction & half-vector
    # constants depend only on the (constant) light_dir, so precompute them.
    key_l = l_unit
    fill_l = torch.stack([-key_l[0], key_l[1].abs(), -key_l[2]])
    view_dir = torch.tensor([0.0, 0.0, 1.0], device=device)
    h = key_l + view_dir
    h = h / h.norm().clamp(min=1e-8)

    def shade(face_idx, bary):
        base = gather_color(face_idx, bary)
        n = gather_n(face_idx, bary)
        key_ndotl = (n * key_l).sum(dim=-1).clamp(min=0)
        fill_ndotl = (n * fill_l).sum(dim=-1).clamp(min=0)
        rim = (1.0 - n[..., 2].clamp(min=0)).pow(2.5) * 0.30
        shade_val = (0.45 + 0.45 * key_ndotl + 0.15 * fill_ndotl + rim * 0.5).clamp(min=0.0, max=1.25)
        ndoth = (n * h).sum(dim=-1).clamp(min=0)
        spec = ndoth.pow(48) * 0.12
        lit = base * shade_val.unsqueeze(-1) + spec.unsqueeze(-1)
        return apply_pastel(lit)
    return shade


def render_pose_data_torch(
    pose_data: dict,
    frame_idx: int,
    W: int,
    H: int,
    background=None,             # Optional[np.ndarray | torch.Tensor] (H, W, 3) fp32 [0, 1]
    composite: str = "over",
    opacity: float = 1.0,
    shader_preset: str = "default",
    base_color: Sequence[float] = (0.68, 0.71, 0.78),
    light_dir: Sequence[float] = (0.4, -0.7, -0.6),
    rainbow_tilt_x_deg: float = 0.0,
    rainbow_tilt_z_deg: float = 0.0,
    person_brightness_falloff: float = 0.6,
    cache: dict | None = None,
) -> torch.Tensor:
    """Render one frame of persons from `pose_data` at resolution WxH.

    Returns an (H, W, 3) float32 torch.Tensor on the comfy compute device,
    ready to be stacked into the node's IMAGE output without a CPU round-trip."""
    device = comfy.model_management.get_torch_device()
    if cache is None:
        cache = {}
    persons = pose_data["frames"][frame_idx] if frame_idx < len(pose_data["frames"]) else []
    if len(persons) == 0:
        if composite == "over" and background is not None:
            if isinstance(background, np.ndarray):
                bg = torch.as_tensor(background, dtype=torch.float32, device=device)
            else:
                bg = background.to(device=device, dtype=torch.float32) if (
                    background.device != device or background.dtype != torch.float32
                ) else background
            return bg.clamp(0.0, 1.0)
        return torch.zeros((H, W, 3), device=device, dtype=torch.float32)

    faces = _faces_to_device(pose_data["faces"], device, cache)

    canonical_colors = pose_data.get("canonical_colors")
    using_canonical = shader_preset in _CANONICAL_PRESETS
    if using_canonical and canonical_colors is None:
        shader_preset = "default"
        using_canonical = False

    vcolor = None
    if using_canonical:
        color_key = ("vcolor", id(canonical_colors), shader_preset,
                     float(rainbow_tilt_x_deg), float(rainbow_tilt_z_deg), device)
        vcolor = cache.get(color_key)
        if vcolor is None:
            vcolor_np = _build_vcolor(canonical_colors, shader_preset,
                                      rainbow_tilt_x_deg, rainbow_tilt_z_deg)
            vcolor = torch.as_tensor(vcolor_np, dtype=torch.float32, device=device)
            cache[color_key] = vcolor

    falloff = max(0.0, min(1.0, float(person_brightness_falloff)))
    person_pastel = [0.0 if k == 0 else (1.0 - falloff ** k) for k in range(len(persons))]

    # Front-to-back draw order so nearer persons overdraw farther ones.
    order = sorted(range(len(persons)),
                   key=lambda i: -float(np.asarray(persons[i]["pred_cam_t"]).reshape(-1)[2]))

    HW = H * W
    z_buf = torch.full((HW,), float('inf'), device=device, dtype=torch.float32)
    color_buf = torch.zeros((HW, 3), device=device, dtype=torch.float32)
    mask_buf = torch.zeros(HW, device=device, dtype=torch.bool)

    for idx in order:
        p = persons[idx]
        verts_np = np.asarray(p["pred_vertices"], dtype=np.float32).reshape(-1, 3)
        cam_t = np.asarray(p["pred_cam_t"], dtype=np.float32).reshape(3)
        verts_world = torch.as_tensor(verts_np + cam_t[None, :],
                                      device=device, dtype=torch.float32)
        focal = float(np.asarray(p.get("focal_length", 5000.0)).reshape(-1)[0])

        # Image-frame (+Y down, +Z forward) → view-space (+Y up, -Z forward)
        # for shading, matching what the GL-style shader math expects.
        view_pos_v = torch.stack(
            [verts_world[:, 0], -verts_world[:, 1], -verts_world[:, 2]], dim=-1,
        )
        normals_world = _vertex_normals(verts_world, faces)
        view_normals_v = torch.stack(
            [normals_world[:, 0], -normals_world[:, 1], -normals_world[:, 2]], dim=-1,
        )

        vcolor_p = vcolor if (vcolor is not None and vcolor.shape[0] == verts_world.shape[0]) else None
        # Only canonical-vcolor shaders need vcolor; geometric shaders
        # ('normals', 'depth') and the lit default work without it.
        if shader_preset in _CANONICAL_PRESETS and vcolor_p is None:
            effective_preset = "default"
        else:
            effective_preset = shader_preset

        shade_fn = _make_shade_fn(
            effective_preset, composite,
            view_normals_v, view_pos_v, vcolor_p, faces,
            base_color, light_dir, person_pastel[idx],
        )

        _rasterize_person(
            verts_world, faces, focal, W, H,
            z_buf, color_buf, mask_buf, shade_fn,
        )

    # Stay on GPU through readback + composite.
    if shader_preset == "depth":
        # z_buf already holds linear image-frame z (smaller=closer; +inf where no mesh covers)
        # Normalize within the rendered mesh's range: near=white, far=black, background=black
        mask_2d = mask_buf.reshape(H, W)
        z_2d = z_buf.reshape(H, W)
        zmin = torch.where(mask_2d, z_2d, torch.full_like(z_2d, float("inf"))).amin()
        zmax = torch.where(mask_2d, z_2d, torch.full_like(z_2d, float("-inf"))).amax()
        zr = (zmax - zmin).clamp(min=1e-6)
        norm = torch.where(mask_2d, 1.0 - (z_2d - zmin) / zr, torch.zeros_like(z_2d))
        rendered = torch.stack([norm, norm, norm], dim=-1)
        mask_f = mask_2d.float()
    else:
        rendered = color_buf.reshape(H, W, 3).clamp(0.0, 1.0)
        mask_f = mask_buf.reshape(H, W).float()

    if composite == "over" and background is not None:
        if isinstance(background, np.ndarray):
            bg = torch.as_tensor(background, dtype=torch.float32, device=device)
        else:
            bg = background.to(device=device, dtype=torch.float32)
        a = mask_f.unsqueeze(-1)
        if opacity != 1.0:
            a = a * float(opacity)
        rendered = torch.lerp(bg, rendered, a)
    elif opacity != 1.0:
        rendered = rendered * float(opacity)

    return rendered
