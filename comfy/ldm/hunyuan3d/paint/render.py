# Torch-native geometry renderer, rasterizer, UV unwrap and multiview baker for the
# Hunyuan3D 2.1 paint pipeline. Pure PyTorch (no C++/CUDA extensions, no diffusers,
# no trimesh/xatlas): a barycentric z-buffer rasterizer feeds the conditioning
# normal/position maps the paint UNet consumes, and back-projects the generated
# multiview PBR images onto a per-triangle UV atlas.
#
# Coordinate / camera conventions follow Tencent's reference renderer
# (hy3dpaint/DifferentiableRenderer): an orthographic camera at distance 1.45 with a
# Z-up look-at, meshes auto-centered and scaled by 1.15, object-space (flat) face
# normals mapped to [0, 1] with a white background, and positions encoded as
# 0.5 - p / scale_factor (white background = "no geometry").

import math

import torch
import torch.nn.functional as F

# Fixed camera intrinsics of the released paint model (hy3dpaint MeshRender defaults).
ORTHO_SCALE = 1.2
CAMERA_DISTANCE = 1.45
NEAR = 0.1
FAR = 100.0
MESH_SCALE_FACTOR = 1.15

# The standard 6-view set the 2.1 pipeline renders (front, right, back, left, top, bottom).
STANDARD_VIEW_AZIMS = [0.0, 90.0, 180.0, 270.0, 0.0, 180.0]
STANDARD_VIEW_ELEVS = [0.0, 0.0, 0.0, 0.0, 90.0, -90.0]
STANDARD_VIEW_WEIGHTS = [1.0, 0.1, 0.5, 0.1, 0.05, 0.05]

# z-buffer depth quantisation levels for the packed scatter-min resolve.
_DEPTH_LEVELS = 1 << 24

# Documented soft bounds for the pure-torch rasterizer/baker. Rasterization work is
# chunked by cumulative bounding-box pixel area (see rasterize()), so across a typical
# mesh face count and resolution trade off against time rather than peak memory. The
# budget is applied at face boundaries, so this does not bound a single triangle: one
# whose bounding box covers much of the frame is still expanded in one go, and peak
# memory then follows that box rather than the chunk budget. CPU envelope (see PR
# notes): 100k faces @ 512 rasterize in well under a second; 1M faces in a few seconds.
MAX_RESOLUTION = 8192
MAX_FACES = 20_000_000


def _validate_mesh(vertices, faces, where):
    """Cheap input validation shared by the public renderer/baker entry points.

    Raises ValueError (with the failing entry point named) for the malformed-input
    classes that would otherwise surface as cryptic indexing errors or silent
    garbage: wrong shapes, empty meshes, non-finite vertices, out-of-range or
    negative face indices, and absurd face counts.
    """
    if vertices.ndim != 2 or vertices.shape[-1] != 3:
        raise ValueError(f"{where}: vertices must be (N, 3), got {tuple(vertices.shape)}")
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError(f"{where}: faces must be (F, 3), got {tuple(faces.shape)}")
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        raise ValueError(f"{where}: mesh is empty ({vertices.shape[0]} vertices, "
                         f"{faces.shape[0]} faces)")
    if faces.shape[0] > MAX_FACES:
        raise ValueError(f"{where}: mesh has {faces.shape[0]} faces "
                         f"(max supported {MAX_FACES}); decimate the mesh first")
    if not torch.isfinite(vertices).all():
        raise ValueError(f"{where}: vertices contain NaN/Inf values")
    fmin, fmax = int(faces.min()), int(faces.max())
    if fmin < 0 or fmax >= vertices.shape[0]:
        raise ValueError(f"{where}: face indices out of range [0, {vertices.shape[0]}) "
                         f"(min {fmin}, max {fmax})")


def _validate_resolution(value, name, where):
    value = int(value)
    if value < 1 or value > MAX_RESOLUTION:
        raise ValueError(f"{where}: {name} must be in [1, {MAX_RESOLUTION}], got {value}")
    return value


class Cameras:
    """Orbit-camera set: per-view elevation/azimuth (degrees) plus bake weights,
    with the fixed orthographic intrinsics of the released paint model."""

    def __init__(self, elevs, azims, weights=None, ortho_scale=ORTHO_SCALE,
                 camera_distance=CAMERA_DISTANCE, near=NEAR, far=FAR):
        if len(elevs) != len(azims):
            raise ValueError(f"Cameras: {len(elevs)} elevations vs {len(azims)} azimuths")
        self.elevs = list(elevs)
        self.azims = list(azims)
        self.weights = list(weights) if weights is not None else [1.0] * len(elevs)
        self.ortho_scale = ortho_scale
        self.camera_distance = camera_distance
        self.near = near
        self.far = far

    def __len__(self):
        return len(self.elevs)


def standard_cameras(num_views=6):
    """The first ``num_views`` cameras of the standard 6-view set."""
    return Cameras(STANDARD_VIEW_ELEVS[:num_views], STANDARD_VIEW_AZIMS[:num_views],
                   STANDARD_VIEW_WEIGHTS[:num_views])


# ---------------------------------------------------------------------------
# Camera matrices (matches the reference camera conventions of
# hy3dpaint/DifferentiableRenderer/camera_utils.py)
# ---------------------------------------------------------------------------
def view_matrix(elev, azim, camera_distance=CAMERA_DISTANCE, device="cpu"):
    """World-to-camera matrix (4, 4) of a look-at orbit camera.

    The camera sits at radius ``camera_distance`` on the orbit given by
    ``elev``/``azim`` (degrees, Z-up world frame) and looks at the origin with
    world +Z as the up reference.
    """
    e = math.radians(elev)
    a = math.radians(azim)
    ce, se = math.cos(e), math.sin(e)
    ca, sa = math.cos(a), math.sin(a)
    pos = torch.tensor([-ce * sa, ce * ca, -se], dtype=torch.float32, device=device)
    pos = pos * camera_distance
    forward = F.normalize(-pos, dim=0)
    if abs(ce) < 1e-8:
        # forward is parallel to the +Z up reference; the limit of the look-at
        # basis as elevation approaches +/-90 degrees is this right vector.
        right = torch.tensor([-ca, -sa, 0.0], dtype=torch.float32, device=device)
    else:
        up_ref = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
        right = F.normalize(torch.linalg.cross(forward, up_ref), dim=0)
    up = torch.linalg.cross(right, forward)

    w2c = torch.eye(4, dtype=torch.float32, device=device)
    w2c[0, :3] = right
    w2c[1, :3] = up
    w2c[2, :3] = -forward
    w2c[:3, 3] = -w2c[:3, :3] @ pos
    return w2c


def orthographic_matrix(ortho_scale=ORTHO_SCALE, near=NEAR, far=FAR, device="cpu"):
    """Symmetric OpenGL-style orthographic projection (4, 4): a world length of
    ``ortho_scale`` spans the full [-1, 1] NDC range in x and y, and camera-space
    depth in [near, far] maps to [-1, 1]."""
    proj = torch.eye(4, dtype=torch.float32, device=device)
    proj[0, 0] = 2.0 / ortho_scale
    proj[1, 1] = 2.0 / ortho_scale
    proj[2, 2] = -2.0 / (far - near)
    proj[2, 3] = -(far + near) / (far - near)
    return proj


# ---------------------------------------------------------------------------
# Mesh normalization (matches MeshRender.set_mesh auto_center path)
# ---------------------------------------------------------------------------
def normalize_mesh(vertices, scale_factor=MESH_SCALE_FACTOR):
    """Canonicalize glTF-frame vertices (N, 3) for the paint renderer.

    Applies the fixed frame change (x, y, z) -> (-x, z, -y), centers the
    axis-aligned bounding box at the origin, and uniformly scales so the
    bounding-sphere diameter equals ``scale_factor``. Computed in fp32.
    """
    v = vertices.to(torch.float32)
    v = torch.stack([-v[:, 0], v[:, 2], -v[:, 1]], dim=1)
    v = v - (v.amin(dim=0) + v.amax(dim=0)) * 0.5
    radius = v.norm(dim=1).max().clamp(min=1e-8)
    return v * (scale_factor / (2.0 * radius))


def face_normals(vertices, faces):
    """Per-face unit normals (F, 3)."""
    tri = vertices[faces.long()]
    n = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return F.normalize(n, dim=-1)


# ---------------------------------------------------------------------------
# Rasterizer
# ---------------------------------------------------------------------------
def _edge(ax, ay, bx, by, px, py):
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def rasterize(verts_ndc, faces, height, width, chunk_area=4_000_000):
    """Barycentric z-buffer rasterizer.

    Args:
        verts_ndc: (N, 3) vertices in normalized device coords (x, y in [-1, 1],
            z is depth with smaller = nearer).
        faces: (F, 3) long triangle indices.
        height, width: output resolution.

    Returns:
        face_id: (H, W) long, winning triangle per pixel (-1 = background).
        bary: (H, W, 3) barycentric weights of the winning triangle (0 on background).
    """
    device = verts_ndc.device
    faces = faces.long()
    Fn = faces.shape[0]
    npix = height * width
    face_id = torch.full((height, width), -1, dtype=torch.long, device=device)
    bary = torch.zeros((height, width, 3), dtype=torch.float32, device=device)
    if Fn == 0:
        return face_id, bary

    # Vertex screen coordinates (pixel space). Matches Tencent's custom_rasterizer
    # (rasterizer.cpp barycentricFromImgcoordCPU): row grows with NDC +y, i.e. NO
    # top/bottom flip. This looks unusual (a "physically up" point ends up in the
    # *last* row of the raw H,W buffer) but the released paint UNet was trained on
    # conditioning images produced with exactly this convention, and render/bake use
    # the same rasterize() for both the geometry-map render and the back-projection,
    # so the mesh<->UV correspondence is self-consistent either way. What matters is
    # matching the pretrained weights' expected pixel layout, not "looking upright"
    # when this raw buffer is viewed directly.
    sx = (verts_ndc[:, 0] * 0.5 + 0.5) * width
    sy = (verts_ndc[:, 1] * 0.5 + 0.5) * height
    sz = verts_ndc[:, 2]

    fx = sx[faces]  # (F, 3)
    fy = sy[faces]
    fz = sz[faces]

    xmin = torch.clamp(torch.floor(fx.amin(dim=1)).long(), 0, width - 1)
    xmax = torch.clamp(torch.ceil(fx.amax(dim=1)).long(), 0, width - 1)
    ymin = torch.clamp(torch.floor(fy.amin(dim=1)).long(), 0, height - 1)
    ymax = torch.clamp(torch.ceil(fy.amax(dim=1)).long(), 0, height - 1)
    bw = (xmax - xmin + 1).clamp(min=0)
    bh = (ymax - ymin + 1).clamp(min=0)
    area = bw * bh

    sentinel = torch.iinfo(torch.int64).max
    key_buf = torch.full((npix,), sentinel, dtype=torch.int64, device=device)

    # split faces into contiguous chunks bounded by cumulative bbox pixel area
    budget = max(int(chunk_area), 1)
    excl = torch.cumsum(area, 0) - area
    chunk_id = torch.div(excl, budget, rounding_mode="floor")
    n_chunks = int(chunk_id[-1].item()) + 1 if Fn > 0 else 0

    # chunk_id comes from a cumulative sum, so it is non-decreasing and every chunk is a
    # contiguous face range. Locating the boundaries once keeps this loop linear in Fn;
    # testing chunk_id == c per chunk would rescan all faces n_chunks times.
    bounds = torch.searchsorted(chunk_id, torch.arange(n_chunks + 1, device=device)).tolist()

    for c in range(n_chunks):
        start, end = bounds[c], bounds[c + 1]
        if end <= start:
            continue
        a_s = area[start:end]
        total = int(a_s.sum().item())
        if total == 0:
            continue
        bw_s = bw[start:end]
        # per-sample -> local face index within the chunk
        local = torch.repeat_interleave(torch.arange(end - start, device=device), a_s)
        off = torch.arange(total, device=device) - (torch.cumsum(a_s, 0) - a_s)[local]
        lw = bw_s[local]
        px = xmin[start:end][local] + (off % lw)
        py = ymin[start:end][local] + torch.div(off, lw, rounding_mode="floor")
        fg = start + local  # global face index per sample

        cx = px.to(torch.float32) + 0.5
        cy = py.to(torch.float32) + 0.5
        ax, ay = fx[fg, 0], fy[fg, 0]
        bx, by = fx[fg, 1], fy[fg, 1]
        cx2, cy2 = fx[fg, 2], fy[fg, 2]

        area2 = _edge(ax, ay, bx, by, cx2, cy2)
        w0 = _edge(bx, by, cx2, cy2, cx, cy)
        w1 = _edge(cx2, cy2, ax, ay, cx, cy)
        w2 = _edge(ax, ay, bx, by, cx, cy)
        denom = torch.where(area2.abs() < 1e-9, torch.ones_like(area2), area2)
        w0 = w0 / denom
        w1 = w1 / denom
        w2 = w2 / denom
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0) & (area2.abs() >= 1e-9)

        depth = w0 * fz[fg, 0] + w1 * fz[fg, 1] + w2 * fz[fg, 2]
        dq = ((depth * 0.5 + 0.5).clamp(0.0, 1.0) * (_DEPTH_LEVELS - 1)).long()
        key = dq * Fn + fg

        pix = (py * width + px)[inside]
        key_i = key[inside]
        if pix.numel() > 0:
            key_buf.scatter_reduce_(0, pix, key_i, reduce="amin", include_self=True)

    valid = key_buf != sentinel
    fid_flat = torch.where(valid, key_buf % Fn, torch.full_like(key_buf, -1))
    face_id = fid_flat.view(height, width)

    # recompute barycentric weights for the winning face at each covered pixel
    idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
    if idx.numel() > 0:
        fsel = fid_flat[idx]
        row = torch.div(idx, width, rounding_mode="floor")
        col = idx % width
        cx = col.to(torch.float32) + 0.5
        cy = row.to(torch.float32) + 0.5
        ax, ay = fx[fsel, 0], fy[fsel, 0]
        bx, by = fx[fsel, 1], fy[fsel, 1]
        cx2, cy2 = fx[fsel, 2], fy[fsel, 2]
        area2 = _edge(ax, ay, bx, by, cx2, cy2)
        denom = torch.where(area2.abs() < 1e-9, torch.ones_like(area2), area2)
        w0 = _edge(bx, by, cx2, cy2, cx, cy) / denom
        w1 = _edge(cx2, cy2, ax, ay, cx, cy) / denom
        w2 = _edge(ax, ay, bx, by, cx, cy) / denom
        bary.view(-1, 3)[idx] = torch.stack([w0, w1, w2], dim=-1)

    return face_id, bary


def interpolate_vertex_attr(attr, faces, face_id, bary):
    """Barycentric-interpolate a per-vertex attribute (N, C) into an (H, W, C) map."""
    H, W = face_id.shape
    C = attr.shape[-1]
    out = torch.zeros((H, W, C), dtype=attr.dtype, device=attr.device)
    valid = face_id >= 0
    if valid.any():
        f = faces.long()[face_id.clamp(min=0)]  # (H, W, 3)
        a = attr[f]  # (H, W, 3, C)
        out = (bary.unsqueeze(-1) * a).sum(dim=-2)
        out = out * valid.unsqueeze(-1)
    return out


def gather_face_attr(attr_face, face_id):
    """Gather a per-face attribute (F, C) into an (H, W, C) map (0 on background)."""
    H, W = face_id.shape
    out = attr_face[face_id.clamp(min=0)]
    out = out * (face_id >= 0).unsqueeze(-1)
    return out


# ---------------------------------------------------------------------------
# Geometry conditioning maps for the paint UNet
# ---------------------------------------------------------------------------
def _project(verts_norm, w2c, proj):
    homog = torch.cat([verts_norm, torch.ones_like(verts_norm[:, :1])], dim=1)
    cam = homog @ w2c.t()
    clip = cam @ proj.t()
    ndc = clip[:, :3] / clip[:, 3:4].clamp(min=1e-8)
    return ndc, cam


def render_geometry_maps(vertices, faces, cameras, resolution=512, scale_factor=MESH_SCALE_FACTOR,
                         normalize=True):
    """Render per-view object-space normal + position maps (the paint UNet's control).

    Args:
        vertices: (N, 3) mesh vertices in the input (GLB) frame.
        faces: (F, 3) triangle indices.
        cameras: a Cameras object.
        resolution: output map resolution.

    Returns:
        normals: (V, H, W, 3) in [0, 1] with a white background.
        positions: (V, H, W, 3) in [0, 1] with a white background.
        masks: (V, H, W) float, 1 where the mesh is visible.
    """
    _validate_mesh(vertices, faces, "render_geometry_maps")
    resolution = _validate_resolution(resolution, "resolution", "render_geometry_maps")
    device = vertices.device
    vtx = normalize_mesh(vertices, scale_factor) if normalize else vertices.to(torch.float32)
    faces = faces.long().to(device)
    fn = face_normals(vtx, faces)  # object-space flat normals
    tex_position = 0.5 - vtx / scale_factor

    proj = orthographic_matrix(cameras.ortho_scale, cameras.near, cameras.far, device=device)
    normals, positions, masks = [], [], []
    for elev, azim in zip(cameras.elevs, cameras.azims):
        w2c = view_matrix(elev, azim, cameras.camera_distance, device=device)
        ndc, _ = _project(vtx, w2c, proj)
        face_id, bary = rasterize(ndc, faces, resolution, resolution)
        mask = (face_id >= 0).to(torch.float32)

        nrm = gather_face_attr(fn, face_id)  # (H, W, 3) object-space normal
        nrm = (nrm + 1.0) * 0.5
        nrm = nrm * mask.unsqueeze(-1) + (1.0 - mask.unsqueeze(-1))  # white background

        pos = interpolate_vertex_attr(tex_position, faces, face_id, bary)
        pos = pos * mask.unsqueeze(-1) + (1.0 - mask.unsqueeze(-1))  # white background

        normals.append(nrm)
        positions.append(pos)
        masks.append(mask)

    return torch.stack(normals), torch.stack(positions), torch.stack(masks)


# ---------------------------------------------------------------------------
# Per-triangle UV atlas (xatlas-free unwrap)
# ---------------------------------------------------------------------------
def pack_per_triangle_uv(vertices, faces, gutter=0.25):
    """Unwrap a mesh by giving each triangle its own cell in a square UV atlas.

    The mesh is "unwelded" (3 unique vertices per face) so UVs are per-vertex and
    1:1 with vertices, as core's MESH/save_glb require. Charts are guaranteed
    non-overlapping (one per grid cell) and every triangle is packed.

    Returns:
        new_vertices: (3F, 3)
        new_faces: (F, 3) long, referencing the unwelded vertices
        uvs: (3F, 2) in [0, 1]
    """
    _validate_mesh(vertices, faces, "pack_per_triangle_uv")
    faces = faces.long()
    Fn = faces.shape[0]
    device = vertices.device
    corners = vertices[faces.reshape(-1)]  # (3F, 3)
    new_faces = torch.arange(3 * Fn, device=device, dtype=torch.long).reshape(Fn, 3)

    g = max(1, int(math.ceil(math.sqrt(Fn))))
    fi = torch.arange(Fn, device=device)
    row = torch.div(fi, g, rounding_mode="floor")
    col = fi % g
    cell = 1.0 / g
    m = gutter * cell  # gutter keeps charts off the cell border to avoid bleed
    u0 = col * cell + m
    v0 = row * cell + m
    u1 = (col + 1) * cell - m
    v1 = (row + 1) * cell - m
    # a right triangle (lower-left) filling most of the cell
    uv = torch.stack([
        torch.stack([u0, v0], dim=-1),
        torch.stack([u1, v0], dim=-1),
        torch.stack([u0, v1], dim=-1),
    ], dim=1).reshape(3 * Fn, 2).to(torch.float32)
    return corners, new_faces, uv


# ---------------------------------------------------------------------------
# Multiview back-projection / bake
# ---------------------------------------------------------------------------
def _splat_bilinear(coords01, values, sample_weight, tex_h, tex_w, color_acc, weight_acc):
    """Bilinearly splat (K, C) values with per-sample weights (K, 1) into UV accumulators.

    coords01 are (K, 2) [row, col] fractional coords in [0, 1]. Each texel receives
    ``corner * sample_weight * value`` in ``color_acc`` and ``corner * sample_weight``
    in ``weight_acc`` so a final divide yields the angle-weighted colour average.
    """
    idx = coords01 * torch.tensor([tex_h - 1, tex_w - 1], device=coords01.device, dtype=coords01.dtype)
    i0 = idx.floor().long()
    i0[:, 0].clamp_(0, tex_h - 2)
    i0[:, 1].clamp_(0, tex_w - 2)
    fr = idx - i0.to(idx.dtype)
    fh, fw = fr[:, 0:1], fr[:, 1:2]
    C = values.shape[1]
    for di, dj, corner in ((0, 0, (1 - fh) * (1 - fw)), (0, 1, (1 - fh) * fw),
                           (1, 0, fh * (1 - fw)), (1, 1, fh * fw)):
        flat = (i0[:, 0] + di) * tex_w + (i0[:, 1] + dj)
        cw = corner * sample_weight
        color_acc.scatter_add_(0, flat.unsqueeze(1).expand(-1, C), values * cw)
        weight_acc.scatter_add_(0, flat.unsqueeze(1), cw)


def bake_multiview(vertices, faces, uvs, views, cameras, texture_size=1024,
                   scale_factor=MESH_SCALE_FACTOR, normalize=True,
                   bake_exp=4.0, cos_thresh_deg=75.0):
    """Back-project multiview images onto the UV atlas via angle-weighted blending.

    Mirrors the reference pipeline's bake: each view is rasterized with the z-buffer
    (so occluded surfaces never receive that view's colors) and every covered pixel
    is splatted into UV space with weight ``view_weight * cos(view angle)**bake_exp``
    (the reference's per-view weights and ``bake_exp=4``), then the per-texel
    weighted average is taken across all views.

    Args:
        vertices: (N, 3) vertices (unwelded, carrying the UV atlas) in the GLB frame.
        faces: (F, 3) triangle indices.
        uvs: (N, 2) per-vertex UVs in [0, 1].
        views: (V, H, W, C) per-view images in [0, 1] (same camera order as cameras).
        cameras: Cameras used to produce views.
        texture_size: output texture resolution.

    Returns:
        texture: (texture_size, texture_size, C) baked texture (holes left at 0).
        mask: (texture_size, texture_size) bool, True where a view contributed.
    """
    _validate_mesh(vertices, faces, "bake_multiview")
    T = _validate_resolution(texture_size, "texture_size", "bake_multiview")
    if uvs.ndim != 2 or uvs.shape[0] != vertices.shape[0] or uvs.shape[-1] != 2:
        raise ValueError(f"bake_multiview: uvs must be ({vertices.shape[0]}, 2) - one per "
                         f"vertex - got {tuple(uvs.shape)}")
    device = vertices.device
    vtx = normalize_mesh(vertices, scale_factor) if normalize else vertices.to(torch.float32)
    faces = faces.long().to(device)
    uvs = uvs.to(device=device, dtype=torch.float32)
    views = views.to(device)
    V, vh, vw, C = views.shape
    cos_thresh = math.cos(math.radians(cos_thresh_deg))

    proj = orthographic_matrix(cameras.ortho_scale, cameras.near, cameras.far, device=device)
    color_acc = torch.zeros((T * T, C), dtype=torch.float32, device=device)
    weight_acc = torch.zeros((T * T, 1), dtype=torch.float32, device=device)

    if V != len(cameras.elevs) or V != len(cameras.azims):
        raise ValueError(
            f"bake_multiview: {V} view images but "
            f"{len(cameras.elevs)} elevations / {len(cameras.azims)} azimuths; "
            "views and cameras must match one-to-one")

    for vi in range(V):
        elev, azim = cameras.elevs[vi], cameras.azims[vi]
        vw_weight = cameras.weights[vi] if vi < len(cameras.weights) else 1.0
        w2c = view_matrix(elev, azim, cameras.camera_distance, device=device)
        ndc, cam = _project(vtx, w2c, proj)
        face_id, bary = rasterize(ndc, faces, vh, vw)
        valid = face_id >= 0
        if not valid.any():
            continue

        # camera-space flat normal -> facing cosine to the view axis. The z-buffer already
        # keeps only the front-most surface, so |z-component| is the grazing cosine
        # irrespective of triangle winding.
        fn_cam = face_normals(cam[:, :3], faces)
        cos_face = fn_cam[:, 2].abs()
        cos_map = gather_face_attr(cos_face.unsqueeze(-1), face_id).squeeze(-1)  # (H, W)
        cos_map = torch.where(cos_map < cos_thresh, torch.zeros_like(cos_map), cos_map)

        uv_map = interpolate_vertex_attr(uvs, faces, face_id, bary)  # (H, W, 2)

        weight = (vw_weight * cos_map.pow(bake_exp)) * valid.to(torch.float32)
        sel = weight > 0
        if not sel.any():
            continue
        w_sel = weight[sel].unsqueeze(-1)
        colors = views[vi][sel]
        # UV (u, v) -> texel (row = v, col = u)
        coords = torch.stack([uv_map[sel][:, 1], uv_map[sel][:, 0]], dim=-1).clamp(0.0, 1.0)
        _splat_bilinear(coords, colors, w_sel, T, T, color_acc, weight_acc)

    mask = (weight_acc.squeeze(-1) > 1e-8)
    texture = torch.zeros_like(color_acc)
    texture[mask] = color_acc[mask] / weight_acc[mask].clamp(min=1e-8)
    return texture.view(T, T, C), mask.view(T, T)


def _dilate_valid(color, valid, iters):
    """Grow valid regions by ``iters`` 3x3 dilation passes (gutter fill).

    color is (1, C, T, T), valid is (1, 1, T, T) in {0, 1}. Hole texels adjacent to
    valid ones take the average of their valid neighbours; original valid texels are
    never modified. Returns the updated (color, valid).
    """
    C = color.shape[1]
    kernel = torch.ones((1, 1, 3, 3), device=color.device, dtype=torch.float32)
    for _ in range(iters):
        holes = valid < 0.5
        if not holes.any():
            break
        neigh_w = F.conv2d(valid, kernel, padding=1)
        neigh_c = F.conv2d(color * valid, kernel.expand(C, 1, 3, 3), padding=1, groups=C)
        can_fill = holes & (neigh_w > 0)
        filled = neigh_c / neigh_w.clamp(min=1e-8)
        color = torch.where(can_fill.expand(-1, C, -1, -1), filled, color)
        valid = torch.where(can_fill, torch.ones_like(valid), valid)
    return color, valid


def _push_pull(color, valid):
    """Fill every remaining hole texel with a push-pull image pyramid.

    Pull: average premultiplied color + coverage down to 1x1. Push: walk back up,
    keeping (partially) covered texels and filling the rest from the coarser level,
    so arbitrarily large unseen regions inherit plausible low-frequency colors from
    the nearest covered surface. Texels that were valid at full resolution are
    returned exactly unchanged.
    """
    levels = []
    c = color * valid
    w = valid
    while min(c.shape[-2:]) > 1:
        levels.append((c, w))
        c = F.avg_pool2d(c, 2, ceil_mode=True)
        w = F.avg_pool2d(w, 2, ceil_mode=True)
    out = torch.where(w > 0, c / w.clamp(min=1e-8), torch.zeros_like(c))
    for cf, wf in reversed(levels):
        up = F.interpolate(out, size=cf.shape[-2:], mode="bilinear", align_corners=False)
        a = wf.clamp(0.0, 1.0)
        out = a * torch.where(wf > 0, cf / wf.clamp(min=1e-8), torch.zeros_like(cf)) + (1.0 - a) * up
    return out


def fill_holes(texture, mask, dilate_iters=8):
    """UV-space inpaint of texels no view contributed to.

    Two stages, mirroring the reference pipeline's UV-space texture inpaint:
    first ``dilate_iters`` gutter-dilation passes extend chart borders outward with
    their exact nearest colors (protecting bilinear/mipmap sampling across seams),
    then a push-pull pyramid fills every remaining texel — unseen interior regions
    of any size and the atlas background — with low-frequency colors from the
    nearest covered areas. Valid texels are returned unchanged.
    """
    T, _, C = texture.shape
    color = texture.permute(2, 0, 1).unsqueeze(0).clone()  # (1, C, T, T)
    valid = mask.to(torch.float32).view(1, 1, T, T).clone()
    color, valid = _dilate_valid(color, valid, dilate_iters)
    color = _push_pull(color, valid)
    return color.squeeze(0).permute(1, 2, 0).contiguous()
