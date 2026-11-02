"""Convention armor for the Hunyuan3D 2.1 paint renderer/baker.

A procedural "chirality cube" fixture (no binary assets) exercises every
coordinate-convention class the render->diffuse->bake pipeline depends on:

  AXIS-MAPPING/CAMERA-SIGN  which glTF axis face each standard view sees
  WINDING/BACKFACE          the normal-map color convention the UNet was trained on
  OCCLUSION/Z-BUFFER        hidden surfaces receive no view color
  CHIRALITY/MIRROR          an extruded chiral 'F' reads unmirrored in view space
  UV-ORIGIN/V-FLIP          the baked texture lands in the authored UV region,
                            unflipped, with the F shadow oriented correctly

Every assertion message names its convention class and the likely code location.
A mutation self-test injects each convention error (V flip, axis negation,
winding reversal, camera-sign swap) and proves the suite fails for each one -
evidence the armor actually catches its classes.

The fixture is a GLB-frame (Y-up, front = +Z) cube with distinct per-axis-face
colors, an extruded chiral 'F' on the +Z face, an oversized limb on +X, and
authored UVs mapping the +Z face to a known texture window. Deterministic,
120 triangles, all CPU.
"""

from __future__ import annotations

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy.ldm.hunyuan3d.paint import render as R  # noqa: E402

# face classes
CLS_PX, CLS_NX, CLS_PY, CLS_NY, CLS_PZ, CLS_NZ, CLS_F, CLS_LIMB = range(8)

CLASS_COLORS = torch.tensor([
    [0.9, 0.1, 0.1],  # +X red
    [0.1, 0.9, 0.9],  # -X cyan
    [0.1, 0.9, 0.1],  # +Y green
    [0.9, 0.1, 0.9],  # -Y magenta
    [0.1, 0.1, 0.9],  # +Z blue
    [0.9, 0.9, 0.1],  # -Z yellow
    [1.0, 0.5, 0.0],  # F  orange
    [0.6, 0.6, 0.6],  # limb grey
])

# the +Z face is authored into this UV window; everything else is parked at PARK_UV
FACE_UV_MIN, FACE_UV_SPAN = 0.05, 0.5
PARK_UV = (0.85, 0.85)


def _quad3d(c0, c1, c2, c3, cls):
    """One outward-wound quad -> (4 verts, 2 tris, per-tri class)."""
    v = torch.tensor([c0, c1, c2, c3], dtype=torch.float32)
    f = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    return v, f, torch.tensor([cls, cls], dtype=torch.long)


def _box(cx, cy, cz, hx, hy, hz, cls):
    """Axis-aligned box, outward winding, all 12 tris tagged ``cls``."""
    v = torch.tensor([
        [cx - hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz + hz],
        [cx + hx, cy + hy, cz + hz], [cx - hx, cy + hy, cz + hz],
    ], dtype=torch.float32)
    f = torch.tensor([
        [0, 3, 2], [0, 2, 1],  # -Z
        [4, 5, 6], [4, 6, 7],  # +Z
        [0, 1, 5], [0, 5, 4],  # -Y
        [3, 7, 6], [3, 6, 2],  # +Y
        [0, 4, 7], [0, 7, 3],  # -X
        [1, 2, 6], [1, 6, 5],  # +X
    ], dtype=torch.long)
    return v, f, torch.full((12,), cls, dtype=torch.long)


def make_chirality_cube():
    """Procedural convention fixture in the glTF frame (Y-up, front = +Z).

    Returns (vertices (N,3), faces (F,3), face_class (F,), face_colors (F,3),
    uvs (N,2)). ~120 deterministic triangles:
      - unit cube with per-axis-face colors (unwelded per face),
      - extruded chiral 'F' (8 prisms) on the +Z face: stem toward -X, arms
        toward +X, long arm toward +Y,
      - oversized limb protruding from the +X face,
      - +Z face UVs authored into [0.05, 0.55]^2 (u tracks +X, v tracks +Y);
        every other vertex parked at a constant far-away UV.
    """
    h = 0.5
    quads = [
        _quad3d([-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h], CLS_PZ),
        _quad3d([h, -h, -h], [-h, -h, -h], [-h, h, -h], [h, h, -h], CLS_NZ),
        _quad3d([h, -h, h], [h, -h, -h], [h, h, -h], [h, h, h], CLS_PX),
        _quad3d([-h, -h, -h], [-h, -h, h], [-h, h, h], [-h, h, -h], CLS_NX),
        _quad3d([-h, h, h], [h, h, h], [h, h, -h], [-h, h, -h], CLS_PY),
        _quad3d([-h, -h, -h], [h, -h, -h], [h, -h, h], [-h, -h, h], CLS_NY),
    ]
    # chiral F on the +Z face: 3x5 cell grid, stem on the -X column, arms to +X,
    # the long arm on the +Y (top) row. Cells: (col, row) with row 0 at +Y.
    pitch, half_cell = 0.14, 0.065
    f_cells = [(0, 0), (1, 0), (2, 0), (0, 1), (0, 2), (1, 2), (0, 3), (0, 4)]
    boxes = []
    for col, row in f_cells:
        x = -0.30 + col * pitch
        y = 0.28 - row * pitch
        boxes.append(_box(x, y, 0.55, half_cell, half_cell, 0.07, CLS_F))
    # oversized limb out of +X (base sunk into the body to avoid coplanar faces)
    boxes.append(_box(0.84, 0.0, 0.0, 0.36, 0.15, 0.15, CLS_LIMB))

    verts, faces, cls, uvs = [], [], [], []
    base = 0
    for v, f, c in quads + boxes:
        verts.append(v)
        faces.append(f + base)
        cls.append(c)
        if int(c[0]) == CLS_PZ:
            u = FACE_UV_MIN + FACE_UV_SPAN * (v[:, 0] + h)
            w = FACE_UV_MIN + FACE_UV_SPAN * (v[:, 1] + h)
            uvs.append(torch.stack([u, w], dim=-1))
        else:
            uvs.append(torch.tensor(PARK_UV).expand(v.shape[0], 2).clone())
        base += v.shape[0]

    vertices = torch.cat(verts)
    faces = torch.cat(faces)
    face_class = torch.cat(cls)
    face_colors = CLASS_COLORS[face_class]
    return vertices, faces, face_class, face_colors, torch.cat(uvs)


# --------------------------------------------------------------------------- #
# Shared rendering helpers (same code path the pipeline uses)
# --------------------------------------------------------------------------- #
RES = 128
TEX = 128


def _view_face_ids(vertices, faces, cameras, res=RES):
    vtx = R.normalize_mesh(vertices)
    proj = R.orthographic_matrix(cameras.ortho_scale, cameras.near, cameras.far)
    fids = []
    for elev, azim in zip(cameras.elevs, cameras.azims):
        w2c = R.view_matrix(elev, azim, cameras.camera_distance)
        ndc, _ = R._project(vtx, w2c, proj)
        fid, _ = R.rasterize(ndc, faces, res, res)
        fids.append(fid)
    return fids


def _class_map(fid, face_class):
    out = torch.full_like(fid, -1)
    covered = fid >= 0
    out[covered] = face_class[fid[covered]]
    return out


def _fixture(mutation=None):
    vertices, faces, face_class, face_colors, uvs = make_chirality_cube()
    cameras = R.standard_cameras(6)
    if mutation == "negate_x":
        vertices = vertices.clone()
        vertices[:, 0] = -vertices[:, 0]
    elif mutation == "reverse_winding":
        faces = faces[:, [0, 2, 1]]
    elif mutation == "flip_v":
        uvs = uvs.clone()
        uvs[:, 1] = 1.0 - uvs[:, 1]
    elif mutation == "swap_camera_sign":
        cameras = R.Cameras(cameras.elevs, [-a for a in cameras.azims], cameras.weights)
    elif mutation is not None:
        raise ValueError(mutation)
    return vertices, faces, face_class, face_colors, uvs, cameras


# --------------------------------------------------------------------------- #
# Assertion groups (each names its convention class + likely location)
# --------------------------------------------------------------------------- #
def _assert_view_axis_faces(vertices, faces, face_class, cameras):
    expected = [
        ("front", CLS_PZ, "+Z"), ("right", CLS_PX, "+X"), ("back", CLS_NZ, "-Z"),
        ("left", CLS_NX, "-X"), ("top", CLS_PY, "+Y"), ("bottom", CLS_NY, "-Y"),
    ]
    fids = _view_face_ids(vertices, faces, cameras)
    for i, (name, want, axis) in enumerate(expected):
        cmap = _class_map(fids[i], face_class)
        counts = torch.bincount(cmap[(cmap >= 0) & (cmap < 6)], minlength=6)
        got = int(counts.argmax())
        assert got == want, (
            f"AXIS-MAPPING/CAMERA-SIGN: the {name} view (elev={cameras.elevs[i]}, "
            f"azim={cameras.azims[i]}) must be dominated by the glTF {axis} face, "
            f"saw class {got} (counts={counts.tolist()}); check normalize_mesh() axis "
            f"swaps or view_matrix() elev/azim signs in comfy/ldm/hunyuan3d/paint/render.py")
    return fids


def _assert_normal_convention(vertices, faces, face_class, cameras):
    front = R.Cameras(cameras.elevs[:1], cameras.azims[:1])
    normals, _positions, _masks = R.render_geometry_maps(vertices, faces, front, resolution=RES)
    fid = _view_face_ids(vertices, faces, front)[0]
    sel = _class_map(fid, face_class) == CLS_PZ
    mean = normals[0][sel].mean(dim=0)
    ok = (abs(float(mean[0]) - 0.5) < 0.15 and float(mean[1]) < 0.15
          and abs(float(mean[2]) - 0.5) < 0.15)
    assert ok, (
        f"WINDING/BACKFACE: the front-view normal map over the glTF +Z face must encode "
        f"the reference convention ~[0.5, 0.0, 0.5] (outward glTF winding through the "
        f"det=-1 normalize_mesh axis map), got {[round(float(x), 3) for x in mean]}; "
        f"triangle winding is reversed or the normal transform changed - check "
        f"face_normals()/normalize_mesh() in comfy/ldm/hunyuan3d/paint/render.py")


def _assert_limb_occlusion(vertices, faces, face_class, cameras):
    fids = _view_face_ids(vertices, faces, cameras)
    front, right, left = fids[0], fids[1], fids[3]
    left_limb = int((_class_map(left, face_class) == CLS_LIMB).sum())
    right_limb = int((_class_map(right, face_class) == CLS_LIMB).sum())
    assert left_limb < 10, (
        f"OCCLUSION/Z-BUFFER: the +X limb must be fully hidden behind the body in the "
        f"left view but {left_limb} limb pixels are visible; the depth comparison or "
        f"camera sign is flipped - check rasterize() depth keys or view_matrix() in "
        f"comfy/ldm/hunyuan3d/paint/render.py")
    assert right_limb > 150, (
        f"OCCLUSION/Z-BUFFER: the +X limb must face the right view head-on but only "
        f"{right_limb} limb pixels are visible; check view_matrix()/rasterize() in "
        f"comfy/ldm/hunyuan3d/paint/render.py")
    limb_cols = torch.nonzero(_class_map(front, face_class) == CLS_LIMB)[:, 1].float()
    assert limb_cols.numel() > 0 and float(limb_cols.mean()) > 0.55 * RES, (
        f"CAMERA-SIGN/HANDEDNESS: the +X limb must appear on the right half of the "
        f"front view (mean col {float(limb_cols.mean()) if limb_cols.numel() else 'n/a'} "
        f"of {RES}); screen-x no longer tracks glTF +X - check view_matrix() "
        f"right/up construction in comfy/ldm/hunyuan3d/paint/render.py")


def _assert_f_chirality_in_view(vertices, faces, face_class, cameras):
    front = R.Cameras(cameras.elevs[:1], cameras.azims[:1])
    fid = _view_face_ids(vertices, faces, front)[0]
    mask = _class_map(fid, face_class) == CLS_F
    pix = torch.nonzero(mask).float()
    assert pix.numel() > 0, "CHIRALITY/MIRROR: the extruded F is not visible in the front view"
    rows, cols = pix[:, 0], pix[:, 1]
    rmid = (rows.min() + rows.max()) / 2
    cmid = (cols.min() + cols.max()) / 2
    top, bottom = int((rows < rmid).sum()), int((rows > rmid).sum())
    left, right = int((cols < cmid).sum()), int((cols > cmid).sum())
    assert top > 1.3 * bottom, (
        f"CHIRALITY/MIRROR: the F's long arm must sit at the top of the front view "
        f"(buffer row 0 = glTF +Y): top mass {top} vs bottom {bottom}; the view is "
        f"vertically flipped - check rasterize() row convention or view_matrix() up "
        f"vector in comfy/ldm/hunyuan3d/paint/render.py")
    assert left > 1.3 * right, (
        f"CHIRALITY/MIRROR: the F's stem must sit on the left of the front view "
        f"(screen-x tracks glTF +X): left mass {left} vs right {right}; the view is "
        f"mirrored - check normalize_mesh() axis negations or view_matrix() in "
        f"comfy/ldm/hunyuan3d/paint/render.py")


def _assert_baked_f_in_uv_window(vertices, faces, face_class, face_colors, uvs, cameras):
    front = R.Cameras(cameras.elevs[:1], cameras.azims[:1])
    fid = _view_face_ids(vertices, faces, front)[0]
    view = torch.zeros((RES, RES, 3))
    covered = fid >= 0
    view[covered] = face_colors[fid[covered]]
    tex, mask = R.bake_multiview(vertices, faces, uvs, view.unsqueeze(0), front,
                                 texture_size=TEX)
    lo = round((FACE_UV_MIN + 0.03) * (TEX - 1))
    hi = round((FACE_UV_MIN + FACE_UV_SPAN - 0.03) * (TEX - 1))
    win_mask = mask[lo:hi + 1, lo:hi + 1]
    win_tex = tex[lo:hi + 1, lo:hi + 1]
    coverage = float(win_mask.float().mean())
    assert coverage > 0.5, (
        f"UV-ORIGIN/V-FLIP: the authored +Z-face UV window [{FACE_UV_MIN}, "
        f"{FACE_UV_MIN + FACE_UV_SPAN}]^2 must be covered by the front-view bake "
        f"(coverage {coverage:.2f}); the bake's V convention no longer matches the "
        f"authored UVs - check the uv->texel mapping in bake_multiview() in "
        f"comfy/ldm/hunyuan3d/paint/render.py")
    mean = win_tex[win_mask].mean(dim=0)
    assert torch.allclose(mean, CLASS_COLORS[CLS_PZ], atol=0.1), (
        f"UV-ORIGIN/CHART: the +Z-face UV window must carry the +Z face color "
        f"{CLASS_COLORS[CLS_PZ].tolist()}, got {[round(float(x), 3) for x in mean]}; "
        f"another chart is landing in this window - check bake_multiview()/authored UVs")
    # the F prisms shadow the +Z face -> unbaked hole in the window shaped like the F.
    hole = torch.nonzero(~win_mask).float()
    assert hole.numel() > 0, "UV-ORIGIN: expected an F-shaped unbaked shadow in the +Z window"
    hrows, hcols = hole[:, 0], hole[:, 1]
    rmid = (hrows.min() + hrows.max()) / 2
    cmid = (hcols.min() + hcols.max()) / 2
    # v tracks glTF +Y and texel row = v * (T-1): the long arm sits at HIGH v (high row)
    high_v, low_v = int((hrows > rmid).sum()), int((hrows < rmid).sum())
    left_u, right_u = int((hcols < cmid).sum()), int((hcols > cmid).sum())
    assert high_v > 1.3 * low_v, (
        f"UV-ORIGIN/V-FLIP: the F shadow's long arm must sit at high V in the baked "
        f"texture (v tracks glTF +Y, texel row = v*(T-1)): high-v mass {high_v} vs "
        f"low-v {low_v}; a V flip was introduced between bake and the authored UVs - "
        f"check bake_multiview() in comfy/ldm/hunyuan3d/paint/render.py")
    assert left_u > 1.3 * right_u, (
        f"CHIRALITY/MIRROR: the F shadow's stem must sit at low U in the baked texture "
        f"(u tracks glTF +X): low-u mass {left_u} vs high-u {right_u}; the texture is "
        f"mirrored - check bake_multiview()/normalize_mesh() in "
        f"comfy/ldm/hunyuan3d/paint/render.py")


def _run_full_suite(mutation=None):
    vertices, faces, face_class, face_colors, uvs, cameras = _fixture(mutation)
    _assert_view_axis_faces(vertices, faces, face_class, cameras)
    _assert_normal_convention(vertices, faces, face_class, cameras)
    _assert_limb_occlusion(vertices, faces, face_class, cameras)
    _assert_f_chirality_in_view(vertices, faces, face_class, cameras)
    _assert_baked_f_in_uv_window(vertices, faces, face_class, face_colors, uvs, cameras)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_chirality_cube_fixture_is_deterministic():
    v1, f1, c1, col1, uv1 = make_chirality_cube()
    v2, f2, c2, col2, uv2 = make_chirality_cube()
    assert torch.equal(v1, v2) and torch.equal(f1, f2) and torch.equal(uv1, uv2)
    assert f1.shape[0] == 120  # 6 quads * 2 + 9 boxes * 12
    assert v1.shape[0] == 24 + 9 * 8
    assert int(f1.max()) < v1.shape[0]
    assert float(uv1.min()) >= 0.0 and float(uv1.max()) <= 1.0
    assert c1.shape[0] == f1.shape[0] and col1.shape == (f1.shape[0], 3)


def test_per_view_dominant_axis_face():
    vertices, faces, face_class, _colors, _uvs, cameras = _fixture()
    _assert_view_axis_faces(vertices, faces, face_class, cameras)


def test_winding_gives_reference_normal_colors():
    vertices, faces, face_class, _colors, _uvs, cameras = _fixture()
    _assert_normal_convention(vertices, faces, face_class, cameras)


def test_limb_occlusion_and_screen_side():
    vertices, faces, face_class, _colors, _uvs, cameras = _fixture()
    _assert_limb_occlusion(vertices, faces, face_class, cameras)


def test_extruded_f_reads_unmirrored_in_view():
    vertices, faces, face_class, _colors, _uvs, cameras = _fixture()
    _assert_f_chirality_in_view(vertices, faces, face_class, cameras)


def test_baked_f_lands_unflipped_in_uv_window():
    vertices, faces, face_class, face_colors, uvs, cameras = _fixture()
    _assert_baked_f_in_uv_window(vertices, faces, face_class, face_colors, uvs, cameras)


@pytest.mark.parametrize("mutation,expect", [
    ("flip_v", "UV-ORIGIN"),
    ("negate_x", "AXIS-MAPPING"),
    ("reverse_winding", "WINDING"),
    ("swap_camera_sign", "AXIS-MAPPING"),
])
def test_convention_armor_catches_injected_errors(mutation, expect):
    """Mutation self-test: each deliberately injected convention error must make the
    suite fail with the matching convention-class message."""
    with pytest.raises(AssertionError, match=expect):
        _run_full_suite(mutation)


def test_full_suite_passes_unmutated():
    _run_full_suite(None)
