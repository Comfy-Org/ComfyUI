"""Unit tests for the torch-native Hunyuan3D 2.1 paint renderer, UV unwrap and baker.

All CPU, no weights. Covers the barycentric z-buffer rasterizer (known triangle ->
known pixels, occlusion), the geometry conditioning maps, the per-triangle UV atlas
(non-overlapping charts, all triangles packed), the multiview baker (flat-colour
round-trip, MR channel preservation, per-view weight blending, cosine-exponent
weighting, grazing-angle cutoff), the two-stage hole fill (gutter dilation +
push-pull inpaint), and node schema/wiring including baking onto a mesh's existing
UVs and the textured-GLB write-back with a metallic-roughness texture.
"""

from __future__ import annotations

import inspect
import json
import math
import struct
from io import BytesIO

import numpy as np
import pytest
import torch
from PIL import Image

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy.ldm.hunyuan3d.paint import render as R  # noqa: E402


# --------------------------------------------------------------------------- #
# Rasterizer
# --------------------------------------------------------------------------- #
def test_rasterize_known_triangle_to_known_pixels():
    # NDC triangle -> screen triangle (0,0),(4,0),(0,4): row grows with NDC +y (no
    # top/bottom flip, matching Tencent's custom_rasterizer pixel convention), so this
    # covers pixels where col + row <= ~4 (dense near row 0, tapering by row 3).
    verts = torch.tensor([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]])
    faces = torch.tensor([[0, 1, 2]])
    face_id, bary = R.rasterize(verts, faces, 4, 4)
    expected = torch.tensor([
        [0, 0, 0, 0],
        [0, 0, 0, -1],
        [0, 0, -1, -1],
        [0, -1, -1, -1],
    ])
    assert torch.equal(face_id, expected)
    # barycentric weights of covered pixels sum to 1
    covered = face_id >= 0
    assert torch.allclose(bary[covered].sum(-1), torch.ones(int(covered.sum())), atol=1e-5)


def test_rasterize_barycentric_center():
    # A triangle whose centroid lands on a pixel center yields ~(1/3, 1/3, 1/3).
    verts = torch.tensor([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    faces = torch.tensor([[0, 1, 2]])
    face_id, bary = R.rasterize(verts, faces, 32, 32)
    covered = face_id >= 0
    assert covered.any()
    assert torch.allclose(bary[covered].sum(-1), torch.ones(int(covered.sum())), atol=1e-5)


def test_rasterize_z_buffer_occlusion():
    # Two full-covering triangles; the nearer (smaller z) one wins everywhere.
    near = torch.tensor([[-1.0, -1.0, -0.5], [3.0, -1.0, -0.5], [-1.0, 3.0, -0.5]])
    far = torch.tensor([[-1.0, -1.0, 0.5], [3.0, -1.0, 0.5], [-1.0, 3.0, 0.5]])
    verts = torch.cat([far, near], dim=0)  # face 0 = far, face 1 = near
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
    face_id, _ = R.rasterize(verts, faces, 8, 8)
    covered = face_id >= 0
    assert covered.any()
    assert torch.all(face_id[covered] == 1)  # the near triangle


def test_rasterize_empty_faces():
    verts = torch.zeros((0, 3))
    faces = torch.zeros((0, 3), dtype=torch.long)
    face_id, bary = R.rasterize(verts, faces, 4, 4)
    assert torch.all(face_id == -1)
    assert bary.shape == (4, 4, 3)


# --------------------------------------------------------------------------- #
# Geometry conditioning maps
# --------------------------------------------------------------------------- #
def _quad():
    v = torch.tensor([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]])
    f = torch.tensor([[0, 1, 2], [0, 2, 3]])
    return v, f


def test_render_geometry_maps_shapes_and_ranges():
    v, f = _quad()
    cams = R.standard_cameras(6)
    normals, positions, masks = R.render_geometry_maps(v, f, cams, resolution=32)
    assert normals.shape == (6, 32, 32, 3)
    assert positions.shape == (6, 32, 32, 3)
    assert masks.shape == (6, 32, 32)
    assert float(normals.min()) >= 0.0 and float(normals.max()) <= 1.0
    assert float(positions.min()) >= 0.0 and float(positions.max()) <= 1.0
    # background is white for both control maps
    bg = masks[0] < 0.5
    assert torch.allclose(normals[0][bg], torch.ones_like(normals[0][bg]))
    assert torch.allclose(positions[0][bg], torch.ones_like(positions[0][bg]))
    # a front-facing quad is visible from the front view
    assert float(masks[0].mean()) > 0.1


# --------------------------------------------------------------------------- #
# Per-triangle UV atlas
# --------------------------------------------------------------------------- #
def test_uv_unwrap_packs_all_triangles_non_overlapping():
    torch.manual_seed(0)
    v = torch.rand(10, 3)
    f = torch.tensor([[0, 1, 2], [2, 3, 4], [4, 5, 6], [1, 3, 5], [0, 2, 7],
                      [6, 7, 8], [8, 9, 0], [1, 4, 7]])
    nv, nf, uv = R.pack_per_triangle_uv(v, f)
    Fn = f.shape[0]
    # mesh is unwelded: 3 unique verts/uvs per face, faces index them 0..3F-1
    assert nv.shape == (3 * Fn, 3)
    assert uv.shape == (3 * Fn, 2)
    assert torch.equal(nf, torch.arange(3 * Fn).reshape(Fn, 3))
    assert float(uv.min()) >= 0.0 and float(uv.max()) <= 1.0
    # each triangle's UVs live in its own grid cell -> charts cannot overlap
    g = int(math.ceil(math.sqrt(Fn)))
    cell = 1.0 / g
    tri_uv = uv.reshape(Fn, 3, 2)
    for i in range(Fn):
        c, rr = i % g, i // g
        assert torch.all(tri_uv[i, :, 0] >= c * cell - 1e-6)
        assert torch.all(tri_uv[i, :, 0] <= (c + 1) * cell + 1e-6)
        assert torch.all(tri_uv[i, :, 1] >= rr * cell - 1e-6)
        assert torch.all(tri_uv[i, :, 1] <= (rr + 1) * cell + 1e-6)


# --------------------------------------------------------------------------- #
# Multiview baker
# --------------------------------------------------------------------------- #
def _quad_y(y, half, base=0):
    """Two triangles of an axis-aligned square at height ``y`` facing +/-Y, in the
    renderer's own Z-up frame (for normalize=False tests)."""
    v = torch.tensor([[-half, y, -half], [half, y, -half], [half, y, half], [-half, y, half]])
    f = torch.tensor([[0, 1, 2], [0, 2, 3]]) + base
    return v, f
def _cube():
    v = torch.tensor([
        [-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
        [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5],
    ])
    f = torch.tensor([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
        [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
    ])
    return v, f


def test_bake_flat_color_round_trip():
    v, f = _cube()
    nv, nf, uv = R.pack_per_triangle_uv(v, f)
    cams = R.standard_cameras(6)
    color = torch.tensor([0.1, 0.7, 0.3])
    views = color.view(1, 1, 1, 3).expand(6, 48, 48, 3).contiguous()
    tex, mask = R.bake_multiview(nv, nf, uv, views, cams, texture_size=256)
    assert tex.shape == (256, 256, 3)
    assert int(mask.sum()) > 0
    covered = tex[mask]
    assert torch.allclose(covered, color.expand_as(covered), atol=1e-3)


def test_bake_channels_are_independent():
    # metallic in R, roughness in G survive baking independently.
    v, f = _cube()
    nv, nf, uv = R.pack_per_triangle_uv(v, f)
    cams = R.standard_cameras(6)
    mr = torch.zeros(6, 48, 48, 3)
    mr[..., 0] = 0.2
    mr[..., 1] = 0.9
    tex, mask = R.bake_multiview(nv, nf, uv, mr, cams, texture_size=256)
    covered = tex[mask]
    assert torch.allclose(covered[:, 0], torch.full((covered.shape[0],), 0.2), atol=1e-3)
    assert torch.allclose(covered[:, 1], torch.full((covered.shape[0],), 0.9), atol=1e-3)
    assert torch.allclose(covered[:, 2], torch.zeros(covered.shape[0]), atol=1e-3)


def _chart_centroid_texel(uv, tri, texture_size):
    tri_uv = uv.reshape(-1, 3, 2)
    c = tri_uv[tri].mean(dim=0)
    return int(torch.round(c[1] * (texture_size - 1))), int(torch.round(c[0] * (texture_size - 1)))


def test_bake_per_view_weights_bias_blend():
    # A single quad facing +/-Y is seen head-on (|cos| = 1) by both the front view
    # (weight 1.0) and the back view (weight 0.5) of the standard 6-view set, and
    # only edge-on (past the grazing cutoff) by the other four. Front red + back
    # green must blend to (1.0*red + 0.5*green) / 1.5.
    v, f = _quad_y(0.0, 0.4)
    nv, nf, uv = R.pack_per_triangle_uv(v, f)
    cams = R.standard_cameras(6)
    views = torch.zeros(6, 32, 32, 3)
    views[0, ..., 0] = 1.0  # front view: red
    views[2, ..., 1] = 1.0  # back view: green
    tex, mask = R.bake_multiview(nv, nf, uv, views, cams, texture_size=64, normalize=False)
    expected = torch.tensor([1.0 / 1.5, 0.5 / 1.5, 0.0])
    for i in range(f.shape[0]):
        row, col = _chart_centroid_texel(uv, i, 64)
        assert bool(mask[row, col]), f"chart {i} centroid texel not covered"
        assert torch.allclose(tex[row, col], expected, atol=0.05), (i, tex[row, col])


def test_bake_cosine_exponent_suppresses_grazing_views():
    # Front view sees the quad head-on (cos = 1, red); a second view 60 degrees
    # off-axis (cos = 0.5, green) is damped by cos^bake_exp: with bake_exp=4 the
    # green share at the chart centroid must be marginal, with bake_exp=1 it is
    # a substantial minority share.
    v, f = _quad_y(0.0, 0.4)
    nv, nf, uv = R.pack_per_triangle_uv(v, f)
    cams = R.Cameras(elevs=[0.0, 0.0], azims=[0.0, 60.0], weights=[1.0, 1.0])
    views = torch.zeros(2, 32, 32, 3)
    views[0, ..., 0] = 1.0
    views[1, ..., 1] = 1.0

    def centroid_color(bake_exp):
        tex, mask = R.bake_multiview(nv, nf, uv, views, cams, texture_size=64,
                                     normalize=False, bake_exp=bake_exp)
        row, col = _chart_centroid_texel(uv, 0, 64)
        assert bool(mask[row, col])
        return tex[row, col]

    sharp = centroid_color(4.0)
    soft = centroid_color(1.0)
    assert float(sharp[0]) > 0.9 and float(sharp[1]) < 0.08  # cos^4 = 0.0625
    assert float(soft[1]) > 0.15  # cos^1 = 0.5 keeps a visible share


def test_bake_skips_views_beyond_grazing_threshold():
    # A single view 80 degrees off-axis is past the 75-degree grazing cutoff:
    # it must contribute nothing at all.
    v, f = _quad_y(0.0, 0.4)
    nv, nf, uv = R.pack_per_triangle_uv(v, f)
    cams = R.Cameras(elevs=[0.0], azims=[80.0], weights=[1.0])
    views = torch.ones(1, 32, 32, 3)
    _tex, mask = R.bake_multiview(nv, nf, uv, views, cams, texture_size=64, normalize=False)
    assert not bool(mask.any())


# --------------------------------------------------------------------------- #
# Hole fill: gutter dilation + push-pull inpaint
# --------------------------------------------------------------------------- #
def test_fill_holes_fills_from_neighbors():
    tex = torch.zeros(16, 16, 3)
    mask = torch.zeros(16, 16, dtype=torch.bool)
    tex[8, 8] = torch.tensor([1.0, 0.0, 0.0])
    mask[8, 8] = True
    filled = R.fill_holes(tex, mask)
    # the single valid red texel fills the whole map (dilation + push-pull)
    assert torch.allclose(filled, tex[8, 8].expand_as(filled), atol=1e-4)


def test_fill_holes_preserves_valid_texels_and_fills_everything():
    # Blue strip on the left, red strip on the right, a big hole in between.
    tex = torch.zeros(64, 64, 3)
    mask = torch.zeros(64, 64, dtype=torch.bool)
    tex[:, :4, 2] = 1.0
    mask[:, :4] = True
    tex[:, 60:, 0] = 1.0
    mask[:, 60:] = True
    filled = R.fill_holes(tex, mask)
    # valid texels come back exactly unchanged
    assert torch.allclose(filled[mask], tex[mask], atol=1e-6)
    # every hole texel is filled
    assert bool((filled.abs().sum(-1) > 0).all())
    # gutter dilation extends each strip with its exact color (2 texels out is
    # only reachable from the blue strip)
    assert torch.allclose(filled[:, 6], tex[:, 0].expand_as(filled[:, 6]), atol=1e-4)
    # the deep-hole centre draws low-frequency color from both sides via push-pull
    mid = filled[32, 32]
    assert float(mid[0]) > 0.01 and float(mid[2]) > 0.01
    assert float(filled.min()) >= 0.0 and float(filled.max()) <= 1.0 + 1e-5


def test_fill_holes_reaches_beyond_dilation_distance():
    # A hole far larger than the dilation reach must still be filled (push-pull),
    # and with only one source color present the fill is exactly that color.
    tex = torch.zeros(128, 128, 3)
    mask = torch.zeros(128, 128, dtype=torch.bool)
    tex[0, :, 1] = 1.0  # single green row at the top
    mask[0, :] = True
    filled = R.fill_holes(tex, mask, dilate_iters=2)
    assert torch.allclose(filled[120], tex[0, 0].expand_as(filled[120]), atol=1e-3)


# --------------------------------------------------------------------------- #
# Input validation / robustness bounds
# --------------------------------------------------------------------------- #
def test_render_rejects_nonfinite_vertices():
    v, f = _quad()
    v = v.clone()
    v[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN/Inf"):
        R.render_geometry_maps(v, f, R.standard_cameras(1), resolution=16)


def test_render_rejects_empty_mesh():
    with pytest.raises(ValueError, match="empty"):
        R.render_geometry_maps(torch.zeros(0, 3), torch.zeros(0, 3, dtype=torch.long),
                               R.standard_cameras(1), resolution=16)
    v, f = _quad()
    with pytest.raises(ValueError, match="empty"):
        R.render_geometry_maps(v, torch.zeros(0, 3, dtype=torch.long),
                               R.standard_cameras(1), resolution=16)


def test_render_rejects_out_of_range_face_indices():
    v, f = _quad()
    bad = f.clone()
    bad[0, 0] = 99
    with pytest.raises(ValueError, match="out of range"):
        R.render_geometry_maps(v, bad, R.standard_cameras(1), resolution=16)
    bad[0, 0] = -2
    with pytest.raises(ValueError, match="out of range"):
        R.render_geometry_maps(v, bad, R.standard_cameras(1), resolution=16)


def test_render_rejects_bad_resolution():
    v, f = _quad()
    with pytest.raises(ValueError, match="resolution"):
        R.render_geometry_maps(v, f, R.standard_cameras(1), resolution=0)
    with pytest.raises(ValueError, match="resolution"):
        R.render_geometry_maps(v, f, R.standard_cameras(1), resolution=R.MAX_RESOLUTION + 1)


def test_bake_rejects_bad_inputs():
    v, f = _cube()
    nv, nf, uv = R.pack_per_triangle_uv(v, f)
    views = torch.zeros(1, 8, 8, 3)
    cams = R.Cameras([0.0], [0.0])
    with pytest.raises(ValueError, match="uvs must be"):
        R.bake_multiview(nv, nf, uv[:-1], views, cams, texture_size=32)
    with pytest.raises(ValueError, match="texture_size"):
        R.bake_multiview(nv, nf, uv, views, cams, texture_size=0)
    with pytest.raises(ValueError, match="empty"):
        R.bake_multiview(torch.zeros(0, 3), torch.zeros(0, 3, dtype=torch.long),
                         torch.zeros(0, 2), views, cams, texture_size=32)


def test_zero_area_triangles_render_cleanly():
    # Degenerate (zero-area) triangles must neither win pixels nor poison the maps.
    v, f = _quad()
    v = torch.cat([v, torch.tensor([[0.1, 0.1, 0.2]])])  # vertex 4
    degen = torch.tensor([[4, 4, 4], [0, 0, 1], [2, 2, 2]])
    f = torch.cat([f, degen])
    normals, positions, masks = R.render_geometry_maps(v, f, R.standard_cameras(6), resolution=32)
    assert torch.isfinite(normals).all() and torch.isfinite(positions).all()
    assert float(masks[0].mean()) > 0.1  # the real quad still renders

    nv, nf, uv = R.pack_per_triangle_uv(v, f)
    views = torch.full((1, 32, 32, 3), 0.5)
    tex, mask = R.bake_multiview(nv, nf, uv, views, R.Cameras([0.0], [0.0]), texture_size=64)
    assert torch.isfinite(tex).all()


def test_large_mesh_renders_and_bakes():
    # A ~180k-face displaced grid exercises the chunked rasterizer path end-to-end.
    n = 300
    ys, xs = torch.meshgrid(torch.linspace(-0.5, 0.5, n), torch.linspace(-0.5, 0.5, n),
                            indexing="ij")
    z = 0.1 * torch.sin(xs * 20.0) * torch.cos(ys * 20.0)
    v = torch.stack([xs, ys, z], dim=-1).reshape(-1, 3)
    idx = torch.arange(n * n).reshape(n, n)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    f = torch.cat([torch.stack([a, b, c], -1).reshape(-1, 3),
                   torch.stack([b, d, c], -1).reshape(-1, 3)])
    assert f.shape[0] == 2 * (n - 1) ** 2  # 178,802 faces
    normals, _positions, masks = R.render_geometry_maps(v, f, R.Cameras([0.0], [0.0]),
                                                        resolution=256)
    assert float(masks[0].mean()) > 0.2
    assert torch.isfinite(normals).all()


# --------------------------------------------------------------------------- #
# sRGB / linear audit: GLB embedding round-trip
# --------------------------------------------------------------------------- #
def _read_glb(path):
    with open(path, "rb") as fh:
        data = fh.read()
    assert data[:4] == b"glTF"
    jlen = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20:20 + jlen])
    bin_off = 20 + jlen
    blen = struct.unpack("<I", data[bin_off:bin_off + 4])[0]
    assert data[bin_off + 4:bin_off + 8] == b"BIN\x00"
    return gltf, data[bin_off + 8:bin_off + 8 + blen]


def _decode_glb_image(gltf, binary, image_index):
    view = gltf["bufferViews"][gltf["images"][image_index]["bufferView"]]
    start = view.get("byteOffset", 0)
    png = binary[start:start + view["byteLength"]]
    return Image.open(BytesIO(png))


def test_glb_textures_round_trip_without_gamma_shift(tmp_path):
    """sRGB/linear audit: the save path must embed texture values verbatim
    (uint8-quantized only). The albedo IMAGE is already sRGB-encoded (VAE output)
    and glTF defines baseColorTexture as sRGB, so a value ramp must survive
    identically; the MR texture is defined by glTF as linear, so applying an sRGB
    transfer anywhere would shift e.g. 0.8 -> ~0.91. Both are checked exactly."""
    from comfy_extras.nodes_save_3d import save_glb
    v, f = _cube()
    nv, nf, uv = R.pack_per_triangle_uv(v, f)

    # float IMAGE tensors -> uint8 exactly as SaveGLB.execute converts them
    ramp01 = torch.arange(256, dtype=torch.float32).view(1, 256, 1).expand(8, 256, 3) / 255.0
    mr01 = torch.zeros(8, 256, 3)
    mr01[..., 0] = 1.0
    mr01[..., 1] = 0.8  # roughness -> G
    mr01[..., 2] = 0.2  # metallic  -> B
    albedo_arr = (ramp01.clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
    mr_arr = (mr01.clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
    assert int(mr_arr[0, 0, 1]) == 204 and int(mr_arr[0, 0, 2]) == 51  # no transfer curve
    path = str(tmp_path / "roundtrip.glb")
    save_glb(nv, nf, path, uvs=uv,
             texture_image=Image.fromarray(albedo_arr, mode="RGB"),
             metallic_roughness_image=Image.fromarray(mr_arr, mode="RGB"))

    gltf, binary = _read_glb(path)
    albedo_png = np.array(_decode_glb_image(gltf, binary, 0))
    mr_png = np.array(_decode_glb_image(gltf, binary, 1))

    # bit-exact round trip: no gamma / colorspace transform was applied
    assert np.array_equal(albedo_png, albedo_arr)
    assert np.array_equal(mr_png, mr_arr)
    # an sRGB encode of linear 0.8 would be ~0.91 (232) and of 0.2 ~0.48 (124):
    assert int(mr_png[0, 0, 1]) == 204 and int(mr_png[0, 0, 2]) == 51
    # no color-profile chunks that could make a viewer re-interpret the MR data
    info = _decode_glb_image(gltf, binary, 1).info
    assert "icc_profile" not in info and "gamma" not in info
    # material factors gate the MR texture at 1.0 (values used as-is, per spec)
    pbr = gltf["materials"][0]["pbrMetallicRoughness"]
    assert pbr["metallicFactor"] == 1.0 and pbr["roughnessFactor"] == 1.0


# --------------------------------------------------------------------------- #
# Node schema / wiring + textured-GLB write-back
# --------------------------------------------------------------------------- #
def _load_nodes():
    import comfy_extras.nodes_hunyuan3d_paint as N
    return N


def test_node_schema_matches_execute_signature():
    N = _load_nodes()
    for cls in (N.Hunyuan3DPaintModelLoader, N.Hunyuan3DPaintConditioning,
                N.Hunyuan3DPaintScheduler, N.Hunyuan3DPaintSplitLatent,
                N.Hunyuan3DBakeMultiView):
        ids = [i.id for i in cls.define_schema().inputs]
        params = [p for p in inspect.signature(cls.execute).parameters if p != "cls"]
        assert ids == params, (cls.__name__, ids, params)


def test_conditioning_node_outputs_declared():
    N = _load_nodes()
    outs = [o.display_name for o in N.Hunyuan3DPaintConditioning.define_schema().outputs]
    assert outs == ["model", "positive", "negative", "latent", "cameras", "normal_maps", "position_maps"]
    outs = [o.display_name for o in N.Hunyuan3DPaintSplitLatent.define_schema().outputs]
    assert outs == ["albedo", "mr"]
    outs = [o.display_name for o in N.Hunyuan3DBakeMultiView.define_schema().outputs]
    assert outs == ["mesh", "albedo_texture", "mr_texture", "albedo_views", "mr_views"]


def test_bake_node_execute_attaches_textures_and_uvs():
    from comfy_api.latest import Types
    N = _load_nodes()
    v, f = _cube()
    mesh = Types.MESH(v.unsqueeze(0), f.unsqueeze(0))
    cams = R.standard_cameras(6)
    albedo = torch.zeros(6, 48, 48, 3)
    albedo[..., 1] = 1.0  # green
    mr = torch.zeros(6, 48, 48, 3)
    mr[..., 0] = 0.2  # metallic
    mr[..., 1] = 0.8  # roughness
    out_mesh, alb_tex, mr_tex, alb_views, mr_views = N.Hunyuan3DBakeMultiView.execute(
        mesh, albedo, mr, cams, 256, 4.0, False)

    assert out_mesh.uvs is not None
    assert out_mesh.uvs.shape[1] == out_mesh.vertices.shape[1]  # UVs 1:1 with vertices
    assert out_mesh.texture is not None and out_mesh.texture.shape == (1, 256, 256, 3)
    assert out_mesh.texture_mr is not None and out_mesh.texture_mr.shape == (1, 256, 256, 3)
    # native MR image output keeps R=metallic, G=roughness
    covered = alb_tex[0].abs().sum(-1) > 0
    assert bool(covered.any()), "expected baked texture coverage"
    assert torch.allclose(mr_tex[0][covered][:, 0], torch.full((int(covered.sum()),), 0.2), atol=1e-3)
    # glTF-packed MR on the mesh: G=roughness, B=metallic
    packed = out_mesh.texture_mr[0]
    assert torch.allclose(packed[covered][:, 1], torch.full((int(covered.sum()),), 0.8), atol=1e-3)
    assert torch.allclose(packed[covered][:, 2], torch.full((int(covered.sum()),), 0.2), atol=1e-3)


def test_bake_node_uses_existing_mesh_uvs():
    # A mesh that already carries per-vertex UVs (artist-authored or from an unwrap
    # node) must be baked onto those UVs as-is: no re-unwrap, no unweld — the output
    # mesh keeps the input vertices, faces and UVs.
    from comfy_api.latest import Types
    N = _load_nodes()
    v, f = _cube()
    torch.manual_seed(0)
    uvs = torch.rand(v.shape[0], 2)
    mesh = Types.MESH(v.unsqueeze(0), f.unsqueeze(0), uvs=uvs.unsqueeze(0))
    cams = R.standard_cameras(6)
    albedo = torch.full((6, 32, 32, 3), 0.5)
    mr = torch.zeros(6, 32, 32, 3)
    out_mesh, _alb_tex, _mr_tex, _av, _mv = N.Hunyuan3DBakeMultiView.execute(
        mesh, albedo, mr, cams, 128, 4.0, False)
    assert torch.allclose(out_mesh.vertices[0], v)
    assert torch.equal(out_mesh.faces[0], f)
    assert torch.allclose(out_mesh.uvs[0], uvs)


def test_mesh_metallic_roughness_adapter_preserves_paint_texture_mr():
    from comfy_api.latest import Types

    mr = torch.rand(2, 8, 8, 3)
    from_legacy = Types.MESH(torch.zeros(2, 3, 3), torch.zeros(2, 1, 3, dtype=torch.long), texture_mr=mr)
    from_current = Types.MESH(torch.zeros(2, 3, 3), torch.zeros(2, 1, 3, dtype=torch.long), metallic_roughness=mr)
    assert from_legacy.metallic_roughness is mr
    assert from_legacy.texture_mr is mr
    assert from_current.metallic_roughness is mr
    assert from_current.texture_mr is mr


def test_mesh_item_glb_uses_matching_batched_mr_texture(tmp_path):
    from comfy_api.latest import Types
    from comfy_extras.nodes_save_3d import mesh_item_to_glb_bytes

    v, f = _cube()
    packed_v, packed_f, packed_uv = R.pack_per_triangle_uv(v, f)
    vertices = packed_v.unsqueeze(0).repeat(2, 1, 1)
    faces = packed_f.unsqueeze(0).repeat(2, 1, 1)
    uvs = packed_uv.unsqueeze(0).repeat(2, 1, 1)
    albedo = torch.ones(2, 32, 32, 3)
    mr = torch.zeros(2, 32, 32, 3)
    mr[0, ..., 1] = 0.1
    mr[1, ..., 1] = 0.9
    mesh = Types.MESH(vertices, faces, uvs=uvs, texture=albedo, metallic_roughness=mr)
    path = str(tmp_path / "batch_1.glb")
    with open(path, "wb") as fh:
        fh.write(mesh_item_to_glb_bytes(mesh, 1))
    gltf, binary = _read_glb(path)
    image = np.array(_decode_glb_image(gltf, binary, 1))
    assert int(image[0, 0, 1]) == int(0.9 * 255)


def test_save_glb_writes_metallic_roughness_texture(tmp_path):
    from comfy_extras.nodes_save_3d import save_glb
    v, f = _cube()
    nv, nf, uv = R.pack_per_triangle_uv(v, f)
    base = Image.new("RGB", (32, 32), (0, 255, 0))
    mrimg = Image.new("RGB", (32, 32), (255, 200, 40))
    path = str(tmp_path / "out.glb")
    save_glb(nv, nf, path, uvs=uv, texture_image=base, metallic_roughness_image=mrimg)

    with open(path, "rb") as fh:
        data = fh.read()
    assert data[:4] == b"glTF"
    jlen = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20:20 + jlen])
    assert len(gltf["images"]) == 2
    assert len(gltf["textures"]) == 2
    pbr = gltf["materials"][0]["pbrMetallicRoughness"]
    assert "baseColorTexture" in pbr
    assert "metallicRoughnessTexture" in pbr
    assert pbr["metallicRoughnessTexture"]["index"] == 1


def _fake_vae():
    class FakeVAE:
        def encode(self, img):
            x = img.permute(0, 3, 1, 2)
            lat = torch.nn.functional.interpolate(x, scale_factor=0.125, mode="bilinear", align_corners=False)
            return torch.cat([lat, lat[:, :1]], dim=1)  # 4-channel latent at H/8

        def decode(self, lat):
            x = torch.nn.functional.interpolate(lat[:, :3], scale_factor=8, mode="bilinear", align_corners=False)
            return x.permute(0, 2, 3, 1).clamp(0, 1)

    return FakeVAE()


def test_conditioning_sampling_then_bake_end_to_end_with_random_model():
    """Exercise the full node glue: MESH + reference -> conditioning node (render +
    reference bank + packed latent) -> core sampling loop -> split -> VAE decode ->
    bake -> textured mesh, on a small random-init UNet + stub VAE."""
    import comfy.ops as comfy_ops
    import comfy.sample
    import comfy.samplers
    from comfy.ldm.hunyuan3d.paint.unet import UNet2p5DConditionModel
    from comfy.ldm.hunyuan3d.paint.loader import load_paint_unet
    from comfy_api.latest import Types
    N = _load_nodes()

    cfg = dict(
        in_channels=12, ref_in_channels=4, out_channels=4,
        block_out_channels=(64, 64, 128, 128), layers_per_block=2, cross_attention_dim=64,
        num_attention_heads=(1, 1, 2, 2), transformer_layers_per_block=1, norm_num_groups=32,
        pbr_setting=("albedo", "mr"), pbr_token_channels=7, dino_embeddings_dim=32, use_dino=True)
    model = UNet2p5DConditionModel(dtype=torch.float32, device="cpu",
                                   operations=comfy_ops.disable_weight_init, **cfg)
    model.eval()
    patcher = load_paint_unet(model.state_dict(), model_options={"dtype": torch.float32})

    v, f = _cube()
    mesh = Types.MESH(v.unsqueeze(0), f.unsqueeze(0))
    ref = torch.rand(1, 128, 128, 3)

    m, positive, negative, latent, cams, normals, positions = N.Hunyuan3DPaintConditioning.execute(
        patcher, _fake_vae(), mesh, ref, 6, 128)
    assert latent["samples"].shape == (1, 4, 12, 16, 16)
    assert normals.shape == (6, 128, 128, 3)
    assert len(cams) == 6

    noise = comfy.sample.prepare_noise(latent["samples"], 0)
    sigmas = N.Hunyuan3DPaintScheduler.execute(m, 2)[0]
    sampler = comfy.samplers.sampler_object("euler")
    samples = comfy.sample.sample_custom(m, noise, 3.0, sampler, sigmas, positive, negative,
                                         latent["samples"], disable_pbar=True, seed=0)

    albedo_lat, mr_lat = N.Hunyuan3DPaintSplitLatent.execute(m, {"samples": samples})
    assert albedo_lat["samples"].shape == (6, 4, 16, 16)
    assert mr_lat["samples"].shape == (6, 4, 16, 16)
    vae = _fake_vae()
    albedo = vae.decode(albedo_lat["samples"])
    mr = vae.decode(mr_lat["samples"])
    assert albedo.shape == (6, 128, 128, 3)
    assert mr.shape == (6, 128, 128, 3)

    out_mesh, alb_tex, mr_tex, _av, _mv = N.Hunyuan3DBakeMultiView.execute(
        mesh, albedo, mr, cams, 256, 4.0, True)
    assert out_mesh.uvs is not None and out_mesh.texture is not None
    assert out_mesh.texture_mr is not None
    assert alb_tex.shape == (1, 256, 256, 3)


def test_save_glb_without_mr_still_single_texture(tmp_path):
    from comfy_extras.nodes_save_3d import save_glb
    v, f = _cube()
    nv, nf, uv = R.pack_per_triangle_uv(v, f)
    base = Image.new("RGB", (16, 16), (128, 128, 128))
    path = str(tmp_path / "out.glb")
    save_glb(nv, nf, path, uvs=uv, texture_image=base)
    with open(path, "rb") as fh:
        data = fh.read()
    jlen = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20:20 + jlen])
    assert len(gltf["images"]) == 1
    pbr = gltf["materials"][0]["pbrMetallicRoughness"]
    assert "metallicRoughnessTexture" not in pbr
