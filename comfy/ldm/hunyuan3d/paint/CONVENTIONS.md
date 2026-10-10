# Coordinate & image conventions — Hunyuan3D 2.1 paint renderer/baker

Every frame this module touches, in pipeline order, and where each (non-)flip
lives. The reference-parity basis is Tencent's `hy3dpaint` renderer
(`DifferentiableRenderer/{MeshRender,camera_utils}.py`, `custom_rasterizer`),
which produced the conditioning images the released
`hunyuan3d-paintpbr-v2-1` UNet was trained on. Matching those weights — not
"looking natural" in any single buffer — is the design goal. Self-consistency
and chirality are enforced by
`tests-unit/comfy_test/test_hunyuan3d_paint_conventions.py` (chirality-cube
armor with a mutation self-test).

## 1. Input mesh frame

glTF/GLB convention: **Y-up, front = +Z**, right-handed, outward triangle
winding. This is what core `MESH` tensors carry.

## 2. World (renderer) frame — `normalize_mesh()`

`(x, y, z)_glb -> (-x, z, -y)_world`, then center on the bbox midpoint and
scale so the bounding-sphere *diameter* is `MESH_SCALE_FACTOR = 1.15`, i.e.
max vertex radius 0.575 (matches `MeshRender.set_mesh(auto_center=True)`;
this is what makes `0.5 - p / 1.15` land in `[0, 1]` for the position maps).

Two properties to be aware of:

- The map has **det = -1** (it is a mirror). Face normals computed from the
  *unchanged* winding therefore flip: a glTF-outward-wound front (+Z) face has
  world-frame flat normal `(0, -1, 0)`, i.e. it points *away* from the front
  camera. This is the reference behaviour; the rasterizer and baker are
  winding-agnostic by construction (z-buffer visibility, `|cos|` view
  weighting), and the normal-map colors the UNet was trained on encode exactly
  this convention (front face ≈ `[0.5, 0.0, 0.5]` after the `(n+1)/2` map).
- glTF "up" (+Y) maps to world **-Z**, which the camera model compensates
  (below). Neither half of this pair may be changed independently.

## 3. Camera — `view_matrix()` (matches `camera_utils.get_mv_matrix`)

Z-up look-at orbit with `elev := -elev` and `azim := azim + 90` applied
internally, orthographic projection (`ortho_scale = 1.2`, near 0.1, far 100,
camera distance 1.45). Net mapping of the standard views to glTF axis faces
(enforced by the armor tests):

| view   | elev | azim | sees glTF face |
|--------|------|------|----------------|
| front  | 0    | 0    | +Z |
| right  | 0    | 90   | +X |
| back   | 0    | 180  | -Z |
| left   | 0    | 270  | -X |
| top    | 90   | 0    | +Y |
| bottom | -90  | 180  | -Y |

In the front view, screen-x tracks glTF **+X** (a character's left hand
appears on the viewer's right, as when facing someone) and camera-up is world
+Z = glTF **-Y**.

## 4. Raster buffer — `rasterize()`

Pixel row grows **with** +NDC y (`sy = (ndc_y * 0.5 + 0.5) * H`, no
top/bottom flip), matching Tencent's `custom_rasterizer`
(`barycentricFromImgcoordCPU`). Because camera-up is glTF -Y (section 3), the
composition leaves the raw buffer **upright**: glTF-up lands at row 0.

There is **no explicit image y-flip anywhere in this pipeline** — not in the
rasterizer, the geometry maps, the baker, or the GLB writer. The upright
appearance is the composition of the two deliberate sign pairs above
(mesh `-y` swap x elevation negation). If you add a flip in one place you must
remove its partner, and the trained UNet will disagree with you; the armor
tests will fail first.

## 5. Conditioning maps — `render_geometry_maps()`

- Normal maps: **object(world)-space flat normals**, `(n + 1) / 2`, white
  background (mask complement). See section 2 for the winding/mirror caveat.
- Position maps: `0.5 - p_world / 1.15`, white background ("no geometry" =
  white). These feed both the VAE conditioning latents and PoseRoPE
  (`multires_voxel_indices` treats all-ones pixels as background).

## 6. UV / texture space — `bake_multiview()`, `pack_per_triangle_uv()`

`(u, v) -> texel (row = v * (T-1), col = u * (T-1))`; v = 0 is texture row 0.
The baked tensor's row 0 is written as the PNG top row by the save path, and
glTF's `TEXCOORD_0` origin is top-left with v growing downward — so UVs are
written to the GLB **verbatim** (`save_glb` performs no V flip) and the
round-trip is consistent end-to-end. Baking targets a mesh's existing
per-vertex UVs when present; UV-less meshes get the per-triangle atlas.

## 7. Bake weighting

Per-view weight = `view_weight * |cos(view angle)| ** bake_exp` with a 75°
grazing cutoff, z-buffer visibility per view, weighted average across views
(reference `bake_exp = 4`, per-view weights `[1.0, 0.1, 0.5, 0.1, 0.05,
0.05]`). `|cos|` (not signed cos) keeps the baker winding-agnostic under the
det = -1 world map.

## Reference-parity basis

Conventions were transcribed from `hy3dpaint` sources (read-only reference:
`MeshRender`, `camera_utils`, `custom_rasterizer`) and validated two ways:
end-to-end chest renders with the released weights (correct texture placement
at ss4 supersampling), and the procedural chirality-cube armor in this repo,
whose mutation self-test proves each convention class (V flip, axis negation,
winding reversal, camera-sign swap) is actually caught.
