"""ComfyUI nodes for the native MoGe (Monocular Geometry Estimation) integration."""


import torch

import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, Types, io
from typing_extensions import override

from comfy.ldm.colormap import turbo as _turbo
from comfy.ldm.moge.model import MoGeModel
from comfy.ldm.moge.geometry import triangulate_grid_mesh
from comfy.ldm.moge.panorama import get_panorama_cameras, split_panorama_image, merge_panorama_depth, spherical_uv_to_directions, _uv_grid
import comfy.model_management
from tqdm.auto import tqdm

MoGeModelType = io.Custom("MOGE_MODEL")
MoGeGeometry = io.Custom("MOGE_GEOMETRY")


# MOGE_GEOMETRY is a dict with these optional keys (absent when the upstream model didn't produce them):
#   "points":     torch.Tensor (B, H, W, 3)
#   "depth":      torch.Tensor (B, H, W)
#   "intrinsics": torch.Tensor (B, 3, 3)   -- perspective only
#   "mask":       torch.Tensor (B, H, W) bool
#   "normal":     torch.Tensor (B, H, W, 3) -- v2 only
#   "image":      torch.Tensor (B, H, W, 3) in [0, 1], CPU (always present)


def _normals_from_points(points: torch.Tensor) -> torch.Tensor:
    """Camera-space surface normals from a (B, H, W, 3) point map (v1 fallback)."""
    finite = torch.isfinite(points).all(dim=-1)
    pts = torch.where(finite.unsqueeze(-1), points, torch.zeros_like(points))
    dx = pts[..., :, 2:, :] - pts[..., :, :-2, :]
    dy = pts[..., 2:, :, :] - pts[..., :-2, :, :]
    dx = torch.nn.functional.pad(dx.permute(0, 3, 1, 2), (1, 1, 0, 0)).permute(0, 2, 3, 1)
    dy = torch.nn.functional.pad(dy.permute(0, 3, 1, 2), (0, 0, 1, 1)).permute(0, 2, 3, 1)
    # dy x dx (not dx x dy) so the result is outward-facing in OpenCV (Y-down flips the right-hand rule), matching v2's predicted normals.
    n = torch.cross(dy, dx, dim=-1)
    n = torch.nn.functional.normalize(n, dim=-1)
    return torch.where(finite.unsqueeze(-1), n, torch.zeros_like(n))


def _normalize_disparity(depth: torch.Tensor) -> torch.Tensor:
    """Per-batch normalize 1/depth to [0, 1] using 0.1/99.9 percentile clipping."""
    out = torch.zeros_like(depth)
    for i in range(depth.shape[0]):
        d = depth[i]
        valid = torch.isfinite(d) & (d > 0)
        if not valid.any():
            continue
        disp = torch.where(valid, 1.0 / d.clamp_min(1e-6), torch.zeros_like(d))
        disp_valid = disp[valid]
        lo = torch.quantile(disp_valid, 0.001)
        hi = torch.quantile(disp_valid, 0.999)
        scale = (hi - lo).clamp_min(1e-6)
        norm = ((disp - lo) / scale).clamp(0.0, 1.0)
        out[i] = torch.where(valid, norm, torch.zeros_like(norm))
    return out


class LoadMoGeModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadMoGeModel",
            display_name="Load MoGe Model",
            category="model/loaders",
            inputs=[
                io.Combo.Input("model_name", options=folder_paths.get_filename_list("geometry_estimation")),
            ],
            outputs=[MoGeModelType.Output()],
        )

    @classmethod
    def execute(cls, model_name) -> io.NodeOutput:
        path = folder_paths.get_full_path_or_raise("geometry_estimation", model_name)
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        return io.NodeOutput(MoGeModel(sd))


class MoGePanoramaInference(io.ComfyNode):
    """Equirectangular panorama inference: split into 12 perspective views, run
    MoGe at fov_x=90 on each, merge via multi-scale Poisson + gradient solve.
    v2's predicted normals and metric scale are ignored (per-view scales would not align across seams).
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MoGePanoramaInference",
            search_aliases=["moge", "panorama", "depth", "geometry", "depth estimation", "geometry estimation"],
            display_name="Run MoGe Panorama Inference",
            category="image/geometry estimation",
            description="Run MoGe on an equirectangular panorama by splitting it into 12 perspective views, running inference on each, and merging the results into a single depth map.",
            inputs=[
                MoGeModelType.Input("moge_model"),
                io.Image.Input("image", tooltip="Equirectangular panorama (any aspect)."),
                io.Int.Input("resolution_level", default=9, min=0, max=9,
                             tooltip="Per-view detail (0 = fastest, 9 = most detailed)."),
                io.Int.Input("split_resolution", default=512, min=256, max=1024,
                             tooltip="Resolution of each perspective split."),
                io.Int.Input("merge_resolution", default=1920, min=256, max=8192,
                             tooltip="Long-side resolution of the merged equirect distance map."),
                io.Int.Input("batch_size", default=4, min=1, max=12,
                             tooltip="Views per inference batch (12 splits total)."),
            ],
            outputs=[MoGeGeometry.Output(display_name="moge_geometry")],
        )

    @classmethod
    def execute(cls, moge_model, image, resolution_level, split_resolution, merge_resolution, batch_size) -> io.NodeOutput:

        if image.shape[0] != 1:
            raise ValueError(f"MoGePanoramaInference takes a single image (got batch of {image.shape[0]})")

        image = image[..., :3]
        H, W = int(image.shape[1]), int(image.shape[2])
        scale = min(merge_resolution / max(H, W), 1.0)
        merge_h, merge_w = max(int(H * scale), 32), max(int(W * scale), 32)

        extrinsics, intrinsics = get_panorama_cameras()

        comfy.model_management.load_model_gpu(moge_model.patcher)
        device = moge_model.load_device
        img_chw = image[0].movedim(-1, -3).to(device=device, dtype=moge_model.dtype)
        splits = split_panorama_image(img_chw, extrinsics, intrinsics, split_resolution)

        n_views = splits.shape[0]

        # Weight each lsmr solve by 4^level so the final-resolution solve doesn't leave the bar idle.
        merge_levels: list[tuple[int, int]] = []
        w_, h_ = merge_w, merge_h
        while True:
            merge_levels.append((w_, h_))
            if max(w_, h_) <= 256:
                break
            w_, h_ = w_ // 2, h_ // 2
        merge_levels.reverse()

        solve_weight = {wh: 4 ** i for i, wh in enumerate(merge_levels)}
        n_merge_view_units = n_views * len(merge_levels)
        n_merge_solve_units = sum(solve_weight.values())

        pbar = comfy.utils.ProgressBar(n_views + n_merge_view_units + n_merge_solve_units)
        done = 0

        distance_maps: list = []
        masks: list = []
        with tqdm(total=n_views, desc="MoGe panorama inference") as tq:
            for i in range(0, n_views, batch_size):
                batch = splits[i:i + batch_size]
                # apply_metric_scale=False: per-view scales would not align across overlap seams.
                result = moge_model.infer(batch, resolution_level=resolution_level,
                                          fov_x=90.0, force_projection=True,
                                          apply_mask=False, apply_metric_scale=False)
                distance_maps.extend(list(result["points"].float().norm(dim=-1).cpu().numpy()))
                masks.extend(list(result["mask"].cpu().numpy()))
                n = batch.shape[0]
                done += n
                pbar.update_absolute(done)
                tq.update(n)

        with tqdm(total=n_merge_view_units + n_merge_solve_units, desc="MoGe panorama merge: views") as tq:
            def _on_merge_view():
                nonlocal done
                done += 1
                pbar.update_absolute(done)
                tq.update(1)

            def _on_solve_start(w, h):
                tq.set_description(f"MoGe panorama merge: solving {w}x{h}")

            def _on_solve_end(w, h):
                nonlocal done
                weight = solve_weight[(w, h)]
                done += weight
                pbar.update_absolute(done)
                tq.update(weight)
                tq.set_description("MoGe panorama merge: views")

            pano_depth, pano_mask = merge_panorama_depth(
                merge_w, merge_h, distance_maps, masks, list(extrinsics), intrinsics,
                on_view=_on_merge_view, on_solve_start=_on_solve_start, on_solve_end=_on_solve_end)

        pano_depth = torch.from_numpy(pano_depth)
        pano_mask = torch.from_numpy(pano_mask)

        if (merge_h, merge_w) != (H, W):
            pano_depth = torch.nn.functional.interpolate(pano_depth[None, None], size=(H, W), mode="bilinear", align_corners=False).squeeze()
            pano_mask = torch.nn.functional.interpolate(pano_mask[None, None].float(), size=(H, W), mode="nearest").squeeze() > 0

        # Pixels uncovered by any view's predicted foreground are unconstrained in the lsmr solve and stay at log_depth=0 (depth=1)
        if pano_mask.any() and not pano_mask.all():
            far = torch.quantile(pano_depth[pano_mask], 0.95) * 5.0
            pano_depth = torch.where(pano_mask, pano_depth, far)

        directions = torch.from_numpy(spherical_uv_to_directions(_uv_grid(H, W)))
        points = (directions * pano_depth[..., None]).unsqueeze(0)
        depth = pano_depth.unsqueeze(0)
        mask = pano_mask.unsqueeze(0)

        # Points stay in MoGe spherical coords; MoGePointMapToMesh applies the spherical->glTF rotation after triangulation
        moge_geometry = {"points": points, "depth": depth, "mask": mask, "image": image.cpu()}
        return io.NodeOutput(moge_geometry)


class MoGeInference(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MoGeInference",
            search_aliases=["moge", "depth", "geometry", "depth estimation", "geometry estimation"],
            display_name="Run MoGe Inference",
            description="Run MoGe on a single image to estimate depth and geometry.",
            category="image/geometry estimation",
            inputs=[
                MoGeModelType.Input("moge_model"),
                io.Image.Input("image"),
                io.Int.Input("resolution_level", default=9, min=0, max=9,
                             tooltip="0 = fastest, 9 = most detail."),
                io.Float.Input("fov_x_degrees", default=0.0, min=0.0, max=170.0, step=0.1, advanced=True,
                               tooltip="Horizontal field of view of the source camera. Sets the focal length used to unproject the depth map into 3D. 0 = auto-recover from the predicted points."),
                io.Int.Input("batch_size", default=4, min=1, max=64,
                             tooltip="Images per inference call. Lower if you OOM on a long video / image set."),
                io.Boolean.Input("force_projection", default=True, advanced=True),
                io.Boolean.Input("apply_mask", default=True, advanced=True,
                                 tooltip="Set masked-out (sky / invalid) pixels to inf in points and depth so meshing culls them. Disable to keep the raw predicted geometry everywhere; the mask is still returned separately."),
            ],
            outputs=[MoGeGeometry.Output(display_name="moge_geometry")],
        )

    @classmethod
    def execute(cls, moge_model, image, resolution_level, fov_x_degrees, batch_size, force_projection, apply_mask) -> io.NodeOutput:

        image = image[..., :3]
        bchw = image.movedim(-1, -3).contiguous()
        B = bchw.shape[0]
        fov = None if fov_x_degrees <= 0 else float(fov_x_degrees)

        pbar = comfy.utils.ProgressBar(B)
        chunks: list[dict] = []
        with tqdm(total=B, desc="MoGe inference") as tq:
            for i in range(0, B, batch_size):
                chunk = bchw[i:i + batch_size]
                chunks.append(moge_model.infer(chunk, resolution_level=resolution_level, fov_x=fov,
                                               force_projection=force_projection, apply_mask=apply_mask))
                pbar.update_absolute(min(i + batch_size, B))
                tq.update(chunk.shape[0])

        def stack(field):
            vals = [c[field] for c in chunks if field in c]
            return torch.cat(vals, dim=0) if vals else None

        moge_geometry = {"image": image.cpu()}
        for field in ("points", "depth", "intrinsics", "mask", "normal"):
            v = stack(field)
            if v is not None:
                moge_geometry[field] = v
        return io.NodeOutput(moge_geometry)


class MoGeRender(io.ComfyNode):
    """Render a visualization or mask from a MOGE_GEOMETRY packet."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MoGeRender",
            search_aliases=["moge", "render", "geometry", "depth", "normal"],
            display_name="Render MoGe Geometry",
            description="Render a depth map or normal map from geometry data",
            category="image/geometry estimation",
            inputs=[
                MoGeGeometry.Input("moge_geometry"),
                io.Combo.Input("output", options=["depth", "depth_colored", "normal_opengl", "normal_directx", "mask"], default="depth",
                    tooltip="DirectX vs OpenGL controls the normal-map green-channel convention. DirectX: green = -Y down (Unreal). OpenGL: green = +Y up (Blender, Substance, Unity, glTF)."),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, moge_geometry, output) -> io.NodeOutput:
        is_normal = output in ("normal_directx", "normal_opengl")
        opengl = output.endswith("_opengl")

        # Pick the input tensor for the chosen mode and validate availability.
        if output in ("depth", "depth_colored"):
            if "depth" not in moge_geometry:
                raise ValueError("moge_geometry has no depth output.")
            src = moge_geometry["depth"]
        elif is_normal:
            if "normal" in moge_geometry:
                src = moge_geometry["normal"]
            elif "points" in moge_geometry:
                src = moge_geometry["points"]
            else:
                raise ValueError("moge_geometry has neither normals nor points to derive normals from.")
        elif output == "mask":
            if "mask" not in moge_geometry:
                raise ValueError("moge_geometry has no mask output.")
            src = moge_geometry["mask"]
        else:
            raise ValueError(f"Unknown output mode: {output}")

        B = src.shape[0]
        pbar = comfy.utils.ProgressBar(B)
        out: list[torch.Tensor] = []
        with tqdm(total=B, desc=f"MoGe render: {output}") as tq:
            for i in range(B):
                slc = src[i:i + 1].float()
                if output in ("depth", "depth_colored"):
                    d = _normalize_disparity(slc)
                    out.append(_turbo(d) if output == "depth_colored"
                               else d.unsqueeze(-1).expand(*d.shape, 3).contiguous())
                elif is_normal:
                    n = slc if "normal" in moge_geometry else _normals_from_points(slc)
                    # MoGe is OpenCV (Z+ into scene); normal-map convention is Z+ out of surface, so flip Z.
                    y_sign = -1.0 if opengl else 1.0
                    n = n * n.new_tensor([1.0, y_sign, -1.0])
                    out.append((n * 0.5 + 0.5).clamp(0.0, 1.0))
                elif output == "mask":
                    out.append(slc.unsqueeze(-1).expand(*slc.shape, 3).contiguous())
                pbar.update_absolute(i + 1)
                tq.update(1)
        result = torch.cat(out, dim=0).to(device=comfy.model_management.intermediate_device(), dtype=comfy.model_management.intermediate_dtype())
        return io.NodeOutput(result)


class MoGePointMapToMesh(io.ComfyNode):
    """Triangulate one image of a MoGe point map into a Types.MESH (UVs + texture)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MoGePointMapToMesh",
            search_aliases=["moge", "mesh", "geometry", "point map"],
            display_name="Convert MoGe Point Map to Mesh",
            description="Convert a MoGe point map into a 3D mesh.",
            category="image/geometry estimation",
            inputs=[
                MoGeGeometry.Input("moge_geometry"),
                io.Int.Input("batch_index", default=0, min=0, max=4096,
                             tooltip="Which image of a batched MoGe geometry to mesh. Per-image vertex counts "
                                     "differ, so batches can't be stacked into a single MESH."),
                io.Int.Input("decimation", default=1, min=1, max=8,
                             tooltip="Vertex stride; 1 = full resolution."),
                io.Float.Input("discontinuity_threshold", default=0.04, min=0.0, max=1.0, step=0.01,
                               tooltip="Drop pixels whose 3x3 depth span exceeds this fraction. 0 = off."),
                io.Boolean.Input("texture", default=True,
                                 tooltip="Carry the source image through as the baseColor texture."),
            ],
            outputs=[io.Mesh.Output()],
        )

    @classmethod
    def execute(cls, moge_geometry, batch_index, decimation, discontinuity_threshold, texture) -> io.NodeOutput:
        if "points" not in moge_geometry:
            raise ValueError("moge_geometry has no points output.")
        points = moge_geometry["points"]
        B = points.shape[0]
        if batch_index >= B:
            raise ValueError(f"batch_index {batch_index} out of range; moge_geometry has batch size {B}.")

        # Pass depth so the rtol edge check sees radial depth -- for panoramas
        # points[..., 2] = cos(phi)*r goes negative below the equator and the rtol clamp would drop the bottom half.
        edge_depth = moge_geometry["depth"][batch_index] if "depth" in moge_geometry else None
        verts, faces, uvs = triangulate_grid_mesh(
            points[batch_index], decimation=decimation,
            discontinuity_threshold=discontinuity_threshold, depth=edge_depth,
        )
        if verts.shape[0] == 0 or faces.shape[0] == 0:
            raise ValueError("MoGe produced an empty mesh; try discontinuity_threshold=0 or apply_mask=False.")

        if "intrinsics" not in moge_geometry:
            # Panorama: rotate MoGe spherical (Z up) -> glTF (Y up, Z back), correct for inside-the-sphere viewing)
            verts = verts[:, [1, 2, 0]].contiguous()
        else:
            # Perspective MoGe (X right, Y down, Z forward) -> glTF; face flip keeps winding CCW after the Y/Z flip.
            verts = verts * torch.tensor([1.0, -1.0, -1.0], dtype=verts.dtype)
            faces = faces[:, [0, 2, 1]].contiguous()

        tex = moge_geometry["image"][batch_index:batch_index + 1] if texture else None
        mesh = Types.MESH(
            vertices=verts.unsqueeze(0),
            faces=faces.unsqueeze(0),
            uvs=uvs.unsqueeze(0),
            texture=tex,
        )
        return io.NodeOutput(mesh)


class MoGeExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [LoadMoGeModel, MoGeInference, MoGePanoramaInference, MoGeRender, MoGePointMapToMesh]


async def comfy_entrypoint() -> MoGeExtension:
    return MoGeExtension()
