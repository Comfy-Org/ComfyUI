"""ComfyUI nodes for Depth Anything 3.
Model capability matrix:

Variant               head_type  has_sky  has_conf  cam_dec
DA3-Small             dualdpt    False    True      yes
DA3-Base              dualdpt    False    True      yes
DA3-Mono-Large        dpt        True     False     no
DA3-Metric-Large      dpt        True     False     no  (raw output is metres)
"""

from __future__ import annotations

import logging
from typing_extensions import override

import torch

import comfy.model_management as mm
import comfy.sd
import folder_paths
from comfy.ldm.colormap import turbo as _turbo
from comfy.ldm.depth_anything_3 import preprocess as da3_preprocess
from comfy_api.latest import ComfyExtension, Types, io
from comfy.ldm.moge.geometry import triangulate_grid_mesh

DA3ModelType = io.Custom("DA3_MODEL")
DA3Geometry = io.Custom("DA3_GEOMETRY")
DA3PointCloud = io.Custom("DA3_POINT_CLOUD")

# DA3_GEOMETRY is a dict with these optional keys (absent when the upstream model didn't produce them):
#
# Per-frame tensors - B = batch size in mono mode; B = S (number of views) in multi-view mode.
#   "depth":       torch.Tensor (B, H, W)         -- raw model depth (always present; matches MoGe convention)
#   "image":       torch.Tensor (B, H, W, 3)      -- source image in [0, 1], CPU (always present)
#   "mode":        str                            -- "mono" or "multiview" (always present)
#   "sky":         torch.Tensor (B, H, W)         -- sky probability in [0, 1] (Mono/Metric variants only)
#   "confidence":  torch.Tensor (B, H, W)         -- raw model confidence output (Small/Base variants only)
#
# Multi-view only - S = number of views; the leading 1 is the scene dimension from the model.
#   "extrinsics":  torch.Tensor (1, S, 3, 4)      -- world-to-camera [R|t] matrices
#   "intrinsics":  torch.Tensor (1, S, 3, 3)      -- pixel-space intrinsics
#
# DA3_POINT_CLOUD is a dict:
#   "points":     torch.Tensor (N, 3)  -- 3-D coords in glTF convention (Y-up, Z-back)
#   "colors":     torch.Tensor (N, 3)  -- RGB in [0, 1], or None
#   "confidence": torch.Tensor (N,)    -- raw confidence per point, or None


def _da3_unproject(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Pixel-space K⁻¹ unprojection: (H,W) depth → (H,W,3) point map in OpenCV space."""
    H, W = depth.shape
    u = torch.arange(W, dtype=torch.float32, device=depth.device)
    v = torch.arange(H, dtype=torch.float32, device=depth.device)
    u, v = torch.meshgrid(u, v, indexing='xy')             # both (H, W)
    pix = torch.stack([u, v, torch.ones_like(u)], dim=-1)  # (H, W, 3)
    rays = torch.einsum('ij,hwj->hwi', torch.linalg.inv(K.to(depth.device)), pix)
    return rays * depth.unsqueeze(-1)                       # (H, W, 3)


def _da3_default_K(H: int, W: int) -> torch.Tensor:
    """Fallback ~60° FOV pinhole K for mono-mode DA3 (no intrinsics in geometry)."""
    fx = fy = float(W) * 0.7
    return torch.tensor([[fx, 0.0, (W - 1) / 2.0],
                         [0.0, fy, (H - 1) / 2.0],
                         [0.0, 0.0, 1.0]], dtype=torch.float32)


def _da3_get_K(geometry: dict, b: int, H: int, W: int) -> torch.Tensor:
    """Return pixel-space K for batch element b, falling back to a default estimate."""
    if "intrinsics" in geometry:
        # shape (1, S, 3, 3) - leading scene dimension from the multiview head
        return geometry["intrinsics"][0, b].float()
    logging.getLogger("comfy").warning(
        "DA3_GEOMETRY has no intrinsics (mono-mode model). "
        "Using a ~60° FOV estimate; 3-D reconstruction may be inaccurate."
    )
    return _da3_default_K(H, W)


def _da3_get_extrinsic(geometry: dict, b: int) -> torch.Tensor | None:
    """Return the world-to-camera extrinsic for batch element b, or None in mono mode.

    The model outputs (1, S, 3, 4) [R|t] matrices; the fallback identity is (4, 4).
    _da3_apply_extrinsic handles both shapes via [:3, :3] / [:3, 3] slicing.
    """
    if "extrinsics" not in geometry:
        return None
    return geometry["extrinsics"][0, b].float()


def _da3_apply_extrinsic(points_cam: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """Transform (H,W,3) OpenCV camera-space points to world space."""
    E = E.to(points_cam.device).float()
    if not torch.isfinite(E).all():
        logging.getLogger("comfy").warning(
            "DA3 extrinsic matrix contains non-finite values (pose estimation may have failed). "
            "Falling back to camera-space coordinates."
        )
        return points_cam
    H, W, _ = points_cam.shape
    R = E[:3, :3]           # (3, 3) rotation
    t = E[:3, 3]            # (3,)   translation
    R_inv = R.T             # rotation inverse = transpose for orthogonal R
    t_inv = -(R_inv @ t)    # (3,)
    pts = points_cam.reshape(-1, 3)                 # (N, 3)
    pts_world = pts @ R_inv.T + t_inv               # (N, 3)
    return pts_world.reshape(H, W, 3)


def _normalize_confidence(conf: torch.Tensor) -> torch.Tensor:
    """Map raw confidence to [0, 1] per image."""
    B = conf.shape[0]
    out = []
    for i in range(B):
        c = conf[i]
        c_min, c_max = c.min(), c.max()
        out.append((c - c_min) / (c_max - c_min) if c_max > c_min else torch.ones_like(c))
    return torch.stack(out, dim=0)


def _da3_build_mask(geometry: dict, b: int, H: int, W: int, confidence_threshold: float, use_sky_mask: bool) -> torch.Tensor:
    """Build (H,W) bool keep-mask from sky probability and confidence."""
    mask = torch.ones(H, W, dtype=torch.bool)
    if use_sky_mask and "sky" in geometry:
        mask = mask & (geometry["sky"][b] < 0.5)
    if "confidence" in geometry and confidence_threshold > 0.0:
        conf_norm = _normalize_confidence(geometry["confidence"][b:b + 1])[0]
        mask = mask & (conf_norm >= confidence_threshold)
    return mask


class LoadDA3Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadDA3Model",
            display_name="Load Depth Anything 3",
            category="model/loaders",
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=folder_paths.get_filename_list("geometry_estimation"),
                ),
                io.Combo.Input(
                    "weight_dtype",
                    options=["default", "fp16", "bf16", "fp32"],
                    default="default",
                ),
            ],
            outputs=[DA3ModelType.Output()],
        )

    @classmethod
    def execute(cls, model_name, weight_dtype) -> io.NodeOutput:
        model_options = {}
        if weight_dtype == "fp16":
            model_options["dtype"] = torch.float16
        elif weight_dtype == "bf16":
            model_options["dtype"] = torch.bfloat16
        elif weight_dtype == "fp32":
            model_options["dtype"] = torch.float32

        path = folder_paths.get_full_path_or_raise("geometry_estimation", model_name)
        model = comfy.sd.load_diffusion_model(path, model_options=model_options)
        return io.NodeOutput(model)


def _run_da3(model_patcher, image: torch.Tensor, process_res: int, method: str = "upper_bound_resize"):
    """Run DA3 on (B,H,W,3), returns depth/conf/sky at original resolution (or None)."""
    assert image.ndim == 4 and image.shape[-1] == 3, f"expected (B,H,W,3) IMAGE; got {tuple(image.shape)}"

    B, H, W, _ = image.shape
    mm.load_model_gpu(model_patcher)
    diffusion = model_patcher.model.diffusion_model
    device = mm.get_torch_device()
    dtype = diffusion.dtype if diffusion.dtype is not None else torch.float32

    depths, confs, skies = [], [], []
    for i in range(B):
        single = image[i:i + 1].to(device)
        x = da3_preprocess.preprocess_image(single, process_res=process_res, method=method)
        x = x.to(dtype=dtype)
        with torch.no_grad():
            out = diffusion(x)

        depth_lr = out["depth"]
        depth_full = torch.nn.functional.interpolate(
            depth_lr.unsqueeze(1).float(), size=(H, W),
            mode="bilinear", align_corners=False,
        ).squeeze(1).cpu()
        depths.append(depth_full)

        if "depth_conf" in out:
            conf_full = torch.nn.functional.interpolate(
                out["depth_conf"].unsqueeze(1).float(), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(1).cpu()
            confs.append(conf_full)
        if "sky" in out:
            sky_full = torch.nn.functional.interpolate(
                out["sky"].unsqueeze(1).float(), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(1).cpu()
            skies.append(sky_full)

    depth = torch.cat(depths, dim=0)
    confidence = torch.cat(confs, dim=0) if confs else None
    sky = torch.cat(skies, dim=0) if skies else None
    return depth, confidence, sky


class DA3Inference(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DA3Inference",
            search_aliases=["depth", "geometry", "da3", "depth anything", "monocular", "pointmap", "sky", "3d", "metric depth", "disparity"],
            display_name="Run Depth Anything 3",
            category="image/geometry estimation",
            description="Run Depth Anything 3 on an image. In multi-view mode each image is treated as a separate view of the same scene.",
            inputs=[
                DA3ModelType.Input("da3_model"),
                io.Image.Input("image"),
                io.Int.Input("resolution", default=504, min=140, max=2520, step=14,
                    tooltip="Resolution the model runs at (longest side, multiple of 14).\n"
                        "Lower = faster / less VRAM.\n"
                        "Higher = more detail.\n"
                        "Output is upsampled back to the original size."),
                io.Combo.Input("resize_method", options=["upper_bound_resize", "lower_bound_resize"], default="upper_bound_resize",
                    tooltip="upper_bound_resize: scale so the longest side = resolution (caps memory, default).\n"
                        "lower_bound_resize: scale so the shortest side = resolution (preserves more detail on tall/wide images, uses more memory)."),
                io.DynamicCombo.Input("mode", tooltip="mono: single view image (works with any model variant).\n"
                    "multiview: all images processed together for geometric consistency + camera pose (for Small/Base models only).",
                    options=[
                        io.DynamicCombo.Option("mono", []),
                        io.DynamicCombo.Option("multiview", [
                        io.Combo.Input("ref_view_strategy", options=["saddle_balanced", "saddle_sim_range", "first", "middle"], default="saddle_balanced",
                            tooltip="Which view acts as the geometric anchor.\n"
                                "- saddle_balanced: the view most 'average' across all others (best general choice).\n"
                                "- saddle_sim_range: the view most visually distinct from the others.\n"
                                "- first / middle: fixed positional picks."),
                        io.Combo.Input("pose_method", options=["cam_dec", "ray_pose"], default="cam_dec",
                            tooltip="How the camera field-of-view is estimated (for Small/Base models only).\n"
                                "- cam_dec: learned from image features.\n"
                                "- ray_pose: derived geometrically from the model's 3D ray output.\n"
                                "Affects perspective correctness of the 3D output. Try both if results look distorted."),
                    ]),
                ]),
            ],
            outputs=[
                DA3Geometry.Output("da3_geometry", tooltip="Dictionary of non-normalized tensors.\n"
                    "Always has the keys: depth, image, mode.\n"
                    "Optional keys: sky (for Mono/Metric), confidence (for Small/Base), extrinsics + intrinsics (for multi-view)."),
            ],
        )

    @classmethod
    def execute(cls, da3_model, image, resolution, resize_method, mode) -> io.NodeOutput:
        mode_val = mode["mode"]  # "mono" or "multiview"

        if mode_val == "mono":
            return cls._execute_mono(da3_model, image, resolution, resize_method)

        # Capability checks for multi-view mode.
        diffusion = da3_model.model.diffusion_model
        pose_method = mode["pose_method"]
        ref_view_strategy = mode["ref_view_strategy"]

        has_cam_dec = diffusion.cam_dec is not None
        has_dualdpt = diffusion.head_type == "dualdpt"

        if not has_cam_dec and not has_dualdpt:
            raise ValueError(
                "multi-view mode requires Small or Base model. The loaded model "
                f"(head_type='{diffusion.head_type}') does not support cross-view "
                "attention or camera pose estimation. Switch mode to 'mono', or "
                "load Small or Base model for mult-view."
            )

        if pose_method == "cam_dec" and not has_cam_dec:
            raise ValueError(
                "pose_method='cam_dec' requires a camera decoder, but the loaded "
                f"model (head_type='{diffusion.head_type}') does not have one. "
                "Use pose_method='ray_pose' instead."
            )
        if pose_method == "ray_pose" and not has_dualdpt:
            raise ValueError(
                "pose_method='ray_pose' requires a DualDPT head, but the loaded "
                f"model has a '{diffusion.head_type}' head. "
                "Use pose_method='cam_dec' instead."
            )

        return cls._execute_multiview(
            da3_model, image, resolution, resize_method,
            ref_view_strategy, pose_method,
        )

    @classmethod
    def _execute_mono(cls, model, image, resolution, resize_method) -> io.NodeOutput:
        depth, confidence, sky = _run_da3(model, image, resolution, method=resize_method)

        geometry: dict = {
            "depth": depth.contiguous(),
            "image": image[..., :3].cpu(),
            "mode": "mono",
        }
        if sky is not None:
            geometry["sky"] = sky.contiguous()
        if confidence is not None:
            geometry["confidence"] = confidence.contiguous()
        return io.NodeOutput(geometry)

    @classmethod
    def _execute_multiview(cls, model, image, resolution, resize_method, ref_view_strategy, pose_method) -> io.NodeOutput:
        assert image.ndim == 4 and image.shape[-1] == 3, \
            f"expected (B,H,W,3) IMAGE; got {tuple(image.shape)}"
        S, H, W, _ = image.shape

        mm.load_model_gpu(model)
        diffusion = model.model.diffusion_model
        device = mm.get_torch_device()
        dtype = diffusion.dtype if diffusion.dtype is not None else torch.float32

        # All views in a single forward pass: (1, S, 3, H', W').
        x = image.to(device)
        x = da3_preprocess.preprocess_image(x, process_res=resolution, method=resize_method)
        x = x.to(dtype=dtype).unsqueeze(0)

        use_ray_pose = (pose_method == "ray_pose")
        with torch.no_grad():
            out = diffusion(x, use_ray_pose=use_ray_pose, ref_view_strategy=ref_view_strategy)

        depth = torch.nn.functional.interpolate(
            out["depth"].float().unsqueeze(1), size=(H, W),
            mode="bilinear", align_corners=False,
        ).squeeze(1).cpu()

        sky = None
        if "sky" in out:
            sky = torch.nn.functional.interpolate(
                out["sky"].unsqueeze(1).float(), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(1).cpu()

        if "extrinsics" in out and "intrinsics" in out:
            extrinsics = out["extrinsics"].float().cpu()
            intrinsics = out["intrinsics"].float().cpu()
        else:
            extrinsics = torch.eye(4)[None, None].expand(1, S, 4, 4).clone()
            intrinsics = torch.eye(3)[None, None].expand(1, S, 3, 3).clone()

        geometry: dict = {
            "depth": depth.contiguous(),
            "image": image[..., :3].cpu(),
            "mode": "multiview",
            "extrinsics": extrinsics.contiguous(),
            "intrinsics": intrinsics.contiguous(),
        }
        if sky is not None:
            geometry["sky"] = sky.contiguous()
        if "depth_conf" in out:
            conf = torch.nn.functional.interpolate(
                out["depth_conf"].unsqueeze(1).float(), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(1).cpu()
            geometry["confidence"] = conf.contiguous()
        return io.NodeOutput(geometry)


class DA3Render(io.ComfyNode):
    """Render a visualization from a DA3_GEOMETRY packet."""

    _DEPTH_RENDER_INPUTS = [
        io.Combo.Input("normalization",
            options=["v2_style", "min_max", "raw"],
            default="v2_style",
            tooltip="- v2_style: mean/std normalisation for perceptually balanced results (default).\n"
                "- min_max: stretches the full depth range to [0, 1] for maximum contrast.\n"
                "- raw: no scaling,preserves metric units for Metric model."),
        io.Boolean.Input("apply_sky_clip", default=False,
            tooltip="Clip sky-region depth to the 99th percentile of foreground depth before normalisation. "
                "Requires a sky key in the da3_geometry input (for Mono/Metric models only)."),
    ]

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DA3Render",
            display_name="Render Depth Anything 3",
            category="image/geometry estimation",
            description="Render a depth map, confidence map, or sky mask from Depth Anything 3 geometry data.",
            inputs=[
                DA3Geometry.Input("da3_geometry"),
                io.DynamicCombo.Input("output",
                    tooltip="- depth: normalised greyscale depth image.\n"
                        "- depth_colored: depth mapped through the Turbo colormap.\n"
                        "- sky_mask: sky probability in [0, 1] (for Mono/Metric models only).\n"
                        "- confidence: normalised depth confidence (for Small/Base models only).",
                options=[
                    io.DynamicCombo.Option("depth", cls._DEPTH_RENDER_INPUTS),
                    io.DynamicCombo.Option("depth_colored", cls._DEPTH_RENDER_INPUTS),
                    io.DynamicCombo.Option("sky_mask", [
                        io.Boolean.Input("colored", default=False, tooltip="Apply the Turbo colormap to the sky mask."),
                    ]),
                    io.DynamicCombo.Option("confidence", [
                        io.Boolean.Input("colored", default=False, tooltip="Apply the Turbo colormap to the confidence map."),
                    ]),
                ]),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, da3_geometry, output) -> io.NodeOutput:
        output_val = output["output"]

        if output_val in ("depth", "depth_colored"):
            normalization = output["normalization"]
            apply_sky_clip = output["apply_sky_clip"]
            if apply_sky_clip and "sky" not in da3_geometry:
                raise ValueError(
                    "apply_sky_clip=True requires a sky tensor in the da3_geometry input, but none is present. "
                    "Run with Mono/Metric models or set apply_sky_clip=False."
                )
            depth = da3_geometry["depth"]
            sky = da3_geometry.get("sky")
            if apply_sky_clip and sky is not None:
                depth = torch.stack([
                    da3_preprocess.apply_sky_aware_clip(depth[i], sky[i])
                    for i in range(depth.shape[0])
                ], dim=0)
            grey = cls._depth_to_image(depth, sky, normalization)  # (B,H,W,3) greyscale
            result = _turbo(grey[..., 0]) if output_val == "depth_colored" else grey

        elif output_val == "sky_mask":
            if "sky" not in da3_geometry:
                raise ValueError("geometry has no sky output; run with Mono/Metric models.")
            sky = da3_geometry["sky"]
            if output["colored"]:
                result = _turbo(sky)
            else:
                result = sky.unsqueeze(-1).expand(*sky.shape, 3).contiguous()

        elif output_val == "confidence":
            if "confidence" not in da3_geometry:
                raise ValueError("da3_geometry has no confidence output; run with Small/Base models.")
            conf = _normalize_confidence(da3_geometry["confidence"])
            if output["colored"]:
                result = _turbo(conf)
            else:
                result = conf.unsqueeze(-1).expand(*conf.shape, 3).contiguous()

        else:
            raise ValueError(f"Unknown output mode: {output_val}")

        return io.NodeOutput(result.float())

    @staticmethod
    def _depth_to_image(depth: torch.Tensor, sky_for_norm: torch.Tensor | None, normalization: str) -> torch.Tensor:
        """Normalise depth and pack as an (B,H,W,3) image tensor."""

        N = depth.shape[0]
        if normalization == "v2_style":
            norm = torch.stack([
                da3_preprocess.normalize_depth_v2_style(
                    depth[i], sky_for_norm[i] if sky_for_norm is not None else None)
                for i in range(N)
            ], dim=0)
        elif normalization == "min_max":
            norm = da3_preprocess.normalize_depth_min_max(depth)
        else:
            norm = depth

        out = norm.unsqueeze(-1).repeat(1, 1, 1, 3)
        if normalization != "raw":
            out = out.clamp(0.0, 1.0)
        return out.contiguous()


class DA3GeometryToMesh(io.ComfyNode):
    """Convert a DA3_GEOMETRY packet into a Types.MESH by unprojecting depth and triangulating."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DA3GeometryToMesh",
            search_aliases=["da3", "depth anything", "mesh", "geometry", "3d", "triangulate"],
            display_name="Convert DA3 Geometry to Mesh",
            category="image/geometry estimation",
            description="Convert a depth map into a triangulated 3D mesh.",
            inputs=[
                DA3Geometry.Input("da3_geometry"),
                io.Int.Input("batch_index", default=0, min=0, max=4096, tooltip="Which image of a batch to convert. Per-image vertex counts differ so batches cannot be stacked."),
                io.Int.Input("decimation", default=1, min=1, max=8, tooltip="Vertex stride. 1 = full resolution, 2 = half, etc."),
                io.Float.Input("discontinuity_threshold", default=0.04, min=0.0, max=1.0, step=0.01, tooltip="Drop triangles whose 3x3 depth span exceeds this fraction. 0 = off."),
                io.Float.Input("confidence_threshold", default=0.1, min=0.0, max=1.0, step=0.01,
                    tooltip="Exclude pixels whose per-image normalised confidence is below this value (0 = keep all, 1 = keep only the single most confident pixel). "
                        "Used when the geometry has a confidence map (Small/Base models)."),
                io.Boolean.Input("use_sky_mask", default=True, tooltip="Exclude sky-probability pixels (sky >= 0.5) from the mesh. Used when the geometry has a sky map (Mono/Metric models)."),
                io.Boolean.Input("texture", default=True, tooltip="Use the source image as a base color texture."),
            ],
            outputs=[io.Mesh.Output()],
        )

    @classmethod
    def execute(cls, da3_geometry, batch_index, decimation, discontinuity_threshold, confidence_threshold, use_sky_mask, texture) -> io.NodeOutput:
        depth_all = da3_geometry["depth"]   # (B, H, W)
        B = depth_all.shape[0]
        if batch_index >= B:
            raise ValueError(f"batch_index {batch_index} is out of range; DA3_GEOMETRY has batch size {B}.")

        depth = depth_all[batch_index]      # (H, W)
        H, W = depth.shape

        # NaN/inf depth would propagate silently through unproject and produce an
        # empty mesh; replace them with 0 here so those pixels are later excluded
        # by the isfinite check inside triangulate_grid_mesh.
        depth = depth.clone()
        n_bad = (~torch.isfinite(depth)).sum().item()
        if n_bad:
            logging.getLogger("comfy").warning(
                f"DA3GeometryToMesh: depth[{batch_index}] has {n_bad} non-finite pixels "
                f"({100*n_bad/(H*W):.1f}%) - zeroed before unproject."
            )
        depth[~torch.isfinite(depth)] = 0.0
        logging.getLogger("comfy").debug(
            f"DA3GeometryToMesh: depth[{batch_index}] range "
            f"[{depth.min():.4g}, {depth.max():.4g}], mean={depth.mean():.4g}"
        )

        K = _da3_get_K(da3_geometry, batch_index, H, W)
        points = _da3_unproject(depth, K)   # (H, W, 3) in OpenCV camera space

        # Apply world-to-camera inverse so multi-view frames share a common world frame.
        E = _da3_get_extrinsic(da3_geometry, batch_index)
        if E is not None:
            points = _da3_apply_extrinsic(points, E)

        # Mask invalid pixels by setting them to inf so triangulate_grid_mesh skips them.
        mask = _da3_build_mask(da3_geometry, batch_index, H, W, confidence_threshold, use_sky_mask)
        # Also exclude pixels where depth was invalid.
        mask = mask & (depth_all[batch_index] > 0) & torch.isfinite(depth_all[batch_index])
        points = points.clone()
        points[~mask] = float('inf')

        verts, faces, uvs = triangulate_grid_mesh(
            points,
            decimation=decimation,
            discontinuity_threshold=discontinuity_threshold,
            depth=depth,
        )
        if verts.shape[0] == 0 or faces.shape[0] == 0:
            raise ValueError(
                "DA3GeometryToMesh produced an empty mesh. "
                "Try raising discontinuity_threshold, lowering confidence_threshold, "
                "or disabling use_sky_mask."
            )

        # OpenCV (X right, Y down, Z forward) → glTF (X right, Y up, Z back).
        # Same transform as MoGePointMapToMesh perspective branch.
        verts = verts * torch.tensor([1.0, -1.0, -1.0], dtype=verts.dtype)
        faces = faces[:, [0, 2, 1]].contiguous()

        tex = da3_geometry["image"][batch_index:batch_index + 1] if texture else None
        mesh = Types.MESH(
            vertices=verts.unsqueeze(0),
            faces=faces.unsqueeze(0),
            uvs=uvs.unsqueeze(0),
            texture=tex,
        )
        return io.NodeOutput(mesh)


class DA3GeometryToPointCloud(io.ComfyNode):
    """Unproject a DA3_GEOMETRY depth map into a filtered DA3_POINT_CLOUD."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DA3GeometryToPointCloud",
            search_aliases=["da3", "depth anything", "point cloud", "pointcloud", "3d", "geometry"],
            display_name="Convert DA3 Geometry to Point Cloud",
            category="image/geometry estimation",
            description="Convert a depth map into a 3D point cloud.",
            inputs=[
                DA3Geometry.Input("da3_geometry"),
                io.Int.Input("batch_index", default=0, min=0, max=4096, tooltip="Which image of a batch to convert."),
                io.Float.Input("confidence_threshold", default=0.1, min=0.0, max=1.0, step=0.01,
                    tooltip="Exclude pixels whose per-image normalised confidence is below this value (0 = keep all). Used when the geometry has a confidence map (Small/Base models)."),
                io.Boolean.Input("use_sky_mask", default=True,
                    tooltip="Exclude sky-probability pixels (sky >= 0.5). Used when the geometry has a sky map (Mono/Metric models)."),
                io.Int.Input("downsample", default=1, min=1, max=16,
                    tooltip="Take every Nth pixel (1 = full resolution). Higher values give fewer points and faster processing."),
            ],
            # TODO: add a proper PointCloud output type
            outputs=[DA3PointCloud.Output(display_name="point_cloud")],
        )

    @classmethod
    def execute(cls, da3_geometry, batch_index, confidence_threshold, use_sky_mask, downsample) -> io.NodeOutput:
        depth_all = da3_geometry["depth"]   # (B, H, W)
        B = depth_all.shape[0]
        if batch_index >= B:
            raise ValueError(f"batch_index {batch_index} is out of range; DA3_GEOMETRY has batch size {B}.")

        depth = depth_all[batch_index].clone()  # (H, W)
        depth[~torch.isfinite(depth)] = 0.0
        H, W = depth.shape

        K = _da3_get_K(da3_geometry, batch_index, H, W)

        if downsample > 1:
            depth = depth[::downsample, ::downsample].contiguous()
            # Scale intrinsics to the downsampled grid.
            K = K.clone()
            K[0, :] /= downsample
            K[1, :] /= downsample

        H_ds, W_ds = depth.shape
        points = _da3_unproject(depth, K)   # (H_ds, W_ds, 3) in OpenCV camera space

        # Apply world-to-camera inverse so multi-view frames share a common world frame.
        E = _da3_get_extrinsic(da3_geometry, batch_index)
        if E is not None:
            points = _da3_apply_extrinsic(points, E)

        # Rebuild mask at downsampled resolution.
        mask = _da3_build_mask(da3_geometry, batch_index, H, W, confidence_threshold, use_sky_mask)
        if downsample > 1:
            mask = mask[::downsample, ::downsample]

        mask = mask & torch.isfinite(depth)

        # OpenCV → glTF: flip Y and Z.
        points_gltf = points.clone()
        points_gltf[..., 1] *= -1.0
        points_gltf[..., 2] *= -1.0

        pts_flat = points_gltf.reshape(-1, 3)[mask.reshape(-1)]

        colors_flat = None
        if "image" in da3_geometry:
            img = da3_geometry["image"][batch_index]     # (H, W, 3)
            if downsample > 1:
                img = img[::downsample, ::downsample]
            colors_flat = img.reshape(-1, 3)[mask.reshape(-1)]

        conf_flat = None
        if "confidence" in da3_geometry:
            conf = da3_geometry["confidence"][batch_index]   # (H, W)
            if downsample > 1:
                conf = conf[::downsample, ::downsample]
            conf_flat = conf.reshape(-1)[mask.reshape(-1)]

        if pts_flat.shape[0] == 0:
            raise ValueError(
                "DA3GeometryToPointCloud produced zero points after filtering. "
                "Try lowering confidence_threshold or disabling use_sky_mask."
            )

        return io.NodeOutput({
            "points": pts_flat,
            "colors": colors_flat,
            "confidence": conf_flat,
        })


class DA3Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LoadDA3Model,
            DA3Inference,
            DA3Render,
            DA3GeometryToMesh,
            # DA3GeometryToPointCloud,  # Keep this commented out for now until we have a proper PointCloud output type
        ]


async def comfy_entrypoint() -> DA3Extension:
    return DA3Extension()
