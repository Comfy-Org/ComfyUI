"""Native ComfyUI nodes for Hunyuan3D 2.1 PBR paint (multiview texture generation).

The paint UNet is a first-class comfy model (detected by comfy.model_detection,
sampled by the standard KSampler machinery). Given a mesh and a reference image,
the conditioning node renders the geometry conditioning (world-space normal +
position maps) with a torch-native rasterizer, precomputes the reference-attention
bank, and emits standard CONDITIONING plus a packed multiview LATENT; after
sampling, the split node unpacks per-view albedo and metallic-roughness latents
for a standard VAEDecode, and the bake node back-projects the decoded views onto
a UV atlas to produce the textured mesh.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing_extensions import override

import comfy.model_management
import comfy.utils
import folder_paths
from comfy.ldm.hunyuan3d.paint import render as paint_render
from comfy.ldm.hunyuan3d.paint.loader import load_paint_unet
from comfy.ldm.hunyuan3d.paint.unet import PaintReferenceBank
from comfy_extras.nodes_save_3d import get_mesh_batch_item
from comfy_api.latest import ComfyExtension, IO, Types

DINO_FEATURES = IO.Custom("HY3D_DINO_FEATURES")
CAMERAS = IO.Custom("HY3D_CAMERAS")


def _first_mesh(mesh):
    """Return (vertices (N,3), faces (M,3) long) for batch item 0 of a MESH."""
    vertices, faces, _colors, uvs, _normals = get_mesh_batch_item(mesh, 0)
    return vertices.float(), faces.long(), uvs


def _prep_reference(image, size):
    """White-composite (if RGBA) and resize a reference IMAGE to (size, size, 3)."""
    img = image[:1]
    if img.shape[-1] == 4:
        rgb, alpha = img[..., :3], img[..., 3:4]
        img = rgb * alpha + (1.0 - alpha)
    else:
        img = img[..., :3]
    img = img.permute(0, 3, 1, 2)
    img = F.interpolate(img, size=(size, size), mode="bilinear", align_corners=False)
    return img.permute(0, 2, 3, 1).contiguous()


def _pack_mr_gltf(mr_native):
    """Repack the model's native MR image (R=metallic, G=roughness) into the glTF
    metallicRoughness layout (R=1, G=roughness, B=metallic)."""
    out = torch.ones_like(mr_native)
    out[..., 1] = mr_native[..., 1]  # roughness -> G
    out[..., 2] = mr_native[..., 0]  # metallic  -> B
    return out


def view_scale_mapping(azimuth):
    """Per-view CFG multiplier as a function of the view azimuth (degrees).

    The front view (azimuth 0) follows the reference image closely and needs
    little guidance; side/back views are underconstrained and get double
    guidance, with linear ramps of slope 1/90 near the front. The function is
    intentionally discontinuous at 330 degrees (trained behavior).
    """
    a = azimuth % 360.0
    if a < 90.0:
        return 1.0 + a / 90.0
    if a < 330.0:
        return 2.0
    return 4.0 / 3.0 - (a - 330.0) / 90.0


def trailing_timesteps(num_steps, num_train_timesteps=1000):
    """DDIM "trailing" timestep spacing (diffusers ``timestep_spacing: trailing``),
    e.g. [999, 932, ..., 66] for 15 steps over a 1000-step train schedule."""
    step_ratio = num_train_timesteps / num_steps
    return np.round(np.arange(num_train_timesteps, 0, -step_ratio)).astype(np.int64) - 1


def _view_scale_pre_cfg(azims):
    """Pre-CFG function applying the reference pipeline's per-view guidance scale.

    The reference composes ``uncond + s*vs*(ref - uncond) + s*vs*(full - ref)``,
    which telescopes exactly to ``uncond + s*vs*(full - uncond)`` (the ref-only
    middle batch cancels; guidance_rescale defaults to 0 and is never set) - i.e.
    standard 2-cond CFG with a per-view scale ``vs`` on the packed view axis.
    Replacing cond with ``uncond + vs*(cond - uncond)`` before the scalar CFG
    reproduces that composition and stays stackable with RescaleCFG."""
    def pre_cfg(args):
        conds_out = args["conds_out"]
        if len(conds_out) < 2 or conds_out[1] is None:
            return conds_out  # cfg == 1.0: no guidance to scale
        cond, uncond = conds_out[0], conds_out[1]
        if cond.ndim != 5 or cond.shape[2] % len(azims) != 0:
            raise ValueError(
                f"Hunyuan3D paint: expected a packed 5D latent whose view axis is a multiple of "
                f"{len(azims)} (the camera count), got shape {tuple(cond.shape)}. Falling back to "
                f"scalar CFG here would silently drop per-view guidance and skew the result."
            )
        vs = torch.tensor([view_scale_mapping(a) for a in azims], device=cond.device, dtype=cond.dtype)
        vs = vs.repeat(cond.shape[2] // len(azims)).reshape(1, 1, -1, 1, 1)
        return [uncond + vs * (cond - uncond)] + list(conds_out[1:])
    return pre_cfg


class Hunyuan3DPaintModelLoader(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Hunyuan3DPaintModelLoader",
            display_name="Hunyuan3D Paint Model Loader",
            category="loaders/hunyuan 3d",
            description=(
                "Load a Hunyuan3D 2.1 PBR paint UNet (hunyuan3d-paintpbr-v2-1) as a "
                "standard MODEL. Equivalent to the generic diffusion-model loader, "
                "with paint-specific error diagnosis for wrong-family checkpoints."
            ),
            inputs=[
                IO.Combo.Input("model_name", options=folder_paths.get_filename_list("diffusion_models")),
            ],
            outputs=[
                IO.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, model_name) -> IO.NodeOutput:
        path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        sd = comfy.utils.load_torch_file(path)
        return IO.NodeOutput(load_paint_unet(sd))


class Hunyuan3DPaintConditioning(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Hunyuan3DPaintConditioning",
            display_name="Hunyuan3D Paint Conditioning",
            category="conditioning/3d_models",
            description=(
                "Prepare everything the 2.1 paint model needs for a standard KSampler run: "
                "renders the mesh's geometry conditioning (normal + position maps), encodes "
                "it with the VAE, precomputes the reference-attention bank from the reference "
                "image (dual-stream write pass), and packs it all into CONDITIONING plus an "
                "empty multiview LATENT. Also outputs the model with the reference pipeline's "
                "per-view guidance scaling attached."
            ),
            inputs=[
                IO.Model.Input("model", tooltip="The paint model (Hunyuan3D Paint Model Loader or Load Diffusion Model)."),
                IO.Vae.Input("vae", tooltip="SD-2.x image VAE from the paint model (hunyuan3d-paintpbr-v2-1/vae)."),
                IO.Mesh.Input("mesh", tooltip="Untextured mesh to paint (e.g. from Hunyuan3D 2.1 shape)."),
                IO.Image.Input("reference_image", tooltip="Reference image; alpha is composited over white."),
                IO.Int.Input("num_views", default=6, min=1, max=6,
                             tooltip="Number of standard views (front, right, back, left, top, bottom)."),
                IO.Int.Input("resolution", default=512, min=256, max=1024, step=64,
                             tooltip="Per-view render/diffusion resolution."),
                DINO_FEATURES.Input("dino_features", optional=True,
                                    tooltip="Optional precomputed DINOv2 reference tokens (improves fidelity)."),
            ],
            outputs=[
                IO.Model.Output(display_name="model"),
                IO.Conditioning.Output(display_name="positive"),
                IO.Conditioning.Output(display_name="negative"),
                IO.Latent.Output(display_name="latent"),
                CAMERAS.Output(display_name="cameras"),
                IO.Image.Output(display_name="normal_maps"),
                IO.Image.Output(display_name="position_maps"),
            ],
        )

    @classmethod
    def execute(cls, model, vae, mesh, reference_image, num_views, resolution,
                dino_features=None) -> IO.NodeOutput:
        base = model.model  # comfy.model_base.Hunyuan3DPaint
        dm = base.diffusion_model
        n_pbr = len(dm.pbr_setting)

        vertices, faces, _uvs = _first_mesh(mesh)
        cameras = paint_render.standard_cameras(num_views)

        normal_maps, position_maps, _masks = paint_render.render_geometry_maps(
            vertices, faces, cameras, resolution=resolution)
        # CPU float for the VAE, which handles its own device placement
        normal_maps = normal_maps.cpu().float()
        position_maps = position_maps.cpu().float()

        reference = _prep_reference(reference_image, resolution)
        ref_latent = vae.encode(reference[:, :, :, :3])
        normal_latents = vae.encode(normal_maps[:, :, :, :3])
        position_latents = vae.encode(position_maps[:, :, :, :3])

        # reference-attention bank: one dual-stream write pass, reused every step
        comfy.model_management.load_models_gpu([model])
        device = model.load_device
        dtype = base.get_dtype_inference()
        ref_in = base.process_latent_in(ref_latent).unsqueeze(1).to(device=device, dtype=dtype)
        bank = PaintReferenceBank(dm.compute_reference_bank(ref_in))

        # the model's learned per-material embeddings ride as regular cross attn
        context = dm.material_context(1).detach().float().cpu()
        context = context.reshape(1, -1, context.shape[-1])  # (1, n_pbr*L, cross)

        # geometry latents -> packed (1, 8, n_pbr*V, h, w), identical per material
        geo = torch.cat([normal_latents.cpu().float(), position_latents.cpu().float()], dim=1)
        geo = geo.movedim(0, 1).unsqueeze(0).repeat(1, 1, n_pbr, 1, 1)

        pos_maps = position_maps[:, :, :, :3].permute(0, 3, 1, 2).unsqueeze(0).contiguous()

        cond = {
            "concat_latent_image": geo,
            "ref_bank": bank,
            "position_maps": pos_maps,
            "ref_scale": 1.0,
        }
        uncond = dict(cond)
        uncond["ref_scale"] = 0.0
        if dino_features is not None:
            dino = dino_features.detach().float().cpu()
            cond["dino_features"] = dino
            uncond["dino_features"] = torch.zeros_like(dino)
        positive = [[context, cond]]
        negative = [[context, uncond]]

        latent = torch.zeros((1, 4, n_pbr * num_views, resolution // 8, resolution // 8),
                             device=comfy.model_management.intermediate_device())

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(_view_scale_pre_cfg(list(cameras.azims)))

        return IO.NodeOutput(m, positive, negative, {"samples": latent}, cameras,
                             normal_maps, position_maps)


class Hunyuan3DPaintScheduler(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Hunyuan3DPaintScheduler",
            display_name="Hunyuan3D Paint Scheduler",
            category="sampling/custom_sampling/schedulers",
            description=(
                "Exact DDIM trailing-spacing sigmas for the paint model's zero-terminal-SNR "
                "schedule (integer-timestep table lookups; sgm_uniform is a close but inexact "
                "approximation). Use with SamplerCustomAdvanced/CFGGuider and an euler sampler "
                "(DDIM eta=0 == Euler on this schedule)."
            ),
            inputs=[
                IO.Model.Input("model"),
                IO.Int.Input("steps", default=15, min=1, max=1000),
            ],
            outputs=[
                IO.Sigmas.Output(display_name="sigmas"),
            ],
        )

    @classmethod
    def execute(cls, model, steps) -> IO.NodeOutput:
        ms = model.get_model_object("model_sampling")
        ts = trailing_timesteps(steps, num_train_timesteps=len(ms.sigmas))
        sigmas = ms.sigma(torch.tensor(ts, dtype=torch.float32))
        sigmas = torch.cat([sigmas.cpu(), torch.zeros(1)])
        return IO.NodeOutput(sigmas)


class Hunyuan3DPaintSplitLatent(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Hunyuan3DPaintSplitLatent",
            display_name="Hunyuan3D Paint Split Latent",
            category="latent/3d",
            description=(
                "Split a sampled packed multiview paint latent (B, 4, n_pbr*V, h, w) into "
                "per-view albedo and metallic-roughness LATENT batches for VAEDecode."
            ),
            inputs=[
                IO.Model.Input("model", tooltip="The paint model (defines the material packing)."),
                IO.Latent.Input("samples"),
            ],
            outputs=[
                IO.Latent.Output(display_name="albedo"),
                IO.Latent.Output(display_name="mr"),
            ],
        )

    @classmethod
    def execute(cls, model, samples) -> IO.NodeOutput:
        pbr_setting = list(model.model.diffusion_model.pbr_setting)
        latent = samples["samples"]
        if latent.ndim != 5 or latent.shape[2] % len(pbr_setting) != 0:
            raise ValueError(
                f"expected a packed paint latent (B, C, {len(pbr_setting)}*V, h, w), got {tuple(latent.shape)}")
        views = latent.shape[2] // len(pbr_setting)
        frames = latent[0].movedim(1, 0)  # (n_pbr*V, C, h, w), material-major
        albedo = frames[:views]
        if "mr" in pbr_setting:
            mr = frames[pbr_setting.index("mr") * views:][:views]
        else:
            mr = torch.zeros_like(albedo)
        return IO.NodeOutput({"samples": albedo}, {"samples": mr})


class Hunyuan3DBakeMultiView(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Hunyuan3DBakeMultiView",
            display_name="Hunyuan3D Bake MultiView",
            category="model/hunyuan 3d",
            description=(
                "Back-project multiview albedo + metallic-roughness images onto the mesh's UV "
                "atlas and attach the baked textures. A mesh that already has UVs (artist-made "
                "or from an unwrap node) is baked onto those UVs; a mesh without UVs falls back "
                "to a built-in per-triangle atlas. Outputs the textured mesh and the baked "
                "albedo/MR textures as images."
            ),
            inputs=[
                IO.Mesh.Input("mesh", tooltip="The mesh that was painted (same one fed to the conditioning node). "
                                              "Existing per-vertex UVs are used as the bake target; a mesh "
                                              "without UVs gets a built-in per-triangle atlas."),
                IO.Image.Input("albedo", tooltip="Per-view albedo images (VAE-decoded albedo latents)."),
                IO.Image.Input("mr", tooltip="Per-view metallic-roughness images (VAE-decoded MR latents)."),
                CAMERAS.Input("cameras", tooltip="Cameras output by the conditioning node."),
                IO.Int.Input("texture_size", default=1024, min=256, max=4096, step=256),
                IO.Float.Input("bake_exponent", default=4.0, min=1.0, max=16.0, step=0.5, advanced=True,
                               tooltip="Cosine weighting exponent; higher favours front-facing views."),
                IO.Boolean.Input("fill_holes", default=True, advanced=True,
                                 tooltip="Fill unseen texels by dilating nearest valid colours."),
            ],
            outputs=[
                IO.Mesh.Output(display_name="mesh"),
                IO.Image.Output(display_name="albedo_texture"),
                IO.Image.Output(display_name="mr_texture"),
                IO.Image.Output(display_name="albedo_views"),
                IO.Image.Output(display_name="mr_views"),
            ],
        )

    @classmethod
    def execute(cls, mesh, albedo, mr, cameras, texture_size, bake_exponent, fill_holes) -> IO.NodeOutput:
        vertices, faces, uvs = _first_mesh(mesh)

        # A mesh that already carries per-vertex UVs (artist-authored or produced by an
        # unwrap node) keeps them as the bake target; only UV-less meshes fall back to
        # the built-in per-triangle atlas.
        if uvs is not None and uvs.shape[0] == vertices.shape[0]:
            v_uv, f_uv, uv = vertices, faces, uvs.float()
        else:
            v_uv, f_uv, uv = paint_render.pack_per_triangle_uv(vertices, faces)

        albedo_views = albedo[..., :3].float()
        mr_views = mr[..., :3].float()

        # Bake both maps in one pass over the stacked channels. Rasterization and UV
        # interpolation depend only on the geometry and cameras, which are shared, so
        # baking albedo and MR separately repeats the expensive half of the work; the
        # accumulators are per-channel, so splitting the result afterwards is exact.
        baked, mask = paint_render.bake_multiview(
            v_uv, f_uv, uv, torch.cat([albedo_views, mr_views], dim=-1),
            cameras, texture_size=texture_size, bake_exp=bake_exponent)
        albedo_tex, mr_tex = baked[..., :3], baked[..., 3:]

        if fill_holes:
            albedo_tex = paint_render.fill_holes(albedo_tex, mask)
            mr_tex = paint_render.fill_holes(mr_tex, mask)

        albedo_tex = albedo_tex.clamp(0.0, 1.0)
        mr_tex = mr_tex.clamp(0.0, 1.0)

        out_mesh = Types.MESH(
            v_uv.unsqueeze(0), f_uv.unsqueeze(0), uvs=uv.unsqueeze(0),
            texture=albedo_tex.unsqueeze(0),
            texture_mr=_pack_mr_gltf(mr_tex).unsqueeze(0))

        return IO.NodeOutput(out_mesh, albedo_tex.unsqueeze(0), mr_tex.unsqueeze(0), albedo, mr)


class Hunyuan3DPaintExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            Hunyuan3DPaintModelLoader,
            Hunyuan3DPaintConditioning,
            Hunyuan3DPaintScheduler,
            Hunyuan3DPaintSplitLatent,
            Hunyuan3DBakeMultiView,
        ]


async def comfy_entrypoint() -> Hunyuan3DPaintExtension:
    return Hunyuan3DPaintExtension()
