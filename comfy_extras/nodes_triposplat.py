# TripoSplat nodes: image -> 3D gaussian splat

import logging

import torch
import torch.nn.functional as F
from typing_extensions import override

import comfy.model_management
import comfy.nested_tensor
import comfy.patcher_extension
import comfy.utils
from comfy_api.latest import ComfyExtension, IO, Types


_Q_TOKEN_LENGTH = 8192
_LATENT_CHANNELS = 16
_CAM_CHANNELS = 5
_DINOV3_MEAN = [0.485, 0.456, 0.406]
_DINOV3_STD = [0.229, 0.224, 0.225]
_NUM_GAUSSIANS_MIN = 32768
_NUM_GAUSSIANS_MAX = 1048576


def _preprocess(image: torch.Tensor, mask: torch.Tensor, erode_radius: int, size: int) -> torch.Tensor:
    # Match original preprocessing:
    # resize min side to `size` -> erode alpha -> alpha bbox -> 1.2x square crop -> resize -> composite on black.
    rgb = image[..., :3].clamp(0, 1).movedim(-1, 0)        # (3, H, W)
    alpha = mask.clamp(0, 1)[None]                         # (1, H, W)
    rgba = torch.cat([rgb, alpha], 0)[None]                # (1, 4, H, W)

    h, w = rgba.shape[-2:]
    s = size / min(w, h)
    rgba = comfy.utils.common_upscale(rgba, max(1, round(w * s)), max(1, round(h * s)), "lanczos", "disabled").clamp(0, 1)

    a = rgba[:, 3:4]
    if erode_radius > 0:
        # min filter over a (2r+1) window == morphological erosion of the alpha matte.
        a = -F.max_pool2d(-a, 2 * erode_radius + 1, stride=1, padding=erode_radius)
        rgba = torch.cat([rgba[:, :3], a], 1)

    ys, xs = torch.nonzero(a[0, 0] > 0, as_tuple=True)
    if xs.numel() == 0:
        raise ValueError("TripoSplatPreprocessImage: mask is empty (no foreground pixels).")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) / 2 * 1.2
    left, upper, right, lower = int(cx - half), int(cy - half), int(cx + half), int(cy + half)

    H, W = rgba.shape[-2:]
    crop = rgba.new_zeros((1, 4, lower - upper, right - left))  # out-of-bounds stays 0, matching PIL.crop
    sx0, sy0, sx1, sy1 = max(left, 0), max(upper, 0), min(right, W), min(lower, H)
    if sx1 > sx0 and sy1 > sy0:
        crop[:, :, sy0 - upper:sy1 - upper, sx0 - left:sx1 - left] = rgba[:, :, sy0:sy1, sx0:sx1]

    crop = comfy.utils.common_upscale(crop, size, size, "lanczos", "disabled").clamp(0, 1)
    out = (crop[:, :3] * crop[:, 3:4])[0].movedim(0, -1)   # composite over black == rgb * alpha
    return out.unsqueeze(0)  # (1, 1024, 1024, 3)


class TripoSplatPreprocessImage(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoSplatPreprocessImage",
            display_name="TripoSplat Preprocess Image",
            category="model/conditioning/triposplat",
            description="Crop center each image to a square canvas on a black background and add padding.",
            inputs=[
                IO.Image.Input("image"),
                IO.Mask.Input("mask"),
                IO.Int.Input("erode_radius", default=1, min=0, max=16,
                             tooltip="Erode the alpha matte by this pixel radius before cropping (avoids border bleed)."),
                IO.Int.Input("size", default=1024, min=256, max=4096, step=16,
                             tooltip="Square image size. The model is trained at 1024; other sizes run but are off-distribution."),
            ],
            outputs=[IO.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(cls, image, mask, erode_radius, size) -> IO.NodeOutput:
        size = max(16, (int(size) // 16) * 16)  # DINOv3 patch / Flux2 VAE stride is 16
        if mask.shape[0] != image.shape[0]:
            mask = comfy.utils.repeat_to_batch_size(mask, image.shape[0])
        if tuple(mask.shape[1:]) != tuple(image.shape[1:3]):
            mask = F.interpolate(mask[:, None].float(), size=tuple(image.shape[1:3]), mode="bilinear", align_corners=False)[:, 0]
        prepared = torch.cat([_preprocess(image[i], mask[i], erode_radius, size) for i in range(image.shape[0])], dim=0)
        return IO.NodeOutput(prepared)


class TripoSplatConditioning(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoSplatConditioning",
            display_name="TripoSplat Conditioning",
            category="model/conditioning/triposplat",
            description="Encode the image with DINOv3 and the Flux2 VAE into TripoSplat positive/negative "
                        "conditioning, and create the fixed size noise target (latent + camera) for the KSampler",
            inputs=[
                IO.ClipVision.Input("clip_vision", tooltip="DINOv3 ViT-H/16+ image encoder"),
                IO.Vae.Input("vae", tooltip="Flux2 VAE"),
                IO.Image.Input("image"),
            ],
            outputs=[
                IO.Conditioning.Output(display_name="positive"),
                IO.Conditioning.Output(display_name="negative"),
                IO.Latent.Output(display_name="latent", tooltip="The fixed size noise target (latent +camera)."),
            ],
        )

    @classmethod
    def execute(cls, clip_vision, vae, image) -> IO.NodeOutput:
        # feature1: DINOv3 token sequence (cls + registers + patches), ImageNet-normalized, with a final non-affine layer norm on top
        comfy.model_management.load_model_gpu(clip_vision.patcher)
        device = clip_vision.load_device
        img = image.movedim(-1, 1).to(device)  # (B,3,H,W) in [0,1]
        mean = torch.tensor(_DINOV3_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(_DINOV3_STD, device=device).view(1, 3, 1, 1)
        img = (img - mean) / std
        seq = clip_vision.model(pixel_values=img.float())[0]
        feature1 = F.layer_norm(seq.float(), seq.shape[-1:]).to(comfy.model_management.intermediate_device())

        # Second conditioning: the Flux2 VAE latent of the image, carried as a standard reference_latents entry
        ref = vae.encode(image).to(comfy.model_management.intermediate_device())  # (B, 128, H, W)
        b = ref.shape[0]

        positive = [[feature1, {"reference_latents": [ref]}]]
        negative = [[torch.zeros_like(feature1), {"reference_latents": [torch.zeros_like(ref)]}]]

        # Fixed noise target: the latent is a constant-shape (8192, 16) shape-code + a (1, 5) camera token
        dev = comfy.model_management.intermediate_device()
        latent_seq = torch.zeros([b, _Q_TOKEN_LENGTH, _LATENT_CHANNELS], device=dev)
        camera = torch.zeros([b, 1, _CAM_CHANNELS], device=dev)
        samples = comfy.nested_tensor.NestedTensor((latent_seq, camera))
        return IO.NodeOutput(positive, negative, {"samples": samples})


class VAEDecodeTripoSplat(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VAEDecodeTripoSplat",
            display_name="TripoSplat Decode",
            category="model/latent/triposplat",
            description="Decode the sampled TripoSplat latent into a 3D gaussian splat. "
                        "Modify the number of gaussians to vary the density.",
            inputs=[
                IO.Latent.Input("samples"),
                IO.Vae.Input("vae", tooltip="TripoSplat VAE decoder"),
                IO.Int.Input("num_gaussians", default=262144, min=_NUM_GAUSSIANS_MIN, max=_NUM_GAUSSIANS_MAX, step=32,
                             tooltip="Number of gaussians to produce (rounded to a multiple of 32). "
                                     "262144 matches the octree's point density; higher oversamples the same points "
                                     "(denser, but no new detail) and costs proportionally more VRAM/time."),
                IO.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff,
                             tooltip="Seeds the octree point sampler (global RNG) for deterministic decodes."),
            ],
            outputs=[IO.Splat.Output(display_name="splat")],
        )

    @classmethod
    def execute(cls, samples, vae, num_gaussians, seed) -> IO.NodeOutput:
        s = samples["samples"]
        latent = s.unbind()[0] if getattr(s, "is_nested", False) else s  # take the latent stream, drop camera

        decoder = vae.first_stage_model
        gpp = decoder.gaussians_per_point
        n = max(_NUM_GAUSSIANS_MIN, min(_NUM_GAUSSIANS_MAX, int(num_gaussians)))
        if n % gpp != 0:
            n = round(n / gpp) * gpp

        dtype_size = comfy.model_management.dtype_size(vae.vae_dtype)
        hidden = decoder.gs.model_channels
        cond_tokens = latent.shape[1]
        memory_required = (cond_tokens * 4 + (n // gpp) * 10) * hidden * dtype_size
        comfy.model_management.load_models_gpu([vae.patcher], memory_required=memory_required)
        latent = latent.to(device=vae.device, dtype=vae.vae_dtype)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        parts = [g.render_tensors() for g in decoder.decode(latent, num_gaussians=n, generator=generator)]
        positions, scales, rotations, opacities, sh = (torch.stack(t) for t in zip(*parts))
        return IO.NodeOutput(Types.SPLAT(positions, scales, rotations, opacities, sh))


class TripoSplatSamplingPreview(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TripoSplatSamplingPreview",
            display_name="TripoSplat Sampling Preview",
            category="model/latent/triposplat",
            description="Patch the TripoSplat model for the standard Ksampler node to show a live decoded "
                        "gaussian splat preview at each step.",
            inputs=[
                IO.Model.Input("model"),
                IO.Vae.Input("vae", tooltip="TripoSplat VAE decoder"),
                IO.Int.Input("octree_level", default=5, min=2, max=8, advanced=True,
                             tooltip="Octree depth for the preview decode (lower = cheaper/coarser)."),
                IO.Int.Input("num_gaussians", default=16384, min=1024, max=262144, step=32,
                             tooltip="Number of gaussians to produce for the preview (rounded to a multiple of 32)."),
                IO.Float.Input("yaw", default=90.0, min=-360.0, max=360.0, step=1.0, tooltip="Preview camera yaw in degrees.", advanced=True,),
                IO.Float.Input("pitch", default=15.0, min=-89.0, max=89.0, step=1.0, tooltip="Preview camera pitch in degrees.", advanced=True,),
                IO.Int.Input("point_size", default=3, min=1, max=16,
                             tooltip="Maximum splat radius in pixels. Each gaussian is sized from its scale and capped here; "
                                     "lower = finer/pointier, higher = chunkier."),
            ],
            outputs=[IO.Model.Output()],
        )

    @classmethod
    def execute(cls, model, vae, octree_level, num_gaussians, yaw, pitch, point_size) -> IO.NodeOutput:
        from comfy.ldm.triposplat.preview import decode_x0_to_image
        cfg = {"gaussians": num_gaussians, "level": octree_level, "yaw": yaw, "pitch": pitch,
               "point_size": point_size}

        fsm = vae.first_stage_model
        cond_tokens = model.model.diffusion_model.q_token_length
        memory_required = (cond_tokens * 4 + (num_gaussians // fsm.gaussians_per_point) * 10) * fsm.gs.model_channels * comfy.model_management.dtype_size(vae.vae_dtype)

        # Live preview via WrappersMP.OUTER_SAMPLE + ProgressBar
        # The wrapper augments the sampler's own callback to decode x0 -> gaussian splat -> preview image each step
        def outer_sample_wrapper(executor, *args, **kwargs):
            args = list(args)
            cb_idx = 5  # outer_sample(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
            orig_cb = args[cb_idx] if len(args) > cb_idx else kwargs.get("callback")
            state = {"ok": True, "pbar": None, "loaded": False}

            def callback(step, x0, x, total_steps):
                if orig_cb is not None:
                    orig_cb(step, x0, x, total_steps)
                if not state["ok"]:
                    return
                try:
                    if not state["loaded"]:
                        loaded_models = comfy.model_management.loaded_models(only_currently_used=True)
                        loaded_models.append(vae.patcher)
                        comfy.model_management.load_models_gpu(loaded_models, memory_required=memory_required)
                        state["loaded"] = True
                    img = decode_x0_to_image(vae, x0, cfg)
                    if state["pbar"] is None:
                        state["pbar"] = comfy.utils.ProgressBar(total_steps)
                    state["pbar"].update_absolute(step + 1, total_steps, ("JPEG", img, 512))
                except Exception as e:
                    logging.warning("TripoSplatSamplingPreview: preview failed, disabling ({})".format(e))
                    state["ok"] = False

            if len(args) > cb_idx:
                args[cb_idx] = callback
            else:
                kwargs["callback"] = callback
            return executor(*args, **kwargs)

        m = model.clone()
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "triposplat_sampling_preview", outer_sample_wrapper)
        return IO.NodeOutput(m)


class TripoSplatExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            TripoSplatPreprocessImage,
            TripoSplatConditioning,
            VAEDecodeTripoSplat,
            TripoSplatSamplingPreview,
        ]


async def comfy_entrypoint() -> TripoSplatExtension:
    return TripoSplatExtension()
