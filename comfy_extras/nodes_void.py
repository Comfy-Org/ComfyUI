import logging

import torch

import comfy
import comfy.model_management
import comfy.model_patcher
import comfy.samplers
import comfy.utils
import folder_paths
import node_helpers
import nodes
from comfy.utils import model_trange as trange
from comfy_api.latest import ComfyExtension, io
from torchvision.models.optical_flow import raft_large
from typing_extensions import override


from comfy_extras.void_noise_warp import RaftOpticalFlow, get_noise_from_video

OpticalFlow = io.Custom("OPTICAL_FLOW")

TEMPORAL_COMPRESSION = 4
PATCH_SIZE_T = 2


def _valid_void_length(length: int) -> int:
    """Round ``length`` down to a value that produces an even latent_t.

    VOID / CogVideoX-Fun-V1.5 uses patch_size_t=2, so the VAE-encoded latent
    must have an even temporal dimension. If latent_t is odd, the transformer
    pad_to_patch_size circular-wraps an extra latent frame onto the end; after
    the post-transformer crop the last real latent frame has been influenced
    by the wrapped phantom frame, producing visible jitter and "disappearing"
    subjects near the end of the decoded video. Rounding down fixes this.
    """
    latent_t = ((length - 1) // TEMPORAL_COMPRESSION) + 1
    if latent_t % PATCH_SIZE_T == 0:
        return length
    # Round latent_t down to the nearest multiple of PATCH_SIZE_T, then invert
    # the ((length - 1) // TEMPORAL_COMPRESSION) + 1 formula. Floor at 1 frame
    # so we never return a non-positive length.
    target_latent_t = max(PATCH_SIZE_T, (latent_t // PATCH_SIZE_T) * PATCH_SIZE_T)
    return (target_latent_t - 1) * TEMPORAL_COMPRESSION + 1


class OpticalFlowLoader(io.ComfyNode):
    """Load an optical flow model from ``models/optical_flow/``.

    Only torchvision's RAFT-large format is recognized today (the model used
    by VOIDWarpedNoise).  The checkpoint must be placed under
    ``models/optical_flow/`` — ComfyUI never downloads optical-flow weights
    at runtime.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OpticalFlowLoader",
            display_name="Load Optical Flow Model",
            category="model/loaders",
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=folder_paths.get_filename_list("optical_flow"),
                    tooltip=(
                        "Optical flow model to load.  Files must be placed in the "
                        "'optical_flow' folder.  Today only torchvision's "
                        "raft_large.pth is supported."
                    ),
                ),
            ],
            outputs=[
                OpticalFlow.Output(),
            ],
        )

    @classmethod
    def execute(cls, model_name) -> io.NodeOutput:

        model_path = folder_paths.get_full_path_or_raise("optical_flow", model_name)
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)

        has_raft_keys = (
            any(k.startswith("feature_encoder.") for k in sd)
            and any(k.startswith("context_encoder.") for k in sd)
            and any(k.startswith("update_block.") for k in sd)
        )
        if not has_raft_keys:
            raise ValueError(
                "Unrecognized optical flow model format: expected a torchvision "
                "RAFT-large state dict with 'feature_encoder.', 'context_encoder.' "
                "and 'update_block.' prefixes."
            )

        model = raft_large(weights=None, progress=False)
        model.load_state_dict(sd)
        model.eval().to(torch.float32)

        patcher = comfy.model_patcher.ModelPatcher(
            model,
            load_device=comfy.model_management.get_torch_device(),
            offload_device=comfy.model_management.unet_offload_device(),
        )
        return io.NodeOutput(patcher)


class VOIDQuadmaskPreprocess(io.ComfyNode):
    """Preprocess a quadmask video for VOID inpainting.

    Quantizes mask values to four semantic levels, inverts, and normalizes:
      0   -> primary object to remove
      63  -> overlap of primary + affected
      127 -> affected region (interactions)
      255 -> background (keep)

    After inversion and normalization, the output mask has values in [0, 1]
    with four discrete levels: 1.0 (remove), ~0.75, ~0.50, 0.0 (keep).
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VOIDQuadmaskPreprocess",
            display_name="VOID Quadmask Preprocessor",
            category="image/mask",
            inputs=[
                io.Mask.Input("mask"),
                io.Int.Input("dilate_width", default=0, min=0, max=50, step=1,
                             tooltip="Dilation radius for the primary mask region (0 = no dilation)"),
            ],
            outputs=[
                io.Mask.Output(display_name="quadmask"),
            ],
        )

    @classmethod
    def execute(cls, mask, dilate_width=0) -> io.NodeOutput:
        m = mask.clone()

        if m.max() <= 1.0:
            m = m * 255.0

        if dilate_width > 0 and m.ndim >= 3:
            binary = (m < 128).float()
            kernel_size = dilate_width * 2 + 1
            if binary.ndim == 3:
                binary = binary.unsqueeze(1)
            dilated = torch.nn.functional.max_pool2d(
                binary, kernel_size=kernel_size, stride=1, padding=dilate_width
            )
            if dilated.ndim == 4:
                dilated = dilated.squeeze(1)
            m = torch.where(dilated > 0.5, torch.zeros_like(m), m)

        m = torch.where(m <= 31, torch.zeros_like(m), m)
        m = torch.where((m > 31) & (m <= 95), torch.full_like(m, 63), m)
        m = torch.where((m > 95) & (m <= 191), torch.full_like(m, 127), m)
        m = torch.where(m > 191, torch.full_like(m, 255), m)

        m = (255.0 - m) / 255.0

        return io.NodeOutput(m)


class VOIDInpaintConditioning(io.ComfyNode):
    """Build VOID inpainting conditioning for CogVideoX.

    Encodes the processed quadmask and masked source video through the VAE,
    producing a 32-channel concat conditioning (16ch mask + 16ch masked video)
    that gets concatenated with the 16ch noise latent by the model.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VOIDInpaintConditioning",
            category="model/conditioning/void",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Image.Input("video", tooltip="Source video frames [T, H, W, 3]"),
                io.Mask.Input("quadmask", tooltip="Preprocessed quadmask from VOIDQuadmaskPreprocess [T, H, W]"),
                io.Int.Input("width", default=672, min=16, max=nodes.MAX_RESOLUTION, step=8),
                io.Int.Input("height", default=384, min=16, max=nodes.MAX_RESOLUTION, step=8),
                io.Int.Input("length", default=45, min=1, max=nodes.MAX_RESOLUTION, step=1,
                             tooltip="Number of pixel frames to process. For CogVideoX-Fun-V1.5 "
                                     "(patch_size_t=2), latent_t must be even — lengths that "
                                     "produce odd latent_t are rounded down (e.g. 49 → 45)."),
                io.Int.Input("batch_size", default=1, min=1, max=64),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, video, quadmask,
                width, height, length, batch_size) -> io.NodeOutput:

        adjusted_length = _valid_void_length(length)
        if adjusted_length != length:
            logging.warning(
                "VOIDInpaintConditioning: rounding length %d down to %d so that "
                "latent_t is even (required by CogVideoX-Fun-V1.5 patch_size_t=2). "
                "Using odd latent_t causes the last frame to be corrupted by "
                "circular padding.", length, adjusted_length,
            )
            length = adjusted_length

        latent_t = ((length - 1) // TEMPORAL_COMPRESSION) + 1
        latent_h = height // 8
        latent_w = width // 8

        vid = video[:length]
        vid = comfy.utils.common_upscale(
            vid.movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)

        qm = quadmask[:length]
        if qm.ndim == 3:
            qm = qm.unsqueeze(-1)
        qm = comfy.utils.common_upscale(
            qm.movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        if qm.ndim == 4 and qm.shape[-1] == 1:
            qm = qm.squeeze(-1)

        mask_condition = qm
        if mask_condition.ndim == 3:
            mask_condition_3ch = mask_condition.unsqueeze(-1).expand(-1, -1, -1, 3)
        else:
            mask_condition_3ch = mask_condition

        inverted_mask_3ch = 1.0 - mask_condition_3ch
        masked_video = vid[:, :, :, :3] * (1.0 - mask_condition_3ch)

        mask_latents = vae.encode(inverted_mask_3ch)
        masked_video_latents = vae.encode(masked_video)

        def _match_temporal(lat, target_t):
            if lat.shape[2] > target_t:
                return lat[:, :, :target_t]
            elif lat.shape[2] < target_t:
                pad = target_t - lat.shape[2]
                return torch.cat([lat, lat[:, :, -1:].repeat(1, 1, pad, 1, 1)], dim=2)
            return lat

        mask_latents = _match_temporal(mask_latents, latent_t)
        masked_video_latents = _match_temporal(masked_video_latents, latent_t)

        inpaint_latents = torch.cat([mask_latents, masked_video_latents], dim=1)

        # No explicit scaling needed here: the model's CogVideoX.concat_cond()
        # applies process_latent_in (×latent_format.scale_factor) to each 16-ch
        # block of the stored conditioning. For 5b-class checkpoints (incl. the
        # VOID/CogVideoX-Fun-V1.5 inpainting model) that scale_factor is auto-
        # selected as 0.7 in supported_models.CogVideoX_T2V, which matches the
        # diffusers vae/config.json scaling_factor VOID was trained with.

        positive = node_helpers.conditioning_set_values(
            positive, {"concat_latent_image": inpaint_latents}
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"concat_latent_image": inpaint_latents}
        )

        noise_latent = torch.zeros(
            [batch_size, 16, latent_t, latent_h, latent_w],
            device=comfy.model_management.intermediate_device()
        )

        return io.NodeOutput(positive, negative, {"samples": noise_latent})


class VOIDWarpedNoise(io.ComfyNode):
    """Generate optical-flow warped noise for VOID Pass 2 refinement.

    Takes the Pass 1 output video and produces temporally-correlated noise
    by warping Gaussian noise along optical flow vectors. This noise is used
    as the initial latent for Pass 2, resulting in better temporal consistency.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VOIDWarpedNoise",
            category="model/latent/void",
            inputs=[
                OpticalFlow.Input(
                    "optical_flow",
                    tooltip="Optical flow model from OpticalFlowLoader (RAFT-large).",
                ),
                io.Image.Input("video", tooltip="Pass 1 output video frames [T, H, W, 3]"),
                io.Int.Input("width", default=672, min=16, max=nodes.MAX_RESOLUTION, step=8),
                io.Int.Input("height", default=384, min=16, max=nodes.MAX_RESOLUTION, step=8),
                io.Int.Input("length", default=45, min=1, max=nodes.MAX_RESOLUTION, step=1,
                             tooltip="Number of pixel frames. Rounded down to make latent_t "
                                     "even (patch_size_t=2 requirement), e.g. 49 → 45."),
                io.Int.Input("batch_size", default=1, min=1, max=64),
            ],
            outputs=[
                io.Latent.Output(display_name="warped_noise"),
            ],
        )

    @classmethod
    def execute(cls, optical_flow, video, width, height, length, batch_size) -> io.NodeOutput:

        adjusted_length = _valid_void_length(length)
        if adjusted_length != length:
            logging.warning(
                "VOIDWarpedNoise: rounding length %d down to %d so that "
                "latent_t is even (required by CogVideoX-Fun-V1.5 patch_size_t=2).",
                length, adjusted_length,
            )
            length = adjusted_length

        latent_t = ((length - 1) // TEMPORAL_COMPRESSION) + 1
        latent_h = height // 8
        latent_w = width // 8

        # RAFT + noise warp is real compute, not an "intermediate" buffer, so
        # we want the actual torch device (CUDA/MPS).  The final latent is
        # moved back to intermediate_device() before returning to match the
        # rest of the ComfyUI pipeline.
        device = comfy.model_management.get_torch_device()

        comfy.model_management.load_model_gpu(optical_flow)
        raft = RaftOpticalFlow(optical_flow.model, device=device)

        vid = video[:length].to(device)
        vid = comfy.utils.common_upscale(
            vid.movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        vid_uint8 = (vid.clamp(0, 1) * 255).to(torch.uint8)

        FRAME = 2**-1
        FLOW = 2**3
        LATENT_SCALE = 8

        warped = get_noise_from_video(
            vid_uint8,
            raft,
            noise_channels=16,
            resize_frames=FRAME,
            resize_flow=FLOW,
            downscale_factor=round(FRAME * FLOW) * LATENT_SCALE,
            device=device,
        )

        if warped.shape[0] != latent_t:
            indices = torch.linspace(0, warped.shape[0] - 1, latent_t,
                                     device=device).long()
            warped = warped[indices]

        if warped.shape[1] != latent_h or warped.shape[2] != latent_w:
            # (T, H, W, C) → (T, C, H, W) → bilinear resize → back
            warped = warped.permute(0, 3, 1, 2)
            warped = torch.nn.functional.interpolate(
                warped, size=(latent_h, latent_w),
                mode="bilinear", align_corners=False,
            )
            warped = warped.permute(0, 2, 3, 1)

        # (T, H, W, C) → (B, C, T, H, W)
        warped_tensor = warped.permute(3, 0, 1, 2).unsqueeze(0)
        if batch_size > 1:
            warped_tensor = warped_tensor.repeat(batch_size, 1, 1, 1, 1)

        warped_tensor = warped_tensor.to(comfy.model_management.intermediate_device())
        return io.NodeOutput({"samples": warped_tensor})


class Noise_FromLatent:
    """Wraps a pre-computed LATENT tensor as a NOISE source."""
    def __init__(self, latent_dict):
        self.seed = 0
        self._samples = latent_dict["samples"]

    def generate_noise(self, input_latent):
        return self._samples.clone().cpu()


class VOIDWarpedNoiseSource(io.ComfyNode):
    """Convert a LATENT (e.g. from VOIDWarpedNoise) into a NOISE source
    for use with SamplerCustomAdvanced."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VOIDWarpedNoiseSource",
            category="model/latent/void",
            inputs=[
                io.Latent.Input("warped_noise",
                    tooltip="Warped noise latent from VOIDWarpedNoise"),
            ],
            outputs=[io.Noise.Output()],
        )

    @classmethod
    def execute(cls, warped_noise) -> io.NodeOutput:
        return io.NodeOutput(Noise_FromLatent(warped_noise))


class VOID_DDIM(comfy.samplers.Sampler):
    """DDIM sampler for VOID inpainting models.

    VOID was trained with the diffusers CogVideoXDDIMScheduler which operates in
    alpha-space (input std ≈ 1). The standard KSampler applies noise_scaling that
    multiplies by sqrt(1+sigma^2) ≈ 4500x, which is incompatible with VOID's
    training. This sampler skips noise_scaling and implements the DDIM update rule
    directly using sigma-to-alpha conversion.
    """

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        x = noise.to(torch.float32)
        model_options = extra_args.get("model_options", {})
        seed = extra_args.get("seed", None)
        s_in = x.new_ones([x.shape[0]])

        for i in trange(len(sigmas) - 1, disable=disable_pbar):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            denoised = model_wrap(x, sigma * s_in, model_options=model_options, seed=seed)

            if callback is not None:
                callback(i, denoised, x, len(sigmas) - 1)

            if sigma_next == 0:
                x = denoised
            else:
                alpha_t = 1.0 / (1.0 + sigma ** 2)
                alpha_prev = 1.0 / (1.0 + sigma_next ** 2)

                pred_eps = (x - (alpha_t ** 0.5) * denoised) / (1.0 - alpha_t) ** 0.5
                x = (alpha_prev ** 0.5) * denoised + (1.0 - alpha_prev) ** 0.5 * pred_eps

        return x


class VOIDSampler(io.ComfyNode):
    """VOID DDIM sampler for use with SamplerCustom / SamplerCustomAdvanced.

    Required for VOID inpainting models. Implements the same DDIM loop that VOID
    was trained with (diffusers CogVideoXDDIMScheduler), without the noise_scaling
    that the standard KSampler applies. Use with RandomNoise or VOIDWarpedNoiseSource.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VOIDSampler",
            category="model/sampling/samplers",
            inputs=[],
            outputs=[io.Sampler.Output()],
        )

    @classmethod
    def execute(cls) -> io.NodeOutput:
        return io.NodeOutput(VOID_DDIM())

    get_sampler = execute


class VOIDExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            OpticalFlowLoader,
            VOIDQuadmaskPreprocess,
            VOIDInpaintConditioning,
            VOIDWarpedNoise,
            VOIDWarpedNoiseSource,
            VOIDSampler,
        ]


async def comfy_entrypoint() -> VOIDExtension:
    return VOIDExtension()
