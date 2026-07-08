import nodes
import node_helpers
import torch
import torchaudio
import comfy.model_management
import comfy.model_sampling
import comfy.samplers
import comfy.utils
import math
import numpy as np
import av
from io import BytesIO
from typing_extensions import override
from comfy.ldm.lightricks.symmetric_patchifier import SymmetricPatchifier, latent_to_pixel_coords
from comfy_api.latest import ComfyExtension, io

ICLoRAParameters = io.Custom("IC_LORA_PARAMETERS")


class GetICLoRAParameters(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GetICLoRAParameters",
            display_name="Get IC-LoRA Parameters",
            description="Extracts IC-LoRA parameters from the safetensors metadata of a LoRA-loaded "
                        "model and outputs them for LTXVAddGuide (eg. reference_downscale_factor).",
            category="model/conditioning/ltxv",
            search_aliases=["ic-lora", "ic lora", "iclora", "downscale factor", "reference downscale"],
            inputs=[
                io.Model.Input(
                    "iclora_model",
                    tooltip="Direct output from a LoRA Loader for the specific IC-LoRA "
                            "from which to extract the metadata.",
                ),
            ],
            outputs=[
                ICLoRAParameters.Output(
                    "iclora_parameters",
                    tooltip="IC-LoRA parameters extracted from the LoRA metadata "
                            "(eg. reference_downscale_factor). Connect to LTXVAddGuide "
                            "if the LoRA requires special handling of the guides.",
                ),
            ],
        )

    @classmethod
    def execute(cls, iclora_model) -> io.NodeOutput:
        metadata = iclora_model.get_attachment("lora_metadata")
        factor = 1
        if metadata:
            try:
                factor = max(1, round(float(metadata.get("reference_downscale_factor", 1))))
            except (TypeError, ValueError):
                factor = 1
        parameters = {"reference_downscale_factor": factor}
        return io.NodeOutput(parameters)


class EmptyLTXVLatentVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptyLTXVLatentVideo",
            category="model/latent/ltxv",
            inputs=[
                io.Int.Input("width", default=768, min=64, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=512, min=64, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=97, min=1, max=nodes.MAX_RESOLUTION, step=8),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, width, height, length, batch_size=1) -> io.NodeOutput:
        latent = torch.zeros([batch_size, 128, ((length - 1) // 8) + 1, height // 32, width // 32], device=comfy.model_management.intermediate_device())
        return io.NodeOutput({"samples": latent, "downscale_ratio_spacial": 32})

    generate = execute  # TODO: remove

class LTXVImgToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVImgToVideo",
            category="model/conditioning/ltxv",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Image.Input("image"),
                io.Int.Input("width", default=768, min=64, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=512, min=64, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=97, min=9, max=nodes.MAX_RESOLUTION, step=8),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, image, vae, width, height, length, batch_size, strength) -> io.NodeOutput:
        pixels = comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        encode_pixels = pixels[:, :, :, :3]
        t = vae.encode(encode_pixels)

        latent = torch.zeros([batch_size, 128, ((length - 1) // 8) + 1, height // 32, width // 32], device=comfy.model_management.intermediate_device())
        latent[:, :, :t.shape[2]] = t

        conditioning_latent_frames_mask = torch.ones(
            (batch_size, 1, latent.shape[2], 1, 1),
            dtype=torch.float32,
            device=latent.device,
        )
        conditioning_latent_frames_mask[:, :, :t.shape[2]] = 1.0 - strength

        return io.NodeOutput(positive, negative, {"samples": latent, "noise_mask": conditioning_latent_frames_mask})

    generate = execute  # TODO: remove


class LTXVImgToVideoInplace(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVImgToVideoInplace",
            category="model/conditioning/ltxv",
            inputs=[
                io.Vae.Input("vae"),
                io.Image.Input("image"),
                io.Latent.Input("latent"),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0),
                io.Boolean.Input("bypass", default=False, tooltip="Bypass the conditioning.")
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, vae, image, latent, strength, bypass=False) -> io.NodeOutput:
        if bypass:
            return (latent,)

        samples = latent["samples"].clone()
        _, height_scale_factor, width_scale_factor = (
            vae.downscale_index_formula
        )

        _, _, _, latent_height, latent_width = samples.shape
        width = latent_width * width_scale_factor
        height = latent_height * height_scale_factor

        if image.shape[1] != height or image.shape[2] != width:
            pixels = comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        else:
            pixels = image
        encode_pixels = pixels[:, :, :, :3]
        t = vae.encode(encode_pixels)

        samples[:, :, :t.shape[2]] = t

        conditioning_latent_frames_mask = get_noise_mask(latent)
        conditioning_latent_frames_mask[:, :, :t.shape[2]] = 1.0 - strength

        return io.NodeOutput({"samples": samples, "noise_mask": conditioning_latent_frames_mask})

    generate = execute  # TODO: remove


def _append_guide_attention_entry(positive, negative, pre_filter_count, latent_shape, strength=1.0, attention_mask=None):
    """Append a guide_attention_entry to both positive and negative conditioning.

    Each entry tracks one guide reference for per-reference attention control.
    Entries are derived independently from each conditioning to avoid cross-contamination.
    """
    new_entry = {
        "pre_filter_count": pre_filter_count,
        "strength": strength,
        "pixel_mask": attention_mask.unsqueeze(0).unsqueeze(0) if attention_mask is not None else None,  # reshape to (1, 1, F, H, W)
        "latent_shape": latent_shape,
    }

    results = []
    for cond in (positive, negative):
        # Read existing entries from this specific conditioning
        existing = []
        for t in cond:
            found = t[1].get("guide_attention_entries", None)
            if found is not None:
                existing = found
                break
        # Shallow copy only and append (pixel_mask is never mutated).
        entries = [*existing, new_entry]
        results.append(node_helpers.conditioning_set_values(
            cond, {"guide_attention_entries": entries}
        ))
    return results[0], results[1]


def conditioning_get_any_value(conditioning, key, default=None):
    for t in conditioning:
        if key in t[1]:
            return t[1][key]
    return default


def get_noise_mask(latent):
    noise_mask = latent.get("noise_mask", None)
    latent_image = latent["samples"]
    if noise_mask is None:
        batch_size, _, latent_length, _, _ = latent_image.shape
        noise_mask = torch.ones(
            (batch_size, 1, latent_length, 1, 1),
            dtype=torch.float32,
            device=latent_image.device,
        )
    else:
        noise_mask = noise_mask.clone()
    return noise_mask

def get_keyframe_idxs(cond, latent_shape=None):
    keyframe_idxs = conditioning_get_any_value(cond, "keyframe_idxs", None)
    if keyframe_idxs is None:
        return None, 0
    # Get number of keyframes from latent_shape or guide_attention_entries if available
    if latent_shape is not None and len(latent_shape) == 5:
        tokens_per_frame = latent_shape[-2] * latent_shape[-1]
        num_keyframes = keyframe_idxs.shape[2] // tokens_per_frame
        return keyframe_idxs, num_keyframes
    entries = conditioning_get_any_value(cond, "guide_attention_entries", None)
    if entries:
        num_keyframes = sum(e["latent_shape"][0] for e in entries)
        return keyframe_idxs, num_keyframes
    # fallback, may under-count if keyframes share t-start
    # keyframe_idxs contains start/end positions (last dimension), checking for unqiue values only for start
    num_keyframes = torch.unique(keyframe_idxs[:, 0, :, 0]).shape[0]
    return keyframe_idxs, num_keyframes

class LTXVAddGuide(io.ComfyNode):
    PATCHIFIER = SymmetricPatchifier(1, start_end=True)

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVAddGuide",
            category="model/conditioning/ltxv",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Latent.Input("latent"),
                io.Image.Input(
                    "image",
                    tooltip="Image or video to condition the latent video on. Must be 8*n + 1 frames. "
                            "If the video is not 8*n + 1 frames, it will be cropped to the nearest 8*n + 1 frames.",
                ),
                io.Int.Input(
                    "frame_idx",
                    default=0,
                    min=-9999,
                    max=9999,
                    tooltip="Frame index to start the conditioning at. "
                            "For single-frame images or videos with 1-8 frames, any frame_idx value is acceptable. "
                            "For videos with 9+ frames, frame_idx must be divisible by 8, otherwise it will be rounded "
                            "down to the nearest multiple of 8. Negative values are counted from the end of the video.",
                ),
                io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Mask.Input(
                    "attention_mask",
                    optional=True,
                    tooltip="Optional pixel-space spatial mask. Controls per-region "
                            "conditioning influence via self-attention, multiplied by strength.",
                ),
                ICLoRAParameters.Input(
                    "iclora_parameters",
                    optional=True,
                    tooltip="Optional IC-LoRA parameters from a Get IC-LoRA Parameters node. "
                            "Used for adjusting guide processing as required by certain IC-LoRAs "
                            "(eg. those with a reference_downscale_factor > 1). "
                            "When chained, each LTXVAddGuide uses only the parameters connected to it.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def encode(cls, vae, latent_width, latent_height, images, scale_factors, latent_downscale_factor=1):
        time_scale_factor, width_scale_factor, height_scale_factor = scale_factors
        images = images[:(images.shape[0] - 1) // time_scale_factor * time_scale_factor + 1]
        target_width = int(latent_width * width_scale_factor / latent_downscale_factor)
        target_height = int(latent_height * height_scale_factor / latent_downscale_factor)
        pixels = comfy.utils.common_upscale(images.movedim(-1, 1), target_width, target_height, "bilinear", crop="center").movedim(1, -1)
        encode_pixels = pixels[:, :, :, :3]
        t = vae.encode(encode_pixels)
        return encode_pixels, t

    @classmethod
    def dilate_latent(cls, guide_latent, latent_downscale_factor):
        if latent_downscale_factor <= 1:
            return guide_latent, None
        scale = int(latent_downscale_factor)
        dilated_shape = guide_latent.shape[:3] + (guide_latent.shape[3] * scale, guide_latent.shape[4] * scale)
        dilated = torch.zeros(dilated_shape, device=guide_latent.device, dtype=guide_latent.dtype)
        dilated[..., ::scale, ::scale] = guide_latent
        dilated_mask = torch.full(
            (dilated.shape[0], 1, dilated.shape[2], dilated.shape[3], dilated.shape[4]),
            -1.0, device=guide_latent.device, dtype=guide_latent.dtype,
        )
        dilated_mask[..., ::scale, ::scale] = 1.0
        return dilated, dilated_mask

    @classmethod
    def get_reference_downscale_factor(cls, iclora_parameters):
        if not iclora_parameters:
            return 1
        try:
            factor = max(1, round(float(iclora_parameters.get("reference_downscale_factor", 1))))
        except (TypeError, ValueError):
            factor = 1
        return factor

    @classmethod
    def get_latent_index(cls, cond, latent_length, guide_length, frame_idx, scale_factors, latent_shape=None):
        time_scale_factor, _, _ = scale_factors
        _, num_keyframes = get_keyframe_idxs(cond, latent_shape)
        latent_count = latent_length - num_keyframes
        frame_idx = frame_idx if frame_idx >= 0 else max((latent_count - 1) * time_scale_factor + 1 + frame_idx, 0)
        if guide_length > 1 and frame_idx != 0:
            frame_idx = (frame_idx - 1) // time_scale_factor * time_scale_factor + 1 # frame index - 1 must be divisible by 8 or frame_idx == 0

        latent_idx = (frame_idx + time_scale_factor - 1) // time_scale_factor

        return frame_idx, latent_idx

    @classmethod
    def add_keyframe_index(cls, cond, frame_idx, guiding_latent, scale_factors, latent_downscale_factor=1, causal_fix=None):
        keyframe_idxs, _ = get_keyframe_idxs(cond)
        _, latent_coords = cls.PATCHIFIER.patchify(guiding_latent)
        if causal_fix is None:
            causal_fix = frame_idx == 0 or guiding_latent.shape[2] == 1
        pixel_coords = latent_to_pixel_coords(latent_coords, scale_factors, causal_fix=causal_fix)
        pixel_coords[:, 0] += frame_idx

        # The following adjusts keyframe end positions for small grid IC-LoRA.
        # After dilation, the small grid has the same size and position as the large grid,
        # but each token encodes a larger image patch. We adjust the end position (not start)
        # so that RoPE represents the correct middle point of each token.
        # keyframe_idxs dims: (batch, spatial_dim [t,h,w], token_id, [start, end])
        # We only adjust h,w (not t) in dim 1, and only end (not start) in dim 3.
        spatial_end_offset = (latent_downscale_factor - 1) * torch.tensor(
            scale_factors[1:],
            device=pixel_coords.device,
        ).view(1, -1, 1, 1)
        pixel_coords[:, 1:, :, 1:] += spatial_end_offset.to(pixel_coords.dtype)

        if keyframe_idxs is None:
            keyframe_idxs = pixel_coords
        else:
            keyframe_idxs = torch.cat([keyframe_idxs, pixel_coords], dim=2)
        return node_helpers.conditioning_set_values(cond, {"keyframe_idxs": keyframe_idxs})

    @classmethod
    def append_keyframe(cls, positive, negative, frame_idx, latent_image, noise_mask, guiding_latent, strength, scale_factors, guide_mask=None, in_channels=128, latent_downscale_factor=1, causal_fix=None):
        if latent_image.shape[1] != in_channels or guiding_latent.shape[1] != in_channels:
            raise ValueError("Adding guide to a combined AV latent is not supported.")

        positive = cls.add_keyframe_index(positive, frame_idx, guiding_latent, scale_factors, latent_downscale_factor, causal_fix=causal_fix)
        negative = cls.add_keyframe_index(negative, frame_idx, guiding_latent, scale_factors, latent_downscale_factor, causal_fix=causal_fix)

        if guide_mask is not None:
            target_h = max(noise_mask.shape[3], guide_mask.shape[3])
            target_w = max(noise_mask.shape[4], guide_mask.shape[4])

            if noise_mask.shape[3] == 1 or noise_mask.shape[4] == 1:
                noise_mask = noise_mask.expand(-1, -1, -1, target_h, target_w)

            if guide_mask.shape[3] == 1 or guide_mask.shape[4] == 1:
                guide_mask = guide_mask.expand(-1, -1, -1, target_h, target_w)
            mask = guide_mask - strength
        else:
            mask = torch.full(
                (noise_mask.shape[0], 1, guiding_latent.shape[2], noise_mask.shape[3], noise_mask.shape[4]),
                max(0.0, 1.0 - strength), # clamp here to amplify only via the attention mask
                dtype=noise_mask.dtype,
                device=noise_mask.device,
            )
        # This solves audio video combined latent case where latent_image has audio latent concatenated
        # in channel dimension with video latent. The solution is to pad guiding latent accordingly.
        if latent_image.shape[1] > guiding_latent.shape[1]:
            pad_len = latent_image.shape[1] - guiding_latent.shape[1]
            guiding_latent = torch.nn.functional.pad(guiding_latent, pad=(0, 0, 0, 0, 0, 0, 0, pad_len), value=0)
        latent_image = torch.cat([latent_image, guiding_latent], dim=2)
        noise_mask = torch.cat([noise_mask, mask], dim=2)
        return positive, negative, latent_image, noise_mask

    @classmethod
    def replace_latent_frames(cls, latent_image, noise_mask, guiding_latent, latent_idx, strength):
        cond_length = guiding_latent.shape[2]
        assert latent_image.shape[2] >= latent_idx + cond_length, "Conditioning frames exceed the length of the latent sequence."

        mask = torch.full(
            (noise_mask.shape[0], 1, cond_length, 1, 1),
            max(0.0, 1.0 - strength), # clamp here to amplify only via the attention mask
            dtype=noise_mask.dtype,
            device=noise_mask.device,
        )

        latent_image = latent_image.clone()
        noise_mask = noise_mask.clone()

        latent_image[:, :, latent_idx : latent_idx + cond_length] = guiding_latent
        noise_mask[:, :, latent_idx : latent_idx + cond_length] = mask

        return latent_image, noise_mask

    @classmethod
    def execute(cls, positive, negative, vae, latent, image, frame_idx, strength, attention_mask=None, iclora_parameters=None) -> io.NodeOutput:
        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"]
        noise_mask = get_noise_mask(latent)

        _, _, latent_length, latent_height, latent_width = latent_image.shape

        latent_downscale_factor = cls.get_reference_downscale_factor(iclora_parameters)
        if latent_downscale_factor > 1:
            if latent_width % latent_downscale_factor != 0 or latent_height % latent_downscale_factor != 0:
                raise ValueError(
                    f"Latent spatial size {latent_width}x{latent_height} must be divisible by "
                    f"reference_downscale_factor {latent_downscale_factor} from the IC-LoRA parameters."
                )

        # For mid-video multi-frame guides, prepend+strip a throwaway first frame so the VAE's "first latent = 1 pixel frame" asymmetry lands on the discarded slot
        time_scale_factor = scale_factors[0]
        num_frames_to_keep = ((image.shape[0] - 1) // time_scale_factor) * time_scale_factor + 1
        resolved_frame_idx = frame_idx
        if frame_idx < 0:
            _, num_keyframes = get_keyframe_idxs(positive, latent_image.shape)
            resolved_frame_idx = max((latent_length - num_keyframes - 1) * time_scale_factor + 1 + frame_idx, 0)
        causal_fix = resolved_frame_idx == 0 or num_frames_to_keep == 1

        if not causal_fix:
            image = torch.cat([image[:1], image], dim=0)

        image, t = cls.encode(vae, latent_width, latent_height, image, scale_factors, latent_downscale_factor)

        if not causal_fix:
            t = t[:, :, 1:, :, :]
            image = image[1:]

        guide_latent_shape = list(t.shape[2:])  # pre-dilation [F, H, W] for spatial-mask downsampling
        guide_mask = None
        if latent_downscale_factor > 1:
            t, guide_mask = cls.dilate_latent(t, latent_downscale_factor)

        frame_idx, latent_idx = cls.get_latent_index(positive, latent_length, len(image), frame_idx, scale_factors, latent_shape=latent_image.shape)
        assert latent_idx + t.shape[2] <= latent_length, "Conditioning frames exceed the length of the latent sequence."

        positive, negative, latent_image, noise_mask = cls.append_keyframe(
            positive,
            negative,
            frame_idx,
            latent_image,
            noise_mask,
            t,
            strength,
            scale_factors,
            guide_mask=guide_mask,
            latent_downscale_factor=latent_downscale_factor,
            causal_fix=causal_fix,
        )

        # Track this guide for per-reference attention control.
        pre_filter_count = t.shape[2] * t.shape[3] * t.shape[4]
        positive, negative = _append_guide_attention_entry(
            positive, negative, pre_filter_count, guide_latent_shape, strength=strength,
            attention_mask=attention_mask,
        )

        return io.NodeOutput(positive, negative, {"samples": latent_image, "noise_mask": noise_mask})

    generate = execute  # TODO: remove


class LTXVCropGuides(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVCropGuides",
            category="model/conditioning/ltxv",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent"),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, latent) -> io.NodeOutput:
        latent_image = latent["samples"].clone()
        noise_mask = get_noise_mask(latent)

        _, num_keyframes = get_keyframe_idxs(positive, latent_image.shape)
        if num_keyframes == 0:
            return io.NodeOutput(positive, negative, {"samples": latent_image, "noise_mask": noise_mask},)

        latent_image = latent_image[:, :, :-num_keyframes]
        noise_mask = noise_mask[:, :, :-num_keyframes]

        positive = node_helpers.conditioning_set_values(positive, {
            "keyframe_idxs": None,
            "guide_attention_entries": None,
        })
        negative = node_helpers.conditioning_set_values(negative, {
            "keyframe_idxs": None,
            "guide_attention_entries": None,
        })

        return io.NodeOutput(positive, negative, {"samples": latent_image, "noise_mask": noise_mask})

    crop = execute  # TODO: remove


class LTXVConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVConditioning",
            category="model/conditioning/ltxv",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Float.Input("frame_rate", default=25.0, min=0.0, max=1000.0, step=0.01),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, frame_rate) -> io.NodeOutput:
        positive = node_helpers.conditioning_set_values(positive, {"frame_rate": frame_rate})
        negative = node_helpers.conditioning_set_values(negative, {"frame_rate": frame_rate})
        return io.NodeOutput(positive, negative)


class ModelSamplingLTXV(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ModelSamplingLTXV",
            category="model/patch/ltxv",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("max_shift", default=2.05, min=0.0, max=100.0, step=0.01),
                io.Float.Input("base_shift", default=0.95, min=0.0, max=100.0, step=0.01),
                io.Latent.Input("latent", optional=True),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, max_shift, base_shift, latent=None) -> io.NodeOutput:
        m = model.clone()

        if latent is None:
            tokens = 4096
        else:
            tokens = math.prod(latent["samples"].shape[2:])

        x1 = 1024
        x2 = 4096
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        shift = (tokens) * mm + b

        sampling_base = comfy.model_sampling.ModelSamplingFlux
        sampling_type = comfy.model_sampling.CONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift)
        m.add_object_patch("model_sampling", model_sampling)

        return io.NodeOutput(m)


class LTXVScheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVScheduler",
            category="model/sampling/schedulers",
            inputs=[
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Float.Input("max_shift", default=2.05, min=0.0, max=100.0, step=0.01),
                io.Float.Input("base_shift", default=0.95, min=0.0, max=100.0, step=0.01),
                io.Boolean.Input(
                    id="stretch",
                    default=True,
                    tooltip="Stretch the sigmas to be in the range [terminal, 1].",
                    advanced=True,
                ),
                io.Float.Input(
                    id="terminal",
                    default=0.1,
                    min=0.0,
                    max=0.99,
                    step=0.01,
                    tooltip="The terminal value of the sigmas after stretching.",
                    advanced=True,
                ),
                io.Latent.Input("latent", optional=True),
            ],
            outputs=[
                io.Sigmas.Output(),
            ],
        )

    @classmethod
    def execute(cls, steps, max_shift, base_shift, stretch, terminal, latent=None) -> io.NodeOutput:
        if latent is None:
            tokens = 4096
        else:
            tokens = math.prod(latent["samples"].shape[2:])

        sigmas = torch.linspace(1.0, 0.0, steps + 1)

        x1 = 1024
        x2 = 4096
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        sigma_shift = (tokens) * mm + b

        power = 1
        sigmas = torch.where(
            sigmas != 0,
            math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1) ** power),
            0,
        )

        # Stretch sigmas so that its final value matches the given terminal value.
        if stretch:
            non_zero_mask = sigmas != 0
            non_zero_sigmas = sigmas[non_zero_mask]
            one_minus_z = 1.0 - non_zero_sigmas
            scale_factor = one_minus_z[-1] / (1.0 - terminal)
            stretched = 1.0 - (one_minus_z / scale_factor)
            sigmas[non_zero_mask] = stretched

        return io.NodeOutput(sigmas)

def encode_single_frame(output_file, image_array: np.ndarray, crf):
    container = av.open(output_file, "w", format="mp4")
    try:
        stream = container.add_stream(
            "libx264", rate=1, options={"crf": str(crf), "preset": "veryfast"}
        )
        stream.height = image_array.shape[0]
        stream.width = image_array.shape[1]
        av_frame = av.VideoFrame.from_ndarray(image_array, format="rgb24").reformat(
            format="yuv420p"
        )
        container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
    finally:
        container.close()


def decode_single_frame(video_file):
    container = av.open(video_file)
    try:
        stream = next(s for s in container.streams if s.type == "video")
        frame = next(container.decode(stream))
    finally:
        container.close()
    return frame.to_ndarray(format="rgb24")


def preprocess(image: torch.Tensor, crf=29):
    if crf == 0:
        return image

    image_array = (image[:(image.shape[0] // 2) * 2, :(image.shape[1] // 2) * 2] * 255.0).byte().cpu().numpy()
    with BytesIO() as output_file:
        encode_single_frame(output_file, image_array, crf)
        video_bytes = output_file.getvalue()
    with BytesIO(video_bytes) as video_file:
        image_array = decode_single_frame(video_file)
    tensor = torch.tensor(image_array, dtype=image.dtype, device=image.device) / 255.0
    return tensor


class LTXVPreprocess(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVPreprocess",
            display_name="LTXV Preprocess",
            category="video/preprocessors",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input(
                    id="img_compression", default=35, min=0, max=100, tooltip="Amount of compression to apply on image."
                ),
            ],
            outputs=[
                io.Image.Output(display_name="output_image"),
            ],
        )

    @classmethod
    def execute(cls, image, img_compression) -> io.NodeOutput:
        output_images = []
        for i in range(image.shape[0]):
            output_images.append(preprocess(image[i], img_compression))
        return io.NodeOutput(torch.stack(output_images))

    preprocess = execute  # TODO: remove


import comfy.nested_tensor
class LTXVConcatAVLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVConcatAVLatent",
            category="model/latent/ltxv",
            inputs=[
                io.Latent.Input("video_latent"),
                io.Latent.Input("audio_latent"),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, video_latent, audio_latent) -> io.NodeOutput:
        output = {}
        output.update(video_latent)
        output.update(audio_latent)
        video_noise_mask = video_latent.get("noise_mask", None)
        audio_noise_mask = audio_latent.get("noise_mask", None)

        if video_noise_mask is not None or audio_noise_mask is not None:
            if video_noise_mask is None:
                video_noise_mask = torch.ones_like(video_latent["samples"])
            if audio_noise_mask is None:
                audio_noise_mask = torch.ones_like(audio_latent["samples"])
            output["noise_mask"] = comfy.nested_tensor.NestedTensor((video_noise_mask, audio_noise_mask))

        output["samples"] = comfy.nested_tensor.NestedTensor((video_latent["samples"], audio_latent["samples"]))

        return io.NodeOutput(output)


class LTXVSeparateAVLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVSeparateAVLatent",
            category="model/latent/ltxv",
            description="LTXV Separate AV Latent",
            inputs=[
                io.Latent.Input("av_latent"),
            ],
            outputs=[
                io.Latent.Output(display_name="video_latent"),
                io.Latent.Output(display_name="audio_latent"),
            ],
        )

    @classmethod
    def execute(cls, av_latent) -> io.NodeOutput:
        latents = av_latent["samples"].unbind()
        video_latent = av_latent.copy()
        video_latent["samples"] = latents[0]
        audio_latent = av_latent.copy()
        audio_latent["samples"] = latents[1]
        if "noise_mask" in av_latent:
            masks = av_latent["noise_mask"]
            if masks is not None:
                masks = masks.unbind()
                video_latent["noise_mask"] = masks[0]
                audio_latent["noise_mask"] = masks[1]
        return io.NodeOutput(video_latent, audio_latent)


class LTXVReferenceAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LTXVReferenceAudio",
            display_name="LTXV Reference Audio (ID-LoRA)",
            category="model/conditioning/ltxv",
            description="Set reference audio for ID-LoRA speaker identity transfer. Encodes a reference audio clip into the conditioning and optionally patches the model with identity guidance (extra forward pass without reference, amplifying the speaker identity effect).",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Audio.Input("reference_audio", tooltip="Reference audio clip whose speaker identity to transfer. ~5 seconds recommended (training duration). Shorter or longer clips may degrade voice identity transfer."),
                io.Vae.Input(id="audio_vae", display_name="Audio VAE", tooltip="LTXV Audio VAE for encoding."),
                io.Float.Input("identity_guidance_scale", default=3.0, min=0.0, max=100.0, step=0.01, round=0.01, tooltip="Strength of identity guidance. Runs an extra forward pass without reference each step to amplify speaker identity. Set to 0 to disable (no extra pass)."),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.001, advanced=True, tooltip="Start of the sigma range where identity guidance is active."),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.001, advanced=True, tooltip="End of the sigma range where identity guidance is active."),
            ],
            outputs=[
                io.Model.Output(),
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
            ],
        )

    @classmethod
    def execute(cls, model, positive, negative, reference_audio, audio_vae, identity_guidance_scale, start_percent, end_percent) -> io.NodeOutput:
        # Encode reference audio to latents and patchify
        sample_rate = reference_audio["sample_rate"]
        vae_sample_rate = getattr(audio_vae, "audio_sample_rate", 44100)
        if vae_sample_rate != sample_rate:
            waveform = torchaudio.functional.resample(reference_audio["waveform"], sample_rate, vae_sample_rate)
        else:
            waveform = reference_audio["waveform"]

        audio_latents = audio_vae.encode(waveform.movedim(1, -1))
        b, c, t, f = audio_latents.shape
        ref_tokens = audio_latents.permute(0, 2, 1, 3).reshape(b, t, c * f)
        ref_audio = {"tokens": ref_tokens}

        positive = node_helpers.conditioning_set_values(positive, {"ref_audio": ref_audio})
        negative = node_helpers.conditioning_set_values(negative, {"ref_audio": ref_audio})

        # Patch model with identity guidance
        m = model.clone()
        scale = identity_guidance_scale
        model_sampling = m.get_model_object("model_sampling")
        sigma_start = model_sampling.percent_to_sigma(start_percent)
        sigma_end = model_sampling.percent_to_sigma(end_percent)

        def post_cfg_function(args):
            if scale == 0:
                return args["denoised"]

            sigma = args["sigma"]
            sigma_ = sigma[0].item()
            if sigma_ > sigma_start or sigma_ < sigma_end:
                return args["denoised"]

            cond_pred = args["cond_denoised"]
            cond = args["cond"]
            cfg_result = args["denoised"]
            model_options = args["model_options"].copy()
            x = args["input"]

            # Strip ref_audio from conditioning for the no-reference pass
            noref_cond = []
            for entry in cond:
                new_entry = entry.copy()
                mc = new_entry.get("model_conds", {}).copy()
                mc.pop("ref_audio", None)
                new_entry["model_conds"] = mc
                noref_cond.append(new_entry)

            (pred_noref,) = comfy.samplers.calc_cond_batch(
                args["model"], [noref_cond], x, sigma, model_options
            )

            return cfg_result + (cond_pred - pred_noref) * scale

        m.set_model_sampler_post_cfg_function(post_cfg_function)

        return io.NodeOutput(m, positive, negative)


class LtxvExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            EmptyLTXVLatentVideo,
            LTXVImgToVideo,
            LTXVImgToVideoInplace,
            ModelSamplingLTXV,
            LTXVConditioning,
            LTXVScheduler,
            GetICLoRAParameters,
            LTXVAddGuide,
            LTXVPreprocess,
            LTXVCropGuides,
            LTXVConcatAVLatent,
            LTXVSeparateAVLatent,
            LTXVReferenceAudio,
        ]


async def comfy_entrypoint() -> LtxvExtension:
    return LtxvExtension()
