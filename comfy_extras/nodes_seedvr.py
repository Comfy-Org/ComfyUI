import logging

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import torch

import comfy.model_management
from comfy.ldm.seedvr.color_fix import (
    adain_color_transfer,
    lab_color_transfer,
    wavelet_color_transfer,
)
from comfy.ldm.seedvr.constants import (
    BYTEDANCE_VAE_SPATIAL_DOWNSAMPLE,
    SEEDVR2_ADAIN_SCALE_MULTIPLIER,
    SEEDVR2_CHUNK_GIB_PER_MPX_FRAME,
    SEEDVR2_CHUNK_RESERVED_GIB,
    SEEDVR2_CHUNK_SIGMA_GIB,
    SEEDVR2_CHUNK_SIGMA_K,
    SEEDVR2_COLOR_MEM_HEADROOM,
    SEEDVR2_DTYPE_BYTES_FLOOR,
    SEEDVR2_LAB_SCALE_MULTIPLIER,
    SEEDVR2_LATENT_CHANNELS,
    SEEDVR2_OOM_BACKOFF_DIVISOR,
    SEEDVR2_WAVELET_SCALE_MULTIPLIER,
)

from torchvision.transforms import functional as TVF
from torchvision.transforms.functional import InterpolationMode


_SEEDVR2_INVALID_MODEL_MSG_PREFIX = "SeedVR2Conditioning: model object does not match expected SeedVR2 structure"
_ATTR_MISSING = object()


def _resolve_seedvr2_diffusion_model(model):
    inner = getattr(model, "model", _ATTR_MISSING)
    if inner is _ATTR_MISSING:
        raise RuntimeError(
            f"{_SEEDVR2_INVALID_MODEL_MSG_PREFIX}: input has no 'model' attribute "
            f"(got type {type(model).__name__})."
        )
    if inner is None:
        raise RuntimeError(
            f"{_SEEDVR2_INVALID_MODEL_MSG_PREFIX}: input.model is None "
            f"(input type {type(model).__name__})."
        )
    diffusion_model = getattr(inner, "diffusion_model", _ATTR_MISSING)
    if diffusion_model is _ATTR_MISSING:
        raise RuntimeError(
            f"{_SEEDVR2_INVALID_MODEL_MSG_PREFIX}: 'model.model' has no "
            f"'diffusion_model' attribute (got type {type(inner).__name__})."
        )
    if diffusion_model is None:
        raise RuntimeError(
            f"{_SEEDVR2_INVALID_MODEL_MSG_PREFIX}: 'model.model.diffusion_model' "
            f"is None (model.model type {type(inner).__name__})."
        )
    return diffusion_model


def div_pad(image, factor):
    height_factor, width_factor = factor
    height, width = image.shape[-2:]

    pad_height = (height_factor - (height % height_factor)) % height_factor
    pad_width = (width_factor - (width % width_factor)) % width_factor

    if pad_height == 0 and pad_width == 0:
        return image

    padding = (0, pad_width, 0, pad_height)
    return torch.nn.functional.pad(image, padding, mode='constant', value=0.0)

def cut_videos(videos):
    t = videos.size(1)
    if t < 1:
        raise ValueError("SeedVR2Preprocess expected at least one frame.")
    if t == 1:
        return videos
    if t <= 4:
        padding = videos[:, -1:].repeat(1, 4 - t + 1, 1, 1, 1)
        return torch.cat([videos, padding], dim=1)
    if (t - 1) % 4 == 0:
        return videos
    padding = videos[:, -1:].repeat(1, 4 - ((t - 1) % 4), 1, 1, 1)
    videos = torch.cat([videos, padding], dim=1)
    if (videos.size(1) - 1) % 4 != 0:
        raise ValueError(f"SeedVR2Preprocess failed to pad video length to 4n+1; got {videos.size(1)} frames.")
    return videos

def _seedvr2_input_shorter_edge(images, node_name):
    if images.dim() == 4:
        return min(images.shape[1], images.shape[2])
    if images.dim() == 5:
        return min(images.shape[2], images.shape[3])
    raise ValueError(
        f"{node_name}: expected 4-D or 5-D IMAGE tensor, "
        f"got shape {tuple(images.shape)}"
    )


def _seedvr2_pad(images, upscaled_shorter_edge, node_name):
    if upscaled_shorter_edge < 2:
        raise ValueError(
            f"{node_name}: input shorter edge must be at least 2 pixels; "
            f"got {upscaled_shorter_edge}."
        )
    if images.shape[-1] > 3:
        images = images[..., :3]
    if images.dim() == 4:
        # Comfy video components arrive as a 4-D IMAGE frame sequence:
        # (frames, H, W, C). SeedVR2 consumes that as one video.
        images = images.unsqueeze(0)
    elif images.dim() != 5:
        raise ValueError(
            f"{node_name}: expected 4-D or 5-D IMAGE tensor, "
            f"got shape {tuple(images.shape)}"
        )
    images = images.permute(0, 1, 4, 2, 3)

    b, t, c, h, w = images.shape
    images = images.reshape(b * t, c, h, w)

    images = torch.clamp(images, 0.0, 1.0)
    images = div_pad(images, (16, 16))
    _, _, new_h, new_w = images.shape

    images = images.reshape(b, t, c, new_h, new_w)
    images = cut_videos(images)
    images_bthwc = images.permute(0, 1, 3, 4, 2).contiguous()

    return io.NodeOutput(images_bthwc)


class SeedVR2Preprocess(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SeedVR2Preprocess",
            display_name="Pre-Process SeedVR2 Input",
            category="image/pre-processors",
            description="Pad a resized image for SeedVR2 model. Alpha channel is dropped. The node Post-Process SeedVR2 Output re-applies it from the original resized image.",
            search_aliases=["seedvr2", "upscale", "video upscale", "pad", "preprocess"],
            inputs=[
                io.Image.Input("resized_images", tooltip="The resized image to process."),
            ],
            outputs=[
                io.Image.Output("images", tooltip="The padded image for VAE encoding."),
            ]
        )

    @classmethod
    def execute(cls, resized_images):
        upscaled_shorter_edge = _seedvr2_input_shorter_edge(resized_images, "SeedVR2Preprocess")
        return _seedvr2_pad(
            resized_images, upscaled_shorter_edge, "SeedVR2Preprocess",
        )


class SeedVR2PostProcessing(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SeedVR2PostProcessing",
            display_name="Post-Process SeedVR2 Output",
            category="image/post-processors",
            description="Align the generated image with the original resized image and apply color correction.",
            search_aliases=["seedvr2", "upscale", "color correction", "color match", "postprocess"],
            inputs=[
                io.Image.Input("images", tooltip="The generated image to process."),
                io.Image.Input("original_resized_images", tooltip="The original resized image before pre-processing, used as reference."),
                io.Combo.Input("color_correction_method", options=["lab", "wavelet", "adain", "none"], default="lab", tooltip="Method to match the generated image colors to the original image. lab: transfer color in CIELAB space, preserving detail (most faithful). wavelet: transfer low-frequency color, keeping upscaled high-frequency detail. adain: match per-channel mean/std (fastest, global tint). none: skip color transfer (geometry alignment only)."),
            ],
            outputs=[io.Image.Output(display_name="images", tooltip="The aligned, color-corrected image.")],
        )

    @classmethod
    def execute(cls, images, original_resized_images, color_correction_method):
        alpha_input = None
        if original_resized_images.shape[-1] == 4:
            alpha_input = original_resized_images[..., 3:4]
            original_resized_images = original_resized_images[..., :3]
        decoded_5d, decoded_was_4d = cls._as_bthwc(images)
        reference_full, _ = cls._as_bthwc(original_resized_images)
        decoded_5d = cls._restore_reference_batch_time(decoded_5d, reference_full)

        b = min(decoded_5d.shape[0], reference_full.shape[0])
        t = min(decoded_5d.shape[1], reference_full.shape[1])
        reference_h = reference_full.shape[2]
        reference_w = reference_full.shape[3]

        decoded_5d = decoded_5d[:b, :t, :, :, :]
        target_h = min(decoded_5d.shape[2], reference_h)
        target_w = min(decoded_5d.shape[3], reference_w)
        decoded_5d = decoded_5d[:, :, :target_h, :target_w, :]
        if color_correction_method in ("lab", "wavelet", "adain"):
            reference_5d = reference_full[:b, :t, :, :, :]
            reference_5d = cls._resize_reference(reference_5d, target_h, target_w)
            output_device = decoded_5d.device
            decoded_raw = cls._to_seedvr2_raw(decoded_5d)
            reference_raw = cls._to_seedvr2_raw(reference_5d)
            decoded_flat = decoded_raw.permute(0, 1, 4, 2, 3).reshape(b * t, decoded_raw.shape[4], target_h, target_w)
            reference_flat = reference_raw.permute(0, 1, 4, 2, 3).reshape(b * t, reference_raw.shape[4], target_h, target_w)
            output = cls._color_transfer_chunked(
                decoded_flat, reference_flat, output_device, color_correction_method,
            )
            output = output.reshape(b, t, output.shape[1], output.shape[2], output.shape[3]).permute(0, 1, 3, 4, 2)
            output = output.add(1.0).div(2.0).clamp(0.0, 1.0)
        elif color_correction_method == "none":
            output = decoded_5d
        else:
            raise ValueError(f"SeedVR2PostProcessing: unknown color_correction_method {color_correction_method!r}")

        if alpha_input is not None:
            alpha_5d, _ = cls._as_bthwc(alpha_input)
            alpha_5d = alpha_5d[:output.shape[0], :output.shape[1], :output.shape[2], :output.shape[3], :]
            output = torch.cat([output, alpha_5d.to(dtype=output.dtype, device=output.device)], dim=-1)
        h2 = output.shape[-3] - (output.shape[-3] % 2)
        w2 = output.shape[-2] - (output.shape[-2] % 2)
        output = output[:, :, :h2, :w2, :]
        if decoded_was_4d:
            output = output.reshape(-1, output.shape[-3], output.shape[-2], output.shape[-1])
        return io.NodeOutput(output)

    @staticmethod
    def _as_bthwc(images):
        if images.ndim == 4:
            return images.unsqueeze(0), True
        if images.ndim == 5:
            return images, False
        raise ValueError(
            f"SeedVR2PostProcessing: expected 4-D or 5-D IMAGE tensor, got shape {tuple(images.shape)}"
        )

    @staticmethod
    def _restore_reference_batch_time(decoded, reference):
        if decoded.shape[0] != 1:
            return decoded
        ref_b, ref_t = reference.shape[:2]
        if ref_b < 1 or decoded.shape[1] % ref_b != 0:
            return decoded
        decoded_t = decoded.shape[1] // ref_b
        if decoded_t < ref_t:
            return decoded
        return decoded.reshape(ref_b, decoded_t, decoded.shape[2], decoded.shape[3], decoded.shape[4])

    @staticmethod
    def _to_seedvr2_raw(images):
        return images.mul(2.0).sub(1.0)

    @staticmethod
    def _color_transfer_on_vae_device(decoded_flat, reference_flat, output_device, transfer_fn):
        color_device = comfy.model_management.vae_device()
        decoded_flat = decoded_flat.to(device=color_device)
        reference_flat = reference_flat.to(device=color_device)
        output = transfer_fn(decoded_flat, reference_flat)
        return output.to(device=output_device)

    @staticmethod
    def _lab_color_transfer_on_vae_device(decoded_flat, reference_flat, output_device):
        color_device = comfy.model_management.vae_device()
        result = None
        for start in range(decoded_flat.shape[0]):
            decoded_frame = decoded_flat[start:start + 1].to(device=color_device).clone()
            reference_frame = reference_flat[start:start + 1].to(device=color_device).clone()
            output = lab_color_transfer(decoded_frame, reference_frame).to(device=output_device)
            if result is None:
                result = torch.empty(
                    (decoded_flat.shape[0],) + tuple(output.shape[1:]),
                    device=output_device,
                    dtype=output.dtype,
                )
            result[start:start + 1].copy_(output)
        if result is None:
            raise ValueError("SeedVR2PostProcessing: LAB color correction requires at least one frame.")
        return result

    @classmethod
    def _color_transfer_chunked(cls, decoded_flat, reference_flat, output_device, color_correction_method):
        chunk_size = cls._estimate_color_correction_chunk_size(decoded_flat, color_correction_method)
        while True:
            try:
                return cls._run_color_transfer_chunks(
                    decoded_flat, reference_flat, output_device, color_correction_method, chunk_size,
                )
            except Exception as e:
                comfy.model_management.raise_non_oom(e)
                if chunk_size <= 1:
                    raise RuntimeError(
                        "SeedVR2PostProcessing: color correction OOM at one frame; "
                        f"color_correction_method={color_correction_method}, shape={tuple(decoded_flat.shape)}."
                    ) from e
                chunk_size = max(1, chunk_size // SEEDVR2_OOM_BACKOFF_DIVISOR)

    @classmethod
    def _run_color_transfer_chunks(cls, decoded_flat, reference_flat, output_device, color_correction_method, chunk_size):
        result = None
        for start in range(0, decoded_flat.shape[0], chunk_size):
            end = min(start + chunk_size, decoded_flat.shape[0])
            decoded_chunk = decoded_flat[start:end]
            reference_chunk = reference_flat[start:end]
            if color_correction_method == "lab":
                output = cls._lab_color_transfer_on_vae_device(decoded_chunk, reference_chunk, output_device)
            elif color_correction_method == "wavelet":
                output = cls._color_transfer_on_vae_device(
                    decoded_chunk, reference_chunk, output_device, wavelet_color_transfer,
                )
            else:
                output = cls._color_transfer_on_vae_device(
                    decoded_chunk, reference_chunk, output_device, adain_color_transfer,
                )
            if result is None:
                result = torch.empty(
                    (decoded_flat.shape[0],) + tuple(output.shape[1:]),
                    device=output_device,
                    dtype=output.dtype,
                )
            result[start:end].copy_(output)
        if result is None:
            raise ValueError("SeedVR2PostProcessing: color correction requires at least one frame.")
        return result

    @classmethod
    def _estimate_color_correction_chunk_size(cls, decoded_flat, color_correction_method):
        multiplier = cls._color_correction_memory_multiplier(color_correction_method)
        frames = decoded_flat.shape[0]
        _, channels, height, width = decoded_flat.shape
        dtype_bytes = max(decoded_flat.element_size(), SEEDVR2_DTYPE_BYTES_FLOOR)
        bytes_per_frame = height * width * channels * dtype_bytes * multiplier
        if bytes_per_frame <= 0:
            return frames
        color_device = comfy.model_management.vae_device()
        free_memory = comfy.model_management.get_free_memory(color_device)
        chunk_size = int((free_memory * SEEDVR2_COLOR_MEM_HEADROOM) // bytes_per_frame)
        return max(1, min(frames, chunk_size))

    @staticmethod
    def _color_correction_memory_multiplier(color_correction_method):
        if color_correction_method == "lab":
            return SEEDVR2_LAB_SCALE_MULTIPLIER
        if color_correction_method == "wavelet":
            return SEEDVR2_WAVELET_SCALE_MULTIPLIER
        if color_correction_method == "adain":
            return SEEDVR2_ADAIN_SCALE_MULTIPLIER
        raise ValueError(f"SeedVR2PostProcessing: unknown color_correction_method {color_correction_method!r}")

    @staticmethod
    def _resize_reference(reference, height, width):
        if reference.shape[2] == height and reference.shape[3] == width:
            return reference
        b, t = reference.shape[:2]
        reference_flat = reference.permute(0, 1, 4, 2, 3).reshape(b * t, reference.shape[4], reference.shape[2], reference.shape[3])
        resized = TVF.resize(
            reference_flat,
            size=(height, width),
            interpolation=InterpolationMode.BICUBIC,
            antialias=not (isinstance(reference_flat, torch.Tensor) and reference_flat.device.type == "mps"),
        )
        return resized.reshape(b, t, resized.shape[1], height, width).permute(0, 1, 3, 4, 2)


class SeedVR2Conditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SeedVR2Conditioning",
            display_name="Apply SeedVR2 Conditioning",
            category="model/conditioning",
            description="Build SeedVR2 positive/negative conditioning from a VAE latent.",
            search_aliases=["seedvr2", "upscale", "conditioning"],
            inputs=[
                io.Model.Input("model", tooltip="The SeedVR2 model."),
                io.Latent.Input("vae_conditioning", display_name="latent"),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive", tooltip="The positive conditioning for sampling."),
                io.Conditioning.Output(display_name="negative", tooltip="The negative conditioning for sampling."),
            ],
        )

    @classmethod
    def execute(cls, model, vae_conditioning) -> io.NodeOutput:

        vae_conditioning = vae_conditioning["samples"]
        if vae_conditioning.ndim != 5:
            raise ValueError(
                "SeedVR2Conditioning expects a 5-D VAE latent in Comfy "
                f"channel-first layout; got shape {tuple(vae_conditioning.shape)}."
            )
        if vae_conditioning.shape[1] != SEEDVR2_LATENT_CHANNELS:
            if vae_conditioning.shape[-1] == SEEDVR2_LATENT_CHANNELS:
                raise ValueError(
                    "SeedVR2Conditioning expects SeedVR2 VAE latents in Comfy "
                    f"channel-first layout (B, {SEEDVR2_LATENT_CHANNELS}, T, H, W); "
                    f"got channel-last shape {tuple(vae_conditioning.shape)}."
                )
            raise ValueError(
                "SeedVR2Conditioning expects SeedVR2 VAE latents with "
                f"{SEEDVR2_LATENT_CHANNELS} channels; got shape {tuple(vae_conditioning.shape)}."
            )
        vae_conditioning = vae_conditioning.movedim(1, -1).contiguous()
        model = _resolve_seedvr2_diffusion_model(model)
        pos_cond = model.positive_conditioning
        neg_cond = model.negative_conditioning

        mask = vae_conditioning.new_ones(vae_conditioning.shape[:-1] + (1,))
        condition = torch.cat((vae_conditioning, mask), dim=-1)
        condition = condition.movedim(-1, 1)

        negative = [[neg_cond.unsqueeze(0), {"condition": condition}]]
        positive = [[pos_cond.unsqueeze(0), {"condition": condition}]]

        return io.NodeOutput(positive, negative)

def _seedvr2_chunk_crossfade_weights(overlap, device, dtype):
    """Descending previous-chunk weights across the overlap (next chunk gets ``1 - w``): a Hann fade over the middle third, flat shoulders on the outer thirds."""
    ramp = torch.linspace(0.0, 1.0, steps=overlap, device=device, dtype=dtype)
    ramp = ((ramp - 1.0 / 3.0) / (1.0 / 3.0)).clamp(0.0, 1.0)
    return 0.5 + 0.5 * torch.cos(torch.pi * ramp)


class SeedVR2TemporalChunk(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SeedVR2TemporalChunk",
            display_name="Split SeedVR2 Latent",
            category="model/latent/seedvr",
            description="Split a SeedVR2 video latent into overlapping temporal chunks small enough to sample one at a time within VRAM, wiring latents outputs to both Apply SeedVR2 Conditioning and the sampler latent input before recombining with Merge SeedVR2 Latents.",
            search_aliases=["seedvr2", "split", "chunk", "temporal", "video upscale", "rebatch"],
            inputs=[
                io.Latent.Input("latent", tooltip="The VAE-encoded SeedVR2 latent to split."),
                io.Int.Input("temporal_overlap", default=0, min=0, max=16384,
                             tooltip="Latent frames shared between adjacent chunks and crossfaded at merge; 0 = no overlap."),
                io.DynamicCombo.Input("chunking_mode",
                                      tooltip="manual = use frames_per_chunk exactly; auto = predict the largest chunk that fits free VRAM.",
                                      options=[
                                          io.DynamicCombo.Option("auto", []),
                                          io.DynamicCombo.Option("manual", [
                                              io.Int.Input("frames_per_chunk", default=21, min=1, max=16384, step=4,
                                                           tooltip="Pixel frames per temporal chunk (4n+1: 1, 5, 9, 13, ...)."),
                                          ]),
                                      ]),
            ],
            outputs=[
                io.Latent.Output(display_name="latents", is_output_list=True,
                                 tooltip="The temporal chunks in sequence order."),
                io.Int.Output(display_name="temporal_overlap",
                              tooltip="The effective latent-frame overlap between adjacent chunks, for Merge SeedVR2 Latents."),
            ],
        )

    @classmethod
    def execute(cls, latent, temporal_overlap, chunking_mode) -> io.NodeOutput:
        samples = latent["samples"]
        if samples.ndim != 5:
            raise ValueError(
                f"SeedVR2TemporalChunk: expected a 5-D video latent (B, C, T, H, W); "
                f"got shape {tuple(samples.shape)}."
            )
        if samples.shape[1] != SEEDVR2_LATENT_CHANNELS:
            raise ValueError(
                f"SeedVR2TemporalChunk: expected {SEEDVR2_LATENT_CHANNELS} latent channels; "
                f"got shape {tuple(samples.shape)}."
            )
        if temporal_overlap < 0:
            raise ValueError(
                f"SeedVR2TemporalChunk: temporal_overlap must be >= 0; got {temporal_overlap}."
            )
        mode = chunking_mode["chunking_mode"]
        if mode not in ("auto", "manual"):
            raise ValueError(
                f"SeedVR2TemporalChunk: chunking_mode must be 'auto' or 'manual'; "
                f"got {mode!r}."
            )
        t_latent = samples.shape[2]
        t_pixel = 4 * (t_latent - 1) + 1

        if mode == "auto":
            free_gb = comfy.model_management.get_free_memory(
                comfy.model_management.get_torch_device()) / (1024 ** 3)
            mpx_per_frame = (samples.shape[0] * samples.shape[3] * samples.shape[4]) * (BYTEDANCE_VAE_SPATIAL_DOWNSAMPLE ** 2) / 1e6
            budget_gb = free_gb - SEEDVR2_CHUNK_RESERVED_GIB - SEEDVR2_CHUNK_SIGMA_K * SEEDVR2_CHUNK_SIGMA_GIB
            chunk_latent_max = max(1, int(budget_gb / (SEEDVR2_CHUNK_GIB_PER_MPX_FRAME * mpx_per_frame)))
            frames_per_chunk = min(4 * (chunk_latent_max - 1) + 1, t_pixel)
            logging.info(
                "SeedVR2TemporalChunk auto: free=%.2fGiB, %.2fMpx -> frames_per_chunk=%d (t_pixel=%d).",
                free_gb, mpx_per_frame, frames_per_chunk, t_pixel,
            )
        else:
            frames_per_chunk = chunking_mode["frames_per_chunk"]
            if frames_per_chunk < 1 or (frames_per_chunk - 1) % 4 != 0:
                raise ValueError(
                    f"SeedVR2TemporalChunk: frames_per_chunk must be a 4n+1 pixel-frame count "
                    f"(1, 5, 9, 13, 17, 21, ...); got {frames_per_chunk}."
                )

        if t_pixel <= frames_per_chunk:
            return io.NodeOutput([latent], 0)

        chunk_latent = (frames_per_chunk - 1) // 4 + 1
        temporal_overlap = min(temporal_overlap, chunk_latent - 1)
        step = chunk_latent - temporal_overlap

        chunks = []
        for start in range(0, t_latent, step):
            end = min(start + chunk_latent, t_latent)
            chunk = latent.copy()
            chunk["samples"] = samples[:, :, start:end].contiguous()
            chunks.append(chunk)
            if end >= t_latent:
                break
        return io.NodeOutput(chunks, temporal_overlap)


class SeedVR2TemporalMerge(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SeedVR2TemporalMerge",
            display_name="Merge SeedVR2 Latents",
            category="model/latent/seedvr",
            is_input_list=True,
            description="Recombine sampled SeedVR2 latent temporal chunks into one latent, crossfading each overlap with a Hann window sized by the temporal_overlap wired from Split SeedVR2 Latent.",
            search_aliases=["seedvr2", "merge", "temporal", "hann", "crossfade"],
            inputs=[
                io.Latent.Input("latents", tooltip="The sampled temporal chunks in sequence order."),
                io.Int.Input("temporal_overlap", default=0, min=0, max=16384, force_input=True,
                             tooltip="The temporal_overlap output of Split SeedVR2 Latent. 0 = plain concatenation."),
            ],
            outputs=[
                io.Latent.Output(display_name="latent", tooltip="The recombined full-length latent."),
            ],
        )

    @classmethod
    def execute(cls, latents, temporal_overlap) -> io.NodeOutput:
        temporal_overlap = temporal_overlap[0]
        if temporal_overlap < 0:
            raise ValueError(
                f"SeedVR2TemporalMerge: temporal_overlap must be >= 0; got {temporal_overlap}."
            )
        chunks = [entry["samples"] for entry in latents]
        first = chunks[0]
        if first.ndim != 5:
            raise ValueError(
                f"SeedVR2TemporalMerge: expected 5-D video latents (B, C, T, H, W); "
                f"chunk 0 has shape {tuple(first.shape)}."
            )
        for i, chunk in enumerate(chunks[1:], start=1):
            if chunk.shape[:2] != first.shape[:2] or chunk.shape[3:] != first.shape[3:]:
                raise ValueError(
                    f"SeedVR2TemporalMerge: chunk {i} shape {tuple(chunk.shape)} does not "
                    f"match chunk 0 shape {tuple(first.shape)} outside the temporal axis."
                )
            if i < len(chunks) - 1 and chunk.shape[2] != first.shape[2]:
                raise ValueError(
                    f"SeedVR2TemporalMerge: chunk {i} has {chunk.shape[2]} latent frames but "
                    f"chunk 0 has {first.shape[2]}; only the final chunk may be shorter."
                )

        out = latents[0].copy()
        out.pop("noise_mask", None)

        if len(chunks) == 1:
            out["samples"] = first
            return io.NodeOutput(out)
        if temporal_overlap == 0:
            out["samples"] = torch.cat(chunks, dim=2)
            return io.NodeOutput(out)

        chunk_latent = first.shape[2]
        step = chunk_latent - min(temporal_overlap, chunk_latent - 1)
        t_total = step * (len(chunks) - 1) + chunks[-1].shape[2]
        b, c, _, h, w = first.shape
        merged = torch.empty((b, c, t_total, h, w), device=first.device, dtype=first.dtype)

        merged[:, :, :chunk_latent] = first
        filled = chunk_latent
        for i, chunk in enumerate(chunks[1:], start=1):
            start = i * step
            end = start + chunk.shape[2]
            # Crossfade width is bounded by the previous fill frontier and by a runt
            # final chunk shorter than the configured overlap.
            fade = min(filled - start, chunk.shape[2])
            if fade > 0:
                w_prev = _seedvr2_chunk_crossfade_weights(
                    fade, chunk.device, chunk.dtype).view(1, 1, fade, 1, 1)
                merged[:, :, start:start + fade] = (
                    merged[:, :, start:start + fade] * w_prev + chunk[:, :, :fade] * (1.0 - w_prev)
                )
                merged[:, :, start + fade:end] = chunk[:, :, fade:]
            else:
                merged[:, :, start:end] = chunk
            filled = end

        out["samples"] = merged
        return io.NodeOutput(out)


class SeedVRExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SeedVR2Conditioning,
            SeedVR2Preprocess,
            SeedVR2PostProcessing,
            SeedVR2TemporalChunk,
            SeedVR2TemporalMerge,
        ]

async def comfy_entrypoint() -> SeedVRExtension:
    return SeedVRExtension()
