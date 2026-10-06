"""Kandinsky 6 nodes for ComfyUI."""
import math

import torch

import comfy.model_management as mm
import comfy.nested_tensor
import node_helpers

from comfy.ldm.kandinsky6.core_contract import (
    DIT_CONFIG,
    GENERATION_DEFAULTS,
    LATENT_DEFAULTS,
)


_VIDEO_CHANNELS = int(LATENT_DEFAULTS["video_channels"])
_VIDEO_SPATIAL_FACTOR = int(LATENT_DEFAULTS["video_spatial_compression_factor"])
_VIDEO_TEMPORAL_FACTOR = int(LATENT_DEFAULTS["video_temporal_compression_factor"])
_AUDIO_CHANNELS = int(LATENT_DEFAULTS["audio_channels"])
_AUDIO_DOWNSAMPLE_FACTOR = int(LATENT_DEFAULTS["audio_downsample_factor"])
_AUDIO_SAMPLE_RATE = int(LATENT_DEFAULTS["audio_sample_rate"])
_DIT_PATCH_SIZE = tuple(int(value) for value in DIT_CONFIG["patch_size"])


def _audio_latent_len(length, fps, downsample_factor):
    seconds = length / fps
    return int(math.ceil(seconds * _AUDIO_SAMPLE_RATE / downsample_factor))


def _validate_generation_shape(width, height, length, batch_size):
    divisibility = int(GENERATION_DEFAULTS["image_divisibility"])
    if width % divisibility or height % divisibility:
        raise ValueError(
            f"Kandinsky 6 requires width and height divisible by {divisibility}, got {width}x{height}."
        )
    if (length - 1) % _VIDEO_TEMPORAL_FACTOR:
        raise ValueError(
            "Kandinsky 6 requires a frame count of the form "
            f"{_VIDEO_TEMPORAL_FACTOR}*n+1, got {length}."
        )
    if batch_size != 1:
        raise ValueError("The initial Kandinsky 6 ComfyUI release supports batch_size=1 only.")

    latent_shape = (
        (length - 1) // _VIDEO_TEMPORAL_FACTOR + 1,
        height // _VIDEO_SPATIAL_FACTOR,
        width // _VIDEO_SPATIAL_FACTOR,
    )
    if any(size % patch for size, patch in zip(latent_shape, _DIT_PATCH_SIZE, strict=True)):
        raise ValueError(
            "Kandinsky 6 latent dimensions must be divisible by the DiT patch "
            f"size {_DIT_PATCH_SIZE}, got {latent_shape}."
        )


def _joint_streams(joint_latent):
    samples = joint_latent.get("samples")
    if not getattr(samples, "is_nested", False):
        raise ValueError(
            "Kandinsky 6 expects a joint LATENT containing video and audio streams."
        )
    streams = samples.unbind()
    if len(streams) != 2:
        raise ValueError(
            f"Kandinsky 6 expects exactly two latent streams, got {len(streams)}."
        )
    return streams


def _validate_joint_latent(joint_latent):
    video, audio = _joint_streams(joint_latent)
    if not torch.is_tensor(video) or video.ndim != 5:
        raise ValueError("Kandinsky 6 video latent must be a 5D BCHTW tensor.")
    if not torch.is_tensor(audio) or audio.ndim != 3:
        raise ValueError("Kandinsky 6 audio latent must be a 3D BTF tensor.")

    batch, channels, frames, height, width = video.shape
    if batch != 1 or audio.shape[0] != batch:
        raise ValueError("The initial Kandinsky 6 ComfyUI release requires matching batch_size=1 latents.")
    if channels != _VIDEO_CHANNELS:
        raise ValueError(
            f"Kandinsky 6 expects {_VIDEO_CHANNELS} video latent channels, got {channels}."
        )
    if audio.shape[-1] != _AUDIO_CHANNELS:
        raise ValueError(
            f"Kandinsky 6 expects {_AUDIO_CHANNELS} audio latent channels, got {audio.shape[-1]}."
        )
    latent_shape = (frames, height, width)
    if any(size < 1 for size in latent_shape) or any(
        size % patch for size, patch in zip(latent_shape, _DIT_PATCH_SIZE, strict=True)
    ):
        raise ValueError(
            "Kandinsky 6 video latent dimensions must be positive and divisible "
            f"by the DiT patch size {_DIT_PATCH_SIZE}."
        )
    if audio.shape[1] < 1:
        raise ValueError("Kandinsky 6 audio latent must contain at least one frame.")

    fps = joint_latent.get("frame_rate")
    if fps is not None:
        sample_frames = (frames - 1) * _VIDEO_TEMPORAL_FACTOR + 1
        expected_audio_frames = _audio_latent_len(
            sample_frames,
            float(fps),
            _AUDIO_DOWNSAMPLE_FACTOR,
        )
        if audio.shape[1] != expected_audio_frames:
            raise ValueError(
                "Kandinsky 6 audio/video duration mismatch: "
                f"expected {expected_audio_frames} audio latent frames, got {audio.shape[1]}."
            )


class Kandinsky6EmptyLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": int(GENERATION_DEFAULTS["width"]), "min": 256, "max": 2048, "step": 16}),
                "height": ("INT", {"default": int(GENERATION_DEFAULTS["height"]), "min": 256, "max": 2048, "step": 16}),
                "length": ("INT", {"default": int(GENERATION_DEFAULTS["sample_frames"]), "min": 1, "max": 1001, "step": _VIDEO_TEMPORAL_FACTOR,
                                   "tooltip": f"Pixel frames. Kandinsky 6 requires {_VIDEO_TEMPORAL_FACTOR}*n+1 frames."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 1}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("joint_latent",)
    FUNCTION = "build"
    CATEGORY = "Kandinsky 6"

    def build(self, width, height, length, batch_size):
        fps = float(GENERATION_DEFAULTS["fps"])
        _validate_generation_shape(width, height, length, batch_size)
        t_lat = (length - 1) // _VIDEO_TEMPORAL_FACTOR + 1
        h_lat = height // _VIDEO_SPATIAL_FACTOR
        w_lat = width // _VIDEO_SPATIAL_FACTOR
        device = mm.intermediate_device()
        video = torch.zeros(
            batch_size, _VIDEO_CHANNELS, t_lat, h_lat, w_lat, device=device
        )
        t_a = _audio_latent_len(
            length,
            fps,
            downsample_factor=_AUDIO_DOWNSAMPLE_FACTOR,
        )
        audio = torch.zeros(
            batch_size,
            t_a,
            _AUDIO_CHANNELS,
            device=device,
        )
        joint = comfy.nested_tensor.NestedTensor((video, audio))
        latent = {
            "samples": joint,
            "frame_rate": fps,
            "sample_rate": _AUDIO_SAMPLE_RATE,
        }
        _validate_joint_latent(latent)
        return latent


class Kandinsky6ImageToVideoAudio:
    """Append the clean I2VA reference tail while keeping stock Comfy inputs.

    ``reference_latent`` is intentionally produced by ComfyUI's standard
    ``VAEEncode`` node.  The only K6-specific work here is the canonical
    ``tail_cond_first_frame`` layout and its denoise mask.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "empty_latent": ("LATENT",),
                "reference_latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "joint_latent")
    FUNCTION = "apply"
    CATEGORY = "Kandinsky 6"

    def apply(self, positive, negative, empty_latent, reference_latent):
        _validate_joint_latent(empty_latent)
        video, audio = _joint_streams(empty_latent)

        reference = reference_latent.get("samples")
        if getattr(reference, "is_nested", False) or not torch.is_tensor(reference):
            raise ValueError(
                "Kandinsky 6 I2VA expects a regular video LATENT from the standard VAEEncode node."
            )
        if reference.ndim == 4:
            reference = reference.unsqueeze(2)
        if reference.ndim != 5:
            raise ValueError(
                "Kandinsky 6 I2VA reference latent must be a 5D BCHTW tensor, "
                f"got {tuple(reference.shape)}."
            )
        if reference.shape[2] != 1:
            raise ValueError(
                "Kandinsky 6 I2VA requires exactly one encoded reference frame, "
                f"got {reference.shape[2]}."
            )
        expected = (video.shape[0], video.shape[1], video.shape[3], video.shape[4])
        actual = (reference.shape[0], reference.shape[1], reference.shape[3], reference.shape[4])
        if actual != expected:
            raise ValueError(
                "Kandinsky 6 I2VA reference latent must match the empty video latent: "
                f"expected B/C/H/W {expected}, got {actual}. Resize the input image "
                "to the generation width and height before VAEEncode."
            )

        reference = reference.to(device=video.device, dtype=video.dtype)
        video_with_reference = torch.cat((video, reference), dim=2)

        # Comfy's standard inpaint mask keeps the reference clean throughout
        # KSampler.  The model adapter additionally returns the clean latent
        # from scale_latent_inpaint, matching core's per-step tail reset.
        video_mask = torch.ones_like(video_with_reference)
        video_mask[:, :, -1] = 0
        audio_mask = torch.ones_like(audio)

        latent = empty_latent.copy()
        latent["samples"] = comfy.nested_tensor.NestedTensor(
            (video_with_reference, audio)
        )
        latent["noise_mask"] = comfy.nested_tensor.NestedTensor(
            (video_mask, audio_mask)
        )
        latent["k6_reference_tail"] = True
        latent["k6_generated_video_latent_frames"] = int(video.shape[2])

        values = {"k6_reference_tail": True}
        positive = node_helpers.conditioning_set_values(positive, values)
        negative = node_helpers.conditioning_set_values(negative, values)
        return positive, negative, latent


class Kandinsky6RemoveReferenceLatent:
    """Remove the I2VA-only reference tail before standard VAE decoding."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"joint_latent": ("LATENT",)}}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("joint_latent",)
    FUNCTION = "remove"
    CATEGORY = "Kandinsky 6"

    def remove(self, joint_latent):
        video, audio = _joint_streams(joint_latent)
        generated_frames = joint_latent.get("k6_generated_video_latent_frames")
        if joint_latent.get("k6_reference_tail") is not True or not isinstance(
            generated_frames, int
        ):
            raise ValueError(
                "Kandinsky 6 reference-tail metadata is missing. Connect the output "
                "of Kandinsky6 Image to Video+Audio through KSampler first."
            )
        if generated_frames < 1 or video.shape[2] != generated_frames + 1:
            raise ValueError(
                "Kandinsky 6 sampled I2VA latent has an unexpected temporal shape: "
                f"metadata={generated_frames}, video={video.shape[2]}."
            )

        output = joint_latent.copy()
        output["samples"] = comfy.nested_tensor.NestedTensor(
            (video[:, :, :generated_frames], audio)
        )
        output.pop("noise_mask", None)
        output.pop("k6_reference_tail", None)
        output.pop("k6_generated_video_latent_frames", None)
        _validate_joint_latent(output)
        return (output,)


NODE_CLASS_MAPPINGS = {
    "Kandinsky6EmptyLatent": Kandinsky6EmptyLatent,
    "Kandinsky6ImageToVideoAudio": Kandinsky6ImageToVideoAudio,
    "Kandinsky6RemoveReferenceLatent": Kandinsky6RemoveReferenceLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Kandinsky6EmptyLatent": "Kandinsky 6 Empty Latent (video+audio)",
    "Kandinsky6ImageToVideoAudio": "Kandinsky 6 Image to Video+Audio",
    "Kandinsky6RemoveReferenceLatent": "Kandinsky 6 Remove I2VA Reference",
}
