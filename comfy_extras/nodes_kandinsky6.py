"""Kandinsky 6 nodes for ComfyUI."""
import math

import torch

import comfy.model_management as mm
import comfy.nested_tensor
import comfy.utils
import node_helpers

from comfy.ldm.kandinsky6.core_contract import (
    GENERATION_DEFAULTS,
    LATENT_DEFAULTS,
)


_VIDEO_CHANNELS = int(LATENT_DEFAULTS["video_channels"])
_VIDEO_SPATIAL_FACTOR = int(LATENT_DEFAULTS["video_spatial_compression_factor"])
_VIDEO_TEMPORAL_FACTOR = int(LATENT_DEFAULTS["video_temporal_compression_factor"])
_AUDIO_CHANNELS = int(LATENT_DEFAULTS["audio_channels"])
_AUDIO_DOWNSAMPLE_FACTOR = int(LATENT_DEFAULTS["audio_downsample_factor"])
_AUDIO_SAMPLE_RATE = int(LATENT_DEFAULTS["audio_sample_rate"])


def _audio_latent_len(length, fps, downsample_factor):
    seconds = length / fps
    return int(math.ceil(seconds * _AUDIO_SAMPLE_RATE / downsample_factor))


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
    if audio.shape[0] != batch:
        raise ValueError("Kandinsky 6 video and audio batch sizes must match.")
    if channels != _VIDEO_CHANNELS:
        raise ValueError(
            f"Kandinsky 6 expects {_VIDEO_CHANNELS} video latent channels, got {channels}."
        )
    if audio.shape[-1] != _AUDIO_CHANNELS:
        raise ValueError(
            f"Kandinsky 6 expects {_AUDIO_CHANNELS} audio latent channels, got {audio.shape[-1]}."
        )
    if any(size < 1 for size in (frames, height, width)):
        raise ValueError("Kandinsky 6 video latent dimensions must be positive.")
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
                "width": ("INT", {"default": int(GENERATION_DEFAULTS["width"]), "min": 256, "max": 2048, "step": 8}),
                "height": ("INT", {"default": int(GENERATION_DEFAULTS["height"]), "min": 256, "max": 2048, "step": 8}),
                "length": ("INT", {"default": int(GENERATION_DEFAULTS["sample_frames"]), "min": 1, "max": 1001, "step": _VIDEO_TEMPORAL_FACTOR,
                                   "tooltip": "Pixel frames. Counts not of the form 4*n+1 are rounded to the nearest lower count."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("joint_latent",)
    FUNCTION = "build"
    CATEGORY = "Kandinsky 6"

    def build(self, width, height, length, batch_size):
        fps = float(GENERATION_DEFAULTS["fps"])
        t_lat = (length - 1) // _VIDEO_TEMPORAL_FACTOR + 1
        # The video rounds down to the nearest 4*n+1 frame count; the audio
        # must be sized from that same effective length or the two streams
        # drift out of sync by a latent frame.
        effective_length = (t_lat - 1) * _VIDEO_TEMPORAL_FACTOR + 1
        h_lat = height // _VIDEO_SPATIAL_FACTOR
        w_lat = width // _VIDEO_SPATIAL_FACTOR
        device = mm.intermediate_device()
        video = torch.zeros(
            batch_size, _VIDEO_CHANNELS, t_lat, h_lat, w_lat, device=device
        )
        t_a = _audio_latent_len(
            effective_length,
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
        return (latent,)


class Kandinsky6ImageToVideoAudio:
    """Animate a reference image into video+audio, or run text-only.

    ``image`` is resized to the generation size and VAE-encoded here, so no
    separate VAEEncode step is needed.  Leave ``image`` unconnected for pure
    text-to-video+audio.  The encoded reference is passed through the
    conditioning; the model appends it as a clean tail frame internally, so
    the latent keeps its native shape through sampling.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "empty_latent": ("LATENT",),
            },
            "optional": {
                "image": ("IMAGE",),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "joint_latent")
    FUNCTION = "apply"
    CATEGORY = "Kandinsky 6"

    def apply(self, positive, negative, empty_latent, image=None, vae=None):
        _validate_joint_latent(empty_latent)

        if image is None:
            return positive, negative, empty_latent
        if vae is None:
            raise ValueError("Connect the video VAE to encode the I2VA reference image.")

        video, _ = _joint_streams(empty_latent)
        down = vae.spacial_compression_encode()
        target_w = int(video.shape[4] * down)
        target_h = int(video.shape[3] * down)
        resized = comfy.utils.common_upscale(
            image[:1, :, :, :3].movedim(-1, 1), target_w, target_h, "bilinear", "center"
        ).movedim(1, -1)
        reference = vae.encode(resized)
        if reference.ndim == 4:
            reference = reference.unsqueeze(2)
        reference = reference.to(device=video.device, dtype=video.dtype)

        values = {"k6_reference": reference}
        positive = node_helpers.conditioning_set_values(positive, values)
        negative = node_helpers.conditioning_set_values(negative, values)
        return positive, negative, empty_latent


NODE_CLASS_MAPPINGS = {
    "Kandinsky6EmptyLatent": Kandinsky6EmptyLatent,
    "Kandinsky6ImageToVideoAudio": Kandinsky6ImageToVideoAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Kandinsky6EmptyLatent": "Kandinsky 6 Empty Latent (video+audio)",
    "Kandinsky6ImageToVideoAudio": "Kandinsky 6 Image to Video+Audio",
}
