"""Kandinsky 6 nodes for ComfyUI."""
import math

import torch
from tqdm.auto import tqdm
from typing_extensions import override

import comfy.model_management as mm
import comfy.nested_tensor
import comfy.samplers
import comfy.utils
import node_helpers
from comfy_api.latest import ComfyExtension, io

from comfy.ldm.kandinsky6.core_contract import (
    GENERATION_DEFAULTS,
    LATENT_DEFAULTS,
)
from comfy.ldm.kandinsky6 import piflow


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


class Kandinsky6EmptyLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Kandinsky6EmptyLatent",
            display_name="Kandinsky 6 Empty Latent (video+audio)",
            category="model/latent/kandinsky6",
            description="Empty joint video+audio latent for Kandinsky 6; the two streams are sized to stay in sync.",
            inputs=[
                io.Int.Input("width", default=int(GENERATION_DEFAULTS["width"]), min=256, max=2048, step=8),
                io.Int.Input("height", default=int(GENERATION_DEFAULTS["height"]), min=256, max=2048, step=8),
                io.Int.Input("length", default=int(GENERATION_DEFAULTS["sample_frames"]), min=1, max=1001, step=_VIDEO_TEMPORAL_FACTOR,
                             tooltip="Pixel frames. Counts not of the form 4*n+1 are rounded to the nearest lower count."),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
            ],
            outputs=[
                io.Latent.Output(display_name="joint_latent"),
            ],
        )

    @classmethod
    def execute(cls, width, height, length, batch_size) -> io.NodeOutput:
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
        return io.NodeOutput(latent)


class Kandinsky6ImageToVideoAudio(io.ComfyNode):
    """Animate a reference image into video+audio, or run text-only.

    ``image`` is resized to the generation size and VAE-encoded here, so no
    separate VAEEncode step is needed.  Leave ``image`` unconnected for pure
    text-to-video+audio.  The encoded reference is passed through the
    conditioning; the model appends it as a clean tail frame internally, so
    the latent keeps its native shape through sampling.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Kandinsky6ImageToVideoAudio",
            display_name="Kandinsky 6 Image to Video+Audio",
            category="model/conditioning/kandinsky6",
            description="Animate a reference image into video+audio, or run text-only. The image is resized and VAE-encoded here; leave it unconnected for pure text-to-video+audio.",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("empty_latent"),
                io.Image.Input("image", optional=True),
                io.Vae.Input("vae", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="joint_latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, empty_latent, image=None, vae=None) -> io.NodeOutput:
        _validate_joint_latent(empty_latent)

        if image is None:
            return io.NodeOutput(positive, negative, empty_latent)
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
        return io.NodeOutput(positive, negative, empty_latent)


class Kandinsky6PiFlowGuider(comfy.samplers.CFGGuider):
    """Runs the vendor PiFlow schedule on a distilled Kandinsky 6 checkpoint.

    Each segment costs one DiT evaluation; the model's n_grid velocity grids are
    then integrated over the segment by a network-free policy, so the distilled
    model runs on its intended step budget. Only the positive conditioning is
    consumed (the distilled release runs without CFG). The video stream is fully
    denoised; a latent noise mask can keep part of the audio stream clean, which
    is how a custom audio track is encoded into the generation. The custom
    sampler's ``sampler`` and sigma values only set the segment count
    (len(sigmas) - 1); PiFlow keeps its own shifted schedule.
    """

    def set_conds(self, positive):
        self.inner_set_conds({"positive": positive})

    def inner_sample(self, noise, latent_image, device, sampler, sigmas,
                      denoise_mask, callback, disable_pbar, seed, latent_shapes=None):
        self.inner_model.latent_shapes = latent_shapes

        audio_mask = None
        if denoise_mask is not None:
            masks = comfy.utils.unpack_latents(denoise_mask, latent_shapes)
            if not bool(torch.all(masks[0] >= 1.0 - 1e-6)):
                raise ValueError("Kandinsky 6 PiFlow sampling supports full video denoise only.")
            if len(masks) > 1:
                audio_mask = masks[1]
        if len(self.conds["positive"]) != 1:
            raise ValueError(
                "Kandinsky 6 PiFlow sampling expects a single joint conditioning, "
                "without regional or scheduled prompts."
            )

        cond = self.conds["positive"][0]
        dtype = self.inner_model.get_dtype_inference()
        context = mm.cast_to_device(cond["cross_attn"], device, dtype)
        pooled = mm.cast_to_device(cond["pooled_output"], device, dtype)
        reference = cond.get("k6_reference", None)
        if reference is not None:
            reference = mm.cast_to_device(reference, device, dtype)

        video, audio = comfy.utils.unpack_latents(noise, latent_shapes)
        audio_clean = None
        if audio_mask is not None:
            _, audio_clean = comfy.utils.unpack_latents(latent_image, latent_shapes)
        steps = max(int(sigmas.shape[-1]) - 1, 1)

        options = dict(self.model_options.get("transformer_options", {}))
        # The rollout evaluates the DiT per segment, outside the sampler's cond
        # batch: carry the cond uuid so model wrappers can key their state on it.
        options["uuids"] = [cond["uuid"]]

        # PiFlow drives its own loop, so it owns the CLI progress bar the way the
        # k_diffusion samplers do; the preview callback only feeds the UI.
        with tqdm(total=steps, disable=disable_pbar) as pbar:
            def segment_callback(step, x0_streams, x_streams, total):
                pbar.update(1)
                if callback is None:
                    return
                x0_packed, _ = comfy.utils.pack_latents(x0_streams)
                x_packed, _ = comfy.utils.pack_latents(x_streams)
                callback(step, x0_packed, x_packed, total)

            video, audio = piflow.rollout(
                self.inner_model.diffusion_model,
                video,
                audio,
                context,
                pooled,
                steps=steps,
                dtype=dtype,
                reference=reference,
                transformer_options=options,
                callback=segment_callback,
                audio_mask=audio_mask,
                audio_clean=audio_clean,
            )
        samples, _ = comfy.utils.pack_latents((video, audio))
        return self.inner_model.process_latent_out(samples.to(torch.float32))


class Kandinsky6PiFlowGuiderNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Kandinsky6PiFlowGuider",
            display_name="Kandinsky 6 PiFlow Guider",
            category="model/sampling/guiders",
            description="Runs the vendor PiFlow schedule on a distilled Kandinsky 6 checkpoint. The video is fully denoised; a latent noise mask can keep part of the audio clean to encode a custom track.",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("conditioning"),
            ],
            outputs=[
                io.Guider.Output(display_name="guider"),
            ],
        )

    @classmethod
    def execute(cls, model, conditioning) -> io.NodeOutput:
        guider = Kandinsky6PiFlowGuider(model)
        guider.set_conds(conditioning)
        return io.NodeOutput(guider)


class Kandinsky6Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            Kandinsky6EmptyLatent,
            Kandinsky6ImageToVideoAudio,
            Kandinsky6PiFlowGuiderNode,
        ]


async def comfy_entrypoint() -> Kandinsky6Extension:
    return Kandinsky6Extension()
