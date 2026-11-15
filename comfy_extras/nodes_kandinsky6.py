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


def _encode_frames(vae, image, width, height):
    """Resize a batched image or clip onto the canvas and encode it to video latent frames."""
    resized = comfy.utils.common_upscale(
        image.movedim(-1, 1), width, height, "bilinear", "center"
    ).movedim(1, -1)
    latent = vae.encode(resized)
    if latent.ndim == 4:
        latent = latent.unsqueeze(2)
    return latent


def _pin_video_frames(joint_latent, frames, latent_idx):
    """Write encoded frames into the video stream at latent_idx and keep them clean via the noise mask."""
    video, audio = _joint_streams(joint_latent)
    frames = frames.to(device=video.device, dtype=video.dtype)
    if frames.shape[0] == 1 and video.shape[0] != 1:
        frames = frames.expand(video.shape[0], -1, -1, -1, -1)
    count = frames.shape[2]
    if frames.shape[3:] != video.shape[3:]:
        raise ValueError(
            f"the guide's spatial size {frames.shape[3]}x{frames.shape[4]} must match the "
            f"latent's {video.shape[3]}x{video.shape[4]}."
        )
    if latent_idx < 0 or latent_idx + count > video.shape[2]:
        raise ValueError(
            f"the guide's {count} latent frame(s) at index {latent_idx} do not fit the "
            f"{video.shape[2]}-frame video latent."
        )

    mask = joint_latent.get("noise_mask")
    if mask is None:
        video_mask = torch.ones(video.shape[0], 1, video.shape[2], 1, 1, device=video.device)
        audio_mask = torch.ones(audio.shape[0], 1, 1, audio.shape[1], device=audio.device)
    elif getattr(mask, "is_nested", False):
        video_mask, audio_mask = list(mask.unbind())
        video_mask = video_mask.clone()
    else:
        if mask.ndim != 5:
            raise ValueError(
                "a Kandinsky 6 joint latent noise mask must be a nested AV mask or a 5D video mask."
            )
        video_mask = mask.clone()
        audio_mask = torch.ones(audio.shape[0], 1, 1, audio.shape[1], device=audio.device)

    video = video.clone()
    video[:, :, latent_idx:latent_idx + count] = frames
    video_mask[:, :, latent_idx:latent_idx + count] = 0.0

    out = dict(joint_latent)
    out["samples"] = comfy.nested_tensor.NestedTensor((video, audio))
    out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
    return out


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
    """Animate frame images into video+audio, or run text-only.

    Connected frames are resized to the generation size and VAE-encoded here,
    so no separate VAEEncode step is needed.  Each one is pinned into the
    latent at the first or last latent frame with the noise mask, so the
    output video starts or ends exactly on it.  The start frame is
    additionally passed through the conditioning as the model's I2VA
    reference.  ``empty_latent`` can be left unconnected: the node then builds
    one from a connected image, using its size and the default frame count.
    Leave both images unconnected for pure text-to-video+audio.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Kandinsky6ImageToVideoAudio",
            display_name="Kandinsky 6 Image to Video+Audio",
            category="model/conditioning/kandinsky6",
            description="Animate a start and/or end frame image into video+audio, or run text-only. Frames are resized and VAE-encoded here and pinned into the returned latent, so the video starts or ends exactly on them. Leave empty_latent unconnected to build it from the image size.",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("empty_latent", optional=True),
                io.Image.Input("start_image", optional=True),
                io.Image.Input("end_image", optional=True),
                io.Vae.Input("vae", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="joint_latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, empty_latent, start_image=None, end_image=None, vae=None) -> io.NodeOutput:
        if empty_latent is None:
            anchor = start_image if start_image is not None else end_image
            if anchor is None:
                raise ValueError("Connect an empty latent or a frame image to size the generation.")
            width = (anchor.shape[2] // _VIDEO_SPATIAL_FACTOR) * _VIDEO_SPATIAL_FACTOR
            height = (anchor.shape[1] // _VIDEO_SPATIAL_FACTOR) * _VIDEO_SPATIAL_FACTOR
            if width < _VIDEO_SPATIAL_FACTOR or height < _VIDEO_SPATIAL_FACTOR:
                raise ValueError(
                    f"the frame image is too small: it needs at least "
                    f"{_VIDEO_SPATIAL_FACTOR}x{_VIDEO_SPATIAL_FACTOR} pixels."
                )
            empty_latent = Kandinsky6EmptyLatent.execute(
                width, height, int(GENERATION_DEFAULTS["sample_frames"]), 1
            )[0]
        else:
            _validate_joint_latent(empty_latent)

        if start_image is None and end_image is None:
            return io.NodeOutput(positive, negative, empty_latent)
        if vae is None:
            raise ValueError("Connect the video VAE to encode the frame images.")

        video, _ = _joint_streams(empty_latent)
        down = vae.spacial_compression_encode()
        target_w = int(video.shape[4] * down)
        target_h = int(video.shape[3] * down)

        if start_image is not None:
            start = _encode_frames(vae, start_image[:1], target_w, target_h)
            empty_latent = _pin_video_frames(empty_latent, start, 0)
            reference = start.to(device=video.device, dtype=video.dtype)
            values = {"k6_reference": reference}
            positive = node_helpers.conditioning_set_values(positive, values)
            negative = node_helpers.conditioning_set_values(negative, values)

        if end_image is not None:
            end = _encode_frames(vae, end_image[:1], target_w, target_h)
            empty_latent = _pin_video_frames(empty_latent, end, video.shape[2] - 1)

        return io.NodeOutput(positive, negative, empty_latent)


class Kandinsky6AddGuide(io.ComfyNode):
    """Anchor an image, or a short clip batched as frames, at a pixel frame of the joint latent.

    The frames are resized and VAE-encoded here and pinned into the latent with
    the noise mask, so the PiFlow rollout keeps them clean and the video
    interpolates between the anchored frames. Chain several nodes to anchor
    several frames.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Kandinsky6AddGuide",
            display_name="Kandinsky 6 Add Guide",
            category="model/conditioning/kandinsky6",
            description="Anchor an image, or a short clip batched as frames, at any pixel frame of a Kandinsky 6 joint latent. The frames are VAE-encoded here and kept clean by the noise mask, so the video interpolates between them. Chain several nodes to anchor several frames.",
            inputs=[
                io.Latent.Input("latent"),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("image", optional=True,
                               tooltip="Image, or a clip batched as frames. Clip guides are cropped to the model's 4*n+1 frame grid and start at the nearest lower multiple of 4."),
                io.Int.Input("frame_idx", default=0, min=-9999, max=9999,
                             tooltip="Pixel frame to anchor the guide at. Negative values are counted from the end of the video. "
                                     "The anchored frame decodes at the start of the 4-frame group containing the requested index."),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, latent, frame_idx, vae=None, image=None) -> io.NodeOutput:
        if image is None:
            raise ValueError("Connect an image or clip to anchor.")
        if vae is None:
            raise ValueError("Connect the video VAE to encode the guide.")

        video, _ = _joint_streams(latent)
        frame_count = (video.shape[2] - 1) * _VIDEO_TEMPORAL_FACTOR + 1
        resolved = frame_idx if frame_idx >= 0 else frame_count + frame_idx
        frames = image.shape[0]
        if frames > 1:
            resolved = (resolved // _VIDEO_TEMPORAL_FACTOR) * _VIDEO_TEMPORAL_FACTOR
            frames = (frames - 1) // _VIDEO_TEMPORAL_FACTOR * _VIDEO_TEMPORAL_FACTOR + 1
            image = image[:frames]

        down = vae.spacial_compression_encode()
        guide = _encode_frames(vae, image, int(video.shape[4] * down), int(video.shape[3] * down))
        return io.NodeOutput(_pin_video_frames(latent, guide, resolved // _VIDEO_TEMPORAL_FACTOR))


class Kandinsky6AddLatentGuide(io.ComfyNode):
    """Pin an already-encoded video latent at a latent frame of the joint latent.

    Same pinning as Kandinsky 6 Add Guide without the VAE round trip, for when
    the guide comes out of an earlier stage (another generation, an upscale,
    a decode/encode pair) as a latent instead of an image.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Kandinsky6AddLatentGuide",
            display_name="Kandinsky 6 Add Latent Guide",
            category="model/conditioning/kandinsky6",
            description="Pin an already-encoded video latent at any latent frame of a Kandinsky 6 joint latent, kept clean by the noise mask. Same effect as Kandinsky 6 Add Guide without the VAE decode/encode round trip.",
            inputs=[
                io.Latent.Input("latent", tooltip="Target Kandinsky 6 joint latent the guide is pinned onto."),
                io.Latent.Input("guide", tooltip="Guide latent: a video latent, or a joint AV latent whose video stream is pinned. Its spatial size must match the target's."),
                io.Int.Input("latent_idx", default=0, min=-9999, max=9999,
                             tooltip="Latent frame to pin the guide at. Negative values are counted from the end of the latent."),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, latent, guide, latent_idx) -> io.NodeOutput:
        video, _ = _joint_streams(latent)
        samples = guide["samples"]
        if getattr(samples, "is_nested", False):
            guide_video = list(samples.unbind())[0]
        else:
            guide_video = samples
        if guide_video.ndim != 5 or guide_video.shape[1] != _VIDEO_CHANNELS:
            raise ValueError(
                f"the guide must carry a 5D video latent with {_VIDEO_CHANNELS} channels."
            )

        resolved = latent_idx if latent_idx >= 0 else video.shape[2] + latent_idx
        return io.NodeOutput(_pin_video_frames(latent, guide_video, resolved))


class Kandinsky6PiFlowGuider(comfy.samplers.CFGGuider):
    """Runs the vendor PiFlow schedule on a distilled Kandinsky 6 checkpoint.

    Each segment costs one DiT evaluation; the model's n_grid velocity grids are
    then integrated over the segment by a network-free policy, so the distilled
    model runs on its intended step budget. Only the positive conditioning is
    consumed (the distilled release runs without CFG). A latent noise mask can
    keep part of either stream clean: pinned video frames are how start/end and
    guide frames stay exact, and a masked audio stream is how a custom audio
    track is encoded into the generation. The custom sampler's ``sampler`` and
    sigma values only set the segment count (len(sigmas) - 1); PiFlow keeps its
    own shifted schedule.
    """

    def set_conds(self, positive):
        self.inner_set_conds({"positive": positive})

    def inner_sample(self, noise, latent_image, device, sampler, sigmas,
                      denoise_mask, callback, disable_pbar, seed, latent_shapes=None):
        self.inner_model.latent_shapes = latent_shapes

        video_mask = None
        audio_mask = None
        if denoise_mask is not None:
            masks = comfy.utils.unpack_latents(denoise_mask, latent_shapes)
            video_mask = masks[0]
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
            # The reference tail is concatenated onto the video sequence inside
            # the DiT, so it must be in the same model space as the trajectory;
            # the conditioning carries VAE-space latents.
            reference = self.inner_model.latent_format.process_in(reference)
            reference = mm.cast_to_device(reference, device, dtype)

        video, audio = comfy.utils.unpack_latents(noise, latent_shapes)
        video_clean = None
        audio_clean = None
        if video_mask is not None or audio_mask is not None:
            # Pin the clean latent into the trajectory space, the same space
            # the standard sampler moves latent_image into before blending in
            # the masked region; process_latent_out brings the output back.
            clean = self.inner_model.process_latent_in(latent_image)
            video_clean, audio_clean = comfy.utils.unpack_latents(clean, latent_shapes)
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
                video_mask=video_mask,
                video_clean=video_clean,
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
            description="Runs the vendor PiFlow schedule on a distilled Kandinsky 6 checkpoint. A latent noise mask can keep part of the video (pinned start/end/guide frames) or of the audio (a custom track) clean.",
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
            Kandinsky6AddGuide,
            Kandinsky6AddLatentGuide,
            Kandinsky6PiFlowGuiderNode,
        ]


async def comfy_entrypoint() -> Kandinsky6Extension:
    return Kandinsky6Extension()
