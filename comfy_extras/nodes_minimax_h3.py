"""MiniMax H3 nodes: AV latent creation and task conditioning (t2va / fl2va / ref2va).

The H3 packed-DiT consumes, via conditioning:
- Qwen3-VL-32B hidden states with per-token modality tags (from the minimax CLIP)
- keyframe / reference condition latents, re-injected every step (never denoised)

Latents are NestedTensor pairs (video [B,24,T,H/16,W/16], audio [B,32,2,T40]);
sampling runs on the flat pack with any stock sampler (the model handles the
audio stream's shifted schedule internally).
"""

import math

import torch
import torch.nn.functional as F
import torchaudio

import nodes
import comfy.model_management
import comfy.model_sampling
import comfy.nested_tensor
import comfy.patcher_extension
import comfy.utils
import node_helpers
from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
from comfy_api.latest import ComfyExtension, io

CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048
FPS = 24
AUDIO_LATENT_FPS = 40

MiniMaxH3ReferenceImage = io.Custom("MINIMAX_H3_REFERENCE_IMAGE")


def align_frame_count(n):
    while n % 17 != 5:
        n += 1
    return n


def video_latent_t(frame_count):
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def temporal_shape(length):
    frame_count = align_frame_count(max(5, length))
    duration = frame_count / FPS
    return frame_count, video_latent_t(frame_count), round(duration * AUDIO_LATENT_FPS)


def adapt_canvas(width, height):
    """768-short-edge canvas with 768*1344 area cap, per-axis round to 32."""
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))


def _resize(image, width, height, crop):
    # image [B, H, W, C] -> [B, height, width, 3]
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _encode_ref_audio(audio_vae, audio):
    waveform = audio["waveform"]  # [B, C, L]
    sr = audio["sample_rate"]
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))  # [1, 32, 2, T]
    return z, z.shape[-1]


def _empty_av_latent(width, height, length, batch_size=1):
    frame_count, latent_t, audio_t = temporal_shape(length)
    video = torch.zeros([batch_size, 24, latent_t, height // 16, width // 16],
                        device=comfy.model_management.intermediate_device())
    audio = torch.zeros([batch_size, 32, 2, audio_t],
                        device=comfy.model_management.intermediate_device())
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


class EmptyMiniMaxH3LatentAV(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptyMiniMaxH3LatentAV",
            display_name="Empty MiniMax H3 AV Latent",
            category="model/latent/minimax",
            description="Joint video+audio latent for MiniMax H3. Duration snaps to the model's 17k+5 frame grid at 24 fps.",
            inputs=[
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17, tooltip="Frame count at 24 fps, snapped up to the model's 17k+5 grid (124 = ~5s; trained range is ~124-362, longer is untested)"),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, width, height, length) -> io.NodeOutput:
        latent, _ = _empty_av_latent(width, height, length)
        return io.NodeOutput(latent)


class MiniMaxH3ImageToVideo(io.ComfyNode):
    """t2va and fl2va: prompt (+ optional first/last keyframes) -> conditioning + AV latent."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ImageToVideo",
            display_name="MiniMax H3 Image to Video",
            category="model/conditioning/minimax",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17, tooltip="Frame count at 24 fps, snapped up to the model's 17k+5 grid (124 = ~5s; trained range is ~124-362, longer is untested)"),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @classmethod
    def execute(cls, clip, vae, prompt, width, height, length,
                first_frame=None, last_frame=None) -> io.NodeOutput:
        latent, frame_count = _empty_av_latent(width, height, length)

        images = []
        keyframes = []
        if first_frame is not None:
            # geometry anchor: plain stretch to canvas
            img = _resize(first_frame[:1], width, height, "disabled")
            images.append(img)
            keyframes.append({"resolved_frame_index": 0, "image": img})
        if last_frame is not None:
            # follower: aspect-preserving cover-crop
            img = _resize(last_frame[:1], width, height, "center")
            images.append(img)
            keyframes.append({"resolved_frame_index": frame_count - 1, "image": img})

        tokens = clip.tokenize(prompt, images=images)
        cond = clip.encode_from_tokens_scheduled(tokens)

        if keyframes:
            for kf in keyframes:
                kf["latent"] = vae.encode(kf.pop("image"))
            cond = node_helpers.conditioning_set_values(cond, {"minimax_keyframes": keyframes})
        return io.NodeOutput(cond, latent)


class MiniMaxH3AddGuide(io.ComfyNode):
    """Anchor image and/or audio guides at an arbitrary pixel frame of the target video."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AddGuide",
            display_name="Add Guide for MiniMax H3",
            category="model/conditioning/minimax",
            description="Anchor an image, a short clip, audio, or a clip with its soundtrack at any frame of a MiniMax H3 video. Chain several nodes to anchor several frames.",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Vae.Input("vae", optional=True, tooltip="Video VAE, needed when an image is connected."),
                io.Vae.Input("audio_vae", optional=True, tooltip="Audio VAE, needed when an audio is connected."),
                io.Latent.Input("latent"),
                io.Image.Input("image", optional=True, tooltip="Image or video frames to anchor. Multi-frame batches are anchored as a clip and cropped down to the model's valid clip lengths: 5, 22, 39... (17k + 5) frames. Batches shorter than 5 frames use only the first image."),
                io.Audio.Input("audio", optional=True,
                               tooltip="Soundtrack to anchor starting at the same frame index, cropped to the video's remaining duration."),
                io.Int.Input("frame_idx", default=0, min=-9999, max=9999,
                             tooltip="Frame index to anchor the image or the clip's first frame at. Negative values are counted from the end of the video."),
            ],
            outputs=[io.Conditioning.Output(display_name="positive")],
        )

    @classmethod
    def execute(cls, positive, latent, frame_idx, vae=None, audio_vae=None, image=None, audio=None) -> io.NodeOutput:
        samples = latent["samples"]
        if not samples.is_nested or len(samples.tensors) != 2 or samples.tensors[0].ndim != 5 or samples.tensors[0].shape[1] != 24:
            raise ValueError("MiniMaxH3AddGuide expects a MiniMax H3 AV latent")
        if image is None and audio is None:
            raise ValueError("MiniMaxH3AddGuide needs an image or an audio to anchor")
        video = samples.tensors[0]
        height = video.shape[3] * 16
        width = video.shape[4] * 16
        frame_count = sum(FRAME_PER_TOKEN[k % 5] for k in range(video.shape[2]))

        guide_frames = 1
        if image is not None:
            if vae is None:
                raise ValueError("anchoring guide frames needs the vae input")
            guide_frames = image.shape[0]
            if guide_frames < 5:
                guide_frames = 1
            else:
                while guide_frames % 17 != 5:
                    guide_frames -= 1

        resolved_frame_index = frame_idx if frame_idx >= 0 else frame_count + frame_idx
        if resolved_frame_index < 0 or resolved_frame_index + guide_frames > frame_count:
            if guide_frames == 1:
                raise ValueError("frame_idx {} is outside the video's {} frames".format(frame_idx, frame_count))
            raise ValueError("a {} frame guide clip at frame_idx {} does not fit in the video's {} frames".format(
                guide_frames, frame_idx, frame_count))

        keyframe = {"resolved_frame_index": resolved_frame_index}
        if image is not None:
            frames = _resize(image[:guide_frames], width, height, "center")
            keyframe["latent"] = vae.encode(frames)

        if audio is not None:
            if audio_vae is None:
                raise ValueError("anchoring guide audio needs the audio_vae input")
            audio_latent, audio_rt = _encode_ref_audio(audio_vae, audio)
            # the streams share one time axis: FRAME_RESCALE per pixel frame, 1.0 per audio latent frame
            max_rt = math.floor(samples.tensors[1].shape[-1] - FRAME_RESCALE * resolved_frame_index)
            if max_rt < 1:
                raise ValueError("frame_idx {} is past the end of the video's audio track".format(frame_idx))
            if audio_rt > max_rt:
                audio_latent = audio_latent[..., :max_rt].clone()
            keyframe["audio_latent"] = audio_latent

        keyframes = list(positive[0][1].get("minimax_keyframes", []))
        keyframes.append(keyframe)
        positive = node_helpers.conditioning_set_values(positive, {"minimax_keyframes": keyframes})
        return io.NodeOutput(positive)


class MiniMaxH3TextEncoderOnlyReference(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3TextEncoderOnlyReference",
            display_name="MiniMax H3 Text Encoder Only Reference",
            description="Mark a reference image to condition the MiniMax H3 text encoder without VAE encoding it for the diffusion model.",
            category="model/conditioning/minimax",
            inputs=[io.Image.Input("image")],
            outputs=[MiniMaxH3ReferenceImage.Output(display_name="reference_image")],
        )

    @classmethod
    def execute(cls, image) -> io.NodeOutput:
        return io.NodeOutput({"image": image})


class MiniMaxH3ReferenceToVideo(io.ComfyNode):
    """ref2va: prompt + reference images / videos / audio -> conditioning + AV latent.

    References enter the presentation in fixed order: images, then videos (each
    soundtrack's <Audio j> label right before its <Video k>), then standalone
    audio. Ordinals are 1-based per type, so the prompt refers to them as
    <Picture i> / <Video k> / <Audio j>.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ReferenceToVideo",
            description="<Picture i> / <Video k> / <Audio j> reference conditioning for MiniMax H3. Use the same tags when prompting.",
            display_name="MiniMax H3 Reference to Video",
            category="model/conditioning/minimax",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae", optional=True, tooltip="Video VAE. Without it reference images/videos only condition the text encoder."),
                io.Vae.Input("audio_vae", optional=True, tooltip="Audio VAE. Without it reference audio only conditions the text encoder."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17, tooltip="Frame count at 24 fps, (124 = ~5s, trained range is ~124-362)"),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match",
                    tooltip="Reference image sizing. 'match' scales each ref (down only, keeping aspect) to the generation's pixel area; 'max' uses the reference pipeline's 2048px short edge for best identity fidelity. Reference tokens ride through every sampling step, so 'max' can be several times slower."),
                io.Autogrow.Input("ref_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.MultiType.Input(
                            io.Image.Input("ref_image", tooltip="Reference image (downscaled to 2048 short edge if larger, never upscaled)"),
                            types=[MiniMaxH3ReferenceImage]),
                        prefix="ref_image_", min=0, max=9)),
                io.Autogrow.Input("ref_videos", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_video", tooltip="Reference video frames at 24 fps (2-15s)"),
                        prefix="ref_video_", min=0, max=3)),
                io.Autogrow.Input("ref_video_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_video_audio", tooltip="Soundtrack of the same-numbered reference video"),
                        prefix="ref_video_audio_", min=0, max=3)),
                io.Autogrow.Input("ref_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio", tooltip="Standalone reference audio"),
                        prefix="ref_audio_", min=0, max=3)),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @classmethod
    def execute(cls, clip, prompt, width, height, length, ref_image_size="match", vae=None, audio_vae=None,
                ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None) -> io.NodeOutput:
        latent, frame_count = _empty_av_latent(width, height, length)

        ref_items = []   # for the tokenizer presentation, in request order
        ref_blocks = []  # VAE-enabled subset for the DiT payload, in request order
        picture_index = 0

        for ref_image in (ref_images or {}).values():
            if ref_image is None:
                continue
            text_encoder_only = isinstance(ref_image, dict)
            img = ref_image["image"] if text_encoder_only else ref_image
            h, w = img.shape[1], img.shape[2]
            if ref_image_size == "match":
                # aspect-preserving scale (down only) to the generation's pixel area
                scale = min(1.0, math.sqrt((width * height) / (w * h)))
            else:
                scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
            tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            resized = _resize(img[:1], tw, th, "disabled")
            ref_items.append({"type": "image", "data": resized})
            if vae is not None and not text_encoder_only:
                z = vae.encode(resized)
                ref_blocks.append({"kind": "image", "picture_index": picture_index,
                                   "latent_h": th // 16, "latent_w": tw // 16, "latent": z})
            picture_index += 1

        ref_video_audios = ref_video_audios or {}
        for name, video_frames in (ref_videos or {}).items():
            if video_frames is None:
                continue
            # index-paired soundtrack: ref_video_audio_N belongs to ref_video_N
            soundtrack = ref_video_audios.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
            vh, vw = video_frames.shape[1], video_frames.shape[2]
            cw, ch = adapt_canvas(vw, vh)
            if vw * vh < cw * ch:
                cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            frames = _resize(video_frames, cw, ch, "disabled")
            if frames.shape[0] > frame_count:
                frames = frames[:frame_count]
            n = frames.shape[0]
            if n < 5:
                raise ValueError("MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps)")
            while n % 17 != 5:
                n -= 1
            frames = frames[:n]
            if soundtrack is not None:
                # the soundtrack gets its own <Audio j> label, emitted before <Video k>
                ref_items.append({"type": "audio"})
            # Qwen sees the video at 2 fps with timestamps
            sample_idx = list(range(0, frames.shape[0], FPS // 2))
            qwen_frames = frames[sample_idx]
            ref_items.append({"type": "video", "data": qwen_frames,
                              "timestamps": [i / 2.0 for i in range(len(sample_idx))]})
            if vae is None:
                continue
            z = vae.encode(frames)
            audio_latent, ref_audio_t = (None, 0)
            if soundtrack is not None and audio_vae is not None:
                audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, soundtrack)
            ref_blocks.append({"kind": "video_audio" if ref_audio_t else "video",
                               "latent_t": z.shape[2], "latent_h": ch // 16, "latent_w": cw // 16,
                               "ref_audio_t": ref_audio_t, "latent": z, "audio_latent": audio_latent})

        for audio in (ref_audios or {}).values():
            if audio is None:
                continue
            ref_items.append({"type": "audio"})
            if audio_vae is not None:
                audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, audio)
                ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})

        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if ref_blocks:
            cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})
        return io.NodeOutput(cond, latent)


class MiniMaxH3SigmaShift(io.ComfyNode):
    """Set the video/audio flow shifts coherently.

    The video shift drives the sampler's sigma schedule (ModelSamplingAV); both
    values are also handed to the DiT, which inverts the video schedule to the
    shared base grid and derives the audio schedule from it.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SigmaShift",
            description="Set the video/audio flow shifts.",
            display_name="ModelSamplingMiniMaxH3",
            search_aliases=["sigma shift", "minimax shift"],
            category="model/patch/minimax",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("shift_video", default=12.0, min=0.01, max=100.0, step=0.01),
                io.Float.Input("shift_audio", default=3.0, min=0.01, max=100.0, step=0.01),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, shift_video, shift_audio) -> io.NodeOutput:
        m = model.clone()

        class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
            pass

        original = m.get_model_object("model_sampling")
        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift_video, audio_shift=shift_audio)
        if hasattr(original, "noise_scale"):
            model_sampling.set_noise_scale(original.noise_scale)
        m.add_object_patch("model_sampling", model_sampling)

        to = m.model_options["transformer_options"] = m.model_options.get("transformer_options", {}).copy()
        to["minimax_h3_sigma_shift_video"] = shift_video
        to["minimax_h3_sigma_shift_audio"] = shift_audio
        return io.NodeOutput(m)


class MiniMaxH3FunControlPatch:
    def __init__(self, model_patch, vae, control_video, mask, source_video, strength, sigma_start, sigma_end):
        self.model_patch = model_patch
        self.vae = vae
        self.control_video = control_video
        self.mask = mask
        self.source_video = source_video
        self.strength = strength
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.control_latent = None
        self.control_latent_shape = None
        self.control_stream = None
        self.pristine_stream = None
        self.active = False

    def _fit_frames(self, frames, frame_count, width, height):
        indices = torch.arange(frame_count, device=frames.device).clamp(max=frames.shape[0] - 1)
        return comfy.utils.common_upscale(frames[indices], width, height, "bilinear", "center")

    def _encode(self, frames, target_shape):
        latent = self.vae.encode(frames.movedim(1, -1)).to(torch.float32)
        if tuple(latent.shape) != target_shape:
            raise ValueError("MiniMax H3 Fun VAE output shape {} does not match the target {}".format(tuple(latent.shape), target_shape))
        return latent

    def prepare_control_latent(self, target_shape):
        target_shape = tuple(target_shape)
        if self.control_latent is not None and self.control_latent_shape == target_shape:
            return

        latent_frames, latent_height, latent_width = target_shape[2:]
        frame_count = max((latent_frames - 2) // 5, 0) * 17 + 5
        spatial_compression = self.vae.spacial_compression_encode()
        width = latent_width * spatial_compression
        height = latent_height * spatial_compression
        loaded_models = comfy.model_management.loaded_models(only_currently_used=True)

        try:
            hint = None
            if self.control_video is not None:
                frames = self._fit_frames(self.control_video, frame_count, width, height)
                hint = self._encode(frames, target_shape)

            if self.mask is not None:
                mask = (self.mask.reshape(-1, 1, self.mask.shape[-2], self.mask.shape[-1]) > 0.5).to(torch.float32)
                indices = torch.arange(frame_count, device=mask.device).clamp(max=mask.shape[0] - 1)
                mask = comfy.utils.common_upscale(mask[indices], width, height, "bilinear", "center")
                visibility = 1.0 - (mask > 0.5).to(torch.float32)
                if self.source_video is None:
                    source = torch.zeros(frame_count, 3, height, width, dtype=visibility.dtype, device=visibility.device)
                else:
                    source = self._fit_frames(self.source_video, frame_count, width, height)
                masked_latent = self._encode(source * visibility.to(source.device), target_shape)
                if hint is None:
                    hint = torch.zeros_like(masked_latent)
                visibility_latent = F.interpolate(
                    visibility.squeeze(1)[None, None], size=(latent_frames, latent_height, latent_width),
                    mode="trilinear", align_corners=False)
                hint = torch.cat([hint, visibility_latent.to(hint.device), masked_latent.to(hint.device)], dim=1)
        finally:
            comfy.model_management.load_models_gpu(loaded_models)

        self.control_latent = hint
        self.control_latent_shape = target_shape

    def diffusion_model_wrapper(self, executor, x, timestep, context, transformer_options={}, **kwargs):
        sigmas = transformer_options.get("sigmas")
        sigma = float(sigmas[0]) if sigmas is not None else float(timestep.flatten()[0]) / 1000.0
        self.active = self.sigma_end <= sigma <= self.sigma_start
        self.control_stream = None
        if self.active:
            self.prepare_control_latent(x[0].shape)
        try:
            return executor(x, timestep, context, transformer_options, **kwargs)
        finally:
            self.control_stream = None
            self.pristine_stream = None

    def before_block(self, block_index, args):
        if not self.active or block_index != self.model_patch.model.injection_layers[0]:
            return
        # stash only: control weight loads here would clobber the base block's freshly staged weights
        self.pristine_stream = args["img"].clone()

    def after_block(self, block_index, args, out):
        if not self.active:
            return out
        control_index = self.model_patch.model.injection_layers.index(block_index)
        if control_index == 0:
            self.control_latent = self.control_latent.to(out["img"].device)
            self.control_stream = self.model_patch.model.init_stream(
                self.pristine_stream, self.control_latent, args["layout"], args["t_emb"])
            self.pristine_stream = None
        self.control_stream, skip = self.model_patch.model.step(
            control_index, self.control_stream, args["t_emb"], args["mod_segments"], args["rope_freqs"],
            transformer_options=args["transformer_options"])
        skip[args["layout"].audio_pos.to(skip.device)] = 0
        out["img"].add_(skip, alpha=self.strength)
        return out

    def to(self, device_or_dtype):
        if isinstance(device_or_dtype, torch.device):
            if self.control_latent is not None:
                self.control_latent = self.control_latent.to(device_or_dtype)
            self.control_stream = None
        return self

    def cleanup(self):
        self.control_latent = None
        self.control_latent_shape = None
        self.control_stream = None
        self.pristine_stream = None
        self.active = False

    def models(self):
        return [self.model_patch]

    def register(self, model):
        model.add_wrapper(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, self.diffusion_model_wrapper)
        for block_index in self.model_patch.model.injection_layers:
            blocks_replace = model.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit", {})
            previous = blocks_replace.get(("double_block", block_index))
            model.set_model_patch_replace(
                MiniMaxH3FunControlBlockPatch(self, block_index, previous), "dit", "double_block", block_index)


class MiniMaxH3FunControlBlockPatch:
    def __init__(self, control_patch, block_index, previous):
        self.control_patch = control_patch
        self.block_index = block_index
        self.previous = previous

    def __call__(self, args, extra_args):
        self.control_patch.before_block(self.block_index, args)
        if self.previous is None:
            out = extra_args["original_block"](args)
        else:
            out = self.previous(args, extra_args)
        return self.control_patch.after_block(self.block_index, args, out)

    def to(self, device_or_dtype):
        self.control_patch.to(device_or_dtype)
        if hasattr(self.previous, "to"):
            self.previous = self.previous.to(device_or_dtype)
        return self

    def cleanup(self):
        self.control_patch.cleanup()
        if hasattr(self.previous, "cleanup"):
            self.previous.cleanup()

    def models(self):
        models = self.control_patch.models()
        if hasattr(self.previous, "models"):
            models += self.previous.models()
        return models


class MiniMaxH3FunControlNetApply(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FunControlNetApply",
            description="Apply a MiniMax H3 Fun ControlNet to a text-to-video model as a model patch.",
            display_name="Apply MiniMax H3 Fun ControlNet",
            search_aliases=["minimax controlnet", "h3 controlnet", "video inpaint controlnet"],
            category="model/patch/minimax",
            inputs=[
                io.Model.Input("model"),
                io.ModelPatch.Input("model_patch"),
                io.Vae.Input("vae"),
                io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.001, advanced=True),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.001, advanced=True),
                io.Image.Input("control_video", optional=True),
                io.Mask.Input("mask", optional=True, tooltip="1 marks the regions to regenerate."),
                io.Image.Input("source_video", optional=True, tooltip="Video behind the mask; only read when a mask is given."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, model_patch, vae, strength, start_percent, end_percent,
                control_video=None, mask=None, source_video=None) -> io.NodeOutput:
        if strength == 0 or (control_video is None and mask is None):
            return io.NodeOutput(model)

        model_patched = model.clone()
        model_sampling = model.get_model_object("model_sampling")
        patch = MiniMaxH3FunControlPatch(
            model_patch,
            vae,
            control_video[..., :3].movedim(-1, 1) if control_video is not None else None,
            mask,
            source_video[..., :3].movedim(-1, 1) if mask is not None and source_video is not None else None,
            strength,
            float(model_sampling.percent_to_sigma(start_percent)),
            float(model_sampling.percent_to_sigma(end_percent)),
        )
        patch.register(model_patched)
        return io.NodeOutput(model_patched)


class MiniMaxH3Extension(ComfyExtension):
    async def get_node_list(self):
        return [
            EmptyMiniMaxH3LatentAV,
            MiniMaxH3ImageToVideo,
            MiniMaxH3AddGuide,
            MiniMaxH3TextEncoderOnlyReference,
            MiniMaxH3ReferenceToVideo,
            MiniMaxH3SigmaShift,
            MiniMaxH3FunControlNetApply,
            ]


async def comfy_entrypoint() -> MiniMaxH3Extension:
    return MiniMaxH3Extension()
