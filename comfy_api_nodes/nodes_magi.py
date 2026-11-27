import uuid
from io import BytesIO

import torch
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input, Types
from comfy_api_nodes.apis.magi import (
    Magi2ImageToVideoRequest,
    Magi2ReferenceToVideoRequest,
    Magi2TextToVideoRequest,
    SandbaseRunResponse,
)
from comfy_api_nodes.util import (
    download_url_to_video_output,
    downscale_video_to_max_pixels,
    get_number_of_images,
    router_op,
    upload_audio_to_comfyapi,
    upload_file_to_comfyapi,
    upload_image_to_comfyapi,
    upload_images_to_comfyapi,
    validate_string,
)

ASPECT_RATIOS = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0, "4:3": 4 / 3, "3:4": 3 / 4, "21:9": 21 / 9}
MIN_REFERENCE_SECONDS = 2.0
MAX_REFERENCE_SECONDS = 30.0
MAX_REFERENCE_VIDEO_BYTES = 200_000_000

PRICE_BADGE = IO.PriceBadge(
    depends_on=IO.PriceBadgeDepends(widgets=["duration"]),
    expr="""{"type":"usd","usd": 0.1144 * widgets.duration}""",
)


def _prompt_input(tooltip: str | None) -> IO.String.Input:
    return IO.String.Input("prompt", multiline=True, default="", tooltip=tooltip)


def _settings_inputs(aspect_ratios: list[str], aspect_default: str, aspect_tooltip: str | None) -> list:
    return [
        IO.Int.Input("duration", default=5, min=4, max=30, step=1, display_mode=IO.NumberDisplay.number),
        IO.Combo.Input("aspect_ratio", options=aspect_ratios, default=aspect_default, tooltip=aspect_tooltip),
        IO.Combo.Input("resolution", options=["1080p"]),
        IO.Boolean.Input("generate_audio", default=True),
        IO.Int.Input(
            "seed",
            default=42,
            min=0,
            max=4294967295,
            control_after_generate=True,
            tooltip="Changing the seed re-runs the node; MAGI-2 results vary even with the same seed.",
        ),
    ]


def _validate_prompt(prompt: str) -> None:
    if not prompt.strip():
        raise ValueError("Prompt is required.")


def _fit_durations(durations: list[float]) -> list[float]:
    fitted = [min(d, MAX_REFERENCE_SECONDS) for d in durations]
    excess = sum(fitted) - MAX_REFERENCE_SECONDS
    for i in reversed(range(len(fitted))):
        if excess <= 0:
            break
        cut = min(excess, fitted[i] - MIN_REFERENCE_SECONDS)
        fitted[i] -= cut
        excess -= cut
    return fitted


def _fit_reference_videos(videos: list[Input.Video]) -> list[Input.Video]:
    durations = []
    for i, video in enumerate(videos, 1):
        seconds = min(video.get_duration(), video.get_frame_count() / float(video.get_frame_rate()))
        if seconds < MIN_REFERENCE_SECONDS:
            raise ValueError(f"Reference video {i} is {seconds:.2f} s; it must be at least 2 s.")
        durations.append(seconds)
    return [
        video.as_trimmed(start_time=0, duration=fitted) if fitted < seconds else video
        for video, seconds, fitted in zip(videos, durations, _fit_durations(durations))
    ]


def _fit_reference_audios(audios: list[Input.Audio]) -> list[Input.Audio]:
    result = []
    durations = []
    for audio in audios:
        waveform, sample_rate = audio["waveform"], audio["sample_rate"]
        min_samples = int(MIN_REFERENCE_SECONDS * sample_rate)
        if waveform.shape[-1] < min_samples:
            waveform = torch.nn.functional.pad(waveform, (0, min_samples - waveform.shape[-1]))
        result.append({"waveform": waveform, "sample_rate": sample_rate})
        durations.append(waveform.shape[-1] / sample_rate)
    for audio, fitted in zip(result, _fit_durations(durations)):
        audio["waveform"] = audio["waveform"][..., : int(fitted * audio["sample_rate"])]
    return result


async def _upload_reference_video(cls: type[IO.ComfyNode], video: Input.Video, index: int) -> str:
    data = BytesIO()
    video.save_to(data, format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264)
    if data.getbuffer().nbytes > MAX_REFERENCE_VIDEO_BYTES:
        data = BytesIO()
        downscale_video_to_max_pixels(video, 1920 * 1080).save_to(
            data, format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264
        )
    return await upload_file_to_comfyapi(cls, data, f"{uuid.uuid4()}.mp4", "video/mp4", f"Uploading video {index}")


async def _run(
    cls: type[IO.ComfyNode],
    mode: str,
    request: Magi2TextToVideoRequest | Magi2ImageToVideoRequest | Magi2ReferenceToVideoRequest,
) -> IO.NodeOutput:
    response = await router_op(cls, f"sandbase/magi-2-{mode}", response_model=SandbaseRunResponse, data=request)
    return IO.NodeOutput(await download_url_to_video_output(response.outputs[0].url))


class Magi2TextToVideoApi(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Magi2TextToVideoApi",
            display_name="MAGI-2 Text to Video",
            category="partner/video/MAGI",
            description="Generate a 1080p video of 4 to 30 seconds with native audio from a text prompt.",
            inputs=[
                _prompt_input(None),
                *_settings_inputs(list(ASPECT_RATIOS), "16:9", None),
            ],
            outputs=[IO.Video.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=PRICE_BADGE,
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        duration: int,
        aspect_ratio: str,
        resolution: str,
        generate_audio: bool,
        seed: int,
    ) -> IO.NodeOutput:
        _validate_prompt(prompt)
        validate_string(prompt, strip_whitespace=False, max_length=15000)
        return await _run(
            cls,
            "text-to-video",
            Magi2TextToVideoRequest(
                prompt=prompt,
                duration=duration,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                generate_audio=generate_audio,
                seed=seed,
            ),
        )


class Magi2ImageToVideoApi(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Magi2ImageToVideoApi",
            display_name="MAGI-2 Image to Video",
            category="partner/video/MAGI",
            description="Animate a first frame, or move from a first frame to a last frame, into a 1080p video "
            "of 4 to 30 seconds with native audio.",
            inputs=[
                IO.Image.Input("first_frame"),
                _prompt_input(None),
                *_settings_inputs(
                    ["auto", *ASPECT_RATIOS],
                    "auto",
                    "auto picks the ratio closest to the first frame. "
                    "MAGI-2 stretches the frames to the chosen ratio.",
                ),
                IO.Image.Input("last_frame", optional=True),
            ],
            outputs=[IO.Video.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=PRICE_BADGE,
        )

    @classmethod
    async def execute(
        cls,
        first_frame: Input.Image,
        prompt: str,
        duration: int,
        aspect_ratio: str,
        resolution: str,
        generate_audio: bool,
        seed: int,
        last_frame: Input.Image | None = None,
    ) -> IO.NodeOutput:
        _validate_prompt(prompt)
        if aspect_ratio == "auto":
            frame_ratio = first_frame.shape[2] / first_frame.shape[1]
            aspect_ratio = min(ASPECT_RATIOS, key=lambda k: abs(ASPECT_RATIOS[k] - frame_ratio))
        return await _run(
            cls,
            "image-to-video",
            Magi2ImageToVideoRequest(
                image=await upload_image_to_comfyapi(cls, first_frame, wait_label="Uploading first frame"),
                end_image=(
                    await upload_image_to_comfyapi(cls, last_frame, wait_label="Uploading last frame")
                    if last_frame is not None
                    else None
                ),
                prompt=prompt,
                duration=duration,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                generate_audio=generate_audio,
                seed=seed,
            ),
        )


class Magi2ReferenceToVideoApi(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        clip_tooltip = (
            "Clips longer than 30 s are trimmed to their first 30 s, and each type is trimmed to 30 s in total. "
            "Audio shorter than 2 s is padded with silence; video must be at least 2 s."
        )
        return IO.Schema(
            node_id="Magi2ReferenceToVideoApi",
            display_name="MAGI-2 Reference to Video",
            category="partner/video/MAGI",
            description="Generate a 1080p video of 4 to 30 seconds with native audio, guided by reference images, "
            "videos and audio to keep characters, locations, camera moves or voices consistent.",
            inputs=[
                _prompt_input(
                    "Refer to references as @Image1, @Video1, @Audio1 "
                    "(numbered per type, in input order; every image of a batch counts)."
                ),
                IO.Autogrow.Input(
                    "reference_images",
                    template=IO.Autogrow.TemplateNames(
                        IO.Image.Input("reference_image"),
                        names=[f"image_{i}" for i in range(1, 11)],
                        min=0,
                    ),
                    tooltip="Up to 10 images in total.",
                ),
                IO.Autogrow.Input(
                    "reference_videos",
                    template=IO.Autogrow.TemplateNames(
                        IO.Video.Input("reference_video"),
                        names=[f"video_{i}" for i in range(1, 6)],
                        min=0,
                    ),
                    tooltip=clip_tooltip,
                ),
                IO.Autogrow.Input(
                    "reference_audios",
                    template=IO.Autogrow.TemplateNames(
                        IO.Audio.Input("reference_audio"),
                        names=[f"audio_{i}" for i in range(1, 6)],
                        min=0,
                    ),
                    tooltip=clip_tooltip,
                ),
                *_settings_inputs(list(ASPECT_RATIOS), "16:9", None),
            ],
            outputs=[IO.Video.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=PRICE_BADGE,
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        reference_images: IO.Autogrow.Type,
        reference_videos: IO.Autogrow.Type,
        reference_audios: IO.Autogrow.Type,
        duration: int,
        aspect_ratio: str,
        resolution: str,
        generate_audio: bool,
        seed: int,
    ) -> IO.NodeOutput:
        _validate_prompt(prompt)
        images = list(reference_images.values())
        videos = list(reference_videos.values())
        audios = list(reference_audios.values())
        if not (images or videos or audios):
            raise ValueError("Connect at least one reference image, video or audio.")
        if sum(get_number_of_images(i) for i in images) > 10:
            raise ValueError("At most 10 reference images are supported.")
        videos = _fit_reference_videos(videos)
        audios = _fit_reference_audios(audios)
        image_urls = await upload_images_to_comfyapi(cls, images, max_images=10) if images else None
        video_urls = [await _upload_reference_video(cls, v, i) for i, v in enumerate(videos, 1)] or None
        audio_urls = [
            await upload_audio_to_comfyapi(
                cls, a, container_format="mp3", codec_name="libmp3lame", mime_type="audio/mpeg"
            )
            for a in audios
        ] or None
        return await _run(
            cls,
            "reference-to-video",
            Magi2ReferenceToVideoRequest(
                prompt=prompt,
                images=image_urls,
                videos=video_urls,
                audios=audio_urls,
                duration=duration,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                generate_audio=generate_audio,
                seed=seed,
            ),
        )


class MagiExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [Magi2TextToVideoApi, Magi2ImageToVideoApi, Magi2ReferenceToVideoApi]


async def comfy_entrypoint() -> MagiExtension:
    return MagiExtension()
