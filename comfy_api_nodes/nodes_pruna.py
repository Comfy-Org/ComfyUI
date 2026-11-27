import asyncio
import uuid
from io import BytesIO

from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input, Types
from comfy_api_nodes.apis.pruna import (
    PrunaAvatarInput,
    PrunaPredictionRequest,
    PrunaPredictionResponse,
    PrunaPredictionStatusResponse,
    PrunaReferenceVideoInput,
    PrunaUpscaleInput,
    PrunaUpscaleRequest,
    PrunaVideoInput,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_video_output,
    downscale_video_to_max_pixels,
    poll_op,
    sync_op,
    upload_audio_to_comfyapi,
    upload_file_to_comfyapi,
    upload_image_to_comfyapi,
    upload_video_to_comfyapi,
    validate_audio_duration,
    validate_images_aspect_ratio_closeness,
    validate_string,
)

PREDICTIONS_PATH = "/proxy/pruna/v1/predictions"
VIDEO_MODELS = ["p-video-2"]
REFERENCE_VIDEO_MODELS = ["p-video-2-pro-reference"]
ASPECT_RATIOS = ["16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "1:1"]
RESOLUTIONS = ["720p", "1080p"]
FPS_OPTIONS = ["24", "48"]
DURATIONS = ["auto"] + [str(i) for i in range(1, 21)]
MAX_PROMPT_LENGTH = 5000
MIN_AUDIO_DURATION = 1.0
UPSCALE_FRAME_LIMITS = {"720p": 2400, "1080p": 1200, "2K": 600, "4K": 300}
UPSCALE_FPS_OPTIONS = ["source", "16", "24", "25", "30", "48", "50", "60"]
UPSCALE_MAX_VIDEO_BYTES = 512 * 1024 * 1024
UPSCALE_MAX_PIXELS_BELOW_2K = 10_000_000
MAX_REFERENCE_IMAGES = 9
MAX_REFERENCES = 12
MAX_REFERENCE_SECONDS = 15.0
MAX_REFERENCE_VIDEO_PIXELS = 1920 * 1080
PRO_RESOLUTIONS = ["480p", "768p", "1080p", "2k", "4k"]
MAX_AVATAR_SECONDS = {"2k": 150.0, "4k": 60.0}
AVATAR_VOICES = [
    "Achernar (Female)",
    "Achird (Male)",
    "Algenib (Male)",
    "Algieba (Male)",
    "Alnilam (Male)",
    "Aoede (Female)",
    "Autonoe (Female)",
    "Callirrhoe (Female)",
    "Charon (Male)",
    "Despina (Female)",
    "Enceladus (Male)",
    "Erinome (Female)",
    "Fenrir (Male)",
    "Gacrux (Female)",
    "Iapetus (Male)",
    "Kore (Female)",
    "Laomedeia (Female)",
    "Leda (Female)",
    "Orus (Male)",
    "Puck (Male)",
    "Pulcherrima (Female)",
    "Rasalgethi (Male)",
    "Sadachbia (Male)",
    "Sadaltager (Male)",
    "Schedar (Male)",
    "Sulafat (Female)",
    "Umbriel (Male)",
    "Vindemiatrix (Female)",
    "Zephyr (Female)",
    "Zubenelgenubi (Male)",
]
AVATAR_LANGUAGES = [
    "English (US)",
    "English (UK)",
    "Spanish",
    "French",
    "German",
    "Italian",
    "Portuguese (Brazil)",
    "Japanese",
    "Korean",
    "Hindi",
]


def _generation_inputs() -> list:
    return [
        IO.Combo.Input(
            "duration",
            options=DURATIONS,
            default="5",
            tooltip="Length of the video in seconds. 'auto' lets the model choose the length from the prompt. "
            "Ignored when audio is connected: the video then follows the audio length, rounded up to a whole "
            "second, up to 20 seconds.",
        ),
        IO.Combo.Input(
            "resolution",
            options=RESOLUTIONS,
            default="720p",
            tooltip="Output resolution. 720p renders about 0.9 megapixels (1280x704 at 16:9), "
            "1080p about 2 megapixels (1920x1088 at 16:9).",
        ),
        IO.Combo.Input(
            "fps",
            options=FPS_OPTIONS,
            default="24",
            tooltip="Frames per second. 48 fps is not available with draft at 1080p.",
        ),
        IO.Boolean.Input(
            "draft",
            default=False,
            tooltip="Faster, less detailed render, billed at 60% of the standard rate.",
        ),
        IO.Boolean.Input(
            "generate_audio",
            default=True,
            tooltip="Generate a soundtrack for the video. Ignored when audio is connected, "
            "which becomes the soundtrack instead.",
        ),
        IO.Boolean.Input(
            "enhance_prompt",
            default=True,
            advanced=True,
            tooltip="Rewrite the prompt with more detail before generation; short prompts need it. "
            "Turn it off to reproduce a result exactly with the same seed.",
        ),
        IO.Audio.Input(
            "audio",
            optional=True,
            tooltip="Audio that drives the motion and becomes the soundtrack. At least 1 second long; "
            "audio longer than 20 seconds is truncated. Sets the video length instead of duration.",
        ),
        IO.Int.Input(
            "seed",
            default=42,
            min=0,
            max=2147483647,
            step=1,
            display_mode=IO.NumberDisplay.number,
            control_after_generate=True,
            tooltip="Seed for the generation. The same seed reproduces a result exactly only when "
            "enhance_prompt is off.",
        ),
    ]


def _text_to_video_option(model_id: str) -> IO.DynamicCombo.Option:
    return IO.DynamicCombo.Option(
        model_id,
        [
            IO.String.Input(
                "prompt",
                multiline=True,
                default="",
                tooltip=f"Describes the video, its motion and its sound. Up to {MAX_PROMPT_LENGTH} characters.",
            ),
            IO.Combo.Input(
                "aspect_ratio",
                options=ASPECT_RATIOS,
                default="16:9",
                tooltip="Aspect ratio of the output video.",
            ),
            *_generation_inputs(),
        ],
    )


def _image_to_video_option(model_id: str) -> IO.DynamicCombo.Option:
    return IO.DynamicCombo.Option(
        model_id,
        [
            IO.Image.Input(
                "first_frame",
                tooltip="Image the video starts from. The output keeps the aspect ratio of this image.",
            ),
            IO.Image.Input(
                "last_frame",
                optional=True,
                tooltip="Image the video ends on. Its aspect ratio must be close to the first frame's.",
            ),
            IO.String.Input(
                "prompt",
                multiline=True,
                default="",
                tooltip=f"Describes how the scene moves and sounds. Up to {MAX_PROMPT_LENGTH} characters.",
            ),
            *_generation_inputs(),
        ],
    )


def _reference_to_video_option(model_id: str) -> IO.DynamicCombo.Option:
    return IO.DynamicCombo.Option(
        model_id,
        [
            IO.Autogrow.Input(
                "images",
                template=IO.Autogrow.TemplatePrefix(
                    IO.Image.Input("image"), prefix="image_", min=0, max=MAX_REFERENCE_IMAGES
                ),
                tooltip="Reference images, referred to in the prompt as @Image1 to @Image9 in connection order. "
                "Up to 9 images, counting the first and last frames, and up to 12 references in total.",
            ),
            IO.Autogrow.Input(
                "videos",
                template=IO.Autogrow.TemplatePrefix(IO.Video.Input("video"), prefix="video_", min=0, max=3),
                tooltip="Reference videos, referred to in the prompt as @Video1 to @Video3 in connection order. "
                "Each 2 to 15 seconds at up to 60 fps, 15 seconds in total. Videos above 1080p are downscaled.",
            ),
            IO.Autogrow.Input(
                "audios",
                template=IO.Autogrow.TemplatePrefix(IO.Audio.Input("audio"), prefix="audio_", min=0, max=3),
                tooltip="Reference audio, referred to in the prompt as @Audio1 to @Audio3 in connection order. "
                "Up to 15 seconds in total.",
            ),
            IO.String.Input(
                "prompt",
                multiline=True,
                default="",
                tooltip=f"Describes the video. Refer to the references as @Image1, @Video1 or @Audio1. "
                f"Up to {MAX_PROMPT_LENGTH} characters.",
            ),
            IO.Image.Input("first_frame", optional=True),
            IO.Image.Input("last_frame", optional=True),
            IO.Boolean.Input(
                "keep_audio",
                default=False,
                tooltip="Use the connected audio as the soundtrack and generate the video to match it. "
                "Needs exactly one audio. Shorter audio is followed by generated sound, longer audio is cut to "
                "the duration, and audio within 1 second of the duration sets the video length.",
            ),
            IO.Combo.Input("duration", options=[str(i) for i in range(5, 15)], default="5"),
            IO.Combo.Input(
                "resolution",
                options=PRO_RESOLUTIONS,
                default="768p",
                tooltip="1080p, 2k and 4k upscale the 768p video.",
            ),
            IO.Combo.Input(
                "mode",
                options=["cost", "speed"],
                default="speed",
                tooltip="'cost' is cheaper, 'speed' renders faster.",
            ),
            IO.Combo.Input(
                "aspect_ratio",
                options=["adaptive", *ASPECT_RATIOS],
                default="16:9",
                tooltip="'adaptive' uses the most common aspect ratio among the references; "
                "when enhance_prompt is on, the prompt rewrite picks the ratio instead.",
            ),
            IO.Combo.Input(
                "enhance_prompt",
                options=["off", "turbo", "max"],
                default="turbo",
                advanced=True,
                tooltip="Rewrite the prompt with more detail before generation; 'max' is slower. "
                "Keep it on when the prompt refers to references by number. "
                "A first or last frame always uses 'turbo'.",
            ),
            IO.Int.Input(
                "seed",
                default=42,
                min=0,
                max=2147483647,
                step=1,
                display_mode=IO.NumberDisplay.number,
                control_after_generate=True,
                tooltip="Seed for the generation. The same seed reproduces a result exactly only when "
                "enhance_prompt is off.",
            ),
        ],
    )


def _price_badge() -> IO.PriceBadge:
    return IO.PriceBadge(
        depends_on=IO.PriceBadgeDepends(
            widgets=["model", "model.duration", "model.resolution", "model.draft"],
            inputs=["model.audio"],
        ),
        expr="""
        (
          $draft := $lookup(widgets, "model.draft");
          $rate := $lookup(widgets, "model.resolution") = "1080p"
            ? ($draft ? 0.0429 : 0.0715)
            : ($draft ? 0.02145 : 0.03575);
          $duration := $lookup(widgets, "model.duration");
          $audio := $lookup(inputs, "model.audio");
          ($audio and $audio.connected) or $type($duration) != "string" or $duration = "auto"
            ? {"type":"usd","usd": $rate, "format": {"suffix": "/second"}}
            : {"type":"usd","usd": $rate * $number($duration)}
        )
        """,
    )


def _reference_price_badge() -> IO.PriceBadge:
    return IO.PriceBadge(
        depends_on=IO.PriceBadgeDepends(
            widgets=["model.duration", "model.resolution", "model.mode"],
            inputs=["model.first_frame", "model.last_frame"],
            input_groups=["model.images", "model.videos", "model.audios"],
        ),
        expr="""
        (
          $speed := $lookup(widgets, "model.mode") = "speed";
          $resolution := $lookup(widgets, "model.resolution");
          $seconds := $number($lookup(widgets, "model.duration"));
          $rates := $speed
            ? {"480p": 0.0286, "768p": 0.05005, "1080p": 0.07865, "2k": 0.10725, "4k": 0.16445}
            : {"480p": 0.02145, "768p": 0.03575, "1080p": 0.06435, "2k": 0.09295, "4k": 0.15015};
          $first := $lookup(inputs, "model.first_frame");
          $last := $lookup(inputs, "model.last_frame");
          $images := $lookup(inputGroups, "model.images")
            + ($first and $first.connected ? 1 : 0) + ($last and $last.connected ? 1 : 0);
          $videos := $lookup(inputGroups, "model.videos");
          $audios := $lookup(inputGroups, "model.audios");
          $videoRate := $resolution = "480p" ? ($speed ? 0.0572 : 0.0429) : 0.0858;
          $base := $seconds * $lookup($rates, $resolution) + $images * ($speed ? 0.0143 : 0.00715);
          $videoMax := $videos * ($seconds + 0.5) > 15 ? 15 : $videos * ($seconds + 0.5);
          $audioMax := $audios * ($seconds + 0.5) > 15 ? 15 : $audios * ($seconds + 0.5);
          $videos + $audios > 0
            ? {"type":"range_usd","min_usd": $base + $videos * 2 * $videoRate,
               "max_usd": $base + $videoMax * $videoRate + $audioMax * ($speed ? 0.003575 : 0.002145),
               "format": {"approximate": true}}
            : {"type":"usd","usd": $base}
        )
        """,
    )


def _validate_generation_inputs(model: dict) -> None:
    validate_string(model["prompt"], strip_whitespace=True, min_length=1, max_length=MAX_PROMPT_LENGTH)
    if model["resolution"] == "1080p" and model["draft"] and model["fps"] == "48":
        raise ValueError("48 fps is not available with draft at 1080p; turn off draft or use 24 fps.")
    if model.get("audio") is not None:
        validate_audio_duration(model["audio"], MIN_AUDIO_DURATION)


async def _upload_audio(cls: type[IO.ComfyNode], audio: Input.Audio | None) -> str | None:
    if audio is None:
        return None
    return await upload_audio_to_comfyapi(
        cls, audio, container_format="mp3", codec_name="libmp3lame", mime_type="audio/mpeg"
    )


def _video_input(model: dict, **media: str | None) -> PrunaVideoInput:
    return PrunaVideoInput(
        prompt=model["prompt"],
        duration=None if model["duration"] == "auto" else int(model["duration"]),
        resolution=model["resolution"],
        fps=int(model["fps"]),
        draft=model["draft"],
        save_audio=model["generate_audio"],
        prompt_upsampling=model["enhance_prompt"],
        seed=model["seed"],
        **media,
    )


async def _generate_video(
    cls: type[IO.ComfyNode], model_id: str, request: PrunaPredictionRequest | PrunaUpscaleRequest
) -> IO.NodeOutput:
    submitted = await sync_op(
        cls,
        ApiEndpoint(path=f"{PREDICTIONS_PATH}/{model_id}", method="POST"),
        response_model=PrunaPredictionResponse,
        data=request,
    )
    result = await poll_op(
        cls,
        ApiEndpoint(path=f"{PREDICTIONS_PATH}/status/{submitted.id}"),
        response_model=PrunaPredictionStatusResponse,
        status_extractor=lambda r: r.status,
        completed_statuses=["succeeded"],
        failed_statuses=["failed", "canceled"],
        queued_statuses=["starting"],
    )
    if not result.generation_url:
        raise Exception("The prediction succeeded but returned no video.")
    return IO.NodeOutput(await download_url_to_video_output(result.generation_url))


class PrunaTextToVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="PrunaTextToVideoNode",
            display_name="Pruna P-Video-2 Text to Video",
            category="partner/video/Pruna",
            description="Generates a video from a text prompt using Pruna video models, "
            "with optional generated sound or an audio track to drive it.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[_text_to_video_option(model_id) for model_id in VIDEO_MODELS],
                    tooltip="Model to use.",
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=_price_badge(),
        )

    @classmethod
    async def execute(cls, model: dict):
        _validate_generation_inputs(model)
        audio_url = await _upload_audio(cls, model.get("audio"))
        return await _generate_video(
            cls,
            model["model"],
            PrunaPredictionRequest(input=_video_input(model, aspect_ratio=model["aspect_ratio"], audio=audio_url)),
        )


class PrunaImageToVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="PrunaImageToVideoNode",
            display_name="Pruna P-Video-2 Image to Video",
            category="partner/video/Pruna",
            description="Animates an image into a video using Pruna video models, "
            "with an optional last frame, generated sound or audio track.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[_image_to_video_option(model_id) for model_id in VIDEO_MODELS],
                    tooltip="Model to use.",
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=_price_badge(),
        )

    @classmethod
    async def execute(cls, model: dict):
        _validate_generation_inputs(model)
        if model.get("last_frame") is not None:
            validate_images_aspect_ratio_closeness(model["first_frame"], model["last_frame"], min_rel=0.8, max_rel=1.25)
        first_frame_url = await upload_image_to_comfyapi(
            cls, model["first_frame"], mime_type="image/png", wait_label="Uploading first frame"
        )
        last_frame_url = None
        if model.get("last_frame") is not None:
            last_frame_url = await upload_image_to_comfyapi(
                cls, model["last_frame"], mime_type="image/png", wait_label="Uploading last frame"
            )
        audio_url = await _upload_audio(cls, model.get("audio"))
        return await _generate_video(
            cls,
            model["model"],
            PrunaPredictionRequest(
                input=_video_input(model, image=first_frame_url, last_frame_image=last_frame_url, audio=audio_url)
            ),
        )


class PrunaPVideoUpscaleNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="PrunaPVideoUpscaleNode",
            display_name="Pruna P-Video Upscale",
            category="partner/video/Pruna",
            description="Upscales a video to 720p, 1080p, 2K or 4K using Pruna P-Video Upscale, "
            "keeping its aspect ratio and audio.",
            inputs=[
                IO.Video.Input(
                    "video",
                    tooltip="Video to upscale, up to 512 MiB.",
                ),
                IO.Combo.Input(
                    "target",
                    options=list(UPSCALE_FRAME_LIMITS),
                    default="1080p",
                    tooltip="Output pixel area: 720p about 0.9 megapixels, 1080p 2.1, 2K 3.7, 4K 8.3. "
                    "A video already this large keeps its size and is only enhanced. Maximum length: "
                    "2400, 1200, 600 or 300 frames, about 100, 50, 25 or 12.5 seconds at 24 fps.",
                ),
                IO.Combo.Input(
                    "fps",
                    options=UPSCALE_FPS_OPTIONS,
                    default="source",
                    tooltip="Output frame rate. 'source' keeps the rate of the video, up to 60. "
                    "A higher rate is interpolated. Output above 30 fps costs twice as much, except with turbo.",
                ),
                IO.Boolean.Input(
                    "turbo",
                    default=False,
                    tooltip="Faster upscale with softer detail, at a lower price.",
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["target", "fps", "turbo"]),
                expr="""
                (
                  $turbo := widgets.turbo = true and $now() >= "2026-11-30T17:00:00.000Z";
                  $rates := $turbo
                    ? {"720p": 0.005005, "1080p": 0.01001, "2k": 0.02002, "4k": 0.04004}
                    : {"720p": 0.0143, "1080p": 0.0286, "2k": 0.0572, "4k": 0.1144};
                  $rate := $lookup($rates, $string(widgets.target));
                  $turbo
                    ? {"type":"usd","usd": $rate,"format": {"suffix": "/s"}}
                    : widgets.fps = "source"
                      ? {"type":"range_usd","min_usd": $rate,"max_usd": $rate * 2,"format": {"suffix": "/s"}}
                      : {"type":"usd","usd": $number(widgets.fps) > 30 ? $rate * 2 : $rate,"format": {"suffix": "/s"}}
                )
                """,
            ),
        )

    @classmethod
    async def execute(cls, video: Input.Video, target: str, fps: str, turbo: bool):
        width, height = video.get_dimensions()
        if target in ("720p", "1080p") and width * height > UPSCALE_MAX_PIXELS_BELOW_2K:
            raise ValueError(f"Videos above 10 megapixels (got {width}x{height}) need target 2K or 4K.")
        output_fps = min(float(video.get_frame_rate()), 60) if fps == "source" else int(fps)
        frames = round(video.get_duration() * min(output_fps, 30))
        if frames > UPSCALE_FRAME_LIMITS[target]:
            raise ValueError(
                f"Video is too long for {target}: it needs {frames} frames (counted at up to 30 fps), "
                f"the limit is {UPSCALE_FRAME_LIMITS[target]}. Use a shorter clip, a lower target or a lower fps."
            )
        video_bytes = BytesIO()
        video.save_to(video_bytes, format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264)
        video_size = video_bytes.getbuffer().nbytes
        if video_size > UPSCALE_MAX_VIDEO_BYTES:
            raise ValueError(f"Video size ({video_size / 1024 / 1024:.1f} MiB) exceeds the 512 MiB limit.")
        video_url = await upload_file_to_comfyapi(cls, video_bytes, f"{uuid.uuid4()}.mp4", "video/mp4", "Uploading video")
        return await _generate_video(
            cls,
            "p-video-upscale",
            PrunaUpscaleRequest(
                input=PrunaUpscaleInput(
                    video=video_url,
                    target=target.lower(),
                    fps=None if fps == "source" else int(fps),
                    turbo=turbo,
                )
            ),
        )


class PrunaReferenceToVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="PrunaReferenceToVideoNode",
            display_name="Pruna P-Video-2 Pro Reference to Video",
            category="partner/video/Pruna",
            description="Generates a video from a prompt guided by reference images, videos and audio "
            "using Pruna video models.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[_reference_to_video_option(model_id) for model_id in REFERENCE_VIDEO_MODELS],
                    tooltip="Model to use.",
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=_reference_price_badge(),
        )

    @classmethod
    async def execute(cls, model: dict):
        validate_string(model["prompt"], strip_whitespace=True, min_length=1, max_length=MAX_PROMPT_LENGTH)
        images = list(model["images"].values())
        videos = list(model["videos"].values())
        audios = list(model["audios"].values())
        first_frame, last_frame = model.get("first_frame"), model.get("last_frame")
        frames = [frame for frame in (first_frame, last_frame) if frame is not None]
        if not images and not videos and not frames:
            raise ValueError("Connect at least one reference image or video, or a first or last frame.")
        if len(images) + len(frames) > MAX_REFERENCE_IMAGES:
            raise ValueError(f"At most {MAX_REFERENCE_IMAGES} images are allowed, counting the first and last frames.")
        if len(images) + len(frames) + len(videos) + len(audios) > MAX_REFERENCES:
            raise ValueError(
                f"At most {MAX_REFERENCES} references are allowed in total, counting images, the first and last frames, "
                "videos and audio."
            )
        if model["keep_audio"] and len(audios) != 1:
            raise ValueError("keep_audio needs exactly one connected audio.")
        video_durations = [video.get_duration() for video in videos]
        for i, (video, duration) in enumerate(zip(videos, video_durations), 1):
            if not 2.0 <= duration <= MAX_REFERENCE_SECONDS:
                raise ValueError(f"Video {i} must be 2 to 15 seconds long, got {duration:.2f} seconds.")
            if video.get_frame_rate() > 60:
                raise ValueError(f"Video {i} must be at most 60 fps, got {float(video.get_frame_rate()):.2f} fps.")
        if sum(video_durations) > MAX_REFERENCE_SECONDS:
            raise ValueError("Reference videos must be 15 seconds or shorter in total.")
        if sum(audio["waveform"].shape[-1] / audio["sample_rate"] for audio in audios) > MAX_REFERENCE_SECONDS:
            raise ValueError("Reference audio must be 15 seconds or shorter in total.")

        videos = [downscale_video_to_max_pixels(video, MAX_REFERENCE_VIDEO_PIXELS) for video in videos]
        image_urls, video_urls, audio_urls = await asyncio.gather(
            asyncio.gather(
                *(
                    upload_image_to_comfyapi(cls, image, mime_type="image/png", wait_label=f"Uploading image {i}")
                    for i, image in enumerate(images, 1)
                )
            ),
            asyncio.gather(
                *(
                    upload_video_to_comfyapi(cls, video, wait_label=f"Uploading video {i}")
                    for i, video in enumerate(videos, 1)
                )
            ),
            asyncio.gather(*(_upload_audio(cls, audio) for audio in audios)),
        )
        first_frame_url = None
        if first_frame is not None:
            first_frame_url = await upload_image_to_comfyapi(
                cls, first_frame, mime_type="image/png", wait_label="Uploading first frame"
            )
        last_frame_url = None
        if last_frame is not None:
            last_frame_url = await upload_image_to_comfyapi(
                cls, last_frame, mime_type="image/png", wait_label="Uploading last frame"
            )
        return await _generate_video(
            cls,
            model["model"],
            PrunaPredictionRequest(
                input=PrunaReferenceVideoInput(
                    prompt=model["prompt"],
                    image=image_urls or None,
                    video=video_urls or None,
                    audio=audio_urls or None,
                    keep_audio=model["keep_audio"],
                    first_frame_image=first_frame_url,
                    last_frame_image=last_frame_url,
                    duration=int(model["duration"]),
                    resolution=model["resolution"],
                    mode=model["mode"],
                    aspect_ratio=model["aspect_ratio"],
                    prompt_upsampler=model["enhance_prompt"],
                    seed=model["seed"],
                )
            ),
        )


class PrunaAvatarNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="PrunaAvatarNode",
            display_name="Pruna P-Video-2 Pro Avatar",
            category="partner/video/Pruna",
            description="Generates a talking video from a portrait and speech using Pruna P-Video-2 Pro Avatar: "
            "your own audio, or a script spoken with text-to-speech.",
            inputs=[
                IO.Image.Input(
                    "image",
                    tooltip="Portrait the video starts from. The video keeps its aspect ratio from 1:2 to 4:1; "
                    "more extreme images, such as 1:5, are cropped to 9:16 or 16:9.",
                ),
                IO.DynamicCombo.Input(
                    "speech",
                    options=[
                        IO.DynamicCombo.Option(
                            "script",
                            [
                                IO.String.Input(
                                    "script",
                                    multiline=True,
                                    default="",
                                    tooltip="Words the person speaks with text-to-speech. The video matches "
                                    "the length of the speech; speech longer than 3 minutes is cut at 3 minutes.",
                                ),
                                IO.Combo.Input("voice", options=AVATAR_VOICES, default="Zephyr (Female)"),
                                IO.Combo.Input("language", options=AVATAR_LANGUAGES, default="English (US)"),
                                IO.String.Input(
                                    "voice_prompt",
                                    default="Say the following.",
                                    tooltip="How the script is spoken: tone, pacing, accent, emotion. "
                                    "It is not spoken itself.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "audio",
                            [
                                IO.Audio.Input(
                                    "audio",
                                    tooltip="Speech the person lip-syncs, at least 1 second. The video matches "
                                    "its length; audio longer than 3 minutes is cut at 3 minutes.",
                                ),
                            ],
                        ),
                    ],
                    tooltip="Drive the video with a text-to-speech script or your own audio.",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="The person is talking.",
                    tooltip=f"Describes what the video shows. Up to {MAX_PROMPT_LENGTH} characters.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=PRO_RESOLUTIONS,
                    default="768p",
                    tooltip="1080p, 2k and 4k upscale the 768p video. 2k is limited to 150 seconds and 4k to 60 seconds.",
                ),
                IO.Boolean.Input(
                    "enhance_prompt",
                    default=True,
                    advanced=True,
                    tooltip="Add the motion seen in the portrait, such as background, lights and body movement, "
                    "to the prompt. Turn it off to use the prompt as written and to reproduce a result exactly "
                    "with the same seed.",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed for the generation. The same seed reproduces a result exactly only when "
                    "enhance_prompt is off.",
                ),
            ],
            outputs=[
                IO.Video.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["resolution"]),
                expr="""
                (
                  $rates := {"480p": 0.0286, "768p": 0.0572, "1080p": 0.0858, "2k": 0.1144, "4k": 0.1716};
                  {"type":"usd","usd": $lookup($rates, widgets.resolution),"format": {"suffix": "/second"}}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        speech: dict,
        prompt: str,
        resolution: str,
        enhance_prompt: bool,
        seed: int,
    ) -> IO.NodeOutput:
        validate_string(prompt, max_length=MAX_PROMPT_LENGTH)
        if speech["speech"] == "audio":
            validate_audio_duration(speech["audio"], MIN_AUDIO_DURATION)
            seconds = speech["audio"]["waveform"].shape[-1] / speech["audio"]["sample_rate"]
            max_seconds = MAX_AVATAR_SECONDS.get(resolution)
            if max_seconds is not None and seconds > max_seconds:
                raise ValueError(
                    f"{resolution} is limited to {max_seconds:g} seconds of audio, got {seconds:.1f} seconds. "
                    "Use shorter audio or a lower resolution."
                )
        else:
            validate_string(speech["script"], strip_whitespace=True, field_name="script", min_length=1)
        image_url = await upload_image_to_comfyapi(cls, image, mime_type="image/png", wait_label="Uploading portrait")
        if speech["speech"] == "audio":
            speech_fields = {"audio": await _upload_audio(cls, speech["audio"])}
        else:
            speech_fields = {
                "voice_script": speech["script"],
                "voice": speech["voice"],
                "voice_language": speech["language"],
                "voice_prompt": speech["voice_prompt"].strip() or None,
            }
        return await _generate_video(
            cls,
            "p-video-2-pro-avatar",
            PrunaPredictionRequest(
                input=PrunaAvatarInput(
                    image=image_url,
                    prompt=prompt.strip() or None,
                    prompt_upsampling=enhance_prompt,
                    resolution=resolution,
                    seed=seed,
                    **speech_fields,
                )
            ),
        )


class PrunaExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            PrunaTextToVideoNode,
            PrunaImageToVideoNode,
            PrunaPVideoUpscaleNode,
            PrunaReferenceToVideoNode,
            PrunaAvatarNode,
        ]


async def comfy_entrypoint() -> PrunaExtension:
    return PrunaExtension()
