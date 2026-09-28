from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.pruna import (
    PrunaPredictionRequest,
    PrunaPredictionResponse,
    PrunaPredictionStatusResponse,
    PrunaVideoInput,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_video_output,
    poll_op,
    sync_op,
    upload_audio_to_comfyapi,
    upload_image_to_comfyapi,
    validate_audio_duration,
    validate_images_aspect_ratio_closeness,
    validate_string,
)

PREDICTIONS_PATH = "/proxy/pruna/v1/predictions"
VIDEO_MODELS = ["p-video-2"]
ASPECT_RATIOS = ["16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "1:1"]
RESOLUTIONS = ["720p", "1080p"]
FPS_OPTIONS = ["24", "48"]
DURATIONS = ["auto"] + [str(i) for i in range(1, 21)]
MAX_PROMPT_LENGTH = 5000
MIN_AUDIO_DURATION = 1.0


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


async def _generate_video(cls: type[IO.ComfyNode], model_id: str, video_input: PrunaVideoInput) -> IO.NodeOutput:
    submitted = await sync_op(
        cls,
        ApiEndpoint(path=f"{PREDICTIONS_PATH}/{model_id}", method="POST"),
        response_model=PrunaPredictionResponse,
        data=PrunaPredictionRequest(input=video_input),
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
            _video_input(model, aspect_ratio=model["aspect_ratio"], audio=audio_url),
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
            _video_input(model, image=first_frame_url, last_frame_image=last_frame_url, audio=audio_url),
        )


class PrunaExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            PrunaTextToVideoNode,
            PrunaImageToVideoNode,
        ]


async def comfy_entrypoint() -> PrunaExtension:
    return PrunaExtension()
