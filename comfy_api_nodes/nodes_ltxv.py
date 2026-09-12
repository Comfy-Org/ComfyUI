
from pydantic import BaseModel, Field
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_video_output,
    get_number_of_images,
    poll_op,
    sync_op,
    upload_audio_to_comfyapi,
    upload_images_to_comfyapi,
    validate_string,
)

V25_MODELS_MAP = {
    "LTX-2.5 (Fast)": "ltx-2-5-fast",
    "LTX-2.5 (Pro)": "ltx-2-5-pro",
}


class ExecuteTaskRequest(BaseModel):
    prompt: str = Field(...)
    model: str = Field(...)
    duration: int = Field(...)
    resolution: str = Field(...)
    fps: int | None = Field(25)
    generate_audio: bool | None = Field(True)
    image_uri: str | None = Field(None)
    last_frame_uri: str | None = Field(None)


class AudioToVideoRequest(BaseModel):
    prompt: str = Field(...)
    model: str = Field(...)
    resolution: str = Field(...)
    audio_uri: str = Field(...)
    image_uri: str | None = Field(None)


class Ltx25SubmitResponse(BaseModel):
    id: str = Field(...)


class Ltx25JobResult(BaseModel):
    video_url: str | None = Field(None)


class Ltx25JobStatusResponse(BaseModel):
    id: str = Field(...)
    status: str = Field(...)
    result: Ltx25JobResult | None = Field(None)


async def _v25_submit_and_poll(cls: type[IO.ComfyNode], route: str, data: BaseModel) -> IO.NodeOutput:
    submit = await sync_op(
        cls,
        ApiEndpoint(f"/proxy/ltx/v2/{route}", "POST"),
        response_model=Ltx25SubmitResponse,
        data=data,
        max_retries=1,
    )
    job = await poll_op(
        cls,
        ApiEndpoint(f"/proxy/ltx/v2/{route}/{submit.id}"),
        response_model=Ltx25JobStatusResponse,
        status_extractor=lambda r: r.status,
    )
    if not job.result or not job.result.video_url:
        raise RuntimeError(f"LTX job {job.id} completed without a video URL.")
    return IO.NodeOutput(await download_url_to_video_output(job.result.video_url, cls=cls))


V25_PRICE_BADGE = IO.PriceBadge(
    depends_on=IO.PriceBadgeDepends(widgets=["model", "model.duration", "model.resolution"]),
    expr="""
    (
      $prices := {
        "ltx-2.5 (fast)": {
          "1280x720":0.1287,"720x1280":0.1287,
          "1920x1080":0.1859,"1080x1920":0.1859,
          "2560x1440":0.2717,"1440x2560":0.2717,
          "3840x2160":0.429,"2160x3840":0.429
        },
        "ltx-2.5 (pro)": {
          "1280x720":0.1716,"720x1280":0.1716,
          "1920x1080":0.2431,"1080x1920":0.2431
        }
      };
      $model := $lookup(widgets, "model");
      $table := $type($model) = "string" ? $lookup($prices, $model) : undefined;
      $res := $lookup(widgets, "model.resolution");
      $pps := $type($table) = "object" and $type($res) = "string" ? $lookup($table, $res) : undefined;
      $durRaw := $lookup(widgets, "model.duration");
      $dur := $type($durRaw) in ["string", "number"] ? $number($durRaw) : undefined;
      $type($pps) = "number" and $type($dur) = "number"
        ? {"type":"usd","usd": $pps * $dur}
        : undefined
    )
    """,
)

V25_A2V_PRICE_BADGE = IO.PriceBadge(
    depends_on=IO.PriceBadgeDepends(widgets=["model"]),
    expr="""
    (
      $rates := {"ltx-2.5 (fast)":0.1859, "ltx-2.5 (pro)":0.2431};
      $model := $lookup(widgets, "model");
      $rate := $type($model) = "string" ? $lookup($rates, $model) : undefined;
      $type($rate) = "number"
        ? {"type":"usd","usd": $rate, "format":{"suffix":"/second"}}
        : undefined
    )
    """,
)


def _v25_generation_inputs(
    durations: list[str], resolutions: list[str], fps_options: list[str], tooltip: str | None
) -> list:
    return [
        IO.Combo.Input(
            "duration",
            options=durations,
            default="8",
            tooltip=tooltip,
        ),
        IO.Combo.Input(
            "resolution",
            options=resolutions,
            default="1920x1080",
        ),
        IO.Combo.Input("fps", options=fps_options, default="25"),
        IO.Boolean.Input(
            "generate_audio",
            default=True,
            tooltip="When true, the generated video will include AI-generated audio matching the scene.",
            advanced=True,
        ),
    ]


def _v25_model_combo() -> IO.DynamicCombo.Input:
    return IO.DynamicCombo.Input(
        "model",
        options=[
            IO.DynamicCombo.Option(
                "LTX-2.5 (Fast)",
                _v25_generation_inputs(
                    ["2", "3", "4", "5", "6", "8", "10", "12", "14", "16", "18", "20"],
                    [
                        "1280x720",
                        "720x1280",
                        "1920x1080",
                        "1080x1920",
                        "2560x1440",
                        "1440x2560",
                        "3840x2160",
                        "2160x3840",
                    ],
                    ["24", "25", "48", "50"],
                    "Video duration in seconds. Durations over 10s require a 720p/1080p resolution and 24/25 FPS.",
                ),
            ),
            IO.DynamicCombo.Option(
                "LTX-2.5 (Pro)",
                _v25_generation_inputs(
                    ["2", "3", "4", "5", "6", "8", "10"],
                    ["1280x720", "720x1280", "1920x1080", "1080x1920"],
                    ["24", "25", "50"],
                    "Video duration in seconds.",
                ),
            ),
        ],
    )


def _v25_seed_input() -> IO.Int.Input:
    return IO.Int.Input(
        "seed",
        default=42,
        min=0,
        max=0xFFFFFFFF,
        control_after_generate=True,
        tooltip="Seed to determine if node should re-run; "
        "actual results are nondeterministic regardless of seed.",
    )


def _v25_validate_settings(model: dict) -> None:
    if int(model["duration"]) > 10 and (
        int(model["fps"]) > 25 or model["resolution"] in ("2560x1440", "1440x2560", "3840x2160", "2160x3840")
    ):
        raise ValueError("Durations over 10s require a 720p or 1080p resolution and 24/25 FPS.")


class Ltx25TextToVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="LtxApi25TextToVideo",
            display_name="LTX 2.5 Text To Video",
            category="partner/video/LTXV",
            description="Professional-quality videos with customizable duration and resolution.",
            inputs=[
                _v25_model_combo(),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                ),
                _v25_seed_input(),
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
            price_badge=V25_PRICE_BADGE,
        )

    @classmethod
    async def execute(
        cls,
        model: dict,
        prompt: str,
        seed: int = 42,
    ) -> IO.NodeOutput:
        validate_string(prompt, min_length=1, max_length=10000)
        _v25_validate_settings(model)
        return await _v25_submit_and_poll(
            cls,
            "text-to-video",
            ExecuteTaskRequest(
                prompt=prompt,
                model=V25_MODELS_MAP[model["model"]],
                duration=int(model["duration"]),
                resolution=model["resolution"],
                fps=int(model["fps"]),
                generate_audio=model["generate_audio"],
            ),
        )


class Ltx25ImageToVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="LtxApi25ImageToVideo",
            display_name="LTX 2.5 Image To Video",
            category="partner/video/LTXV",
            description="Professional-quality videos with customizable duration and resolution based on start image.",
            inputs=[
                IO.Image.Input("image", tooltip="First frame to be used for the video."),
                _v25_model_combo(),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                ),
                _v25_seed_input(),
                IO.Image.Input(
                    "last_frame",
                    optional=True,
                    tooltip="Last frame to be used for the video.",
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
            price_badge=V25_PRICE_BADGE,
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        model: dict,
        prompt: str,
        seed: int = 42,
        last_frame: Input.Image | None = None,
    ) -> IO.NodeOutput:
        validate_string(prompt, min_length=1, max_length=10000)
        _v25_validate_settings(model)
        if get_number_of_images(image) != 1:
            raise ValueError("Currently only one input image is supported.")
        last_frame_uri = None
        if last_frame is not None:
            if get_number_of_images(last_frame) != 1:
                raise ValueError("Currently only one last frame image is supported.")
            last_frame_uri = (await upload_images_to_comfyapi(cls, last_frame, max_images=1, mime_type="image/png"))[0]
        return await _v25_submit_and_poll(
            cls,
            "image-to-video",
            ExecuteTaskRequest(
                image_uri=(await upload_images_to_comfyapi(cls, image, max_images=1, mime_type="image/png"))[0],
                last_frame_uri=last_frame_uri,
                prompt=prompt,
                model=V25_MODELS_MAP[model["model"]],
                duration=int(model["duration"]),
                resolution=model["resolution"],
                fps=int(model["fps"]),
                generate_audio=model["generate_audio"],
            ),
        )


class Ltx25AudioToVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="LtxApi25AudioToVideo",
            display_name="LTX 2.5 Audio To Video",
            category="partner/video/LTXV",
            description="Generate a video driven by an audio track, with an optional first frame image.",
            inputs=[
                IO.Audio.Input(
                    "audio",
                    tooltip="Audio track driving the video. Its length (2-20 seconds) sets the video duration.",
                ),
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "LTX-2.5 (Fast)",
                            [IO.Combo.Input("resolution", options=["1920x1080", "1080x1920"])],
                        ),
                        IO.DynamicCombo.Option(
                            "LTX-2.5 (Pro)",
                            [IO.Combo.Input("resolution", options=["1920x1080", "1080x1920"])],
                        ),
                    ],
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                ),
                _v25_seed_input(),
                IO.Image.Input(
                    "image",
                    optional=True,
                    tooltip="Optional first frame to be used for the video.",
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
            price_badge=V25_A2V_PRICE_BADGE,
        )

    @classmethod
    async def execute(
        cls,
        audio: Input.Audio,
        model: dict,
        prompt: str,
        seed: int = 42,
        image: Input.Image | None = None,
    ) -> IO.NodeOutput:
        validate_string(prompt, min_length=1, max_length=10000)
        audio_duration = audio["waveform"].shape[-1] / audio["sample_rate"]
        if not 2 <= audio_duration <= 20:
            raise ValueError(f"Audio duration must be between 2 and 20 seconds, got {audio_duration:.1f}s.")
        image_uri = None
        if image is not None:
            if get_number_of_images(image) != 1:
                raise ValueError("Currently only one input image is supported.")
            image_uri = (await upload_images_to_comfyapi(cls, image, max_images=1, mime_type="image/png"))[0]
        return await _v25_submit_and_poll(
            cls,
            "audio-to-video",
            AudioToVideoRequest(
                prompt=prompt,
                model=V25_MODELS_MAP[model["model"]],
                resolution=model["resolution"],
                audio_uri=await upload_audio_to_comfyapi(cls, audio),
                image_uri=image_uri,
            ),
        )


class LtxvApiExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            Ltx25TextToVideoNode,
            Ltx25ImageToVideoNode,
            Ltx25AudioToVideoNode,
        ]


async def comfy_entrypoint() -> LtxvApiExtension:
    return LtxvApiExtension()
