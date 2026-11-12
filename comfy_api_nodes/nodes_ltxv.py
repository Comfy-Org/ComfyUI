
from pydantic import BaseModel, Field
from typing_extensions import override

from comfy_api.v3 import IO, ComfyExtension, Input
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_video_output,
    get_number_of_images,
    poll_op,
    sync_op,
    upload_audio_to_comfyapi,
    upload_images_to_comfyapi,
    upload_video_to_comfyapi,
    validate_string,
    validate_video_dimensions,
    validate_video_duration,
    validate_video_frame_count,
)

V25_MODELS_MAP = {
    "LTX-2.5 (Fast)": "ltx-2-5-fast",
    "LTX-2.5 (Pro)": "ltx-2-5-pro",
}

UPSCALE_MAX_FRAMES = {"1080p": 2401, "4k": 601, "8k": 241}


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


class Ltx25JobStatusResponse(BaseModel):
    id: str = Field(...)
    status: str = Field(...)
    result: dict[str, str] | None = Field(None)


class VideoInputRequest(BaseModel):
    video_uri: str = Field(...)


class VideoUpscaleRequest(BaseModel):
    video_uri: str = Field(...)
    resolution: str = Field(...)


class ReframeAnchor(BaseModel):
    anchor: str = Field(...)


class ReframeRectangle(BaseModel):
    x: float = Field(...)
    y: float = Field(...)
    width: float = Field(...)


class VideoReframeRequest(BaseModel):
    video_uri: str = Field(...)
    resolution: str = Field(...)
    placement: ReframeAnchor | ReframeRectangle | None = Field(None)


async def _v25_submit_and_poll(cls: type[IO.ComfyNode], route: str, data: BaseModel, result_key: str) -> IO.NodeOutput:
    submit = await sync_op(
        cls,
        ApiEndpoint(f"/proxy/ltx/v2/{route}", "POST"),
        response_model=Ltx25SubmitResponse,
        data=data,
    )
    job = await poll_op(
        cls,
        ApiEndpoint(f"/proxy/ltx/v2/{route}/{submit.id}"),
        response_model=Ltx25JobStatusResponse,
        status_extractor=lambda r: r.status,
        max_poll_attempts=1200,
    )
    if not job.result or not job.result.get(result_key):
        raise RuntimeError(f"LTX job {job.id} completed without a {result_key}.")
    return IO.NodeOutput(await download_url_to_video_output(job.result[result_key], cls=cls))


def _validate_video_fits(video: Input.Video, max_long_side: int, max_short_side: int) -> tuple[int, int]:
    width, height = video.get_dimensions()
    if max(width, height) > max_long_side or min(width, height) > max_short_side:
        raise ValueError(
            f"Video must fit within {max_long_side}x{max_short_side} in either orientation, got {width}x{height}."
        )
    return width, height


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
            "video_url",
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
            "video_url",
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
            "video_url",
        )


class LtxVideoAlphaGenNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="LtxApiVideoAlphaGen",
            display_name="LTX Video AlphaGen",
            category="partner/video/LTXV",
            description="Extract a grayscale alpha matte (white is foreground) that matches the input video "
            "frame for frame. Use Get Video Components and Image To Mask to turn it into a mask. Audio is not kept.",
            inputs=[
                IO.Video.Input(
                    "video",
                    tooltip="Up to 10 seconds, at most 3840x2160 in either orientation.",
                ),
                _v25_seed_input(),
            ],
            outputs=[
                IO.Video.Output(display_name="matte"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"range_usd","min_usd":0.143,"max_usd":1.43,"format":{"approximate":true,"suffix":"/second"}}""",
            ),
        )

    @classmethod
    async def execute(cls, video: Input.Video, seed: int = 42) -> IO.NodeOutput:
        validate_video_duration(video, max_duration=10.05)
        _validate_video_fits(video, 3840, 2160)
        return await _v25_submit_and_poll(
            cls,
            "video-alpha-gen",
            VideoInputRequest(video_uri=await upload_video_to_comfyapi(cls, video)),
            "matte_video_url",
        )


class LtxVideoSdrToHdrNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="LtxApiVideoSdrToHdr",
            display_name="LTX Video SDR to HDR",
            category="partner/video/LTXV",
            description="Convert an SDR video into a 10-bit HDR video (BT.2020, HLG) at the same resolution. "
            "Audio is not kept.",
            inputs=[
                IO.Video.Input(
                    "video",
                    tooltip="At most 3840x2160 in either orientation. "
                    "Up to 273 frames at 1080p, 193 frames at 1440p and 65 frames at 4K.",
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
            price_badge=IO.PriceBadge(
                expr="""{"type":"range_usd","min_usd":0.572,"max_usd":2.288,"format":{"approximate":true,"suffix":"/second"}}""",
            ),
        )

    @classmethod
    async def execute(cls, video: Input.Video, seed: int = 42) -> IO.NodeOutput:
        width, height = _validate_video_fits(video, 3840, 2160)
        if width * height <= 1920 * 1080:
            max_frames = 273
        elif width * height <= 2560 * 1440:
            max_frames = 193
        else:
            max_frames = 65
        validate_video_frame_count(video, max_frame_count=max_frames)
        return await _v25_submit_and_poll(
            cls,
            "video-to-video-hdr",
            VideoInputRequest(video_uri=await upload_video_to_comfyapi(cls, video)),
            "video_url",
        )


class LtxVideoReframeNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="LtxApiVideoReframe",
            display_name="LTX Video Reframe",
            category="partner/video/LTXV",
            description="Reframe a video to a new resolution and aspect ratio. "
            "The areas outside the original video are generated to match it.",
            inputs=[
                IO.Video.Input(
                    "video",
                    tooltip="Up to 60 seconds and 1800 frames, each side 64-4096 pixels.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=[
                        "(720p) 720x720 (1:1)",
                        "(720p) 720x900 (4:5)",
                        "(720p) 900x720 (5:4)",
                        "(720p) 720x1280 (9:16)",
                        "(720p) 1280x720 (16:9)",
                        "(1080p) 1080x1080 (1:1)",
                        "(1080p) 1080x1350 (4:5)",
                        "(1080p) 1350x1080 (5:4)",
                        "(1080p) 1080x1920 (9:16)",
                        "(1080p) 1920x1080 (16:9)",
                    ],
                    default="(1080p) 1080x1920 (9:16)",
                ),
                IO.DynamicCombo.Input(
                    "placement",
                    options=[
                        IO.DynamicCombo.Option("center", []),
                        IO.DynamicCombo.Option("top", []),
                        IO.DynamicCombo.Option("bottom", []),
                        IO.DynamicCombo.Option("left", []),
                        IO.DynamicCombo.Option("right", []),
                        IO.DynamicCombo.Option(
                            "custom",
                            [
                                IO.Float.Input(
                                    "x",
                                    default=0.25,
                                    min=0.0,
                                    max=0.99,
                                    step=0.01,
                                    tooltip="Left edge of the video as a fraction of the output width.",
                                ),
                                IO.Float.Input(
                                    "y",
                                    default=0.25,
                                    min=0.0,
                                    max=0.99,
                                    step=0.01,
                                    tooltip="Top edge of the video as a fraction of the output height.",
                                ),
                                IO.Float.Input(
                                    "width",
                                    default=0.5,
                                    min=0.01,
                                    max=1.0,
                                    step=0.01,
                                    tooltip="Width of the video as a fraction of the output width; "
                                    "the height follows the video's aspect ratio. "
                                    "The video must stay inside the frame and can be enlarged at most 2x.",
                                ),
                            ],
                        ),
                    ],
                    tooltip="Where the original video sits in the new frame. "
                    "The video always fills one side of the frame, so an anchor moves it along the other side only.",
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
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["resolution"]),
                expr="""
                {
                  "type":"usd",
                  "usd": $contains(widgets.resolution, "(1080p)") ? 0.286 : 0.143,
                  "format":{"suffix":"/second"}
                }
                """,
            ),
        )

    @classmethod
    async def execute(cls, video: Input.Video, resolution: str, placement: dict, seed: int = 42) -> IO.NodeOutput:
        validate_video_dimensions(video, min_width=64, max_width=4096, min_height=64, max_height=4096)
        validate_video_duration(video, max_duration=60)
        validate_video_frame_count(video, max_frame_count=1800)
        if placement["placement"] == "custom":
            reframe_placement = ReframeRectangle(x=placement["x"], y=placement["y"], width=placement["width"])
        elif placement["placement"] == "center":
            reframe_placement = None
        else:
            reframe_placement = ReframeAnchor(anchor=placement["placement"])
        return await _v25_submit_and_poll(
            cls,
            "video-to-video-reframe",
            VideoReframeRequest(
                video_uri=await upload_video_to_comfyapi(cls, video),
                resolution=resolution.split(" ")[1],
                placement=reframe_placement,
            ),
            "video_url",
        )


class LtxVideoUpscaleNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="LtxApiVideoUpscale",
            display_name="LTX Video Upscale",
            category="partner/video/LTXV",
            description="Upscale a video to 1080p, 4K or 8K, keeping its aspect ratio, frame rate and audio.",
            inputs=[
                IO.Video.Input(
                    "video",
                    tooltip="Short side at least 480 pixels, at most 7680x4320 in either orientation. "
                    "Up to 2401 frames for 1080p, 601 frames for 4k and 241 frames for 8k.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["1080p", "4k", "8k"],
                    default="4k",
                    tooltip="The output is as large as fits in 1920x1080, 3840x2160 or 7680x4320 "
                    "(turned for portrait videos).",
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
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["resolution"]),
                expr="""
                {
                  "type":"usd",
                  "usd": $lookup({"1080p":0.1144,"4k":0.429,"8k":1.716}, widgets.resolution),
                  "format":{"suffix":"/second"}
                }
                """,
            ),
        )

    @classmethod
    async def execute(cls, video: Input.Video, resolution: str, seed: int = 42) -> IO.NodeOutput:
        width, height = _validate_video_fits(video, 7680, 4320)
        if min(width, height) < 480:
            raise ValueError(f"Video short side must be at least 480 pixels, got {width}x{height}.")
        validate_video_frame_count(video, min_frame_count=2, max_frame_count=UPSCALE_MAX_FRAMES[resolution])
        return await _v25_submit_and_poll(
            cls,
            "video-upscale",
            VideoUpscaleRequest(video_uri=await upload_video_to_comfyapi(cls, video), resolution=resolution),
            "video_url",
        )


class LtxvApiExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            Ltx25TextToVideoNode,
            Ltx25ImageToVideoNode,
            Ltx25AudioToVideoNode,
            LtxVideoAlphaGenNode,
            LtxVideoSdrToHdrNode,
            LtxVideoReframeNode,
            LtxVideoUpscaleNode,
        ]


async def comfy_entrypoint() -> LtxvApiExtension:
    return LtxvApiExtension()
