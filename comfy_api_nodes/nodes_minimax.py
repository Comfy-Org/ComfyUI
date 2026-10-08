from typing import Optional

import torch
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.minimax import (
    Hailuo03AudioContent,
    Hailuo03AudioContentUrl,
    Hailuo03ContextIRRequest,
    Hailuo03ImageContent,
    Hailuo03ImageContentUrl,
    Hailuo03MaxTaskCreationResponse,
    Hailuo03MaxTaskStatusResponse,
    Hailuo03MaxVideoRequest,
    Hailuo03MaxVideoResult,
    Hailuo03RegenerationRequest,
    Hailuo03TaskCreationRequest,
    Hailuo03TaskCreationResponse,
    Hailuo03TaskQueryResponse,
    Hailuo03TextContent,
    Hailuo03VideoContent,
    Hailuo03VideoContentUrl,
    MinimaxFileRetrieveResponse,
    MiniMaxModel,
    MinimaxTaskResultResponse,
    MinimaxVideoGenerationRequest,
    MinimaxVideoGenerationResponse,
    SubjectReferenceItem,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_video_output,
    downscale_image_tensor_by_max_sides,
    poll_op,
    sync_op,
    upload_audio_to_comfyapi,
    upload_images_to_comfyapi,
    upload_video_to_comfyapi,
    validate_image_aspect_ratio,
    validate_image_dimensions,
    validate_string,
)

I2V_AVERAGE_DURATION = 114
T2V_AVERAGE_DURATION = 234


async def _generate_mm_video(
    cls: type[IO.ComfyNode],
    *,
    prompt_text: str,
    seed: int,
    model: str,
    image: Optional[torch.Tensor] = None,  # used for ImageToVideo
    subject: Optional[torch.Tensor] = None,  # used for SubjectToVideo
    average_duration: Optional[int] = None,
) -> IO.NodeOutput:
    if image is None:
        validate_string(prompt_text, field_name="prompt_text")
    image_url = None
    if image is not None:
        image_url = (await upload_images_to_comfyapi(cls, image, max_images=1))[0]

    # TODO: figure out how to deal with subject properly, API returns invalid params when using S2V-01 model
    subject_reference = None
    if subject is not None:
        subject_url = (await upload_images_to_comfyapi(cls, subject, max_images=1))[0]
        subject_reference = [SubjectReferenceItem(image=subject_url)]

    response = await sync_op(
        cls,
        ApiEndpoint(path="/proxy/minimax/video_generation", method="POST"),
        response_model=MinimaxVideoGenerationResponse,
        data=MinimaxVideoGenerationRequest(
            model=MiniMaxModel(model),
            prompt=prompt_text,
            callback_url=None,
            first_frame_image=image_url,
            subject_reference=subject_reference,
            prompt_optimizer=None,
        ),
    )

    task_id = response.task_id
    if not task_id:
        raise Exception(f"MiniMax generation failed: {response.base_resp}")

    task_result = await poll_op(
        cls,
        ApiEndpoint(path="/proxy/minimax/query/video_generation", query_params={"task_id": task_id}),
        response_model=MinimaxTaskResultResponse,
        status_extractor=lambda x: x.status.value,
        estimated_duration=average_duration,
    )

    file_id = task_result.file_id
    if file_id is None:
        raise Exception("Request was not successful. Missing file ID.")
    file_result = await sync_op(
        cls,
        ApiEndpoint(path="/proxy/minimax/files/retrieve", query_params={"file_id": int(file_id)}),
        response_model=MinimaxFileRetrieveResponse,
    )

    file_url = file_result.file.download_url
    if file_url is None:
        raise Exception(f"No video was found in the response. Full response: {file_result.model_dump()}")
    if file_result.file.backup_download_url:
        try:
            return IO.NodeOutput(await download_url_to_video_output(file_url, timeout=10, max_retries=2))
        except Exception:  # if we have a second URL to retrieve the result, try again using that one
            return IO.NodeOutput(
                await download_url_to_video_output(file_result.file.backup_download_url, max_retries=3)
            )
    return IO.NodeOutput(await download_url_to_video_output(file_url))


class MinimaxTextToVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="MinimaxTextToVideoNode",
            display_name="MiniMax Text to Video",
            category="partner/video/MiniMax",
            description="Generates videos synchronously based on a prompt, and optional parameters.",
            inputs=[
                IO.String.Input(
                    "prompt_text",
                    multiline=True,
                    default="",
                    tooltip="Text prompt to guide the video generation",
                ),
                IO.Combo.Input(
                    "model",
                    options=["T2V-01", "T2V-01-Director"],
                    default="T2V-01",
                    tooltip="Model to use for video generation",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    step=1,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
                    optional=True,
                ),
            ],
            outputs=[IO.Video.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.43}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt_text: str,
        model: str = "T2V-01",
        seed: int = 0,
    ) -> IO.NodeOutput:
        return await _generate_mm_video(
            cls,
            prompt_text=prompt_text,
            seed=seed,
            model=model,
            image=None,
            subject=None,
            average_duration=T2V_AVERAGE_DURATION,
        )


class MinimaxImageToVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="MinimaxImageToVideoNode",
            display_name="MiniMax Image to Video",
            category="partner/video/MiniMax",
            description="Generates videos synchronously based on an image and prompt, and optional parameters.",
            inputs=[
                IO.Image.Input(
                    "image",
                    tooltip="Image to use as first frame of video generation",
                ),
                IO.String.Input(
                    "prompt_text",
                    multiline=True,
                    default="",
                    tooltip="Text prompt to guide the video generation",
                ),
                IO.Combo.Input(
                    "model",
                    options=["I2V-01-Director", "I2V-01", "I2V-01-live"],
                    default="I2V-01",
                    tooltip="Model to use for video generation",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    step=1,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
                    optional=True,
                ),
            ],
            outputs=[IO.Video.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.43}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        prompt_text: str,
        model: str = "I2V-01",
        seed: int = 0,
    ) -> IO.NodeOutput:
        return await _generate_mm_video(
            cls,
            prompt_text=prompt_text,
            seed=seed,
            model=model,
            image=image,
            subject=None,
            average_duration=I2V_AVERAGE_DURATION,
        )


class MinimaxSubjectToVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="MinimaxSubjectToVideoNode",
            display_name="MiniMax Subject to Video",
            category="partner/video/MiniMax",
            description="Generates videos synchronously based on an image and prompt, and optional parameters.",
            inputs=[
                IO.Image.Input(
                    "subject",
                    tooltip="Image of subject to reference for video generation",
                ),
                IO.String.Input(
                    "prompt_text",
                    multiline=True,
                    default="",
                    tooltip="Text prompt to guide the video generation",
                ),
                IO.Combo.Input(
                    "model",
                    options=["S2V-01"],
                    default="S2V-01",
                    tooltip="Model to use for video generation",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    step=1,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
                    optional=True,
                ),
            ],
            outputs=[IO.Video.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        subject: torch.Tensor,
        prompt_text: str,
        model: str = "S2V-01",
        seed: int = 0,
    ) -> IO.NodeOutput:
        return await _generate_mm_video(
            cls,
            prompt_text=prompt_text,
            seed=seed,
            model=model,
            image=None,
            subject=subject,
            average_duration=T2V_AVERAGE_DURATION,
        )


class MinimaxHailuoVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="MinimaxHailuoVideoNode",
            display_name="MiniMax Hailuo 02 Video",
            category="partner/video/MiniMax",
            description="Generates videos from prompt, with optional start frame using the MiniMax Hailuo-02 model.",
            inputs=[
                IO.String.Input(
                    "prompt_text",
                    multiline=True,
                    default="",
                    tooltip="Text prompt to guide the video generation.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    step=1,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
                    optional=True,
                ),
                IO.Image.Input(
                    "first_frame_image",
                    tooltip="Optional image to use as the first frame to generate a video.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "prompt_optimizer",
                    default=True,
                    tooltip="Optimize prompt to improve generation quality when needed.",
                    optional=True,
                ),
                IO.Combo.Input(
                    "duration",
                    options=[6, 10],
                    default=6,
                    tooltip="The length of the output video in seconds.",
                    optional=True,
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["768P", "1080P"],
                    default="768P",
                    tooltip="The dimensions of the video display. 1080p is 1920x1080, 768p is 1366x768.",
                    optional=True,
                ),
            ],
            outputs=[IO.Video.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["resolution", "duration"]),
                expr="""
                (
                  $prices := {
                    "768p": {"6": 0.28, "10": 0.56},
                    "1080p": {"6": 0.49}
                  };
                  $resPrices := $lookup($prices, $lowercase(widgets.resolution));
                  $price := $lookup($resPrices, $string(widgets.duration));
                  {"type":"usd","usd": $price ? $price : 0.43}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt_text: str,
        seed: int = 0,
        first_frame_image: Optional[torch.Tensor] = None,  # used for ImageToVideo
        prompt_optimizer: bool = True,
        duration: int = 6,
        resolution: str = "768P",
        model: str = "MiniMax-Hailuo-02",
    ) -> IO.NodeOutput:
        if first_frame_image is None:
            validate_string(prompt_text, field_name="prompt_text")

        if model == "MiniMax-Hailuo-02" and resolution.upper() == "1080P" and duration != 6:
            raise Exception(
                "When model is MiniMax-Hailuo-02 and resolution is 1080P, duration is limited to 6 seconds."
            )

        # upload image, if passed in
        image_url = None
        if first_frame_image is not None:
            image_url = (await upload_images_to_comfyapi(cls, first_frame_image, max_images=1))[0]

        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/minimax/video_generation", method="POST"),
            response_model=MinimaxVideoGenerationResponse,
            data=MinimaxVideoGenerationRequest(
                model=MiniMaxModel(model),
                prompt=prompt_text,
                callback_url=None,
                first_frame_image=image_url,
                prompt_optimizer=prompt_optimizer,
                duration=duration,
                resolution=resolution,
            ),
        )

        task_id = response.task_id
        if not task_id:
            raise Exception(f"MiniMax generation failed: {response.base_resp}")

        average_duration = 120 if resolution == "768P" else 240
        task_result = await poll_op(
            cls,
            ApiEndpoint(path="/proxy/minimax/query/video_generation", query_params={"task_id": task_id}),
            response_model=MinimaxTaskResultResponse,
            status_extractor=lambda x: x.status.value,
            estimated_duration=average_duration,
        )

        file_id = task_result.file_id
        if file_id is None:
            raise Exception("Request was not successful. Missing file ID.")
        file_result = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/minimax/files/retrieve", query_params={"file_id": int(file_id)}),
            response_model=MinimaxFileRetrieveResponse,
        )

        file_url = file_result.file.download_url
        if file_url is None:
            raise Exception(f"No video was found in the response. Full response: {file_result.model_dump()}")

        if file_result.file.backup_download_url:
            try:
                return IO.NodeOutput(await download_url_to_video_output(file_url, timeout=10, max_retries=2))
            except Exception:  # if we have a second URL to retrieve the result, try again using that one
                return IO.NodeOutput(
                    await download_url_to_video_output(file_result.file.backup_download_url, max_retries=3)
                )
        return IO.NodeOutput(await download_url_to_video_output(file_url))


HAILUO_03_CREATE_ENDPOINT = "/proxy/minimax/v2/video_generation"
HAILUO_03_QUERY_ENDPOINT = "/proxy/minimax/v2/query/video_generation"  # + /{task_id}
HAILUO_03_MODELS = {"MiniMax H3": "MiniMax-H3"}
HAILUO_03_FAILED_STATUSES = ["failed", "cancelled", "expired"]

HAILUO_03_CONTEXT_IR_ENDPOINT = "/proxy/minimax/v2/h3_context_ir"
HAILUO_03_REGENERATION_ENDPOINT = "/proxy/minimax/v2/video_regeneration"

HAILUO_03_MAX_MODEL = "MiniMax H3 Max"
HAILUO_03_MAX_TURBO_MODEL = "MiniMax H3 Max Turbo"
HAILUO_03_MAX_ENDPOINTS = {
    HAILUO_03_MAX_MODEL: "/proxy/fal/minimax/h3-max",
    HAILUO_03_MAX_TURBO_MODEL: "/proxy/fal/minimax/h3-max-turbo",
}
HAILUO_03_MAX_PROMPT_MAX_LENGTH = 50000
HAILUO_03_MAX_REFERENCE_IMAGE_MAX_SIDES = {"high": (5120, 2048), "standard": (2048, 1024)}


def _hailuo03_model_inputs(include_ratio: bool = True, allow_adaptive: bool = True):
    inputs = [
        IO.String.Input(
            "prompt",
            multiline=True,
            default="",
            tooltip="Text prompt for video generation.",
        ),
        IO.Combo.Input(
            "resolution",
            options=["768P", "2K"],
            tooltip="Resolution of the output video.",
        ),
    ]
    if include_ratio:
        ratio_options = ["16:9", "4:3", "1:1", "3:4", "9:16", "21:9"]
        if allow_adaptive:
            ratio_options.insert(0, "adaptive")
        inputs.append(
            IO.Combo.Input(
                "ratio",
                options=ratio_options,
                default=ratio_options[0],
                tooltip="Aspect ratio of the output video.",
            )
        )
    inputs.append(
        IO.Int.Input(
            "duration",
            default=5,
            min=4,
            max=15,
            step=1,
            tooltip="Duration of the output video in seconds (4-15).",
            display_mode=IO.NumberDisplay.slider,
        )
    )
    return inputs


async def _hailuo03_run_task(
    cls: type[IO.ComfyNode],
    *,
    model_id: str,
    content: list,
    resolution: str,
    duration: int,
    ratio: str | None,
    seed: int,
    watermark: bool,
) -> IO.NodeOutput:
    response = await sync_op(
        cls,
        ApiEndpoint(path=HAILUO_03_CREATE_ENDPOINT, method="POST"),
        response_model=Hailuo03TaskCreationResponse,
        data=Hailuo03TaskCreationRequest(
            model=model_id,
            content=content,
            resolution=resolution,
            duration=duration,
            ratio=ratio,
            seed=seed,
            aigc_watermark=watermark,
        ),
    )
    task_result = await poll_op(
        cls,
        ApiEndpoint(path=f"{HAILUO_03_QUERY_ENDPOINT}/{response.task_id}"),
        response_model=Hailuo03TaskQueryResponse,
        status_extractor=lambda r: r.task.status,
        failed_statuses=HAILUO_03_FAILED_STATUSES,
        poll_interval=15,
    )
    video_url = task_result.task.content.url if task_result.task.content else None
    if not video_url:
        raise Exception(f"No video URL in the response: {task_result.model_dump()}")
    return IO.NodeOutput(await download_url_to_video_output(video_url))


def _hailuo03_max_model_inputs(include_ratio: bool = True, allow_adaptive: bool = False):
    inputs = [
        IO.String.Input(
            "prompt",
            multiline=True,
            default="",
            tooltip="Text prompt for video generation.",
        ),
        IO.Combo.Input(
            "resolution",
            options=["480P", "768P"],
            default="768P",
            tooltip="Resolution of the output video.",
        ),
    ]
    if include_ratio:
        ratio_options = ["16:9", "4:3", "1:1", "3:4", "9:16", "21:9"]
        if allow_adaptive:
            ratio_options.insert(0, "adaptive")
        inputs.append(
            IO.Combo.Input(
                "ratio",
                options=ratio_options,
                default=ratio_options[0],
                tooltip="Aspect ratio of the output video.",
            )
        )
    inputs.extend(
        [
            IO.Int.Input(
                "duration",
                default=5,
                min=5,
                max=15,
                step=1,
                tooltip="Duration of the output video in seconds (5-15).",
                display_mode=IO.NumberDisplay.slider,
            ),
            IO.Combo.Input(
                "prompt_expansion_mode",
                options=["balanced", "quality"],
                default="balanced",
                tooltip="How much effort is spent rewriting the prompt before generation.",
            ),
        ]
    )
    return inputs


async def _hailuo03_max_run_task(
    cls: type[IO.ComfyNode],
    *,
    model: str,
    endpoint: str,
    request: Hailuo03MaxVideoRequest,
) -> IO.NodeOutput:
    base_endpoint = HAILUO_03_MAX_ENDPOINTS[model]
    submit = await sync_op(
        cls,
        ApiEndpoint(path=f"{base_endpoint}/{endpoint}", method="POST"),
        response_model=Hailuo03MaxTaskCreationResponse,
        data=request,
    )
    await poll_op(
        cls,
        ApiEndpoint(path=f"{base_endpoint}/requests/{submit.request_id}/status"),
        response_model=Hailuo03MaxTaskStatusResponse,
        status_extractor=lambda r: r.status,
        completed_statuses=["COMPLETED"],
        queued_statuses=["IN_QUEUE"],
        poll_interval=5,
    )
    result = await sync_op(
        cls,
        ApiEndpoint(path=f"{base_endpoint}/requests/{submit.request_id}"),
        response_model=Hailuo03MaxVideoResult,
    )
    return IO.NodeOutput(await download_url_to_video_output(result.video.url))


class MinimaxHailuo03TextToVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MinimaxHailuo03TextToVideoNode",
            display_name="MiniMax H3 Text to Video",
            category="partner/video/MiniMax",
            description="Generate video from a text prompt using the MiniMax H3 models.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option("MiniMax H3", _hailuo03_model_inputs(allow_adaptive=False)),
                        IO.DynamicCombo.Option(HAILUO_03_MAX_MODEL, _hailuo03_max_model_inputs()),
                        IO.DynamicCombo.Option(HAILUO_03_MAX_TURBO_MODEL, _hailuo03_max_model_inputs()),
                    ],
                    tooltip="Model to use for video generation.",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=4294967295,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Random seed. The same request with the same seed gives similar, "
                    "but not guaranteed identical, results.",
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip="Whether to add an AIGC watermark to the video.",
                    advanced=True,
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
                depends_on=IO.PriceBadgeDepends(widgets=["model", "model.resolution", "model.duration"]),
                expr="""
                (
                  $dur := $lookup(widgets, "model.duration");
                  $res := $lookup(widgets, "model.resolution");
                  $model := $lookup(widgets, "model");
                  $rate := $model = "minimax h3 max turbo"
                    ? ($res = "480p" ? 0.03575 : 0.0572)
                    : $model = "minimax h3 max"
                      ? ($res = "480p" ? 0.0715 : 0.1144)
                      : ($res = "768p" ? 0.1287 : 0.1859);
                  {"type": "usd", "usd": $dur * $rate}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: dict,
        seed: int,
        watermark: bool,
    ) -> IO.NodeOutput:
        validate_string(model["prompt"], strip_whitespace=True, min_length=1)
        if model["model"] in HAILUO_03_MAX_ENDPOINTS:
            if watermark:
                raise ValueError("Watermark is only supported by MiniMax H3.")
            validate_string(model["prompt"], strip_whitespace=False, max_length=HAILUO_03_MAX_PROMPT_MAX_LENGTH)
            return await _hailuo03_max_run_task(
                cls,
                model=model["model"],
                endpoint="text-to-video",
                request=Hailuo03MaxVideoRequest(
                    prompt=model["prompt"],
                    duration=model["duration"],
                    resolution=model["resolution"],
                    prompt_expansion_mode=model["prompt_expansion_mode"],
                    seed=seed,
                    aspect_ratio=model["ratio"],
                ),
            )
        return await _hailuo03_run_task(
            cls,
            model_id=HAILUO_03_MODELS[model["model"]],
            content=[Hailuo03TextContent(text=model["prompt"])],
            resolution=model["resolution"],
            duration=model["duration"],
            ratio=model["ratio"],
            seed=seed,
            watermark=watermark,
        )


class MinimaxHailuo03FirstLastFrameNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MinimaxHailuo03FirstLastFrameNode",
            display_name="MiniMax H3 First-Last-Frame to Video",
            category="partner/video/MiniMax",
            description="Generate video from a first frame image and an optional last frame image "
            "using the MiniMax H3 models. The aspect ratio of the video follows the supplied images.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option("MiniMax H3", _hailuo03_model_inputs(include_ratio=False)),
                        IO.DynamicCombo.Option(HAILUO_03_MAX_MODEL, _hailuo03_max_model_inputs(include_ratio=False)),
                        IO.DynamicCombo.Option(
                            HAILUO_03_MAX_TURBO_MODEL, _hailuo03_max_model_inputs(include_ratio=False)
                        ),
                    ],
                    tooltip="Model to use for video generation.",
                ),
                IO.Image.Input(
                    "first_frame",
                    tooltip="First frame image for the video.",
                ),
                IO.Image.Input(
                    "last_frame",
                    tooltip="Optional last frame image for the video.",
                    optional=True,
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=4294967295,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Random seed. The same request with the same seed gives similar, "
                    "but not guaranteed identical, results.",
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip="Whether to add an AIGC watermark to the video.",
                    advanced=True,
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
                depends_on=IO.PriceBadgeDepends(widgets=["model", "model.resolution", "model.duration"]),
                expr="""
                (
                  $dur := $lookup(widgets, "model.duration");
                  $res := $lookup(widgets, "model.resolution");
                  $model := $lookup(widgets, "model");
                  $rate := $model = "minimax h3 max turbo"
                    ? ($res = "480p" ? 0.03575 : 0.0572)
                    : $model = "minimax h3 max"
                      ? ($res = "480p" ? 0.0715 : 0.1144)
                      : ($res = "768p" ? 0.1287 : 0.1859);
                  {"type": "usd", "usd": $dur * $rate}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: dict,
        first_frame: torch.Tensor,
        seed: int,
        watermark: bool,
        last_frame: torch.Tensor | None = None,
    ) -> IO.NodeOutput:
        validate_string(model["prompt"], strip_whitespace=True, min_length=1)
        for frame in (first_frame, last_frame):
            if frame is not None:
                validate_image_aspect_ratio(frame, (2, 5), (5, 2), strict=False)  # 0.4 to 2.5
                validate_image_dimensions(frame, min_width=256, min_height=256)
        if model["model"] in HAILUO_03_MAX_ENDPOINTS:
            if watermark:
                raise ValueError("Watermark is only supported by MiniMax H3.")
            validate_string(model["prompt"], strip_whitespace=False, max_length=HAILUO_03_MAX_PROMPT_MAX_LENGTH)
            image_url = (
                await upload_images_to_comfyapi(cls, first_frame, max_images=1, wait_label="Uploading first frame")
            )[0]
            end_image_url = None
            if last_frame is not None:
                end_image_url = (
                    await upload_images_to_comfyapi(cls, last_frame, max_images=1, wait_label="Uploading last frame")
                )[0]
            return await _hailuo03_max_run_task(
                cls,
                model=model["model"],
                endpoint="image-to-video",
                request=Hailuo03MaxVideoRequest(
                    prompt=model["prompt"],
                    duration=model["duration"],
                    resolution=model["resolution"],
                    prompt_expansion_mode=model["prompt_expansion_mode"],
                    seed=seed,
                    image_url=image_url,
                    end_image_url=end_image_url,
                ),
            )

        content: list = [
            Hailuo03TextContent(text=model["prompt"]),
            Hailuo03ImageContent(
                image_url=Hailuo03ImageContentUrl(
                    url=(
                        await upload_images_to_comfyapi(
                            cls, first_frame, max_images=1, wait_label="Uploading first frame"
                        )
                    )[0],
                ),
                role="first_frame",
            ),
        ]
        if last_frame is not None:
            content.append(
                Hailuo03ImageContent(
                    image_url=Hailuo03ImageContentUrl(
                        url=(
                            await upload_images_to_comfyapi(
                                cls, last_frame, max_images=1, wait_label="Uploading last frame"
                            )
                        )[0],
                    ),
                    role="last_frame",
                )
            )
        return await _hailuo03_run_task(
            cls,
            model_id=HAILUO_03_MODELS[model["model"]],
            content=content,
            resolution=model["resolution"],
            duration=model["duration"],
            ratio=None,
            seed=seed,
            watermark=watermark,
        )


def _hailuo03_reference_inputs():
    return [
        IO.Autogrow.Input(
            "reference_images",
            template=IO.Autogrow.TemplateNames(
                IO.Image.Input("reference_image"),
                names=[
                    "image_1",
                    "image_2",
                    "image_3",
                    "image_4",
                    "image_5",
                    "image_6",
                    "image_7",
                    "image_8",
                    "image_9",
                ],
                min=0,
            ),
            tooltip="Subject or style reference images, referred to in the prompt "
            "as 'Image 1'..'Image 9' in connection order. Up to 9 images.",
        ),
        IO.Autogrow.Input(
            "reference_videos",
            template=IO.Autogrow.TemplateNames(
                IO.Video.Input("reference_video"),
                names=["video_1", "video_2", "video_3"],
                min=0,
            ),
            tooltip="Motion or scene reference videos, referred to in the prompt "
            "as 'Video 1'..'Video 3' in connection order. Up to 3 videos, "
            "2-15 seconds each, 15 seconds in total.",
        ),
        IO.Autogrow.Input(
            "reference_audios",
            template=IO.Autogrow.TemplateNames(
                IO.Audio.Input("reference_audio"),
                names=["audio_1", "audio_2", "audio_3"],
                min=0,
            ),
            tooltip="Audio references, referred to in the prompt as "
            "'Audio 1'..'Audio 3' in connection order. Up to 3 clips, "
            "2-15 seconds each, 15 seconds in total. Cannot be used without "
            "a reference image or video.",
        ),
    ]


class MinimaxHailuo03ReferenceNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MinimaxHailuo03ReferenceNode",
            display_name="MiniMax H3 Reference to Video",
            category="partner/video/MiniMax",
            description="Generate video conditioned on reference images, videos, and audio using the "
            "MiniMax H3 models. Refer to the references in the prompt by their order: "
            "'Image 1', 'Image 2', 'Video 1', 'Audio 1', and so on.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "MiniMax H3",
                            [*_hailuo03_model_inputs(), *_hailuo03_reference_inputs()],
                        ),
                        IO.DynamicCombo.Option(
                            HAILUO_03_MAX_MODEL,
                            [
                                *_hailuo03_max_model_inputs(allow_adaptive=True),
                                IO.Combo.Input(
                                    "reference_detail",
                                    options=["high", "standard"],
                                    default="standard",
                                    tooltip="Detail level at which reference images are sent. 'high' sends them at "
                                    "the largest size the model uses (up to a 2048 pixel short side); "
                                    "'standard' downsizes them to at most 2048x1024 to reduce the reference cost.",
                                ),
                                *_hailuo03_reference_inputs(),
                            ],
                        ),
                    ],
                    tooltip="Model to use for video generation.",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=4294967295,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Random seed. The same request with the same seed gives similar, "
                    "but not guaranteed identical, results.",
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip="Whether to add an AIGC watermark to the video.",
                    advanced=True,
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
                depends_on=IO.PriceBadgeDepends(
                    widgets=["model", "model.resolution", "model.duration", "model.reference_detail"],
                    input_groups=["model.reference_images", "model.reference_videos", "model.reference_audios"],
                ),
                expr="""
                (
                  $dur := $lookup(widgets, "model.duration");
                  $res := $lookup(widgets, "model.resolution");
                  $imgsRaw := $lookup(inputGroups, "model.reference_images");
                  $imgs := $imgsRaw ? $imgsRaw : 0;
                  $vidsRaw := $lookup(inputGroups, "model.reference_videos");
                  $vids := $vidsRaw ? $vidsRaw : 0;
                  $audsRaw := $lookup(inputGroups, "model.reference_audios");
                  $auds := $audsRaw ? $audsRaw : 0;
                  $lookup(widgets, "model") = "minimax h3 max"
                    ? (
                        $unitRate := $res = "480p" ? 1 : 1.6;
                        $frameTokens := $res = "480p" ? 390 : 1008;
                        $base := $dur * $unitRate;
                        $minTokens := $imgs * 576 + $vids * 12 * $frameTokens + $auds * 160;
                        $capFrames := $vids * (7 * $dur + 2);
                        $maxFrames := $capFrames > 265 ? 265 : $capFrames;
                        $imgMax := $lookup(widgets, "model.reference_detail") = "standard" ? 2560 : 10240;
                        $maxTokens := $imgs * $imgMax + $maxFrames * $frameTokens + ($auds > 0 ? 1200 : 0);
                        $minUnits := $base + ($minTokens > 4096 ? ($minTokens - 4096) * 0.0004 : 0);
                        $maxUnits := $base + ($maxTokens > 4096 ? ($maxTokens - 4096) * 0.0004 : 0);
                        $minUnits = $maxUnits
                          ? {"type": "usd", "usd": $base * 0.0715}
                          : {"type": "range_usd", "min_usd": $minUnits * 0.0715, "max_usd": $maxUnits * 0.0715,
                             "format": {"approximate": true}}
                      )
                    : (
                        $rate := $res = "768p" ? 0.1287 : 0.1859;
                        $base := $dur * $rate + ($imgs > 5 ? ($imgs - 5) * 0.0572 : 0);
                        $vids > 0
                          ? {"type": "range_usd", "min_usd": $base + $vids * 2 * $rate,
                             "max_usd": $base + 15 * $rate, "format": {"approximate": true}}
                          : {"type": "usd", "usd": $base}
                      )
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: dict,
        seed: int,
        watermark: bool,
    ) -> IO.NodeOutput:
        validate_string(model["prompt"], strip_whitespace=True, min_length=1)

        reference_images = model.get("reference_images", {})
        reference_videos = model.get("reference_videos", {})
        reference_audios = model.get("reference_audios", {})
        if not reference_images and not reference_videos:
            raise ValueError("At least one reference image or video is required.")

        is_max = model["model"] == HAILUO_03_MAX_MODEL
        min_clip_duration = 2.0 if is_max else 1.8
        max_total_duration = 15.0 if is_max else 15.1
        max_fps = 60.19 if is_max else 60.5

        for image in reference_images.values():
            validate_image_aspect_ratio(image, (2, 5), (5, 2), strict=False)  # 0.4 to 2.5
            validate_image_dimensions(image, min_width=256, min_height=256)

        total_video_duration = 0.0
        for i, video in enumerate(reference_videos.values(), 1):
            try:
                fps = float(video.get_frame_rate())
            except Exception:
                fps = 0.0
            if fps and not (23.9 <= fps <= max_fps):
                raise ValueError(f"Reference video {i} is {fps:.2f} FPS. Supported range is 23.976-60 FPS.")
            try:
                dur = video.get_duration()
            except Exception:
                continue
            if dur < min_clip_duration:
                raise ValueError(f"Reference video {i} is too short: {dur:.1f}s. Minimum duration is 2 seconds.")
            total_video_duration += dur
        if total_video_duration > max_total_duration:
            raise ValueError(
                f"Total reference video duration is {total_video_duration:.2f}s. Maximum is 15 seconds."
            )

        total_audio_duration = 0.0
        for i, audio in enumerate(reference_audios.values(), 1):
            dur = int(audio["waveform"].shape[-1]) / int(audio["sample_rate"])
            if dur < min_clip_duration:
                raise ValueError(f"Reference audio {i} is too short: {dur:.1f}s. Minimum duration is 2 seconds.")
            total_audio_duration += dur
        if total_audio_duration > max_total_duration:
            raise ValueError(
                f"Total reference audio duration is {total_audio_duration:.2f}s. Maximum is 15 seconds."
            )

        if is_max:
            if watermark:
                raise ValueError("Watermark is only supported by MiniMax H3.")
            validate_string(model["prompt"], strip_whitespace=False, max_length=HAILUO_03_MAX_PROMPT_MAX_LENGTH)
            if len(reference_images) + len(reference_videos) + len(reference_audios) > 12:
                raise ValueError("MiniMax H3 Max accepts at most 12 reference files in total.")
            max_long_side, max_short_side = HAILUO_03_MAX_REFERENCE_IMAGE_MAX_SIDES[model["reference_detail"]]
            reference_image_urls = [
                (
                    await upload_images_to_comfyapi(
                        cls,
                        downscale_image_tensor_by_max_sides(
                            image, max_long_side=max_long_side, max_short_side=max_short_side
                        ),
                        max_images=1,
                        total_pixels=None,
                        wait_label=f"Uploading image {i}",
                    )
                )[0]
                for i, image in enumerate(reference_images.values(), 1)
            ]
            reference_video_urls = [
                await upload_video_to_comfyapi(cls, video, wait_label=f"Uploading video {i}")
                for i, video in enumerate(reference_videos.values(), 1)
            ]
            reference_audio_urls = [
                await upload_audio_to_comfyapi(
                    cls,
                    audio,
                    container_format="mp3",
                    codec_name="libmp3lame",
                    mime_type="audio/mpeg",
                )
                for audio in reference_audios.values()
            ]
            return await _hailuo03_max_run_task(
                cls,
                model=HAILUO_03_MAX_MODEL,
                endpoint="reference-to-video",
                request=Hailuo03MaxVideoRequest(
                    prompt=model["prompt"],
                    duration=model["duration"],
                    resolution=model["resolution"],
                    prompt_expansion_mode=model["prompt_expansion_mode"],
                    seed=seed,
                    aspect_ratio=model["ratio"],
                    reference_image_urls=reference_image_urls or None,
                    reference_video_urls=reference_video_urls or None,
                    reference_audio_urls=reference_audio_urls or None,
                ),
            )

        content: list = [Hailuo03TextContent(text=model["prompt"])]
        for i, image in enumerate(reference_images.values(), 1):
            content.append(
                Hailuo03ImageContent(
                    image_url=Hailuo03ImageContentUrl(
                        url=(
                            await upload_images_to_comfyapi(
                                cls, image, max_images=1, wait_label=f"Uploading image {i}"
                            )
                        )[0],
                    ),
                    role="reference_image",
                )
            )
        for i, video in enumerate(reference_videos.values(), 1):
            content.append(
                Hailuo03VideoContent(
                    video_url=Hailuo03VideoContentUrl(
                        url=await upload_video_to_comfyapi(cls, video, wait_label=f"Uploading video {i}"),
                    ),
                )
            )
        for audio in reference_audios.values():
            content.append(
                Hailuo03AudioContent(
                    audio_url=Hailuo03AudioContentUrl(
                        url=await upload_audio_to_comfyapi(
                            cls,
                            audio,
                            container_format="mp3",
                            codec_name="libmp3lame",
                            mime_type="audio/mpeg",
                        ),
                    ),
                )
            )
        return await _hailuo03_run_task(
            cls,
            model_id=HAILUO_03_MODELS[model["model"]],
            content=content,
            resolution=model["resolution"],
            duration=model["duration"],
            ratio=model["ratio"],
            seed=seed,
            watermark=watermark,
        )


class MinimaxHailuo03ContextIRNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MinimaxHailuo03ContextIRNode",
            display_name="MiniMax H3 Context IR (Prompt Enhancer)",
            category="partner/video/MiniMax",
            description="Analyze text and media context with MiniMax H3 Context IR and produce an enhanced, "
            "structured video prompt. Feed the output into the prompt of a MiniMax H3 video node and attach "
            "the same media there in the same order, because the enhanced prompt refers to the attached "
            "media by position.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "MiniMax H3",
                            [
                                IO.String.Input(
                                    "prompt",
                                    multiline=True,
                                    default="",
                                    tooltip="Description of the video you intend to generate.",
                                ),
                                IO.Int.Input(
                                    "duration",
                                    default=5,
                                    min=4,
                                    max=15,
                                    step=1,
                                    tooltip="Duration of the video you intend to generate, in seconds (4-15).",
                                    display_mode=IO.NumberDisplay.slider,
                                ),
                                IO.Combo.Input(
                                    "ratio",
                                    options=["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"],
                                    default="adaptive",
                                    tooltip="Aspect ratio of the video you intend to generate. 'adaptive' "
                                    "requires at least one image, video, or audio input.",
                                ),
                                IO.Autogrow.Input(
                                    "reference_images",
                                    template=IO.Autogrow.TemplateNames(
                                        IO.Image.Input("reference_image"),
                                        names=[
                                            "image_1",
                                            "image_2",
                                            "image_3",
                                            "image_4",
                                            "image_5",
                                            "image_6",
                                            "image_7",
                                            "image_8",
                                            "image_9",
                                        ],
                                        min=0,
                                    ),
                                    tooltip="Subject or style reference images, referred to in the prompt "
                                    "as 'Image 1'..'Image 9' in connection order. Up to 9 images.",
                                ),
                                IO.Autogrow.Input(
                                    "reference_videos",
                                    template=IO.Autogrow.TemplateNames(
                                        IO.Video.Input("reference_video"),
                                        names=["video_1", "video_2", "video_3"],
                                        min=0,
                                    ),
                                    tooltip="Motion or scene reference videos, referred to in the prompt "
                                    "as 'Video 1'..'Video 3' in connection order. Up to 3 videos, "
                                    "2-15 seconds each, 15 seconds in total.",
                                ),
                                IO.Autogrow.Input(
                                    "reference_audios",
                                    template=IO.Autogrow.TemplateNames(
                                        IO.Audio.Input("reference_audio"),
                                        names=["audio_1", "audio_2", "audio_3"],
                                        min=0,
                                    ),
                                    tooltip="Audio references, referred to in the prompt as "
                                    "'Audio 1'..'Audio 3' in connection order. Up to 3 clips, "
                                    "2-15 seconds each, 15 seconds in total. Cannot be used without "
                                    "a reference image or video.",
                                ),
                            ],
                        )
                    ],
                    tooltip="Model to use for prompt enhancement.",
                ),
                IO.Image.Input(
                    "first_frame",
                    tooltip="First frame of the video you intend to generate. Cannot be combined with "
                    "reference media.",
                    optional=True,
                ),
                IO.Image.Input(
                    "last_frame",
                    tooltip="Last frame of the video you intend to generate. Cannot be combined with "
                    "reference media.",
                    optional=True,
                ),
            ],
            outputs=[
                IO.String.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(
                    inputs=["first_frame", "last_frame"],
                    input_groups=["model.reference_images", "model.reference_videos", "model.reference_audios"],
                ),
                expr="""
                (
                  $imgsRaw := $lookup(inputGroups, "model.reference_images");
                  $imgs := $imgsRaw ? $imgsRaw : 0;
                  $vidsRaw := $lookup(inputGroups, "model.reference_videos");
                  $vids := $vidsRaw ? $vidsRaw : 0;
                  $audsRaw := $lookup(inputGroups, "model.reference_audios");
                  $auds := $audsRaw ? $audsRaw : 0;
                  $frames := (inputs.first_frame.connected ? 1 : 0) + (inputs.last_frame.connected ? 1 : 0);
                  ($imgs + $vids + $auds) > 0
                    ? {"type": "range_usd", "min_usd": 0.06, "max_usd": 0.11, "format": {"approximate": true}}
                    : $frames > 0
                      ? {"type": "usd", "usd": 0.05, "format": {"approximate": true}}
                      : {"type": "usd", "usd": 0.02, "format": {"approximate": true}}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: dict,
        first_frame: torch.Tensor | None = None,
        last_frame: torch.Tensor | None = None,
    ) -> IO.NodeOutput:
        validate_string(model["prompt"], strip_whitespace=True, min_length=1)

        reference_images = {k: v for k, v in (model.get("reference_images") or {}).items() if v is not None}
        reference_videos = {k: v for k, v in (model.get("reference_videos") or {}).items() if v is not None}
        reference_audios = {k: v for k, v in (model.get("reference_audios") or {}).items() if v is not None}
        has_frames = first_frame is not None or last_frame is not None
        has_references = bool(reference_images) or bool(reference_videos) or bool(reference_audios)
        if has_frames and has_references:
            raise ValueError(
                "First/last frame and reference media are mutually exclusive. Use frames for an "
                "image-to-video prompt, or reference media for a reference-to-video prompt."
            )
        if reference_audios and not reference_images and not reference_videos:
            raise ValueError("Reference audio cannot be used without a reference image or video.")
        if not has_frames and not has_references and model["ratio"] == "adaptive":
            raise ValueError(
                "Ratio 'adaptive' is not supported for text-only requests; select an explicit aspect ratio."
            )

        for frame in (first_frame, last_frame):
            if frame is not None:
                validate_image_aspect_ratio(frame, (2, 5), (5, 2), strict=False)  # 0.4 to 2.5
                validate_image_dimensions(frame, min_width=256, min_height=256)
        for image in reference_images.values():
            validate_image_aspect_ratio(image, (2, 5), (5, 2), strict=False)  # 0.4 to 2.5
            validate_image_dimensions(image, min_width=256, min_height=256)

        total_video_duration = 0.0
        for i, video in enumerate(reference_videos.values(), 1):
            try:
                fps = float(video.get_frame_rate())
            except Exception:
                fps = 0.0
            if fps and not (23.9 <= fps <= 60.5):
                raise ValueError(f"Reference video {i} is {fps:.2f} FPS. Supported range is 23.976-60 FPS.")
            try:
                dur = video.get_duration()
            except Exception:
                continue
            if dur < 1.8:
                raise ValueError(f"Reference video {i} is too short: {dur:.1f}s. Minimum duration is 2 seconds.")
            total_video_duration += dur
        if total_video_duration > 15.1:
            raise ValueError(
                f"Total reference video duration is {total_video_duration:.1f}s. Maximum is 15 seconds."
            )

        total_audio_duration = 0.0
        for i, audio in enumerate(reference_audios.values(), 1):
            dur = int(audio["waveform"].shape[-1]) / int(audio["sample_rate"])
            if dur < 1.8:
                raise ValueError(f"Reference audio {i} is too short: {dur:.1f}s. Minimum duration is 2 seconds.")
            total_audio_duration += dur
        if total_audio_duration > 15.1:
            raise ValueError(
                f"Total reference audio duration is {total_audio_duration:.1f}s. Maximum is 15 seconds."
            )

        content: list = [Hailuo03TextContent(text=model["prompt"])]
        if first_frame is not None:
            content.append(
                Hailuo03ImageContent(
                    image_url=Hailuo03ImageContentUrl(
                        url=(
                            await upload_images_to_comfyapi(
                                cls, first_frame, max_images=1, wait_label="Uploading first frame"
                            )
                        )[0],
                    ),
                    role="first_frame",
                )
            )
        if last_frame is not None:
            content.append(
                Hailuo03ImageContent(
                    image_url=Hailuo03ImageContentUrl(
                        url=(
                            await upload_images_to_comfyapi(
                                cls, last_frame, max_images=1, wait_label="Uploading last frame"
                            )
                        )[0],
                    ),
                    role="last_frame",
                )
            )
        for i, image in enumerate(reference_images.values(), 1):
            content.append(
                Hailuo03ImageContent(
                    image_url=Hailuo03ImageContentUrl(
                        url=(
                            await upload_images_to_comfyapi(
                                cls, image, max_images=1, wait_label=f"Uploading image {i}"
                            )
                        )[0],
                    ),
                    role="reference_image",
                )
            )
        for i, video in enumerate(reference_videos.values(), 1):
            content.append(
                Hailuo03VideoContent(
                    video_url=Hailuo03VideoContentUrl(
                        url=await upload_video_to_comfyapi(cls, video, wait_label=f"Uploading video {i}"),
                    ),
                )
            )
        for audio in reference_audios.values():
            content.append(
                Hailuo03AudioContent(
                    audio_url=Hailuo03AudioContentUrl(
                        url=await upload_audio_to_comfyapi(
                            cls,
                            audio,
                            container_format="mp3",
                            codec_name="libmp3lame",
                            mime_type="audio/mpeg",
                        ),
                    ),
                )
            )

        response = await sync_op(
            cls,
            ApiEndpoint(path=HAILUO_03_CONTEXT_IR_ENDPOINT, method="POST"),
            response_model=Hailuo03TaskCreationResponse,
            data=Hailuo03ContextIRRequest(
                model=HAILUO_03_MODELS[model["model"]],
                content=content,
                duration=model["duration"],
                ratio=None if model["ratio"] == "adaptive" else model["ratio"],
            ),
        )
        task_result = await poll_op(
            cls,
            ApiEndpoint(path=f"{HAILUO_03_QUERY_ENDPOINT}/{response.task_id}"),
            response_model=Hailuo03TaskQueryResponse,
            status_extractor=lambda r: r.task.status,
            failed_statuses=HAILUO_03_FAILED_STATUSES,
            poll_interval=5,
        )
        prompt = task_result.task.content.prompt if task_result.task.content else None
        if not prompt:
            raise Exception(f"No enhanced prompt in the response: {task_result.model_dump()}")
        return IO.NodeOutput(prompt)


class MinimaxHailuo03RegenerateNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MinimaxHailuo03RegenerateNode",
            display_name="MiniMax H3 Regenerate to 2K",
            category="partner/video/MiniMax",
            description="Re-render a MiniMax H3 768P output at 2K resolution. Connect the unmodified 768P "
            "video and the exact prompt used to generate it; if the original generation used first/last "
            "frames or reference media, attach the same inputs.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "MiniMax H3",
                            [
                                IO.String.Input(
                                    "prompt",
                                    multiline=True,
                                    default="",
                                    tooltip="The exact prompt used to generate the source video.",
                                ),
                                IO.Combo.Input(
                                    "resolution",
                                    options=["2K"],
                                    tooltip="Resolution to re-render the source video at.",
                                ),
                                IO.Autogrow.Input(
                                    "reference_images",
                                    template=IO.Autogrow.TemplateNames(
                                        IO.Image.Input("reference_image"),
                                        names=[
                                            "image_1",
                                            "image_2",
                                            "image_3",
                                            "image_4",
                                            "image_5",
                                            "image_6",
                                            "image_7",
                                            "image_8",
                                            "image_9",
                                        ],
                                        min=0,
                                    ),
                                    tooltip="Reference images from the original generation, in the same "
                                    "order. Up to 9 images.",
                                ),
                                IO.Autogrow.Input(
                                    "reference_videos",
                                    template=IO.Autogrow.TemplateNames(
                                        IO.Video.Input("reference_video"),
                                        names=["video_1", "video_2", "video_3"],
                                        min=0,
                                    ),
                                    tooltip="Reference videos from the original generation, in the same "
                                    "order. Up to 3 videos, 2-15 seconds each, 15 seconds in total.",
                                ),
                                IO.Autogrow.Input(
                                    "reference_audios",
                                    template=IO.Autogrow.TemplateNames(
                                        IO.Audio.Input("reference_audio"),
                                        names=["audio_1", "audio_2", "audio_3"],
                                        min=0,
                                    ),
                                    tooltip="Audio references from the original generation, in the same "
                                    "order. Up to 3 clips, 2-15 seconds each, 15 seconds in total. "
                                    "Cannot be used without a reference image or video.",
                                ),
                            ],
                        )
                    ],
                    tooltip="Model to use for video regeneration.",
                ),
                IO.Video.Input(
                    "video",
                    tooltip="The MiniMax H3 768P output video to re-render. Connect the unmodified output "
                    "of a MiniMax H3 video node (24 FPS, 4-15 seconds). 2K outputs cannot be used.",
                ),
                IO.Image.Input(
                    "first_frame",
                    tooltip="First frame image from the original generation, if one was used.",
                    optional=True,
                ),
                IO.Image.Input(
                    "last_frame",
                    tooltip="Last frame image from the original generation, if one was used.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip="Whether to add an AIGC watermark to the video.",
                    advanced=True,
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
                expr="""{"type": "usd", "usd": 0.0715, "format": {"suffix": "/second"}}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: dict,
        video: Input.Video,
        watermark: bool,
        first_frame: torch.Tensor | None = None,
        last_frame: torch.Tensor | None = None,
    ) -> IO.NodeOutput:
        validate_string(model["prompt"], strip_whitespace=True, min_length=1)

        try:
            fps = float(video.get_frame_rate())
        except Exception:
            fps = 0.0
        if fps and not (23.9 <= fps <= 24.1):
            raise ValueError(
                f"The source video is {fps:.2f} FPS. Regeneration accepts unmodified MiniMax H3 768P "
                "outputs, which are 24 FPS."
            )
        try:
            width, height = video.get_dimensions()
        except Exception:
            width = height = 0
        if width and height and (width % 32 or height % 32 or width * height > 1_032_192):
            raise ValueError(
                f"The source video is {width}x{height}. Regeneration accepts MiniMax H3 768P outputs "
                "(width and height divisible by 32, at most 1,032,192 total pixels); 2K outputs cannot "
                "be used as a source."
            )
        try:
            frame_count = video.get_frame_count()
        except Exception:
            frame_count = 0
        if frame_count and (frame_count < 107 or frame_count > 362 or (frame_count - 107) % 17):
            raise ValueError(
                f"The source video has {frame_count} frames. Regeneration accepts unmodified "
                "MiniMax H3 outputs, whose length is 107 to 362 frames in steps of 17 "
                "(4 to 15 seconds at 24 FPS)."
            )

        reference_images = {k: v for k, v in (model.get("reference_images") or {}).items() if v is not None}
        reference_videos = {k: v for k, v in (model.get("reference_videos") or {}).items() if v is not None}
        reference_audios = {k: v for k, v in (model.get("reference_audios") or {}).items() if v is not None}
        if (first_frame is not None or last_frame is not None) and (
            reference_images or reference_videos or reference_audios
        ):
            raise ValueError(
                "First/last frame and reference media are mutually exclusive. Use frames for an "
                "image-to-video prompt, or reference media for a reference-to-video prompt."
            )
        if reference_audios and not reference_images and not reference_videos:
            raise ValueError("Reference audio cannot be used without a reference image or video.")

        for frame in (first_frame, last_frame):
            if frame is not None:
                validate_image_aspect_ratio(frame, (2, 5), (5, 2), strict=False)  # 0.4 to 2.5
                validate_image_dimensions(frame, min_width=256, min_height=256)
        for image in reference_images.values():
            validate_image_aspect_ratio(image, (2, 5), (5, 2), strict=False)  # 0.4 to 2.5
            validate_image_dimensions(image, min_width=256, min_height=256)

        total_video_duration = 0.0
        for i, ref_video in enumerate(reference_videos.values(), 1):
            try:
                ref_fps = float(ref_video.get_frame_rate())
            except Exception:
                ref_fps = 0.0
            if ref_fps and not (23.9 <= ref_fps <= 60.5):
                raise ValueError(f"Reference video {i} is {ref_fps:.2f} FPS. Supported range is 23.976-60 FPS.")
            try:
                dur = ref_video.get_duration()
            except Exception:
                continue
            if dur < 1.8:
                raise ValueError(f"Reference video {i} is too short: {dur:.1f}s. Minimum duration is 2 seconds.")
            total_video_duration += dur
        if total_video_duration > 15.1:
            raise ValueError(
                f"Total reference video duration is {total_video_duration:.1f}s. Maximum is 15 seconds."
            )

        total_audio_duration = 0.0
        for i, audio in enumerate(reference_audios.values(), 1):
            dur = int(audio["waveform"].shape[-1]) / int(audio["sample_rate"])
            if dur < 1.8:
                raise ValueError(f"Reference audio {i} is too short: {dur:.1f}s. Minimum duration is 2 seconds.")
            total_audio_duration += dur
        if total_audio_duration > 15.1:
            raise ValueError(
                f"Total reference audio duration is {total_audio_duration:.1f}s. Maximum is 15 seconds."
            )

        content: list = [
            Hailuo03VideoContent(
                video_url=Hailuo03VideoContentUrl(
                    url=await upload_video_to_comfyapi(cls, video, wait_label="Uploading source video"),
                ),
                role="base_video",
            ),
            Hailuo03TextContent(text=model["prompt"]),
        ]
        if first_frame is not None:
            content.append(
                Hailuo03ImageContent(
                    image_url=Hailuo03ImageContentUrl(
                        url=(
                            await upload_images_to_comfyapi(
                                cls, first_frame, max_images=1, wait_label="Uploading first frame"
                            )
                        )[0],
                    ),
                    role="first_frame",
                )
            )
        if last_frame is not None:
            content.append(
                Hailuo03ImageContent(
                    image_url=Hailuo03ImageContentUrl(
                        url=(
                            await upload_images_to_comfyapi(
                                cls, last_frame, max_images=1, wait_label="Uploading last frame"
                            )
                        )[0],
                    ),
                    role="last_frame",
                )
            )
        for i, image in enumerate(reference_images.values(), 1):
            content.append(
                Hailuo03ImageContent(
                    image_url=Hailuo03ImageContentUrl(
                        url=(
                            await upload_images_to_comfyapi(
                                cls, image, max_images=1, wait_label=f"Uploading image {i}"
                            )
                        )[0],
                    ),
                    role="reference_image",
                )
            )
        for i, ref_video in enumerate(reference_videos.values(), 1):
            content.append(
                Hailuo03VideoContent(
                    video_url=Hailuo03VideoContentUrl(
                        url=await upload_video_to_comfyapi(cls, ref_video, wait_label=f"Uploading video {i}"),
                    ),
                )
            )
        for audio in reference_audios.values():
            content.append(
                Hailuo03AudioContent(
                    audio_url=Hailuo03AudioContentUrl(
                        url=await upload_audio_to_comfyapi(
                            cls,
                            audio,
                            container_format="mp3",
                            codec_name="libmp3lame",
                            mime_type="audio/mpeg",
                        ),
                    ),
                )
            )

        response = await sync_op(
            cls,
            ApiEndpoint(path=HAILUO_03_REGENERATION_ENDPOINT, method="POST"),
            response_model=Hailuo03TaskCreationResponse,
            data=Hailuo03RegenerationRequest(
                model=HAILUO_03_MODELS[model["model"]],
                content=content,
                resolution=model["resolution"],
                aigc_watermark=watermark,
            ),
        )
        task_result = await poll_op(
            cls,
            ApiEndpoint(path=f"{HAILUO_03_QUERY_ENDPOINT}/{response.task_id}"),
            response_model=Hailuo03TaskQueryResponse,
            status_extractor=lambda r: r.task.status,
            failed_statuses=HAILUO_03_FAILED_STATUSES,
            poll_interval=10,
        )
        video_url = task_result.task.content.url if task_result.task.content else None
        if not video_url:
            raise Exception(f"No video URL in the response: {task_result.model_dump()}")
        return IO.NodeOutput(await download_url_to_video_output(video_url))


class MinimaxExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            MinimaxTextToVideoNode,
            MinimaxImageToVideoNode,
            # MinimaxSubjectToVideoNode,
            MinimaxHailuoVideoNode,
            MinimaxHailuo03TextToVideoNode,
            MinimaxHailuo03FirstLastFrameNode,
            MinimaxHailuo03ReferenceNode,
            MinimaxHailuo03ContextIRNode,
            MinimaxHailuo03RegenerateNode,
        ]


async def comfy_entrypoint() -> MinimaxExtension:
    return MinimaxExtension()
