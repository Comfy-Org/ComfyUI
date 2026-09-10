import re

import torch
from typing_extensions import override
from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.pixverse import (
    PixverseTextVideoRequest,
    PixverseImageVideoRequest,
    PixverseTransitionVideoRequest,
    PixverseImageUploadResponse,
    PixverseMediaUploadResponse,
    PixverseVideoResponse,
    PixverseGenerationStatusResponse,
    PixverseAspectRatio,
    PixverseImageReference,
    PixverseQuality,
    PixverseDuration,
    PixverseMotionMode,
    PixverseReferenceType,
    PixverseStatus,
    PixverseV6AspectRatio,
    PixverseV6ExtendVideoRequest,
    PixverseV6FusionVideoRequest,
    PixverseV6ImageVideoRequest,
    PixverseV6Style,
    PixverseV6TextVideoRequest,
    PixverseV6TransitionVideoRequest,
    PixverseVideoReference,
    PixverseIO,
    pixverse_templates,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_video_output,
    poll_op,
    sync_op,
    upload_images_to_comfyapi,
    upload_video_to_comfyapi,
    validate_string,
    validate_video_dimensions,
    validate_video_duration,
)

AVERAGE_DURATION_T2V = 32
AVERAGE_DURATION_I2V = 30
AVERAGE_DURATION_T2T = 52

V6_MAX_PROMPT_LENGTH = 5000
V6_MAX_NEGATIVE_PROMPT_LENGTH = 2048
V6_MIN_DURATION = 1
V6_MAX_DURATION = 15
V6_MAX_SUBJECTS = 8
V6_MAX_BACKGROUNDS = 2
V6_MAX_REFERENCE_IMAGES_OMNI = 10
V6_MAX_REFERENCE_IMAGES_PLAIN = 7
V6_MAX_REFERENCE_VIDEOS = 2
V6_MAX_REFERENCE_VIDEO_SECONDS = 15
V6_EXTEND_SOURCE_SECONDS_LIMIT = 40
V6_MAX_SOURCE_VIDEO_SIDE = 1920

PIXVERSE_STATUS_LABELS = {
    PixverseStatus.successful: "completed",
    PixverseStatus.generating: "generating",
    PixverseStatus.deleted: "deleted",
    PixverseStatus.contents_moderation: "content moderation failed",
    PixverseStatus.failed: "generation failed",
}
PIXVERSE_COMPLETED_STATUSES = ["completed"]
PIXVERSE_FAILED_STATUSES = ["deleted", "content moderation failed", "generation failed"]


def _pixverse_status(response) -> str:
    resp = getattr(response, "Resp", None)
    status = getattr(resp, "status", None)
    return PIXVERSE_STATUS_LABELS.get(status, f"unknown status {status}")


async def upload_video_to_pixverse(cls: type[IO.ComfyNode], video: Input.Video) -> int:
    response_upload = await sync_op(
        cls,
        ApiEndpoint(path="/proxy/pixverse/media/upload", method="POST"),
        response_model=PixverseMediaUploadResponse,
        data={"file_url": await upload_video_to_comfyapi(cls, video)},
        content_type="multipart/form-data",
    )
    if response_upload.Resp is None or response_upload.Resp.media_id is None:
        raise Exception(f"PixVerse video upload request failed: '{response_upload.ErrMsg}'")
    return response_upload.Resp.media_id


async def upload_image_to_pixverse(cls: type[IO.ComfyNode], image: torch.Tensor) -> int:
    image_urls = await upload_images_to_comfyapi(cls, image, max_images=1)
    response_upload = await sync_op(
        cls,
        ApiEndpoint(path="/proxy/pixverse/image/upload", method="POST"),
        response_model=PixverseImageUploadResponse,
        data={"image_url": image_urls[0]},
        content_type="multipart/form-data",
    )
    if response_upload.Resp is None or response_upload.Resp.img_id is None:
        raise Exception(f"PixVerse image upload request failed: '{response_upload.ErrMsg}'")
    return response_upload.Resp.img_id


class PixverseTemplateNode(IO.ComfyNode):
    """
    Select template for PixVerse Video generation.
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="PixverseTemplateNode",
            display_name="PixVerse Template",
            category="partner/video/PixVerse",
            inputs=[
                IO.Combo.Input("template", options=list(pixverse_templates.keys())),
            ],
            outputs=[IO.Custom(PixverseIO.TEMPLATE).Output(display_name="pixverse_template")],
        )

    @classmethod
    def execute(cls, template: str) -> IO.NodeOutput:
        template_id = pixverse_templates.get(template, None)
        if template_id is None:
            raise Exception(f"Template '{template}' is not recognized.")
        return IO.NodeOutput(template_id)


class PixverseTextToVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="PixverseTextToVideoNode",
            display_name="PixVerse Text to Video",
            category="partner/video/PixVerse",
            description="Generates videos based on prompt and output_size.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt for the video generation",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=PixverseAspectRatio,
                ),
                IO.Combo.Input(
                    "quality",
                    options=PixverseQuality,
                    default=PixverseQuality.res_540p,
                ),
                IO.Combo.Input(
                    "duration_seconds",
                    options=PixverseDuration,
                ),
                IO.Combo.Input(
                    "motion_mode",
                    options=PixverseMotionMode,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    tooltip="Seed for video generation.",
                ),
                IO.String.Input(
                    "negative_prompt",
                    default="",
                    multiline=True,
                    tooltip="An optional text description of undesired elements on an image.",
                    optional=True,
                ),
                IO.Custom(PixverseIO.TEMPLATE).Input(
                    "pixverse_template",
                    tooltip="An optional template to influence style of generation, created by the PixVerse Template node.",
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
            price_badge=PRICE_BADGE_VIDEO,
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        aspect_ratio: str,
        quality: str,
        duration_seconds: int,
        motion_mode: str,
        seed,
        negative_prompt: str = None,
        pixverse_template: int = None,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=False, min_length=1)
        # 1080p is limited to 5 seconds duration
        # only normal motion_mode supported for 1080p or for non-5 second duration
        if quality == PixverseQuality.res_1080p:
            motion_mode = PixverseMotionMode.normal
            duration_seconds = PixverseDuration.dur_5
        elif duration_seconds != PixverseDuration.dur_5:
            motion_mode = PixverseMotionMode.normal

        response_api = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/pixverse/video/text/generate", method="POST"),
            response_model=PixverseVideoResponse,
            data=PixverseTextVideoRequest(
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                quality=quality,
                duration=duration_seconds,
                motion_mode=motion_mode,
                negative_prompt=negative_prompt if negative_prompt else None,
                template_id=pixverse_template,
                seed=seed,
            ),
        )
        if response_api.Resp is None:
            raise Exception(f"PixVerse request failed: '{response_api.ErrMsg}'")

        response_poll = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/pixverse/video/result/{response_api.Resp.video_id}"),
            response_model=PixverseGenerationStatusResponse,
            completed_statuses=PIXVERSE_COMPLETED_STATUSES,
            failed_statuses=PIXVERSE_FAILED_STATUSES,
            status_extractor=_pixverse_status,
            estimated_duration=AVERAGE_DURATION_T2V,
        )
        return IO.NodeOutput(await download_url_to_video_output(response_poll.Resp.url))


class PixverseImageToVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="PixverseImageToVideoNode",
            display_name="PixVerse Image to Video",
            category="partner/video/PixVerse",
            description="Generates videos based on prompt and output_size.",
            inputs=[
                IO.Image.Input("image"),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt for the video generation",
                ),
                IO.Combo.Input(
                    "quality",
                    options=PixverseQuality,
                    default=PixverseQuality.res_540p,
                ),
                IO.Combo.Input(
                    "duration_seconds",
                    options=PixverseDuration,
                ),
                IO.Combo.Input(
                    "motion_mode",
                    options=PixverseMotionMode,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    tooltip="Seed for video generation.",
                ),
                IO.String.Input(
                    "negative_prompt",
                    default="",
                    multiline=True,
                    tooltip="An optional text description of undesired elements on an image.",
                    optional=True,
                ),
                IO.Custom(PixverseIO.TEMPLATE).Input(
                    "pixverse_template",
                    tooltip="An optional template to influence style of generation, created by the PixVerse Template node.",
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
            price_badge=PRICE_BADGE_VIDEO,
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        prompt: str,
        quality: str,
        duration_seconds: int,
        motion_mode: str,
        seed,
        negative_prompt: str = None,
        pixverse_template: int = None,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=False)
        img_id = await upload_image_to_pixverse(cls, image)

        # 1080p is limited to 5 seconds duration
        # only normal motion_mode supported for 1080p or for non-5 second duration
        if quality == PixverseQuality.res_1080p:
            motion_mode = PixverseMotionMode.normal
            duration_seconds = PixverseDuration.dur_5
        elif duration_seconds != PixverseDuration.dur_5:
            motion_mode = PixverseMotionMode.normal

        response_api = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/pixverse/video/img/generate", method="POST"),
            response_model=PixverseVideoResponse,
            data=PixverseImageVideoRequest(
                img_id=img_id,
                prompt=prompt,
                quality=quality,
                duration=duration_seconds,
                motion_mode=motion_mode,
                negative_prompt=negative_prompt if negative_prompt else None,
                template_id=pixverse_template,
                seed=seed,
            ),
        )

        if response_api.Resp is None:
            raise Exception(f"PixVerse request failed: '{response_api.ErrMsg}'")

        response_poll = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/pixverse/video/result/{response_api.Resp.video_id}"),
            response_model=PixverseGenerationStatusResponse,
            completed_statuses=PIXVERSE_COMPLETED_STATUSES,
            failed_statuses=PIXVERSE_FAILED_STATUSES,
            status_extractor=_pixverse_status,
            estimated_duration=AVERAGE_DURATION_I2V,
        )
        return IO.NodeOutput(await download_url_to_video_output(response_poll.Resp.url))


class PixverseTransitionVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="PixverseTransitionVideoNode",
            display_name="PixVerse Transition Video",
            category="partner/video/PixVerse",
            description="Generates videos based on prompt and output_size.",
            inputs=[
                IO.Image.Input("first_frame"),
                IO.Image.Input("last_frame"),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt for the video generation",
                ),
                IO.Combo.Input(
                    "quality",
                    options=PixverseQuality,
                    default=PixverseQuality.res_540p,
                ),
                IO.Combo.Input(
                    "duration_seconds",
                    options=PixverseDuration,
                ),
                IO.Combo.Input(
                    "motion_mode",
                    options=PixverseMotionMode,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    tooltip="Seed for video generation.",
                ),
                IO.String.Input(
                    "negative_prompt",
                    default="",
                    multiline=True,
                    tooltip="An optional text description of undesired elements on an image.",
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
            price_badge=PRICE_BADGE_VIDEO,
        )

    @classmethod
    async def execute(
        cls,
        first_frame: torch.Tensor,
        last_frame: torch.Tensor,
        prompt: str,
        quality: str,
        duration_seconds: int,
        motion_mode: str,
        seed,
        negative_prompt: str = None,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=False)
        first_frame_id = await upload_image_to_pixverse(cls, first_frame)
        last_frame_id = await upload_image_to_pixverse(cls, last_frame)

        # 1080p is limited to 5 seconds duration
        # only normal motion_mode supported for 1080p or for non-5 second duration
        if quality == PixverseQuality.res_1080p:
            motion_mode = PixverseMotionMode.normal
            duration_seconds = PixverseDuration.dur_5
        elif duration_seconds != PixverseDuration.dur_5:
            motion_mode = PixverseMotionMode.normal

        response_api = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/pixverse/video/transition/generate", method="POST"),
            response_model=PixverseVideoResponse,
            data=PixverseTransitionVideoRequest(
                first_frame_img=first_frame_id,
                last_frame_img=last_frame_id,
                prompt=prompt,
                quality=quality,
                duration=duration_seconds,
                motion_mode=motion_mode,
                negative_prompt=negative_prompt if negative_prompt else None,
                seed=seed,
            ),
        )

        if response_api.Resp is None:
            raise Exception(f"PixVerse request failed: '{response_api.ErrMsg}'")

        response_poll = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/pixverse/video/result/{response_api.Resp.video_id}"),
            response_model=PixverseGenerationStatusResponse,
            completed_statuses=PIXVERSE_COMPLETED_STATUSES,
            failed_statuses=PIXVERSE_FAILED_STATUSES,
            status_extractor=_pixverse_status,
            estimated_duration=AVERAGE_DURATION_T2V,
        )
        return IO.NodeOutput(await download_url_to_video_output(response_poll.Resp.url))


PRICE_BADGE_VIDEO = IO.PriceBadge(
    depends_on=IO.PriceBadgeDepends(widgets=["duration_seconds", "quality", "motion_mode"]),
    expr="""
    (
      $prices := {
        "5": {
          "1080p": {"normal": 1.716},
          "720p": {"normal": 0.858, "fast": 1.716},
          "540p": {"normal": 0.6435, "fast": 1.287},
          "360p": {"normal": 0.6435, "fast": 1.287}
        },
        "8": {
          "720p": {"normal": 1.716},
          "540p": {"normal": 1.287},
          "360p": {"normal": 1.287}
        }
      };
      $quality := $lowercase($string($lookup(widgets, "quality")));
      $duration := $quality = "1080p" ? "5" : $string($lookup(widgets, "duration_seconds"));
      $motion := $quality = "1080p" or $duration != "5"
        ? "normal" : $lowercase($string($lookup(widgets, "motion_mode")));
      $price := $lookup($lookup($lookup($prices, $duration), $quality), $motion);
      $type($price) = "number" ? {"type":"usd","usd": $price} : undefined
    )
    """,
)


PIXVERSE_MODELS = {
    "PixVerse V6": "v6",
}

PRICE_BADGE_PIXVERSE = IO.PriceBadge(
    depends_on=IO.PriceBadgeDepends(
        widgets=["model", "model.quality", "model.duration_seconds", "model.generate_audio"]
    ),
    expr="""
    (
      $prices := {
        "pixverse v6": {
          "360p": {"false": 0.0715, "true": 0.1001},
          "540p": {"false": 0.1001, "true": 0.1287},
          "720p": {"false": 0.1287, "true": 0.1716},
          "1080p": {"false": 0.2574, "true": 0.3289}
        }
      };
      $model := $lookup(widgets, "model");
      $table := $type($model) = "string" ? $lookup($prices, $lowercase($model)) : undefined;
      $quality := $lookup(widgets, "model.quality");
      $row := $type($table) = "object" and $type($quality) = "string"
        ? $lookup($table, $lowercase($quality)) : undefined;
      $audio := $string($lookup(widgets, "model.generate_audio")) = "true" ? "true" : "false";
      $pps := $type($row) = "object" ? $lookup($row, $audio) : undefined;
      $durationRaw := $lookup(widgets, "model.duration_seconds");
      $duration := $type($durationRaw) in ["string", "number"] ? $number($durationRaw) : undefined;
      $type($pps) = "number" and $type($duration) = "number"
        ? {"type":"usd","usd": $pps * $duration}
        : undefined
    )
    """,
)


PRICE_BADGE_PIXVERSE_FUSION = IO.PriceBadge(
    depends_on=IO.PriceBadgeDepends(
        widgets=["model", "model.quality", "model.duration_seconds", "model.generate_audio"],
        input_groups=["videos"],
    ),
    expr="""
    (
      $rates := {
        "pixverse v6": {
          "360p": {"false": 0.0715, "true": 0.1001},
          "540p": {"false": 0.1001, "true": 0.1287},
          "720p": {"false": 0.1287, "true": 0.1716},
          "1080p": {"false": 0.2574, "true": 0.3289}
        }
      };
      $model := $lookup(widgets, "model");
      $table := $type($model) = "string" ? $lookup($rates, $lowercase($model)) : undefined;
      $quality := $lookup(widgets, "model.quality");
      $row := $type($table) = "object" and $type($quality) = "string"
        ? $lookup($table, $lowercase($quality)) : undefined;
      $audio := $string($lookup(widgets, "model.generate_audio")) = "true" ? "true" : "false";
      $pps := $type($row) = "object" ? $lookup($row, $audio) : undefined;
      $hasVideo := $exists(inputGroups) and $lookup(inputGroups, "videos") > 0;
      $durationRaw := $lookup(widgets, "model.duration_seconds");
      $duration := $type($durationRaw) in ["string", "number"] ? $number($durationRaw) : undefined;
      $type($pps) = "number"
        ? ($hasVideo
            ? {"type":"usd","usd": $pps * 2, "format":{"suffix":"/second of reference video"}}
            : ($type($duration) = "number" ? {"type":"usd","usd": $pps * $duration} : undefined))
        : undefined
    )
    """,
)

def _pixverse6_inputs(
    *,
    aspect_ratio_options: list | None = None,
    with_multi_clip: bool = False,
    prompt_tooltip: str = "Prompt for the video generation.",
    quality_tooltip: str = "Output resolution. Sets the long edge: 360p is 640px, 540p 1024px, "
    "720p 1280px, 1080p 1920px.",
) -> list:
    inputs = [
        IO.String.Input("prompt", multiline=True, default="", tooltip=prompt_tooltip),
    ]
    if aspect_ratio_options is not None:
        inputs.append(
            IO.Combo.Input(
                "aspect_ratio",
                options=aspect_ratio_options,
                tooltip="Output aspect ratio.",
            )
        )
    inputs += [
        IO.Combo.Input(
            "quality",
            options=PixverseQuality,
            default=PixverseQuality.res_720p,
            tooltip=quality_tooltip,
        ),
        IO.Int.Input(
            "duration_seconds",
            default=5,
            min=V6_MIN_DURATION,
            max=V6_MAX_DURATION,
            tooltip="Length of the generated video in seconds.",
        ),
        IO.Boolean.Input(
            "generate_audio",
            default=True,
            tooltip="Generate a native audio track together with the video.",
        ),
    ]
    if with_multi_clip:
        inputs.append(
            IO.Boolean.Input(
                "multi_clip",
                default=False,
                tooltip="Let the model cut the video into several shots instead of one continuous take.",
            )
        )
    inputs += [
        IO.Int.Input(
            "seed",
            default=42,
            min=0,
            max=2147483647,
            control_after_generate=True,
            tooltip="Seed for video generation. PixVerse records it but does not reproduce a run from it.",
        ),
        IO.String.Input(
            "negative_prompt",
            default="",
            multiline=True,
            optional=True,
            tooltip="An optional text description of undesired elements in the video.",
        ),
        IO.Combo.Input(
            "style",
            options=PixverseV6Style,
            default=PixverseV6Style.none,
            optional=True,
            tooltip="An optional visual style applied to the whole video.",
        ),
    ]
    return inputs


def _pixverse_model_input(**kwargs) -> IO.DynamicCombo.Input:
    return IO.DynamicCombo.Input(
        "model",
        options=[IO.DynamicCombo.Option("PixVerse V6", _pixverse6_inputs(**kwargs))],
        tooltip="Model and generation settings.",
    )


def _validate_prompts(prompt: str, negative_prompt: str | None) -> None:
    validate_string(prompt, strip_whitespace=True, min_length=1, max_length=V6_MAX_PROMPT_LENGTH)
    if negative_prompt and len(negative_prompt) > V6_MAX_NEGATIVE_PROMPT_LENGTH:
        raise ValueError(
            f"Negative prompt must be at most {V6_MAX_NEGATIVE_PROMPT_LENGTH} characters, "
            f"got {len(negative_prompt)}."
        )


def _model_style(model: dict) -> str | None:
    style = model.get("style")
    if not style or style == PixverseV6Style.none:
        return None
    return style


def _model_common(model: dict) -> dict:
    _validate_prompts(model["prompt"], model.get("negative_prompt"))
    return {
        "model": PIXVERSE_MODELS[model["model"]],
        "prompt": model["prompt"],
        "quality": model["quality"],
        "duration": model["duration_seconds"],
        "generate_audio_switch": model["generate_audio"],
        "negative_prompt": model.get("negative_prompt") or None,
        "style": _model_style(model),
        "seed": model["seed"],
    }


def _pixverse_error(response) -> Exception:
    code = response.ErrCode
    message = response.ErrMsg or "unknown error"
    if code == 500044:
        return Exception(
            "PixVerse is already running the maximum number of simultaneous generations. "
            "Try again in a moment."
        )
    if code == 500090:
        return Exception("PixVerse rejected the request: the provider account is out of credits.")
    if code == 500063:
        return Exception(f"PixVerse content moderation rejected the request: {message}")
    return Exception(f"PixVerse request failed ({code}): '{message}'")


async def _pixverse_generate(cls: type[IO.ComfyNode], path: str, request) -> IO.NodeOutput:
    response_api = await sync_op(
        cls,
        ApiEndpoint(path=path, method="POST"),
        response_model=PixverseVideoResponse,
        data=request,
    )
    if response_api.Resp is None:
        raise _pixverse_error(response_api)
    response_poll = await poll_op(
        cls,
        ApiEndpoint(path=f"/proxy/pixverse/video/result/{response_api.Resp.video_id}"),
        response_model=PixverseGenerationStatusResponse,
        completed_statuses=PIXVERSE_COMPLETED_STATUSES,
        failed_statuses=PIXVERSE_FAILED_STATUSES,
        status_extractor=_pixverse_status,
    )
    return IO.NodeOutput(await download_url_to_video_output(response_poll.Resp.url))


class PixverseV6TextToVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="PixverseV6TextToVideoNode",
            display_name="PixVerse V6 Text to Video",
            category="partner/video/PixVerse",
            description="Generates a video from a text prompt with PixVerse, optionally with native audio.",
            inputs=[
                _pixverse_model_input(
                    aspect_ratio_options=PixverseV6AspectRatio,
                    with_multi_clip=True,
                ),
            ],
            outputs=[IO.Video.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=PRICE_BADGE_PIXVERSE,
        )

    @classmethod
    async def execute(cls, model: dict) -> IO.NodeOutput:
        return await _pixverse_generate(
            cls,
            "/proxy/pixverse/video/text/generate",
            PixverseV6TextVideoRequest(
                **_model_common(model),
                aspect_ratio=model["aspect_ratio"],
                generate_multi_clip_switch=model["multi_clip"],
            ),
        )


class PixverseV6ImageToVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="PixverseV6ImageToVideoNode",
            display_name="PixVerse V6 Image to Video",
            category="partner/video/PixVerse",
            description="Animates an image with PixVerse, optionally with native audio. "
            "The output keeps the aspect ratio of the input image.",
            inputs=[
                IO.Image.Input("image"),
                _pixverse_model_input(with_multi_clip=True),
            ],
            outputs=[IO.Video.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=PRICE_BADGE_PIXVERSE,
        )

    @classmethod
    async def execute(cls, image: torch.Tensor, model: dict) -> IO.NodeOutput:
        common = _model_common(model)
        return await _pixverse_generate(
            cls,
            "/proxy/pixverse/video/img/generate",
            PixverseV6ImageVideoRequest(
                **common,
                img_id=await upload_image_to_pixverse(cls, image),
                generate_multi_clip_switch=model["multi_clip"],
            ),
        )


class PixverseV6FirstLastFrameNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="PixverseV6FirstLastFrameNode",
            display_name="PixVerse V6 First-Last-Frame to Video",
            category="partner/video/PixVerse",
            description="Generates a video that transitions from a first frame to a last frame with PixVerse, "
            "optionally with native audio. The output keeps the aspect ratio of the first frame.",
            inputs=[
                IO.Image.Input("first_frame"),
                IO.Image.Input("last_frame"),
                _pixverse_model_input(prompt_tooltip="Prompt describing the transition."),
            ],
            outputs=[IO.Video.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=PRICE_BADGE_PIXVERSE,
        )

    @classmethod
    async def execute(cls, first_frame: torch.Tensor, last_frame: torch.Tensor, model: dict) -> IO.NodeOutput:
        common = _model_common(model)
        return await _pixverse_generate(
            cls,
            "/proxy/pixverse/video/transition/generate",
            PixverseV6TransitionVideoRequest(
                **common,
                first_frame_img=await upload_image_to_pixverse(cls, first_frame),
                last_frame_img=await upload_image_to_pixverse(cls, last_frame),
            ),
        )


class PixverseV6ExtendVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="PixverseV6ExtendVideoNode",
            display_name="PixVerse V6 Extend Video",
            category="partner/video/PixVerse",
            description="Continues an existing video with PixVerse, optionally with native audio. The source must "
            "be under 40 seconds and at most 1920px on either side. The output keeps the source's resolution, so "
            "quality sets how well the continuation is rendered rather than the frame size.",
            inputs=[
                IO.Video.Input("video", tooltip="Video to continue."),
                _pixverse_model_input(
                    prompt_tooltip="Prompt describing how the video should continue.",
                    quality_tooltip="Render quality of the generated continuation: 1080p looks markedly better than "
                    "540p or 360p. It never resizes - the output keeps the source video's resolution.",
                ),
            ],
            outputs=[IO.Video.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=PRICE_BADGE_PIXVERSE,
        )

    @classmethod
    async def execute(cls, video: Input.Video, model: dict) -> IO.NodeOutput:
        common = _model_common(model)
        source_seconds = video.get_duration()
        if source_seconds >= V6_EXTEND_SOURCE_SECONDS_LIMIT:
            raise ValueError(
                f"PixVerse extends videos shorter than {V6_EXTEND_SOURCE_SECONDS_LIMIT} seconds; "
                f"this one is {source_seconds:.2f}s."
            )
        validate_video_dimensions(video, max_width=V6_MAX_SOURCE_VIDEO_SIDE, max_height=V6_MAX_SOURCE_VIDEO_SIDE)
        return await _pixverse_generate(
            cls,
            "/proxy/pixverse/video/extend/generate",
            PixverseV6ExtendVideoRequest(
                **common,
                video_media_id=await upload_video_to_pixverse(cls, video),
            ),
        )


def _rewrite_reference_tags(prompt: str, ref_names: set[str]) -> str:
    def repl(match: re.Match) -> str:
        name = match.group(1).lower() + match.group(2)
        if name not in ref_names:
            available = ", ".join(f"@{n}" for n in sorted(ref_names)) or "none"
            raise ValueError(f"@{match.group(1)}{match.group(2)} is not connected. Available references: {available}.")
        return f"@{name} "

    return re.sub(
        r"(?<!\S)@(subject|background|video)([0-9]+)\b\s*",
        repl,
        prompt,
        flags=re.IGNORECASE,
    )


class PixverseV6FusionVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="PixverseV6FusionVideoNode",
            display_name="PixVerse V6 Fusion (Reference to Video)",
            category="partner/video/PixVerse",
            description="Composes a video from reference subjects, backgrounds and videos with PixVerse. "
            "Place a reference in the scene by naming it in the prompt, for example "
            "'@Subject1 walks through @Background1'. Connecting a reference video switches the model to Omni mode, "
            "where the output length matches the longest reference video.",
            inputs=[
                IO.Autogrow.Input(
                    "subjects",
                    template=IO.Autogrow.TemplateNames(
                        IO.Image.Input("subject"),
                        names=[f"subject{i}" for i in range(1, V6_MAX_SUBJECTS + 1)],
                        min=0,
                    ),
                    tooltip="Reference images of the subjects to place in the scene.",
                ),
                IO.Autogrow.Input(
                    "backgrounds",
                    template=IO.Autogrow.TemplateNames(
                        IO.Image.Input("background"),
                        names=[f"background{i}" for i in range(1, V6_MAX_BACKGROUNDS + 1)],
                        min=0,
                    ),
                    tooltip="Reference images of the scene the subjects are placed into.",
                ),
                IO.Autogrow.Input(
                    "videos",
                    template=IO.Autogrow.TemplateNames(
                        IO.Video.Input("video"),
                        names=[f"video{i}" for i in range(1, V6_MAX_REFERENCE_VIDEOS + 1)],
                        min=0,
                    ),
                    tooltip="Reference videos to borrow subjects, motion, framing or style from. "
                    "At most two, at most 15 seconds in total.",
                ),
                _pixverse_model_input(
                    aspect_ratio_options=[*[ratio.value for ratio in PixverseV6AspectRatio], "auto"],
                    prompt_tooltip="Prompt for the video generation. Refer to connected references as "
                    "@Subject1, @Background1, @Video1.",
                ),
            ],
            outputs=[IO.Video.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=PRICE_BADGE_PIXVERSE_FUSION,
        )

    @classmethod
    async def execute(
        cls,
        subjects: IO.Autogrow.Type,
        backgrounds: IO.Autogrow.Type,
        videos: IO.Autogrow.Type,
        model: dict,
    ) -> IO.NodeOutput:
        common = _model_common(model)
        aspect_ratio = model["aspect_ratio"]
        subjects = subjects or {}
        backgrounds = backgrounds or {}
        videos = videos or {}
        if not subjects and not backgrounds and not videos:
            raise ValueError("Connect at least one subject, background or reference video.")

        is_omni = bool(videos)
        image_count = len(subjects) + len(backgrounds)
        max_images = V6_MAX_REFERENCE_IMAGES_OMNI if is_omni else V6_MAX_REFERENCE_IMAGES_PLAIN
        if image_count > max_images:
            raise ValueError(
                f"PixVerse accepts at most {max_images} reference images "
                f"{'in Omni mode' if is_omni else 'without a reference video'}, got {image_count}."
            )
        if aspect_ratio == "auto" and not is_omni:
            raise ValueError("aspect_ratio 'auto' requires at least one connected reference video.")

        total_video_seconds = 0.0
        for video in videos.values():
            validate_video_duration(video, max_duration=V6_MAX_REFERENCE_VIDEO_SECONDS)
            total_video_seconds += video.get_duration()
        if total_video_seconds > V6_MAX_REFERENCE_VIDEO_SECONDS:
            raise ValueError(
                f"Reference videos must total at most {V6_MAX_REFERENCE_VIDEO_SECONDS} seconds, "
                f"got {total_video_seconds:.2f}."
            )

        common["prompt"] = _rewrite_reference_tags(
            common["prompt"], set(subjects) | set(backgrounds) | set(videos)
        )
        common["duration"] = 0 if is_omni else common["duration"]

        image_references = []
        for ref_name, image in subjects.items():
            image_references.append(
                PixverseImageReference(
                    img_id=await upload_image_to_pixverse(cls, image),
                    ref_name=ref_name,
                    type=PixverseReferenceType.subject,
                )
            )
        for ref_name, image in backgrounds.items():
            image_references.append(
                PixverseImageReference(
                    img_id=await upload_image_to_pixverse(cls, image),
                    ref_name=ref_name,
                    type=PixverseReferenceType.background,
                )
            )
        video_references = []
        for ref_name, video in videos.items():
            video_references.append(
                PixverseVideoReference(
                    ref_name=ref_name,
                    video_media_id=await upload_video_to_pixverse(cls, video),
                )
            )

        return await _pixverse_generate(
            cls,
            "/proxy/pixverse/video/fusion/generate",
            PixverseV6FusionVideoRequest(
                **common,
                aspect_ratio=aspect_ratio,
                image_references=image_references or None,
                video_references=video_references or None,
                reference_mode="omni" if is_omni else None,
            ),
        )


class PixVerseExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            PixverseTextToVideoNode,
            PixverseImageToVideoNode,
            PixverseTransitionVideoNode,
            PixverseTemplateNode,
            PixverseV6TextToVideoNode,
            PixverseV6ImageToVideoNode,
            PixverseV6FirstLastFrameNode,
            PixverseV6ExtendVideoNode,
            PixverseV6FusionVideoNode,
        ]


async def comfy_entrypoint() -> PixVerseExtension:
    return PixVerseExtension()
