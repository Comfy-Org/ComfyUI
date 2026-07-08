import torch
from typing_extensions import override
from comfy_api.latest import IO, ComfyExtension
from comfy_api_nodes.apis.pixverse import (
    PixverseTextVideoRequest,
    PixverseImageVideoRequest,
    PixverseTransitionVideoRequest,
    PixverseImageUploadResponse,
    PixverseVideoResponse,
    PixverseGenerationStatusResponse,
    PixverseAspectRatio,
    PixverseQuality,
    PixverseDuration,
    PixverseMotionMode,
    PixverseStatus,
    PixverseIO,
    pixverse_templates,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_video_output,
    poll_op,
    sync_op,
    tensor_to_bytesio,
    validate_string,
)

AVERAGE_DURATION_T2V = 32
AVERAGE_DURATION_I2V = 30
AVERAGE_DURATION_T2T = 52


async def upload_image_to_pixverse(cls: type[IO.ComfyNode], image: torch.Tensor):
    response_upload = await sync_op(
        cls,
        ApiEndpoint(path="/proxy/pixverse/image/upload", method="POST"),
        response_model=PixverseImageUploadResponse,
        files={"image": tensor_to_bytesio(image)},
        content_type="multipart/form-data",
    )
    if response_upload.Resp is None:
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
            completed_statuses=[PixverseStatus.successful],
            failed_statuses=[
                PixverseStatus.contents_moderation,
                PixverseStatus.failed,
                PixverseStatus.deleted,
            ],
            status_extractor=lambda x: x.Resp.status,
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
            completed_statuses=[PixverseStatus.successful],
            failed_statuses=[
                PixverseStatus.contents_moderation,
                PixverseStatus.failed,
                PixverseStatus.deleted,
            ],
            status_extractor=lambda x: x.Resp.status,
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
            completed_statuses=[PixverseStatus.successful],
            failed_statuses=[
                PixverseStatus.contents_moderation,
                PixverseStatus.failed,
                PixverseStatus.deleted,
            ],
            status_extractor=lambda x: x.Resp.status,
            estimated_duration=AVERAGE_DURATION_T2V,
        )
        return IO.NodeOutput(await download_url_to_video_output(response_poll.Resp.url))


PRICE_BADGE_VIDEO = IO.PriceBadge(
    depends_on=IO.PriceBadgeDepends(widgets=["duration_seconds", "quality", "motion_mode"]),
    expr="""
    (
      $prices := {
        "5": {
          "1080p": {"normal": 1.2, "fast": 1.2},
          "720p": {"normal": 0.6, "fast": 1.2},
          "540p": {"normal": 0.45, "fast": 0.9},
          "360p": {"normal": 0.45, "fast": 0.9}
        },
        "8": {
          "1080p": {"normal": 1.2, "fast": 1.2},
          "720p": {"normal": 1.2, "fast": 1.2},
          "540p": {"normal": 0.9, "fast": 1.2},
          "360p": {"normal": 0.9, "fast": 1.2}
        }
      };
      $durPrices := $lookup($prices, $string(widgets.duration_seconds));
      $qualityPrices := $lookup($durPrices, widgets.quality);
      $price := $lookup($qualityPrices, widgets.motion_mode);
      {"type":"usd","usd": $price ? $price : 0.9}
    )
    """,
)


class PixVerseExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            PixverseTextToVideoNode,
            PixverseImageToVideoNode,
            PixverseTransitionVideoNode,
            PixverseTemplateNode,
        ]


async def comfy_entrypoint() -> PixVerseExtension:
    return PixVerseExtension()
