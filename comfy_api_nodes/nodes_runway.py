"""Runway API Nodes

API Docs:
  - https://docs.dev.runwayml.com/api/#tag/Task-management/paths/~1v1~1tasks~1%7Bid%7D/delete

User Guides:
  - https://help.runwayml.com/hc/en-us/sections/30265301423635-Gen-3-Alpha
  - https://help.runwayml.com/hc/en-us/articles/37327109429011-Creating-with-Gen-4-Video
  - https://help.runwayml.com/hc/en-us/articles/33927968552339-Creating-with-Act-One-on-Gen-3-Alpha-and-Turbo
  - https://help.runwayml.com/hc/en-us/articles/34170748696595-Creating-with-Keyframes-on-Gen-3

"""

from enum import Enum

from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input, InputImpl
from comfy_api_nodes.apis.runway import (
    RunwayImageToVideoRequest,
    RunwayImageToVideoResponse,
    RunwayTaskStatusResponse as TaskStatusResponse,
    RunwayModelEnum as Model,
    RunwayDurationEnum as Duration,
    RunwayAspectRatioEnum as AspectRatio,
    RunwayPromptImageObject,
    RunwayPromptImageDetailedObject,
    RunwayTextToImageRequest,
    RunwayTextToImageResponse,
    Model4,
    ReferenceImage,
    RunwayTextToImageAspectRatioEnum,
    RunwayAleph2IO,
    RunwayAleph2KeyframeChain,
    RunwayAleph2KeyframeItem,
    RunwayAleph2PromptImageChain,
    RunwayAleph2PromptImageItem,
    RunwayAleph2Request,
    RunwayAleph2Response,
    RunwayAleph2KeyframeSeconds,
    RunwayAleph2KeyframeAt,
    RunwayAleph2PromptImage,
    RunwayAleph2TimestampPosition,
    RunwayAleph2RelativePosition,
    RunwayAleph2ContentModeration,
    KEYFRAME_MODE_SECONDS,
    KEYFRAME_MODE_AT,
    PROMPT_IMAGE_MODE_TIMESTAMP,
    PROMPT_IMAGE_MODE_POSITION,
)
from comfy_api_nodes.util import (
    image_tensor_pair_to_batch,
    validate_string,
    validate_image_dimensions,
    validate_image_aspect_ratio,
    validate_video_duration,
    upload_images_to_comfyapi,
    upload_image_to_comfyapi,
    upload_video_to_comfyapi,
    download_url_to_video_output,
    download_url_to_image_tensor,
    ApiEndpoint,
    sync_op,
    poll_op,
)

PATH_IMAGE_TO_VIDEO = "/proxy/runway/image_to_video"
PATH_VIDEO_TO_VIDEO = "/proxy/runway/video_to_video"
PATH_TEXT_TO_IMAGE = "/proxy/runway/text_to_image"
PATH_GET_TASK_STATUS = "/proxy/runway/tasks"

AVERAGE_DURATION_I2V_SECONDS = 64
AVERAGE_DURATION_FLF_SECONDS = 256
AVERAGE_DURATION_T2I_SECONDS = 41


class RunwayGen4TurboAspectRatio(str, Enum):
    """Aspect ratios supported for Image to Video API when using gen4_turbo model."""

    field_1280_720 = "1280:720"
    field_720_1280 = "720:1280"
    field_1104_832 = "1104:832"
    field_832_1104 = "832:1104"
    field_960_960 = "960:960"
    field_1584_672 = "1584:672"


class RunwayGen3aAspectRatio(str, Enum):
    """Aspect ratios supported for Image to Video API when using gen3a_turbo model."""

    field_768_1280 = "768:1280"
    field_1280_768 = "1280:768"


def get_video_url_from_task_status(response: TaskStatusResponse) -> str | None:
    """Returns the video URL from the task status response if it exists."""
    if hasattr(response, "output") and len(response.output) > 0:
        return response.output[0]
    return None


def get_image_url_from_task_status(response: TaskStatusResponse) -> str | None:
    """Returns the image URL from the task status response if it exists."""
    if hasattr(response, "output") and len(response.output) > 0:
        return response.output[0]
    return None


async def get_response(
    cls: type[IO.ComfyNode], task_id: str, estimated_duration: int | None = None
) -> TaskStatusResponse:
    return await poll_op(
        cls,
        ApiEndpoint(path=f"{PATH_GET_TASK_STATUS}/{task_id}"),
        response_model=TaskStatusResponse,
        status_extractor=lambda r: r.status,
        estimated_duration=estimated_duration,
        progress_extractor=lambda r: r.progress * 100 if r.progress is not None else None,
    )


async def generate_video(
    cls: type[IO.ComfyNode],
    request: RunwayImageToVideoRequest,
    estimated_duration: int | None = None,
) -> InputImpl.VideoFromFile:
    initial_response = await sync_op(
        cls,
        endpoint=ApiEndpoint(path=PATH_IMAGE_TO_VIDEO, method="POST"),
        response_model=RunwayImageToVideoResponse,
        data=request,
    )

    final_response = await get_response(cls, initial_response.id, estimated_duration)
    if not final_response.output:
        raise ValueError("Runway task succeeded but no video data found in response.")

    video_url = get_video_url_from_task_status(final_response)
    return await download_url_to_video_output(video_url)


class RunwayImageToVideoNodeGen3a(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RunwayImageToVideoNodeGen3a",
            display_name="Runway Image to Video (Gen3a Turbo)",
            category="partner/video/Runway",
            description="Generate a video from a single starting frame using Gen3a Turbo model. "
            "Before diving in, review these best practices to ensure that "
            "your input selections will set your generation up for success: "
            "https://help.runwayml.com/hc/en-us/articles/33927968552339-Creating-with-Act-One-on-Gen-3-Alpha-and-Turbo.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text prompt for the generation",
                ),
                IO.Image.Input(
                    "start_frame",
                    tooltip="Start frame to be used for the video",
                ),
                IO.Combo.Input(
                    "duration",
                    options=Duration,
                ),
                IO.Combo.Input(
                    "ratio",
                    options=RunwayGen3aAspectRatio,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=4294967295,
                    step=1,
                    control_after_generate=True,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Random seed for generation",
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
                depends_on=IO.PriceBadgeDepends(widgets=["duration"]),
                expr="""{"type":"usd","usd": 0.0715 * widgets.duration}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        start_frame: Input.Image,
        duration: str,
        ratio: str,
        seed: int,
    ) -> IO.NodeOutput:
        validate_string(prompt, min_length=1)
        validate_image_dimensions(start_frame, max_width=7999, max_height=7999)
        validate_image_aspect_ratio(start_frame, (1, 2), (2, 1))

        download_urls = await upload_images_to_comfyapi(
            cls,
            start_frame,
            max_images=1,
            mime_type="image/png",
        )

        return IO.NodeOutput(
            await generate_video(
                cls,
                RunwayImageToVideoRequest(
                    promptText=prompt,
                    seed=seed,
                    model=Model("gen3a_turbo"),
                    duration=Duration(duration),
                    ratio=AspectRatio(ratio),
                    promptImage=RunwayPromptImageObject(
                        root=[RunwayPromptImageDetailedObject(uri=str(download_urls[0]), position="first")]
                    ),
                ),
            )
        )


class RunwayImageToVideoNodeGen4(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RunwayImageToVideoNodeGen4",
            display_name="Runway Image to Video (Gen4 Turbo)",
            category="partner/video/Runway",
            description="Generate a video from a single starting frame using Gen4 Turbo model. "
            "Before diving in, review these best practices to ensure that "
            "your input selections will set your generation up for success: "
            "https://help.runwayml.com/hc/en-us/articles/37327109429011-Creating-with-Gen-4-Video.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text prompt for the generation",
                ),
                IO.Image.Input(
                    "start_frame",
                    tooltip="Start frame to be used for the video",
                ),
                IO.Combo.Input(
                    "duration",
                    options=Duration,
                ),
                IO.Combo.Input(
                    "ratio",
                    options=RunwayGen4TurboAspectRatio,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=4294967295,
                    step=1,
                    control_after_generate=True,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Random seed for generation",
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
                depends_on=IO.PriceBadgeDepends(widgets=["duration"]),
                expr="""{"type":"usd","usd": 0.0715 * widgets.duration}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        start_frame: Input.Image,
        duration: str,
        ratio: str,
        seed: int,
    ) -> IO.NodeOutput:
        validate_string(prompt, min_length=1)
        validate_image_dimensions(start_frame, max_width=7999, max_height=7999)
        validate_image_aspect_ratio(start_frame, (1, 2), (2, 1))

        download_urls = await upload_images_to_comfyapi(
            cls,
            start_frame,
            max_images=1,
            mime_type="image/png",
        )

        return IO.NodeOutput(
            await generate_video(
                cls,
                RunwayImageToVideoRequest(
                    promptText=prompt,
                    seed=seed,
                    model=Model("gen4_turbo"),
                    duration=Duration(duration),
                    ratio=AspectRatio(ratio),
                    promptImage=RunwayPromptImageObject(
                        root=[RunwayPromptImageDetailedObject(uri=str(download_urls[0]), position="first")]
                    ),
                ),
                estimated_duration=AVERAGE_DURATION_FLF_SECONDS,
            )
        )


class RunwayFirstLastFrameNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RunwayFirstLastFrameNode",
            display_name="Runway First-Last-Frame to Video",
            category="partner/video/Runway",
            description="Upload first and last keyframes, draft a prompt, and generate a video. "
            "More complex transitions, such as cases where the Last frame is completely different "
            "from the First frame, may benefit from the longer 10s duration. "
            "This would give the generation more time to smoothly transition between the two inputs. "
            "Before diving in, review these best practices to ensure that your input selections "
            "will set your generation up for success: "
            "https://help.runwayml.com/hc/en-us/articles/34170748696595-Creating-with-Keyframes-on-Gen-3.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text prompt for the generation",
                ),
                IO.Image.Input(
                    "start_frame",
                    tooltip="Start frame to be used for the video",
                ),
                IO.Image.Input(
                    "end_frame",
                    tooltip="End frame to be used for the video. Supported for gen3a_turbo only.",
                ),
                IO.Combo.Input(
                    "duration",
                    options=Duration,
                ),
                IO.Combo.Input(
                    "ratio",
                    options=RunwayGen3aAspectRatio,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=4294967295,
                    step=1,
                    control_after_generate=True,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Random seed for generation",
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
                depends_on=IO.PriceBadgeDepends(widgets=["duration"]),
                expr="""{"type":"usd","usd": 0.0715 * widgets.duration}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        start_frame: Input.Image,
        end_frame: Input.Image,
        duration: str,
        ratio: str,
        seed: int,
    ) -> IO.NodeOutput:
        validate_string(prompt, min_length=1)
        validate_image_dimensions(start_frame, max_width=7999, max_height=7999)
        validate_image_dimensions(end_frame, max_width=7999, max_height=7999)
        validate_image_aspect_ratio(start_frame, (1, 2), (2, 1))
        validate_image_aspect_ratio(end_frame, (1, 2), (2, 1))

        stacked_input_images = image_tensor_pair_to_batch(start_frame, end_frame)
        download_urls = await upload_images_to_comfyapi(
            cls,
            stacked_input_images,
            max_images=2,
            mime_type="image/png",
        )
        if len(download_urls) != 2:
            raise ValueError("Failed to upload one or more images to comfy api.")

        return IO.NodeOutput(
            await generate_video(
                cls,
                RunwayImageToVideoRequest(
                    promptText=prompt,
                    seed=seed,
                    model=Model("gen3a_turbo"),
                    duration=Duration(duration),
                    ratio=AspectRatio(ratio),
                    promptImage=RunwayPromptImageObject(
                        root=[
                            RunwayPromptImageDetailedObject(uri=str(download_urls[0]), position="first"),
                            RunwayPromptImageDetailedObject(uri=str(download_urls[1]), position="last"),
                        ]
                    ),
                ),
                estimated_duration=AVERAGE_DURATION_FLF_SECONDS,
            )
        )


class RunwayTextToImageNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RunwayTextToImageNode",
            display_name="Runway Text to Image",
            category="partner/image/Runway",
            description="Generate an image from a text prompt using Runway's Gen 4 model. "
            "You can also include reference image to guide the generation.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text prompt for the generation",
                ),
                IO.Combo.Input(
                    "ratio",
                    options=[model.value for model in RunwayTextToImageAspectRatioEnum],
                ),
                IO.Image.Input(
                    "reference_image",
                    tooltip="Optional reference image to guide the generation",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Image.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.11}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        ratio: str,
        reference_image: Input.Image | None = None,
    ) -> IO.NodeOutput:
        validate_string(prompt, min_length=1)

        # Prepare reference images if provided
        reference_images = None
        if reference_image is not None:
            validate_image_dimensions(reference_image, max_width=7999, max_height=7999)
            validate_image_aspect_ratio(reference_image, (1, 2), (2, 1))
            download_urls = await upload_images_to_comfyapi(
                cls,
                reference_image,
                max_images=1,
                mime_type="image/png",
            )
            reference_images = [ReferenceImage(uri=str(download_urls[0]))]

        initial_response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path=PATH_TEXT_TO_IMAGE, method="POST"),
            response_model=RunwayTextToImageResponse,
            data=RunwayTextToImageRequest(
                promptText=prompt,
                model=Model4.gen4_image,
                ratio=ratio,
                referenceImages=reference_images,
            ),
        )

        final_response = await get_response(
            cls,
            initial_response.id,
            estimated_duration=AVERAGE_DURATION_T2I_SECONDS,
        )
        if not final_response.output:
            raise ValueError("Runway task succeeded but no image data found in response.")

        return IO.NodeOutput(await download_url_to_image_tensor(get_image_url_from_task_status(final_response)))


_TIMING_ABSOLUTE = "Absolute time (seconds)"
_TIMING_FRACTION = "Fraction of duration (0.0-1.0)"


class RunwayAleph2KeyframeNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RunwayAleph2KeyframeNode",
            display_name="Runway Aleph2 Keyframe",
            category="partner/video/Runway",
            description="Anchor a guidance image to a moment of the input (source) video, so Aleph2 "
            "steers the edit at that point of your footage. Connect this to the 'keyframes' input of "
            "the Runway Aleph2 Video to Video node; chain several together (up to 5) via the optional "
            "'keyframes' input below.",
            inputs=[
                IO.Image.Input(
                    "image",
                    tooltip="The guidance image to apply at the chosen moment of the input video.",
                ),
                IO.DynamicCombo.Input(
                    "timing",
                    options=[
                        IO.DynamicCombo.Option(
                            _TIMING_ABSOLUTE,
                            [
                                IO.Float.Input(
                                    "seconds",
                                    default=0.0,
                                    min=0.0,
                                    max=30.0,
                                    step=0.1,
                                    display_mode=IO.NumberDisplay.number,
                                    tooltip="Time in seconds from start of the input video where this image applies.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            _TIMING_FRACTION,
                            [
                                IO.Float.Input(
                                    "fraction",
                                    default=0.0,
                                    min=0.0,
                                    max=1.0,
                                    step=0.01,
                                    display_mode=IO.NumberDisplay.number,
                                    tooltip="Where in the input video this image applies, "
                                    "as a fraction of its duration (0.0 = start, 1.0 = end).",
                                ),
                            ],
                        ),
                    ],
                    tooltip="How to place this image on the input video's timeline.",
                ),
                IO.Custom(RunwayAleph2IO.KEYFRAME).Input(
                    "keyframes",
                    optional=True,
                    tooltip="Optional earlier keyframes to chain with this one.",
                ),
            ],
            outputs=[IO.Custom(RunwayAleph2IO.KEYFRAME).Output(display_name="keyframes")],
        )

    @classmethod
    def execute(
        cls,
        image: Input.Image,
        timing: dict,
        keyframes: RunwayAleph2KeyframeChain | None = None,
    ) -> IO.NodeOutput:
        chain = keyframes.clone() if keyframes is not None else RunwayAleph2KeyframeChain()
        if timing["timing"] == _TIMING_ABSOLUTE:
            mode, value = KEYFRAME_MODE_SECONDS, float(timing["seconds"])
        else:
            mode, value = KEYFRAME_MODE_AT, float(timing["fraction"])
        chain.add(RunwayAleph2KeyframeItem(image=image, mode=mode, value=value))
        return IO.NodeOutput(chain)


class RunwayAleph2PromptImageNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RunwayAleph2PromptImageNode",
            display_name="Runway Aleph2 Prompt Image",
            category="partner/video/Runway",
            description="Anchor a guidance image to a moment of the output (result) video, to guide what "
            "the edited video looks like at that point. Connect this to the 'prompt_images' input of the "
            "Runway Aleph2 Video to Video node; chain several together (up to 5) via the optional "
            "'prompt_images' input below.",
            inputs=[
                IO.Image.Input(
                    "image",
                    tooltip="The guidance image to place at the chosen moment of the output video.",
                ),
                IO.DynamicCombo.Input(
                    "position",
                    options=[
                        IO.DynamicCombo.Option(
                            _TIMING_ABSOLUTE,
                            [
                                IO.Float.Input(
                                    "seconds",
                                    default=0.0,
                                    min=0.0,
                                    max=30.0,
                                    step=0.1,
                                    display_mode=IO.NumberDisplay.number,
                                    tooltip="Time in seconds from start of the output video where this image applies.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            _TIMING_FRACTION,
                            [
                                IO.Float.Input(
                                    "fraction",
                                    default=0.0,
                                    min=0.0,
                                    max=1.0,
                                    step=0.01,
                                    display_mode=IO.NumberDisplay.number,
                                    tooltip="Where in the output video this image applies, "
                                    "as a fraction of its duration (0.0 = start, 1.0 = end).",
                                ),
                            ],
                        ),
                    ],
                    tooltip="How to place this image on the output video's timeline.",
                ),
                IO.Custom(RunwayAleph2IO.PROMPT_IMAGE).Input(
                    "prompt_images",
                    optional=True,
                    tooltip="Optional earlier prompt images to chain with this one.",
                ),
            ],
            outputs=[IO.Custom(RunwayAleph2IO.PROMPT_IMAGE).Output(display_name="prompt_images")],
        )

    @classmethod
    def execute(
        cls,
        image: Input.Image,
        position: dict,
        prompt_images: RunwayAleph2PromptImageChain | None = None,
    ) -> IO.NodeOutput:
        chain = prompt_images.clone() if prompt_images is not None else RunwayAleph2PromptImageChain()
        if position["position"] == _TIMING_ABSOLUTE:
            mode, value = PROMPT_IMAGE_MODE_TIMESTAMP, float(position["seconds"])
        else:
            mode, value = PROMPT_IMAGE_MODE_POSITION, float(position["fraction"])
        chain.add(RunwayAleph2PromptImageItem(image=image, mode=mode, value=value))
        return IO.NodeOutput(chain)


class RunwayAleph2VideoToVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RunwayAleph2VideoToVideoNode",
            display_name="Runway Aleph2 Video to Video",
            category="partner/video/Runway",
            description="Edit a video with a text prompt using Runway's Aleph2 model. Aleph2 transforms "
            "your footage (restyle, relight, add or remove elements, change the viewpoint) while keeping "
            "the original motion and timing; the output resolution matches the input video, which must be "
            "2-30 seconds at 30 fps or lower. Optionally steer the edit with either keyframes (anchored to "
            "the input video) or prompt images (anchored to the output video) - use one or the other, not both.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Describes what should appear in the output (1-1000 characters).",
                ),
                IO.Video.Input(
                    "video",
                    tooltip="Input video to edit. Must be 2-30 seconds at 30 fps or lower.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=4294967295,
                    step=1,
                    control_after_generate=True,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Random seed for generation",
                ),
                IO.Combo.Input(
                    "public_figure_threshold",
                    options=["auto", "low"],
                    default="low",
                    tooltip="Content moderation for recognizable public figures.",
                ),
                IO.Custom(RunwayAleph2IO.KEYFRAME).Input(
                    "keyframes",
                    optional=True,
                    tooltip="Guidance images anchored to the input video, from Aleph2 Keyframe nodes (up to 5). "
                    "Use keyframes or prompt images, not both.",
                ),
                IO.Custom(RunwayAleph2IO.PROMPT_IMAGE).Input(
                    "prompt_images",
                    optional=True,
                    tooltip="Guidance images anchored to the output video, from Aleph2 Prompt Image nodes (up to 5). "
                    "Use keyframes or prompt images, not both.",
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
                expr="""{"type":"usd","usd": 0.4004, "format":{"suffix":"/second"}}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        video: Input.Video,
        seed: int,
        public_figure_threshold: str = "low",
        keyframes: RunwayAleph2KeyframeChain | None = None,
        prompt_images: RunwayAleph2PromptImageChain | None = None,
    ) -> IO.NodeOutput:
        validate_string(prompt, min_length=1, max_length=1000)
        validate_video_duration(
            video,
            min_duration=2.0,
            max_duration=30.0,
        )
        try:
            fps = float(video.get_frame_rate())
        except Exception:
            fps = None
        if fps is not None and fps > 30.0 + 0.01:
            raise ValueError(f"Input video frame rate ({fps:.2f} fps) exceeds Aleph2's maximum of 30 fps.")

        if (keyframes and keyframes.items) and (prompt_images and prompt_images.items):
            raise ValueError("Aleph2 accepts either keyframes or prompt images, not both.")

        video_duration: float | None = None
        try:
            video_duration = video.get_duration()
        except Exception:
            video_duration = None

        def _check_seconds(value: float, label: str) -> None:
            if video_duration is not None and value > video_duration + 0.0001:
                raise ValueError(f"{label} {value:.2f}s exceeds the input video duration ({video_duration:.2f}s).")

        video_url = await upload_video_to_comfyapi(cls, video)

        keyframe_models: list[RunwayAleph2KeyframeSeconds | RunwayAleph2KeyframeAt] = []
        if keyframes is not None:
            if len(keyframes.items) > 5:
                raise ValueError("Aleph2 supports at most 5 keyframes.")
            for item in keyframes.items:
                image_url = await upload_image_to_comfyapi(cls, item.image, mime_type="image/png")
                if item.mode == KEYFRAME_MODE_SECONDS:
                    _check_seconds(item.value, "Keyframe timestamp")
                    keyframe_models.append(RunwayAleph2KeyframeSeconds(seconds=item.value, uri=image_url))
                else:
                    keyframe_models.append(RunwayAleph2KeyframeAt(at=item.value, uri=image_url))

        prompt_image_models: list[RunwayAleph2PromptImage] = []
        if prompt_images is not None:
            if len(prompt_images.items) > 5:
                raise ValueError("Aleph2 supports at most 5 prompt images.")
            for item in prompt_images.items:
                image_url = await upload_image_to_comfyapi(cls, item.image, mime_type="image/png")
                position: RunwayAleph2TimestampPosition | RunwayAleph2RelativePosition
                if item.mode == PROMPT_IMAGE_MODE_TIMESTAMP:
                    _check_seconds(item.value, "Prompt image timestamp")
                    position = RunwayAleph2TimestampPosition(timestampSeconds=item.value)
                else:
                    position = RunwayAleph2RelativePosition(positionPercentage=item.value)
                prompt_image_models.append(RunwayAleph2PromptImage(position=position, uri=image_url))

        initial_response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path=PATH_VIDEO_TO_VIDEO, method="POST"),
            response_model=RunwayAleph2Response,
            data=RunwayAleph2Request(
                promptText=prompt,
                videoUri=video_url,
                seed=seed,
                contentModeration=RunwayAleph2ContentModeration(publicFigureThreshold=public_figure_threshold),
                keyframes=keyframe_models or None,
                promptImage=prompt_image_models or None,
            ),
        )

        final_response = await get_response(cls, initial_response.id)
        if not final_response.output:
            raise ValueError("Runway task succeeded but no video data found in response.")

        return IO.NodeOutput(await download_url_to_video_output(get_video_url_from_task_status(final_response)))


class RunwayExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            RunwayFirstLastFrameNode,
            RunwayImageToVideoNodeGen3a,
            RunwayImageToVideoNodeGen4,
            RunwayTextToImageNode,
            RunwayAleph2VideoToVideoNode,
            RunwayAleph2KeyframeNode,
            RunwayAleph2PromptImageNode,
        ]


async def comfy_entrypoint() -> RunwayExtension:
    return RunwayExtension()
