from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.vidu import (
    FrameSetting,
    SubjectReference,
    TaskCreationRequest,
    TaskCreationResponse,
    TaskExtendCreationRequest,
    TaskMultiFrameCreationRequest,
    TaskResult,
    TaskStatusResponse,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_video_output,
    get_number_of_images,
    poll_op,
    sync_op,
    upload_image_to_comfyapi,
    upload_images_to_comfyapi,
    upload_video_to_comfyapi,
    validate_image_aspect_ratio,
    validate_image_dimensions,
    validate_images_aspect_ratio_closeness,
    validate_string,
    validate_video_duration,
)

VIDU_TEXT_TO_VIDEO = "/proxy/vidu/text2video"
VIDU_IMAGE_TO_VIDEO = "/proxy/vidu/img2video"
VIDU_REFERENCE_VIDEO = "/proxy/vidu/reference2video"
VIDU_START_END_VIDEO = "/proxy/vidu/start-end2video"
VIDU_GET_GENERATION_STATUS = "/proxy/vidu/tasks/%s/creations"


async def execute_task(
    cls: type[IO.ComfyNode],
    vidu_endpoint: str,
    payload: TaskCreationRequest | TaskExtendCreationRequest | TaskMultiFrameCreationRequest,
    max_poll_attempts: int = 480,
) -> list[TaskResult]:
    task_creation_response = await sync_op(
        cls,
        endpoint=ApiEndpoint(path=vidu_endpoint, method="POST"),
        response_model=TaskCreationResponse,
        data=payload,
    )
    if task_creation_response.state == "failed":
        raise RuntimeError(f"Vidu request failed. Code: {task_creation_response.code}")
    response = await poll_op(
        cls,
        ApiEndpoint(path=VIDU_GET_GENERATION_STATUS % task_creation_response.task_id),
        response_model=TaskStatusResponse,
        status_extractor=lambda r: r.state,
        progress_extractor=lambda r: r.progress,
        price_extractor=lambda r: r.credits * 0.005 if r.credits is not None else None,
        max_poll_attempts=max_poll_attempts,
    )
    if not response.creations:
        raise RuntimeError(
            f"Vidu request does not contain results. State: {response.state}, Error Code: {response.err_code}"
        )
    return response.creations


class ViduTextToVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ViduTextToVideoNode",
            display_name="Vidu Text To Video Generation",
            category="partner/video/Vidu",
            description="Generate video from a text prompt",
            inputs=[
                IO.Combo.Input("model", options=["viduq1"], tooltip="Model name"),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="A textual description for video generation",
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=5,
                    max=5,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Duration of the output video in seconds",
                    optional=True,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed for video generation (0 for random)",
                    optional=True,
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["16:9", "9:16", "1:1"],
                    tooltip="The aspect ratio of the output video",
                    optional=True,
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["1080p"],
                    tooltip="Supported values may vary by model & duration",
                    optional=True,
                    advanced=True,
                ),
                IO.Combo.Input(
                    "movement_amplitude",
                    options=["auto", "small", "medium", "large"],
                    tooltip="The movement amplitude of objects in the frame",
                    optional=True,
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
                expr="""{"type":"usd","usd":0.4}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        duration: int,
        seed: int,
        aspect_ratio: str,
        resolution: str,
        movement_amplitude: str,
    ) -> IO.NodeOutput:
        if not prompt:
            raise ValueError("The prompt field is required and cannot be empty.")
        payload = TaskCreationRequest(
            model=model,
            prompt=prompt,
            duration=duration,
            seed=seed,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            movement_amplitude=movement_amplitude,
        )
        results = await execute_task(cls, VIDU_TEXT_TO_VIDEO, payload)
        return IO.NodeOutput(await download_url_to_video_output(results[0].url))


class ViduImageToVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ViduImageToVideoNode",
            display_name="Vidu Image To Video Generation",
            category="partner/video/Vidu",
            description="Generate video from image and optional prompt",
            inputs=[
                IO.Combo.Input("model", options=["viduq1"], tooltip="Model name"),
                IO.Image.Input(
                    "image",
                    tooltip="An image to be used as the start frame of the generated video",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="A textual description for video generation",
                    optional=True,
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=5,
                    max=5,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Duration of the output video in seconds",
                    optional=True,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed for video generation (0 for random)",
                    optional=True,
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["1080p"],
                    tooltip="Supported values may vary by model & duration",
                    optional=True,
                    advanced=True,
                ),
                IO.Combo.Input(
                    "movement_amplitude",
                    options=["auto", "small", "medium", "large"],
                    tooltip="The movement amplitude of objects in the frame",
                    optional=True,
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
                expr="""{"type":"usd","usd":0.4}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        image: Input.Image,
        prompt: str,
        duration: int,
        seed: int,
        resolution: str,
        movement_amplitude: str,
    ) -> IO.NodeOutput:
        if get_number_of_images(image) > 1:
            raise ValueError("Only one input image is allowed.")
        validate_image_aspect_ratio(image, (1, 4), (4, 1))
        payload = TaskCreationRequest(
            model=model,
            prompt=prompt,
            duration=duration,
            seed=seed,
            resolution=resolution,
            movement_amplitude=movement_amplitude,
        )
        payload.images = await upload_images_to_comfyapi(
            cls,
            image,
            max_images=1,
            mime_type="image/png",
        )
        results = await execute_task(cls, VIDU_IMAGE_TO_VIDEO, payload)
        return IO.NodeOutput(await download_url_to_video_output(results[0].url))


class ViduReferenceVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ViduReferenceVideoNode",
            display_name="Vidu Reference To Video Generation",
            category="partner/video/Vidu",
            description="Generate video from multiple images and a prompt",
            inputs=[
                IO.Combo.Input("model", options=["viduq1"], tooltip="Model name"),
                IO.Image.Input(
                    "images",
                    tooltip="Images to use as references to generate a video with consistent subjects (max 7 images).",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="A textual description for video generation",
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=5,
                    max=5,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Duration of the output video in seconds",
                    optional=True,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed for video generation (0 for random)",
                    optional=True,
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["16:9", "9:16", "1:1"],
                    tooltip="The aspect ratio of the output video",
                    optional=True,
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["1080p"],
                    tooltip="Supported values may vary by model & duration",
                    optional=True,
                    advanced=True,
                ),
                IO.Combo.Input(
                    "movement_amplitude",
                    options=["auto", "small", "medium", "large"],
                    tooltip="The movement amplitude of objects in the frame",
                    optional=True,
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
                expr="""{"type":"usd","usd":0.4}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        images: Input.Image,
        prompt: str,
        duration: int,
        seed: int,
        aspect_ratio: str,
        resolution: str,
        movement_amplitude: str,
    ) -> IO.NodeOutput:
        if not prompt:
            raise ValueError("The prompt field is required and cannot be empty.")
        a = get_number_of_images(images)
        if a > 7:
            raise ValueError("Too many images, maximum allowed is 7.")
        for image in images:
            validate_image_aspect_ratio(image, (1, 4), (4, 1))
            validate_image_dimensions(image, min_width=128, min_height=128)
        payload = TaskCreationRequest(
            model=model,
            prompt=prompt,
            duration=duration,
            seed=seed,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            movement_amplitude=movement_amplitude,
        )
        payload.images = await upload_images_to_comfyapi(
            cls,
            images,
            max_images=7,
            mime_type="image/png",
        )
        results = await execute_task(cls, VIDU_REFERENCE_VIDEO, payload)
        return IO.NodeOutput(await download_url_to_video_output(results[0].url))


class ViduStartEndToVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ViduStartEndToVideoNode",
            display_name="Vidu Start End To Video Generation",
            category="partner/video/Vidu",
            description="Generate a video from start and end frames and a prompt",
            inputs=[
                IO.Combo.Input("model", options=["viduq1"], tooltip="Model name"),
                IO.Image.Input(
                    "first_frame",
                    tooltip="Start frame",
                ),
                IO.Image.Input(
                    "end_frame",
                    tooltip="End frame",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="A textual description for video generation",
                    optional=True,
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=5,
                    max=5,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Duration of the output video in seconds",
                    optional=True,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed for video generation (0 for random)",
                    optional=True,
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["1080p"],
                    tooltip="Supported values may vary by model & duration",
                    optional=True,
                    advanced=True,
                ),
                IO.Combo.Input(
                    "movement_amplitude",
                    options=["auto", "small", "medium", "large"],
                    tooltip="The movement amplitude of objects in the frame",
                    optional=True,
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
                expr="""{"type":"usd","usd":0.4}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        first_frame: Input.Image,
        end_frame: Input.Image,
        prompt: str,
        duration: int,
        seed: int,
        resolution: str,
        movement_amplitude: str,
    ) -> IO.NodeOutput:
        validate_images_aspect_ratio_closeness(first_frame, end_frame, min_rel=0.8, max_rel=1.25, strict=False)
        payload = TaskCreationRequest(
            model=model,
            prompt=prompt,
            duration=duration,
            seed=seed,
            resolution=resolution,
            movement_amplitude=movement_amplitude,
        )
        payload.images = [
            (await upload_images_to_comfyapi(cls, frame, max_images=1, mime_type="image/png"))[0]
            for frame in (first_frame, end_frame)
        ]
        results = await execute_task(cls, VIDU_START_END_VIDEO, payload)
        return IO.NodeOutput(await download_url_to_video_output(results[0].url))


class Vidu2TextToVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Vidu2TextToVideoNode",
            display_name="Vidu2 Text-to-Video Generation",
            category="partner/video/Vidu",
            description="Generate video from a text prompt",
            inputs=[
                IO.Combo.Input("model", options=["viduq2"]),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="A textual description for video generation, with a maximum length of 2000 characters.",
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=1,
                    max=10,
                    step=1,
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Int.Input(
                    "seed",
                    default=1,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                ),
                IO.Combo.Input("aspect_ratio", options=["16:9", "9:16", "3:4", "4:3", "1:1"]),
                IO.Combo.Input("resolution", options=["720p", "1080p"], advanced=True),
                IO.Boolean.Input(
                    "background_music",
                    default=False,
                    tooltip="Whether to add background music to the generated video.",
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
                depends_on=IO.PriceBadgeDepends(widgets=["duration", "resolution"]),
                expr="""
                (
                  $is1080 := widgets.resolution = "1080p";
                  $base := $is1080 ? 0.1 : 0.075;
                  $perSec := $is1080 ? 0.05 : 0.025;
                  {"type":"usd","usd": $base + $perSec * (widgets.duration - 1)}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        duration: int,
        seed: int,
        aspect_ratio: str,
        resolution: str,
        background_music: bool,
    ) -> IO.NodeOutput:
        validate_string(prompt, min_length=1, max_length=2000)
        results = await execute_task(
            cls,
            VIDU_TEXT_TO_VIDEO,
            TaskCreationRequest(
                model=model,
                prompt=prompt,
                duration=duration,
                seed=seed,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                bgm=background_music,
            ),
        )
        return IO.NodeOutput(await download_url_to_video_output(results[0].url))


class Vidu2ImageToVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Vidu2ImageToVideoNode",
            display_name="Vidu2 Image-to-Video Generation",
            category="partner/video/Vidu",
            description="Generate a video from an image and an optional prompt.",
            inputs=[
                IO.Combo.Input("model", options=["viduq2-pro-fast", "viduq2-pro", "viduq2-turbo"]),
                IO.Image.Input(
                    "image",
                    tooltip="An image to be used as the start frame of the generated video.",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="An optional text prompt for video generation (max 2000 characters).",
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=1,
                    max=10,
                    step=1,
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Int.Input(
                    "seed",
                    default=1,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["720p", "1080p"],
                    advanced=True,
                ),
                IO.Combo.Input(
                    "movement_amplitude",
                    options=["auto", "small", "medium", "large"],
                    tooltip="The movement amplitude of objects in the frame.",
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
                depends_on=IO.PriceBadgeDepends(widgets=["model", "duration", "resolution"]),
                expr="""
                (
                  $m := widgets.model;
                  $d := widgets.duration;
                  $is1080 := widgets.resolution = "1080p";
                  $contains($m, "pro-fast")
                    ? (
                        $base := $is1080 ? 0.08 : 0.04;
                        $perSec := $is1080 ? 0.02 : 0.01;
                        {"type":"usd","usd": $base + $perSec * ($d - 1)}
                      )
                    : $contains($m, "pro")
                      ? (
                          $base := $is1080 ? 0.275 : 0.075;
                          $perSec := $is1080 ? 0.075 : 0.05;
                          {"type":"usd","usd": $base + $perSec * ($d - 1)}
                        )
                      : $contains($m, "turbo")
                        ? (
                            $is1080
                              ? {"type":"usd","usd": 0.175 + 0.05 * ($d - 1)}
                              : (
                                  $d <= 1 ? {"type":"usd","usd": 0.04}
                                  : $d <= 2 ? {"type":"usd","usd": 0.05}
                                  : {"type":"usd","usd": 0.05 + 0.05 * ($d - 2)}
                                )
                          )
                        : {"type":"usd","usd": 0.04}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        image: Input.Image,
        prompt: str,
        duration: int,
        seed: int,
        resolution: str,
        movement_amplitude: str,
    ) -> IO.NodeOutput:
        if get_number_of_images(image) > 1:
            raise ValueError("Only one input image is allowed.")
        validate_image_aspect_ratio(image, (1, 4), (4, 1))
        validate_string(prompt, max_length=2000)
        results = await execute_task(
            cls,
            VIDU_IMAGE_TO_VIDEO,
            TaskCreationRequest(
                model=model,
                prompt=prompt,
                duration=duration,
                seed=seed,
                resolution=resolution,
                movement_amplitude=movement_amplitude,
                images=await upload_images_to_comfyapi(
                    cls,
                    image,
                    max_images=1,
                    mime_type="image/png",
                ),
            ),
        )
        return IO.NodeOutput(await download_url_to_video_output(results[0].url))


class Vidu2ReferenceVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Vidu2ReferenceVideoNode",
            display_name="Vidu2 Reference-to-Video Generation",
            category="partner/video/Vidu",
            description="Generate a video from multiple reference images and a prompt.",
            inputs=[
                IO.Combo.Input("model", options=["viduq2"]),
                IO.Autogrow.Input(
                    "subjects",
                    template=IO.Autogrow.TemplateNames(
                        IO.Image.Input("reference_images"),
                        names=["subject1", "subject2", "subject3", "subject4", "subject5", "subject6", "subject7"],
                        min=1,
                    ),
                    tooltip="For each subject, provide up to 3 reference images (7 images total across all subjects). "
                    "Reference them in prompts via @subject{subject_id}.",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="When enabled, the video will include generated speech and background music "
                    "based on the prompt.",
                ),
                IO.Boolean.Input(
                    "audio",
                    default=False,
                    tooltip="When enabled video will contain generated speech and background music based on the prompt.",
                    advanced=True,
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=1,
                    max=10,
                    step=1,
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Int.Input(
                    "seed",
                    default=1,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                ),
                IO.Combo.Input("aspect_ratio", options=["16:9", "9:16", "4:3", "3:4", "1:1"]),
                IO.Combo.Input("resolution", options=["720p", "1080p"], advanced=True),
                IO.Combo.Input(
                    "movement_amplitude",
                    options=["auto", "small", "medium", "large"],
                    tooltip="The movement amplitude of objects in the frame.",
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
                depends_on=IO.PriceBadgeDepends(widgets=["audio", "duration", "resolution"]),
                expr="""
                (
                  $is1080 := widgets.resolution = "1080p";
                  $base := $is1080 ? 0.375 : 0.125;
                  $perSec := $is1080 ? 0.05 : 0.025;
                  $audioCost := widgets.audio = true ? 0.075 : 0;
                  {"type":"usd","usd": $base + $perSec * (widgets.duration - 1) + $audioCost}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        subjects: IO.Autogrow.Type,
        prompt: str,
        audio: bool,
        duration: int,
        seed: int,
        aspect_ratio: str,
        resolution: str,
        movement_amplitude: str,
    ) -> IO.NodeOutput:
        validate_string(prompt, min_length=1, max_length=2000)
        total_images = 0
        for i in subjects:
            if get_number_of_images(subjects[i]) > 3:
                raise ValueError("Maximum number of images per subject is 3.")
            for im in subjects[i]:
                total_images += 1
                validate_image_aspect_ratio(im, (1, 4), (4, 1))
                validate_image_dimensions(im, min_width=128, min_height=128)
        if total_images > 7:
            raise ValueError("Too many reference images; the maximum allowed is 7.")
        subjects_param: list[SubjectReference] = []
        for i in subjects:
            subjects_param.append(
                SubjectReference(
                    id=i,
                    images=await upload_images_to_comfyapi(
                        cls,
                        subjects[i],
                        max_images=3,
                        mime_type="image/png",
                        wait_label=f"Uploading reference images for {i}",
                    ),
                ),
            )
        payload = TaskCreationRequest(
            model=model,
            prompt=prompt,
            audio=audio,
            duration=duration,
            seed=seed,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            movement_amplitude=movement_amplitude,
            subjects=subjects_param,
        )
        results = await execute_task(cls, VIDU_REFERENCE_VIDEO, payload)
        return IO.NodeOutput(await download_url_to_video_output(results[0].url))


class Vidu2StartEndToVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Vidu2StartEndToVideoNode",
            display_name="Vidu2 Start/End Frame-to-Video Generation",
            category="partner/video/Vidu",
            description="Generate a video from a start frame, an end frame, and a prompt.",
            inputs=[
                IO.Combo.Input("model", options=["viduq2-pro-fast", "viduq2-pro", "viduq2-turbo"]),
                IO.Image.Input("first_frame"),
                IO.Image.Input("end_frame"),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="Prompt description (max 2000 characters).",
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=2,
                    max=8,
                    step=1,
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Int.Input(
                    "seed",
                    default=1,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                ),
                IO.Combo.Input("resolution", options=["720p", "1080p"], advanced=True),
                IO.Combo.Input(
                    "movement_amplitude",
                    options=["auto", "small", "medium", "large"],
                    tooltip="The movement amplitude of objects in the frame.",
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
                depends_on=IO.PriceBadgeDepends(widgets=["model", "duration", "resolution"]),
                expr="""
                (
                  $m := widgets.model;
                  $d := widgets.duration;
                  $is1080 := widgets.resolution = "1080p";
                  $contains($m, "pro-fast")
                    ? (
                        $base := $is1080 ? 0.08 : 0.04;
                        $perSec := $is1080 ? 0.02 : 0.01;
                        {"type":"usd","usd": $base + $perSec * ($d - 1)}
                      )
                    : $contains($m, "pro")
                      ? (
                          $base := $is1080 ? 0.275 : 0.075;
                          $perSec := $is1080 ? 0.075 : 0.05;
                          {"type":"usd","usd": $base + $perSec * ($d - 1)}
                        )
                      : $contains($m, "turbo")
                        ? (
                            $is1080
                              ? {"type":"usd","usd": 0.175 + 0.05 * ($d - 1)}
                              : (
                                  $d <= 2 ? {"type":"usd","usd": 0.05}
                                  : {"type":"usd","usd": 0.05 + 0.05 * ($d - 2)}
                                )
                          )
                        : {"type":"usd","usd": 0.04}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        first_frame: Input.Image,
        end_frame: Input.Image,
        prompt: str,
        duration: int,
        seed: int,
        resolution: str,
        movement_amplitude: str,
    ) -> IO.NodeOutput:
        validate_string(prompt, max_length=2000)
        if get_number_of_images(first_frame) > 1:
            raise ValueError("Only one input image is allowed for `first_frame`.")
        if get_number_of_images(end_frame) > 1:
            raise ValueError("Only one input image is allowed for `end_frame`.")
        validate_images_aspect_ratio_closeness(first_frame, end_frame, min_rel=0.8, max_rel=1.25, strict=False)
        payload = TaskCreationRequest(
            model=model,
            prompt=prompt,
            duration=duration,
            seed=seed,
            resolution=resolution,
            movement_amplitude=movement_amplitude,
            images=[
                (await upload_images_to_comfyapi(cls, frame, max_images=1, mime_type="image/png"))[0]
                for frame in (first_frame, end_frame)
            ],
        )
        results = await execute_task(cls, VIDU_START_END_VIDEO, payload)
        return IO.NodeOutput(await download_url_to_video_output(results[0].url))


class ViduExtendVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ViduExtendVideoNode",
            display_name="Vidu Video Extension",
            category="partner/video/Vidu",
            description="Extend an existing video by generating additional frames.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "viduq2-pro",
                            [
                                IO.Int.Input(
                                    "duration",
                                    default=4,
                                    min=1,
                                    max=7,
                                    step=1,
                                    display_mode=IO.NumberDisplay.slider,
                                    tooltip="Duration of the extended video in seconds.",
                                ),
                                IO.Combo.Input(
                                    "resolution",
                                    options=["720p", "1080p"],
                                    tooltip="Resolution of the output video.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "viduq2-turbo",
                            [
                                IO.Int.Input(
                                    "duration",
                                    default=4,
                                    min=1,
                                    max=7,
                                    step=1,
                                    display_mode=IO.NumberDisplay.slider,
                                    tooltip="Duration of the extended video in seconds.",
                                ),
                                IO.Combo.Input(
                                    "resolution",
                                    options=["720p", "1080p"],
                                    tooltip="Resolution of the output video.",
                                ),
                            ],
                        ),
                    ],
                    tooltip="Model to use for video extension.",
                ),
                IO.Video.Input(
                    "video",
                    tooltip="The source video to extend.",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="An optional text prompt for the extended video (max 2000 characters).",
                ),
                IO.Int.Input(
                    "seed",
                    default=1,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                ),
                IO.Image.Input("end_frame", optional=True),
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
                depends_on=IO.PriceBadgeDepends(widgets=["model", "model.duration", "model.resolution"]),
                expr="""
                (
                  $m := widgets.model;
                  $d := $lookup(widgets, "model.duration");
                  $res := $lookup(widgets, "model.resolution");
                  $contains($m, "pro")
                    ? (
                        $base := $lookup({"720p": 0.15, "1080p": 0.3}, $res);
                        $perSec := $lookup({"720p": 0.05, "1080p": 0.075}, $res);
                        {"type":"usd","usd": $base + $perSec * ($d - 1)}
                      )
                    : (
                        $base := $lookup({"720p": 0.075, "1080p": 0.2}, $res);
                        $perSec := $lookup({"720p": 0.025, "1080p": 0.05}, $res);
                        {"type":"usd","usd": $base + $perSec * ($d - 1)}
                      )
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: dict,
        video: Input.Video,
        prompt: str,
        seed: int,
        end_frame: Input.Image | None = None,
    ) -> IO.NodeOutput:
        validate_string(prompt, max_length=2000)
        validate_video_duration(video, min_duration=4, max_duration=55)
        image_url = None
        if end_frame is not None:
            validate_image_aspect_ratio(end_frame, (1, 4), (4, 1))
            validate_image_dimensions(end_frame, min_width=128, min_height=128)
            image_url = await upload_image_to_comfyapi(cls, end_frame, wait_label="Uploading end frame")
        results = await execute_task(
            cls,
            "/proxy/vidu/extend",
            TaskExtendCreationRequest(
                model=model["model"],
                prompt=prompt,
                duration=model["duration"],
                seed=seed,
                resolution=model["resolution"],
                video_url=await upload_video_to_comfyapi(cls, video, wait_label="Uploading video"),
                images=[image_url] if image_url else None,
            ),
        )
        return IO.NodeOutput(await download_url_to_video_output(results[0].url))


def _generate_frame_inputs(count: int) -> list:
    """Generate input widgets for a given number of frames."""
    inputs = []
    for i in range(1, count + 1):
        inputs.extend(
            [
                IO.String.Input(
                    f"prompt{i}",
                    multiline=True,
                    default="",
                    tooltip=f"Text prompt for frame {i} transition.",
                ),
                IO.Image.Input(
                    f"end_image{i}",
                    tooltip=f"End frame image for segment {i}. Aspect ratio must be between 1:4 and 4:1.",
                ),
                IO.Int.Input(
                    f"duration{i}",
                    default=4,
                    min=2,
                    max=7,
                    step=1,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip=f"Duration for segment {i} in seconds.",
                ),
            ]
        )
    return inputs


class ViduMultiFrameVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ViduMultiFrameVideoNode",
            display_name="Vidu Multi-Frame Video Generation",
            category="partner/video/Vidu",
            description="Generate a video with multiple keyframe transitions.",
            inputs=[
                IO.Combo.Input("model", options=["viduq2-pro", "viduq2-turbo"]),
                IO.Image.Input(
                    "start_image",
                    tooltip="The starting frame image. Aspect ratio must be between 1:4 and 4:1.",
                ),
                IO.Int.Input(
                    "seed",
                    default=1,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                ),
                IO.Combo.Input("resolution", options=["720p", "1080p"]),
                IO.DynamicCombo.Input(
                    "frames",
                    options=[
                        IO.DynamicCombo.Option("2", _generate_frame_inputs(2)),
                        IO.DynamicCombo.Option("3", _generate_frame_inputs(3)),
                        IO.DynamicCombo.Option("4", _generate_frame_inputs(4)),
                        IO.DynamicCombo.Option("5", _generate_frame_inputs(5)),
                        IO.DynamicCombo.Option("6", _generate_frame_inputs(6)),
                        IO.DynamicCombo.Option("7", _generate_frame_inputs(7)),
                        IO.DynamicCombo.Option("8", _generate_frame_inputs(8)),
                        IO.DynamicCombo.Option("9", _generate_frame_inputs(9)),
                    ],
                    tooltip="Number of keyframe transitions (2-9).",
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
                    widgets=[
                        "model",
                        "resolution",
                        "frames",
                        "frames.duration1",
                        "frames.duration2",
                        "frames.duration3",
                        "frames.duration4",
                        "frames.duration5",
                        "frames.duration6",
                        "frames.duration7",
                        "frames.duration8",
                        "frames.duration9",
                    ]
                ),
                expr="""
                (
                  $m := widgets.model;
                  $n := $number(widgets.frames);
                  $is1080 := widgets.resolution = "1080p";
                  $d1 := $lookup(widgets, "frames.duration1");
                  $d2 := $lookup(widgets, "frames.duration2");
                  $d3 := $n >= 3 ? $lookup(widgets, "frames.duration3") : 0;
                  $d4 := $n >= 4 ? $lookup(widgets, "frames.duration4") : 0;
                  $d5 := $n >= 5 ? $lookup(widgets, "frames.duration5") : 0;
                  $d6 := $n >= 6 ? $lookup(widgets, "frames.duration6") : 0;
                  $d7 := $n >= 7 ? $lookup(widgets, "frames.duration7") : 0;
                  $d8 := $n >= 8 ? $lookup(widgets, "frames.duration8") : 0;
                  $d9 := $n >= 9 ? $lookup(widgets, "frames.duration9") : 0;
                  $totalDuration := $d1 + $d2 + $d3 + $d4 + $d5 + $d6 + $d7 + $d8 + $d9;
                  $contains($m, "pro")
                    ? (
                        $base := $is1080 ? 0.3 : 0.15;
                        $perSec := $is1080 ? 0.075 : 0.05;
                        {"type":"usd","usd": $n * $base + $perSec * $totalDuration}
                      )
                    : (
                        $base := $is1080 ? 0.2 : 0.075;
                        $perSec := $is1080 ? 0.05 : 0.025;
                        {"type":"usd","usd": $n * $base + $perSec * $totalDuration}
                      )
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        start_image: Input.Image,
        seed: int,
        resolution: str,
        frames: dict,
    ) -> IO.NodeOutput:
        validate_image_aspect_ratio(start_image, (1, 4), (4, 1))
        frame_count = int(frames["frames"])
        image_settings: list[FrameSetting] = []
        for i in range(1, frame_count + 1):
            validate_image_aspect_ratio(frames[f"end_image{i}"], (1, 4), (4, 1))
            validate_string(frames[f"prompt{i}"], max_length=2000)
        start_image_url = await upload_image_to_comfyapi(
            cls,
            start_image,
            mime_type="image/png",
            wait_label="Uploading start image",
        )
        for i in range(1, frame_count + 1):
            image_settings.append(
                FrameSetting(
                    prompt=frames[f"prompt{i}"],
                    key_image=await upload_image_to_comfyapi(
                        cls,
                        frames[f"end_image{i}"],
                        mime_type="image/png",
                        wait_label=f"Uploading end image({i})",
                    ),
                    duration=frames[f"duration{i}"],
                )
            )
        results = await execute_task(
            cls,
            "/proxy/vidu/multiframe",
            TaskMultiFrameCreationRequest(
                model=model,
                seed=seed,
                resolution=resolution,
                start_image=start_image_url,
                image_settings=image_settings,
            ),
            max_poll_attempts=480 * frame_count,
        )
        return IO.NodeOutput(await download_url_to_video_output(results[0].url))


class Vidu3TextToVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Vidu3TextToVideoNode",
            display_name="Vidu Q3 Text-to-Video Generation",
            category="partner/video/Vidu",
            description="Generate video from a text prompt.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "viduq3-pro",
                            [
                                IO.Combo.Input(
                                    "aspect_ratio",
                                    options=["16:9", "9:16", "3:4", "4:3", "1:1"],
                                    tooltip="The aspect ratio of the output video.",
                                ),
                                IO.Combo.Input(
                                    "resolution",
                                    options=["720p", "1080p"],
                                    tooltip="Resolution of the output video.",
                                ),
                                IO.Int.Input(
                                    "duration",
                                    default=5,
                                    min=1,
                                    max=16,
                                    step=1,
                                    display_mode=IO.NumberDisplay.slider,
                                    tooltip="Duration of the output video in seconds.",
                                ),
                                IO.Boolean.Input(
                                    "audio",
                                    default=False,
                                    tooltip="When enabled, outputs video with sound "
                                    "(including dialogue and sound effects).",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "viduq3-turbo",
                            [
                                IO.Combo.Input(
                                    "aspect_ratio",
                                    options=["16:9", "9:16", "3:4", "4:3", "1:1"],
                                    tooltip="The aspect ratio of the output video.",
                                ),
                                IO.Combo.Input(
                                    "resolution",
                                    options=["720p", "1080p"],
                                    tooltip="Resolution of the output video.",
                                ),
                                IO.Int.Input(
                                    "duration",
                                    default=5,
                                    min=1,
                                    max=16,
                                    step=1,
                                    display_mode=IO.NumberDisplay.slider,
                                    tooltip="Duration of the output video in seconds.",
                                ),
                                IO.Boolean.Input(
                                    "audio",
                                    default=False,
                                    tooltip="When enabled, outputs video with sound "
                                    "(including dialogue and sound effects).",
                                ),
                            ],
                        ),
                    ],
                    tooltip="Model to use for video generation.",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="A textual description for video generation, with a maximum length of 2000 characters.",
                ),
                IO.Int.Input(
                    "seed",
                    default=1,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
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
                depends_on=IO.PriceBadgeDepends(widgets=["model", "model.duration", "model.resolution"]),
                expr="""
                (
                  $res := $lookup(widgets, "model.resolution");
                  $d := $lookup(widgets, "model.duration");
                  $contains(widgets.model, "turbo")
                    ? (
                        $rate := $lookup({"720p": 0.06, "1080p": 0.08}, $res);
                        {"type":"usd","usd": $rate * $d}
                      )
                    : (
                        $rate := $lookup({"720p": 0.15, "1080p": 0.16}, $res);
                        {"type":"usd","usd": $rate * $d}
                      )
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: dict,
        prompt: str,
        seed: int,
    ) -> IO.NodeOutput:
        validate_string(prompt, min_length=1, max_length=2000)
        results = await execute_task(
            cls,
            VIDU_TEXT_TO_VIDEO,
            TaskCreationRequest(
                model=model["model"],
                prompt=prompt,
                duration=model["duration"],
                seed=seed,
                aspect_ratio=model["aspect_ratio"],
                resolution=model["resolution"],
                audio=model["audio"],
            ),
            max_poll_attempts=640,
        )
        return IO.NodeOutput(await download_url_to_video_output(results[0].url))


class Vidu3ImageToVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Vidu3ImageToVideoNode",
            display_name="Vidu Q3 Image-to-Video Generation",
            category="partner/video/Vidu",
            description="Generate a video from an image and an optional prompt.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "viduq3-pro",
                            [
                                IO.Combo.Input(
                                    "resolution",
                                    options=["720p", "1080p", "2K"],
                                    tooltip="Resolution of the output video.",
                                ),
                                IO.Int.Input(
                                    "duration",
                                    default=5,
                                    min=1,
                                    max=16,
                                    step=1,
                                    display_mode=IO.NumberDisplay.slider,
                                    tooltip="Duration of the output video in seconds.",
                                ),
                                IO.Boolean.Input(
                                    "audio",
                                    default=False,
                                    tooltip="When enabled, outputs video with sound "
                                    "(including dialogue and sound effects).",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "viduq3-turbo",
                            [
                                IO.Combo.Input(
                                    "resolution",
                                    options=["720p", "1080p"],
                                    tooltip="Resolution of the output video.",
                                ),
                                IO.Int.Input(
                                    "duration",
                                    default=5,
                                    min=1,
                                    max=16,
                                    step=1,
                                    display_mode=IO.NumberDisplay.slider,
                                    tooltip="Duration of the output video in seconds.",
                                ),
                                IO.Boolean.Input(
                                    "audio",
                                    default=False,
                                    tooltip="When enabled, outputs video with sound "
                                    "(including dialogue and sound effects).",
                                ),
                            ],
                        ),
                    ],
                    tooltip="Model to use for video generation.",
                ),
                IO.Image.Input(
                    "image",
                    tooltip="An image to be used as the start frame of the generated video.",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="An optional text prompt for video generation (max 2000 characters).",
                ),
                IO.Int.Input(
                    "seed",
                    default=1,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
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
                depends_on=IO.PriceBadgeDepends(widgets=["model", "model.duration", "model.resolution"]),
                expr="""
                (
                  $res := $lookup(widgets, "model.resolution");
                  $d := $lookup(widgets, "model.duration");
                  $contains(widgets.model, "turbo")
                    ? (
                        $rate := $lookup({"720p": 0.06, "1080p": 0.08}, $res);
                        {"type":"usd","usd": $rate * $d}
                      )
                    : (
                        $rate := $lookup({"720p": 0.15, "1080p": 0.16, "2k": 0.2}, $res);
                        {"type":"usd","usd": $rate * $d}
                      )
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: dict,
        image: Input.Image,
        prompt: str,
        seed: int,
    ) -> IO.NodeOutput:
        validate_image_aspect_ratio(image, (1, 4), (4, 1))
        validate_string(prompt, max_length=2000)
        results = await execute_task(
            cls,
            VIDU_IMAGE_TO_VIDEO,
            TaskCreationRequest(
                model=model["model"],
                prompt=prompt,
                duration=model["duration"],
                seed=seed,
                resolution=model["resolution"],
                audio=model["audio"],
                images=[await upload_image_to_comfyapi(cls, image)],
            ),
            max_poll_attempts=720,
        )
        return IO.NodeOutput(await download_url_to_video_output(results[0].url))


class Vidu3StartEndToVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Vidu3StartEndToVideoNode",
            display_name="Vidu Q3 Start/End Frame-to-Video Generation",
            category="partner/video/Vidu",
            description="Generate a video from a start frame, an end frame, and a prompt.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "viduq3-pro",
                            [
                                IO.Combo.Input(
                                    "resolution",
                                    options=["720p", "1080p"],
                                    tooltip="Resolution of the output video.",
                                ),
                                IO.Int.Input(
                                    "duration",
                                    default=5,
                                    min=1,
                                    max=16,
                                    step=1,
                                    display_mode=IO.NumberDisplay.slider,
                                    tooltip="Duration of the output video in seconds.",
                                ),
                                IO.Boolean.Input(
                                    "audio",
                                    default=False,
                                    tooltip="When enabled, outputs video with sound "
                                    "(including dialogue and sound effects).",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "viduq3-turbo",
                            [
                                IO.Combo.Input(
                                    "resolution",
                                    options=["720p", "1080p"],
                                    tooltip="Resolution of the output video.",
                                ),
                                IO.Int.Input(
                                    "duration",
                                    default=5,
                                    min=1,
                                    max=16,
                                    step=1,
                                    display_mode=IO.NumberDisplay.slider,
                                    tooltip="Duration of the output video in seconds.",
                                ),
                                IO.Boolean.Input(
                                    "audio",
                                    default=False,
                                    tooltip="When enabled, outputs video with sound "
                                    "(including dialogue and sound effects).",
                                ),
                            ],
                        ),
                    ],
                    tooltip="Model to use for video generation.",
                ),
                IO.Image.Input("first_frame"),
                IO.Image.Input("end_frame"),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="Prompt description (max 2000 characters).",
                ),
                IO.Int.Input(
                    "seed",
                    default=1,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
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
                depends_on=IO.PriceBadgeDepends(widgets=["model", "model.duration", "model.resolution"]),
                expr="""
                (
                  $res := $lookup(widgets, "model.resolution");
                  $d := $lookup(widgets, "model.duration");
                  $contains(widgets.model, "turbo")
                    ? (
                        $rate := $lookup({"720p": 0.06, "1080p": 0.08}, $res);
                        {"type":"usd","usd": $rate * $d}
                      )
                    : (
                        $rate := $lookup({"720p": 0.15, "1080p": 0.16}, $res);
                        {"type":"usd","usd": $rate * $d}
                      )
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: dict,
        first_frame: Input.Image,
        end_frame: Input.Image,
        prompt: str,
        seed: int,
    ) -> IO.NodeOutput:
        validate_string(prompt, max_length=2000)
        validate_images_aspect_ratio_closeness(first_frame, end_frame, min_rel=0.8, max_rel=1.25, strict=False)
        payload = TaskCreationRequest(
            model=model["model"],
            prompt=prompt,
            duration=model["duration"],
            seed=seed,
            resolution=model["resolution"],
            audio=model["audio"],
            images=[
                (await upload_images_to_comfyapi(cls, frame, max_images=1, mime_type="image/png"))[0]
                for frame in (first_frame, end_frame)
            ],
        )
        results = await execute_task(cls, VIDU_START_END_VIDEO, payload)
        return IO.NodeOutput(await download_url_to_video_output(results[0].url))


class ViduExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            ViduTextToVideoNode,
            ViduImageToVideoNode,
            ViduReferenceVideoNode,
            ViduStartEndToVideoNode,
            Vidu2TextToVideoNode,
            Vidu2ImageToVideoNode,
            Vidu2ReferenceVideoNode,
            Vidu2StartEndToVideoNode,
            ViduExtendVideoNode,
            ViduMultiFrameVideoNode,
            Vidu3TextToVideoNode,
            Vidu3ImageToVideoNode,
            Vidu3StartEndToVideoNode,
        ]


async def comfy_entrypoint() -> ViduExtension:
    return ViduExtension()
