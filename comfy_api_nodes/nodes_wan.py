import re

from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.wan import (
    Image2ImageInputField,
    Image2ImageParametersField,
    Image2ImageTaskCreationRequest,
    Image2VideoInputField,
    Image2VideoParametersField,
    Image2VideoTaskCreationRequest,
    ImageTaskStatusResponse,
    Reference2VideoInputField,
    Reference2VideoParametersField,
    Reference2VideoTaskCreationRequest,
    TaskCreationResponse,
    Text2ImageInputField,
    Text2ImageTaskCreationRequest,
    Text2VideoInputField,
    Text2VideoParametersField,
    Text2VideoTaskCreationRequest,
    Txt2ImageParametersField,
    VideoTaskStatusResponse,
    Wan27ImageToVideoInputField,
    Wan27ImageToVideoParametersField,
    Wan27ImageToVideoTaskCreationRequest,
    Wan27MediaItem,
    Wan27ReferenceVideoInputField,
    Wan27ReferenceVideoParametersField,
    Wan27ReferenceVideoTaskCreationRequest,
    Wan27Text2VideoParametersField,
    Wan27Text2VideoTaskCreationRequest,
    Wan27VideoEditInputField,
    Wan27VideoEditParametersField,
    Wan27VideoEditTaskCreationRequest,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    audio_to_base64_string,
    download_url_to_image_tensor,
    download_url_to_video_output,
    get_number_of_images,
    poll_op,
    sync_op,
    tensor_to_base64_string,
    upload_audio_to_comfyapi,
    upload_image_to_comfyapi,
    upload_video_to_comfyapi,
    validate_audio_duration,
    validate_image_aspect_ratio,
    validate_image_dimensions,
    validate_string,
    validate_video_duration,
)


RES_IN_PARENS = re.compile(r"\((\d+)\s*[x×]\s*(\d+)\)")


class WanTextToImageApi(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="WanTextToImageApi",
            display_name="Wan Text to Image",
            category="partner/image/Wan",
            description="Generates an image based on a text prompt.",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=["wan2.5-t2i-preview"],
                    tooltip="Model to use.",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt describing the elements and visual features. Supports English and Chinese.",
                ),
                IO.String.Input(
                    "negative_prompt",
                    multiline=True,
                    default="",
                    tooltip="Negative prompt describing what to avoid.",
                    optional=True,
                ),
                IO.Int.Input(
                    "width",
                    default=1024,
                    min=768,
                    max=1440,
                    step=32,
                    optional=True,
                ),
                IO.Int.Input(
                    "height",
                    default=1024,
                    min=768,
                    max=1440,
                    step=32,
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
                    tooltip="Seed to use for generation.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "prompt_extend",
                    default=True,
                    tooltip="Whether to enhance the prompt with AI assistance.",
                    optional=True,
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip="Whether to add an AI-generated watermark to the result.",
                    optional=True,
                    advanced=True,
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
                expr="""{"type":"usd","usd":0.03}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        seed: int = 0,
        prompt_extend: bool = True,
        watermark: bool = False,
    ):
        initial_response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/wan/api/v1/services/aigc/text2image/image-synthesis", method="POST"),
            response_model=TaskCreationResponse,
            data=Text2ImageTaskCreationRequest(
                model=model,
                input=Text2ImageInputField(prompt=prompt, negative_prompt=negative_prompt),
                parameters=Txt2ImageParametersField(
                    size=f"{width}*{height}",
                    seed=seed,
                    prompt_extend=prompt_extend,
                    watermark=watermark,
                ),
            ),
        )
        if not initial_response.output:
            raise Exception(f"An unknown error occurred: {initial_response.code} - {initial_response.message}")
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/wan/api/v1/tasks/{initial_response.output.task_id}"),
            response_model=ImageTaskStatusResponse,
            status_extractor=lambda x: x.output.task_status,
            estimated_duration=9,
            poll_interval=3,
        )
        return IO.NodeOutput(await download_url_to_image_tensor(str(response.output.results[0].url)))


class WanImageToImageApi(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="WanImageToImageApi",
            display_name="Wan Image to Image",
            category="partner/image/Wan",
            description="Generates an image from one or two input images and a text prompt. "
            "The output image is currently fixed at 1.6 MP, and its aspect ratio matches the input image(s).",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=["wan2.5-i2i-preview"],
                    default="wan2.5-i2i-preview",
                    tooltip="Model to use.",
                ),
                IO.Image.Input(
                    "image",
                    tooltip="Single-image editing or multi-image fusion. Maximum 2 images.",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt describing the elements and visual features. Supports English and Chinese.",
                ),
                IO.String.Input(
                    "negative_prompt",
                    multiline=True,
                    default="",
                    tooltip="Negative prompt describing what to avoid.",
                    optional=True,
                ),
                # redo this later as an optional combo of recommended resolutions
                # IO.Int.Input(
                #     "width",
                #     default=1280,
                #     min=384,
                #     max=1440,
                #     step=16,
                #     optional=True,
                # ),
                # IO.Int.Input(
                #     "height",
                #     default=1280,
                #     min=384,
                #     max=1440,
                #     step=16,
                #     optional=True,
                # ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip="Whether to add an AI-generated watermark to the result.",
                    optional=True,
                    advanced=True,
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
                expr="""{"type":"usd","usd":0.03}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        image: Input.Image,
        prompt: str,
        negative_prompt: str = "",
        # width: int = 1024,
        # height: int = 1024,
        seed: int = 0,
        watermark: bool = False,
    ):
        n_images = get_number_of_images(image)
        if n_images not in (1, 2):
            raise ValueError(f"Expected 1 or 2 input images, but got {n_images}.")
        images = []
        for i in image:
            images.append("data:image/png;base64," + tensor_to_base64_string(i, total_pixels=4096 * 4096))
        initial_response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/wan/api/v1/services/aigc/image2image/image-synthesis", method="POST"),
            response_model=TaskCreationResponse,
            data=Image2ImageTaskCreationRequest(
                model=model,
                input=Image2ImageInputField(prompt=prompt, negative_prompt=negative_prompt, images=images),
                parameters=Image2ImageParametersField(
                    # size=f"{width}*{height}",
                    seed=seed,
                    watermark=watermark,
                ),
            ),
        )
        if not initial_response.output:
            raise Exception(f"An unknown error occurred: {initial_response.code} - {initial_response.message}")
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/wan/api/v1/tasks/{initial_response.output.task_id}"),
            response_model=ImageTaskStatusResponse,
            status_extractor=lambda x: x.output.task_status,
            estimated_duration=42,
            poll_interval=4,
        )
        return IO.NodeOutput(await download_url_to_image_tensor(str(response.output.results[0].url)))


class WanTextToVideoApi(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="WanTextToVideoApi",
            display_name="Wan Text to Video",
            category="partner/video/Wan",
            description="Generates a video based on a text prompt.",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=["wan2.5-t2v-preview", "wan2.6-t2v"],
                    default="wan2.6-t2v",
                    tooltip="Model to use.",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt describing the elements and visual features. Supports English and Chinese.",
                ),
                IO.String.Input(
                    "negative_prompt",
                    multiline=True,
                    default="",
                    tooltip="Negative prompt describing what to avoid.",
                    optional=True,
                ),
                IO.Combo.Input(
                    "size",
                    options=[
                        "480p: 1:1 (624x624)",
                        "480p: 16:9 (832x480)",
                        "480p: 9:16 (480x832)",
                        "720p: 1:1 (960x960)",
                        "720p: 16:9 (1280x720)",
                        "720p: 9:16 (720x1280)",
                        "720p: 4:3 (1088x832)",
                        "720p: 3:4 (832x1088)",
                        "1080p: 1:1 (1440x1440)",
                        "1080p: 16:9 (1920x1080)",
                        "1080p: 9:16 (1080x1920)",
                        "1080p: 4:3 (1632x1248)",
                        "1080p: 3:4 (1248x1632)",
                    ],
                    default="720p: 1:1 (960x960)",
                    optional=True,
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=5,
                    max=15,
                    step=5,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="A 15-second duration is available only for the Wan 2.6 model.",
                    optional=True,
                ),
                IO.Audio.Input(
                    "audio",
                    optional=True,
                    tooltip="Audio must contain a clear, loud voice, without extraneous noise or background music.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "generate_audio",
                    default=False,
                    optional=True,
                    tooltip="If no audio input is provided, generate audio automatically.",
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "prompt_extend",
                    default=True,
                    tooltip="Whether to enhance the prompt with AI assistance.",
                    optional=True,
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip="Whether to add an AI-generated watermark to the result.",
                    optional=True,
                    advanced=True,
                ),
                IO.Combo.Input(
                    "shot_type",
                    options=["single", "multi"],
                    tooltip="Specifies the shot type for the generated video, that is, whether the video is a "
                    "single continuous shot or multiple shots with cuts. "
                    "This parameter takes effect only when prompt_extend is True.",
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
                depends_on=IO.PriceBadgeDepends(widgets=["duration", "size"]),
                expr="""
                (
                  $ppsTable := { "480p": 0.05, "720p": 0.1, "1080p": 0.15 };
                  $resKey := $substringBefore(widgets.size, ":");
                  $pps := $lookup($ppsTable, $resKey);
                  { "type": "usd", "usd": $round($pps * widgets.duration, 2) }
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        negative_prompt: str = "",
        size: str = "720p: 1:1 (960x960)",
        duration: int = 5,
        audio: Input.Audio | None = None,
        seed: int = 0,
        generate_audio: bool = False,
        prompt_extend: bool = True,
        watermark: bool = False,
        shot_type: str = "single",
    ):
        if "480p" in size and model == "wan2.6-t2v":
            raise ValueError("The Wan 2.6 model does not support 480p.")
        if duration == 15 and model == "wan2.5-t2v-preview":
            raise ValueError("A 15-second duration is supported only by the Wan 2.6 model.")
        width, height = RES_IN_PARENS.search(size).groups()
        audio_url = None
        if audio is not None:
            validate_audio_duration(audio, 3.0, 29.0)
            audio_url = "data:audio/mp3;base64," + audio_to_base64_string(audio, "mp3", "libmp3lame")

        initial_response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/wan/api/v1/services/aigc/video-generation/video-synthesis", method="POST"),
            response_model=TaskCreationResponse,
            data=Text2VideoTaskCreationRequest(
                model=model,
                input=Text2VideoInputField(prompt=prompt, negative_prompt=negative_prompt, audio_url=audio_url),
                parameters=Text2VideoParametersField(
                    size=f"{width}*{height}",
                    duration=duration,
                    seed=seed,
                    audio=generate_audio,
                    prompt_extend=prompt_extend,
                    watermark=watermark,
                    shot_type=shot_type,
                ),
            ),
        )
        if not initial_response.output:
            raise Exception(f"An unknown error occurred: {initial_response.code} - {initial_response.message}")
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/wan/api/v1/tasks/{initial_response.output.task_id}"),
            response_model=VideoTaskStatusResponse,
            status_extractor=lambda x: x.output.task_status,
            estimated_duration=120 * int(duration / 5),
            poll_interval=6,
        )
        return IO.NodeOutput(await download_url_to_video_output(response.output.video_url))


class WanImageToVideoApi(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="WanImageToVideoApi",
            display_name="Wan Image to Video",
            category="partner/video/Wan",
            description="Generates a video from the first frame and a text prompt.",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=["wan2.5-i2v-preview", "wan2.6-i2v"],
                    default="wan2.6-i2v",
                    tooltip="Model to use.",
                ),
                IO.Image.Input(
                    "image",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt describing the elements and visual features. Supports English and Chinese.",
                ),
                IO.String.Input(
                    "negative_prompt",
                    multiline=True,
                    default="",
                    tooltip="Negative prompt describing what to avoid.",
                    optional=True,
                ),
                IO.Combo.Input(
                    "resolution",
                    options=[
                        "480P",
                        "720P",
                        "1080P",
                    ],
                    default="720P",
                    optional=True,
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=5,
                    max=15,
                    step=5,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Duration 15 available only for WAN2.6 model.",
                    optional=True,
                ),
                IO.Audio.Input(
                    "audio",
                    optional=True,
                    tooltip="Audio must contain a clear, loud voice, without extraneous noise or background music.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "generate_audio",
                    default=False,
                    optional=True,
                    tooltip="If no audio input is provided, generate audio automatically.",
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "prompt_extend",
                    default=True,
                    tooltip="Whether to enhance the prompt with AI assistance.",
                    optional=True,
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip="Whether to add an AI-generated watermark to the result.",
                    optional=True,
                    advanced=True,
                ),
                IO.Combo.Input(
                    "shot_type",
                    options=["single", "multi"],
                    tooltip="Specifies the shot type for the generated video, that is, whether the video is a "
                    "single continuous shot or multiple shots with cuts. "
                    "This parameter takes effect only when prompt_extend is True.",
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
                depends_on=IO.PriceBadgeDepends(widgets=["duration", "resolution"]),
                expr="""
                (
                  $ppsTable := { "480p": 0.05, "720p": 0.1, "1080p": 0.15 };
                  $pps := $lookup($ppsTable, widgets.resolution);
                  { "type": "usd", "usd": $round($pps * widgets.duration, 2) }
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
        negative_prompt: str = "",
        resolution: str = "720P",
        duration: int = 5,
        audio: Input.Audio | None = None,
        seed: int = 0,
        generate_audio: bool = False,
        prompt_extend: bool = True,
        watermark: bool = False,
        shot_type: str = "single",
    ):
        if get_number_of_images(image) != 1:
            raise ValueError("Exactly one input image is required.")
        if "480P" in resolution and model == "wan2.6-i2v":
            raise ValueError("The Wan 2.6 model does not support 480P.")
        if duration == 15 and model == "wan2.5-i2v-preview":
            raise ValueError("A 15-second duration is supported only by the Wan 2.6 model.")
        image_url = "data:image/png;base64," + tensor_to_base64_string(image, total_pixels=2000 * 2000)
        audio_url = None
        if audio is not None:
            validate_audio_duration(audio, 3.0, 29.0)
            audio_url = "data:audio/mp3;base64," + audio_to_base64_string(audio, "mp3", "libmp3lame")
        initial_response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/wan/api/v1/services/aigc/video-generation/video-synthesis", method="POST"),
            response_model=TaskCreationResponse,
            data=Image2VideoTaskCreationRequest(
                model=model,
                input=Image2VideoInputField(
                    prompt=prompt, negative_prompt=negative_prompt, img_url=image_url, audio_url=audio_url
                ),
                parameters=Image2VideoParametersField(
                    resolution=resolution,
                    duration=duration,
                    seed=seed,
                    audio=generate_audio,
                    prompt_extend=prompt_extend,
                    watermark=watermark,
                    shot_type=shot_type,
                ),
            ),
        )
        if not initial_response.output:
            raise Exception(f"An unknown error occurred: {initial_response.code} - {initial_response.message}")
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/wan/api/v1/tasks/{initial_response.output.task_id}"),
            response_model=VideoTaskStatusResponse,
            status_extractor=lambda x: x.output.task_status,
            estimated_duration=120 * int(duration / 5),
            poll_interval=6,
        )
        return IO.NodeOutput(await download_url_to_video_output(response.output.video_url))


class WanReferenceVideoApi(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="WanReferenceVideoApi",
            display_name="Wan Reference to Video",
            category="partner/video/Wan",
            description="Use the character and voice from input videos, combined with a prompt, "
            "to generate a new video that maintains character consistency.",
            inputs=[
                IO.Combo.Input("model", options=["wan2.6-r2v"]),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt describing the elements and visual features. Supports English and Chinese. "
                    "Use identifiers such as `character1` and `character2` to refer to the reference characters.",
                ),
                IO.String.Input(
                    "negative_prompt",
                    multiline=True,
                    default="",
                    tooltip="Negative prompt describing what to avoid.",
                ),
                IO.Autogrow.Input(
                    "reference_videos",
                    template=IO.Autogrow.TemplateNames(
                        IO.Video.Input("reference_video"),
                        names=["character1", "character2", "character3"],
                        min=1,
                    ),
                ),
                IO.Combo.Input(
                    "size",
                    options=[
                        "720p: 1:1 (960x960)",
                        "720p: 16:9 (1280x720)",
                        "720p: 9:16 (720x1280)",
                        "720p: 4:3 (1088x832)",
                        "720p: 3:4 (832x1088)",
                        "1080p: 1:1 (1440x1440)",
                        "1080p: 16:9 (1920x1080)",
                        "1080p: 9:16 (1080x1920)",
                        "1080p: 4:3 (1632x1248)",
                        "1080p: 3:4 (1248x1632)",
                    ],
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=5,
                    max=10,
                    step=5,
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                ),
                IO.Combo.Input(
                    "shot_type",
                    options=["single", "multi"],
                    tooltip="Specifies the shot type for the generated video, that is, whether the video is a "
                    "single continuous shot or multiple shots with cuts.",
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip="Whether to add an AI-generated watermark to the result.",
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
                depends_on=IO.PriceBadgeDepends(widgets=["size", "duration"]),
                expr="""
                (
                  $rate := $contains(widgets.size, "1080p") ? 0.15 : 0.10;
                  $inputMin := 2 * $rate;
                  $inputMax := 5 * $rate;
                  $outputPrice := widgets.duration * $rate;
                  {
                    "type": "range_usd",
                    "min_usd": $inputMin + $outputPrice,
                    "max_usd": $inputMax + $outputPrice
                  }
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        negative_prompt: str,
        reference_videos: IO.Autogrow.Type,
        size: str,
        duration: int,
        seed: int,
        shot_type: str,
        watermark: bool,
    ):
        reference_video_urls = []
        for i in reference_videos:
            validate_video_duration(reference_videos[i], min_duration=2, max_duration=30)
        for i in reference_videos:
            reference_video_urls.append(await upload_video_to_comfyapi(cls, reference_videos[i]))
        width, height = RES_IN_PARENS.search(size).groups()
        initial_response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/wan/api/v1/services/aigc/video-generation/video-synthesis", method="POST"),
            response_model=TaskCreationResponse,
            data=Reference2VideoTaskCreationRequest(
                model=model,
                input=Reference2VideoInputField(
                    prompt=prompt, negative_prompt=negative_prompt, reference_video_urls=reference_video_urls
                ),
                parameters=Reference2VideoParametersField(
                    size=f"{width}*{height}",
                    duration=duration,
                    shot_type=shot_type,
                    watermark=watermark,
                    seed=seed,
                ),
            ),
        )
        if not initial_response.output:
            raise Exception(f"An unknown error occurred: {initial_response.code} - {initial_response.message}")
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/wan/api/v1/tasks/{initial_response.output.task_id}"),
            response_model=VideoTaskStatusResponse,
            status_extractor=lambda x: x.output.task_status,
            poll_interval=6,
        )
        return IO.NodeOutput(await download_url_to_video_output(response.output.video_url))


class Wan2TextToVideoApi(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Wan2TextToVideoApi",
            display_name="Wan 2.7 Text to Video",
            category="partner/video/Wan",
            description="Generates a video based on a text prompt using the Wan 2.7 model.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "wan2.7-t2v",
                            [
                                IO.String.Input(
                                    "prompt",
                                    multiline=True,
                                    default="",
                                    tooltip="Prompt describing the elements and visual features. "
                                    "Supports English and Chinese.",
                                ),
                                IO.String.Input(
                                    "negative_prompt",
                                    multiline=True,
                                    default="",
                                    tooltip="Negative prompt describing what to avoid.",
                                ),
                                IO.Combo.Input(
                                    "resolution",
                                    options=["720P", "1080P"],
                                ),
                                IO.Combo.Input(
                                    "ratio",
                                    options=["16:9", "9:16", "1:1", "4:3", "3:4"],
                                ),
                                IO.Int.Input(
                                    "duration",
                                    default=5,
                                    min=2,
                                    max=15,
                                    step=1,
                                    display_mode=IO.NumberDisplay.number,
                                ),
                            ],
                        ),
                    ],
                ),
                IO.Audio.Input(
                    "audio",
                    optional=True,
                    tooltip="Audio for driving video generation (e.g., lip sync, beat-matched motion). "
                    "Duration: 3s-30s. If not provided, the model automatically generates matching "
                    "background music or sound effects.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                ),
                IO.Boolean.Input(
                    "prompt_extend",
                    default=True,
                    tooltip="Whether to enhance the prompt with AI assistance.",
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip="Whether to add an AI-generated watermark to the result.",
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
                  $res := $lookup(widgets, "model.resolution");
                  $dur := $lookup(widgets, "model.duration");
                  $ppsTable := { "720p": 0.1, "1080p": 0.15 };
                  $pps := $lookup($ppsTable, $res);
                  { "type": "usd", "usd": $pps * $dur }
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: dict,
        seed: int,
        prompt_extend: bool,
        watermark: bool,
        audio: Input.Audio | None = None,
    ):
        validate_string(model["prompt"], strip_whitespace=False, min_length=1)
        audio_url = None
        if audio is not None:
            validate_audio_duration(audio, 1.5, 60.0)
            audio_url = await upload_audio_to_comfyapi(
                cls, audio, container_format="mp3", codec_name="libmp3lame", mime_type="audio/mpeg"
            )
        initial_response = await sync_op(
            cls,
            ApiEndpoint(
                path="/proxy/wan/api/v1/services/aigc/video-generation/video-synthesis",
                method="POST",
            ),
            response_model=TaskCreationResponse,
            data=Wan27Text2VideoTaskCreationRequest(
                model=model["model"],
                input=Text2VideoInputField(
                    prompt=model["prompt"],
                    negative_prompt=model["negative_prompt"] or None,
                    audio_url=audio_url,
                ),
                parameters=Wan27Text2VideoParametersField(
                    resolution=model["resolution"],
                    ratio=model["ratio"],
                    duration=model["duration"],
                    seed=seed,
                    prompt_extend=prompt_extend,
                    watermark=watermark,
                ),
            ),
        )
        if not initial_response.output:
            raise Exception(f"An unknown error occurred: {initial_response.code} - {initial_response.message}")
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/wan/api/v1/tasks/{initial_response.output.task_id}"),
            response_model=VideoTaskStatusResponse,
            status_extractor=lambda x: x.output.task_status,
            poll_interval=7,
        )
        return IO.NodeOutput(await download_url_to_video_output(response.output.video_url))


class Wan2ImageToVideoApi(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Wan2ImageToVideoApi",
            display_name="Wan 2.7 Image to Video",
            category="partner/video/Wan",
            description="Generate a video from a first-frame image, with optional last-frame image and audio.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "wan2.7-i2v",
                            [
                                IO.String.Input(
                                    "prompt",
                                    multiline=True,
                                    default="",
                                    tooltip="Prompt describing the elements and visual features. "
                                    "Supports English and Chinese.",
                                ),
                                IO.String.Input(
                                    "negative_prompt",
                                    multiline=True,
                                    default="",
                                    tooltip="Negative prompt describing what to avoid.",
                                ),
                                IO.Combo.Input(
                                    "resolution",
                                    options=["720P", "1080P"],
                                ),
                                IO.Int.Input(
                                    "duration",
                                    default=5,
                                    min=2,
                                    max=15,
                                    step=1,
                                    display_mode=IO.NumberDisplay.number,
                                ),
                            ],
                        ),
                    ],
                ),
                IO.Image.Input(
                    "first_frame",
                    tooltip="First frame image. The output aspect ratio is derived from this image.",
                ),
                IO.Image.Input(
                    "last_frame",
                    optional=True,
                    tooltip="Last frame image. The model generates a video transitioning from first to last frame.",
                ),
                IO.Audio.Input(
                    "audio",
                    optional=True,
                    tooltip="Audio for driving video generation (e.g., lip sync, beat-matched motion). "
                    "Duration: 2s-30s. If not provided, the model automatically generates matching "
                    "background music or sound effects.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                ),
                IO.Boolean.Input(
                    "prompt_extend",
                    default=True,
                    tooltip="Whether to enhance the prompt with AI assistance.",
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip="Whether to add an AI-generated watermark to the result.",
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
                  $res := $lookup(widgets, "model.resolution");
                  $dur := $lookup(widgets, "model.duration");
                  $ppsTable := { "720p": 0.1, "1080p": 0.15 };
                  $pps := $lookup($ppsTable, $res);
                  { "type": "usd", "usd": $pps * $dur }
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: dict,
        first_frame: Input.Image,
        seed: int,
        prompt_extend: bool,
        watermark: bool,
        last_frame: Input.Image | None = None,
        audio: Input.Audio | None = None,
    ):
        media = [
            Wan27MediaItem(
                type="first_frame",
                url=await upload_image_to_comfyapi(cls, image=first_frame),
            )
        ]
        if last_frame is not None:
            media.append(
                Wan27MediaItem(
                    type="last_frame",
                    url=await upload_image_to_comfyapi(cls, image=last_frame),
                )
            )
        if audio is not None:
            validate_audio_duration(audio, 2.0, 30.0)
            audio_url = await upload_audio_to_comfyapi(
                cls, audio, container_format="mp3", codec_name="libmp3lame", mime_type="audio/mpeg"
            )
            media.append(Wan27MediaItem(type="driving_audio", url=audio_url))
        initial_response = await sync_op(
            cls,
            ApiEndpoint(
                path="/proxy/wan/api/v1/services/aigc/video-generation/video-synthesis",
                method="POST",
            ),
            response_model=TaskCreationResponse,
            data=Wan27ImageToVideoTaskCreationRequest(
                model=model["model"],
                input=Wan27ImageToVideoInputField(
                    prompt=model["prompt"] or None,
                    negative_prompt=model["negative_prompt"] or None,
                    media=media,
                ),
                parameters=Wan27ImageToVideoParametersField(
                    resolution=model["resolution"],
                    duration=model["duration"],
                    seed=seed,
                    prompt_extend=prompt_extend,
                    watermark=watermark,
                ),
            ),
        )
        if not initial_response.output:
            raise Exception(f"An unknown error occurred: {initial_response.code} - {initial_response.message}")
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/wan/api/v1/tasks/{initial_response.output.task_id}"),
            response_model=VideoTaskStatusResponse,
            status_extractor=lambda x: x.output.task_status,
            poll_interval=7,
        )
        return IO.NodeOutput(await download_url_to_video_output(response.output.video_url))


class Wan2VideoContinuationApi(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Wan2VideoContinuationApi",
            display_name="Wan 2.7 Video Continuation",
            category="partner/video/Wan",
            description="Continue a video from where it left off, with optional last-frame control.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "wan2.7-i2v",
                            [
                                IO.String.Input(
                                    "prompt",
                                    multiline=True,
                                    default="",
                                    tooltip="Prompt describing the elements and visual features. Supports English and Chinese.",
                                ),
                                IO.String.Input(
                                    "negative_prompt",
                                    multiline=True,
                                    default="",
                                    tooltip="Negative prompt describing what to avoid.",
                                ),
                                IO.Combo.Input(
                                    "resolution",
                                    options=["720P", "1080P"],
                                ),
                                IO.Int.Input(
                                    "duration",
                                    default=5,
                                    min=2,
                                    max=15,
                                    step=1,
                                    display_mode=IO.NumberDisplay.number,
                                    tooltip="Total output duration in seconds. The model generates continuation "
                                    "to fill the remaining time after the input clip.",
                                ),
                            ],
                        ),
                    ],
                ),
                IO.Video.Input(
                    "first_clip",
                    tooltip="Input video to continue from. Duration: 2s-10s. "
                    "The output aspect ratio is derived from this video.",
                ),
                IO.Image.Input(
                    "last_frame",
                    optional=True,
                    tooltip="Last frame image. The continuation will transition towards this frame.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                ),
                IO.Boolean.Input(
                    "prompt_extend",
                    default=True,
                    tooltip="Whether to enhance the prompt with AI assistance.",
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip="Whether to add an AI-generated watermark to the result.",
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
                  $res := $lookup(widgets, "model.resolution");
                  $dur := $lookup(widgets, "model.duration");
                  $ppsTable := { "720p": 0.1, "1080p": 0.15 };
                  $pps := $lookup($ppsTable, $res);
                  $outputPrice := $pps * $dur;
                  {
                    "type": "range_usd",
                    "min_usd": 2 * $pps + $outputPrice,
                    "max_usd": 5 * $pps + $outputPrice
                  }
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: dict,
        first_clip: Input.Video,
        prompt: str = "",
        negative_prompt: str = "",
        last_frame: Input.Image | None = None,
        seed: int = 0,
        prompt_extend: bool = True,
        watermark: bool = False,
    ):
        validate_video_duration(first_clip, min_duration=2, max_duration=10)
        media = [
            Wan27MediaItem(
                type="first_clip",
                url=await upload_video_to_comfyapi(cls, first_clip),
            )
        ]
        if last_frame is not None:
            media.append(
                Wan27MediaItem(
                    type="last_frame",
                    url=await upload_image_to_comfyapi(cls, image=last_frame),
                )
            )
        initial_response = await sync_op(
            cls,
            ApiEndpoint(
                path="/proxy/wan/api/v1/services/aigc/video-generation/video-synthesis",
                method="POST",
            ),
            response_model=TaskCreationResponse,
            data=Wan27ImageToVideoTaskCreationRequest(
                model=model["model"],
                input=Wan27ImageToVideoInputField(
                    prompt=model["prompt"] or None,
                    negative_prompt=model["negative_prompt"] or None,
                    media=media,
                ),
                parameters=Wan27ImageToVideoParametersField(
                    resolution=model["resolution"],
                    duration=model["duration"],
                    seed=seed,
                    prompt_extend=prompt_extend,
                    watermark=watermark,
                ),
            ),
        )
        if not initial_response.output:
            raise Exception(f"An unknown error occurred: {initial_response.code} - {initial_response.message}")
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/wan/api/v1/tasks/{initial_response.output.task_id}"),
            response_model=VideoTaskStatusResponse,
            status_extractor=lambda x: x.output.task_status,
            poll_interval=7,
        )
        return IO.NodeOutput(await download_url_to_video_output(response.output.video_url))


class Wan2VideoEditApi(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Wan2VideoEditApi",
            display_name="Wan 2.7 Video Edit",
            category="partner/video/Wan",
            description="Edit a video using text instructions, reference images, or style transfer.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "wan2.7-videoedit",
                            [
                                IO.String.Input(
                                    "prompt",
                                    multiline=True,
                                    default="",
                                    tooltip="Editing instructions or style transfer requirements.",
                                ),
                                IO.Combo.Input(
                                    "resolution",
                                    options=["720P", "1080P"],
                                ),
                                IO.Combo.Input(
                                    "ratio",
                                    options=["16:9", "9:16", "1:1", "4:3", "3:4"],
                                    tooltip="Aspect ratio. If not changed, approximates the input video ratio.",
                                ),
                                IO.Combo.Input(
                                    "duration",
                                    options=["auto", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                                    default="auto",
                                    tooltip="Output duration in seconds. 'auto' matches the input video duration. "
                                    "A specific value truncates from the start of the video.",
                                ),
                                IO.Autogrow.Input(
                                    "reference_images",
                                    template=IO.Autogrow.TemplateNames(
                                        IO.Image.Input("reference_image"),
                                        names=[
                                            "image1",
                                            "image2",
                                            "image3",
                                            "image4",
                                        ],
                                        min=0,
                                    ),
                                ),
                            ],
                        ),
                    ],
                ),
                IO.Video.Input(
                    "video",
                    tooltip="The video to edit.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                ),
                IO.Combo.Input(
                    "audio_setting",
                    options=["auto", "origin"],
                    default="auto",
                    tooltip="'auto': model decides whether to regenerate audio based on the prompt. "
                    "'origin': preserve the original audio from the input video.",
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip="Whether to add an AI-generated watermark to the result.",
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
                  $res := $lookup(widgets, "model.resolution");
                  $dur := $lookup(widgets, "model.duration");
                  $ppsTable := { "720p": 0.1, "1080p": 0.15 };
                  $pps := $lookup($ppsTable, $res);
                  { "type": "usd", "usd": $pps, "format": { "suffix": "/second", "note": "(input + output)" } }
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: dict,
        video: Input.Video,
        seed: int,
        audio_setting: str,
        watermark: bool,
    ):
        validate_string(model["prompt"], strip_whitespace=False, min_length=1)
        validate_video_duration(video, min_duration=2, max_duration=10)
        duration = 0 if model["duration"] == "auto" else int(model["duration"])
        media = [Wan27MediaItem(type="video", url=await upload_video_to_comfyapi(cls, video))]
        reference_images = model.get("reference_images", {})
        for key in reference_images:
            media.append(
                Wan27MediaItem(
                    type="reference_image", url=await upload_image_to_comfyapi(cls, image=reference_images[key])
                )
            )
        initial_response = await sync_op(
            cls,
            ApiEndpoint(
                path="/proxy/wan/api/v1/services/aigc/video-generation/video-synthesis",
                method="POST",
            ),
            response_model=TaskCreationResponse,
            data=Wan27VideoEditTaskCreationRequest(
                model=model["model"],
                input=Wan27VideoEditInputField(prompt=model["prompt"], media=media),
                parameters=Wan27VideoEditParametersField(
                    resolution=model["resolution"],
                    ratio=model["ratio"],
                    duration=duration,
                    audio_setting=audio_setting,
                    watermark=watermark,
                    seed=seed,
                ),
            ),
        )
        if not initial_response.output:
            raise Exception(f"An unknown error occurred: {initial_response.code} - {initial_response.message}")
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/wan/api/v1/tasks/{initial_response.output.task_id}"),
            response_model=VideoTaskStatusResponse,
            status_extractor=lambda x: x.output.task_status,
            poll_interval=7,
        )
        return IO.NodeOutput(await download_url_to_video_output(response.output.video_url))


class Wan2ReferenceVideoApi(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Wan2ReferenceVideoApi",
            display_name="Wan 2.7 Reference to Video",
            category="partner/video/Wan",
            description="Generate a video featuring a person or object from reference materials. "
            "Supports single-character performances and multi-character interactions.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "wan2.7-r2v",
                            [
                                IO.String.Input(
                                    "prompt",
                                    multiline=True,
                                    default="",
                                    tooltip="Prompt describing the video. Use identifiers such as 'character1' and "
                                    "'character2' to refer to the reference characters.",
                                ),
                                IO.String.Input(
                                    "negative_prompt",
                                    multiline=True,
                                    default="",
                                    tooltip="Negative prompt describing what to avoid.",
                                ),
                                IO.Combo.Input(
                                    "resolution",
                                    options=["720P", "1080P"],
                                ),
                                IO.Combo.Input(
                                    "ratio",
                                    options=["16:9", "9:16", "1:1", "4:3", "3:4"],
                                ),
                                IO.Int.Input(
                                    "duration",
                                    default=5,
                                    min=2,
                                    max=10,
                                    step=1,
                                    display_mode=IO.NumberDisplay.number,
                                ),
                                IO.Autogrow.Input(
                                    "reference_videos",
                                    template=IO.Autogrow.TemplateNames(
                                        IO.Video.Input("reference_video"),
                                        names=["video1", "video2", "video3"],
                                        min=0,
                                    ),
                                ),
                                IO.Autogrow.Input(
                                    "reference_images",
                                    template=IO.Autogrow.TemplateNames(
                                        IO.Image.Input("reference_image"),
                                        names=["image1", "image2", "image3", "image4", "image5"],
                                        min=0,
                                    ),
                                ),
                            ],
                        ),
                    ],
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip="Whether to add an AI-generated watermark to the result.",
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
                  $res := $lookup(widgets, "model.resolution");
                  $dur := $lookup(widgets, "model.duration");
                  $ppsTable := { "720p": 0.1, "1080p": 0.15 };
                  $pps := $lookup($ppsTable, $res);
                  $outputPrice := $pps * $dur;
                  {
                    "type": "range_usd",
                    "min_usd": $outputPrice,
                    "max_usd": 5 * $pps + $outputPrice
                  }
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
    ):
        validate_string(model["prompt"], strip_whitespace=False, min_length=1)
        media = []
        reference_videos = model.get("reference_videos", {})
        for key in reference_videos:
            media.append(
                Wan27MediaItem(type="reference_video", url=await upload_video_to_comfyapi(cls, reference_videos[key]))
            )
        reference_images = model.get("reference_images", {})
        for key in reference_images:
            media.append(
                Wan27MediaItem(
                    type="reference_image",
                    url=await upload_image_to_comfyapi(cls, image=reference_images[key]),
                )
            )
        if not media:
            raise ValueError("At least one reference video or reference image must be provided.")
        if len(media) > 5:
            raise ValueError(
                f"Too many references ({len(media)}). The maximum total of reference videos and images is 5."
            )

        initial_response = await sync_op(
            cls,
            ApiEndpoint(
                path="/proxy/wan/api/v1/services/aigc/video-generation/video-synthesis",
                method="POST",
            ),
            response_model=TaskCreationResponse,
            data=Wan27ReferenceVideoTaskCreationRequest(
                model=model["model"],
                input=Wan27ReferenceVideoInputField(
                    prompt=model["prompt"],
                    negative_prompt=model["negative_prompt"] or None,
                    media=media,
                ),
                parameters=Wan27ReferenceVideoParametersField(
                    resolution=model["resolution"],
                    ratio=model["ratio"],
                    duration=model["duration"],
                    watermark=watermark,
                    seed=seed,
                ),
            ),
        )
        if not initial_response.output:
            raise Exception(f"An unknown error occurred: {initial_response.code} - {initial_response.message}")
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/wan/api/v1/tasks/{initial_response.output.task_id}"),
            response_model=VideoTaskStatusResponse,
            status_extractor=lambda x: x.output.task_status,
            poll_interval=7,
        )
        return IO.NodeOutput(await download_url_to_video_output(response.output.video_url))


class HappyHorseTextToVideoApi(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HappyHorseTextToVideoApi",
            display_name="HappyHorse Text to Video",
            category="partner/video/Wan",
            description="Generates a video based on a text prompt using the HappyHorse model.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "happyhorse-1.1-t2v",
                            [
                                IO.String.Input(
                                    "prompt",
                                    multiline=True,
                                    default="",
                                    tooltip="Prompt describing the elements and visual features. "
                                    "Supports English and Chinese.",
                                ),
                                IO.Combo.Input(
                                    "resolution",
                                    options=["720P", "1080P"],
                                ),
                                IO.Combo.Input(
                                    "ratio",
                                    options=[
                                        "16:9",
                                        "9:16",
                                        "1:1",
                                        "4:3",
                                        "3:4",
                                        "21:9",
                                        "9:21",
                                        "5:4",
                                        "4:5",
                                    ],
                                ),
                                IO.Int.Input(
                                    "duration",
                                    default=5,
                                    min=3,
                                    max=15,
                                    step=1,
                                    display_mode=IO.NumberDisplay.number,
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "happyhorse-1.0-t2v",
                            [
                                IO.String.Input(
                                    "prompt",
                                    multiline=True,
                                    default="",
                                    tooltip="Prompt describing the elements and visual features. "
                                    "Supports English and Chinese.",
                                ),
                                IO.Combo.Input(
                                    "resolution",
                                    options=["720P", "1080P"],
                                ),
                                IO.Combo.Input(
                                    "ratio",
                                    options=["16:9", "9:16", "1:1", "4:3", "3:4"],
                                ),
                                IO.Int.Input(
                                    "duration",
                                    default=5,
                                    min=3,
                                    max=15,
                                    step=1,
                                    display_mode=IO.NumberDisplay.number,
                                ),
                            ],
                        ),
                    ],
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip="Whether to add an AI-generated watermark to the result.",
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
                  $res := $lookup(widgets, "model.resolution");
                  $dur := $lookup(widgets, "model.duration");
                  $ppsTable := $contains(widgets.model, "1.1")
                    ? { "720p": 0.2002, "1080p": 0.2574 }
                    : { "720p": 0.14, "1080p": 0.24 };
                  $pps := $lookup($ppsTable, $res);
                  { "type": "usd", "usd": $pps * $dur }
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
    ):
        validate_string(model["prompt"], strip_whitespace=False, min_length=1)
        initial_response = await sync_op(
            cls,
            ApiEndpoint(
                path="/proxy/wan/api/v1/services/aigc/video-generation/video-synthesis",
                method="POST",
            ),
            response_model=TaskCreationResponse,
            data=Wan27Text2VideoTaskCreationRequest(
                model=model["model"],
                input=Text2VideoInputField(
                    prompt=model["prompt"],
                    negative_prompt=None,
                ),
                parameters=Wan27Text2VideoParametersField(
                    resolution=model["resolution"],
                    ratio=model["ratio"],
                    duration=model["duration"],
                    seed=seed,
                    watermark=watermark,
                ),
            ),
        )
        if not initial_response.output:
            raise Exception(f"An unknown error occurred: {initial_response.code} - {initial_response.message}")
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/wan/api/v1/tasks/{initial_response.output.task_id}"),
            response_model=VideoTaskStatusResponse,
            status_extractor=lambda x: x.output.task_status,
            poll_interval=7,
        )
        return IO.NodeOutput(await download_url_to_video_output(response.output.video_url))


class HappyHorseImageToVideoApi(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HappyHorseImageToVideoApi",
            display_name="HappyHorse Image to Video",
            category="partner/video/Wan",
            description="Generate a video from a first-frame image using the HappyHorse model.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "happyhorse-1.1-i2v",
                            [
                                IO.String.Input(
                                    "prompt",
                                    multiline=True,
                                    default="",
                                    tooltip="Prompt describing the elements and visual features. "
                                    "Supports English and Chinese.",
                                ),
                                IO.Combo.Input(
                                    "resolution",
                                    options=["720P", "1080P"],
                                ),
                                IO.Int.Input(
                                    "duration",
                                    default=5,
                                    min=3,
                                    max=15,
                                    step=1,
                                    display_mode=IO.NumberDisplay.number,
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "happyhorse-1.0-i2v",
                            [
                                IO.String.Input(
                                    "prompt",
                                    multiline=True,
                                    default="",
                                    tooltip="Prompt describing the elements and visual features. "
                                    "Supports English and Chinese.",
                                ),
                                IO.Combo.Input(
                                    "resolution",
                                    options=["720P", "1080P"],
                                ),
                                IO.Int.Input(
                                    "duration",
                                    default=5,
                                    min=3,
                                    max=15,
                                    step=1,
                                    display_mode=IO.NumberDisplay.number,
                                ),
                            ],
                        ),
                    ],
                ),
                IO.Image.Input(
                    "first_frame",
                    tooltip="First frame image. The output aspect ratio is derived from this image.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip="Whether to add an AI-generated watermark to the result.",
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
                  $res := $lookup(widgets, "model.resolution");
                  $dur := $lookup(widgets, "model.duration");
                  $ppsTable := $contains(widgets.model, "1.1")
                    ? { "720p": 0.2002, "1080p": 0.2574 }
                    : { "720p": 0.14, "1080p": 0.24 };
                  $pps := $lookup($ppsTable, $res);
                  { "type": "usd", "usd": $pps * $dur }
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: dict,
        first_frame: Input.Image,
        seed: int,
        watermark: bool,
    ):
        validate_image_dimensions(first_frame, min_width=300, min_height=300)
        validate_image_aspect_ratio(first_frame, (1, 2.5), (2.5, 1), strict=False)
        media = [
            Wan27MediaItem(
                type="first_frame",
                url=await upload_image_to_comfyapi(cls, image=first_frame),
            )
        ]
        initial_response = await sync_op(
            cls,
            ApiEndpoint(
                path="/proxy/wan/api/v1/services/aigc/video-generation/video-synthesis",
                method="POST",
            ),
            response_model=TaskCreationResponse,
            data=Wan27ImageToVideoTaskCreationRequest(
                model=model["model"],
                input=Wan27ImageToVideoInputField(
                    prompt=model["prompt"] or None,
                    negative_prompt=None,
                    media=media,
                ),
                parameters=Wan27ImageToVideoParametersField(
                    resolution=model["resolution"],
                    duration=model["duration"],
                    seed=seed,
                    watermark=watermark,
                ),
            ),
        )
        if not initial_response.output:
            raise Exception(f"An unknown error occurred: {initial_response.code} - {initial_response.message}")
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/wan/api/v1/tasks/{initial_response.output.task_id}"),
            response_model=VideoTaskStatusResponse,
            status_extractor=lambda x: x.output.task_status,
            poll_interval=7,
        )
        return IO.NodeOutput(await download_url_to_video_output(response.output.video_url))


class HappyHorseVideoEditApi(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HappyHorseVideoEditApi",
            display_name="HappyHorse Video Edit",
            category="partner/video/Wan",
            description="Edit a video using text instructions or reference images with the HappyHorse model. "
            "Output duration is 3-15s and matches the input video; inputs longer than 15s are truncated.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "happyhorse-1.0-video-edit",
                            [
                                IO.String.Input(
                                    "prompt",
                                    multiline=True,
                                    default="",
                                    tooltip="Editing instructions or style transfer requirements.",
                                ),
                                IO.Combo.Input(
                                    "resolution",
                                    options=["720P", "1080P"],
                                ),
                                IO.Combo.Input(
                                    "ratio",
                                    options=["16:9", "9:16", "1:1", "4:3", "3:4"],
                                    tooltip="Aspect ratio. If not changed, approximates the input video ratio.",
                                ),
                                IO.Autogrow.Input(
                                    "reference_images",
                                    template=IO.Autogrow.TemplateNames(
                                        IO.Image.Input("reference_image"),
                                        names=[
                                            "image1",
                                            "image2",
                                            "image3",
                                            "image4",
                                            "image5",
                                        ],
                                        min=0,
                                    ),
                                ),
                            ],
                        ),
                    ],
                ),
                IO.Video.Input(
                    "video",
                    tooltip="The video to edit.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip="Whether to add an AI-generated watermark to the result.",
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
                depends_on=IO.PriceBadgeDepends(widgets=["model", "model.resolution"]),
                expr="""
                (
                  $res := $lookup(widgets, "model.resolution");
                  $ppsTable := { "720p": 0.14, "1080p": 0.24 };
                  $pps := $lookup($ppsTable, $res);
                  { "type": "usd", "usd": $pps, "format": { "suffix": "/second" } }
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: dict,
        video: Input.Video,
        seed: int,
        watermark: bool,
    ):
        validate_string(model["prompt"], strip_whitespace=False, min_length=1)
        validate_video_duration(video, min_duration=3, max_duration=60)
        media = [Wan27MediaItem(type="video", url=await upload_video_to_comfyapi(cls, video))]
        reference_images = model.get("reference_images", {})
        for key in reference_images:
            media.append(
                Wan27MediaItem(
                    type="reference_image", url=await upload_image_to_comfyapi(cls, image=reference_images[key])
                )
            )
        initial_response = await sync_op(
            cls,
            ApiEndpoint(
                path="/proxy/wan/api/v1/services/aigc/video-generation/video-synthesis",
                method="POST",
            ),
            response_model=TaskCreationResponse,
            data=Wan27VideoEditTaskCreationRequest(
                model=model["model"],
                input=Wan27VideoEditInputField(prompt=model["prompt"], media=media),
                parameters=Wan27VideoEditParametersField(
                    resolution=model["resolution"],
                    ratio=model["ratio"],
                    duration=None,
                    watermark=watermark,
                    seed=seed,
                ),
            ),
        )
        if not initial_response.output:
            raise Exception(f"An unknown error occurred: {initial_response.code} - {initial_response.message}")
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/wan/api/v1/tasks/{initial_response.output.task_id}"),
            response_model=VideoTaskStatusResponse,
            status_extractor=lambda x: x.output.task_status,
            poll_interval=7,
        )
        return IO.NodeOutput(await download_url_to_video_output(response.output.video_url))


class HappyHorseReferenceVideoApi(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HappyHorseReferenceVideoApi",
            display_name="HappyHorse Reference to Video",
            category="partner/video/Wan",
            description="Generate a video featuring a person or object from reference materials with the HappyHorse "
            "model. Supports single-character performances and multi-character interactions.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "happyhorse-1.1-r2v",
                            [
                                IO.String.Input(
                                    "prompt",
                                    multiline=True,
                                    default="",
                                    tooltip="Prompt describing the video. Use identifiers such as 'character1' and "
                                    "'character2' to refer to the reference characters.",
                                ),
                                IO.Combo.Input(
                                    "resolution",
                                    options=["720P", "1080P"],
                                ),
                                IO.Combo.Input(
                                    "ratio",
                                    options=[
                                        "16:9",
                                        "9:16",
                                        "1:1",
                                        "4:3",
                                        "3:4",
                                        "21:9",
                                        "9:21",
                                        "5:4",
                                        "4:5",
                                    ],
                                ),
                                IO.Int.Input(
                                    "duration",
                                    default=5,
                                    min=3,
                                    max=15,
                                    step=1,
                                    display_mode=IO.NumberDisplay.number,
                                ),
                                IO.Autogrow.Input(
                                    "reference_images",
                                    template=IO.Autogrow.TemplateNames(
                                        IO.Image.Input("reference_image"),
                                        names=[
                                            "image1",
                                            "image2",
                                            "image3",
                                            "image4",
                                            "image5",
                                            "image6",
                                            "image7",
                                            "image8",
                                            "image9",
                                        ],
                                        min=1,
                                    ),
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "happyhorse-1.0-r2v",
                            [
                                IO.String.Input(
                                    "prompt",
                                    multiline=True,
                                    default="",
                                    tooltip="Prompt describing the video. Use identifiers such as 'character1' and "
                                    "'character2' to refer to the reference characters.",
                                ),
                                IO.Combo.Input(
                                    "resolution",
                                    options=["720P", "1080P"],
                                ),
                                IO.Combo.Input(
                                    "ratio",
                                    options=["16:9", "9:16", "1:1", "4:3", "3:4"],
                                ),
                                IO.Int.Input(
                                    "duration",
                                    default=5,
                                    min=3,
                                    max=15,
                                    step=1,
                                    display_mode=IO.NumberDisplay.number,
                                ),
                                IO.Autogrow.Input(
                                    "reference_images",
                                    template=IO.Autogrow.TemplateNames(
                                        IO.Image.Input("reference_image"),
                                        names=[
                                            "image1",
                                            "image2",
                                            "image3",
                                            "image4",
                                            "image5",
                                            "image6",
                                            "image7",
                                            "image8",
                                            "image9",
                                        ],
                                        min=1,
                                    ),
                                ),
                            ],
                        ),
                    ],
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip="Whether to add an AI-generated watermark to the result.",
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
                  $res := $lookup(widgets, "model.resolution");
                  $dur := $lookup(widgets, "model.duration");
                  $ppsTable := $contains(widgets.model, "1.1")
                    ? { "720p": 0.2002, "1080p": 0.2574 }
                    : { "720p": 0.14, "1080p": 0.24 };
                  $pps := $lookup($ppsTable, $res);
                  { "type": "usd", "usd": $pps * $dur }
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
    ):
        validate_string(model["prompt"], strip_whitespace=False, min_length=1)
        reference_images = model.get("reference_images", {})
        for key in reference_images:
            validate_image_dimensions(reference_images[key], min_width=400, min_height=400)
            validate_image_aspect_ratio(reference_images[key], (1, 2.5), (2.5, 1), strict=False)
        media = []
        for key in reference_images:
            media.append(
                Wan27MediaItem(
                    type="reference_image",
                    url=await upload_image_to_comfyapi(cls, image=reference_images[key]),
                )
            )
        if not media:
            raise ValueError("At least one reference image must be provided.")

        initial_response = await sync_op(
            cls,
            ApiEndpoint(
                path="/proxy/wan/api/v1/services/aigc/video-generation/video-synthesis",
                method="POST",
            ),
            response_model=TaskCreationResponse,
            data=Wan27ReferenceVideoTaskCreationRequest(
                model=model["model"],
                input=Wan27ReferenceVideoInputField(
                    prompt=model["prompt"],
                    negative_prompt=None,
                    media=media,
                ),
                parameters=Wan27ReferenceVideoParametersField(
                    resolution=model["resolution"],
                    ratio=model["ratio"],
                    duration=model["duration"],
                    watermark=watermark,
                    seed=seed,
                ),
            ),
        )
        if not initial_response.output:
            raise Exception(f"An unknown error occurred: {initial_response.code} - {initial_response.message}")
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/wan/api/v1/tasks/{initial_response.output.task_id}"),
            response_model=VideoTaskStatusResponse,
            status_extractor=lambda x: x.output.task_status,
            poll_interval=7,
        )
        return IO.NodeOutput(await download_url_to_video_output(response.output.video_url))


class WanApiExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            WanTextToImageApi,
            WanImageToImageApi,
            WanTextToVideoApi,
            WanImageToVideoApi,
            WanReferenceVideoApi,
            Wan2TextToVideoApi,
            Wan2ImageToVideoApi,
            Wan2VideoContinuationApi,
            Wan2VideoEditApi,
            Wan2ReferenceVideoApi,
            HappyHorseTextToVideoApi,
            HappyHorseImageToVideoApi,
            HappyHorseVideoEditApi,
            HappyHorseReferenceVideoApi,
        ]


async def comfy_entrypoint() -> WanApiExtension:
    return WanApiExtension()
