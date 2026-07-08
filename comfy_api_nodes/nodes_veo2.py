import base64
from io import BytesIO

from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input, InputImpl
from comfy_api_nodes.apis.veo import (
    VeoGenVidPollRequest,
    VeoGenVidPollResponse,
    VeoGenVidRequest,
    VeoGenVidResponse,
    VeoRequestInstance,
    VeoRequestInstanceImage,
    VeoRequestParameters,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_video_output,
    poll_op,
    sync_op,
    tensor_to_base64_string,
)

AVERAGE_DURATION_VIDEO_GEN = 32
MODELS_MAP = {
    "veo-2.0-generate-001": "veo-2.0-generate-001",
    "veo-3.1-generate": "veo-3.1-generate-001",
    "veo-3.1-fast-generate": "veo-3.1-fast-generate-001",
    "veo-3.1-lite": "veo-3.1-lite-generate-001",
    "veo-3.0-generate-001": "veo-3.0-generate-001",
    "veo-3.0-fast-generate-001": "veo-3.0-fast-generate-001",
}


class VeoVideoGenerationNode(IO.ComfyNode):
    """
    Generates videos from text prompts using Google's Veo API.

    This node can create videos from text descriptions and optional image inputs,
    with control over parameters like aspect ratio, duration, and more.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VeoVideoGenerationNode",
            display_name="Google Veo 2 Video Generation",
            category="partner/video/Veo",
            description="Generates videos from text prompts using Google's Veo 2 API",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text description of the video",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["16:9", "9:16"],
                    default="16:9",
                    tooltip="Aspect ratio of the output video",
                ),
                IO.String.Input(
                    "negative_prompt",
                    multiline=True,
                    default="",
                    tooltip="Negative text prompt to guide what to avoid in the video",
                    optional=True,
                ),
                IO.Int.Input(
                    "duration_seconds",
                    default=5,
                    min=5,
                    max=8,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Duration of the output video in seconds",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "enhance_prompt",
                    default=True,
                    tooltip="Whether to enhance the prompt with AI assistance",
                    optional=True,
                    advanced=True,
                ),
                IO.Combo.Input(
                    "person_generation",
                    options=["ALLOW", "BLOCK"],
                    default="ALLOW",
                    tooltip="Whether to allow generating people in the video",
                    optional=True,
                    advanced=True,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFF,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed for video generation (0 for random)",
                    optional=True,
                ),
                IO.Image.Input(
                    "image",
                    tooltip="Optional reference image to guide video generation",
                    optional=True,
                ),
                IO.Combo.Input(
                    "model",
                    options=["veo-2.0-generate-001"],
                    default="veo-2.0-generate-001",
                    tooltip="Veo 2 model to use for video generation",
                    optional=True,
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
                depends_on=IO.PriceBadgeDepends(widgets=["duration_seconds"]),
                expr="""{"type":"usd","usd": 0.5 * widgets.duration_seconds}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt,
        aspect_ratio="16:9",
        negative_prompt="",
        duration_seconds=5,
        enhance_prompt=True,
        person_generation="ALLOW",
        seed=0,
        image=None,
        model="veo-2.0-generate-001",
        generate_audio=False,
    ):
        model = MODELS_MAP[model]
        # Prepare the instances for the request
        instances = []

        instance = {"prompt": prompt}

        # Add image if provided
        if image is not None:
            image_base64 = tensor_to_base64_string(image)
            if image_base64:
                instance["image"] = {"bytesBase64Encoded": image_base64, "mimeType": "image/png"}

        instances.append(instance)

        # Create parameters dictionary
        parameters = {
            "aspectRatio": aspect_ratio,
            "personGeneration": person_generation,
            "durationSeconds": duration_seconds,
            "enhancePrompt": enhance_prompt,
        }

        # Add optional parameters if provided
        if negative_prompt:
            parameters["negativePrompt"] = negative_prompt
        if seed > 0:
            parameters["seed"] = seed
        # Only add generateAudio for Veo 3 models
        if model.find("veo-2.0") == -1:
            parameters["generateAudio"] = generate_audio
            # force "enhance_prompt" to True for Veo3 models
            parameters["enhancePrompt"] = True

        initial_response = await sync_op(
            cls,
            ApiEndpoint(path=f"/proxy/veo/{model}/generate", method="POST"),
            response_model=VeoGenVidResponse,
            data=VeoGenVidRequest(
                instances=instances,
                parameters=parameters,
            ),
        )

        def status_extractor(response):
            # Only return "completed" if the operation is done, regardless of success or failure
            # We'll check for errors after polling completes
            return "completed" if response.done else "pending"

        poll_response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/veo/{model}/poll", method="POST"),
            response_model=VeoGenVidPollResponse,
            status_extractor=status_extractor,
            data=VeoGenVidPollRequest(
                operationName=initial_response.name,
            ),
            poll_interval=5.0,
            estimated_duration=AVERAGE_DURATION_VIDEO_GEN,
        )

        # Now check for errors in the final response
        # Check for error in poll response
        if poll_response.error:
            raise Exception(f"Veo API error: {poll_response.error.message} (code: {poll_response.error.code})")

        # Check for RAI filtered content
        if (
            hasattr(poll_response.response, "raiMediaFilteredCount")
            and poll_response.response.raiMediaFilteredCount > 0
        ):

            # Extract reason message if available
            if (
                hasattr(poll_response.response, "raiMediaFilteredReasons")
                and poll_response.response.raiMediaFilteredReasons
            ):
                reason = poll_response.response.raiMediaFilteredReasons[0]
                error_message = f"Content filtered by Google's Responsible AI practices: {reason} ({poll_response.response.raiMediaFilteredCount} videos filtered.)"
            else:
                error_message = f"Content filtered by Google's Responsible AI practices ({poll_response.response.raiMediaFilteredCount} videos filtered.)"

            raise Exception(error_message)

        # Extract video data
        if (
            poll_response.response
            and hasattr(poll_response.response, "videos")
            and poll_response.response.videos
            and len(poll_response.response.videos) > 0
        ):
            video = poll_response.response.videos[0]

            # Check if video is provided as base64 or URL
            if hasattr(video, "bytesBase64Encoded") and video.bytesBase64Encoded:
                return IO.NodeOutput(InputImpl.VideoFromFile(BytesIO(base64.b64decode(video.bytesBase64Encoded))))

            if hasattr(video, "gcsUri") and video.gcsUri:
                return IO.NodeOutput(await download_url_to_video_output(video.gcsUri))

            raise Exception("Video returned but no data or URL was provided")
        raise Exception("Video generation completed but no video was returned")


class Veo3VideoGenerationNode(IO.ComfyNode):
    """Generates videos from text prompts using Google's Veo 3 API."""

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Veo3VideoGenerationNode",
            display_name="Google Veo 3 Video Generation",
            category="partner/video/Veo",
            description="Generates videos from text prompts using Google's Veo 3 API",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text description of the video",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["16:9", "9:16"],
                    default="16:9",
                    tooltip="Aspect ratio of the output video",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["720p", "1080p", "4k"],
                    default="720p",
                    tooltip="Output video resolution. 4K is not available for veo-3.1-lite and veo-3.0 models.",
                    optional=True,
                ),
                IO.String.Input(
                    "negative_prompt",
                    multiline=True,
                    default="",
                    tooltip="Negative text prompt to guide what to avoid in the video",
                    optional=True,
                ),
                IO.Int.Input(
                    "duration_seconds",
                    default=8,
                    min=4,
                    max=8,
                    step=2,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Duration of the output video in seconds",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "enhance_prompt",
                    default=True,
                    tooltip="This parameter is deprecated and ignored.",
                    optional=True,
                    advanced=True,
                ),
                IO.Combo.Input(
                    "person_generation",
                    options=["ALLOW", "BLOCK"],
                    default="ALLOW",
                    tooltip="Whether to allow generating people in the video",
                    optional=True,
                    advanced=True,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFF,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed for video generation (0 for random)",
                    optional=True,
                ),
                IO.Image.Input(
                    "image",
                    tooltip="Optional reference image to guide video generation",
                    optional=True,
                ),
                IO.Combo.Input(
                    "model",
                    options=[
                        "veo-3.1-generate",
                        "veo-3.1-fast-generate",
                        "veo-3.1-lite",
                        "veo-3.0-generate-001",
                        "veo-3.0-fast-generate-001",
                    ],
                    tooltip="Veo 3 model to use for video generation",
                    optional=True,
                ),
                IO.Boolean.Input(
                    "generate_audio",
                    default=False,
                    tooltip="Generate audio for the video. Supported by all Veo 3 models.",
                    optional=True,
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
                depends_on=IO.PriceBadgeDepends(widgets=["model", "generate_audio", "resolution", "duration_seconds"]),
                expr="""
                (
                  $m := widgets.model;
                  $r := widgets.resolution;
                  $a := widgets.generate_audio;
                  $seconds := widgets.duration_seconds;
                  $pps :=
                    $contains($m, "lite")
                      ? ($r = "1080p" ? ($a ? 0.08 : 0.05) : ($a ? 0.05 : 0.03))
                    : $contains($m, "3.1-fast")
                      ? ($r = "4k" ? ($a ? 0.30 : 0.25) : $r = "1080p" ? ($a ? 0.12 : 0.10) : ($a ? 0.10 : 0.08))
                    : $contains($m, "3.1-generate")
                      ? ($r = "4k" ? ($a ? 0.60 : 0.40) : ($a ? 0.40 : 0.20))
                    : $contains($m, "3.0-fast")
                      ? ($a ? 0.15 : 0.10)
                    : ($a ? 0.40 : 0.20);
                  {"type":"usd","usd": $pps * $seconds}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt,
        aspect_ratio="16:9",
        resolution="720p",
        negative_prompt="",
        duration_seconds=8,
        enhance_prompt=True,
        person_generation="ALLOW",
        seed=0,
        image=None,
        model="veo-3.0-generate-001",
        generate_audio=False,
    ):
        if resolution == "4k" and ("lite" in model or "3.0" in model):
            raise Exception("4K resolution is not supported by the veo-3.1-lite or veo-3.0 models.")

        model = MODELS_MAP[model]

        instances = [{"prompt": prompt}]
        if image is not None:
            image_base64 = tensor_to_base64_string(image)
            if image_base64:
                instances[0]["image"] = {"bytesBase64Encoded": image_base64, "mimeType": "image/png"}

        parameters = {
            "aspectRatio": aspect_ratio,
            "personGeneration": person_generation,
            "durationSeconds": duration_seconds,
            "enhancePrompt": True,
            "generateAudio": generate_audio,
        }
        if negative_prompt:
            parameters["negativePrompt"] = negative_prompt
        if seed > 0:
            parameters["seed"] = seed
        if "veo-3.1" in model:
            parameters["resolution"] = resolution

        initial_response = await sync_op(
            cls,
            ApiEndpoint(path=f"/proxy/veo/{model}/generate", method="POST"),
            response_model=VeoGenVidResponse,
            data=VeoGenVidRequest(
                instances=instances,
                parameters=parameters,
            ),
        )

        poll_response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/veo/{model}/poll", method="POST"),
            response_model=VeoGenVidPollResponse,
            status_extractor=lambda r: "completed" if r.done else "pending",
            data=VeoGenVidPollRequest(operationName=initial_response.name),
            poll_interval=9.0,
            estimated_duration=AVERAGE_DURATION_VIDEO_GEN,
        )

        if poll_response.error:
            raise Exception(f"Veo API error: {poll_response.error.message} (code: {poll_response.error.code})")

        response = poll_response.response
        filtered_count = response.raiMediaFilteredCount
        if filtered_count:
            reasons = response.raiMediaFilteredReasons or []
            reason_part = f": {reasons[0]}" if reasons else ""
            raise Exception(
                f"Content blocked by Google's Responsible AI filters{reason_part} "
                f"({filtered_count} video{'s' if filtered_count != 1 else ''} filtered)."
            )

        if response.videos:
            video = response.videos[0]
            if video.bytesBase64Encoded:
                return IO.NodeOutput(InputImpl.VideoFromFile(BytesIO(base64.b64decode(video.bytesBase64Encoded))))
            if video.gcsUri:
                return IO.NodeOutput(await download_url_to_video_output(video.gcsUri))
            raise Exception("Video returned but no data or URL was provided")
        raise Exception("Video generation completed but no video was returned")


class Veo3FirstLastFrameNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Veo3FirstLastFrameNode",
            display_name="Google Veo 3 First-Last-Frame to Video",
            category="partner/video/Veo",
            description="Generate video using prompt and first and last frames.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text description of the video",
                ),
                IO.String.Input(
                    "negative_prompt",
                    multiline=True,
                    default="",
                    tooltip="Negative text prompt to guide what to avoid in the video",
                ),
                IO.Combo.Input("resolution", options=["720p", "1080p", "4k"]),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["16:9", "9:16"],
                    default="16:9",
                    tooltip="Aspect ratio of the output video",
                ),
                IO.Int.Input(
                    "duration",
                    default=8,
                    min=4,
                    max=8,
                    step=2,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Duration of the output video in seconds",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFF,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed for video generation",
                ),
                IO.Image.Input("first_frame", tooltip="Start frame"),
                IO.Image.Input("last_frame", tooltip="End frame"),
                IO.Combo.Input(
                    "model",
                    options=["veo-3.1-generate", "veo-3.1-fast-generate", "veo-3.1-lite"],
                ),
                IO.Boolean.Input(
                    "generate_audio",
                    default=True,
                    tooltip="Generate audio for the video.",
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
                depends_on=IO.PriceBadgeDepends(widgets=["model", "generate_audio", "duration", "resolution"]),
                expr="""
                (
                  $m := widgets.model;
                  $r := widgets.resolution;
                  $ga := widgets.generate_audio;
                  $seconds := widgets.duration;
                  $pps :=
                    $contains($m, "lite")
                      ? ($r = "1080p" ? ($ga ? 0.08 : 0.05) : ($ga ? 0.05 : 0.03))
                    : $contains($m, "fast")
                      ? ($r = "4k" ? ($ga ? 0.30 : 0.25) : $r = "1080p" ? ($ga ? 0.12 : 0.10) : ($ga ? 0.10 : 0.08))
                    : ($r = "4k" ? ($ga ? 0.60 : 0.40) : ($ga ? 0.40 : 0.20));
                  {"type":"usd","usd": $pps * $seconds}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        negative_prompt: str,
        resolution: str,
        aspect_ratio: str,
        duration: int,
        seed: int,
        first_frame: Input.Image,
        last_frame: Input.Image,
        model: str,
        generate_audio: bool,
    ):
        if "lite" in model and resolution == "4k":
            raise Exception("4K resolution is not supported by the veo-3.1-lite model.")

        model = MODELS_MAP[model]
        initial_response = await sync_op(
            cls,
            ApiEndpoint(path=f"/proxy/veo/{model}/generate", method="POST"),
            response_model=VeoGenVidResponse,
            data=VeoGenVidRequest(
                instances=[
                    VeoRequestInstance(
                        prompt=prompt,
                        image=VeoRequestInstanceImage(
                            bytesBase64Encoded=tensor_to_base64_string(first_frame), mimeType="image/png"
                        ),
                        lastFrame=VeoRequestInstanceImage(
                            bytesBase64Encoded=tensor_to_base64_string(last_frame), mimeType="image/png"
                        ),
                    ),
                ],
                parameters=VeoRequestParameters(
                    aspectRatio=aspect_ratio,
                    personGeneration="ALLOW",
                    durationSeconds=duration,
                    enhancePrompt=True,  # cannot be False for Veo3
                    seed=seed,
                    generateAudio=generate_audio,
                    negativePrompt=negative_prompt,
                    resolution=resolution,
                ),
            ),
        )
        poll_response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/veo/{model}/poll", method="POST"),
            response_model=VeoGenVidPollResponse,
            status_extractor=lambda r: "completed" if r.done else "pending",
            data=VeoGenVidPollRequest(
                operationName=initial_response.name,
            ),
            poll_interval=9.0,
            estimated_duration=AVERAGE_DURATION_VIDEO_GEN,
        )

        if poll_response.error:
            raise Exception(f"Veo API error: {poll_response.error.message} (code: {poll_response.error.code})")

        response = poll_response.response
        filtered_count = response.raiMediaFilteredCount
        if filtered_count:
            reasons = response.raiMediaFilteredReasons or []
            reason_part = f": {reasons[0]}" if reasons else ""
            raise Exception(
                f"Content blocked by Google's Responsible AI filters{reason_part} "
                f"({filtered_count} video{'s' if filtered_count != 1 else ''} filtered)."
            )

        if response.videos:
            video = response.videos[0]
            if video.bytesBase64Encoded:
                return IO.NodeOutput(InputImpl.VideoFromFile(BytesIO(base64.b64decode(video.bytesBase64Encoded))))
            if video.gcsUri:
                return IO.NodeOutput(await download_url_to_video_output(video.gcsUri))
            raise Exception("Video returned but no data or URL was provided")
        raise Exception("Video generation completed but no video was returned")


class VeoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            VeoVideoGenerationNode,
            Veo3VideoGenerationNode,
            Veo3FirstLastFrameNode,
        ]


async def comfy_entrypoint() -> VeoExtension:
    return VeoExtension()
