import math

from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.clarityai import (
    CrystalUpscalerRequest,
    CrystalUpscalerResult,
    CrystalUpscalerStatusResponse,
    CrystalUpscalerSubmitResponse,
    CrystalVideoUpscalerRequest,
    CrystalVideoUpscalerResult,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_image_tensor,
    download_url_to_video_output,
    get_image_dimensions,
    get_number_of_images,
    poll_op,
    sync_op,
    upload_image_to_comfyapi,
    upload_video_to_comfyapi,
    validate_image_dimensions,
    validate_video_dimensions,
)

CRYSTAL_UPSCALER_ENDPOINT = "/proxy/fal/clarityai/crystal-upscaler"
MAX_OUTPUT_PIXELS = 100_000_000
MAX_OUTPUT_SIDE = 65535
CRYSTAL_VIDEO_UPSCALER_ENDPOINT = "/proxy/fal/clarityai/crystal-video-upscaler"
MAX_VIDEO_FRAME_PIXELS = 9_000_000
MAX_VIDEO_PIXEL_SECONDS = 360_000_000
MAX_VIDEO_SIDE = 5120


class ClarityCrystalUpscaleNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ClarityCrystalUpscaleNode",
            display_name="Clarity AI Crystal Upscale",
            category="partner/image/Clarity AI",
            description="Upscale an image with Clarity AI's Crystal Upscaler, a high-fidelity upscaler that stays "
            "faithful to the original and restores faces, skin, and fine texture. Maximum output: 100 megapixels.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "crystal-upscaler",
                            [
                                IO.Image.Input("image"),
                                IO.Float.Input(
                                    "scale_factor",
                                    default=2.0,
                                    min=1.0,
                                    max=200.0,
                                    step=0.1,
                                    display_mode=IO.NumberDisplay.number,
                                    tooltip="Factor to multiply the image width and height by. "
                                    "The output is limited to 100 megapixels.",
                                ),
                                IO.Int.Input(
                                    "creativity",
                                    default=0,
                                    min=0,
                                    max=10,
                                    display_mode=IO.NumberDisplay.slider,
                                    tooltip="Higher values let the model reconstruct more detail instead of "
                                    "strictly preserving the original. Has no effect on images whose shorter "
                                    "side is 256 pixels or less.",
                                ),
                            ],
                        ),
                    ],
                    tooltip="Model to use.",
                ),
            ],
            outputs=[IO.Image.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(expr='{"type":"usd","usd":0.02288,"format":{"suffix":"/MP"}}'),
        )

    @classmethod
    async def execute(cls, model: dict) -> IO.NodeOutput:
        image = model["image"]
        if get_number_of_images(image) != 1:
            raise ValueError("The image input must contain exactly one image, not a batch.")
        validate_image_dimensions(image, min_width=2, min_height=2)
        height, width = get_image_dimensions(image)
        scale_factor = model["scale_factor"]
        output_width, output_height = round(width * scale_factor), round(height * scale_factor)
        if output_width * output_height > MAX_OUTPUT_PIXELS or max(output_width, output_height) > MAX_OUTPUT_SIDE:
            raise ValueError(
                f"The output would be {output_width}x{output_height} pixels, but at most 100 megapixels and "
                f"{MAX_OUTPUT_SIDE} pixels per side are supported. Use a smaller image or a lower scale factor."
            )
        submit = await sync_op(
            cls,
            ApiEndpoint(path=CRYSTAL_UPSCALER_ENDPOINT, method="POST"),
            response_model=CrystalUpscalerSubmitResponse,
            data=CrystalUpscalerRequest(
                image_url=await upload_image_to_comfyapi(cls, image, total_pixels=None),
                scale_factor=scale_factor,
                creativity=model["creativity"],
                output_format="png",
            ),
        )
        await poll_op(
            cls,
            ApiEndpoint(path=f"{CRYSTAL_UPSCALER_ENDPOINT}/requests/{submit.request_id}/status"),
            response_model=CrystalUpscalerStatusResponse,
            status_extractor=lambda r: r.status,
            completed_statuses=["COMPLETED"],
            queued_statuses=["IN_QUEUE"],
            poll_interval=2,
        )
        result = await sync_op(
            cls,
            ApiEndpoint(path=f"{CRYSTAL_UPSCALER_ENDPOINT}/requests/{submit.request_id}"),
            response_model=CrystalUpscalerResult,
        )
        return IO.NodeOutput(await download_url_to_image_tensor(result.images[0].url))


class ClarityCrystalVideoUpscaleNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ClarityCrystalVideoUpscaleNode",
            display_name="Clarity AI Crystal Video Upscale",
            category="partner/video/Clarity AI",
            description="Upscale a video with Clarity AI's Crystal Video Upscaler, a high-fidelity upscaler that stays "
            "faithful to the source. Output is limited to 9 megapixels per frame (for example 4000x2250), and long "
            "clips come back at a lower resolution. Clips above 30 fps cost 2x, above 60 fps 3x, and so on.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "crystal-video-upscaler",
                            [
                                IO.Video.Input("video"),
                                IO.Float.Input(
                                    "scale_factor",
                                    default=2.0,
                                    min=1.0,
                                    max=200.0,
                                    step=0.1,
                                    display_mode=IO.NumberDisplay.number,
                                    tooltip="Factor to multiply the video width and height by. The output is "
                                    "limited to 9 megapixels per frame and 360 megapixel-seconds per clip, so "
                                    "large or long videos are upscaled by less.",
                                ),
                            ],
                        ),
                    ],
                    tooltip="Model to use.",
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
                expr='{"type":"range_usd","min_usd":0.2965,"max_usd":1.287,"format":{"approximate":true,"suffix":"/s","note":"(1080p-4K output)"}}',
            ),
        )

    @classmethod
    async def execute(cls, model: dict) -> IO.NodeOutput:
        video: Input.Video = model["video"]
        validate_video_dimensions(video, max_width=MAX_VIDEO_SIDE, max_height=MAX_VIDEO_SIDE)
        width, height = video.get_dimensions()
        if width * height > MAX_VIDEO_FRAME_PIXELS:
            raise ValueError(
                f"The video is {width}x{height} pixels, but at most 9 megapixels per frame are supported "
                "(for example 4000x2250). Use a smaller video."
            )
        duration = video.get_duration()
        frame_rate = video.get_frame_rate()
        max_duration = MAX_VIDEO_PIXEL_SECONDS / (width * height * math.ceil(frame_rate / 30))
        if duration > max_duration:
            raise ValueError(
                f"A {width}x{height} video at {float(frame_rate):.2f} fps can be at most {max_duration:.1f} seconds "
                f"long, got {duration:.1f} seconds. Use a shorter, smaller or lower frame rate video."
            )
        submit = await sync_op(
            cls,
            ApiEndpoint(path=CRYSTAL_VIDEO_UPSCALER_ENDPOINT, method="POST"),
            response_model=CrystalUpscalerSubmitResponse,
            data=CrystalVideoUpscalerRequest(
                video_url=await upload_video_to_comfyapi(cls, video),
                scale_factor=min(model["scale_factor"], math.floor(MAX_VIDEO_SIDE / max(width, height) * 100) / 100),
            ),
        )
        await poll_op(
            cls,
            ApiEndpoint(path=f"{CRYSTAL_VIDEO_UPSCALER_ENDPOINT}/requests/{submit.request_id}/status"),
            response_model=CrystalUpscalerStatusResponse,
            status_extractor=lambda r: r.status,
            completed_statuses=["COMPLETED"],
            queued_statuses=["IN_QUEUE"],
        )
        result = await sync_op(
            cls,
            ApiEndpoint(path=f"{CRYSTAL_VIDEO_UPSCALER_ENDPOINT}/requests/{submit.request_id}"),
            response_model=CrystalVideoUpscalerResult,
        )
        return IO.NodeOutput(await download_url_to_video_output(result.video.url))


class ClarityAIExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [ClarityCrystalUpscaleNode, ClarityCrystalVideoUpscaleNode]


async def comfy_entrypoint() -> ClarityAIExtension:
    return ClarityAIExtension()
