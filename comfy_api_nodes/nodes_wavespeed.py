from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.wavespeed import (
    FlashVSRRequest,
    TaskCreatedResponse,
    TaskResultResponse,
    SeedVR2ImageRequest,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_video_output,
    poll_op,
    sync_op,
    upload_video_to_comfyapi,
    validate_container_format_is_mp4,
    validate_video_duration,
    upload_images_to_comfyapi,
    get_number_of_images,
    download_url_to_image_tensor,
)


class WavespeedFlashVSRNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="WavespeedFlashVSRNode",
            display_name="FlashVSR Video Upscale",
            category="partner/video/WaveSpeed",
            description="Fast, high-quality video upscaler that "
            "boosts resolution and restores clarity for low-resolution or blurry footage.",
            inputs=[
                IO.Video.Input("video"),
                IO.Combo.Input("target_resolution", options=["720p", "1080p", "2K", "4K"]),
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
                depends_on=IO.PriceBadgeDepends(widgets=["target_resolution"]),
                expr="""
                (
                  $price_for_1sec := {"720p": 0.012, "1080p": 0.018, "2k": 0.024, "4k": 0.032};
                  {
                    "type":"usd",
                    "usd": $lookup($price_for_1sec, widgets.target_resolution),
                    "format":{"suffix": "/second", "approximate": true}
                  }
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        video: Input.Video,
        target_resolution: str,
    ) -> IO.NodeOutput:
        validate_container_format_is_mp4(video)
        validate_video_duration(video, min_duration=5, max_duration=60 * 10)
        initial_res = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/wavespeed/api/v3/wavespeed-ai/flashvsr", method="POST"),
            response_model=TaskCreatedResponse,
            data=FlashVSRRequest(
                target_resolution=target_resolution.lower(),
                video=await upload_video_to_comfyapi(cls, video),
                duration=video.get_duration(),
            ),
        )
        if initial_res.code != 200:
            raise ValueError(f"Task creation fails with code={initial_res.code} and message={initial_res.message}")
        final_response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/wavespeed/api/v3/predictions/{initial_res.data.id}/result"),
            response_model=TaskResultResponse,
            status_extractor=lambda x: "failed" if x.data is None else x.data.status,
            poll_interval=10.0,
        )
        if final_response.code != 200:
            raise ValueError(
                f"Task processing failed with code={final_response.code} and message={final_response.message}"
            )
        return IO.NodeOutput(await download_url_to_video_output(final_response.data.outputs[0]))


class WavespeedImageUpscaleNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="WavespeedImageUpscaleNode",
            display_name="WaveSpeed Image Upscale",
            category="partner/image/WaveSpeed",
            description="Boost image resolution and quality, upscaling photos to 4K or 8K for sharp, detailed results.",
            inputs=[
                IO.Combo.Input("model", options=["SeedVR2", "Ultimate"]),
                IO.Image.Input("image"),
                IO.Combo.Input("target_resolution", options=["2K", "4K", "8K"]),
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
                depends_on=IO.PriceBadgeDepends(widgets=["model"]),
                expr="""
                (
                  $prices := {"seedvr2": 0.01, "ultimate": 0.06};
                  {"type":"usd", "usd": $lookup($prices, widgets.model)}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        image: Input.Image,
        target_resolution: str,
    ) -> IO.NodeOutput:
        if get_number_of_images(image) != 1:
            raise ValueError("Exactly one input image is required.")
        if model == "SeedVR2":
            model_path = "seedvr2/image"
        else:
            model_path = "ultimate-image-upscaler"
        initial_res = await sync_op(
            cls,
            ApiEndpoint(path=f"/proxy/wavespeed/api/v3/wavespeed-ai/{model_path}", method="POST"),
            response_model=TaskCreatedResponse,
            data=SeedVR2ImageRequest(
                target_resolution=target_resolution.lower(),
                image=(await upload_images_to_comfyapi(cls, image, max_images=1))[0],
            ),
        )
        if initial_res.code != 200:
            raise ValueError(f"Task creation fails with code={initial_res.code} and message={initial_res.message}")
        final_response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/wavespeed/api/v3/predictions/{initial_res.data.id}/result"),
            response_model=TaskResultResponse,
            status_extractor=lambda x: "failed" if x.data is None else x.data.status,
            poll_interval=10.0,
        )
        if final_response.code != 200:
            raise ValueError(
                f"Task processing failed with code={final_response.code} and message={final_response.message}"
            )
        return IO.NodeOutput(await download_url_to_image_tensor(final_response.data.outputs[0]))


class WavespeedExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            WavespeedFlashVSRNode,
            WavespeedImageUpscaleNode,
        ]


async def comfy_entrypoint() -> WavespeedExtension:
    return WavespeedExtension()
