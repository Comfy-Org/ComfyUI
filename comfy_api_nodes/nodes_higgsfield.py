from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension
from comfy_api_nodes.apis.higgsfield import HiggsfieldRequestStatus, MotionTransferRequest
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_video_output,
    get_number_of_images,
    poll_op,
    sync_op,
    upload_images_to_comfyapi,
    upload_video_to_comfyapi,
    validate_string,
    validate_video_duration,
)

MOTION_TRANSFER_ENDPOINT = "/proxy/higgsfield/higgsfield/genjutsu/motion-transfer/v1.0"


class HiggsfieldMotionTransferNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HiggsfieldMotionTransferNode",
            display_name="Higgsfield Genjutsu Motion Transfer",
            category="partner/video/Higgsfield",
            description="Recast a video with new characters, locations or styles from reference images while "
            "keeping its motion, camera movement and timing.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "genjutsu-v1.0",
                            [
                                IO.Video.Input(
                                    "video",
                                    tooltip="Source video whose motion, camera movement, timing and audio are "
                                    "kept. At least 4 seconds; only the first 30 seconds are used.",
                                ),
                                IO.Autogrow.Input(
                                    "reference_images",
                                    template=IO.Autogrow.TemplateNames(
                                        IO.Image.Input("reference_image"),
                                        names=[f"image_{i}" for i in range(1, 9)],
                                        min=1,
                                    ),
                                    tooltip="1 to 8 images of the characters, locations or styles to use.",
                                ),
                                IO.String.Input(
                                    "prompt",
                                    multiline=True,
                                    default="",
                                    tooltip="Optional. Describe the result, for example which reference "
                                    "replaces which person, or a new setting.",
                                ),
                                IO.Combo.Input("resolution", options=["480p", "720p", "1080p"], default="720p"),
                                IO.Int.Input(
                                    "seed",
                                    default=42,
                                    min=0,
                                    max=2147483647,
                                    control_after_generate=True,
                                    tooltip="Seed controls whether the node should re-run; "
                                    "results are non-deterministic regardless of seed.",
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
                depends_on=IO.PriceBadgeDepends(widgets=["model.resolution"]),
                expr="""
                (
                  $rates := {"480p": 0.45474, "720p": 0.97383, "1080p": 2.33376};
                  {"type":"usd","usd": $lookup($rates, $lookup(widgets, "model.resolution")),
                   "format":{"suffix":"/second"}}
                )
                """,
            ),
        )

    @classmethod
    async def execute(cls, model: dict) -> IO.NodeOutput:
        video = model["video"]
        validate_video_duration(video, min_duration=4)
        if video.get_duration() > 30:
            video = video.as_trimmed(duration=30)
        images = list(model["reference_images"].values())
        if sum(get_number_of_images(i) for i in images) > 8:
            raise ValueError("At most 8 reference images are supported.")
        validate_string(model["prompt"], strip_whitespace=False, max_length=10000)
        submit = await sync_op(
            cls,
            ApiEndpoint(path=MOTION_TRANSFER_ENDPOINT, method="POST"),
            response_model=HiggsfieldRequestStatus,
            data=MotionTransferRequest(
                prompt=model["prompt"],
                video_url=await upload_video_to_comfyapi(cls, video),
                image_urls=await upload_images_to_comfyapi(cls, images, max_images=8),
                resolution=model["resolution"],
            ),
        )
        result = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/higgsfield/requests/{submit.request_id}/status"),
            response_model=HiggsfieldRequestStatus,
            status_extractor=lambda r: r.status,
            completed_statuses=["completed", "failed", "nsfw", "canceled"],
            queued_statuses=["queued"],
            poll_interval=10,
        )
        if result.status == "nsfw":
            raise ValueError("Higgsfield's content moderation rejected the input or the result.")
        if result.status != "completed":
            raise ValueError(f"Higgsfield could not generate the video: {result.error or result.status}")
        return IO.NodeOutput(await download_url_to_video_output(result.video.url))


class HiggsfieldExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [HiggsfieldMotionTransferNode]


async def comfy_entrypoint() -> HiggsfieldExtension:
    return HiggsfieldExtension()
