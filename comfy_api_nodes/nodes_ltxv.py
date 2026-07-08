from io import BytesIO

from pydantic import BaseModel, Field
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input, InputImpl
from comfy_api_nodes.util import (
    ApiEndpoint,
    get_number_of_images,
    sync_op_raw,
    upload_images_to_comfyapi,
    validate_string,
)

MODELS_MAP = {
    "LTX-2 (Pro)": "ltx-2-pro",
    "LTX-2 (Fast)": "ltx-2-fast",
}


class ExecuteTaskRequest(BaseModel):
    prompt: str = Field(...)
    model: str = Field(...)
    duration: int = Field(...)
    resolution: str = Field(...)
    fps: int | None = Field(25)
    generate_audio: bool | None = Field(True)
    image_uri: str | None = Field(None)


PRICE_BADGE = IO.PriceBadge(
    depends_on=IO.PriceBadgeDepends(widgets=["model", "duration", "resolution"]),
    expr="""
    (
      $prices := {
        "ltx-2 (pro)": {"1920x1080":0.06,"2560x1440":0.12,"3840x2160":0.24},
        "ltx-2 (fast)": {"1920x1080":0.04,"2560x1440":0.08,"3840x2160":0.16}
      };
      $modelPrices := $lookup($prices, $lowercase(widgets.model));
      $pps := $lookup($modelPrices, widgets.resolution);
      {"type":"usd","usd": $pps * widgets.duration}
    )
    """,
)


class TextToVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="LtxvApiTextToVideo",
            display_name="LTXV Text To Video",
            category="partner/video/LTXV",
            description="Professional-quality videos with customizable duration and resolution.",
            inputs=[
                IO.Combo.Input("model", options=list(MODELS_MAP.keys())),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                ),
                IO.Combo.Input("duration", options=[6, 8, 10, 12, 14, 16, 18, 20], default=8),
                IO.Combo.Input(
                    "resolution",
                    options=[
                        "1920x1080",
                        "2560x1440",
                        "3840x2160",
                    ],
                ),
                IO.Combo.Input("fps", options=[25, 50], default=25),
                IO.Boolean.Input(
                    "generate_audio",
                    default=False,
                    optional=True,
                    tooltip="When true, the generated video will include AI-generated audio matching the scene.",
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
            price_badge=PRICE_BADGE,
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        duration: int,
        resolution: str,
        fps: int = 25,
        generate_audio: bool = False,
    ) -> IO.NodeOutput:
        validate_string(prompt, min_length=1, max_length=10000)
        if duration > 10 and (model != "LTX-2 (Fast)" or resolution != "1920x1080" or fps != 25):
            raise ValueError(
                "Durations over 10s are only available for the Fast model at 1920x1080 resolution and 25 FPS."
            )
        response = await sync_op_raw(
            cls,
            ApiEndpoint("/proxy/ltx/v1/text-to-video", "POST"),
            data=ExecuteTaskRequest(
                prompt=prompt,
                model=MODELS_MAP[model],
                duration=duration,
                resolution=resolution,
                fps=fps,
                generate_audio=generate_audio,
            ),
            as_binary=True,
            max_retries=1,
        )
        return IO.NodeOutput(InputImpl.VideoFromFile(BytesIO(response)))


class ImageToVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="LtxvApiImageToVideo",
            display_name="LTXV Image To Video",
            category="partner/video/LTXV",
            description="Professional-quality videos with customizable duration and resolution based on start image.",
            inputs=[
                IO.Image.Input("image", tooltip="First frame to be used for the video."),
                IO.Combo.Input("model", options=list(MODELS_MAP.keys())),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                ),
                IO.Combo.Input("duration", options=[6, 8, 10, 12, 14, 16, 18, 20], default=8),
                IO.Combo.Input(
                    "resolution",
                    options=[
                        "1920x1080",
                        "2560x1440",
                        "3840x2160",
                    ],
                ),
                IO.Combo.Input("fps", options=[25, 50], default=25),
                IO.Boolean.Input(
                    "generate_audio",
                    default=False,
                    optional=True,
                    tooltip="When true, the generated video will include AI-generated audio matching the scene.",
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
            price_badge=PRICE_BADGE,
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        model: str,
        prompt: str,
        duration: int,
        resolution: str,
        fps: int = 25,
        generate_audio: bool = False,
    ) -> IO.NodeOutput:
        validate_string(prompt, min_length=1, max_length=10000)
        if duration > 10 and (model != "LTX-2 (Fast)" or resolution != "1920x1080" or fps != 25):
            raise ValueError(
                "Durations over 10s are only available for the Fast model at 1920x1080 resolution and 25 FPS."
            )
        if get_number_of_images(image) != 1:
            raise ValueError("Currently only one input image is supported.")
        response = await sync_op_raw(
            cls,
            ApiEndpoint("/proxy/ltx/v1/image-to-video", "POST"),
            data=ExecuteTaskRequest(
                image_uri=(await upload_images_to_comfyapi(cls, image, max_images=1, mime_type="image/png"))[0],
                prompt=prompt,
                model=MODELS_MAP[model],
                duration=duration,
                resolution=resolution,
                fps=fps,
                generate_audio=generate_audio,
            ),
            as_binary=True,
            max_retries=1,
        )
        return IO.NodeOutput(InputImpl.VideoFromFile(BytesIO(response)))


class LtxvApiExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            TextToVideoNode,
            ImageToVideoNode,
        ]


async def comfy_entrypoint() -> LtxvApiExtension:
    return LtxvApiExtension()
