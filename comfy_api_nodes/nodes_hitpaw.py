import math

from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.hitpaw import (
    ImageEnhanceTaskCreateRequest,
    InputVideoModel,
    TaskCreateDataResponse,
    TaskCreateResponse,
    TaskStatusPollRequest,
    TaskStatusResponse,
    VideoEnhanceTaskCreateRequest,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_image_tensor,
    download_url_to_video_output,
    downscale_image_tensor,
    get_image_dimensions,
    poll_op,
    sync_op,
    upload_image_to_comfyapi,
    upload_video_to_comfyapi,
    validate_video_duration,
)

VIDEO_MODELS_MODELS_MAP = {
    "Portrait Restore Model (1x)": "portrait_restore_1x",
    "Portrait Restore Model (2x)": "portrait_restore_2x",
    "General Restore Model (1x)": "general_restore_1x",
    "General Restore Model (2x)": "general_restore_2x",
    "General Restore Model (4x)": "general_restore_4x",
    "Ultra HD Model (2x)": "ultrahd_restore_2x",
    "Generative Model (1x)": "generative_1x",
}

# Resolution name to target dimension (shorter side) in pixels
RESOLUTION_TARGET_MAP = {
    "720p": 720,
    "1080p": 1080,
    "2K/QHD": 1440,
    "4K/UHD": 2160,
    "8K": 4320,
}

# Square (1:1) resolutions use standard square dimensions
RESOLUTION_SQUARE_MAP = {
    "720p": 720,
    "1080p": 1080,
    "2K/QHD": 1440,
    "4K/UHD": 2048,  # DCI 4K square
    "8K": 4096,  # DCI 8K square
}

# Models with limited resolution support (no 8K)
LIMITED_RESOLUTION_MODELS = {"Generative Model (1x)"}

# Resolution options for different model types
RESOLUTIONS_LIMITED = ["original", "720p", "1080p", "2K/QHD", "4K/UHD"]
RESOLUTIONS_FULL = ["original", "720p", "1080p", "2K/QHD", "4K/UHD", "8K"]

# Maximum output resolution in pixels
MAX_PIXELS_GENERATIVE = 32_000_000
MAX_MP_GENERATIVE = MAX_PIXELS_GENERATIVE // 1_000_000


class HitPawGeneralImageEnhance(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HitPawGeneralImageEnhance",
            display_name="HitPaw General Image Enhance",
            category="partner/image/HitPaw",
            description="Upscale low-resolution images to super-resolution, eliminate artifacts and noise. "
            f"Maximum output: {MAX_MP_GENERATIVE} megapixels.",
            inputs=[
                IO.Combo.Input("model", options=["generative_portrait", "generative"]),
                IO.Image.Input("image"),
                IO.Combo.Input("upscale_factor", options=[1, 2, 4]),
                IO.Boolean.Input(
                    "auto_downscale",
                    default=False,
                    tooltip="Automatically downscale input image if output would exceed the limit.",
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
                depends_on=IO.PriceBadgeDepends(widgets=["model"]),
                expr="""
                (
                  $prices := {
                    "generative_portrait": {"min": 0.02, "max": 0.06},
                    "generative": {"min": 0.05, "max": 0.15}
                  };
                  $price := $lookup($prices, widgets.model);
                  {
                    "type": "range_usd",
                    "min_usd": $price.min,
                    "max_usd": $price.max
                  }
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        image: Input.Image,
        upscale_factor: int,
        auto_downscale: bool,
    ) -> IO.NodeOutput:
        height, width = get_image_dimensions(image)
        requested_scale = upscale_factor
        output_pixels = height * width * requested_scale * requested_scale
        if output_pixels > MAX_PIXELS_GENERATIVE:
            if auto_downscale:
                input_pixels = width * height
                scale = 1
                max_input_pixels = MAX_PIXELS_GENERATIVE

                for candidate in [4, 2, 1]:
                    if candidate > requested_scale:
                        continue
                    scale_output_pixels = input_pixels * candidate * candidate
                    if scale_output_pixels <= MAX_PIXELS_GENERATIVE:
                        scale = candidate
                        max_input_pixels = None
                        break
                    # Check if we can downscale input by at most 2x to fit
                    downscale_ratio = math.sqrt(scale_output_pixels / MAX_PIXELS_GENERATIVE)
                    if downscale_ratio <= 2.0:
                        scale = candidate
                        max_input_pixels = MAX_PIXELS_GENERATIVE // (candidate * candidate)
                        break

                if max_input_pixels is not None:
                    image = downscale_image_tensor(image, total_pixels=max_input_pixels)
                upscale_factor = scale
            else:
                output_width = width * requested_scale
                output_height = height * requested_scale
                raise ValueError(
                    f"Output size ({output_width}x{output_height} = {output_pixels:,} pixels) "
                    f"exceeds maximum allowed size of {MAX_PIXELS_GENERATIVE:,} pixels ({MAX_MP_GENERATIVE}MP). "
                    f"Enable auto_downscale or use a smaller input image or a lower upscale factor."
                )

        initial_res = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/hitpaw/api/photo-enhancer", method="POST"),
            response_model=TaskCreateResponse,
            data=ImageEnhanceTaskCreateRequest(
                model_name=f"{model}_{upscale_factor}x",
                img_url=await upload_image_to_comfyapi(cls, image, total_pixels=None),
            ),
            wait_label="Creating task",
            final_label_on_success="Task created",
        )
        if initial_res.code != 200:
            raise ValueError(f"Task creation failed with code {initial_res.code}: {initial_res.message}")
        request_price = initial_res.data.consume_coins / 1000
        final_response = await poll_op(
            cls,
            ApiEndpoint(path="/proxy/hitpaw/api/task-status", method="POST"),
            data=TaskCreateDataResponse(job_id=initial_res.data.job_id),
            response_model=TaskStatusResponse,
            status_extractor=lambda x: x.data.status,
            price_extractor=lambda x: request_price,
            poll_interval=10.0,
        )
        return IO.NodeOutput(await download_url_to_image_tensor(final_response.data.res_url))


class HitPawVideoEnhance(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        model_options = []
        for model_name in VIDEO_MODELS_MODELS_MAP:
            if model_name in LIMITED_RESOLUTION_MODELS:
                resolutions = RESOLUTIONS_LIMITED
            else:
                resolutions = RESOLUTIONS_FULL
            model_options.append(
                IO.DynamicCombo.Option(
                    model_name,
                    [IO.Combo.Input("resolution", options=resolutions)],
                )
            )

        return IO.Schema(
            node_id="HitPawVideoEnhance",
            display_name="HitPaw Video Enhance",
            category="partner/video/HitPaw",
            description="Upscale low-resolution videos to high resolution, eliminate artifacts and noise. "
            "Prices shown are per second of video.",
            inputs=[
                IO.DynamicCombo.Input("model", options=model_options),
                IO.Video.Input("video"),
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
                  $m := $lookup(widgets, "model");
                  $res := $lookup(widgets, "model.resolution");
                  $standard_model_prices := {
                    "original": {"min": 0.01, "max": 0.198},
                    "720p": {"min": 0.01, "max": 0.06},
                    "1080p": {"min": 0.015, "max": 0.09},
                    "2k/qhd": {"min": 0.02, "max": 0.117},
                    "4k/uhd": {"min": 0.025, "max": 0.152},
                    "8k": {"min": 0.033, "max": 0.198}
                  };
                  $ultra_hd_model_prices := {
                    "original": {"min": 0.015, "max": 0.264},
                    "720p": {"min": 0.015, "max": 0.092},
                    "1080p": {"min": 0.02, "max": 0.12},
                    "2k/qhd": {"min": 0.026, "max": 0.156},
                    "4k/uhd": {"min": 0.034, "max": 0.203},
                    "8k": {"min": 0.044, "max": 0.264}
                  };
                  $generative_model_prices := {
                    "original": {"min": 0.015, "max": 0.338},
                    "720p": {"min": 0.008, "max": 0.090},
                    "1080p": {"min": 0.05, "max": 0.15},
                    "2k/qhd": {"min": 0.038, "max": 0.225},
                    "4k/uhd": {"min": 0.056, "max": 0.338}
                  };
                  $prices := $contains($m, "ultra hd") ? $ultra_hd_model_prices :
                             $contains($m, "generative") ? $generative_model_prices :
                             $standard_model_prices;
                  $price := $lookup($prices, $res);
                  {
                    "type": "range_usd",
                    "min_usd": $price.min,
                    "max_usd": $price.max,
                    "format": {"approximate": true, "suffix": "/second"}
                  }
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: InputVideoModel,
        video: Input.Video,
    ) -> IO.NodeOutput:
        validate_video_duration(video, min_duration=0.5, max_duration=60 * 60)
        resolution = model["resolution"]
        src_width, src_height = video.get_dimensions()

        if resolution == "original":
            output_width = src_width
            output_height = src_height
        else:
            if src_width == src_height:
                target_size = RESOLUTION_SQUARE_MAP[resolution]
                if target_size < src_width:
                    raise ValueError(
                        f"Selected resolution {resolution} ({target_size}x{target_size}) is smaller than "
                        f"the input video ({src_width}x{src_height}). Please select a higher resolution or 'original'."
                    )
                output_width = target_size
                output_height = target_size
            else:
                min_dimension = min(src_width, src_height)
                target_size = RESOLUTION_TARGET_MAP[resolution]
                if target_size < min_dimension:
                    raise ValueError(
                        f"Selected resolution {resolution} ({target_size}p) is smaller than "
                        f"the input video's shorter dimension ({min_dimension}p). "
                        f"Please select a higher resolution or 'original'."
                    )
                if src_width > src_height:
                    output_height = target_size
                    output_width = int(target_size * (src_width / src_height))
                else:
                    output_width = target_size
                    output_height = int(target_size * (src_height / src_width))
        initial_res = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/hitpaw/api/video-enhancer", method="POST"),
            response_model=TaskCreateResponse,
            data=VideoEnhanceTaskCreateRequest(
                video_url=await upload_video_to_comfyapi(cls, video),
                resolution=[output_width, output_height],
                original_resolution=[src_width, src_height],
                model_name=VIDEO_MODELS_MODELS_MAP[model["model"]],
            ),
            wait_label="Creating task",
            final_label_on_success="Task created",
        )
        request_price = initial_res.data.consume_coins / 1000
        if initial_res.code != 200:
            raise ValueError(f"Task creation failed with code {initial_res.code}: {initial_res.message}")
        final_response = await poll_op(
            cls,
            ApiEndpoint(path="/proxy/hitpaw/api/task-status", method="POST"),
            data=TaskStatusPollRequest(job_id=initial_res.data.job_id),
            response_model=TaskStatusResponse,
            status_extractor=lambda x: x.data.status,
            price_extractor=lambda x: request_price,
            poll_interval=10.0,
        )
        return IO.NodeOutput(await download_url_to_video_output(final_response.data.res_url))


class HitPawExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            HitPawGeneralImageEnhance,
            HitPawVideoEnhance,
        ]


async def comfy_entrypoint() -> HitPawExtension:
    return HitPawExtension()
