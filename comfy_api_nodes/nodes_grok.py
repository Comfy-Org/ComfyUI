import torch
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.grok import (
    ImageEditRequest,
    ImageGenerationRequest,
    ImageGenerationResponse,
    InputUrlObject,
    VideoEditRequest,
    VideoExtensionRequest,
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoStatusResponse,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_image_tensor,
    download_url_to_video_output,
    get_fs_object_size,
    get_number_of_images,
    poll_op,
    sync_op,
    tensor_to_base64_string,
    upload_images_to_comfyapi,
    upload_video_to_comfyapi,
    validate_string,
    validate_video_duration,
)


_GROK_VIDEO_MODEL_API_IDS = {
    "grok-imagine-video-1.5": "grok-imagine-video-1.5",
}


def _extract_grok_price(response) -> float | None:
    if response.usage and response.usage.cost_in_usd_ticks is not None:
        return response.usage.cost_in_usd_ticks / 10_000_000_000
    return None


def _extract_grok_video_price(response) -> float | None:
    price = _extract_grok_price(response)
    if price is not None:
        return price * 1.43
    return None


class GrokImageNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GrokImageNode",
            display_name="Grok Image",
            category="partner/image/Grok",
            description="Generate images using Grok based on a text prompt",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=[
                        "grok-imagine-image-quality",
                        "grok-imagine-image-pro",
                        "grok-imagine-image",
                    ],
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="The text prompt used to generate the image",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=[
                        "1:1",
                        "2:3",
                        "3:2",
                        "3:4",
                        "4:3",
                        "9:16",
                        "16:9",
                        "9:19.5",
                        "19.5:9",
                        "9:20",
                        "20:9",
                        "1:2",
                        "2:1",
                    ],
                ),
                IO.Int.Input(
                    "number_of_images",
                    default=1,
                    min=1,
                    max=10,
                    step=1,
                    tooltip="Number of images to generate",
                    display_mode=IO.NumberDisplay.number,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to determine if node should re-run; "
                    "actual results are nondeterministic regardless of seed.",
                ),
                IO.Combo.Input("resolution", options=["1K", "2K"], optional=True),
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
                depends_on=IO.PriceBadgeDepends(widgets=["model", "number_of_images", "resolution"]),
                expr="""
                (
                  $rate := widgets.model = "grok-imagine-image-quality"
                    ? (widgets.resolution = "1k" ? 0.05 : 0.07)
                    : ($contains(widgets.model, "pro") ? 0.07 : 0.02);
                  {"type":"usd","usd": $rate * widgets.number_of_images}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        aspect_ratio: str,
        number_of_images: int,
        seed: int,
        resolution: str = "1K",
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/xai/v1/images/generations", method="POST"),
            data=ImageGenerationRequest(
                model=model,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                n=number_of_images,
                seed=seed,
                resolution=resolution.lower(),
            ),
            response_model=ImageGenerationResponse,
            price_extractor=_extract_grok_price,
        )
        if len(response.data) == 1:
            return IO.NodeOutput(await download_url_to_image_tensor(response.data[0].url))
        return IO.NodeOutput(
            torch.cat(
                [await download_url_to_image_tensor(i) for i in [str(d.url) for d in response.data if d.url]],
            )
        )


_GROK_IMAGE_EDIT_ASPECT_RATIO_OPTIONS = [
    "auto",
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "9:16",
    "16:9",
    "9:19.5",
    "19.5:9",
    "9:20",
    "20:9",
    "1:2",
    "2:1",
]


def _grok_image_edit_model_inputs(*, max_ref_images: int, with_aspect_ratio: bool):
    inputs = [
        IO.Autogrow.Input(
            "images",
            template=IO.Autogrow.TemplateNames(
                IO.Image.Input("image"),
                names=[f"image_{i}" for i in range(1, max_ref_images + 1)],
                min=1,
            ),
            tooltip=(
                "Reference image to edit."
                if max_ref_images == 1
                else f"Reference image(s) to edit. Up to {max_ref_images} images."
            ),
        ),
        IO.Combo.Input("resolution", options=["1K", "2K"]),
        IO.Int.Input(
            "number_of_images",
            default=1,
            min=1,
            max=10,
            step=1,
            tooltip="Number of edited images to generate",
            display_mode=IO.NumberDisplay.number,
        ),
    ]
    if with_aspect_ratio:
        inputs.append(
            IO.Combo.Input(
                "aspect_ratio",
                options=_GROK_IMAGE_EDIT_ASPECT_RATIO_OPTIONS,
                tooltip="Only allowed when multiple images are connected.",
            )
        )
    return inputs


class GrokImageEditNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GrokImageEditNode",
            display_name="Grok Image Edit",
            category="partner/image/Grok",
            description="Modify an existing image based on a text prompt",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=[
                        "grok-imagine-image-quality",
                        "grok-imagine-image-pro",
                        "grok-imagine-image",
                    ],
                ),
                IO.Image.Input("image", display_name="images"),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="The text prompt used to generate the image",
                ),
                IO.Combo.Input("resolution", options=["1K", "2K"]),
                IO.Int.Input(
                    "number_of_images",
                    default=1,
                    min=1,
                    max=10,
                    step=1,
                    tooltip="Number of edited images to generate",
                    display_mode=IO.NumberDisplay.number,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to determine if node should re-run; "
                    "actual results are nondeterministic regardless of seed.",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=[
                        "auto",
                        "1:1",
                        "2:3",
                        "3:2",
                        "3:4",
                        "4:3",
                        "9:16",
                        "16:9",
                        "9:19.5",
                        "19.5:9",
                        "9:20",
                        "20:9",
                        "1:2",
                        "2:1",
                    ],
                    optional=True,
                    tooltip="Only allowed when multiple images are connected to the image input.",
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
                depends_on=IO.PriceBadgeDepends(widgets=["model", "number_of_images", "resolution"]),
                expr="""
                (
                  $isQualityModel := widgets.model = "grok-imagine-image-quality";
                  $isPro := $contains(widgets.model, "pro");
                  $rate := $isQualityModel
                    ? (widgets.resolution = "1k" ? 0.05 : 0.07)
                    : ($isPro ? 0.07 : 0.02);
                  $base := $isQualityModel ? 0.01 : 0.002;
                  $output := $rate * widgets.number_of_images;
                  $isPro
                    ? {"type":"usd","usd": $base + $output}
                    : {"type":"range_usd","min_usd": $base + $output, "max_usd": 3 * $base + $output}
                )
                """,
            ),
            is_deprecated=True,
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        image: Input.Image,
        prompt: str,
        resolution: str,
        number_of_images: int,
        seed: int,
        aspect_ratio: str = "auto",
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        if model == "grok-imagine-image-pro":
            if get_number_of_images(image) > 1:
                raise ValueError("The pro model supports only 1 input image.")
        elif get_number_of_images(image) > 3:
            raise ValueError("A maximum of 3 input images is supported.")
        if aspect_ratio != "auto" and get_number_of_images(image) == 1:
            raise ValueError(
                "Custom aspect ratio is only allowed when multiple images are connected to the image input."
            )
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/xai/v1/images/edits", method="POST"),
            data=ImageEditRequest(
                model=model,
                images=[InputUrlObject(url=f"data:image/png;base64,{tensor_to_base64_string(i)}") for i in image],
                prompt=prompt,
                resolution=resolution.lower(),
                n=number_of_images,
                seed=seed,
                aspect_ratio=None if aspect_ratio == "auto" else aspect_ratio,
            ),
            response_model=ImageGenerationResponse,
            price_extractor=_extract_grok_price,
        )
        if len(response.data) == 1:
            return IO.NodeOutput(await download_url_to_image_tensor(response.data[0].url))
        return IO.NodeOutput(
            torch.cat(
                [await download_url_to_image_tensor(i) for i in [str(d.url) for d in response.data if d.url]],
            )
        )


class GrokImageEditNodeV2(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GrokImageEditNodeV2",
            display_name="Grok Image Edit",
            category="partner/image/Grok",
            description="Modify an existing image based on a text prompt",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="The text prompt used to generate the image",
                ),
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "grok-imagine-image-quality",
                            _grok_image_edit_model_inputs(max_ref_images=3, with_aspect_ratio=True),
                        ),
                        IO.DynamicCombo.Option(
                            "grok-imagine-image-pro",
                            _grok_image_edit_model_inputs(max_ref_images=1, with_aspect_ratio=False),
                        ),
                        IO.DynamicCombo.Option(
                            "grok-imagine-image",
                            _grok_image_edit_model_inputs(max_ref_images=3, with_aspect_ratio=True),
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
                    tooltip="Seed to determine if node should re-run; "
                    "actual results are nondeterministic regardless of seed.",
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
                depends_on=IO.PriceBadgeDepends(
                    widgets=["model", "model.resolution", "model.number_of_images"],
                ),
                expr="""
                (
                  $isQualityModel := widgets.model = "grok-imagine-image-quality";
                  $isPro := $contains(widgets.model, "pro");
                  $res := $lookup(widgets, "model.resolution");
                  $n := $lookup(widgets, "model.number_of_images");
                  $rate := $isQualityModel
                    ? ($res = "1k" ? 0.05 : 0.07)
                    : ($isPro ? 0.07 : 0.02);
                  $base := $isQualityModel ? 0.01 : 0.002;
                  $output := $rate * $n;
                  $isPro
                    ? {"type":"usd","usd": $base + $output}
                    : {"type":"range_usd","min_usd": $base + $output, "max_usd": 3 * $base + $output}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        model: dict,
        seed: int,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        model_id = model["model"]
        resolution = model["resolution"]
        number_of_images = model["number_of_images"]
        images_dict = model.get("images") or {}
        aspect_ratio = model.get("aspect_ratio", "auto")

        image_tensors: list[Input.Image] = [t for t in images_dict.values() if t is not None]
        n_images = sum(get_number_of_images(t) for t in image_tensors)
        if n_images < 1:
            raise ValueError("At least one image is required for editing.")
        if model_id == "grok-imagine-image-pro" and n_images > 1:
            raise ValueError("The pro model supports only 1 input image.")
        if model_id != "grok-imagine-image-pro" and n_images > 3:
            raise ValueError("A maximum of 3 input images is supported.")
        if aspect_ratio != "auto" and n_images == 1:
            raise ValueError(
                "Custom aspect ratio is only allowed when multiple images are connected to the image input."
            )

        flat_tensors: list[torch.Tensor] = []
        for tensor in image_tensors:
            if len(tensor.shape) == 4:
                flat_tensors.extend(tensor[i] for i in range(tensor.shape[0]))
            else:
                flat_tensors.append(tensor)

        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/xai/v1/images/edits", method="POST"),
            data=ImageEditRequest(
                model=model_id,
                images=[
                    InputUrlObject(url=f"data:image/png;base64,{tensor_to_base64_string(i)}") for i in flat_tensors
                ],
                prompt=prompt,
                resolution=resolution.lower(),
                n=number_of_images,
                seed=seed,
                aspect_ratio=None if aspect_ratio == "auto" else aspect_ratio,
            ),
            response_model=ImageGenerationResponse,
            price_extractor=_extract_grok_price,
        )
        if len(response.data) == 1:
            return IO.NodeOutput(await download_url_to_image_tensor(response.data[0].url))
        return IO.NodeOutput(
            torch.cat(
                [await download_url_to_image_tensor(i) for i in [str(d.url) for d in response.data if d.url]],
            )
        )


class GrokVideoNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GrokVideoNode",
            display_name="Grok Video",
            category="partner/video/Grok",
            description="Generate video from a prompt or an image",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=["grok-imagine-video", "grok-imagine-video-1.5"],
                    tooltip="grok-imagine-video-1.5 currently always requires an input image.",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="Text description of the desired video.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["480p", "720p", "1080p"],
                    tooltip="The resolution of the output video. 1080p is only available for grok-imagine-video-1.5.",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["auto", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16"],
                    tooltip="The aspect ratio of the output video.",
                ),
                IO.Int.Input(
                    "duration",
                    default=6,
                    min=1,
                    max=15,
                    step=1,
                    tooltip="The duration of the output video in seconds.",
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
                    tooltip="Seed to determine if node should re-run; "
                    "actual results are nondeterministic regardless of seed.",
                ),
                IO.Image.Input(
                    "image",
                    optional=True,
                    tooltip="Optional starting image for grok-imagine-video. Required for grok-imagine-video-1.5.",
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
                depends_on=IO.PriceBadgeDepends(widgets=["model", "duration", "resolution"], inputs=["image"]),
                expr="""
                (
                  $is15 := $contains(widgets.model, "1.5");
                  $rate := $is15
                    ? (widgets.resolution = "1080p" ? 0.25 : (widgets.resolution = "720p" ? 0.14 : 0.08))
                    : (widgets.resolution = "720p" ? 0.07 : 0.05);
                  $imgCost := $is15 ? 0.01 : 0.002;
                  $base := $rate * widgets.duration;
                  $total := inputs.image.connected ? $base + $imgCost : $base;
                  {"type":"usd","usd": $is15 ? $total * 1.43 : $total}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
        duration: int,
        seed: int,
        image: Input.Image | None = None,
    ) -> IO.NodeOutput:
        if image is None and model == "grok-imagine-video-1.5":
            raise ValueError(f"The '{model}' model requires an input image; connect one to the 'image' input.")
        if resolution == "1080p" and model != "grok-imagine-video-1.5":
            raise ValueError(f"1080p resolution is only available for grok-imagine-video-1.5, not '{model}'.")
        image_url = None
        if image is not None:
            if get_number_of_images(image) != 1:
                raise ValueError("Only one input image is supported.")
            image_url = InputUrlObject(url=f"data:image/png;base64,{tensor_to_base64_string(image)}")
        validate_string(prompt, strip_whitespace=True, min_length=1)
        initial_response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/xai/v1/videos/generations", method="POST"),
            data=VideoGenerationRequest(
                model=_GROK_VIDEO_MODEL_API_IDS.get(model, model),
                image=image_url,
                prompt=prompt,
                resolution=resolution,
                duration=duration,
                aspect_ratio=None if aspect_ratio == "auto" else aspect_ratio,
                seed=seed,
            ),
            response_model=VideoGenerationResponse,
        )
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/xai/v1/videos/{initial_response.request_id}"),
            status_extractor=lambda r: r.status if r.status is not None else "complete",
            response_model=VideoStatusResponse,
            price_extractor=_extract_grok_video_price if model == "grok-imagine-video-1.5" else _extract_grok_price,
        )
        return IO.NodeOutput(await download_url_to_video_output(response.video.url))


class GrokVideoEditNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GrokVideoEditNode",
            display_name="Grok Video Edit",
            category="partner/video/Grok",
            description="Edit an existing video based on a text prompt.",
            inputs=[
                IO.Combo.Input("model", options=["grok-imagine-video"]),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="Text description of the desired video.",
                ),
                IO.Video.Input("video", tooltip="Maximum supported duration is 8.7 seconds and 50MB file size."),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to determine if node should re-run; "
                    "actual results are nondeterministic regardless of seed.",
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
                expr="""{"type":"usd","usd": 0.06, "format": {"suffix": "/sec", "approximate": true}}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        video: Input.Video,
        seed: int,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        validate_video_duration(video, min_duration=1, max_duration=8.7)
        video_stream = video.get_stream_source()
        video_size = get_fs_object_size(video_stream)
        if video_size > 50 * 1024 * 1024:
            raise ValueError(f"Video size ({video_size / 1024 / 1024:.1f}MB) exceeds 50MB limit.")
        initial_response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/xai/v1/videos/edits", method="POST"),
            data=VideoEditRequest(
                model=model,
                video=InputUrlObject(url=await upload_video_to_comfyapi(cls, video)),
                prompt=prompt,
                seed=seed,
            ),
            response_model=VideoGenerationResponse,
        )
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/xai/v1/videos/{initial_response.request_id}"),
            status_extractor=lambda r: r.status if r.status is not None else "complete",
            response_model=VideoStatusResponse,
            price_extractor=_extract_grok_price,
        )
        return IO.NodeOutput(await download_url_to_video_output(response.video.url))


class GrokVideoReferenceNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GrokVideoReferenceNode",
            display_name="Grok Reference-to-Video",
            category="partner/video/Grok",
            description="Generate video guided by reference images as style and content references.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="Text description of the desired video.",
                ),
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "grok-imagine-video",
                            [
                                IO.Autogrow.Input(
                                    "reference_images",
                                    template=IO.Autogrow.TemplatePrefix(
                                        IO.Image.Input("image"),
                                        prefix="reference_",
                                        min=1,
                                        max=7,
                                    ),
                                    tooltip="Up to 7 reference images to guide the video generation.",
                                ),
                                IO.Combo.Input(
                                    "resolution",
                                    options=["480p", "720p"],
                                    tooltip="The resolution of the output video.",
                                ),
                                IO.Combo.Input(
                                    "aspect_ratio",
                                    options=["16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16"],
                                    tooltip="The aspect ratio of the output video.",
                                ),
                                IO.Int.Input(
                                    "duration",
                                    default=6,
                                    min=2,
                                    max=10,
                                    step=1,
                                    tooltip="The duration of the output video in seconds.",
                                    display_mode=IO.NumberDisplay.slider,
                                ),
                            ],
                        ),
                    ],
                    tooltip="The model to use for video generation.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to determine if node should re-run; "
                    "actual results are nondeterministic regardless of seed.",
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
                    widgets=["model.duration", "model.resolution"],
                    input_groups=["model.reference_images"],
                ),
                expr="""
                (
                  $res := $lookup(widgets, "model.resolution");
                  $dur := $lookup(widgets, "model.duration");
                  $refs := $lookup(inputGroups, "model.reference_images");
                  $rate := $res = "720p" ? 0.07 : 0.05;
                  $price := ($rate * $dur + 0.002 * $refs) * 1.43;
                  {"type":"usd","usd": $price}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        model: dict,
        seed: int,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        ref_image_urls = await upload_images_to_comfyapi(
            cls,
            list(model["reference_images"].values()),
            mime_type="image/png",
            wait_label="Uploading base images",
            max_images=7,
        )
        initial_response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/xai/v1/videos/generations", method="POST"),
            data=VideoGenerationRequest(
                model=model["model"],
                reference_images=[InputUrlObject(url=i) for i in ref_image_urls],
                prompt=prompt,
                resolution=model["resolution"],
                duration=model["duration"],
                aspect_ratio=model["aspect_ratio"],
                seed=seed,
            ),
            response_model=VideoGenerationResponse,
        )
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/xai/v1/videos/{initial_response.request_id}"),
            status_extractor=lambda r: r.status if r.status is not None else "complete",
            response_model=VideoStatusResponse,
            price_extractor=_extract_grok_video_price,
        )
        return IO.NodeOutput(await download_url_to_video_output(response.video.url))


class GrokVideoExtendNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GrokVideoExtendNode",
            display_name="Grok Video Extend",
            category="partner/video/Grok",
            description="Extend an existing video with a seamless continuation based on a text prompt.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="Text description of what should happen next in the video.",
                ),
                IO.Video.Input("video", tooltip="Source video to extend. MP4 format, 2-15 seconds."),
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "grok-imagine-video",
                            [
                                IO.Int.Input(
                                    "duration",
                                    default=8,
                                    min=2,
                                    max=10,
                                    step=1,
                                    tooltip="Length of the extension in seconds.",
                                    display_mode=IO.NumberDisplay.slider,
                                ),
                            ],
                        ),
                    ],
                    tooltip="The model to use for video extension.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to determine if node should re-run; "
                    "actual results are nondeterministic regardless of seed.",
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
                depends_on=IO.PriceBadgeDepends(widgets=["model.duration"]),
                expr="""
                (
                  $dur := $lookup(widgets, "model.duration");
                  {
                    "type": "range_usd",
                    "min_usd": (0.02 + 0.05 * $dur) * 1.43,
                    "max_usd": (0.15 + 0.05 * $dur) * 1.43
                  }
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        video: Input.Video,
        model: dict,
        seed: int,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        validate_video_duration(video, min_duration=2, max_duration=15)
        video_size = get_fs_object_size(video.get_stream_source())
        if video_size > 50 * 1024 * 1024:
            raise ValueError(f"Video size ({video_size / 1024 / 1024:.1f}MB) exceeds 50MB limit.")
        initial_response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/xai/v1/videos/extensions", method="POST"),
            data=VideoExtensionRequest(
                prompt=prompt,
                video=InputUrlObject(url=await upload_video_to_comfyapi(cls, video)),
                duration=model["duration"],
            ),
            response_model=VideoGenerationResponse,
        )
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/xai/v1/videos/{initial_response.request_id}"),
            status_extractor=lambda r: r.status if r.status is not None else "complete",
            response_model=VideoStatusResponse,
            price_extractor=_extract_grok_video_price,
        )
        return IO.NodeOutput(await download_url_to_video_output(response.video.url))


class GrokExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            GrokImageNode,
            GrokImageEditNode,
            GrokImageEditNodeV2,
            GrokVideoNode,
            GrokVideoReferenceNode,
            GrokVideoEditNode,
            GrokVideoExtendNode,
        ]


async def comfy_entrypoint() -> GrokExtension:
    return GrokExtension()
