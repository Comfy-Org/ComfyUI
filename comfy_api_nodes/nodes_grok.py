import re

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
    VoiceReferenceObject,
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

_GROK_IMAGE_MODEL_API_IDS = {
    "grok-imagine-image-2.0": "grok-imagine-image-2.0",
}

_GROK_IMAGE_QUALITY_MODELS = {"grok-imagine-image-2.0"}

_GROK_IMAGE_QUALITY_OPTIONS = ["medium", "low"]

_GROK_IMAGE_EDIT_MAX_IMAGES = {
    "grok-imagine-image-2.0": 3,
    "grok-imagine-image-pro": 1,
    "grok-imagine-image-quality": 3,
    "grok-imagine-image": 3,
}

_GROK_IMAGE_EDIT_ASPECT_RATIO_NEEDS_MULTIPLE = {
    "grok-imagine-image-quality",
    "grok-imagine-image",
}

_GROK_VOICE_OPTIONS = [
    "none",
    "ara",
    "eve",
    "leo",
    "rex",
    "sal",
    "carina",
    "zagan",
    "helix",
    "orion",
    "luna",
    "iris",
    "altair",
    "zenith",
    "perseus",
    "helios",
    "lux",
    "kepler",
    "rigel",
    "cosmo",
    "celeste",
    "ursa",
    "sirius",
    "lumen",
    "castor",
    "naksh",
    "atlas",
]


_GROK_REF_TAG_RE = re.compile(r"(?<!\w)@(image|audio)(?P<idx>\d*)(?!\w)", re.IGNORECASE | re.ASCII)


def _normalize_grok_reference_prompt(prompt: str, total_images: int, voices: list[str]) -> str:
    """Rewrite @Image1/@Audio1 style references (1-based, shared partner-node syntax)
    into Grok's native <IMAGE_0>/<AUDIO_0> tags; an unnumbered @image/@audio means the first one.
    Native tags pass through untouched. @ImageN refers to the Nth reference image overall, in
    input order — a batched input contributes one number per image. @AudioN refers to the
    'voice_N' widget; the API only accepts compact arrays, so voices are remapped to array
    positions and 'none' slots between selected voices are harmless. Substitution repeats until
    stable so adjacent tags like '@Image1@Image2' all resolve."""
    audio_indices: dict[int, int] = {}
    for slot, voice in enumerate(voices, start=1):
        if voice != "none":
            audio_indices[slot] = len(audio_indices)

    def repl(match: re.Match) -> str:
        kind = match.group(1).lower()
        idx = int(match.group("idx") or 1)
        if kind == "image":
            if not 1 <= idx <= total_images:
                raise ValueError(
                    f"The prompt references @Image{idx}, but only {total_images} "
                    f"reference images are connected (a batched input counts once per image)."
                )
            return f"<IMAGE_{idx - 1}>"
        if idx not in audio_indices:
            if 1 <= idx <= len(voices):
                raise ValueError(f"The prompt references @Audio{idx}, but 'voice_{idx}' is set to 'none'.")
            raise ValueError(f"The prompt references @Audio{idx}, but only voices 1..{len(voices)} exist.")
        return f"<AUDIO_{audio_indices[idx]}>"

    prev = None
    while prev != prompt:
        prev = prompt
        prompt = _GROK_REF_TAG_RE.sub(repl, prompt)
    return prompt


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
                        "grok-imagine-image-2.0",
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
                IO.Combo.Input(
                    "quality",
                    options=_GROK_IMAGE_QUALITY_OPTIONS,
                    optional=True,
                    tooltip="Quality level, supported only by the grok-imagine-image-2.0 model.",
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
                depends_on=IO.PriceBadgeDepends(widgets=["model", "number_of_images", "resolution", "quality"]),
                expr="""
                (
                  $is1k := widgets.resolution = "1k";
                  $rate := widgets.model = "grok-imagine-image-2.0"
                    ? (widgets.quality = "low" ? ($is1k ? 0.04 : 0.06) : ($is1k ? 0.06 : 0.08))
                    : (widgets.model = "grok-imagine-image-quality"
                        ? ($is1k ? 0.05 : 0.07)
                        : ($contains(widgets.model, "pro") ? 0.07 : 0.02));
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
        quality: str = "medium",
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/xai/v1/images/generations", method="POST"),
            data=ImageGenerationRequest(
                model=_GROK_IMAGE_MODEL_API_IDS.get(model, model),
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                n=number_of_images,
                seed=seed,
                resolution=resolution.lower(),
                quality=quality if model in _GROK_IMAGE_QUALITY_MODELS else None,
            ),
            response_model=ImageGenerationResponse,
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


def _grok_image_edit_model_inputs(
    *, max_ref_images: int, with_aspect_ratio: bool, with_quality: bool = False, aspect_ratio_needs_multiple: bool = True
):
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
    if with_quality:
        inputs.append(IO.Combo.Input("quality", options=_GROK_IMAGE_QUALITY_OPTIONS))
    if with_aspect_ratio:
        inputs.append(
            IO.Combo.Input(
                "aspect_ratio",
                options=_GROK_IMAGE_EDIT_ASPECT_RATIO_OPTIONS,
                tooltip=(
                    "Only allowed when multiple images are connected."
                    if aspect_ratio_needs_multiple
                    else "Aspect ratio of the edited image."
                ),
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
                            "grok-imagine-image-2.0",
                            _grok_image_edit_model_inputs(
                                max_ref_images=3,
                                with_aspect_ratio=True,
                                with_quality=True,
                                aspect_ratio_needs_multiple=False,
                            ),
                        ),
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
                    widgets=["model", "model.resolution", "model.number_of_images", "model.quality"],
                ),
                expr="""
                (
                  $is20 := widgets.model = "grok-imagine-image-2.0";
                  $isPro := $contains(widgets.model, "pro");
                  $res := $lookup(widgets, "model.resolution");
                  $n := $lookup(widgets, "model.number_of_images");
                  $is1k := $res = "1k";
                  $rate := $is20
                    ? ($lookup(widgets, "model.quality") = "low"
                        ? ($is1k ? 0.04 : 0.06)
                        : ($is1k ? 0.06 : 0.08))
                    : (widgets.model = "grok-imagine-image-quality"
                        ? ($is1k ? 0.05 : 0.07)
                        : ($isPro ? 0.07 : 0.02));
                  $base := ($is20 or widgets.model = "grok-imagine-image-quality") ? 0.01 : 0.002;
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
        max_images = _GROK_IMAGE_EDIT_MAX_IMAGES.get(model_id, 3)
        if n_images < 1:
            raise ValueError("At least one image is required for editing.")
        if n_images > max_images:
            raise ValueError(
                f"The {model_id} model supports at most {max_images} input "
                f"image{'s' if max_images > 1 else ''}; {n_images} are connected."
            )
        if aspect_ratio != "auto" and model_id in _GROK_IMAGE_EDIT_ASPECT_RATIO_NEEDS_MULTIPLE and n_images == 1:
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
                model=_GROK_IMAGE_MODEL_API_IDS.get(model_id, model_id),
                images=[
                    InputUrlObject(url=f"data:image/png;base64,{tensor_to_base64_string(i)}") for i in flat_tensors
                ],
                prompt=prompt,
                resolution=resolution.lower(),
                n=number_of_images,
                seed=seed,
                aspect_ratio=None if aspect_ratio == "auto" else aspect_ratio,
                quality=model.get("quality") if model_id in _GROK_IMAGE_QUALITY_MODELS else None,
            ),
            response_model=ImageGenerationResponse,
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
                    tooltip="The model to use for video generation.",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="Text description of the desired video. "
                    "Optional for grok-imagine-video-1.5 when an input image is provided.",
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
                    tooltip="Optional starting image. If omitted, the video is generated from the text prompt alone.",
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
        if resolution == "1080p" and model != "grok-imagine-video-1.5":
            raise ValueError(f"1080p resolution is only available for grok-imagine-video-1.5, not '{model}'.")
        image_url = None
        if image is not None:
            if get_number_of_images(image) != 1:
                raise ValueError("Only one input image is supported.")
            image_url = InputUrlObject(url=f"data:image/png;base64,{tensor_to_base64_string(image)}")
        if image is None or model != "grok-imagine-video-1.5":
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
        )
        return IO.NodeOutput(await download_url_to_video_output(response.video.url))


class GrokVideoReferenceNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GrokVideoReferenceNode",
            display_name="Grok Reference-to-Video",
            category="partner/video/Grok",
            description="Generate video guided by reference images, with optional preset voice references.",
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
                            "grok-imagine-video-1.5",
                            [
                                IO.Autogrow.Input(
                                    "reference_images",
                                    template=IO.Autogrow.TemplateNames(
                                        IO.Image.Input("image"),
                                        names=[f"reference_{i}" for i in range(1, 8)],
                                        min=1,
                                    ),
                                    tooltip="Up to 7 reference images to guide the video generation. "
                                    "Refer to them in the prompt as @Image1 ... @Image7, numbered "
                                    "in input order; a batched input counts once per image.",
                                ),
                                IO.Combo.Input(
                                    "voice_1",
                                    options=_GROK_VOICE_OPTIONS,
                                    tooltip="Optional preset voice reference; refer to it in the prompt as @Audio1. "
                                    "The API supports only these preset voices, not custom audio.",
                                ),
                                IO.Combo.Input(
                                    "voice_2",
                                    options=_GROK_VOICE_OPTIONS,
                                    tooltip="Optional second voice reference; @Audio2 in the prompt.",
                                ),
                                IO.Combo.Input(
                                    "voice_3",
                                    options=_GROK_VOICE_OPTIONS,
                                    tooltip="Optional third voice reference; @Audio3 in the prompt.",
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
                                    min=1,
                                    max=15,
                                    step=1,
                                    tooltip="The duration of the output video in seconds.",
                                    display_mode=IO.NumberDisplay.slider,
                                ),
                            ],
                        ),
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
                    widgets=["model", "model.duration", "model.resolution"],
                    input_groups=["model.reference_images"],
                ),
                expr="""
                (
                  $is15 := $contains(widgets.model, "1.5");
                  $res := $lookup(widgets, "model.resolution");
                  $dur := $lookup(widgets, "model.duration");
                  $refs := $lookup(inputGroups, "model.reference_images");
                  $rate := $is15
                    ? ($res = "720p" ? 0.14 : 0.08)
                    : ($res = "720p" ? 0.07 : 0.05);
                  $imgCost := $is15 ? 0.01 : 0.002;
                  $price := ($rate * $dur + $imgCost * $refs) * 1.43;
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
        total_images = sum(get_number_of_images(t) for t in model["reference_images"].values())
        if total_images > 7:
            raise ValueError(f"A maximum of 7 reference images is supported; {total_images} are connected.")
        reference_audios = None
        if model["model"] == "grok-imagine-video-1.5":
            voices = [model.get(f"voice_{i}", "none") for i in range(1, 4)]
            reference_audios = [VoiceReferenceObject(voice_id=v) for v in voices if v != "none"] or None
            prompt = _normalize_grok_reference_prompt(prompt, total_images=total_images, voices=voices)
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
                model=_GROK_VIDEO_MODEL_API_IDS.get(model["model"], model["model"]),
                reference_images=[InputUrlObject(url=i) for i in ref_image_urls],
                reference_audios=reference_audios,
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
