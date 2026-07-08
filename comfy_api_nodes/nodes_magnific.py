import math

from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.magnific import (
    ImageRelightAdvancedSettingsRequest,
    ImageRelightRequest,
    ImageSkinEnhancerCreativeRequest,
    ImageSkinEnhancerFaithfulRequest,
    ImageSkinEnhancerFlexibleRequest,
    ImageStyleTransferRequest,
    ImageUpscalerCreativeRequest,
    ImageUpscalerPrecisionV2Request,
    InputAdvancedSettings,
    InputPortraitMode,
    InputSkinEnhancerMode,
    TaskResponse,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_image_tensor,
    downscale_image_tensor,
    get_image_dimensions,
    get_number_of_images,
    poll_op,
    sync_op,
    upload_images_to_comfyapi,
    validate_image_aspect_ratio,
    validate_image_dimensions,
)

_EUR_TO_USD = 1.19


def _tier_price_eur(megapixels: float) -> float:
    """Price in EUR for a single Magnific upscaling step based on input megapixels."""
    if megapixels <= 1.3:
        return 0.143
    if megapixels <= 3.0:
        return 0.286
    if megapixels <= 6.4:
        return 0.429
    return 1.716


def _calculate_magnific_upscale_price_usd(width: int, height: int, scale: int) -> float:
    """Calculate total Magnific upscale price in USD for given input dimensions and scale factor."""
    num_steps = int(math.log2(scale))
    total_eur = 0.0
    pixels = width * height
    for _ in range(num_steps):
        total_eur += _tier_price_eur(pixels / 1_000_000)
        pixels *= 4
    return round(total_eur * _EUR_TO_USD, 2)


class MagnificImageUpscalerCreativeNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MagnificImageUpscalerCreativeNode",
            display_name="Magnific Image Upscale (Creative)",
            category="partner/image/Magnific",
            description="Prompt‑guided enhancement, stylization, and 2x/4x/8x/16x upscaling. "
            "Maximum output: 25.3 megapixels.",
            inputs=[
                IO.Image.Input("image"),
                IO.String.Input("prompt", multiline=True, default=""),
                IO.Combo.Input("scale_factor", options=["2x", "4x", "8x", "16x"]),
                IO.Combo.Input(
                    "optimized_for",
                    options=[
                        "standard",
                        "soft_portraits",
                        "hard_portraits",
                        "art_n_illustration",
                        "videogame_assets",
                        "nature_n_landscapes",
                        "films_n_photography",
                        "3d_renders",
                        "science_fiction_n_horror",
                    ],
                ),
                IO.Int.Input("creativity", min=-10, max=10, default=0, display_mode=IO.NumberDisplay.slider),
                IO.Int.Input(
                    "hdr",
                    min=-10,
                    max=10,
                    default=0,
                    tooltip="The level of definition and detail.",
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Int.Input(
                    "resemblance",
                    min=-10,
                    max=10,
                    default=0,
                    tooltip="The level of resemblance to the original image.",
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Int.Input(
                    "fractality",
                    min=-10,
                    max=10,
                    default=0,
                    tooltip="The strength of the prompt and intricacy per square pixel.",
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Combo.Input(
                    "engine",
                    options=["automatic", "magnific_illusio", "magnific_sharpy", "magnific_sparkle"],
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "auto_downscale",
                    default=False,
                    tooltip="Automatically downscale input image if output would exceed maximum pixel limit.",
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
                depends_on=IO.PriceBadgeDepends(widgets=["scale_factor", "auto_downscale"]),
                expr="""
                (
                  $ad := widgets.auto_downscale;
                  $mins := $ad
                    ? {"2x": 0.172, "4x": 0.343, "8x": 0.515, "16x": 0.515}
                    : {"2x": 0.172, "4x": 0.343, "8x": 0.515, "16x": 0.844};
                  $maxs := {"2x": 0.515, "4x": 0.844, "8x": 1.015, "16x": 1.187};
                  {
                    "type": "range_usd",
                    "min_usd": $lookup($mins, widgets.scale_factor),
                    "max_usd": $lookup($maxs, widgets.scale_factor),
                    "format": { "approximate": true }
                  }
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        prompt: str,
        scale_factor: str,
        optimized_for: str,
        creativity: int,
        hdr: int,
        resemblance: int,
        fractality: int,
        engine: str,
        auto_downscale: bool,
    ) -> IO.NodeOutput:
        if get_number_of_images(image) != 1:
            raise ValueError("Exactly one input image is required.")
        validate_image_aspect_ratio(image, (1, 3), (3, 1), strict=False)
        validate_image_dimensions(image, min_height=160, min_width=160)

        max_output_pixels = 25_300_000
        height, width = get_image_dimensions(image)
        requested_scale = int(scale_factor.rstrip("x"))
        output_pixels = height * width * requested_scale * requested_scale

        if output_pixels > max_output_pixels:
            if auto_downscale:
                # Find optimal scale factor that doesn't require >2x downscale.
                # Server upscales in 2x steps, so aggressive downscaling degrades quality.
                input_pixels = width * height
                scale = 2
                max_input_pixels = max_output_pixels // 4
                for candidate in [16, 8, 4, 2]:
                    if candidate > requested_scale:
                        continue
                    scale_output_pixels = input_pixels * candidate * candidate
                    if scale_output_pixels <= max_output_pixels:
                        scale = candidate
                        max_input_pixels = None
                        break
                    downscale_ratio = math.sqrt(scale_output_pixels / max_output_pixels)
                    if downscale_ratio <= 2.0:
                        scale = candidate
                        max_input_pixels = max_output_pixels // (candidate * candidate)
                        break

                if max_input_pixels is not None:
                    image = downscale_image_tensor(image, total_pixels=max_input_pixels)
                scale_factor = f"{scale}x"
            else:
                raise ValueError(
                    f"Output size ({width * requested_scale}x{height * requested_scale} = {output_pixels:,} pixels) "
                    f"exceeds maximum allowed size of {max_output_pixels:,} pixels. "
                    f"Use a smaller input image or lower scale factor."
                )

        final_height, final_width = get_image_dimensions(image)
        actual_scale = int(scale_factor.rstrip("x"))
        price_usd = _calculate_magnific_upscale_price_usd(final_width, final_height, actual_scale)

        initial_res = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/freepik/v1/ai/image-upscaler", method="POST"),
            response_model=TaskResponse,
            data=ImageUpscalerCreativeRequest(
                image=(await upload_images_to_comfyapi(cls, image, max_images=1, total_pixels=None))[0],
                scale_factor=scale_factor,
                optimized_for=optimized_for,
                creativity=creativity,
                hdr=hdr,
                resemblance=resemblance,
                fractality=fractality,
                engine=engine,
                prompt=prompt if prompt else None,
            ),
        )
        final_response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/freepik/v1/ai/image-upscaler/{initial_res.task_id}"),
            response_model=TaskResponse,
            status_extractor=lambda x: x.status,
            price_extractor=lambda _: price_usd,
            poll_interval=10.0,
        )
        return IO.NodeOutput(await download_url_to_image_tensor(final_response.generated[0]))


class MagnificImageUpscalerPreciseV2Node(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MagnificImageUpscalerPreciseV2Node",
            display_name="Magnific Image Upscale (Precise V2)",
            category="partner/image/Magnific",
            description="High-fidelity upscaling with fine control over sharpness, grain, and detail. "
            "Maximum output: 10060×10060 pixels.",
            inputs=[
                IO.Image.Input("image"),
                IO.Combo.Input("scale_factor", options=["2x", "4x", "8x", "16x"]),
                IO.Combo.Input(
                    "flavor",
                    options=["sublime", "photo", "photo_denoiser"],
                    tooltip="Processing style: "
                    "sublime for general use, photo for photographs, photo_denoiser for noisy photos.",
                ),
                IO.Int.Input(
                    "sharpen",
                    min=0,
                    max=100,
                    default=7,
                    tooltip="Image sharpness intensity. Higher values increase edge definition and clarity.",
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Int.Input(
                    "smart_grain",
                    min=0,
                    max=100,
                    default=7,
                    tooltip="Intelligent grain/texture enhancement to prevent the image from "
                    "looking too smooth or artificial.",
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Int.Input(
                    "ultra_detail",
                    min=0,
                    max=100,
                    default=30,
                    tooltip="Controls fine detail, textures, and micro-details added during upscaling.",
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Boolean.Input(
                    "auto_downscale",
                    default=False,
                    tooltip="Automatically downscale input image if output would exceed maximum resolution.",
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
                depends_on=IO.PriceBadgeDepends(widgets=["scale_factor"]),
                expr="""
                (
                  $mins := {"2x": 0.172, "4x": 0.343, "8x": 0.515, "16x": 0.844};
                  $maxs := {"2x": 2.045, "4x": 2.545, "8x": 2.889, "16x": 3.06};
                  {
                    "type": "range_usd",
                    "min_usd": $lookup($mins, widgets.scale_factor),
                    "max_usd": $lookup($maxs, widgets.scale_factor),
                    "format": { "approximate": true }
                  }
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        scale_factor: str,
        flavor: str,
        sharpen: int,
        smart_grain: int,
        ultra_detail: int,
        auto_downscale: bool,
    ) -> IO.NodeOutput:
        if get_number_of_images(image) != 1:
            raise ValueError("Exactly one input image is required.")
        validate_image_aspect_ratio(image, (1, 3), (3, 1), strict=False)
        validate_image_dimensions(image, min_height=160, min_width=160)

        max_output_dimension = 10060
        height, width = get_image_dimensions(image)
        requested_scale = int(scale_factor.strip("x"))
        output_width = width * requested_scale
        output_height = height * requested_scale

        if output_width > max_output_dimension or output_height > max_output_dimension:
            if auto_downscale:
                # Find optimal scale factor that doesn't require >2x downscale.
                # Server upscales in 2x steps, so aggressive downscaling degrades quality.
                max_dim = max(width, height)
                scale = 2
                max_input_dim = max_output_dimension // 2
                scale_ratio = max_input_dim / max_dim
                max_input_pixels = int(width * height * scale_ratio * scale_ratio)
                for candidate in [16, 8, 4, 2]:
                    if candidate > requested_scale:
                        continue
                    output_dim = max_dim * candidate
                    if output_dim <= max_output_dimension:
                        scale = candidate
                        max_input_pixels = None
                        break
                    downscale_ratio = output_dim / max_output_dimension
                    if downscale_ratio <= 2.0:
                        scale = candidate
                        max_input_dim = max_output_dimension // candidate
                        scale_ratio = max_input_dim / max_dim
                        max_input_pixels = int(width * height * scale_ratio * scale_ratio)
                        break

                if max_input_pixels is not None:
                    image = downscale_image_tensor(image, total_pixels=max_input_pixels)
                requested_scale = scale
            else:
                raise ValueError(
                    f"Output dimensions ({output_width}x{output_height}) exceed maximum allowed "
                    f"resolution of {max_output_dimension}x{max_output_dimension} pixels. "
                    f"Use a smaller input image or lower scale factor."
                )

        final_height, final_width = get_image_dimensions(image)
        price_usd = _calculate_magnific_upscale_price_usd(final_width, final_height, requested_scale)

        initial_res = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/freepik/v1/ai/image-upscaler-precision-v2", method="POST"),
            response_model=TaskResponse,
            data=ImageUpscalerPrecisionV2Request(
                image=(await upload_images_to_comfyapi(cls, image, max_images=1, total_pixels=None))[0],
                scale_factor=requested_scale,
                flavor=flavor,
                sharpen=sharpen,
                smart_grain=smart_grain,
                ultra_detail=ultra_detail,
            ),
        )
        final_response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/freepik/v1/ai/image-upscaler-precision-v2/{initial_res.task_id}"),
            response_model=TaskResponse,
            status_extractor=lambda x: x.status,
            price_extractor=lambda _: price_usd,
            poll_interval=10.0,
        )
        return IO.NodeOutput(await download_url_to_image_tensor(final_response.generated[0]))


class MagnificImageStyleTransferNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MagnificImageStyleTransferNode",
            display_name="Magnific Image Style Transfer",
            category="partner/image/Magnific",
            description="Transfer the style from a reference image to your input image.",
            inputs=[
                IO.Image.Input("image", tooltip="The image to apply style transfer to."),
                IO.Image.Input("reference_image", tooltip="The reference image to extract style from."),
                IO.String.Input("prompt", multiline=True, default=""),
                IO.Int.Input(
                    "style_strength",
                    min=0,
                    max=100,
                    default=100,
                    tooltip="Percentage of style strength.",
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Int.Input(
                    "structure_strength",
                    min=0,
                    max=100,
                    default=50,
                    tooltip="Maintains the structure of the original image.",
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Combo.Input(
                    "flavor",
                    options=["faithful", "gen_z", "psychedelia", "detaily", "clear", "donotstyle", "donotstyle_sharp"],
                    tooltip="Style transfer flavor.",
                ),
                IO.Combo.Input(
                    "engine",
                    options=[
                        "balanced",
                        "definio",
                        "illusio",
                        "3d_cartoon",
                        "colorful_anime",
                        "caricature",
                        "real",
                        "super_real",
                        "softy",
                    ],
                    tooltip="Processing engine selection.",
                    advanced=True,
                ),
                IO.DynamicCombo.Input(
                    "portrait_mode",
                    options=[
                        IO.DynamicCombo.Option("disabled", []),
                        IO.DynamicCombo.Option(
                            "enabled",
                            [
                                IO.Combo.Input(
                                    "portrait_style",
                                    options=["standard", "pop", "super_pop"],
                                    tooltip="Visual style applied to portrait images.",
                                ),
                                IO.Combo.Input(
                                    "portrait_beautifier",
                                    options=["none", "beautify_face", "beautify_face_max"],
                                    tooltip="Facial beautification intensity on portraits.",
                                ),
                            ],
                        ),
                    ],
                    tooltip="Enable portrait mode for facial enhancements.",
                ),
                IO.Boolean.Input(
                    "fixed_generation",
                    default=True,
                    tooltip="When disabled, expect each generation to introduce a degree of randomness, "
                    "leading to more diverse outcomes.",
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
                expr="""{"type":"usd","usd":0.11}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        reference_image: Input.Image,
        prompt: str,
        style_strength: int,
        structure_strength: int,
        flavor: str,
        engine: str,
        portrait_mode: InputPortraitMode,
        fixed_generation: bool,
    ) -> IO.NodeOutput:
        if get_number_of_images(image) != 1:
            raise ValueError("Exactly one input image is required.")
        if get_number_of_images(reference_image) != 1:
            raise ValueError("Exactly one reference image is required.")
        validate_image_aspect_ratio(image, (1, 3), (3, 1), strict=False)
        validate_image_aspect_ratio(reference_image, (1, 3), (3, 1), strict=False)
        validate_image_dimensions(image, min_height=160, min_width=160)
        validate_image_dimensions(reference_image, min_height=160, min_width=160)

        is_portrait = portrait_mode["portrait_mode"] == "enabled"
        portrait_style = portrait_mode.get("portrait_style", "standard")
        portrait_beautifier = portrait_mode.get("portrait_beautifier", "none")

        uploaded_urls = await upload_images_to_comfyapi(cls, [image, reference_image], max_images=2)

        initial_res = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/freepik/v1/ai/image-style-transfer", method="POST"),
            response_model=TaskResponse,
            data=ImageStyleTransferRequest(
                image=uploaded_urls[0],
                reference_image=uploaded_urls[1],
                prompt=prompt if prompt else None,
                style_strength=style_strength,
                structure_strength=structure_strength,
                is_portrait=is_portrait,
                portrait_style=portrait_style if is_portrait else None,
                portrait_beautifier=portrait_beautifier if is_portrait and portrait_beautifier != "none" else None,
                flavor=flavor,
                engine=engine,
                fixed_generation=fixed_generation,
            ),
        )
        final_response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/freepik/v1/ai/image-style-transfer/{initial_res.task_id}"),
            response_model=TaskResponse,
            status_extractor=lambda x: x.status,
            poll_interval=10.0,
        )
        return IO.NodeOutput(await download_url_to_image_tensor(final_response.generated[0]))


class MagnificImageRelightNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MagnificImageRelightNode",
            display_name="Magnific Image Relight",
            category="partner/image/Magnific",
            description="Relight an image with lighting adjustments and optional reference-based light transfer.",
            inputs=[
                IO.Image.Input("image", tooltip="The image to relight."),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Descriptive guidance for lighting. Supports emphasis notation (1-1.4).",
                ),
                IO.Int.Input(
                    "light_transfer_strength",
                    min=0,
                    max=100,
                    default=100,
                    tooltip="Intensity of light transfer application.",
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Combo.Input(
                    "style",
                    options=[
                        "standard",
                        "darker_but_realistic",
                        "clean",
                        "smooth",
                        "brighter",
                        "contrasted_n_hdr",
                        "just_composition",
                    ],
                    tooltip="Stylistic output preference.",
                ),
                IO.Boolean.Input(
                    "interpolate_from_original",
                    default=False,
                    tooltip="Restricts generation freedom to match original more closely.",
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "change_background",
                    default=True,
                    tooltip="Modifies background based on prompt/reference.",
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "preserve_details",
                    default=True,
                    tooltip="Maintains texture and fine details from original.",
                    advanced=True,
                ),
                IO.DynamicCombo.Input(
                    "advanced_settings",
                    options=[
                        IO.DynamicCombo.Option("disabled", []),
                        IO.DynamicCombo.Option(
                            "enabled",
                            [
                                IO.Int.Input(
                                    "whites",
                                    min=0,
                                    max=100,
                                    default=50,
                                    tooltip="Adjusts the brightest tones in the image.",
                                    display_mode=IO.NumberDisplay.slider,
                                ),
                                IO.Int.Input(
                                    "blacks",
                                    min=0,
                                    max=100,
                                    default=50,
                                    tooltip="Adjusts the darkest tones in the image.",
                                    display_mode=IO.NumberDisplay.slider,
                                ),
                                IO.Int.Input(
                                    "brightness",
                                    min=0,
                                    max=100,
                                    default=50,
                                    tooltip="Overall brightness adjustment.",
                                    display_mode=IO.NumberDisplay.slider,
                                ),
                                IO.Int.Input(
                                    "contrast",
                                    min=0,
                                    max=100,
                                    default=50,
                                    tooltip="Contrast adjustment.",
                                    display_mode=IO.NumberDisplay.slider,
                                ),
                                IO.Int.Input(
                                    "saturation",
                                    min=0,
                                    max=100,
                                    default=50,
                                    tooltip="Color saturation adjustment.",
                                    display_mode=IO.NumberDisplay.slider,
                                ),
                                IO.Combo.Input(
                                    "engine",
                                    options=[
                                        "automatic",
                                        "balanced",
                                        "cool",
                                        "real",
                                        "illusio",
                                        "fairy",
                                        "colorful_anime",
                                        "hard_transform",
                                        "softy",
                                    ],
                                    tooltip="Processing engine selection.",
                                ),
                                IO.Combo.Input(
                                    "transfer_light_a",
                                    options=["automatic", "low", "medium", "normal", "high", "high_on_faces"],
                                    tooltip="The intensity of light transfer.",
                                ),
                                IO.Combo.Input(
                                    "transfer_light_b",
                                    options=[
                                        "automatic",
                                        "composition",
                                        "straight",
                                        "smooth_in",
                                        "smooth_out",
                                        "smooth_both",
                                        "reverse_both",
                                        "soft_in",
                                        "soft_out",
                                        "soft_mid",
                                        # "strong_mid",  # Commented out because requests fail when this is set.
                                        "style_shift",
                                        "strong_shift",
                                    ],
                                    tooltip="Also modifies light transfer intensity. "
                                    "Can be combined with the previous control for varied effects.",
                                ),
                                IO.Boolean.Input(
                                    "fixed_generation",
                                    default=True,
                                    tooltip="Ensures consistent output with the same settings.",
                                ),
                            ],
                        ),
                    ],
                    tooltip="Fine-tuning options for advanced lighting control.",
                ),
                IO.Image.Input(
                    "reference_image",
                    optional=True,
                    tooltip="Optional reference image to transfer lighting from.",
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
                expr="""{"type":"usd","usd":0.11}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        prompt: str,
        light_transfer_strength: int,
        style: str,
        interpolate_from_original: bool,
        change_background: bool,
        preserve_details: bool,
        advanced_settings: InputAdvancedSettings,
        reference_image: Input.Image | None = None,
    ) -> IO.NodeOutput:
        if get_number_of_images(image) != 1:
            raise ValueError("Exactly one input image is required.")
        if reference_image is not None and get_number_of_images(reference_image) != 1:
            raise ValueError("Exactly one reference image is required.")
        validate_image_aspect_ratio(image, (1, 3), (3, 1), strict=False)
        validate_image_dimensions(image, min_height=160, min_width=160)
        if reference_image is not None:
            validate_image_aspect_ratio(reference_image, (1, 3), (3, 1), strict=False)
            validate_image_dimensions(reference_image, min_height=160, min_width=160)

        image_url = (await upload_images_to_comfyapi(cls, image, max_images=1))[0]
        reference_url = None
        if reference_image is not None:
            reference_url = (await upload_images_to_comfyapi(cls, reference_image, max_images=1))[0]

        adv_settings = None
        if advanced_settings["advanced_settings"] == "enabled":
            adv_settings = ImageRelightAdvancedSettingsRequest(
                whites=advanced_settings["whites"],
                blacks=advanced_settings["blacks"],
                brightness=advanced_settings["brightness"],
                contrast=advanced_settings["contrast"],
                saturation=advanced_settings["saturation"],
                engine=advanced_settings["engine"],
                transfer_light_a=advanced_settings["transfer_light_a"],
                transfer_light_b=advanced_settings["transfer_light_b"],
                fixed_generation=advanced_settings["fixed_generation"],
            )

        initial_res = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/freepik/v1/ai/image-relight", method="POST"),
            response_model=TaskResponse,
            data=ImageRelightRequest(
                image=image_url,
                prompt=prompt if prompt else None,
                transfer_light_from_reference_image=reference_url,
                light_transfer_strength=light_transfer_strength,
                interpolate_from_original=interpolate_from_original,
                change_background=change_background,
                style=style,
                preserve_details=preserve_details,
                advanced_settings=adv_settings,
            ),
        )
        final_response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/freepik/v1/ai/image-relight/{initial_res.task_id}"),
            response_model=TaskResponse,
            status_extractor=lambda x: x.status,
            poll_interval=10.0,
        )
        return IO.NodeOutput(await download_url_to_image_tensor(final_response.generated[0]))


class MagnificImageSkinEnhancerNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MagnificImageSkinEnhancerNode",
            display_name="Magnific Image Skin Enhancer",
            category="partner/image/Magnific",
            description="Skin enhancement for portraits with multiple processing modes.",
            inputs=[
                IO.Image.Input("image", tooltip="The portrait image to enhance."),
                IO.Int.Input(
                    "sharpen",
                    min=0,
                    max=100,
                    default=0,
                    tooltip="Sharpening intensity level.",
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.Int.Input(
                    "smart_grain",
                    min=0,
                    max=100,
                    default=2,
                    tooltip="Smart grain intensity level.",
                    display_mode=IO.NumberDisplay.slider,
                ),
                IO.DynamicCombo.Input(
                    "mode",
                    options=[
                        IO.DynamicCombo.Option("creative", []),
                        IO.DynamicCombo.Option(
                            "faithful",
                            [
                                IO.Int.Input(
                                    "skin_detail",
                                    min=0,
                                    max=100,
                                    default=80,
                                    tooltip="Skin detail enhancement level.",
                                    display_mode=IO.NumberDisplay.slider,
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "flexible",
                            [
                                IO.Combo.Input(
                                    "optimized_for",
                                    options=[
                                        "enhance_skin",
                                        "improve_lighting",
                                        "enhance_everything",
                                        "transform_to_real",
                                        "no_make_up",
                                    ],
                                    tooltip="Enhancement optimization target.",
                                ),
                            ],
                        ),
                    ],
                    tooltip="Processing mode: creative for artistic enhancement, "
                    "faithful for preserving original appearance, "
                    "flexible for targeted optimization.",
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
                depends_on=IO.PriceBadgeDepends(widgets=["mode"]),
                expr="""
                (
                  $rates := {"creative": 0.29, "faithful": 0.37, "flexible": 0.45};
                  {"type":"usd","usd": $lookup($rates, widgets.mode)}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        sharpen: int,
        smart_grain: int,
        mode: InputSkinEnhancerMode,
    ) -> IO.NodeOutput:
        if get_number_of_images(image) != 1:
            raise ValueError("Exactly one input image is required.")
        validate_image_aspect_ratio(image, (1, 3), (3, 1), strict=False)
        validate_image_dimensions(image, min_height=160, min_width=160)

        image_url = (await upload_images_to_comfyapi(cls, image, max_images=1, total_pixels=4096 * 4096))[0]
        selected_mode = mode["mode"]

        if selected_mode == "creative":
            endpoint = "creative"
            data = ImageSkinEnhancerCreativeRequest(
                image=image_url,
                sharpen=sharpen,
                smart_grain=smart_grain,
            )
        elif selected_mode == "faithful":
            endpoint = "faithful"
            data = ImageSkinEnhancerFaithfulRequest(
                image=image_url,
                sharpen=sharpen,
                smart_grain=smart_grain,
                skin_detail=mode["skin_detail"],
            )
        else:  # flexible
            endpoint = "flexible"
            data = ImageSkinEnhancerFlexibleRequest(
                image=image_url,
                sharpen=sharpen,
                smart_grain=smart_grain,
                optimized_for=mode["optimized_for"],
            )

        initial_res = await sync_op(
            cls,
            ApiEndpoint(path=f"/proxy/freepik/v1/ai/skin-enhancer/{endpoint}", method="POST"),
            response_model=TaskResponse,
            data=data,
        )
        final_response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/freepik/v1/ai/skin-enhancer/{initial_res.task_id}"),
            response_model=TaskResponse,
            status_extractor=lambda x: x.status,
            poll_interval=10.0,
        )
        return IO.NodeOutput(await download_url_to_image_tensor(final_response.generated[0]))


class MagnificExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            MagnificImageUpscalerCreativeNode,
            MagnificImageUpscalerPreciseV2Node,
            MagnificImageStyleTransferNode,
            MagnificImageRelightNode,
            MagnificImageSkinEnhancerNode,
        ]


async def comfy_entrypoint() -> MagnificExtension:
    return MagnificExtension()
