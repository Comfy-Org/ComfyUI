from io import BytesIO

from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension
from comfy_api_nodes.apis.quiver import (
    QuiverImageObject,
    QuiverImageToSVGRequest,
    QuiverSVGAttributes,
    QuiverSVGResponse,
    QuiverTextToSVGRequest,
    QuiverViewBox,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    sync_op,
    upload_image_to_comfyapi,
    validate_string,
)
from comfy_extras.nodes_images import SVG

_ARROW_MODELS = ["arrow-2", "arrow-2-telos", "arrow-1.1", "arrow-1.1-max", "arrow-preview"]
_EFFORT_LEVELS = ["low", "medium", "high", "xhigh"]
_TOKEN_MODEL_RATES = {"arrow-2": (4, 20), "arrow-2-telos": (6, 30)}
_NO_SAMPLING_MODELS = ("arrow-2-telos",)
_REFERENCE_CAPS = {"arrow-2": 14, "arrow-2-telos": 14, "arrow-1.1": 4, "arrow-1.1-max": 14, "arrow-preview": 4}
_FIXED_GENERATION_USD = {"arrow-1.1": 0.286, "arrow-1.1-max": 0.3575, "arrow-preview": 0.429}
_FIXED_VECTORIZATION_USD = {"arrow-1.1": 0.2145, "arrow-1.1-max": 0.286, "arrow-preview": 0.429}

_GENERATION_TOKENS = {
    "arrow-2": {
        "low": (713, 1100, 723, 20000),
        "medium": (713, 2200, 723, 34000),
        "high": (713, 2200, 723, 36000),
        "xhigh": (713, 2200, 723, 38000),
    },
    "arrow-2-telos": {
        "low": (476, 1100, 749, 36000),
        "medium": (476, 1300, 749, 67000),
        "high": (476, 1300, 749, 71000),
        "xhigh": (476, 10000, 749, 106000),
    },
}
_VECTORIZATION_TOKENS = {
    "arrow-2": {
        "low": (591, 600, 783, 23000),
        "medium": (591, 1300, 783, 36000),
        "high": (591, 1300, 783, 38000),
        "xhigh": (591, 1300, 783, 40000),
    },
    "arrow-2-telos": {
        "low": (394, 300, 522, 47000),
        "medium": (394, 300, 522, 48000),
        "high": (394, 300, 522, 101000),
        "xhigh": (394, 12000, 522, 115000),
    },
}


def _effort_bands(model, tokens, effort):
    rate_in, rate_out = _TOKEN_MODEL_RATES[model]

    def band(level):
        min_in, min_out, max_in, max_out = tokens[model][level]
        return (
            '{"type":"range_usd",'
            f'"min_usd":({min_in} * {rate_in} + {min_out} * {rate_out}) * 1.43 / 1000000,'
            f'"max_usd":({max_in} * {rate_in} + {max_out} * {rate_out}) * 1.43 / 1000000,'
            '"format":{"approximate":true}}'
        )

    levels = [f'{effort} = "{level}" ? {band(level)}' for level in ("low", "medium", "xhigh")]
    return " : ".join([*levels, band("high")])


def _arrow_price_badge(tokens, fixed_usd, effort="widgets.reasoning_effort", effort_widget="reasoning_effort"):
    branches = [f'widgets.model = "{model}" ? {{"type":"usd","usd":{usd}}}' for model, usd in fixed_usd.items()]
    branches.append(f'widgets.model = "arrow-2-telos" ? ({_effort_bands("arrow-2-telos", tokens, effort)})')
    branches.append(f'({_effort_bands("arrow-2", tokens, effort)})')
    return IO.PriceBadge(
        depends_on=IO.PriceBadgeDepends(widgets=["model", effort_widget]),
        expr="(" + " : ".join(branches) + ")",
    )


def _reasoning_effort_input():
    return IO.Combo.Input(
        "reasoning_effort",
        options=_EFFORT_LEVELS,
        default="high",
        optional=True,
        tooltip="How much reasoning the model spends before drawing. Higher levels improve "
        "detail and cost more tokens. Only used by the Arrow 2 models.",
    )


def _arrow_sampling_inputs():
    """Shared sampling inputs for all Arrow model variants."""
    return [
        IO.Float.Input(
            "temperature",
            default=1.0,
            min=0.0,
            max=2.0,
            step=0.1,
            display_mode=IO.NumberDisplay.slider,
            tooltip="Randomness control. Higher values increase randomness.",
            advanced=True,
        ),
        IO.Float.Input(
            "top_p",
            default=1.0,
            min=0.05,
            max=1.0,
            step=0.05,
            display_mode=IO.NumberDisplay.slider,
            tooltip="Nucleus sampling parameter.",
            advanced=True,
        ),
        IO.Float.Input(
            "presence_penalty",
            default=0.0,
            min=-2.0,
            max=2.0,
            step=0.1,
            display_mode=IO.NumberDisplay.slider,
            tooltip="Token presence penalty.",
            advanced=True,
        ),
    ]


def _v2_view_box_inputs():
    return [
        IO.Int.Input(
            "width",
            default=0,
            min=0,
            max=8192,
            tooltip="Width of the output SVG canvas (viewBox), in user units. Set both width and "
            "height to control the output size and aspect ratio; leave either at 0 to let the model "
            "choose, which usually gives a square canvas.",
            advanced=True,
        ),
        IO.Int.Input(
            "height",
            default=0,
            min=0,
            max=8192,
            tooltip="Height of the output SVG canvas (viewBox), in user units. Set both width and "
            "height to control the output size and aspect ratio; leave either at 0 to let the model "
            "choose, which usually gives a square canvas.",
            advanced=True,
        ),
    ]


def _v2_seed_input():
    return IO.Int.Input(
        "seed",
        default=42,
        min=0,
        max=2147483647,
        control_after_generate=True,
        tooltip="Seed to determine if node should re-run; "
        "actual results are nondeterministic regardless of seed.",
    )


def _v2_model_tail(model):
    inputs = []
    if model in _TOKEN_MODEL_RATES:
        inputs.append(
            IO.Combo.Input(
                "reasoning_effort",
                options=_EFFORT_LEVELS,
                default="high",
                tooltip="How much reasoning the model spends before drawing. Higher levels improve "
                "detail and cost more tokens.",
            )
        )
    if model not in _NO_SAMPLING_MODELS:
        inputs.extend(_arrow_sampling_inputs())
    inputs.extend(_v2_view_box_inputs())
    inputs.append(_v2_seed_input())
    return inputs


def _v2_text_option(model):
    cap = _REFERENCE_CAPS[model]
    return IO.DynamicCombo.Option(
        model,
        [
            IO.String.Input(
                "prompt",
                multiline=True,
                default="",
                tooltip="Text description of the desired SVG output.",
            ),
            IO.String.Input(
                "instructions",
                multiline=True,
                default="",
                tooltip="Additional style or formatting guidance.",
                optional=True,
                advanced=True,
            ),
            IO.Autogrow.Input(
                "reference_images",
                template=IO.Autogrow.TemplatePrefix(
                    IO.Image.Input("image"),
                    prefix="ref_",
                    min=0,
                    max=cap,
                ),
                tooltip=f"Up to {cap} reference images to guide the generation.",
                optional=True,
            ),
            *_v2_model_tail(model),
        ],
    )


def _v2_image_option(model):
    return IO.DynamicCombo.Option(
        model,
        [
            IO.Image.Input("image", tooltip="Input image to vectorize."),
            IO.Boolean.Input(
                "auto_crop",
                default=False,
                tooltip="Automatically crop to the dominant subject.",
                advanced=True,
            ),
            IO.Int.Input(
                "target_size",
                default=0,
                min=0,
                max=4096,
                tooltip="Square resize applied to the input image before vectorizing, in pixels, "
                "128 to 4096. 0 keeps the source size, which vectorizes more cleanly than forcing a "
                "resize. This does not set the output canvas; use width and height for that.",
                advanced=True,
            ),
            *_v2_model_tail(model),
        ],
    )


def _target_size(model):
    target_size = model.get("target_size") or 0
    return max(target_size, 128) if target_size else None


def _view_box_attributes(model):
    width = model.get("width") or 0
    height = model.get("height") or 0
    if width and height:
        return QuiverSVGAttributes(viewBox=QuiverViewBox(width=width, height=height))
    return None


def _v2_price_badge(tokens, fixed_usd):
    return _arrow_price_badge(
        tokens,
        fixed_usd,
        effort='$lookup(widgets, "model.reasoning_effort")',
        effort_widget="model.reasoning_effort",
    )


class QuiverTextToSVGNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="QuiverTextToSVGNode",
            display_name="Quiver Text to SVG (Legacy)",
            category="partner/image/Quiver",
            description="Generate an SVG from a text prompt using Quiver AI.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text description of the desired SVG output.",
                ),
                IO.String.Input(
                    "instructions",
                    multiline=True,
                    default="",
                    tooltip="Additional style or formatting guidance.",
                    optional=True,
                    advanced=True,
                ),
                IO.Autogrow.Input(
                    "reference_images",
                    template=IO.Autogrow.TemplatePrefix(
                        IO.Image.Input("image"),
                        prefix="ref_",
                        min=0,
                        max=4,
                    ),
                    tooltip="Up to 4 reference images to guide the generation.",
                    optional=True,
                ),
                IO.DynamicCombo.Input(
                    "model",
                    options=[IO.DynamicCombo.Option(m, _arrow_sampling_inputs()) for m in _ARROW_MODELS],
                    tooltip="Model to use for SVG generation.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    tooltip="Seed to determine if node should re-run; "
                    "actual results are nondeterministic regardless of seed.",
                ),
                _reasoning_effort_input(),
            ],
            outputs=[
                IO.SVG.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            is_deprecated=True,
            price_badge=_arrow_price_badge(_GENERATION_TOKENS, _FIXED_GENERATION_USD),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        model: dict,
        seed: int,
        reasoning_effort: str = None,
        instructions: str = None,
        reference_images: IO.Autogrow.Type = None,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=False, min_length=1)

        references = None
        if reference_images:
            references = []
            for key in reference_images:
                url = await upload_image_to_comfyapi(cls, reference_images[key], mime_type="image/png")
                references.append(QuiverImageObject(url=url))

        instructions_val = instructions.strip() if instructions else None
        if instructions_val == "":
            instructions_val = None

        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/quiver/v1/svgs/generations", method="POST"),
            response_model=QuiverSVGResponse,
            data=QuiverTextToSVGRequest(
                model=model["model"],
                reasoning_effort=reasoning_effort if model["model"] in _TOKEN_MODEL_RATES else None,
                prompt=prompt,
                instructions=instructions_val,
                references=references,
                temperature=model.get("temperature"),
                top_p=model.get("top_p"),
                presence_penalty=model.get("presence_penalty"),
            ),
        )

        svg_data = [BytesIO(item.svg.encode("utf-8")) for item in response.data]
        return IO.NodeOutput(SVG(svg_data))


class QuiverImageToSVGNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="QuiverImageToSVGNode",
            display_name="Quiver Image to SVG (Legacy)",
            category="partner/image/Quiver",
            description="Vectorize a raster image into SVG using Quiver AI.",
            inputs=[
                IO.Image.Input(
                    "image",
                    tooltip="Input image to vectorize.",
                ),
                IO.Boolean.Input(
                    "auto_crop",
                    default=False,
                    tooltip="Automatically crop to the dominant subject.",
                    advanced=True,
                ),
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            m,
                            [
                                IO.Int.Input(
                                    "target_size",
                                    default=0,
                                    min=0,
                                    max=4096,
                                    tooltip="Square resize target in pixels, 128 to 4096. 0 keeps the source "
                                    "image size, which vectorizes more cleanly than forcing a resize.",
                                    advanced=True,
                                ),
                                *_arrow_sampling_inputs(),
                            ],
                        )
                        for m in _ARROW_MODELS
                    ],
                    tooltip="Model to use for SVG vectorization.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    tooltip="Seed to determine if node should re-run; "
                    "actual results are nondeterministic regardless of seed.",
                ),
                _reasoning_effort_input(),
            ],
            outputs=[
                IO.SVG.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            is_deprecated=True,
            price_badge=_arrow_price_badge(_VECTORIZATION_TOKENS, _FIXED_VECTORIZATION_USD),
        )

    @classmethod
    async def execute(
        cls,
        image,
        auto_crop: bool,
        model: dict,
        seed: int,
        reasoning_effort: str = None,
    ) -> IO.NodeOutput:
        image_url = await upload_image_to_comfyapi(cls, image, mime_type="image/png")

        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/quiver/v1/svgs/vectorizations", method="POST"),
            response_model=QuiverSVGResponse,
            data=QuiverImageToSVGRequest(
                model=model["model"],
                reasoning_effort=reasoning_effort if model["model"] in _TOKEN_MODEL_RATES else None,
                image=QuiverImageObject(url=image_url),
                auto_crop=auto_crop if auto_crop else None,
                target_size=model.get("target_size") or None,
                temperature=model.get("temperature"),
                top_p=model.get("top_p"),
                presence_penalty=model.get("presence_penalty"),
            ),
        )

        svg_data = [BytesIO(item.svg.encode("utf-8")) for item in response.data]
        return IO.NodeOutput(SVG(svg_data))


class QuiverTextToSVGNodeV2(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="QuiverTextToSVGNodeV2",
            display_name="Quiver Text to SVG",
            category="partner/image/Quiver",
            description="Generate an SVG from a text prompt using Quiver AI.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[_v2_text_option(model) for model in _ARROW_MODELS],
                    tooltip="Model to use for SVG generation.",
                ),
            ],
            outputs=[
                IO.SVG.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=_v2_price_badge(_GENERATION_TOKENS, _FIXED_GENERATION_USD),
        )

    @classmethod
    async def execute(cls, model: dict) -> IO.NodeOutput:
        validate_string(model["prompt"], strip_whitespace=False, min_length=1)

        references = None
        reference_images = model.get("reference_images")
        if reference_images:
            references = []
            for key in reference_images:
                url = await upload_image_to_comfyapi(cls, reference_images[key], mime_type="image/png")
                references.append(QuiverImageObject(url=url))

        instructions = (model.get("instructions") or "").strip() or None

        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/quiver/v1/svgs/generations", method="POST"),
            response_model=QuiverSVGResponse,
            data=QuiverTextToSVGRequest(
                model=model["model"],
                prompt=model["prompt"],
                instructions=instructions,
                references=references,
                reasoning_effort=model.get("reasoning_effort"),
                temperature=model.get("temperature"),
                top_p=model.get("top_p"),
                presence_penalty=model.get("presence_penalty"),
                attributes=_view_box_attributes(model),
            ),
        )

        svg_data = [BytesIO(item.svg.encode("utf-8")) for item in response.data]
        return IO.NodeOutput(SVG(svg_data))


class QuiverImageToSVGNodeV2(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="QuiverImageToSVGNodeV2",
            display_name="Quiver Image to SVG",
            category="partner/image/Quiver",
            description="Vectorize a raster image into SVG using Quiver AI.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[_v2_image_option(model) for model in _ARROW_MODELS],
                    tooltip="Model to use for SVG vectorization.",
                ),
            ],
            outputs=[
                IO.SVG.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=_v2_price_badge(_VECTORIZATION_TOKENS, _FIXED_VECTORIZATION_USD),
        )

    @classmethod
    async def execute(cls, model: dict) -> IO.NodeOutput:
        image_url = await upload_image_to_comfyapi(cls, model["image"], mime_type="image/png")

        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/quiver/v1/svgs/vectorizations", method="POST"),
            response_model=QuiverSVGResponse,
            data=QuiverImageToSVGRequest(
                model=model["model"],
                image=QuiverImageObject(url=image_url),
                auto_crop=model.get("auto_crop") or None,
                target_size=_target_size(model),
                reasoning_effort=model.get("reasoning_effort"),
                temperature=model.get("temperature"),
                top_p=model.get("top_p"),
                presence_penalty=model.get("presence_penalty"),
                attributes=_view_box_attributes(model),
            ),
        )

        svg_data = [BytesIO(item.svg.encode("utf-8")) for item in response.data]
        return IO.NodeOutput(SVG(svg_data))


class QuiverExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            QuiverTextToSVGNode,
            QuiverImageToSVGNode,
            QuiverTextToSVGNodeV2,
            QuiverImageToSVGNodeV2,
        ]


async def comfy_entrypoint() -> QuiverExtension:
    return QuiverExtension()
