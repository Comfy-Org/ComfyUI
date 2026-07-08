from io import BytesIO

from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension
from comfy_api_nodes.apis.quiver import (
    QuiverImageObject,
    QuiverImageToSVGRequest,
    QuiverSVGResponse,
    QuiverTextToSVGRequest,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    sync_op,
    upload_image_to_comfyapi,
    validate_string,
)
from comfy_extras.nodes_images import SVG

_ARROW_MODELS = ["arrow-1.1", "arrow-1.1-max", "arrow-preview"]


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


class QuiverTextToSVGNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="QuiverTextToSVGNode",
            display_name="Quiver Text to SVG",
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
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["model"]),
                expr="""
                (
                  $contains(widgets.model, "max")
                    ? {"type":"usd","usd":0.3575}
                    : $contains(widgets.model, "preview")
                      ? {"type":"usd","usd":0.429}
                      : {"type":"usd","usd":0.286}
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
            if len(references) > 4:
                raise ValueError("Maximum 4 reference images are allowed.")

        instructions_val = instructions.strip() if instructions else None
        if instructions_val == "":
            instructions_val = None

        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/quiver/v1/svgs/generations", method="POST"),
            response_model=QuiverSVGResponse,
            data=QuiverTextToSVGRequest(
                model=model["model"],
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
            display_name="Quiver Image to SVG",
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
                                    default=1024,
                                    min=128,
                                    max=4096,
                                    tooltip="Square resize target in pixels.",
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
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["model"]),
                expr="""
                (
                  $contains(widgets.model, "max")
                    ? {"type":"usd","usd":0.3575}
                    : $contains(widgets.model, "preview")
                      ? {"type":"usd","usd":0.429}
                      : {"type":"usd","usd":0.286}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        image,
        auto_crop: bool,
        model: dict,
        seed: int,
    ) -> IO.NodeOutput:
        image_url = await upload_image_to_comfyapi(cls, image, mime_type="image/png")

        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/quiver/v1/svgs/vectorizations", method="POST"),
            response_model=QuiverSVGResponse,
            data=QuiverImageToSVGRequest(
                model=model["model"],
                image=QuiverImageObject(url=image_url),
                auto_crop=auto_crop if auto_crop else None,
                target_size=model.get("target_size"),
                temperature=model.get("temperature"),
                top_p=model.get("top_p"),
                presence_penalty=model.get("presence_penalty"),
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
        ]


async def comfy_entrypoint() -> QuiverExtension:
    return QuiverExtension()
