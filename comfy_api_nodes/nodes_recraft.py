import base64
import uuid
from io import BytesIO

import aiohttp
import torch
from PIL import UnidentifiedImageError
from typing_extensions import override

from comfy.utils import ProgressBar
from comfy_api.latest import IO, ComfyExtension
from comfy_api_nodes.apis.recraft import (
    RECRAFT_STYLE_MATCH_OPTIONS,
    RECRAFT_STYLE_REFERENCES_MAX,
    RECRAFT_STYLE_REFERENCES_MAX_BYTES,
    RECRAFT_V4_PRO_SIZES,
    RECRAFT_V4_SIZES,
    RECRAFT_V4_STYLES_MODELS,
    RECRAFT_V4_VECTOR_MODEL_FOR_STYLE,
    RecraftColor,
    RecraftColorChain,
    RecraftControls,
    RecraftCreateStyleRequest,
    RecraftCreateStyleResponse,
    RecraftImageGenerationRequest,
    RecraftImageGenerationResponse,
    RecraftImageSize,
    RecraftIO,
    RecraftStyle,
    RecraftStyleV3,
    get_v3_substyles,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    bytesio_to_image_tensor,
    download_url_as_bytesio,
    pad_images_to_common_channels,
    resize_mask_to_image,
    sync_op,
    tensor_to_bytesio,
    validate_string,
)
from comfy_extras.nodes_images import SVG


async def handle_recraft_file_request(
    cls: type[IO.ComfyNode],
    image: torch.Tensor,
    path: str,
    mask: torch.Tensor | None = None,
    total_pixels: int = 4096 * 4096,
    timeout: int = 1024,
    request=None,
) -> list[BytesIO]:
    """Handle sending common Recraft file-only request to get back file bytes."""

    files = {"image": tensor_to_bytesio(image, total_pixels=total_pixels).read()}
    if mask is not None:
        files["mask"] = tensor_to_bytesio(mask, total_pixels=total_pixels).read()

    response = await sync_op(
        cls,
        endpoint=ApiEndpoint(path=path, method="POST"),
        response_model=RecraftImageGenerationResponse,
        data=request if request else None,
        files=files,
        content_type="multipart/form-data",
        multipart_parser=recraft_multipart_parser,
        max_retries=1,
    )
    all_bytesio = []
    if response.image is not None:
        all_bytesio.append(await download_url_as_bytesio(response.image.url, timeout=timeout))
    else:
        for data in response.data:
            all_bytesio.append(await download_url_as_bytesio(data.url, timeout=timeout))

    return all_bytesio


def recraft_multipart_parser(
    data,
    parent_key=None,
    formatter: type[callable] | None = None,
    converted_to_check: list[list] | None = None,
    is_list: bool = False,
    return_mode: str = "formdata",  # "dict" | "formdata"
) -> dict | aiohttp.FormData:
    """
    Formats data such that multipart/form-data will work with aiohttp library when both files and data are present.

    The OpenAI client that Recraft uses has a bizarre way of serializing lists:

    It does NOT keep track of indeces of each list, so for background_color, that must be serialized as:
        'background_color[rgb][]' = [0, 0, 255]
    where the array is assigned to a key that has '[]' at the end, to signal it's an array.

    This has the consequence of nested lists having the exact same key, forcing arrays to merge; all colors inputs fall under the same key:
        if 1 color  -> 'controls[colors][][rgb][]' = [0, 0, 255]
        if 2 colors -> 'controls[colors][][rgb][]' = [0, 0, 255, 255, 0, 0]
        if 3 colors -> 'controls[colors][][rgb][]' = [0, 0, 255, 255, 0, 0, 0, 255, 0]
        etc.
    Whoever made this serialization up at OpenAI added the constraint that lists must be of uniform length on objects of same 'type'.
    """
    # Modification of a function that handled a different type of multipart parsing, big ups:
    # https://gist.github.com/kazqvaizer/4cebebe5db654a414132809f9f88067b

    def handle_converted_lists(item, parent_key, lists_to_check=list[list]):
        # if list already exists, just extend list with data
        for check_list in lists_to_check:
            for conv_tuple in check_list:
                if conv_tuple[0] == parent_key and isinstance(conv_tuple[1], list):
                    conv_tuple[1].append(formatter(item))
                    return True
        return False

    if converted_to_check is None:
        converted_to_check = []

    effective_mode = return_mode if parent_key is None else "dict"
    if formatter is None:
        formatter = lambda v: v  # Multipart representation of value

    if not isinstance(data, dict):
        # if list already exists, just extend list with data
        added = handle_converted_lists(data, parent_key, converted_to_check)
        if added:
            return {}
        # otherwise if is_list, create new list with data
        if is_list:
            return {parent_key: [formatter(data)]}
        # return new key with data
        return {parent_key: formatter(data)}

    converted = []
    next_check = [converted]
    next_check.extend(converted_to_check)

    for key, value in data.items():
        current_key = key if parent_key is None else f"{parent_key}[{key}]"
        if isinstance(value, dict):
            converted.extend(recraft_multipart_parser(value, current_key, formatter, next_check).items())
        elif isinstance(value, list):
            for ind, list_value in enumerate(value):
                iter_key = f"{current_key}[]"
                converted.extend(
                    recraft_multipart_parser(list_value, iter_key, formatter, next_check, is_list=True).items()
                )
        else:
            converted.append((current_key, formatter(value)))

    if effective_mode == "formdata":
        fd = aiohttp.FormData()
        for k, v in dict(converted).items():
            if isinstance(v, list):
                for item in v:
                    fd.add_field(k, str(item))
            else:
                fd.add_field(k, str(v))
        return fd
    return dict(converted)


class handle_recraft_image_output:
    """
    Catch an exception related to receiving SVG data instead of image, when Infinite Style Library style_id is in use.
    """

    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and exc_type is UnidentifiedImageError:
            raise Exception(
                "Received output data was not an image; likely an SVG. "
                "If you used style_id, make sure it is not a Vector art style."
            )


def style_reference_images(images: IO.Autogrow.Type | None) -> list[torch.Tensor]:
    references = []
    for batch in (images or {}).values():
        if batch is None:
            continue
        if batch.ndim == 3:
            batch = batch.unsqueeze(0)
        references.extend(batch[i] for i in range(batch.shape[0]))
    return references


def encode_style_references(references: list[torch.Tensor]) -> list[bytes]:
    if not references:
        raise ValueError("At least one style reference image is required.")
    if len(references) > RECRAFT_STYLE_REFERENCES_MAX:
        raise ValueError(
            f"At most {RECRAFT_STYLE_REFERENCES_MAX} style reference images are allowed; got {len(references)}."
        )
    encoded = []
    total_size = 0
    for reference in references:
        data = tensor_to_bytesio(reference, total_pixels=2048 * 2048, mime_type="image/webp").read()
        total_size += len(data)
        if total_size > RECRAFT_STYLE_REFERENCES_MAX_BYTES:
            raise ValueError("Total size of style reference images exceeds the 10 MB limit.")
        encoded.append(data)
    return encoded


def resolve_v4_style(
    model: str, style_id: str, style_references: IO.Autogrow.Type | None
) -> tuple[str | None, list[str] | None]:
    style_id = style_id.strip()
    references = style_reference_images(style_references)
    if style_id and references:
        raise ValueError("Provide either a style_id or style reference images, not both.")
    if style_id:
        try:
            uuid.UUID(style_id)
        except ValueError:
            raise ValueError(f"style_id must be a UUID; got '{style_id}'.") from None
        return style_id, None
    if references:
        return None, [
            f"data:image/webp;base64,{base64.b64encode(data).decode()}"
            for data in encode_style_references(references)
        ]
    if model in RECRAFT_V4_STYLES_MODELS:
        raise ValueError(
            f"Model '{model}' always requires a style: connect style reference images or provide a style_id."
        )
    return None, None


class RecraftColorRGBNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RecraftColorRGB",
            display_name="Recraft Color RGB",
            category="partner/image/Recraft",
            description="Create Recraft Color by choosing specific RGB values.",
            inputs=[
                IO.Int.Input("r", default=0, min=0, max=255, tooltip="Red value of color."),
                IO.Int.Input("g", default=0, min=0, max=255, tooltip="Green value of color."),
                IO.Int.Input("b", default=0, min=0, max=255, tooltip="Blue value of color."),
                IO.Custom(RecraftIO.COLOR).Input("recraft_color", optional=True),
            ],
            outputs=[
                IO.Custom(RecraftIO.COLOR).Output(display_name="recraft_color"),
            ],
        )

    @classmethod
    def execute(cls, r: int, g: int, b: int, recraft_color: RecraftColorChain = None) -> IO.NodeOutput:
        recraft_color = recraft_color.clone() if recraft_color else RecraftColorChain()
        recraft_color.add(RecraftColor(r, g, b))
        return IO.NodeOutput(recraft_color)


class RecraftControlsNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RecraftControls",
            display_name="Recraft Controls",
            category="partner/image/Recraft",
            description="Create Recraft Controls for customizing Recraft generation.",
            inputs=[
                IO.Custom(RecraftIO.COLOR).Input("colors", optional=True),
                IO.Custom(RecraftIO.COLOR).Input("background_color", optional=True),
            ],
            outputs=[
                IO.Custom(RecraftIO.CONTROLS).Output(display_name="recraft_controls"),
            ],
        )

    @classmethod
    def execute(cls, colors: RecraftColorChain = None, background_color: RecraftColorChain = None) -> IO.NodeOutput:
        return IO.NodeOutput(RecraftControls(colors=colors, background_color=background_color))


class RecraftStyleV3RealisticImageNode(IO.ComfyNode):
    RECRAFT_STYLE = RecraftStyleV3.realistic_image

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RecraftStyleV3RealisticImage",
            display_name="Recraft Style - Realistic Image",
            category="partner/image/Recraft",
            description="Select realistic_image style and optional substyle.",
            inputs=[
                IO.Combo.Input("substyle", options=get_v3_substyles(cls.RECRAFT_STYLE)),
            ],
            outputs=[
                IO.Custom(RecraftIO.STYLEV3).Output(display_name="recraft_style"),
            ],
        )

    @classmethod
    def execute(cls, substyle: str) -> IO.NodeOutput:
        if substyle == "None":
            substyle = None
        return IO.NodeOutput(RecraftStyle(cls.RECRAFT_STYLE, substyle))


class RecraftStyleV3DigitalIllustrationNode(RecraftStyleV3RealisticImageNode):
    RECRAFT_STYLE = RecraftStyleV3.digital_illustration

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RecraftStyleV3DigitalIllustration",
            display_name="Recraft Style - Digital Illustration",
            category="partner/image/Recraft",
            description="Select realistic_image style and optional substyle.",
            inputs=[
                IO.Combo.Input("substyle", options=get_v3_substyles(cls.RECRAFT_STYLE)),
            ],
            outputs=[
                IO.Custom(RecraftIO.STYLEV3).Output(display_name="recraft_style"),
            ],
        )


class RecraftStyleV3VectorIllustrationNode(RecraftStyleV3RealisticImageNode):
    RECRAFT_STYLE = RecraftStyleV3.vector_illustration

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RecraftStyleV3VectorIllustrationNode",
            display_name="Recraft Style - Vector Illustration",
            category="partner/image/Recraft",
            description="Select vector_illustration style and optional substyle.",
            inputs=[
                IO.Combo.Input("substyle", options=get_v3_substyles(cls.RECRAFT_STYLE)),
            ],
            outputs=[
                IO.Custom(RecraftIO.STYLEV3).Output(display_name="recraft_style"),
            ],
        )


class RecraftStyleV3LogoRasterNode(RecraftStyleV3RealisticImageNode):
    RECRAFT_STYLE = RecraftStyleV3.logo_raster

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RecraftStyleV3LogoRaster",
            display_name="Recraft Style - Logo Raster",
            category="partner/image/Recraft",
            description="Select logo_raster style and substyle.",
            inputs=[
                IO.Combo.Input("substyle", options=get_v3_substyles(cls.RECRAFT_STYLE, include_none=False)),
            ],
            outputs=[
                IO.Custom(RecraftIO.STYLEV3).Output(display_name="recraft_style"),
            ],
        )


class RecraftStyleInfiniteStyleLibrary(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RecraftStyleV3InfiniteStyleLibrary",
            display_name="Recraft Style - Infinite Style Library",
            category="partner/image/Recraft",
            description="Choose style based on preexisting UUID from Recraft's Infinite Style Library.",
            inputs=[
                IO.String.Input("style_id", default="", tooltip="UUID of style from Infinite Style Library."),
            ],
            outputs=[
                IO.Custom(RecraftIO.STYLEV3).Output(display_name="recraft_style"),
            ],
        )

    @classmethod
    def execute(cls, style_id: str) -> IO.NodeOutput:
        if not style_id:
            raise Exception("The style_id input cannot be empty.")
        return IO.NodeOutput(RecraftStyle(style_id=style_id))


class RecraftCreateStyleNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RecraftCreateStyleNode",
            display_name="Recraft Create Style",
            category="partner/image/Recraft",
            description="Create a custom style from reference images. "
            "Upload 1-5 images to use as style references. "
            "Total size of all images is limited to 5 MB.",
            inputs=[
                IO.Combo.Input(
                    "style",
                    options=["realistic_image", "digital_illustration"],
                    tooltip="The base style of the generated images.",
                ),
                IO.Autogrow.Input(
                    "images",
                    template=IO.Autogrow.TemplatePrefix(
                        IO.Image.Input("image"),
                        prefix="image",
                        min=1,
                        max=5,
                    ),
                ),
            ],
            outputs=[
                IO.String.Output(display_name="style_id"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd": 0.04}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        style: str,
        images: IO.Autogrow.Type,
    ) -> IO.NodeOutput:
        files = []
        total_size = 0
        max_total_size = 5 * 1024 * 1024  # 5 MB limit
        for i, img in enumerate(list(images.values())):
            file_bytes = tensor_to_bytesio(img, total_pixels=2048 * 2048, mime_type="image/webp").read()
            total_size += len(file_bytes)
            if total_size > max_total_size:
                raise Exception("Total size of all images exceeds 5 MB limit.")
            files.append((f"file{i + 1}", file_bytes))

        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/recraft/styles", method="POST"),
            response_model=RecraftCreateStyleResponse,
            files=files,
            data=RecraftCreateStyleRequest(style=style),
            content_type="multipart/form-data",
            max_retries=1,
        )

        return IO.NodeOutput(response.id)


class RecraftV4CreateStyleNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RecraftV4CreateStyleNode",
            display_name="Recraft V4 Create Style",
            category="partner/image/Recraft",
            description="Create a reusable Recraft V4 style from 1-10 reference images. "
            "The returned style_id works with every Recraft V4 and V4.1 model of the same output type. "
            "Total size of all images is limited to 10 MB.",
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=[
                        "recraftv4_styles",
                        "recraftv4_styles_vector",
                        "recraftv4_styles_pro",
                        "recraftv4_styles_pro_vector",
                    ],
                    tooltip="Model the style is created for. Standard and Pro share one style pool: raster styles "
                    "work with every Recraft V4 and V4.1 raster model, vector styles (*_vector) with every "
                    "V4 and V4.1 vector model.",
                ),
                IO.Autogrow.Input(
                    "images",
                    template=IO.Autogrow.TemplatePrefix(
                        IO.Image.Input("image"),
                        prefix="image",
                        min=1,
                        max=RECRAFT_STYLE_REFERENCES_MAX,
                    ),
                    tooltip="Reference images defining the style. Similar references sharpen the match, "
                    "varied references widen it.",
                ),
            ],
            outputs=[
                IO.String.Output(display_name="style_id"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd": 0.005}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        images: IO.Autogrow.Type,
    ) -> IO.NodeOutput:
        files = [
            (f"file{i + 1}", data)
            for i, data in enumerate(encode_style_references(style_reference_images(images)))
        ]
        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/recraft/styles", method="POST"),
            response_model=RecraftCreateStyleResponse,
            files=files,
            data=RecraftCreateStyleRequest(
                style="vector_illustration" if model.endswith("_vector") else "any",
                model=model,
            ),
            content_type="multipart/form-data",
            max_retries=1,
        )
        return IO.NodeOutput(response.id)


class RecraftTextToImageNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RecraftTextToImageNode",
            display_name="Recraft V3 Text to Image",
            category="partner/image/Recraft",
            description="Generates images synchronously based on prompt and resolution.",
            inputs=[
                IO.String.Input("prompt", multiline=True, default="", tooltip="Prompt for the image generation."),
                IO.Combo.Input(
                    "size",
                    options=[res.value for res in RecraftImageSize],
                    default=RecraftImageSize.res_1024x1024,
                    tooltip="The size of the generated image.",
                ),
                IO.Int.Input(
                    "n",
                    default=1,
                    min=1,
                    max=6,
                    tooltip="The number of images to generate.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Seed to determine if node should re-run; "
                    "actual results are nondeterministic regardless of seed.",
                ),
                IO.Custom(RecraftIO.STYLEV3).Input("recraft_style", optional=True),
                IO.String.Input(
                    "negative_prompt",
                    default="",
                    force_input=True,
                    tooltip="An optional text description of undesired elements on an image.",
                    optional=True,
                ),
                IO.Custom(RecraftIO.CONTROLS).Input(
                    "recraft_controls",
                    tooltip="Optional additional controls over the generation via the Recraft Controls node.",
                    optional=True,
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
                depends_on=IO.PriceBadgeDepends(widgets=["n"]),
                expr="""{"type":"usd","usd": $round(0.04 * widgets.n, 2)}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        size: str,
        n: int,
        seed,
        recraft_style: RecraftStyle = None,
        negative_prompt: str = None,
        recraft_controls: RecraftControls = None,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=False, min_length=1, max_length=1000)
        default_style = RecraftStyle(RecraftStyleV3.realistic_image)
        if recraft_style is None:
            recraft_style = default_style

        controls_api = None
        if recraft_controls:
            controls_api = recraft_controls.create_api_model()

        if not negative_prompt:
            negative_prompt = None

        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/recraft/image_generation", method="POST"),
            response_model=RecraftImageGenerationResponse,
            data=RecraftImageGenerationRequest(
                prompt=prompt,
                negative_prompt=negative_prompt,
                model="recraftv3",
                size=size,
                n=n,
                style=recraft_style.style,
                substyle=recraft_style.substyle,
                style_id=recraft_style.style_id,
                controls=controls_api,
            ),
            max_retries=1,
        )
        images = []
        for data in response.data:
            with handle_recraft_image_output():
                image = bytesio_to_image_tensor(await download_url_as_bytesio(data.url, timeout=1024))
            if len(image.shape) < 4:
                image = image.unsqueeze(0)
            images.append(image)

        return IO.NodeOutput(torch.cat(images, dim=0))


class RecraftImageToImageNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RecraftImageToImageNode",
            display_name="Recraft V3 Image to Image",
            category="partner/image/Recraft",
            description="Modify image based on prompt and strength.",
            inputs=[
                IO.Image.Input("image"),
                IO.String.Input("prompt", multiline=True, default="", tooltip="Prompt for the image generation."),
                IO.Int.Input(
                    "n",
                    default=1,
                    min=1,
                    max=6,
                    tooltip="The number of images to generate.",
                ),
                IO.Float.Input(
                    "strength",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Defines the difference with the original image, should lie in [0, 1], "
                    "where 0 means almost identical, and 1 means miserable similarity.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Seed to determine if node should re-run; "
                    "actual results are nondeterministic regardless of seed.",
                ),
                IO.Custom(RecraftIO.STYLEV3).Input("recraft_style", optional=True),
                IO.String.Input(
                    "negative_prompt",
                    default="",
                    force_input=True,
                    tooltip="An optional text description of undesired elements on an image.",
                    optional=True,
                ),
                IO.Custom(RecraftIO.CONTROLS).Input(
                    "recraft_controls",
                    tooltip="Optional additional controls over the generation via the Recraft Controls node.",
                    optional=True,
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
                depends_on=IO.PriceBadgeDepends(widgets=["n"]),
                expr="""{"type":"usd","usd": $round(0.04 * widgets.n, 2)}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        prompt: str,
        n: int,
        strength: float,
        seed,
        recraft_style: RecraftStyle = None,
        negative_prompt: str = None,
        recraft_controls: RecraftControls = None,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=False, max_length=1000)
        default_style = RecraftStyle(RecraftStyleV3.realistic_image)
        if recraft_style is None:
            recraft_style = default_style

        controls_api = None
        if recraft_controls:
            controls_api = recraft_controls.create_api_model()

        if not negative_prompt:
            negative_prompt = None

        request = RecraftImageGenerationRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            model="recraftv3",
            n=n,
            strength=round(strength, 2),
            style=recraft_style.style,
            substyle=recraft_style.substyle,
            style_id=recraft_style.style_id,
            controls=controls_api,
        )

        images = []
        total = image.shape[0]
        pbar = ProgressBar(total)
        for i in range(total):
            sub_bytes = await handle_recraft_file_request(
                cls,
                image=image[i],
                path="/proxy/recraft/images/imageToImage",
                request=request,
            )
            with handle_recraft_image_output():
                images.append(torch.cat([bytesio_to_image_tensor(x) for x in sub_bytes], dim=0))
            pbar.update(1)

        return IO.NodeOutput(torch.cat(pad_images_to_common_channels(images), dim=0))


class RecraftImageInpaintingNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RecraftImageInpaintingNode",
            display_name="Recraft Image Inpainting",
            category="partner/image/Recraft",
            description="Modify image based on prompt and mask.",
            inputs=[
                IO.Image.Input("image"),
                IO.Mask.Input("mask"),
                IO.String.Input("prompt", multiline=True, default="", tooltip="Prompt for the image generation."),
                IO.Int.Input(
                    "n",
                    default=1,
                    min=1,
                    max=6,
                    tooltip="The number of images to generate.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Seed to determine if node should re-run; "
                    "actual results are nondeterministic regardless of seed.",
                ),
                IO.Custom(RecraftIO.STYLEV3).Input("recraft_style", optional=True),
                IO.String.Input(
                    "negative_prompt",
                    default="",
                    force_input=True,
                    tooltip="An optional text description of undesired elements on an image.",
                    optional=True,
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
                depends_on=IO.PriceBadgeDepends(widgets=["n"]),
                expr="""{"type":"usd","usd": $round(0.04 * widgets.n, 2)}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        mask: torch.Tensor,
        prompt: str,
        n: int,
        seed,
        recraft_style: RecraftStyle = None,
        negative_prompt: str = None,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=False, max_length=1000)
        default_style = RecraftStyle(RecraftStyleV3.realistic_image)
        if recraft_style is None:
            recraft_style = default_style

        if not negative_prompt:
            negative_prompt = None

        request = RecraftImageGenerationRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            model="recraftv3",
            n=n,
            style=recraft_style.style,
            substyle=recraft_style.substyle,
            style_id=recraft_style.style_id,
        )

        # prepare mask tensor
        mask = resize_mask_to_image(mask, image, allow_gradient=False, add_channel_dim=True)

        images = []
        total = image.shape[0]
        pbar = ProgressBar(total)
        for i in range(total):
            sub_bytes = await handle_recraft_file_request(
                cls,
                image=image[i],
                mask=mask[i : i + 1],
                path="/proxy/recraft/images/inpaint",
                request=request,
            )
            with handle_recraft_image_output():
                images.append(torch.cat([bytesio_to_image_tensor(x) for x in sub_bytes], dim=0))
            pbar.update(1)

        return IO.NodeOutput(torch.cat(pad_images_to_common_channels(images), dim=0))


class RecraftTextToVectorNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RecraftTextToVectorNode",
            display_name="Recraft V3 Text to Vector",
            category="partner/image/Recraft",
            description="Generates SVG synchronously based on prompt and resolution.",
            inputs=[
                IO.String.Input("prompt", default="", tooltip="Prompt for the image generation.", multiline=True),
                IO.Combo.Input("substyle", options=get_v3_substyles(RecraftStyleV3.vector_illustration)),
                IO.Combo.Input(
                    "size",
                    options=[res.value for res in RecraftImageSize],
                    default=RecraftImageSize.res_1024x1024,
                    tooltip="The size of the generated image.",
                ),
                IO.Int.Input("n", default=1, min=1, max=6, tooltip="The number of images to generate."),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Seed to determine if node should re-run; "
                    "actual results are nondeterministic regardless of seed.",
                ),
                IO.String.Input(
                    "negative_prompt",
                    default="",
                    force_input=True,
                    tooltip="An optional text description of undesired elements on an image.",
                    optional=True,
                ),
                IO.Custom(RecraftIO.CONTROLS).Input(
                    "recraft_controls",
                    tooltip="Optional additional controls over the generation via the Recraft Controls node.",
                    optional=True,
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
                depends_on=IO.PriceBadgeDepends(widgets=["n"]),
                expr="""{"type":"usd","usd": $round(0.08 * widgets.n, 2)}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        substyle: str,
        size: str,
        n: int,
        seed,
        negative_prompt: str = None,
        recraft_controls: RecraftControls = None,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=False, max_length=1000)
        # create RecraftStyle so strings will be formatted properly (i.e. "None" will become None)
        recraft_style = RecraftStyle(RecraftStyleV3.vector_illustration, substyle=substyle)

        controls_api = None
        if recraft_controls:
            controls_api = recraft_controls.create_api_model()

        if not negative_prompt:
            negative_prompt = None

        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/recraft/image_generation", method="POST"),
            response_model=RecraftImageGenerationResponse,
            data=RecraftImageGenerationRequest(
                prompt=prompt,
                negative_prompt=negative_prompt,
                model="recraftv3",
                size=size,
                n=n,
                style=recraft_style.style,
                substyle=recraft_style.substyle,
                controls=controls_api,
            ),
            max_retries=1,
        )
        svg_data = []
        for data in response.data:
            svg_data.append(await download_url_as_bytesio(data.url, timeout=1024))

        return IO.NodeOutput(SVG(svg_data))


class RecraftVectorizeImageNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RecraftVectorizeImageNode",
            display_name="Recraft Vectorize Image",
            category="partner/image/Recraft",
            essentials_category="Image Tools",
            description="Generates SVG synchronously from an input image.",
            inputs=[
                IO.Image.Input("image"),
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
                depends_on=IO.PriceBadgeDepends(),
                expr="""{"type":"usd","usd": 0.01}""",
            ),
        )

    @classmethod
    async def execute(cls, image: torch.Tensor) -> IO.NodeOutput:
        svgs = []
        total = image.shape[0]
        pbar = ProgressBar(total)
        for i in range(total):
            sub_bytes = await handle_recraft_file_request(
                cls,
                image=image[i],
                path="/proxy/recraft/images/vectorize",
            )
            svgs.append(SVG(sub_bytes))
            pbar.update(1)

        return IO.NodeOutput(SVG.combine_all(svgs))


class RecraftReplaceBackgroundNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RecraftReplaceBackgroundNode",
            display_name="Recraft Replace Background",
            category="partner/image/Recraft",
            description="Replace background on image, based on provided prompt.",
            inputs=[
                IO.Image.Input("image"),
                IO.String.Input("prompt", tooltip="Prompt for the image generation.", default="", multiline=True),
                IO.Int.Input("n", default=1, min=1, max=6, tooltip="The number of images to generate."),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Seed to determine if node should re-run; "
                    "actual results are nondeterministic regardless of seed.",
                ),
                IO.Custom(RecraftIO.STYLEV3).Input("recraft_style", optional=True),
                IO.String.Input(
                    "negative_prompt",
                    default="",
                    force_input=True,
                    tooltip="An optional text description of undesired elements on an image.",
                    optional=True,
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
                expr="""{"type":"usd","usd":0.04}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        prompt: str,
        n: int,
        seed,
        recraft_style: RecraftStyle = None,
        negative_prompt: str = None,
    ) -> IO.NodeOutput:
        default_style = RecraftStyle(RecraftStyleV3.realistic_image)
        if recraft_style is None:
            recraft_style = default_style

        if not negative_prompt:
            negative_prompt = None

        request = RecraftImageGenerationRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            model="recraftv3",
            n=n,
            style=recraft_style.style,
            substyle=recraft_style.substyle,
            style_id=recraft_style.style_id,
        )

        images = []
        total = image.shape[0]
        pbar = ProgressBar(total)
        for i in range(total):
            sub_bytes = await handle_recraft_file_request(
                cls,
                image=image[i],
                path="/proxy/recraft/images/replaceBackground",
                request=request,
            )
            images.append(torch.cat([bytesio_to_image_tensor(x) for x in sub_bytes], dim=0))
            pbar.update(1)

        return IO.NodeOutput(torch.cat(pad_images_to_common_channels(images), dim=0))


class RecraftRemoveBackgroundNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RecraftRemoveBackgroundNode",
            display_name="Recraft Remove Background",
            category="partner/image/Recraft",
            essentials_category="Image Tools",
            description="Remove background from image, and return processed image and mask.",
            inputs=[
                IO.Image.Input("image"),
            ],
            outputs=[
                IO.Image.Output(),
                IO.Mask.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.01}""",
            ),
        )

    @classmethod
    async def execute(cls, image: torch.Tensor) -> IO.NodeOutput:
        images = []
        total = image.shape[0]
        pbar = ProgressBar(total)
        for i in range(total):
            sub_bytes = await handle_recraft_file_request(
                cls,
                image=image[i],
                path="/proxy/recraft/images/removeBackground",
            )
            images.append(torch.cat([bytesio_to_image_tensor(x, mode="RGBA") for x in sub_bytes], dim=0))
            pbar.update(1)

        images_tensor = torch.cat(images, dim=0)
        # use alpha channel as masks, in B,H,W format
        masks_tensor = images_tensor[:, :, :, -1:].squeeze(-1)
        return IO.NodeOutput(images_tensor, masks_tensor)


class RecraftCrispUpscaleNode(IO.ComfyNode):
    RECRAFT_PATH = "/proxy/recraft/images/crispUpscale"

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RecraftCrispUpscaleNode",
            display_name="Recraft Crisp Upscale Image",
            category="partner/image/Recraft",
            description="Upscale image synchronously.\n"
            "Enhances a given raster image using ‘crisp upscale’ tool, "
            "increasing image resolution, making the image sharper and cleaner.",
            inputs=[
                IO.Image.Input("image"),
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
                expr="""{"type":"usd","usd":0.004}""",
            ),
        )

    @classmethod
    async def execute(cls, image: torch.Tensor) -> IO.NodeOutput:
        images = []
        total = image.shape[0]
        pbar = ProgressBar(total)
        for i in range(total):
            sub_bytes = await handle_recraft_file_request(
                cls,
                image=image[i],
                path=cls.RECRAFT_PATH,
            )
            images.append(torch.cat([bytesio_to_image_tensor(x) for x in sub_bytes], dim=0))
            pbar.update(1)

        return IO.NodeOutput(torch.cat(pad_images_to_common_channels(images), dim=0))


class RecraftCreativeUpscaleNode(RecraftCrispUpscaleNode):
    RECRAFT_PATH = "/proxy/recraft/images/creativeUpscale"

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RecraftCreativeUpscaleNode",
            display_name="Recraft Creative Upscale Image",
            category="partner/image/Recraft",
            description="Upscale image synchronously.\n"
            "Enhances a given raster image using ‘creative upscale’ tool, "
            "boosting resolution with a focus on refining small details and faces.",
            inputs=[
                IO.Image.Input("image"),
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
                expr="""{"type":"usd","usd":0.25}""",
            ),
        )


class RecraftV4TextToImageNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RecraftV4TextToImageNode",
            display_name="Recraft V4 Text to Image",
            category="partner/image/Recraft",
            description="Generates images using Recraft V4 and V4.1 models.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="Prompt for the image generation. Maximum 10,000 characters.",
                ),
                IO.String.Input(
                    "negative_prompt",
                    multiline=True,
                    tooltip="This input is ignored: negative prompt is not supported by "
                    "Recraft V4 and V4.1 models.",
                ),
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "recraftv4_1",
                            [
                                IO.Combo.Input(
                                    "size",
                                    options=RECRAFT_V4_SIZES,
                                    default="1024x1024",
                                    tooltip="The size of the generated image.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "recraftv4_1_utility",
                            [
                                IO.Combo.Input(
                                    "size",
                                    options=RECRAFT_V4_SIZES,
                                    default="1024x1024",
                                    tooltip="The size of the generated image.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "recraftv4_1_pro",
                            [
                                IO.Combo.Input(
                                    "size",
                                    options=RECRAFT_V4_PRO_SIZES,
                                    default="2048x2048",
                                    tooltip="The size of the generated image.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "recraftv4_1_utility_pro",
                            [
                                IO.Combo.Input(
                                    "size",
                                    options=RECRAFT_V4_PRO_SIZES,
                                    default="2048x2048",
                                    tooltip="The size of the generated image.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "recraftv4",
                            [
                                IO.Combo.Input(
                                    "size",
                                    options=RECRAFT_V4_SIZES,
                                    default="1024x1024",
                                    tooltip="The size of the generated image.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "recraftv4_pro",
                            [
                                IO.Combo.Input(
                                    "size",
                                    options=RECRAFT_V4_PRO_SIZES,
                                    default="2048x2048",
                                    tooltip="The size of the generated image.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "recraftv4_styles",
                            [
                                IO.Combo.Input(
                                    "size",
                                    options=RECRAFT_V4_SIZES,
                                    default="1024x1024",
                                    tooltip="The size of the generated image.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "recraftv4_styles_pro",
                            [
                                IO.Combo.Input(
                                    "size",
                                    options=RECRAFT_V4_PRO_SIZES,
                                    default="2048x2048",
                                    tooltip="The size of the generated image.",
                                ),
                            ],
                        ),
                    ],
                    tooltip="The model to use for generation. The recraftv4_styles models are built for "
                    "style-consistent generation and always require a style_id or style_references.",
                ),
                IO.Int.Input(
                    "n",
                    default=1,
                    min=1,
                    max=6,
                    tooltip="The number of images to generate.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Seed to determine if node should re-run; "
                    "actual results are nondeterministic regardless of seed.",
                ),
                IO.Custom(RecraftIO.CONTROLS).Input(
                    "recraft_controls",
                    tooltip="Optional additional controls over the generation via the Recraft Controls node.",
                    optional=True,
                ),
                IO.String.Input(
                    "style_id",
                    default="",
                    optional=True,
                    tooltip="UUID of a Recraft V4 style to apply, e.g. from the Recraft V4 Create Style node "
                    "or the style_id output of a previous run. Cannot be combined with style_references.",
                ),
                IO.Combo.Input(
                    "style_match",
                    options=RECRAFT_STYLE_MATCH_OPTIONS,
                    optional=True,
                    tooltip="How closely to follow the style: precise reproduces it in detail, "
                    "flexible matches the general look. Only used when a style is provided.",
                ),
                IO.Autogrow.Input(
                    "style_references",
                    template=IO.Autogrow.TemplatePrefix(
                        IO.Image.Input("style_reference"),
                        prefix="style_reference",
                        min=0,
                        max=RECRAFT_STYLE_REFERENCES_MAX,
                    ),
                    optional=True,
                    tooltip="Reference images to create a style from on the fly, billed on top of the generation. "
                    "The created style is returned as style_id for reuse. Cannot be combined with style_id.",
                ),
            ],
            outputs=[
                IO.Image.Output(),
                IO.String.Output(id="style_id", display_name="style_id"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["model", "n"], input_groups=["style_references"]),
                expr="""
                (
                    $prices := {
                        "recraftv4_1": 0.035,
                        "recraftv4_1_utility": 0.035,
                        "recraftv4_1_pro": 0.21,
                        "recraftv4_1_utility_pro": 0.21,
                        "recraftv4": 0.04,
                        "recraftv4_pro": 0.25,
                        "recraftv4_styles": 0.035,
                        "recraftv4_styles_pro": 0.10
                    };
                    $references := $lookup(inputGroups, "style_references");
                    $style := ($references ? $references : 0) > 0 ? 0.005 : 0;
                    {"type":"usd","usd": $lookup($prices, widgets.model) * widgets.n + $style}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        negative_prompt: str,
        model: dict,
        n: int,
        seed: int,
        recraft_controls: RecraftControls | None = None,
        style_id: str = "",
        style_match: str = "precise",
        style_references: IO.Autogrow.Type | None = None,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1, max_length=10000)
        style_id, style_reference_urls = resolve_v4_style(model["model"], style_id, style_references)
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/recraft/image_generation", method="POST"),
            response_model=RecraftImageGenerationResponse,
            data=RecraftImageGenerationRequest(
                prompt=prompt,
                model=model["model"],
                size=model["size"],
                n=n,
                style_id=style_id,
                style_match=style_match if style_id or style_reference_urls else None,
                style_reference_urls=style_reference_urls,
                controls=recraft_controls.create_api_model() if recraft_controls else None,
            ),
            max_retries=1,
        )
        images = []
        for data in response.data:
            with handle_recraft_image_output():
                image = bytesio_to_image_tensor(await download_url_as_bytesio(data.url, timeout=1024))
            if len(image.shape) < 4:
                image = image.unsqueeze(0)
            images.append(image)
        return IO.NodeOutput(torch.cat(images, dim=0), response.style_id or "")


class RecraftV4TextToVectorNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RecraftV4TextToVectorNode",
            display_name="Recraft V4 Text to Vector",
            category="partner/image/Recraft",
            description="Generates SVG using Recraft V4 and V4.1 models.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="Prompt for the image generation. Maximum 10,000 characters.",
                ),
                IO.String.Input(
                    "negative_prompt",
                    multiline=True,
                    tooltip="This input is ignored: negative prompt is not supported by "
                    "Recraft V4 and V4.1 models.",
                ),
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "recraftv4_1_vector",
                            [
                                IO.Combo.Input(
                                    "size",
                                    options=RECRAFT_V4_SIZES,
                                    default="1024x1024",
                                    tooltip="The size of the generated image.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "recraftv4_1_utility_vector",
                            [
                                IO.Combo.Input(
                                    "size",
                                    options=RECRAFT_V4_SIZES,
                                    default="1024x1024",
                                    tooltip="The size of the generated image.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "recraftv4_1_pro_vector",
                            [
                                IO.Combo.Input(
                                    "size",
                                    options=RECRAFT_V4_PRO_SIZES,
                                    default="2048x2048",
                                    tooltip="The size of the generated image.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "recraftv4_1_utility_pro_vector",
                            [
                                IO.Combo.Input(
                                    "size",
                                    options=RECRAFT_V4_PRO_SIZES,
                                    default="2048x2048",
                                    tooltip="The size of the generated image.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "recraftv4",
                            [
                                IO.Combo.Input(
                                    "size",
                                    options=RECRAFT_V4_SIZES,
                                    default="1024x1024",
                                    tooltip="The size of the generated image.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "recraftv4_pro",
                            [
                                IO.Combo.Input(
                                    "size",
                                    options=RECRAFT_V4_PRO_SIZES,
                                    default="2048x2048",
                                    tooltip="The size of the generated image.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "recraftv4_styles_vector",
                            [
                                IO.Combo.Input(
                                    "size",
                                    options=RECRAFT_V4_SIZES,
                                    default="1024x1024",
                                    tooltip="The size of the generated image.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "recraftv4_styles_pro_vector",
                            [
                                IO.Combo.Input(
                                    "size",
                                    options=RECRAFT_V4_PRO_SIZES,
                                    default="2048x2048",
                                    tooltip="The size of the generated image.",
                                ),
                            ],
                        ),
                    ],
                    tooltip="The model to use for generation. The recraftv4_styles models are built for "
                    "style-consistent generation and always require a style_id or style_references.",
                ),
                IO.Int.Input(
                    "n",
                    default=1,
                    min=1,
                    max=6,
                    tooltip="The number of images to generate.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Seed to determine if node should re-run; "
                    "actual results are nondeterministic regardless of seed.",
                ),
                IO.Custom(RecraftIO.CONTROLS).Input(
                    "recraft_controls",
                    tooltip="Optional additional controls over the generation via the Recraft Controls node.",
                    optional=True,
                ),
                IO.String.Input(
                    "style_id",
                    default="",
                    optional=True,
                    tooltip="UUID of a Recraft V4 vector style to apply, e.g. from the Recraft V4 Create Style node "
                    "or the style_id output of a previous run. Cannot be combined with style_references.",
                ),
                IO.Combo.Input(
                    "style_match",
                    options=RECRAFT_STYLE_MATCH_OPTIONS,
                    optional=True,
                    tooltip="How closely to follow the style: precise reproduces it in detail, "
                    "flexible matches the general look. Only used when a style is provided.",
                ),
                IO.Autogrow.Input(
                    "style_references",
                    template=IO.Autogrow.TemplatePrefix(
                        IO.Image.Input("style_reference"),
                        prefix="style_reference",
                        min=0,
                        max=RECRAFT_STYLE_REFERENCES_MAX,
                    ),
                    optional=True,
                    tooltip="Reference images to create a vector style from on the fly, billed on top of the generation. "
                    "The created style is returned as style_id for reuse. Cannot be combined with style_id.",
                ),
            ],
            outputs=[
                IO.SVG.Output(),
                IO.String.Output(id="style_id", display_name="style_id"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["model", "n"], input_groups=["style_references"]),
                expr="""
                (
                    $prices := {
                        "recraftv4_1_vector": 0.08,
                        "recraftv4_1_utility_vector": 0.08,
                        "recraftv4_1_pro_vector": 0.30,
                        "recraftv4_1_utility_pro_vector": 0.30,
                        "recraftv4": 0.08,
                        "recraftv4_pro": 0.30,
                        "recraftv4_styles_vector": 0.05,
                        "recraftv4_styles_pro_vector": 0.12
                    };
                    $references := $lookup(inputGroups, "style_references");
                    $style := ($references ? $references : 0) > 0 ? 0.005 : 0;
                    {"type":"usd","usd": $lookup($prices, widgets.model) * widgets.n + $style}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        negative_prompt: str,
        model: dict,
        n: int,
        seed: int,
        recraft_controls: RecraftControls | None = None,
        style_id: str = "",
        style_match: str = "precise",
        style_references: IO.Autogrow.Type | None = None,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1, max_length=10000)
        model_id = model["model"]
        style_id, style_reference_urls = resolve_v4_style(model_id, style_id, style_references)
        has_style = bool(style_id or style_reference_urls)
        if has_style:
            model_id = RECRAFT_V4_VECTOR_MODEL_FOR_STYLE.get(model_id, model_id)
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/recraft/image_generation", method="POST"),
            response_model=RecraftImageGenerationResponse,
            data=RecraftImageGenerationRequest(
                prompt=prompt,
                model=model_id,
                size=model["size"],
                n=n,
                style=None if has_style or model_id.endswith("_vector") else "vector_illustration",
                substyle=None,
                style_id=style_id,
                style_match=style_match if has_style else None,
                style_reference_urls=style_reference_urls,
                controls=recraft_controls.create_api_model() if recraft_controls else None,
            ),
            max_retries=1,
        )
        svg_data = []
        for data in response.data:
            svg_data.append(await download_url_as_bytesio(data.url, timeout=1024))
        return IO.NodeOutput(SVG(svg_data), response.style_id or "")


class RecraftExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            RecraftTextToImageNode,
            RecraftImageToImageNode,
            RecraftImageInpaintingNode,
            RecraftTextToVectorNode,
            RecraftVectorizeImageNode,
            RecraftRemoveBackgroundNode,
            RecraftReplaceBackgroundNode,
            RecraftCrispUpscaleNode,
            RecraftCreativeUpscaleNode,
            RecraftStyleV3RealisticImageNode,
            RecraftStyleV3DigitalIllustrationNode,
            RecraftStyleV3LogoRasterNode,
            RecraftStyleInfiniteStyleLibrary,
            RecraftCreateStyleNode,
            RecraftColorRGBNode,
            RecraftControlsNode,
            RecraftV4TextToImageNode,
            RecraftV4TextToVectorNode,
            RecraftV4CreateStyleNode,
        ]


async def comfy_entrypoint() -> RecraftExtension:
    return RecraftExtension()
