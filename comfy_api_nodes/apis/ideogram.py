from enum import Enum
from typing import Optional, List, Dict, Any, Union
from datetime import datetime

from pydantic import BaseModel, Field, RootModel, StrictBytes


class IdeogramColorPalette1(BaseModel):
    name: str = Field(..., description='Name of the preset color palette')


class Member(BaseModel):
    color: Optional[str] = Field(
        None, description='Hexadecimal color code', pattern='^#[0-9A-Fa-f]{6}$'
    )
    weight: Optional[float] = Field(
        None, description='Optional weight for the color (0-1)', ge=0.0, le=1.0
    )


class IdeogramColorPalette2(BaseModel):
    members: List[Member] = Field(
        ..., description='Array of color definitions with optional weights'
    )


class IdeogramColorPalette(
    RootModel[Union[IdeogramColorPalette1, IdeogramColorPalette2]]
):
    root: Union[IdeogramColorPalette1, IdeogramColorPalette2] = Field(
        ...,
        description='A color palette specification that can either use a preset name or explicit color definitions with weights',
    )


class ImageRequest(BaseModel):
    aspect_ratio: Optional[str] = Field(
        None,
        description="Optional. The aspect ratio (e.g., 'ASPECT_16_9', 'ASPECT_1_1'). Cannot be used with resolution. Defaults to 'ASPECT_1_1' if unspecified.",
    )
    color_palette: Optional[Dict[str, Any]] = Field(
        None, description='Optional. Color palette object. Only for V_2, V_2_TURBO.'
    )
    magic_prompt_option: Optional[str] = Field(
        None, description="Optional. MagicPrompt usage ('AUTO', 'ON', 'OFF')."
    )
    model: str = Field(..., description="The model used (e.g., 'V_2', 'V_2A_TURBO')")
    negative_prompt: Optional[str] = Field(
        None,
        description='Optional. Description of what to exclude. Only for V_1, V_1_TURBO, V_2, V_2_TURBO.',
    )
    num_images: Optional[int] = Field(
        1,
        description='Optional. Number of images to generate (1-8). Defaults to 1.',
        ge=1,
        le=8,
    )
    prompt: str = Field(
        ..., description='Required. The prompt to use to generate the image.'
    )
    resolution: Optional[str] = Field(
        None,
        description="Optional. Resolution (e.g., 'RESOLUTION_1024_1024'). Only for model V_2. Cannot be used with aspect_ratio.",
    )
    seed: Optional[int] = Field(
        None,
        description='Optional. A number between 0 and 2147483647.',
        ge=0,
        le=2147483647,
    )
    style_type: Optional[str] = Field(
        None,
        description="Optional. Style type ('AUTO', 'GENERAL', 'REALISTIC', 'DESIGN', 'RENDER_3D', 'ANIME'). Only for models V_2 and above.",
    )


class IdeogramGenerateRequest(BaseModel):
    image_request: ImageRequest = Field(
        ..., description='The image generation request parameters.'
    )


class Datum(BaseModel):
    is_image_safe: Optional[bool] = Field(
        None, description='Indicates whether the image is considered safe.'
    )
    prompt: Optional[str] = Field(
        None, description='The prompt used to generate this image.'
    )
    resolution: Optional[str] = Field(
        None, description="The resolution of the generated image (e.g., '1024x1024')."
    )
    seed: Optional[int] = Field(
        None, description='The seed value used for this generation.'
    )
    style_type: Optional[str] = Field(
        None,
        description="The style type used for generation (e.g., 'REALISTIC', 'ANIME').",
    )
    url: Optional[str] = Field(None, description='URL to the generated image.')


class IdeogramGenerateResponse(BaseModel):
    created: Optional[datetime] = Field(
        None, description='Timestamp when the generation was created.'
    )
    data: Optional[List[Datum]] = Field(
        None, description='Array of generated image information.'
    )


class StyleCode(RootModel[str]):
    root: str = Field(..., pattern='^[0-9A-Fa-f]{8}$')


class Datum1(BaseModel):
    is_image_safe: Optional[bool] = None
    prompt: Optional[str] = None
    resolution: Optional[str] = None
    seed: Optional[int] = None
    style_type: Optional[str] = None
    url: Optional[str] = None


class IdeogramV3IdeogramResponse(BaseModel):
    created: Optional[datetime] = None
    data: Optional[List[Datum1]] = None


class RenderingSpeed1(str, Enum):
    TURBO = 'TURBO'
    DEFAULT = 'DEFAULT'
    QUALITY = 'QUALITY'


class IdeogramV3ReframeRequest(BaseModel):
    color_palette: Optional[Dict[str, Any]] = None
    image: Optional[StrictBytes] = None
    num_images: Optional[int] = Field(None, ge=1, le=8)
    rendering_speed: Optional[RenderingSpeed1] = None
    resolution: str
    seed: Optional[int] = Field(None, ge=0, le=2147483647)
    style_codes: Optional[List[str]] = None
    style_reference_images: Optional[List[StrictBytes]] = None


class MagicPrompt(str, Enum):
    AUTO = 'AUTO'
    ON = 'ON'
    OFF = 'OFF'


class StyleType(str, Enum):
    AUTO = 'AUTO'
    GENERAL = 'GENERAL'
    REALISTIC = 'REALISTIC'
    DESIGN = 'DESIGN'


class IdeogramV3RemixRequest(BaseModel):
    aspect_ratio: Optional[str] = None
    color_palette: Optional[Dict[str, Any]] = None
    image: Optional[StrictBytes] = None
    image_weight: Optional[int] = Field(50, ge=1, le=100)
    magic_prompt: Optional[MagicPrompt] = None
    negative_prompt: Optional[str] = None
    num_images: Optional[int] = Field(None, ge=1, le=8)
    prompt: str
    rendering_speed: Optional[RenderingSpeed1] = None
    resolution: Optional[str] = None
    seed: Optional[int] = Field(None, ge=0, le=2147483647)
    style_codes: Optional[List[str]] = None
    style_reference_images: Optional[List[StrictBytes]] = None
    style_type: Optional[StyleType] = None


class IdeogramV3ReplaceBackgroundRequest(BaseModel):
    color_palette: Optional[Dict[str, Any]] = None
    image: Optional[StrictBytes] = None
    magic_prompt: Optional[MagicPrompt] = None
    num_images: Optional[int] = Field(None, ge=1, le=8)
    prompt: str
    rendering_speed: Optional[RenderingSpeed1] = None
    seed: Optional[int] = Field(None, ge=0, le=2147483647)
    style_codes: Optional[List[str]] = None
    style_reference_images: Optional[List[StrictBytes]] = None


class ColorPalette(BaseModel):
    name: str = Field(..., description='Name of the color palette', examples=['PASTEL'])


class MagicPrompt2(str, Enum):
    ON = 'ON'
    OFF = 'OFF'


class StyleType1(str, Enum):
    AUTO = 'AUTO'
    GENERAL = 'GENERAL'
    REALISTIC = 'REALISTIC'
    DESIGN = 'DESIGN'
    FICTION = 'FICTION'


class RenderingSpeed(str, Enum):
    DEFAULT = 'DEFAULT'
    TURBO = 'TURBO'
    QUALITY = 'QUALITY'


class IdeogramV3EditRequest(BaseModel):
    color_palette: Optional[IdeogramColorPalette] = None
    image: Optional[StrictBytes] = Field(
        None,
        description='The image being edited (max size 10MB); only JPEG, WebP and PNG formats are supported at this time.',
    )
    magic_prompt: Optional[str] = Field(
        None,
        description='Determine if MagicPrompt should be used in generating the request or not.',
    )
    mask: Optional[StrictBytes] = Field(
        None,
        description='A black and white image of the same size as the image being edited (max size 10MB). Black regions in the mask should match up with the regions of the image that you would like to edit; only JPEG, WebP and PNG formats are supported at this time.',
    )
    num_images: Optional[int] = Field(
        None, description='The number of images to generate.'
    )
    prompt: str = Field(
        ..., description='The prompt used to describe the edited result.'
    )
    rendering_speed: RenderingSpeed
    seed: Optional[int] = Field(
        None, description='Random seed. Set for reproducible generation.'
    )
    style_codes: Optional[List[StyleCode]] = Field(
        None,
        description='A list of 8 character hexadecimal codes representing the style of the image. Cannot be used in conjunction with style_reference_images or style_type.',
    )
    style_reference_images: Optional[List[StrictBytes]] = Field(
        None,
        description='A set of images to use as style references (maximum total size 10MB across all style references). The images should be in JPEG, PNG or WebP format.',
    )
    character_reference_images: Optional[List[str]] = Field(
        None,
        description='Generations with character reference are subject to the character reference pricing. A set of images to use as character references (maximum total size 10MB across all character references), currently only supports 1 character reference image. The images should be in JPEG, PNG or WebP format.'
    )
    character_reference_images_mask: Optional[List[str]] = Field(
        None,
        description='Optional masks for character reference images. When provided, must match the number of character_reference_images. Each mask should be a grayscale image of the same dimensions as the corresponding character reference image. The images should be in JPEG, PNG or WebP format.'
    )


class IdeogramV3Request(BaseModel):
    aspect_ratio: Optional[str] = Field(
        None, description='Aspect ratio in format WxH', examples=['1x3']
    )
    color_palette: Optional[ColorPalette] = None
    magic_prompt: Optional[MagicPrompt2] = Field(
        None, description='Whether to enable magic prompt enhancement'
    )
    negative_prompt: Optional[str] = Field(
        None, description='Text prompt specifying what to avoid in the generation'
    )
    num_images: Optional[int] = Field(
        None, description='Number of images to generate', ge=1
    )
    prompt: str = Field(..., description='The text prompt for image generation')
    rendering_speed: RenderingSpeed
    resolution: Optional[str] = Field(
        None, description='Image resolution in format WxH', examples=['1280x800']
    )
    seed: Optional[int] = Field(
        None, description='Seed value for reproducible generation'
    )
    style_codes: Optional[List[StyleCode]] = Field(
        None, description='Array of style codes in hexadecimal format'
    )
    style_reference_images: Optional[List[str]] = Field(
        None, description='Array of reference image URLs or identifiers'
    )
    style_type: Optional[StyleType1] = Field(
        None, description='The type of style to apply'
    )
    character_reference_images: Optional[List[str]] = Field(
        None,
        description='Generations with character reference are subject to the character reference pricing. A set of images to use as character references (maximum total size 10MB across all character references), currently only supports 1 character reference image. The images should be in JPEG, PNG or WebP format.'
    )
    character_reference_images_mask: Optional[List[str]] = Field(
        None,
        description='Optional masks for character reference images. When provided, must match the number of character_reference_images. Each mask should be a grayscale image of the same dimensions as the corresponding character reference image. The images should be in JPEG, PNG or WebP format.'
    )


class IdeogramV4Request(BaseModel):
    text_prompt: str | None = Field(
        None,
        description="Natural-language prompt; Magic Prompt is applied automatically. "
        "Supply exactly one of text_prompt or json_prompt.",
    )
    json_prompt: dict[str, Any] | None = Field(
        None,
        description="Structured V4 prompt object consumed directly (disables Magic Prompt). "
        "Supply exactly one of text_prompt or json_prompt.",
    )
    resolution: str | None = Field(None, description="Output resolution in WIDTHxHEIGHT (e.g. '2048x2048').")
    rendering_speed: str | None = Field(None, description="Rendering speed: 'TURBO', 'DEFAULT', or 'QUALITY'.")
    enable_copyright_detection: bool | None = Field(None, description="Opt into post-generation copyright detection.")
