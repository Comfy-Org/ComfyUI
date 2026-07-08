from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, confloat


class StabilityFormat(str, Enum):
    png = 'png'
    jpeg = 'jpeg'
    webp = 'webp'


class StabilityAspectRatio(str, Enum):
    ratio_1_1 = "1:1"
    ratio_16_9 = "16:9"
    ratio_9_16 = "9:16"
    ratio_3_2 = "3:2"
    ratio_2_3 = "2:3"
    ratio_5_4 = "5:4"
    ratio_4_5 = "4:5"
    ratio_21_9 = "21:9"
    ratio_9_21 = "9:21"


def get_stability_style_presets(include_none=True):
    presets = []
    if include_none:
        presets.append("None")
    return presets + [x.value for x in StabilityStylePreset]


class StabilityStylePreset(str, Enum):
    _3d_model = "3d-model"
    analog_film = "analog-film"
    anime = "anime"
    cinematic = "cinematic"
    comic_book = "comic-book"
    digital_art = "digital-art"
    enhance = "enhance"
    fantasy_art = "fantasy-art"
    isometric = "isometric"
    line_art = "line-art"
    low_poly = "low-poly"
    modeling_compound = "modeling-compound"
    neon_punk = "neon-punk"
    origami = "origami"
    photographic = "photographic"
    pixel_art = "pixel-art"
    tile_texture = "tile-texture"


class Stability_SD3_5_Model(str, Enum):
    sd3_5_large = "sd3.5-large"
    # sd3_5_large_turbo = "sd3.5-large-turbo"
    sd3_5_medium = "sd3.5-medium"


class Stability_SD3_5_GenerationMode(str, Enum):
    text_to_image = "text-to-image"
    image_to_image = "image-to-image"


class StabilityStable3_5Request(BaseModel):
    model: str = Field(...)
    mode: str = Field(...)
    prompt: str = Field(...)
    negative_prompt: Optional[str] = Field(None)
    aspect_ratio: Optional[str] = Field(None)
    seed: Optional[int] = Field(None)
    output_format: Optional[str] = Field(StabilityFormat.png.value)
    image: Optional[str] = Field(None)
    style_preset: Optional[str] = Field(None)
    cfg_scale: float = Field(...)
    strength: Optional[confloat(ge=0.0, le=1.0)] = Field(None)


class StabilityUpscaleConservativeRequest(BaseModel):
    prompt: str = Field(...)
    negative_prompt: Optional[str] = Field(None)
    seed: Optional[int] = Field(None)
    output_format: Optional[str] = Field(StabilityFormat.png.value)
    image: Optional[str] = Field(None)
    creativity: Optional[confloat(ge=0.2, le=0.5)] = Field(None)


class StabilityUpscaleCreativeRequest(BaseModel):
    prompt: str = Field(...)
    negative_prompt: Optional[str] = Field(None)
    seed: Optional[int] = Field(None)
    output_format: Optional[str] = Field(StabilityFormat.png.value)
    image: Optional[str] = Field(None)
    creativity: Optional[confloat(ge=0.1, le=0.5)] = Field(None)
    style_preset: Optional[str] = Field(None)


class StabilityStableUltraRequest(BaseModel):
    prompt: str = Field(...)
    negative_prompt: Optional[str] = Field(None)
    aspect_ratio: Optional[str] = Field(None)
    seed: Optional[int] = Field(None)
    output_format: Optional[str] = Field(StabilityFormat.png.value)
    image: Optional[str] = Field(None)
    style_preset: Optional[str] = Field(None)
    strength: Optional[confloat(ge=0.0, le=1.0)] = Field(None)


class StabilityStableUltraResponse(BaseModel):
    image: Optional[str] = Field(None)
    finish_reason: Optional[str] = Field(None)
    seed: Optional[int] = Field(None)


class StabilityResultsGetResponse(BaseModel):
    image: Optional[str] = Field(None)
    finish_reason: Optional[str] = Field(None)
    seed: Optional[int] = Field(None)
    id: Optional[str] = Field(None)
    name: Optional[str] = Field(None)
    errors: Optional[list[str]] = Field(None)
    status: Optional[str] = Field(None)
    result: Optional[str] = Field(None)


class StabilityAsyncResponse(BaseModel):
    id: Optional[str] = Field(None)


class StabilityTextToAudioRequest(BaseModel):
    model: str = Field(...)
    prompt: str = Field(...)
    duration: int = Field(190, ge=1, le=190)
    seed: int = Field(0, ge=0, le=4294967294)
    steps: int = Field(8, ge=4, le=8)
    output_format: str = Field("wav")


class StabilityAudioToAudioRequest(StabilityTextToAudioRequest):
    strength: float = Field(0.01, ge=0.01, le=1.0)


class StabilityAudioInpaintRequest(StabilityTextToAudioRequest):
    mask_start: int = Field(30, ge=0, le=190)
    mask_end: int = Field(190, ge=0, le=190)


class StabilityAudioResponse(BaseModel):
    audio: Optional[str] = Field(None)
