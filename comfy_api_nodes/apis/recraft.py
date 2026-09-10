from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RecraftColor:
    def __init__(self, r: int, g: int, b: int):
        self.color = [r, g, b]

    def create_api_model(self):
        return RecraftColorObject(rgb=self.color)


class RecraftColorChain:
    def __init__(self):
        self.colors: list[RecraftColor] = []

    def get_first(self):
        if len(self.colors) > 0:
            return self.colors[0]
        return None

    def add(self, color: RecraftColor):
        self.colors.append(color)

    def create_api_model(self):
        if not self.colors:
            return None
        colors_api = [x.create_api_model() for x in self.colors]
        return colors_api

    def clone(self):
        c = RecraftColorChain()
        for color in self.colors:
            c.add(color)
        return c

    def clone_and_merge(self, other: RecraftColorChain):
        c = self.clone()
        for color in other.colors:
            c.add(color)
        return c


class RecraftControls:
    def __init__(self, colors: RecraftColorChain=None, background_color: RecraftColorChain=None,
                 artistic_level: int=None, no_text: bool=None):
        self.colors = colors
        self.background_color = background_color
        self.artistic_level = artistic_level
        self.no_text = no_text

    def create_api_model(self):
        if self.colors is None and self.background_color is None and self.artistic_level is None and self.no_text is None:
            return None
        colors_api = None
        background_color_api = None
        if self.colors:
            colors_api = self.colors.create_api_model()
        if self.background_color:
            first_background = self.background_color.get_first()
            background_color_api = first_background.create_api_model() if first_background else None

        return RecraftControlsObject(colors=colors_api, background_color=background_color_api,
                                             artistic_level=self.artistic_level, no_text=self.no_text)


class RecraftStyle:
    def __init__(self, style: str=None, substyle: str=None, style_id: str=None):
        self.style = style
        if substyle == "None":
            substyle = None
        self.substyle = substyle
        self.style_id = style_id


class RecraftIO:
    STYLEV3 = "RECRAFT_V3_STYLE"
    COLOR = "RECRAFT_COLOR"
    CONTROLS = "RECRAFT_CONTROLS"


class RecraftStyleV3(str, Enum):
    #any = 'any' NOTE: this does not work for some reason... why?
    realistic_image = 'realistic_image'
    digital_illustration = 'digital_illustration'
    vector_illustration = 'vector_illustration'
    logo_raster = 'logo_raster'


def get_v3_substyles(style_v3: str, include_none=True) -> list[str]:
    substyles: list[str] = []
    if include_none:
        substyles.append("None")
    return substyles + dict_recraft_substyles_v3.get(style_v3, [])


dict_recraft_substyles_v3 = {
    RecraftStyleV3.realistic_image: [
        "b_and_w",
        "enterprise",
        "evening_light",
        "faded_nostalgia",
        "forest_life",
        "hard_flash",
        "hdr",
        "motion_blur",
        "mystic_naturalism",
        "natural_light",
        "natural_tones",
        "organic_calm",
        "real_life_glow",
        "retro_realism",
        "retro_snapshot",
        "studio_portrait",
        "urban_drama",
        "village_realism",
        "warm_folk"
    ],
    RecraftStyleV3.digital_illustration: [
        "2d_art_poster",
        "2d_art_poster_2",
        "antiquarian",
        "bold_fantasy",
        "child_book",
        "child_books",
        "cover",
        "crosshatch",
        "digital_engraving",
        "engraving_color",
        "expressionism",
        "freehand_details",
        "grain",
        "grain_20",
        "graphic_intensity",
        "hand_drawn",
        "hand_drawn_outline",
        "handmade_3d",
        "hard_comics",
        "infantile_sketch",
        "long_shadow",
        "modern_folk",
        "multicolor",
        "neon_calm",
        "noir",
        "nostalgic_pastel",
        "outline_details",
        "pastel_gradient",
        "pastel_sketch",
        "pixel_art",
        "plastic",
        "pop_art",
        "pop_renaissance",
        "seamless",
        "street_art",
        "tablet_sketch",
        "urban_glow",
        "urban_sketching",
        "vanilla_dreams",
        "young_adult_book",
        "young_adult_book_2"
    ],
    RecraftStyleV3.vector_illustration: [
        "bold_stroke",
        "chemistry",
        "colored_stencil",
        "contour_pop_art",
        "cosmics",
        "cutout",
        "depressive",
        "editorial",
        "emotional_flat",
        "engraving",
        "infographical",
        "line_art",
        "line_circuit",
        "linocut",
        "marker_outline",
        "mosaic",
        "naivector",
        "roundish_flat",
        "seamless",
        "segmented_colors",
        "sharp_contrast",
        "thin",
        "vector_photo",
        "vivid_shapes"
    ],
    RecraftStyleV3.logo_raster: [
        "emblem_graffiti",
        "emblem_pop_art",
        "emblem_punk",
        "emblem_stamp",
        "emblem_vintage"
    ],
}


class RecraftImageSize(str, Enum):
    res_1024x1024 = '1024x1024'
    res_1365x1024 = '1365x1024'
    res_1024x1365 = '1024x1365'
    res_1536x1024 = '1536x1024'
    res_1024x1536 = '1024x1536'
    res_1820x1024 = '1820x1024'
    res_1024x1820 = '1024x1820'
    res_1024x2048 = '1024x2048'
    res_2048x1024 = '2048x1024'
    res_1434x1024 = '1434x1024'
    res_1024x1434 = '1024x1434'
    res_1024x1280 = '1024x1280'
    res_1280x1024 = '1280x1024'
    res_1024x1707 = '1024x1707'
    res_1707x1024 = '1707x1024'


RECRAFT_V4_SIZES = [
    "1024x1024",
    "1536x768",
    "768x1536",
    "1280x832",
    "832x1280",
    "1216x896",
    "896x1216",
    "1152x896",
    "896x1152",
    "832x1344",
    "1280x896",
    "896x1280",
    "1344x768",
    "768x1344",
]

RECRAFT_V4_PRO_SIZES = [
    "2048x2048",
    "3072x1536",
    "1536x3072",
    "2560x1664",
    "1664x2560",
    "2432x1792",
    "1792x2432",
    "2304x1792",
    "1792x2304",
    "1664x2688",
    "2560x1792",
    "1792x2560",
    "2688x1536",
    "1536x2688",
]

RECRAFT_V4_STYLES_MODELS = frozenset(
    {
        "recraftv4_styles",
        "recraftv4_styles_vector",
        "recraftv4_styles_pro",
        "recraftv4_styles_pro_vector",
    }
)

RECRAFT_V4_VECTOR_MODEL_FOR_STYLE = {
    "recraftv4": "recraftv4_vector",
    "recraftv4_pro": "recraftv4_pro_vector",
}

RECRAFT_STYLE_MATCH_OPTIONS = ["precise", "flexible"]

RECRAFT_STYLE_REFERENCES_MAX = 10

RECRAFT_STYLE_REFERENCES_MAX_BYTES = 10 * 1024 * 1024


class RecraftColorObject(BaseModel):
    rgb: list[int] = Field(..., description='An array of 3 integer values in range of 0...255 defining RGB Color Model')


class RecraftControlsObject(BaseModel):
    colors: list[RecraftColorObject] | None = Field(None, description='An array of preferable colors')
    background_color: RecraftColorObject | None = Field(None, description='Use given color as a desired background color')
    no_text: bool | None = Field(None, description='Do not embed text layouts')
    artistic_level: int | None = Field(None, description='Defines artistic tone of your image. At a simple level, the person looks straight at the camera in a static and clean style. Dynamic and eccentric levels introduce movement and creativity. The value should be in range [0..5].')


class RecraftImageGenerationRequest(BaseModel):
    prompt: str = Field(..., description='The text prompt describing the image to generate')
    size: str | None = Field(None, description='The size of the generated image (e.g., "1024x1024")')
    n: int = Field(..., description='The number of images to generate')
    negative_prompt: str | None = Field(None, description='A text description of undesired elements on an image')
    model: str = Field(...)
    style: str | None = Field(None, description='The style to apply to the generated image (e.g., "digital_illustration")')
    substyle: str | None = Field(None, description='The substyle to apply to the generated image, depending on the style input')
    controls: RecraftControlsObject | None = Field(None, description='A set of custom parameters to tweak generation process')
    style_id: str | None = Field(None, description='Use a previously uploaded style as a reference; UUID')
    style_match: str | None = Field(None, description='How closely to follow the referenced style: "precise" or "flexible" for V4 models')
    style_reference_urls: list[str] | None = Field(None, description='URLs or data URLs of style reference images; a private style is created from them and returned as style_id')
    strength: float | None = Field(None, description='Defines the difference with the original image, should lie in [0, 1], where 0 means almost identical, and 1 means miserable similarity')
    random_seed: int | None = Field(None, description="Seed for video generation")


class RecraftReturnedObject(BaseModel):
    image_id: str = Field(..., description='Unique identifier for the generated image')
    url: str = Field(..., description='URL to access the generated image')


class RecraftImageGenerationResponse(BaseModel):
    created: int = Field(..., description='Unix timestamp when the generation was created')
    credits: int = Field(..., description='Number of credits used for the generation')
    data: list[RecraftReturnedObject] | None = Field(None, description='Array of generated image information')
    image: RecraftReturnedObject | None = Field(None, description='Single generated image')
    style_id: str | None = Field(None, description='The style applied to the generation, including one auto-created from style references')


class RecraftCreateStyleRequest(BaseModel):
    style: str = Field(..., description="any, realistic_image, digital_illustration, vector_illustration, or icon")
    model: str | None = Field(None, description="The model family the style is created for, e.g. recraftv4_styles")


class RecraftCreateStyleResponse(BaseModel):
    id: str = Field(..., description="UUID of the created style")
