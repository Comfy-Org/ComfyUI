from pydantic import BaseModel, Field


class PatinaSubmitResponse(BaseModel):
    request_id: str = Field(...)
    status: str | None = Field(None)


class PatinaStatusResponse(BaseModel):
    status: str | None = Field(None)


class PatinaImage(BaseModel):
    url: str = Field(...)
    map_type: str | None = Field(None, description="PBR map type; None for a base texture image.")
    width: int | None = Field(None)
    height: int | None = Field(None)
    content_type: str | None = Field(None)


class PatinaResult(BaseModel):
    images: list[PatinaImage] = Field(default_factory=list)
    seed: int | None = Field(None)
    prompt: str | None = Field(None)


class ImageSize(BaseModel):
    width: int = Field(...)
    height: int = Field(...)


class PatinaPBRMapsRequest(BaseModel):
    image_url: str = Field(...)
    maps: list[str] | None = Field(None)
    seed: int | None = Field(None)
    output_format: str = Field("png")
    enable_safety_checker: bool = Field(False)


class PatinaMaterialRequest(BaseModel):
    prompt: str = Field(...)
    image_size: str | ImageSize = Field("square_hd")
    maps: list[str] | None = Field(None)
    upscale_factor: int = Field(0, description="0, 2, or 4 - SeedVR upscaling of the PBR maps.")
    tiling_mode: str = Field("both")
    num_inference_steps: int = Field(8)
    enable_prompt_expansion: bool = Field(False)
    enable_safety_checker: bool = Field(False)
    tile_size: int = Field(128, description="Tile size in latent space (64 = 512px, 128 = 1024px).")
    tile_stride: int = Field(64, description="Tile stride in latent space.")
    image_url: str | None = Field(
        None, description="Optional source for img2img, or inpaint when combined with mask_url."
    )
    mask_url: str | None = Field(
        None, description="Inpaint mask (white = regenerate, black = keep); requires image_url."
    )
    strength: float = Field(0.6)
    seed: int | None = Field(None)
    output_format: str = Field("png")


class PatinaExtractRequest(BaseModel):
    prompt: str = Field(...)
    image_url: str = Field(...)
    image_size: str | ImageSize = Field("square_hd")
    maps: list[str] | None = Field(None)
    upscale_factor: int = Field(0, description="0, 2, or 4 - SeedVR upscaling of the PBR maps.")
    tiling_mode: str = Field("both")
    num_inference_steps: int = Field(8)
    enable_prompt_expansion: bool = Field(False)
    enable_safety_checker: bool = Field(False)
    tile_size: int = Field(128, description="Tile size in latent space (64 = 512px, 128 = 1024px).")
    tile_stride: int = Field(64, description="Tile stride in latent space.")
    strength: float = Field(0.6)
    seed: int | None = Field(None)
    output_format: str = Field("png")
