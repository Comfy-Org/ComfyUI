from pydantic import BaseModel, Field


class CrystalUpscalerRequest(BaseModel):
    image_url: str = Field(...)
    scale_factor: float = Field(..., ge=1, le=200)
    creativity: int = Field(..., ge=0, le=10)
    output_format: str = Field(...)


class CrystalUpscalerSubmitResponse(BaseModel):
    request_id: str = Field(...)


class CrystalUpscalerStatusResponse(BaseModel):
    status: str | None = Field(None)


class CrystalUpscalerImage(BaseModel):
    url: str = Field(...)


class CrystalUpscalerResult(BaseModel):
    images: list[CrystalUpscalerImage] = Field(...)
