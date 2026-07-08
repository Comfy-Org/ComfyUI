"""Pydantic models for the Krea image-generation API."""

from pydantic import BaseModel, Field


class KreaMoodboard(BaseModel):
    id: str = Field(...)
    strength: float = Field(default=0.35, ge=-0.5, le=1.5)


class KreaImageStyleReference(BaseModel):
    strength: float = Field(..., ge=-2.0, le=2.0)
    url: str | None = Field(default=None)


class KreaGenerateImageRequest(BaseModel):
    prompt: str = Field(...)
    aspect_ratio: str = Field(...)
    resolution: str = Field(...)
    seed: int | None = Field(default=None)
    creativity: str = Field(default="medium")
    moodboards: list[KreaMoodboard] | None = Field(default=None)
    image_style_references: list[KreaImageStyleReference] | None = Field(default=None)


class KreaJobResult(BaseModel):
    urls: list[str] | None = Field(default=None)
    style_id: str | None = Field(default=None)


class KreaJob(BaseModel):
    job_id: str = Field(...)
    status: str = Field(...)
    created_at: str = Field(...)
    completed_at: str | None = Field(default=None)
    result: KreaJobResult | None = Field(default=None)


class KreaAssetResponse(BaseModel):
    id: str = Field(...)
    image_url: str = Field(...)
    uploaded_at: str = Field(...)
    width: float | None = Field(default=None)
    height: float | None = Field(default=None)
    size_bytes: float | None = Field(default=None)
    mime_type: str | None = Field(default=None)
