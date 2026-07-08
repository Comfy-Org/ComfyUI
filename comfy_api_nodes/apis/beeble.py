from pydantic import BaseModel, Field


class CreateSwitchXRequest(BaseModel):
    generation_type: str = Field(...)
    source_uri: str = Field(...)
    alpha_mode: str = Field(...)
    prompt: str | None = Field(None, max_length=2000)
    reference_image_uri: str | None = Field(None)
    alpha_uri: str | None = Field(None)
    max_resolution: int = Field(1080)
    callback_url: str | None = Field(None)
    idempotency_key: str | None = Field(None, max_length=256, min_length=1)


class SwitchXOutputUrls(BaseModel):
    render: str | None = Field(None)
    source: str | None = Field(None)
    alpha: str | None = Field(None)


class SwitchXStatusResponse(BaseModel):
    id: str = Field(...)
    status: str = Field(...)
    progress: int | None = Field(None)
    generation_type: str | None = Field(None)
    alpha_mode: str | None = Field(None)
    output: SwitchXOutputUrls | None = Field(None)
    error: str | None = Field(None)
    created_at: str | None = Field(None)
    modified_at: str | None = Field(None)
    completed_at: str | None = Field(None)
