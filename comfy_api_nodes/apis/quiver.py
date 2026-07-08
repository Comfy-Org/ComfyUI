from pydantic import BaseModel, Field


class QuiverImageObject(BaseModel):
    url: str = Field(...)


class QuiverTextToSVGRequest(BaseModel):
    model: str = Field(default="arrow-preview")
    prompt: str = Field(...)
    instructions: str | None = Field(default=None)
    references: list[QuiverImageObject] | None = Field(default=None, max_length=4)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)


class QuiverImageToSVGRequest(BaseModel):
    model: str = Field(default="arrow-preview")
    image: QuiverImageObject = Field(...)
    auto_crop: bool | None = Field(default=None)
    target_size: int | None = Field(default=None, ge=128, le=4096)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)


class QuiverSVGResponseItem(BaseModel):
    svg: str = Field(...)
    mime_type: str | None = Field(default="image/svg+xml")


class QuiverSVGUsage(BaseModel):
    total_tokens: int | None = Field(default=None)
    input_tokens: int | None = Field(default=None)
    output_tokens: int | None = Field(default=None)


class QuiverSVGResponse(BaseModel):
    id: str | None = Field(default=None)
    created: int | None = Field(default=None)
    data: list[QuiverSVGResponseItem] = Field(...)
    usage: QuiverSVGUsage | None = Field(default=None)
