from typing import TypedDict

from pydantic import BaseModel, Field, model_validator


class InputGenerateType(TypedDict):
    generate_type: str
    polygon_type: str
    pbr: bool


class Hunyuan3D36GenerateType(TypedDict):
    generate_type: str
    texture_resolution: str
    pbr: bool
    bake_lighting: bool


class Hunyuan3D36ViewImage(BaseModel):
    view_type: str = Field(...)
    view_image_url: str = Field(...)


class Hunyuan3D36Request(BaseModel):
    prompt: str | None = Field(None)
    image_url: str | None = Field(None)
    multi_view_images: list[Hunyuan3D36ViewImage] | None = Field(None)
    style: str | None = Field(None)
    generate_type: str = Field(...)
    face_count: int = Field(...)
    tex_res: str | None = Field(None)
    enable_pbr: bool | None = Field(None)
    bake_lighting: bool | None = Field(None)


class Hunyuan3D36ResultFile(BaseModel):
    type: str = Field(...)
    url: str = Field(...)


class Hunyuan3D36Result(BaseModel):
    request_id: str = Field("")
    data: list[Hunyuan3D36ResultFile] = Field([])


class Hunyuan3DViewImage(BaseModel):
    ViewType: str = Field(..., description="Valid values: back, left, right.")
    ViewImageUrl: str = Field(...)


class To3DProTaskRequest(BaseModel):
    Model: str = Field(...)
    Prompt: str | None = Field(None)
    ImageUrl: str | None = Field(None)
    MultiViewImages: list[Hunyuan3DViewImage] | None = Field(None)
    EnablePBR: bool | None = Field(...)
    FaceCount: int | None = Field(...)
    GenerateType: str | None = Field(...)
    PolygonType: str | None = Field(...)


class RequestError(BaseModel):
    Code: str = Field("")
    Message: str = Field("")


class To3DProTaskCreateResponse(BaseModel):
    JobId: str | None = Field(None)
    Error: RequestError | None = Field(None)

    @model_validator(mode="before")
    @classmethod
    def unwrap_data(cls, values: dict) -> dict:
        if "Response" in values and isinstance(values["Response"], dict):
            return values["Response"]
        return values


class ResultFile3D(BaseModel):
    Type: str = Field(...)
    Url: str = Field(...)
    PreviewImageUrl: str = Field("")


class To3DProTaskResultResponse(BaseModel):
    ErrorCode: str = Field("")
    ErrorMessage: str = Field("")
    ResultFile3Ds: list[ResultFile3D] = Field([])
    Status: str = Field(...)

    @model_validator(mode="before")
    @classmethod
    def unwrap_data(cls, values: dict) -> dict:
        if "Response" in values and isinstance(values["Response"], dict):
            return values["Response"]
        return values


class To3DProTaskQueryRequest(BaseModel):
    JobId: str = Field(...)


class TaskFile3DInput(BaseModel):
    Type: str = Field(..., description="File type: GLB, OBJ, or FBX")
    Url: str = Field(...)


class To3DUVTaskRequest(BaseModel):
    File: TaskFile3DInput = Field(...)


class To3DPartTaskRequest(BaseModel):
    File: TaskFile3DInput = Field(...)
    EnableStagedGeneration: bool | None = Field(None)


class TextureEditImageInfo(BaseModel):
    Url: str = Field(...)


class TextureEditTaskRequest(BaseModel):
    File3D: TaskFile3DInput = Field(...)
    Image: TextureEditImageInfo | None = Field(None)
    Prompt: str | None = Field(None)
    EnablePBR: bool | None = Field(None)


class SmartTopologyRequest(BaseModel):
    File3D: TaskFile3DInput = Field(...)
    PolygonType: str | None = Field(...)
    FaceLevel: str | None = Field(...)
