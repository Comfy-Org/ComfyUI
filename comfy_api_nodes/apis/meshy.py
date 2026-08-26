from typing import TypedDict

from pydantic import BaseModel, Field

from comfy_api.latest import Input


class InputShouldRemesh(TypedDict):
    should_remesh: str
    topology: str
    target_polycount: int


class InputShouldTexture(TypedDict):
    should_texture: str
    enable_pbr: bool
    texture_resolution: str
    texture_prompt: str
    texture_image: Input.Image | None


class MeshyTaskResponse(BaseModel):
    result: str = Field(...)


class MeshyTextToModelRequest(BaseModel):
    mode: str = Field("preview")
    prompt: str = Field(..., max_length=600)
    art_style: str = Field(...)
    ai_model: str = Field(...)
    topology: str | None = Field(..., description="'quad' or 'triangle'")
    target_polycount: int | None = Field(..., ge=100, le=300000)
    should_remesh: bool = Field(
        True,
        description="False returns the original mesh, ignoring topology and polycount.",
    )
    symmetry_mode: str = Field(..., description="'auto', 'off' or 'on'")
    pose_mode: str = Field(...)
    ultra_mode: bool = Field(False)
    seed: int = Field(...)
    moderation: bool = Field(False)


class MeshyRefineTask(BaseModel):
    mode: str = Field("refine")
    preview_task_id: str = Field(...)
    enable_pbr: bool | None = Field(...)
    texture_resolution: str = Field(...)
    texture_prompt: str | None = Field(...)
    texture_image_url: str | None = Field(...)
    ai_model: str = Field(...)
    moderation: bool = Field(False)


class MeshyImageToModelRequest(BaseModel):
    image_url: str = Field(...)
    ai_model: str = Field(...)
    topology: str | None = Field(..., description="'quad' or 'triangle'")
    target_polycount: int | None = Field(..., ge=100, le=300000)
    symmetry_mode: str = Field(..., description="'auto', 'off' or 'on'")
    should_remesh: bool = Field(
        True,
        description="False returns the original mesh, ignoring topology and polycount.",
    )
    should_texture: bool = Field(...)
    enable_pbr: bool | None = Field(...)
    texture_resolution: str | None = Field(None)
    pose_mode: str = Field(...)
    ultra_mode: bool = Field(False)
    texture_prompt: str | None = Field(None, max_length=600)
    texture_image_url: str | None = Field(None)
    seed: int = Field(...)
    moderation: bool = Field(False)


class MeshyMultiImageToModelRequest(BaseModel):
    image_urls: list[str] = Field(...)
    ai_model: str = Field(...)
    topology: str | None = Field(..., description="'quad' or 'triangle'")
    target_polycount: int | None = Field(..., ge=100, le=300000)
    symmetry_mode: str = Field(..., description="'auto', 'off' or 'on'")
    should_remesh: bool = Field(
        True,
        description="False returns the original mesh, ignoring topology and polycount.",
    )
    should_texture: bool = Field(...)
    enable_pbr: bool | None = Field(...)
    texture_resolution: str | None = Field(None)
    pose_mode: str = Field(...)
    texture_prompt: str | None = Field(None, max_length=600)
    texture_image_url: str | None = Field(None)
    seed: int = Field(...)
    moderation: bool = Field(False)


class MeshyRiggingRequest(BaseModel):
    input_task_id: str = Field(...)
    height_meters: float = Field(...)
    texture_image_url: str | None = Field(...)


class MeshyAnimationRequest(BaseModel):
    rig_task_id: str = Field(...)
    action_id: int = Field(...)


class MeshyTextureRequest(BaseModel):
    input_task_id: str = Field(...)
    ai_model: str = Field(...)
    enable_original_uv: bool = Field(...)
    enable_pbr: bool = Field(...)
    texture_resolution: str = Field(...)
    text_style_prompt: str | None = Field(None)
    image_style_url: str | None = Field(None)
    multiview_image_urls: list[str] | None = Field(None)


class MeshyModelsUrls(BaseModel):
    glb: str = Field("")
    fbx: str = Field("")
    usdz: str = Field("")
    obj: str = Field("")


class MeshyRiggedModelsUrls(BaseModel):
    rigged_character_glb_url: str = Field("")
    rigged_character_fbx_url: str = Field("")


class MeshyAnimatedModelsUrls(BaseModel):
    animation_glb_url: str = Field("")
    animation_fbx_url: str = Field("")


class MeshyResultTextureUrls(BaseModel):
    base_color: str = Field(...)
    metallic: str | None = Field(None)
    normal: str | None = Field(None)
    roughness: str | None = Field(None)


class MeshyTaskError(BaseModel):
    message: str | None = Field(None)


class MeshyModelResult(BaseModel):
    id: str = Field(...)
    type: str = Field(...)
    model_urls: MeshyModelsUrls = Field(MeshyModelsUrls())
    thumbnail_url: str = Field(...)
    video_url: str | None = Field(None)
    status: str = Field(...)
    progress: int = Field(0)
    texture_urls: list[MeshyResultTextureUrls] | None = Field([])
    task_error: MeshyTaskError | None = Field(None)


class MeshyRiggedResult(BaseModel):
    id: str = Field(...)
    type: str = Field(...)
    status: str = Field(...)
    progress: int = Field(0)
    result: MeshyRiggedModelsUrls = Field(MeshyRiggedModelsUrls())
    task_error: MeshyTaskError | None = Field(None)


class MeshyAnimationResult(BaseModel):
    id: str = Field(...)
    type: str = Field(...)
    status: str = Field(...)
    progress: int = Field(0)
    result: MeshyAnimatedModelsUrls = Field(MeshyAnimatedModelsUrls())
    task_error: MeshyTaskError | None = Field(None)
