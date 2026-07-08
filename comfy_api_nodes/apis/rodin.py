from enum import Enum

from pydantic import BaseModel, Field


class Rodin3DGenerateRequest(BaseModel):
    seed: int = Field(..., description="seed_")
    tier: str = Field(..., description="Tier of generation.")
    material: str = Field(..., description="The material type.")
    quality_override: int = Field(..., description="The poly count of the mesh.")
    mesh_mode: str = Field(..., description="It controls the type of faces of generated models.")
    TAPose: bool | None = Field(None, description="")


class Rodin3DGen25Request(BaseModel):

    tier: str = Field(..., description="Gen-2.5 tier (e.g. Gen-2.5-High).")
    prompt: str | None = Field(None, description="Required for Text-to-3D; ignored otherwise.")
    seed: int | None = Field(None, description="0-65535.")
    material: str | None = Field(None, description="PBR | Shaded | All | None.")
    geometry_file_format: str | None = Field(None, description="glb | usdz | fbx | obj | stl.")
    texture_mode: str | None = Field(None, description="legacy | extreme-low | low | medium | high.")
    mesh_mode: str | None = Field(None, description="Raw (triangular) | Quad.")
    quality_override: int | None = Field(None, description="Mesh face count override.")
    geometry_instruct_mode: str | None = Field(None, description="faithful | creative.")
    bbox_condition: list[int] | None = Field(None, description="Bounding box [Width(Y), Height(Z), Length(X)] in cm.")
    height: int | None = Field(None, description="Approximate model height in cm.")
    TAPose: bool | None = Field(None, description="T/A pose for human-like models.")
    hd_texture: bool | None = Field(None, description="Enhanced texture quality.")
    texture_delight: bool | None = Field(None, description="Remove baked lighting from textures.")
    is_micro: bool | None = Field(None, description="Micro detail (Extreme-High only).")
    use_original_alpha: bool | None = Field(None, description="Preserve image transparency.")
    preview_render: bool | None = Field(None, description="Generate high-quality preview render.")
    addons: list[str] | None = Field(None, description='Optional addons, e.g. ["HighPack"].')


class GenerateJobsData(BaseModel):
    uuids: list[str] = Field(..., description="str LIST")
    subscription_key: str = Field(..., description="subscription key")


class Rodin3DGenerateResponse(BaseModel):
    message: str | None = Field(None, description="Return message.")
    prompt: str | None = Field(None, description="Generated Prompt from image.")
    submit_time: str | None = Field(None, description="Submit Time")
    uuid: str | None = Field(None, description="Task str")
    jobs: GenerateJobsData | None = Field(None, description="Details of jobs")


class JobStatus(str, Enum):
    """
    Status for jobs
    """

    Done = "Done"
    Failed = "Failed"
    Generating = "Generating"
    Waiting = "Waiting"


class Rodin3DCheckStatusRequest(BaseModel):
    subscription_key: str = Field(..., description="subscription from generate endpoint")


class JobItem(BaseModel):
    uuid: str = Field(..., description="uuid")
    status: JobStatus = Field(..., description="Status Currently")


class Rodin3DCheckStatusResponse(BaseModel):
    jobs: list[JobItem] = Field(..., description="Job status List")


class Rodin3DDownloadRequest(BaseModel):
    task_uuid: str = Field(..., description="Task str")


class RodinResourceItem(BaseModel):
    url: str = Field(..., description="Download Url")
    name: str = Field(..., description="File name with ext")


class Rodin3DDownloadResponse(BaseModel):
    items: list[RodinResourceItem] = Field(..., alias="list", description="Source List")
