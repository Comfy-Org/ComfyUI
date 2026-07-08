from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, RootModel


class TripoModelVersion(str, Enum):
    v3_1_20260211 = "v3.1-20260211"
    v3_0_20250812 = "v3.0-20250812"
    v2_5_20250123 = "v2.5-20250123"
    v2_0_20240919 = "v2.0-20240919"
    v1_4_20240625 = "v1.4-20240625"


class TripoGeometryQuality(str, Enum):
    standard = "standard"
    detailed = "detailed"


class TripoTextureQuality(str, Enum):
    standard = "standard"
    detailed = "detailed"


class TripoStyle(str, Enum):
    PERSON_TO_CARTOON = "person:person2cartoon"
    ANIMAL_VENOM = "animal:venom"
    OBJECT_CLAY = "object:clay"
    OBJECT_STEAMPUNK = "object:steampunk"
    OBJECT_CHRISTMAS = "object:christmas"
    OBJECT_BARBIE = "object:barbie"
    GOLD = "gold"
    ANCIENT_BRONZE = "ancient_bronze"
    NONE = "None"


class TripoTaskType(str, Enum):
    TEXT_TO_MODEL = "text_to_model"
    IMAGE_TO_MODEL = "image_to_model"
    MULTIVIEW_TO_MODEL = "multiview_to_model"
    TEXTURE_MODEL = "texture_model"
    REFINE_MODEL = "refine_model"
    ANIMATE_PRERIGCHECK = "animate_prerigcheck"
    ANIMATE_RIG = "animate_rig"
    ANIMATE_RETARGET = "animate_retarget"
    STYLIZE_MODEL = "stylize_model"
    CONVERT_MODEL = "convert_model"


class TripoTextureAlignment(str, Enum):
    ORIGINAL_IMAGE = "original_image"
    GEOMETRY = "geometry"


class TripoOrientation(str, Enum):
    ALIGN_IMAGE = "align_image"
    DEFAULT = "default"


class TripoOutFormat(str, Enum):
    GLB = "glb"
    FBX = "fbx"


class TripoSpec(str, Enum):
    MIXAMO = "mixamo"
    TRIPO = "tripo"


class TripoAnimation(str, Enum):
    IDLE = "preset:idle"
    WALK = "preset:walk"
    RUN = "preset:run"
    DIVE = "preset:dive"
    CLIMB = "preset:climb"
    JUMP = "preset:jump"
    SLASH = "preset:slash"
    SHOOT = "preset:shoot"
    HURT = "preset:hurt"
    FALL = "preset:fall"
    TURN = "preset:turn"
    QUADRUPED_WALK = "preset:quadruped:walk"
    HEXAPOD_WALK = "preset:hexapod:walk"
    OCTOPOD_WALK = "preset:octopod:walk"
    SERPENTINE_MARCH = "preset:serpentine:march"
    AQUATIC_MARCH = "preset:aquatic:march"


class TripoConvertFormat(str, Enum):
    GLTF = "GLTF"
    USDZ = "USDZ"
    FBX = "FBX"
    OBJ = "OBJ"
    STL = "STL"
    _3MF = "3MF"


class TripoTextureFormat(str, Enum):
    BMP = "BMP"
    DPX = "DPX"
    HDR = "HDR"
    JPEG = "JPEG"
    OPEN_EXR = "OPEN_EXR"
    PNG = "PNG"
    TARGA = "TARGA"
    TIFF = "TIFF"
    WEBP = "WEBP"


class TripoTaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"
    BANNED = "banned"
    EXPIRED = "expired"


class TripoFbxPreset(str, Enum):
    BLENDER = "blender"
    MIXAMO = "mixamo"
    _3DSMAX = "3dsmax"


class TripoFileTokenReference(BaseModel):
    type: str | None = Field(None, description="The type of the reference")
    file_token: str


class TripoUrlReference(BaseModel):
    type: str | None = Field(None, description="The type of the reference")
    url: str


class TripoObjectStorage(BaseModel):
    bucket: str
    key: str


class TripoObjectReference(BaseModel):
    type: str
    object: TripoObjectStorage


class TripoFileEmptyReference(BaseModel):
    pass


class TripoFileReference(RootModel):
    root: TripoFileTokenReference | TripoUrlReference | TripoObjectReference | TripoFileEmptyReference


class TripoTextToModelRequest(BaseModel):
    type: TripoTaskType = Field(TripoTaskType.TEXT_TO_MODEL, description="Type of task")
    prompt: str = Field(..., description="The text prompt describing the model to generate", max_length=1024)
    negative_prompt: str | None = Field(None, description="The negative text prompt", max_length=1024)
    model_version: TripoModelVersion | None = TripoModelVersion.v2_5_20250123
    face_limit: int | None = Field(None, description="The number of faces to limit the generation to")
    texture: bool | None = Field(True, description="Whether to apply texture to the generated model")
    pbr: bool | None = Field(True, description="Whether to apply PBR to the generated model")
    image_seed: int | None = Field(None, description="The seed for the text")
    model_seed: int | None = Field(None, description="The seed for the model")
    texture_seed: int | None = Field(None, description="The seed for the texture")
    texture_quality: TripoTextureQuality | None = TripoTextureQuality.standard
    geometry_quality: TripoGeometryQuality | None = TripoGeometryQuality.standard
    style: TripoStyle | None = None
    auto_size: bool | None = Field(False, description="Whether to auto-size the model")
    quad: bool | None = Field(False, description="Whether to apply quad to the generated model")


class TripoImageToModelRequest(BaseModel):
    type: TripoTaskType = Field(TripoTaskType.IMAGE_TO_MODEL, description="Type of task")
    file: TripoFileReference = Field(..., description="The file reference to convert to a model")
    model_version: TripoModelVersion | None = Field(None, description="The model version to use for generation")
    face_limit: int | None = Field(None, description="The number of faces to limit the generation to")
    texture: bool | None = Field(True, description="Whether to apply texture to the generated model")
    pbr: bool | None = Field(True, description="Whether to apply PBR to the generated model")
    model_seed: int | None = Field(None, description="The seed for the model")
    texture_seed: int | None = Field(None, description="The seed for the texture")
    texture_quality: TripoTextureQuality | None = TripoTextureQuality.standard
    geometry_quality: TripoGeometryQuality | None = TripoGeometryQuality.standard
    texture_alignment: TripoTextureAlignment | None = Field(
        TripoTextureAlignment.ORIGINAL_IMAGE, description="The texture alignment method"
    )
    style: TripoStyle | None = Field(None, description="The style to apply to the generated model")
    auto_size: bool | None = Field(False, description="Whether to auto-size the model")
    orientation: TripoOrientation | None = TripoOrientation.DEFAULT
    quad: bool | None = Field(False, description="Whether to apply quad to the generated model")


class TripoMultiviewToModelRequest(BaseModel):
    type: TripoTaskType = TripoTaskType.MULTIVIEW_TO_MODEL
    files: list[TripoFileReference] = Field(..., description="The file references to convert to a model")
    model_version: TripoModelVersion | None = Field(None, description="The model version to use for generation")
    orthographic_projection: bool | None = Field(False, description="Whether to use orthographic projection")
    face_limit: int | None = Field(None, description="The number of faces to limit the generation to")
    texture: bool | None = Field(True, description="Whether to apply texture to the generated model")
    pbr: bool | None = Field(True, description="Whether to apply PBR to the generated model")
    model_seed: int | None = Field(None, description="The seed for the model")
    texture_seed: int | None = Field(None, description="The seed for the texture")
    texture_quality: TripoTextureQuality | None = TripoTextureQuality.standard
    geometry_quality: TripoGeometryQuality | None = TripoGeometryQuality.standard
    texture_alignment: TripoTextureAlignment | None = TripoTextureAlignment.ORIGINAL_IMAGE
    auto_size: bool | None = Field(False, description="Whether to auto-size the model")
    orientation: TripoOrientation | None = Field(TripoOrientation.DEFAULT, description="The orientation for the model")
    quad: bool | None = Field(False, description="Whether to apply quad to the generated model")


class TripoTexturePrompt(BaseModel):
    text: str | None = Field(None, description="Text guidance for texture generation")


class TripoTextureModelRequest(BaseModel):
    type: TripoTaskType = Field(TripoTaskType.TEXTURE_MODEL, description="Type of task")
    original_model_task_id: str = Field(..., description="The task ID of the original model")
    texture: bool | None = Field(True, description="Whether to apply texture to the model")
    pbr: bool | None = Field(True, description="Whether to apply PBR to the model")
    model_seed: int | None = Field(None, description="The seed for the model")
    texture_seed: int | None = Field(None, description="The seed for the texture")
    texture_quality: TripoTextureQuality | None = Field(None, description="The quality of the texture")
    texture_alignment: TripoTextureAlignment | None = Field(
        TripoTextureAlignment.ORIGINAL_IMAGE, description="The texture alignment method"
    )
    texture_prompt: TripoTexturePrompt | None = Field(
        None,
        description="Optional guidance for texturing. Required in practice for imported models, "
        "which carry no source image to infer texture from.",
    )


class TripoRefineModelRequest(BaseModel):
    type: TripoTaskType = Field(TripoTaskType.REFINE_MODEL, description="Type of task")
    draft_model_task_id: str = Field(..., description="The task ID of the draft model")


class TripoAnimateRigRequest(BaseModel):
    type: TripoTaskType = Field(TripoTaskType.ANIMATE_RIG, description="Type of task")
    original_model_task_id: str = Field(..., description="The task ID of the original model")
    out_format: TripoOutFormat | None = Field(TripoOutFormat.GLB, description="The output format")
    spec: TripoSpec | None = Field(TripoSpec.TRIPO, description="The specification for rigging")


class TripoAnimateRetargetRequest(BaseModel):
    type: TripoTaskType = Field(TripoTaskType.ANIMATE_RETARGET, description="Type of task")
    original_model_task_id: str = Field(..., description="The task ID of the original model")
    animation: TripoAnimation = Field(..., description="The animation to apply")
    out_format: TripoOutFormat | None = Field(TripoOutFormat.GLB, description="The output format")
    bake_animation: bool | None = Field(True, description="Whether to bake the animation")


class TripoConvertModelRequest(BaseModel):
    type: TripoTaskType = Field(TripoTaskType.CONVERT_MODEL, description="Type of task")
    format: TripoConvertFormat = Field(..., description="The format to convert to")
    original_model_task_id: str = Field(..., description="The task ID of the original model")
    quad: bool | None = Field(None, description="Whether to apply quad to the model")
    force_symmetry: bool | None = Field(None, description="Whether to force symmetry")
    face_limit: int | None = Field(None, description="The number of faces to limit the conversion to")
    flatten_bottom: bool | None = Field(None, description="Whether to flatten the bottom of the model")
    flatten_bottom_threshold: float | None = Field(None, description="The threshold for flattening the bottom")
    texture_size: int | None = Field(None, description="The size of the texture")
    texture_format: TripoTextureFormat | None = Field(TripoTextureFormat.JPEG, description="The format of the texture")
    pivot_to_center_bottom: bool | None = Field(None, description="Whether to pivot to the center bottom")
    scale_factor: float | None = Field(None, description="The scale factor for the model")
    with_animation: bool | None = Field(None, description="Whether to include animations")
    pack_uv: bool | None = Field(None, description="Whether to pack the UVs")
    bake: bool | None = Field(None, description="Whether to bake the model")
    part_names: list[str] | None = Field(None, description="The names of the parts to include")
    fbx_preset: TripoFbxPreset | None = Field(None, description="The preset for the FBX export")
    export_vertex_colors: bool | None = Field(None, description="Whether to export the vertex colors")
    export_orientation: TripoOrientation | None = Field(None, description="The orientation for the export")
    animate_in_place: bool | None = Field(None, description="Whether to animate in place")


class TripoP1CommonRequest(BaseModel):
    """Fields supported by Tripo P1 across all input types."""

    model_version: str = Field("P1-20260311")
    model_seed: int | None = Field(None, description="Random seed for geometry generation")
    face_limit: int | None = Field(None, ge=48, le=20000, description="Target face count (48-20000)")
    texture: bool | None = Field(None, description="Enable texturing; pbr=True forces this true")
    pbr: bool | None = Field(None, description="Enable PBR maps; when true, texture is also enabled")
    texture_seed: int | None = Field(None, description="Random seed for texture generation")
    texture_quality: str | None = Field(None, description='"standard" or "detailed"')
    auto_size: bool | None = Field(None, description="Scale to real-world meters")
    compress: str | None = Field(None, description='Only "geometry" is supported')
    export_uv: bool | None = Field(None, description="Perform UV unwrapping during generation")


class TripoP1TextToModelRequest(TripoP1CommonRequest):
    type: str = "text_to_model"
    prompt: str = Field(..., max_length=1024)
    negative_prompt: str | None = Field(None, max_length=255)
    image_seed: int | None = None


class TripoP1ImageToModelRequest(TripoP1CommonRequest):
    type: str = "image_to_model"
    file: TripoFileReference
    enable_image_autofix: bool | None = None
    texture_alignment: str | None = Field(None, description='"original_image" or "geometry"')
    orientation: str | None = Field(None, description='"default" or "align_image"; needs texture=true')


class TripoP1MultiviewToModelRequest(TripoP1CommonRequest):
    """P1 multiview generation.

    Tripo requires `files` to be exactly four entries in [front, left, back, right] order with `{}`
    (TripoFileEmptyReference) for omitted slots; front is required and at least two images total must be provided.
    """

    type: str = "multiview_to_model"
    files: list[TripoFileReference]
    texture_alignment: str | None = None
    orientation: str | None = None


class TripoImportModelRequest(BaseModel):
    """Request for the comfy-api composite import endpoint (/proxy/tripo/v2/openapi/import).

    The model file is uploaded to ComfyUI API storage first; the backend downloads it from
    `url`, re-uploads it to Tripo's storage and creates the import_model task server-side.
    """

    url: str = Field(..., description="ComfyUI API storage download URL of the model file")
    format: str = Field(..., description='File format: "glb", "fbx", "obj" or "stl"')


class TripoTaskOutput(BaseModel):
    model: str | None = Field(None, description="URL to the model")
    base_model: str | None = Field(None, description="URL to the base model")
    pbr_model: str | None = Field(None, description="URL to the PBR model")
    rendered_image: str | None = Field(None, description="URL to the rendered image")
    riggable: bool | None = Field(None, description="Whether the model is riggable")


class TripoTask(BaseModel):
    task_id: str = Field(..., description="The task ID")
    type: str | None = Field(None, description="The type of task")
    status: TripoTaskStatus | None = Field(None, description="The status of the task")
    input: dict[str, Any] | None = Field(None, description="The input parameters for the task")
    output: TripoTaskOutput | None = Field(None, description="The output of the task")
    progress: int | None = Field(None, description="The progress of the task", ge=0, le=100)
    create_time: int | None = Field(None, description="The creation time of the task")
    running_left_time: int | None = Field(None, description="The estimated time left for the task")
    queue_position: int | None = Field(None, description="The position in the queue")
    consumed_credit: int | None = Field(None)


class TripoTaskResponse(BaseModel):
    code: int = Field(0, description="The response code")
    data: TripoTask = Field(..., description="The task data")


class TripoErrorResponse(BaseModel):
    code: int = Field(..., description="The error code")
    message: str = Field(..., description="The error message")
    suggestion: str = Field(..., description="The suggestion for fixing the error")
