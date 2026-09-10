from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, RootModel


class TripoModelVersion(str, Enum):
    v3_1_20260211 = "v3.1-20260211"
    v3_0_20250812 = "v3.0-20250812"
    v2_5_20250123 = "v2.5-20250123"


class TripoTextureModelVersion(str, Enum):
    v3_0_20250812 = "v3.0-20250812"
    v2_5_20250123 = "v2.5-20250123"


class TripoGeometryQuality(str, Enum):
    standard = "standard"
    detailed = "detailed"


class TripoTextureQuality(str, Enum):
    standard = "standard"
    detailed = "detailed"
    extreme = "extreme"


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


class TripoRigModelVersion(str, Enum):
    v1_0_20240301 = "v1.0-20240301"
    v2_5_20260210 = "v2.5-20260210"


class TripoRigType(str, Enum):
    BIPED = "biped"
    QUADRUPED = "quadruped"
    HEXAPOD = "hexapod"
    OCTOPOD = "octopod"
    AVIAN = "avian"
    SERPENTINE = "serpentine"
    AQUATIC = "aquatic"


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


TRIPO_BIPED_ANIMATIONS = (
    "preset:biped:afraid",
    "preset:biped:agree",
    "preset:biped:angry_01",
    "preset:biped:angry_02",
    "preset:biped:angry_03",
    "preset:biped:basketball_shot",
    "preset:biped:bow",
    "preset:biped:box_01",
    "preset:biped:box_02",
    "preset:biped:box_03",
    "preset:biped:cast_a_spell",
    "preset:biped:cheer",
    "preset:biped:chop",
    "preset:biped:clap",
    "preset:biped:climb",
    "preset:biped:complain_01",
    "preset:biped:complain_02",
    "preset:biped:cross_body_crunch",
    "preset:biped:crossover_dribble",
    "preset:biped:cry",
    "preset:biped:dance_01",
    "preset:biped:dance_02",
    "preset:biped:dance_03",
    "preset:biped:dance_04",
    "preset:biped:dance_05",
    "preset:biped:dance_06",
    "preset:biped:defeat_02",
    "preset:biped:defeat_03",
    "preset:biped:depressed",
    "preset:biped:dig",
    "preset:biped:dive",
    "preset:biped:dribble",
    "preset:biped:fall",
    "preset:biped:fire",
    "preset:biped:flee_01",
    "preset:biped:flee_02",
    "preset:biped:flip",
    "preset:biped:fold_arms",
    "preset:biped:football_catch",
    "preset:biped:football_save",
    "preset:biped:football_pass",
    "preset:biped:freaky",
    "preset:biped:frightened",
    "preset:biped:front_kick_01",
    "preset:biped:front_kick_02",
    "preset:biped:frustrated_01",
    "preset:biped:frustrated_02",
    "preset:biped:golf",
    "preset:biped:greet_01",
    "preset:biped:greet_02",
    "preset:biped:greet_03",
    "preset:biped:greet_04",
    "preset:biped:heart_pose",
    "preset:biped:hit_to_body_01",
    "preset:biped:hit_to_body_02",
    "preset:biped:hit_to_head",
    "preset:biped:hit_to_side",
    "preset:biped:hit_to_stomach",
    "preset:biped:hug",
    "preset:biped:hurt",
    "preset:biped:idle",
    "preset:biped:jump_down",
    "preset:biped:jump",
    "preset:biped:jump_rope_01",
    "preset:biped:jump_rope_02",
    "preset:biped:laugh_01",
    "preset:biped:laugh_02",
    "preset:biped:lift_heavy",
    "preset:biped:look_around",
    "preset:biped:make_a_call_01",
    "preset:biped:make_a_call_02",
    "preset:biped:pitch_baseball",
    "preset:biped:play_mobile_game",
    "preset:biped:play_video_game",
    "preset:biped:run_upstairs",
    "preset:biped:run",
    "preset:biped:scared_01",
    "preset:biped:scared_02",
    "preset:biped:scratch",
    "preset:biped:shoot",
    "preset:biped:shovel",
    "preset:biped:sing_01",
    "preset:biped:sing_02",
    "preset:biped:sing_03",
    "preset:biped:sing_04",
    "preset:biped:sit",
    "preset:biped:slash",
    "preset:biped:sob",
    "preset:biped:standing_relax",
    "preset:biped:surf",
    "preset:biped:swagger",
    "preset:biped:swim",
    "preset:biped:turn",
    "preset:biped:victory_celebration",
    "preset:biped:volleyball",
    "preset:biped:wait",
    "preset:biped:walk",
    "preset:biped:warm_up",
    "preset:biped:wave_goodbye_01",
    "preset:biped:wave_goodbye_02",
)


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
    BAKE_SCALE = "bake_scale"


class TripoExportOrientation(str, Enum):
    PLUS_X = "+x"
    MINUS_X = "-x"
    PLUS_Y = "+y"
    MINUS_Y = "-y"


class TripoFileTokenReference(BaseModel):
    type: str | None = Field(None, description="The type of the reference")
    file_token: str


class TripoUrlReference(BaseModel):
    type: str | None = Field(None, description="The type of the reference")
    url: str


class TripoFileEmptyReference(BaseModel):
    pass


class TripoFileReference(RootModel):
    root: TripoFileTokenReference | TripoUrlReference | TripoFileEmptyReference


class TripoTextToModelRequest(BaseModel):
    prompt: str = Field(..., description="The text prompt describing the model to generate", max_length=1024)
    model: TripoModelVersion = TripoModelVersion.v3_1_20260211
    negative_prompt: str | None = Field(None, description="The negative text prompt", max_length=255)
    face_limit: int | None = Field(None, description="The number of faces to limit the generation to")
    texture: bool | None = Field(True, description="Whether to apply texture to the generated model")
    pbr: bool | None = Field(True, description="Whether to apply PBR to the generated model")
    image_seed: int | None = Field(None, description="The seed for the text")
    model_seed: int | None = Field(None, description="The seed for the model")
    texture_seed: int | None = Field(None, description="The seed for the texture")
    texture_quality: TripoTextureQuality | None = TripoTextureQuality.standard
    geometry_quality: TripoGeometryQuality | None = TripoGeometryQuality.standard
    auto_size: bool | None = Field(False, description="Whether to auto-size the model")
    quad: bool | None = Field(False, description="Whether to apply quad to the generated model")
    smart_low_poly: bool | None = Field(None, description="Low-poly output with clean, hand-crafted style topology")


class TripoImageToModelRequest(BaseModel):
    input: str = Field(..., description="Image URL or file token")
    model: TripoModelVersion = TripoModelVersion.v3_1_20260211
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
    auto_size: bool | None = Field(False, description="Whether to auto-size the model")
    orientation: TripoOrientation | None = TripoOrientation.DEFAULT
    quad: bool | None = Field(False, description="Whether to apply quad to the generated model")
    smart_low_poly: bool | None = Field(None, description="Low-poly output with clean, hand-crafted style topology")


class TripoMultiviewToModelRequest(BaseModel):
    inputs: list[dict[str, str]] = Field(..., description="View-keyed image references: front, left, back, right")
    model: TripoModelVersion = TripoModelVersion.v3_1_20260211
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
    smart_low_poly: bool | None = Field(None, description="Low-poly output with clean, hand-crafted style topology")


class TripoTexturePrompt(BaseModel):
    text: str | None = Field(None, description="Text guidance for texture generation")
    style_image: TripoFileReference | None = Field(None, description="Style reference, only together with text")
    image: TripoFileReference | None = Field(None, description="Single reference image")
    images: list[TripoFileReference] | None = Field(None, description="Exactly 4 reference images: front, left, back, right")


class TripoTextureModelRequest(BaseModel):
    input: str = Field(..., description="The task ID of the model to texture")
    model: TripoTextureModelVersion | None = Field(None, description="Texture model version")
    pbr: bool | None = Field(True, description="Whether to apply PBR to the model")
    texture_seed: int | None = Field(None, description="The seed for the texture")
    texture_quality: TripoTextureQuality | None = Field(None, description="The quality of the texture")
    texture_alignment: TripoTextureAlignment | None = Field(
        TripoTextureAlignment.ORIGINAL_IMAGE, description="The texture alignment method"
    )
    texture_prompt: TripoTexturePrompt | None = Field(None, description="Optional guidance for texturing")
    part_names: list[str] | None = Field(None, description="Parts of a segmented model to texture; all parts when omitted")


class TripoAnimatePrerigcheckRequest(BaseModel):
    input: str = Field(..., description="The task ID of the model")


class TripoAnimateRigRequest(BaseModel):
    input: str = Field(..., description="The task ID of the model")
    model: TripoRigModelVersion | None = Field(None, description="Rigging model version")
    rig_type: TripoRigType | None = Field(None, description="Skeleton type")
    out_format: TripoOutFormat | None = Field(TripoOutFormat.GLB, description="The output format")
    spec: TripoSpec | None = Field(TripoSpec.TRIPO, description="The specification for rigging")


class TripoAnimateRetargetRequest(BaseModel):
    input: str = Field(..., description="The task ID of the rigged model")
    animation: str = Field(..., description="The animation preset to apply")
    out_format: TripoOutFormat | None = Field(TripoOutFormat.GLB, description="The output format")
    export_with_geometry: bool | None = Field(None, description="Whether to export geometry with the animation")
    animate_in_place: bool | None = Field(None, description="Whether to play the animation in place")


class TripoMeshSegmentationRequest(BaseModel):
    input: str = Field(..., description="The task ID of the model to segment")


class TripoMeshCompletionRequest(BaseModel):
    input: str = Field(..., description="The task ID of a mesh segmentation task")
    part_names: list[str] | None = Field(None, description="Parts to complete; all parts when omitted")


class TripoHighpolyToLowpolyRequest(BaseModel):
    input: str = Field(..., description="The task ID of the model to retopologize")
    face_limit: int | None = Field(None, description="Target face count; adaptive when omitted")
    quad: bool | None = Field(None, description="Whether to output a quad mesh")
    bake: bool | None = Field(None, description="Whether to bake textures onto the low-poly mesh")
    part_names: list[str] | None = Field(None, description="Parts to retopologize; whole model when omitted")


class TripoGenerateMultiviewImageRequest(BaseModel):
    input: str = Field(..., description="URL or file token of the source image")


class TripoMultiviewEditPrompt(BaseModel):
    view: str = Field(..., description="front, left, back or right")
    prompt: str = Field(..., description="Edit instruction for the view", max_length=1024)


class TripoEditMultiviewImageRequest(BaseModel):
    input: str = Field(..., description="The task ID of the multiview images to edit")
    prompts: list[TripoMultiviewEditPrompt] = Field(..., description="Per-view edit instructions")


class TripoMeshSmartSegmentRequest(BaseModel):
    input: str = Field(..., description="URL or file token of a GLB model, or of an image")
    seg_type: str = Field(..., description='"model" or "image"')
    transform: list[float] | None = Field(None, description="Column-major 4x4 matrix, required for models")
    granularity: str | None = Field(None, description="coarse, medium or fine")
    hint: str | None = Field(None, description="Text naming the parts to look for")


class TripoConvertModelRequest(BaseModel):
    input: str = Field(..., description="The task ID of the model to convert")
    format: TripoConvertFormat = Field(..., description="The format to convert to")
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
    export_orientation: TripoExportOrientation | None = Field(None, description="Forward axis of the exported model")
    animate_in_place: bool | None = Field(None, description="Whether to animate in place")


class TripoP1CommonRequest(BaseModel):
    """Fields supported by Tripo P1 across all input types."""

    model: str = Field("P1-20260311")
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
    prompt: str = Field(..., max_length=1024)
    negative_prompt: str | None = Field(None, max_length=255)
    image_seed: int | None = None


class TripoP1ImageToModelRequest(TripoP1CommonRequest):
    input: str
    enable_image_autofix: bool | None = None
    texture_alignment: str | None = Field(None, description='"original_image" or "geometry"')
    orientation: str | None = Field(None, description='"default" or "align_image"; needs texture=true')


class TripoP1MultiviewToModelRequest(TripoP1CommonRequest):
    inputs: list[dict[str, str]]
    texture_alignment: str | None = None
    orientation: str | None = None


class TripoImportModelRequest(BaseModel):
    input: str = Field(..., description="URL or file token of the model file")


class TripoTaskOutput(BaseModel):
    model_url: str | None = Field(None, description="URL to the model")
    rendered_image_url: str | None = Field(None, description="URL to the rendered preview")
    generated_image_url: str | None = Field(None, description="URL to the intermediate image of text-to-model")
    riggable: bool | None = Field(None, description="Whether the model is riggable")
    rig_type: str | None = Field(None, description="Recommended rig type")
    generate_multiview_image: dict[str, str] | None = Field(None, description="View name to image URL")
    prompt: str | None = Field(None, description="Smart segmentation: the parts Tripo found")
    mask_url: str | None = Field(None, description="Smart segmentation: part mask image")
    seg_model_url: str | None = Field(None, description="Smart segmentation: segmented GLB")
    seg_task_id: str | None = Field(None, description="Smart segmentation: the mesh_segmentation task")
    model_task_id: str | None = Field(None, description="Smart segmentation: the imported or generated model task")


class TripoTask(BaseModel):
    task_id: str = Field(..., description="The task ID")
    type: str | None = Field(None, description="The type of task")
    status: TripoTaskStatus | None = Field(None, description="The status of the task")
    input: dict[str, Any] | None = Field(None, description="The input parameters for the task")
    output: TripoTaskOutput | None = Field(None, description="The output of the task")
    progress: int | None = Field(None, description="The progress of the task", ge=0, le=100)
    created_at: str | None = Field(None, description="The creation time of the task")
    completed_at: str | None = Field(None, description="The completion time of the task")
    credits_consumed: float | None = Field(None)
    error_code: int | None = Field(None)
    error_message: str | None = Field(None)


class TripoTaskResponse(BaseModel):
    code: int = Field(0, description="The response code")
    data: TripoTask = Field(..., description="The task data")


class TripoFileData(BaseModel):
    file_token: str


class TripoFileResponse(BaseModel):
    code: int = Field(0, description="The response code")
    data: TripoFileData


class TripoErrorResponse(BaseModel):
    code: int = Field(..., description="The error code")
    message: str = Field(..., description="The error message")
    suggestion: str = Field(..., description="The suggestion for fixing the error")
