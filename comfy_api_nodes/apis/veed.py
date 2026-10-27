from pydantic import BaseModel, Field


class VeedFabricRequest(BaseModel):
    image_url: str = Field(...)
    audio_url: str = Field(...)
    resolution: str = Field(...)


class VeedLipsyncRequest(BaseModel):
    video_url: str = Field(...)
    audio_url: str = Field(...)


class VeedBackgroundRemovalRequest(BaseModel):
    video_url: str = Field(...)
    output_codec: str = Field(...)
    refine_foreground_edges: bool = Field(...)
    subject_is_person: bool = Field(...)


class VeedGreenScreenRequest(BaseModel):
    video_url: str = Field(...)
    output_codec: str = Field(...)
    spill_suppression_strength: float = Field(...)


class VeedCleanAudioRequest(BaseModel):
    audio_url: str = Field(...)
    strength: float = Field(...)
    normalize_loudness: bool = Field(...)
    target_lufs: float | None = Field(None)
    output_format: str = Field(...)


class VeedCleanAudioVideoRequest(BaseModel):
    video_url: str = Field(...)
    strength: float = Field(...)
    normalize_loudness: bool = Field(...)
    target_lufs: float | None = Field(None)


class VeedFile(BaseModel):
    url: str = Field(...)


class VeedJobResult(BaseModel):
    video: VeedFile | None = Field(None)
    audio: VeedFile | None = Field(None)
    files: list[VeedFile] | None = Field(None)


class VeedJobError(BaseModel):
    code: str = Field(...)
    message: str = Field(...)


class VeedJob(BaseModel):
    job_id: str = Field(...)
    status: str = Field(...)
    result: VeedJobResult | None = Field(None)
    error: VeedJobError | None = Field(None)


class VeedJobResponse(BaseModel):
    data: VeedJob = Field(...)
