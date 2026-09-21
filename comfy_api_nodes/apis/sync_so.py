from pydantic import BaseModel, Field


class SyncInputItem(BaseModel):
    type: str = Field(..., description="Input kind: 'video', 'image' or 'audio'.")
    url: str = Field(...)


class SyncActiveSpeakerDetection(BaseModel):
    auto_detect: bool | None = Field(
        None, description="Detect the active speaker automatically. Video input only; rejected for images."
    )
    frame_number: int | None = Field(
        None, description="Frame used for manual speaker selection. Must be 0 for image inputs."
    )
    coordinates: list[int] | None = Field(
        None, description="Pixel [x, y] of the speaker's face in the frame selected by frame_number."
    )


class SyncGenerationOptions(BaseModel):
    sync_mode: str | None = Field(
        None,
        description="How to resolve an audio/video duration mismatch: "
        "cut_off, bounce, loop, silence or remap. Ignored for image inputs.",
    )
    i2v_prompt: str | None = Field(
        None, description="Motion prompt for image-to-video generation. Image input only."
    )
    active_speaker_detection: SyncActiveSpeakerDetection | None = Field(None)


class SyncGenerationRequest(BaseModel):
    model: str = Field(..., description="Generation model, e.g. 'sync-3'.")
    input: list[SyncInputItem] = Field(
        ..., description="Exactly one visual input (video or image) plus one audio input."
    )
    options: SyncGenerationOptions | None = Field(None)


class SyncGeneration(BaseModel):
    """Subset of the Generation object returned by POST /v2/generate and GET /v2/generate/{id}."""

    id: str = Field(...)
    status: str = Field(..., description="PENDING | PROCESSING | COMPLETED | FAILED | REJECTED")
    outputUrl: str | None = Field(None)
    outputDuration: float | None = Field(None)
    error: str | None = Field(None, description="Human-readable failure message.")
    errorCode: str | None = Field(None, description="Stable machine-readable code from the GET /v2/errors catalog.")
