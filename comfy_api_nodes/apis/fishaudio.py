from pydantic import BaseModel, Field


class FishAudioProsody(BaseModel):
    speed: float = Field(1.0, description="Speaking rate multiplier, 0.5-2.0")
    volume: float = Field(0.0, description="Volume adjustment in decibels")


class FishAudioTTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    reference_id: str | list[str] | None = Field(None, description="Voice model ID or list of IDs")
    temperature: float = Field(0.7, description="Expressiveness, 0-1")
    top_p: float = Field(0.7, description="Nucleus sampling diversity, (0, 1]")
    prosody: FishAudioProsody = Field(..., description="Speed and volume adjustments")
    normalize: bool = Field(True, description="Normalize numbers and text for English and Chinese")
    format: str = Field("wav", description="Output audio format")


class FishAudioASRRequest(BaseModel):
    language: str | None = Field(None, description="Optional ISO 639-1 language hint")
    ignore_timestamps: bool = Field(True, description="Skip precise timestamp computation")


class FishAudioASRSegment(BaseModel):
    text: str | None = Field(None, description="Segment text")
    start: float | None = Field(None, description="Segment start time in seconds")
    end: float | None = Field(None, description="Segment end time in seconds")


class FishAudioASRResponse(BaseModel):
    text: str | None = Field(None, description="Transcribed text")
    duration: float | None = Field(None, description="Audio duration in seconds")
    segments: list[FishAudioASRSegment] | None = Field(None, description="Timestamped transcript segments")
    language_code: str | None = Field(None, description="Detected language as ISO 639-1 code")
    language: str | None = Field(None, description="Detected language display name")


class FishAudioCreateModelRequest(BaseModel):
    type: str = Field("tts", description="Model type")
    title: str = Field(..., description="Voice model name")
    train_mode: str = Field("fast", description="Training mode; fast is instantly available")
    visibility: str = Field("private", description="Model visibility")
    enhance_audio_quality: bool = Field(..., description="Enhance reference audio quality")


class FishAudioCreateModelResponse(BaseModel):
    id: str = Field(..., alias="_id", description="Voice model ID for use as reference_id")
    state: str | None = Field(None, description="Training state")
    visibility: str | None = Field(None, description="Model visibility")
