from pydantic import BaseModel, Field


class CambAITTSRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech")
    voice_id: int = Field(..., description="Voice ID for TTS")
    language: str = Field(..., description="Language code (e.g., 'en-us')")
    speech_model: str = Field(..., description="TTS model to use")
    output_configuration: dict = Field(
        default_factory=lambda: {"format": "wav"},
        description="Output format configuration",
    )


class CambAITranslateRequest(BaseModel):
    source_language: int = Field(..., description="Source language ID")
    target_language: int = Field(..., description="Target language ID")
    texts: list[str] = Field(..., description="Texts to translate")


class CambAITaskResponse(BaseModel):
    task_id: str = Field(..., description="Async task ID")


class CambAIPollResult(BaseModel):
    status: str = Field(..., description="Task status")
    run_id: int | None = Field(None, description="Run ID for fetching results")


class CambAITranslateResult(BaseModel):
    texts: list[str] = Field(default_factory=list, description="Translated texts")


class CambAIDialogueItem(BaseModel):
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Dialogue text")
    speaker: str = Field(..., description="Speaker identifier")


class CambAIVoiceCloneResponse(BaseModel):
    voice_id: int = Field(..., description="Cloned voice ID")


class CambAITextToSoundRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for sound generation")
    audio_type: str = Field(..., description="Type of audio: 'sound' or 'music'")
    duration: float = Field(..., description="Duration in seconds")
