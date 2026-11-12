from pydantic import BaseModel, Field


class BreezeVoiceSettings(BaseModel):
    guidance_scale: float | None = Field(None, description="How strongly generation follows the instructions, 1-10")
    speed: float | None = Field(None, description="Speech speed without pitch change, 0.5-2.0")
    use_speaker_boost: bool | None = Field(None, description="Keep the voice closer to its reference")


class BreezeTextToSpeechRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize, up to 1000 characters")
    model_id: str = Field(...)
    language_code: str = Field(..., description="ISO 639-1 language of the text")
    instructions: str | None = Field(None, description="Performance instructions in the language of the text")
    voice_settings: BreezeVoiceSettings = Field(...)


class BreezeVoiceDesignRequest(BaseModel):
    voice_description: str = Field(..., description="Text description of the voice, 3-500 characters")
    text: str = Field(..., description="Preview script, up to 500 characters")
    model_id: str = Field(...)
    language_code: str = Field(...)
    guidance_scale: float = Field(..., description="How strongly the voice follows the description, 1-10")
    preview_count: int = Field(1)


class BreezeVoiceDesignPreview(BaseModel):
    generated_voice_id: str = Field(...)
    audio_base_64: str = Field(...)
    media_type: str | None = Field(None)
    duration_secs: float | None = Field(None)


class BreezeVoiceDesignResponse(BaseModel):
    previews: list[BreezeVoiceDesignPreview] = Field(...)
    text: str | None = Field(None, description="The script the previews read")


class BreezeVoiceCloneRequest(BaseModel):
    remove_background_noise: bool = Field(...)


class BreezeVoiceCloneResponse(BaseModel):
    generated_voice_id: str = Field(...)


class BreezeVoiceRemixRequest(BaseModel):
    voice_id: str = Field(...)
    prompt: str | None = Field(None, description="Desired voice change; required unless preset_id is set")
    preset_id: str | None = Field(None)
    guidance_scale: float = Field(..., description="Strength of the change, 1-10")


class BreezeVoiceLocalizeRequest(BaseModel):
    voice_id: str = Field(...)
    language_code: str = Field(..., description="Target speaking language")


class BreezeVoicePreviewJob(BaseModel):
    generation_job_id: str = Field(...)
    status: str | None = Field(None)


class BreezeVoicePreviewJobError(BaseModel):
    code: str | None = Field(None)
    detail: str | None = Field(None)


class BreezeVoicePreviewJobStatus(BaseModel):
    generation_job_id: str = Field(...)
    status: str = Field(...)
    generated_voice_id: str | None = Field(None)
    language_code: str | None = Field(None)
    text: str | None = Field(None)
    error: BreezeVoicePreviewJobError | None = Field(None)


class BreezeSaveVoiceRequest(BaseModel):
    voice_name: str | None = Field(None, description="Display name, up to 80 characters")
    language_code: str | None = Field(None, description="Language of the voice; omitted for clones to keep the detected one")


class BreezeSavedVoice(BaseModel):
    voice_id: str = Field(...)
