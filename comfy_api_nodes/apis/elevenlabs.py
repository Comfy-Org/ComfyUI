from pydantic import BaseModel, Field


class SpeechToTextRequest(BaseModel):
    model_id: str = Field(...)
    cloud_storage_url: str = Field(...)
    language_code: str | None = Field(None, description="ISO-639-1 or ISO-639-3 language code")
    tag_audio_events: bool | None = Field(None, description="Annotate sounds like (laughter) in transcript")
    num_speakers: int | None = Field(None, description="Max speakers predicted")
    timestamps_granularity: str = Field(default="word", description="Timing precision: none, word, or character")
    diarize: bool | None = Field(None, description="Annotate which speaker is talking")
    diarization_threshold: float | None = Field(None, description="Speaker separation sensitivity")
    temperature: float | None = Field(None, description="Randomness control")
    seed: int = Field(..., description="Seed for deterministic sampling")


class SpeechToTextWord(BaseModel):
    text: str = Field(..., description="The word text")
    type: str = Field(default="word", description="Type of text element (word, spacing, etc.)")
    start: float | None = Field(None, description="Start time in seconds (when timestamps enabled)")
    end: float | None = Field(None, description="End time in seconds (when timestamps enabled)")
    speaker_id: str | None = Field(None, description="Speaker identifier when diarization is enabled")
    logprob: float | None = Field(None, description="Log probability of the word")


class SpeechToTextResponse(BaseModel):
    language_code: str = Field(..., description="Detected or specified language code")
    language_probability: float | None = Field(None, description="Confidence of language detection")
    text: str = Field(..., description="Full transcript text")
    words: list[SpeechToTextWord] | None = Field(None, description="Word-level timing information")


class TextToSpeechVoiceSettings(BaseModel):
    stability: float | None = Field(None, description="Voice stability")
    similarity_boost: float | None = Field(None, description="Similarity boost")
    style: float | None = Field(None, description="Style exaggeration")
    use_speaker_boost: bool | None = Field(None, description="Boost similarity to original speaker")
    speed: float | None = Field(None, description="Speech speed")


class TextToSpeechRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech")
    model_id: str = Field(..., description="Model ID for TTS")
    language_code: str | None = Field(None, description="ISO-639-1 or ISO-639-3 language code")
    voice_settings: TextToSpeechVoiceSettings | None = Field(None, description="Voice settings")
    seed: int = Field(..., description="Seed for deterministic sampling")
    apply_text_normalization: str | None = Field(None, description="Text normalization mode: auto, on, off")


class TextToSoundEffectsRequest(BaseModel):
    text: str = Field(..., description="Text prompt to convert into a sound effect")
    model_id: str = Field(..., description="Model ID for sound effect generation")
    duration_seconds: float = Field(..., description="Duration of generated sound in seconds")
    prompt_influence: float = Field(..., description="How closely generation follows the prompt")
    loop: bool | None = Field(None, description="Whether to create a smoothly looping sound effect")


class AddVoiceRequest(BaseModel):
    name: str = Field(..., description="Name that identifies the voice")
    remove_background_noise: bool = Field(..., description="Remove background noise from voice samples")


class AddVoiceResponse(BaseModel):
    voice_id: str = Field(..., description="The newly created voice's unique identifier")


class SpeechToSpeechRequest(BaseModel):
    model_id: str = Field(..., description="Model ID for speech-to-speech")
    voice_settings: str = Field(..., description="JSON string of voice settings")
    seed: int = Field(..., description="Seed for deterministic sampling")
    remove_background_noise: bool = Field(..., description="Remove background noise from input audio")


class DialogueInput(BaseModel):
    text: str = Field(..., description="Text content to convert to speech")
    voice_id: str = Field(..., description="Voice identifier for this dialogue segment")


class DialogueSettings(BaseModel):
    stability: float | None = Field(None, description="Voice stability (0-1)")


class TextToDialogueRequest(BaseModel):
    inputs: list[DialogueInput] = Field(..., description="List of dialogue segments")
    model_id: str = Field(..., description="Model ID for dialogue generation")
    language_code: str | None = Field(None, description="ISO-639-1 language code")
    settings: DialogueSettings | None = Field(None, description="Voice settings")
    seed: int | None = Field(None, description="Seed for deterministic sampling")
    apply_text_normalization: str | None = Field(None, description="Text normalization mode: auto, on, off")
