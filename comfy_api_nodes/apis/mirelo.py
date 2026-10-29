from pydantic import BaseModel, Field


class MireloMediaURL(BaseModel):
    type: str = Field("url")
    url: str = Field(...)


class MireloTextInput(BaseModel):
    prompt: str = Field(...)
    negative_prompt: str | None = Field(None)


class MireloTextToSFXRequest(BaseModel):
    model: str = Field(...)
    duration_ms: int = Field(...)
    seed: int = Field(...)
    input: MireloTextInput = Field(...)


class MireloVideoInput(BaseModel):
    video: MireloMediaURL = Field(...)
    audio: MireloMediaURL | None = Field(None)
    prompt: str | None = Field(None)
    negative_prompt: str | None = Field(None)


class MireloVideoControls(BaseModel):
    preserve_speech: bool = Field(...)
    separate_speech: bool = Field(...)


class MireloVideoToSFXRequest(BaseModel):
    model: str = Field(...)
    seed: int = Field(...)
    input: MireloVideoInput = Field(...)
    controls: MireloVideoControls = Field(...)


class MireloAudioFile(BaseModel):
    url: str = Field(...)


class MireloVariantFiles(BaseModel):
    audio: MireloAudioFile | None = Field(None)


class MireloVariant(BaseModel):
    status: str | None = Field(None)
    files: MireloVariantFiles | None = Field(None)


class MireloOutput(BaseModel):
    type: str | None = Field(None)
    variants: list[MireloVariant] = Field(default_factory=list)


class MireloResult(BaseModel):
    output: MireloOutput | None = Field(None)
    outputs: list[MireloOutput] = Field(default_factory=list)


class MireloJobError(BaseModel):
    code: str | None = Field(None)
    message: str | None = Field(None)


class MireloJob(BaseModel):
    id: str = Field(...)
    status: str = Field(...)
    result: MireloResult | None = Field(None)
    errors: list[MireloJobError] = Field(default_factory=list)
