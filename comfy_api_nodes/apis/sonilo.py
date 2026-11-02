from pydantic import BaseModel, Field


class SoniloTextToMusicLyricsRequest(BaseModel):
    prompt: str = Field(...)
    duration: int | None = Field(None)


class SoniloTaskSubmitResponse(BaseModel):
    task_id: str = Field(...)


class SoniloTaskAudio(BaseModel):
    url: str = Field(...)


class SoniloTask(BaseModel):
    status: str = Field(...)
    audio: SoniloTaskAudio | None = Field(None)
