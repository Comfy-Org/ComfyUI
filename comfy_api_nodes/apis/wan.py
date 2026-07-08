from pydantic import BaseModel, Field


class Text2ImageInputField(BaseModel):
    prompt: str = Field(...)
    negative_prompt: str | None = Field(None)


class Image2ImageInputField(BaseModel):
    prompt: str = Field(...)
    negative_prompt: str | None = Field(None)
    images: list[str] = Field(..., min_length=1, max_length=2)


class Text2VideoInputField(BaseModel):
    prompt: str = Field(...)
    negative_prompt: str | None = Field(None)
    audio_url: str | None = Field(None)


class Image2VideoInputField(BaseModel):
    prompt: str = Field(...)
    negative_prompt: str | None = Field(None)
    img_url: str = Field(...)
    audio_url: str | None = Field(None)


class Reference2VideoInputField(BaseModel):
    prompt: str = Field(...)
    negative_prompt: str | None = Field(None)
    reference_video_urls: list[str] = Field(...)


class Txt2ImageParametersField(BaseModel):
    size: str = Field(...)
    n: int = Field(1, description="Number of images to generate.")  # we support only value=1
    seed: int = Field(..., ge=0, le=2147483647)
    prompt_extend: bool = Field(True)
    watermark: bool = Field(False)


class Image2ImageParametersField(BaseModel):
    size: str | None = Field(None)
    n: int = Field(1, description="Number of images to generate.")  # we support only value=1
    seed: int = Field(..., ge=0, le=2147483647)
    watermark: bool = Field(False)


class Text2VideoParametersField(BaseModel):
    size: str = Field(...)
    seed: int = Field(..., ge=0, le=2147483647)
    duration: int = Field(5, ge=5, le=15)
    prompt_extend: bool = Field(True)
    watermark: bool = Field(False)
    audio: bool = Field(False, description="Whether to generate audio automatically.")
    shot_type: str = Field("single")


class Image2VideoParametersField(BaseModel):
    resolution: str = Field(...)
    seed: int = Field(..., ge=0, le=2147483647)
    duration: int = Field(5, ge=5, le=15)
    prompt_extend: bool = Field(True)
    watermark: bool = Field(False)
    audio: bool = Field(False, description="Whether to generate audio automatically.")
    shot_type: str = Field("single")


class Reference2VideoParametersField(BaseModel):
    size: str = Field(...)
    duration: int = Field(5, ge=5, le=15)
    shot_type: str = Field("single")
    seed: int = Field(..., ge=0, le=2147483647)
    watermark: bool = Field(False)


class Text2ImageTaskCreationRequest(BaseModel):
    model: str = Field(...)
    input: Text2ImageInputField = Field(...)
    parameters: Txt2ImageParametersField = Field(...)


class Image2ImageTaskCreationRequest(BaseModel):
    model: str = Field(...)
    input: Image2ImageInputField = Field(...)
    parameters: Image2ImageParametersField = Field(...)


class Text2VideoTaskCreationRequest(BaseModel):
    model: str = Field(...)
    input: Text2VideoInputField = Field(...)
    parameters: Text2VideoParametersField = Field(...)


class Image2VideoTaskCreationRequest(BaseModel):
    model: str = Field(...)
    input: Image2VideoInputField = Field(...)
    parameters: Image2VideoParametersField = Field(...)


class Reference2VideoTaskCreationRequest(BaseModel):
    model: str = Field(...)
    input: Reference2VideoInputField = Field(...)
    parameters: Reference2VideoParametersField = Field(...)


class Wan27MediaItem(BaseModel):
    type: str = Field(...)
    url: str = Field(...)


class Wan27ReferenceVideoInputField(BaseModel):
    prompt: str = Field(...)
    negative_prompt: str | None = Field(None)
    media: list[Wan27MediaItem] = Field(...)


class Wan27ReferenceVideoParametersField(BaseModel):
    resolution: str = Field(...)
    ratio: str | None = Field(None)
    duration: int = Field(5, ge=2, le=15)
    watermark: bool = Field(False)
    seed: int = Field(..., ge=0, le=2147483647)


class Wan27ReferenceVideoTaskCreationRequest(BaseModel):
    model: str = Field(...)
    input: Wan27ReferenceVideoInputField = Field(...)
    parameters: Wan27ReferenceVideoParametersField = Field(...)


class Wan27ImageToVideoInputField(BaseModel):
    prompt: str | None = Field(None)
    negative_prompt: str | None = Field(None)
    media: list[Wan27MediaItem] = Field(...)


class Wan27ImageToVideoParametersField(BaseModel):
    resolution: str = Field(...)
    duration: int = Field(5, ge=2, le=15)
    prompt_extend: bool = Field(True)
    watermark: bool = Field(False)
    seed: int = Field(..., ge=0, le=2147483647)


class Wan27ImageToVideoTaskCreationRequest(BaseModel):
    model: str = Field(...)
    input: Wan27ImageToVideoInputField = Field(...)
    parameters: Wan27ImageToVideoParametersField = Field(...)


class Wan27VideoEditInputField(BaseModel):
    prompt: str = Field(...)
    media: list[Wan27MediaItem] = Field(...)


class Wan27VideoEditParametersField(BaseModel):
    resolution: str = Field(...)
    ratio: str | None = Field(None)
    duration: int | None = Field(0)
    audio_setting: str = Field("auto")
    watermark: bool = Field(False)
    seed: int = Field(..., ge=0, le=2147483647)


class Wan27VideoEditTaskCreationRequest(BaseModel):
    model: str = Field(...)
    input: Wan27VideoEditInputField = Field(...)
    parameters: Wan27VideoEditParametersField = Field(...)


class Wan27Text2VideoParametersField(BaseModel):
    resolution: str = Field(...)
    ratio: str | None = Field(None)
    duration: int = Field(5, ge=2, le=15)
    prompt_extend: bool = Field(True)
    watermark: bool = Field(False)
    seed: int = Field(..., ge=0, le=2147483647)


class Wan27Text2VideoTaskCreationRequest(BaseModel):
    model: str = Field(...)
    input: Text2VideoInputField = Field(...)
    parameters: Wan27Text2VideoParametersField = Field(...)


class TaskCreationOutputField(BaseModel):
    task_id: str = Field(...)
    task_status: str = Field(...)


class TaskCreationResponse(BaseModel):
    output: TaskCreationOutputField | None = Field(None)
    request_id: str = Field(...)
    code: str | None = Field(None, description="Error code for the failed request.")
    message: str | None = Field(None, description="Details about the failed request.")


class TaskResult(BaseModel):
    url: str | None = Field(None)
    code: str | None = Field(None)
    message: str | None = Field(None)


class ImageTaskStatusOutputField(TaskCreationOutputField):
    task_id: str = Field(...)
    task_status: str = Field(...)
    results: list[TaskResult] | None = Field(None)


class VideoTaskStatusOutputField(TaskCreationOutputField):
    task_id: str = Field(...)
    task_status: str = Field(...)
    video_url: str | None = Field(None)
    code: str | None = Field(None)
    message: str | None = Field(None)


class ImageTaskStatusResponse(BaseModel):
    output: ImageTaskStatusOutputField | None = Field(None)
    request_id: str = Field(...)


class VideoTaskStatusResponse(BaseModel):
    output: VideoTaskStatusOutputField | None = Field(None)
    request_id: str = Field(...)
