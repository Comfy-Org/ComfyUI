from typing import TypedDict

from pydantic import BaseModel, Field


class InputModerationSettings(TypedDict):
    prompt_content_moderation: bool
    visual_input_moderation: bool
    visual_output_moderation: bool


class BriaEditImageRequest(BaseModel):
    instruction: str | None = Field(...)
    structured_instruction: str | None = Field(
        ...,
        description="Use this instead of instruction for precise, programmatic control.",
    )
    images: list[str] = Field(
        ...,
        description="Required. Publicly available URL or Base64-encoded. Must contain exactly one item.",
    )
    mask: str | None = Field(
        None,
        description="Mask image (black and white). Black areas will be preserved, white areas will be edited. "
        "If omitted, the edit applies to the entire image. "
        "The input image and the input mask must be of the same size.",
    )
    negative_prompt: str | None = Field(None)
    guidance_scale: float = Field(...)
    model_version: str = Field(...)
    steps_num: int = Field(...)
    seed: int = Field(...)
    ip_signal: bool = Field(
        False,
        description="If true, returns a warning for potential IP content in the instruction.",
    )
    prompt_content_moderation: bool = Field(
        False, description="If true, returns 422 on instruction moderation failure."
    )
    visual_input_content_moderation: bool = Field(
        False, description="If true, returns 422 on images or mask moderation failure."
    )
    visual_output_content_moderation: bool = Field(
        False, description="If true, returns 422 on visual output moderation failure."
    )


class BriaRemoveBackgroundRequest(BaseModel):
    image: str = Field(...)
    sync: bool = Field(False)
    visual_input_content_moderation: bool = Field(
        False, description="If true, returns 422 on input image moderation failure."
    )
    visual_output_content_moderation: bool = Field(
        False, description="If true, returns 422 on visual output moderation failure."
    )
    seed: int = Field(...)


class BriaStatusResponse(BaseModel):
    request_id: str = Field(...)
    status_url: str = Field(...)
    warning: str | None = Field(None)


class BriaRemoveBackgroundResult(BaseModel):
    image_url: str = Field(...)


class BriaRemoveBackgroundResponse(BaseModel):
    status: str = Field(...)
    result: BriaRemoveBackgroundResult | None = Field(None)


class BriaImageEditResult(BaseModel):
    structured_prompt: str = Field(...)
    image_url: str = Field(...)


class BriaImageEditResponse(BaseModel):
    status: str = Field(...)
    result: BriaImageEditResult | None = Field(None)


class BriaRemoveVideoBackgroundRequest(BaseModel):
    video: str = Field(...)
    background_color: str = Field(default="transparent", description="Background color for the output video.")
    output_container_and_codec: str = Field(...)
    preserve_audio: bool = Field(True)
    seed: int = Field(...)


class BriaRemoveVideoBackgroundResult(BaseModel):
    video_url: str = Field(...)


class BriaRemoveVideoBackgroundResponse(BaseModel):
    status: str = Field(...)
    result: BriaRemoveVideoBackgroundResult | None = Field(None)


class BriaVideoGreenScreenRequest(BaseModel):
    video: str = Field(..., description="Publicly accessible URL of the input video.")
    green_shade: str = Field(
        default="broadcast_green",
        description="Solid chroma-key shade applied behind the foreground "
        "(broadcast_green, chroma_green, or blue_screen).",
    )
    output_container_and_codec: str = Field(...)
    preserve_audio: bool = Field(True)
    seed: int = Field(...)


class BriaVideoReplaceBackgroundRequest(BaseModel):
    video: str = Field(..., description="Publicly accessible URL of the input (foreground) video.")
    background_url: str = Field(
        ...,
        description="Publicly accessible URL of the background image or video to composite behind "
        "the foreground. Stretched to the foreground frame; match its aspect ratio for "
        "undistorted results.",
    )
    output_container_and_codec: str = Field(...)
    preserve_audio: bool = Field(True)
    seed: int = Field(...)
