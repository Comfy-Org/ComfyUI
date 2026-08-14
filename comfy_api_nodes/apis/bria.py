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


class BriaGenFillRequest(BaseModel):
    image: str = Field(...)
    mask: str = Field(
        ...,
        description="Binary mask defining the region to fill: white (255) pixels are generated, "
        "black (0) pixels are preserved. Must have the same aspect ratio as the image.",
    )
    prompt: str = Field(...)
    negative_prompt: str | None = Field(None)
    refine_prompt: bool = Field(True)
    seed: int = Field(...)
    prompt_content_moderation: bool = Field(False, description="If true, returns 422 on prompt moderation failure.")
    visual_input_content_moderation: bool = Field(
        False, description="If true, returns 422 on image or mask moderation failure."
    )
    visual_output_content_moderation: bool = Field(
        False, description="If true, returns 422 on visual output moderation failure."
    )


class BriaEraseRequest(BaseModel):
    image: str = Field(...)
    mask: str = Field(
        ...,
        description="Binary mask defining the region to erase: white (255) pixels are removed, "
        "black (0) pixels are preserved. Must have the same aspect ratio as the image.",
    )
    mask_type: str = Field("manual", description="'manual' for hand-drawn masks, 'automatic' for segmentation masks.")
    visual_input_content_moderation: bool = Field(
        False, description="If true, returns 422 on image or mask moderation failure."
    )
    visual_output_content_moderation: bool = Field(
        False, description="If true, returns 422 on visual output moderation failure."
    )


class BriaExpandRequest(BaseModel):
    image: str = Field(...)
    aspect_ratio: str | float | None = Field(
        None,
        description="Target ratio: a preset string (1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9) "
        "or a float between 0.5 and 3.0. When set, the canvas/placement fields are ignored.",
    )
    canvas_size: list[int] | None = Field(None, description="Output canvas [width, height]; area up to 5000x5000.")
    original_image_size: list[int] | None = Field(
        None, description="Size [width, height] of the original image inside the canvas."
    )
    original_image_location: list[int] | None = Field(
        None,
        description="Top-left corner [x, y] of the original image inside the canvas; "
        "values may fall outside the canvas, cropping the image.",
    )
    prompt: str | None = Field(None, description="If omitted, Bria auto-generates a prompt from the image.")
    negative_prompt: str | None = Field(None)
    seed: int = Field(...)
    prompt_content_moderation: bool = Field(False, description="If true, returns 422 on prompt moderation failure.")
    visual_input_content_moderation: bool = Field(
        False, description="If true, returns 422 on image moderation failure."
    )
    visual_output_content_moderation: bool = Field(
        False, description="If true, returns 422 on visual output moderation failure."
    )


class BriaIncreaseResolutionRequest(BaseModel):
    image: str = Field(...)
    desired_increase: int = Field(..., description="Resolution multiplier, 2 or 4.")
    visual_input_content_moderation: bool = Field(
        False, description="If true, returns 422 on image moderation failure."
    )
    visual_output_content_moderation: bool = Field(
        False, description="If true, returns 422 on visual output moderation failure."
    )


class BriaStatusResponse(BaseModel):
    request_id: str = Field(...)
    status_url: str = Field(...)
    warning: str | None = Field(None)


class BriaRemoveBackgroundResult(BaseModel):
    image_url: str = Field(...)


class BriaRemoveBackgroundResponse(BaseModel):
    status: str = Field(...)
    result: BriaRemoveBackgroundResult | None = Field(None)


class BriaImageResult(BaseModel):
    image_url: str = Field(...)


class BriaImageResultResponse(BaseModel):
    status: str = Field(...)
    result: BriaImageResult | None = Field(None)


class BriaExpandResult(BaseModel):
    image_url: str = Field(...)
    prompt: str | None = Field(None)
    seed: int | None = Field(None)


class BriaExpandResponse(BaseModel):
    status: str = Field(...)
    result: BriaExpandResult | None = Field(None)


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
