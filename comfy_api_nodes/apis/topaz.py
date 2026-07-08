from typing import Optional

from pydantic import BaseModel, Field


class ImageEnhanceRequest(BaseModel):
    model: str = Field("Reimagine")
    output_format: str = Field("jpeg")
    subject_detection: str = Field("All")
    face_enhancement: bool = Field(True)
    face_enhancement_creativity: float = Field(0, description="Is ignored if face_enhancement is false")
    face_enhancement_strength: float = Field(0.8, description="Is ignored if face_enhancement is false")
    source_url: str = Field(...)
    output_width: Optional[int] = Field(None)
    output_height: Optional[int] = Field(None)
    crop_to_fill: bool = Field(False)
    prompt: Optional[str] = Field(None, description="Text prompt for creative upscaling guidance")
    creativity: int = Field(3, description="Creativity settings range from 1 to 9")
    face_preservation: str = Field("true", description="To preserve the identity of characters")
    color_preservation: str = Field("true", description="To preserve the original color")


class ImageAsyncTaskResponse(BaseModel):
    process_id: str = Field(...)


class ImageStatusResponse(BaseModel):
    process_id: str = Field(...)
    status: str = Field(...)
    progress: Optional[int] = Field(None)
    credits: int = Field(...)


class ImageDownloadResponse(BaseModel):
    download_url: str = Field(...)
    expiry: int = Field(...)


class Resolution(BaseModel):
    width: int = Field(...)
    height: int = Field(...)


class CreateVideoRequestSource(BaseModel):
    container: str = Field(...)
    size: int = Field(..., description="Size of the video file in bytes")
    duration: int = Field(..., description="Duration of the video file in seconds")
    frameCount: int = Field(..., description="Total number of frames in the video")
    frameRate: int = Field(...)
    resolution: Resolution = Field(...)


class VideoFrameInterpolationFilter(BaseModel):
    model: str = Field(...)
    slowmo: Optional[int] = Field(None)
    fps: int = Field(...)
    duplicate: bool = Field(...)
    duplicate_threshold: float = Field(...)


class VideoEnhancementFilter(BaseModel):
    model: str = Field(...)
    auto: Optional[str] = Field(None, description="Auto, Manual, Relative")
    focusFixLevel: Optional[str] = Field(None, description="Downscales video input for correction of blurred subjects")
    compression: Optional[float] = Field(None, description="Strength of compression recovery")
    details: Optional[float] = Field(None, description="Amount of detail reconstruction")
    prenoise: Optional[float] = Field(None, description="Amount of noise to add to input to reduce over-smoothing")
    noise: Optional[float] = Field(None, description="Amount of noise reduction")
    halo: Optional[float] = Field(None, description="Amount of halo reduction")
    preblur: Optional[float] = Field(None, description="Anti-aliasing and deblurring strength")
    blur: Optional[float] = Field(None, description="Amount of sharpness applied")
    grain: Optional[float] = Field(None, description="Grain after AI model processing")
    grainSize: Optional[float] = Field(None, description="Size of generated grain")
    recoverOriginalDetailValue: Optional[float] = Field(None, description="Source details into the output video")
    creativity: float | str | None = Field(None, description="slc-1/slp-2.5: enum (low/middle/high). ast-2: decimal 0.0-1.0.")
    isOptimizedMode: Optional[bool] = Field(None, description="Set to true for Starlight Creative (slc-1) only")
    prompt: str | None = Field(None, description="Descriptive scene prompt (ast-2 only)")
    sharp: float | None = Field(None, description="ast-2 pre-enhance sharpness")
    realism: float | None = Field(None, description="ast-2 realism control")


class OutputInformationVideo(BaseModel):
    resolution: Resolution = Field(...)
    frameRate: int = Field(...)
    audioCodec: Optional[str] = Field(..., description="Required if audioTransfer is Copy or Convert")
    audioTransfer: str = Field(..., description="Copy, Convert, None")
    dynamicCompressionLevel: str = Field(..., description="Low, Mid, High")


class Overrides(BaseModel):
    isPaidDiffusion: bool = Field(True)


class CreateVideoRequest(BaseModel):
    source: CreateVideoRequestSource = Field(...)
    filters: list[VideoFrameInterpolationFilter | VideoEnhancementFilter] = Field(...)
    output: OutputInformationVideo = Field(...)
    overrides: Overrides = Field(Overrides(isPaidDiffusion=True))


class CreateVideoResponse(BaseModel):
    requestId: str = Field(...)


class VideoAcceptResponse(BaseModel):
    uploadId: str = Field(...)
    urls: list[str] = Field(...)


class VideoCompleteUploadRequestPart(BaseModel):
    partNum: int = Field(...)
    eTag: str = Field(...)


class VideoCompleteUploadRequest(BaseModel):
    uploadResults: list[VideoCompleteUploadRequestPart] = Field(...)


class VideoCompleteUploadResponse(BaseModel):
    message: str = Field(..., description="Confirmation message")


class VideoStatusResponseEstimates(BaseModel):
    cost: list[int] = Field(...)


class VideoStatusResponseDownloadUrl(BaseModel):
    url: str = Field(...)


class VideoStatusResponse(BaseModel):
    status: str = Field(...)
    estimates: Optional[VideoStatusResponseEstimates] = Field(None)
    progress: Optional[float] = Field(None)
    message: Optional[str] = Field("")
    download: Optional[VideoStatusResponseDownloadUrl] = Field(None)
