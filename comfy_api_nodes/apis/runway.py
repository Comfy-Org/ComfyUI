from enum import Enum
from typing import Optional, List, Union
from datetime import datetime

from pydantic import BaseModel, Field, RootModel


class RunwayAspectRatioEnum(str, Enum):
    field_1280_720 = '1280:720'
    field_720_1280 = '720:1280'
    field_1104_832 = '1104:832'
    field_832_1104 = '832:1104'
    field_960_960 = '960:960'
    field_1584_672 = '1584:672'
    field_1280_768 = '1280:768'
    field_768_1280 = '768:1280'


class Position(str, Enum):
    first = 'first'
    last = 'last'


class RunwayPromptImageDetailedObject(BaseModel):
    position: Position = Field(
        ...,
        description="The position of the image in the output video. 'last' is currently supported for gen3a_turbo only.",
    )
    uri: str = Field(
        ..., description='A HTTPS URL or data URI containing an encoded image.'
    )


class RunwayPromptImageObject(
    RootModel[Union[str, List[RunwayPromptImageDetailedObject]]]
):
    root: Union[str, List[RunwayPromptImageDetailedObject]] = Field(
        ...,
        description='Image(s) to use for the video generation. Can be a single URI or an array of image objects with positions.',
    )


class RunwayModelEnum(str, Enum):
    gen4_turbo = 'gen4_turbo'
    gen3a_turbo = 'gen3a_turbo'


class RunwayDurationEnum(int, Enum):
    integer_5 = 5
    integer_10 = 10


class RunwayImageToVideoRequest(BaseModel):
    duration: RunwayDurationEnum
    model: RunwayModelEnum
    promptImage: RunwayPromptImageObject
    promptText: Optional[str] = Field(
        None, description='Text prompt for the generation', max_length=1000
    )
    ratio: RunwayAspectRatioEnum
    seed: int = Field(
        ..., description='Random seed for generation', ge=0, le=4294967295
    )


class RunwayImageToVideoResponse(BaseModel):
    id: Optional[str] = Field(None, description='Task ID')


class RunwayTaskStatusResponse(BaseModel):
    createdAt: datetime = Field(..., description='Task creation timestamp')
    id: str = Field(..., description='Task ID')
    output: Optional[List[str]] = Field(None, description='Array of output video URLs')
    progress: Optional[float] = Field(
        None,
        description='Float value between 0 and 1 representing the progress of the task. Only available if status is RUNNING.',
        ge=0.0,
        le=1.0,
    )
    status: str = Field(..., description="SUCCEEDED, RUNNING, FAILED, PENDING, CANCELLED or THROTTLED")


class Model4(str, Enum):
    gen4_image = 'gen4_image'


class ReferenceImage(BaseModel):
    uri: Optional[str] = Field(
        None, description='A HTTPS URL or data URI containing an encoded image'
    )


class RunwayTextToImageAspectRatioEnum(str, Enum):
    field_1920_1080 = '1920:1080'
    field_1080_1920 = '1080:1920'
    field_1024_1024 = '1024:1024'
    field_1360_768 = '1360:768'
    field_1080_1080 = '1080:1080'
    field_1168_880 = '1168:880'
    field_1440_1080 = '1440:1080'
    field_1080_1440 = '1080:1440'
    field_1808_768 = '1808:768'
    field_2112_912 = '2112:912'


class RunwayTextToImageRequest(BaseModel):
    model: Model4 = Field(..., description='Model to use for generation')
    promptText: str = Field(
        ..., description='Text prompt for the image generation', max_length=1000
    )
    ratio: RunwayTextToImageAspectRatioEnum
    referenceImages: Optional[List[ReferenceImage]] = Field(
        None, description='Array of reference images to guide the generation'
    )


class RunwayTextToImageResponse(BaseModel):
    id: Optional[str] = Field(None, description='Task ID')


class RunwayAleph2IO:
    """Custom socket types for chaining Aleph2 guidance images."""

    KEYFRAME = "RUNWAY_ALEPH2_KEYFRAME"
    PROMPT_IMAGE = "RUNWAY_ALEPH2_PROMPT_IMAGE"


# Keyframe timing modes (anchored to the INPUT video). Stored on the chain item and used to
# choose the request model below. The values match the Aleph2 keyframe union field names.
KEYFRAME_MODE_SECONDS = "seconds"  # absolute time, in seconds, from the start of the input video
KEYFRAME_MODE_AT = "at"  # fraction [0.0, 1.0] of the input video duration

# Prompt-image position modes (anchored to the OUTPUT video). Values match the Aleph2 position `type`.
PROMPT_IMAGE_MODE_TIMESTAMP = "timestamp"  # absolute time, in seconds, from the start of the output video
PROMPT_IMAGE_MODE_POSITION = "position"  # fraction [0.0, 1.0] of the output video duration


class RunwayAleph2KeyframeItem:
    """A guidance image anchored to a point of the INPUT video (one Aleph2 ``keyframe``)."""

    def __init__(self, image, mode: str, value: float):
        self.image = image
        self.mode = mode  # KEYFRAME_MODE_SECONDS | KEYFRAME_MODE_AT
        self.value = value


class RunwayAleph2KeyframeChain:
    """An ordered collection of keyframes, built by chaining Runway Aleph2 Keyframe nodes."""

    def __init__(self):
        self.items: list[RunwayAleph2KeyframeItem] = []

    def add(self, item: RunwayAleph2KeyframeItem) -> None:
        self.items.append(item)

    def clone(self) -> "RunwayAleph2KeyframeChain":
        c = RunwayAleph2KeyframeChain()
        c.items = list(self.items)
        return c


class RunwayAleph2PromptImageItem:
    """A guidance image anchored to a point of the OUTPUT video (one Aleph2 ``promptImage``)."""

    def __init__(self, image, mode: str, value: float):
        self.image = image
        self.mode = mode  # PROMPT_IMAGE_MODE_TIMESTAMP | PROMPT_IMAGE_MODE_POSITION
        self.value = value


class RunwayAleph2PromptImageChain:
    """An ordered collection of prompt images, built by chaining Runway Aleph2 Prompt Image nodes."""

    def __init__(self):
        self.items: list[RunwayAleph2PromptImageItem] = []

    def add(self, item: RunwayAleph2PromptImageItem) -> None:
        self.items.append(item)

    def clone(self) -> "RunwayAleph2PromptImageChain":
        c = RunwayAleph2PromptImageChain()
        c.items = list(self.items)
        return c


class RunwayAleph2KeyframeSeconds(BaseModel):
    seconds: float = Field(
        ...,
        description="Absolute timestamp in seconds from the start of the input video when this guidance image should apply.",
        ge=0.0,
    )
    uri: str = Field(...)


class RunwayAleph2KeyframeAt(BaseModel):
    at: float = Field(
        ...,
        description="Position as a fraction [0.0, 1.0] of the input video duration.",
        ge=0.0,
        le=1.0,
    )
    uri: str = Field(...)


class RunwayAleph2TimestampPosition(BaseModel):
    type: str = Field(default="timestamp")
    timestampSeconds: float = Field(
        ...,
        description="Absolute timestamp in seconds from the start of the output video.",
        ge=0.0,
    )


class RunwayAleph2RelativePosition(BaseModel):
    type: str = Field(default="position")
    positionPercentage: float = Field(
        ...,
        description="Position as a fraction [0.0, 1.0] of the total output video duration.",
        ge=0.0,
        le=1.0,
    )


class RunwayAleph2PromptImage(BaseModel):
    position: RunwayAleph2TimestampPosition | RunwayAleph2RelativePosition
    uri: str = Field(...)


class RunwayAleph2ContentModeration(BaseModel):
    publicFigureThreshold: str = Field(
        ...,
        description='When set to "low", the content moderation system is less strict about '
        'recognizable public figures. One of "auto" or "low".',
    )


class RunwayAleph2Request(BaseModel):
    model: str = Field(default="aleph2")
    promptText: str = Field(
        ...,
        description="A non-empty string describing what should appear in the output.",
        min_length=1,
        max_length=1000,
    )
    videoUri: str = Field(...)
    seed: int = Field(..., description="Random seed for generation", ge=0, le=4294967295)
    contentModeration: RunwayAleph2ContentModeration = Field(...)
    keyframes: list[RunwayAleph2KeyframeSeconds | RunwayAleph2KeyframeAt] | None = Field(
        None,
        description="Timed guidance images placed at specific points in the input video. Up to 5.",
    )
    promptImage: list[RunwayAleph2PromptImage] | None = Field(
        None,
        description="Up to 5 image keyframes for guiding the edit at specific points in the output video.",
    )


class RunwayAleph2Response(BaseModel):
    id: str | None = Field(None, description="Task ID")
