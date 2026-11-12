"""The V3 API under its numbered name.

`comfy_api.latest` is the same API and is deprecated. A pack moves here by
changing its import line.
"""

import sys
from importlib import import_module
from typing import TYPE_CHECKING

from comfy_api.internal.async_to_sync import create_sync_class
from comfy_api.latest import (
    IO,
    UI,
    Caching,
    ComfyAPI_latest,
    ComfyExtension,
    Input,
    InputImpl,
    Types,
    io,
    ui,
    ABC as ABC,
    AudioInput as AudioInput,
    ComfyAPIBase as ComfyAPIBase,
    File3D as File3D,
    Image as Image,
    ImageInput as ImageInput,
    LatentInput as LatentInput,
    MESH as MESH,
    MaskInput as MaskInput,
    PreviewImageTuple as PreviewImageTuple,
    ProxiedSingleton as ProxiedSingleton,
    SPLAT as SPLAT,
    VOXEL as VOXEL,
    VideoCodec as VideoCodec,
    VideoComponents as VideoComponents,
    VideoContainer as VideoContainer,
    VideoFromComponents as VideoFromComponents,
    VideoFromFile as VideoFromFile,
    VideoFromList as VideoFromList,
    VideoInput as VideoInput,
    _caching as _caching,
    _input as _input,
    _input_impl as _input_impl,
    _io as _io,
    _io_public as _io_public,
    _ui as _ui,
    _ui_public as _ui_public,
    _util as _util,
    abstractmethod as abstractmethod,
    args as args,
    get_executing_context as get_executing_context,
    get_progress_state as get_progress_state,
    np as np,
)


class ComfyAPI_v3(ComfyAPI_latest):
    VERSION = "3.0.0"
    STABLE = False


ComfyAPI = ComfyAPI_v3
if TYPE_CHECKING:
    from .generated.ComfyAPISyncStub import ComfyAPISyncStub

    ComfyAPISync: type[ComfyAPISyncStub]
ComfyAPISync = create_sync_class(ComfyAPI_v3)

for _name in (
    "_caching",
    "_input",
    "_input.basic_types",
    "_input.curve_types",
    "_input.range_types",
    "_input.video_types",
    "_input_impl",
    "_input_impl.video_types",
    "_io",
    "_io_public",
    "_ui",
    "_ui_public",
    "_util",
    "_util.geometry_types",
    "_util.image_types",
    "_util.video_types",
):
    sys.modules[f"{__name__}.{_name}"] = import_module(f"comfy_api.latest.{_name}")


__all__ = [
    "ComfyAPI",
    "ComfyAPISync",
    "Input",
    "InputImpl",
    "Types",
    "Caching",
    "ComfyExtension",
    "io",
    "IO",
    "ui",
    "UI",
]
