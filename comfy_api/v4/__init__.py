import re
from typing import TYPE_CHECKING

from comfy_api._exceptions import InterruptProcessingException

from comfy_api.internal import ComfyAPIBase
from comfy_api.latest import (
    Input,
    InputImpl,
    Types,
    io,
    ui,
)
from ._clip import (
    Clip,
    ClipEncoding,
    ClipEncodingResult,
    ClipInfo,
    ClipTokens,
    WeightedToken,
)
from ._device import Device, DeviceInfo
from ._execution import Execution
from ._extension import ComfyExtension, Route, RouteRequest, RouteResponse
from ._guider import Guider
from ._memory import Memory, ModelLoadOptions
from ._model import Model, ModelInfo
from ._model_hooks import (
    ApplyModel,
    AttentionInfo,
    AttentionOverride,
    AttentionPatch,
    BlockInfo,
    CfgArgs,
    DenoiseMaskFunction,
    DenoiseMaskInfo,
    PostCfgArgs,
    PreCfgArgs,
    UnetArgs,
)
from ._models import (
    CheckpointOptions,
    DiffusionModelOptions,
    ModelFile,
    Models,
    WeightDtype,
)
from ._node import Node, NodeRunOutput
from ._noise import Noise, NoiseGenerator
from ._progress import Progress
from ._sampler import (
    Denoiser,
    Sampler,
    SamplerArgs,
    SamplerFunction,
    SamplerStep,
)
from ._services import (
    CacheContext,
    CacheProvider,
    CacheValue,
    Caching,
    NodeReplacement,
)
from ._settings import Settings
from ._sync import SyncGroup
from ._ui import Ui
from ._vae import Vae, VaeInfo
from . import util


class ComfyAPI_v4(ComfyAPIBase):
    VERSION = "4.0.0"
    STABLE = False

    caching: Caching
    clip: Clip
    device: Device
    execution: Execution
    guider: Guider
    memory: Memory
    model: Model
    models: Models
    node: Node
    node_replacement: NodeReplacement
    noise: Noise
    progress: Progress
    sampler: Sampler
    settings: Settings
    ui: Ui
    vae: Vae

    def __init__(self):
        super().__init__()
        self.caching = Caching()
        self.clip = Clip()
        self.device = Device()
        self.execution = Execution()
        self.guider = Guider()
        self.memory = Memory()
        self.model = Model()
        self.models = Models()
        self.node = Node()
        self.node_replacement = NodeReplacement()
        self.noise = Noise()
        self.progress = Progress()
        self.sampler = Sampler()
        self.settings = Settings()
        self.ui = Ui()
        self.vae = Vae()

    def get_api_version(self) -> tuple[int, int, int]:
        """Return this package's (major, minor, patch) version."""
        major, minor, patch = self.VERSION.split(".")
        return int(major), int(minor), int(patch)

    def is_api_version_at_least(self, version: str) -> bool:
        """Return True if this package is `version` or newer.

        `version` is three numbers joined by dots, with or without a leading `v`, such as
        `v4.0.0` or `4.0.0`; anything else raises `ValueError`.
        """
        numbers = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", version, re.ASCII)
        if numbers is None:
            raise ValueError(f"{version!r} is not a version written like 'v4.0.0'")
        return self.get_api_version() >= tuple(map(int, numbers.groups()))


ComfyAPI = ComfyAPI_v4


class ComfyAPISync:
    """The API for nodes that are not async: the same calls, without `await`.

    Make one and keep it: each one made is a separate object, built in about a millisecond.
    """

    def __init__(self):
        api = ComfyAPI_v4()
        self.get_api_version = api.get_api_version
        self.is_api_version_at_least = api.is_api_version_at_least
        for name, declared in ComfyAPI_v4.__annotations__.items():
            setattr(self, name, SyncGroup(declared, getattr(api, name)))


IO = io
UI = ui

if TYPE_CHECKING:
    eng: ComfyAPI_v4


# Resolve eng when it is read, so the host can inject its instances until then;
# `from comfy_api.v4 import eng` reads it.
def __getattr__(name: str):
    if name == "eng":
        return ComfyAPI_v4()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ApplyModel",
    "AttentionInfo",
    "AttentionOverride",
    "AttentionPatch",
    "BlockInfo",
    "CacheContext",
    "CacheProvider",
    "CacheValue",
    "CfgArgs",
    "CheckpointOptions",
    "ClipEncoding",
    "ClipEncodingResult",
    "ClipInfo",
    "ClipTokens",
    "ComfyAPI",
    "ComfyAPISync",
    "ComfyExtension",
    "DenoiseMaskFunction",
    "DenoiseMaskInfo",
    "Denoiser",
    "DeviceInfo",
    "DiffusionModelOptions",
    "IO",
    "Input",
    "InputImpl",
    "InterruptProcessingException",
    "ModelFile",
    "ModelInfo",
    "ModelLoadOptions",
    "NodeRunOutput",
    "NoiseGenerator",
    "PostCfgArgs",
    "PreCfgArgs",
    "Route",
    "RouteRequest",
    "RouteResponse",
    "SamplerArgs",
    "SamplerFunction",
    "SamplerStep",
    "Types",
    "UI",
    "UnetArgs",
    "VaeInfo",
    "WeightDtype",
    "WeightedToken",
    "eng",
    "io",
    "ui",
    "util",
]
