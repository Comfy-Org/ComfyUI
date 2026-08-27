from comfy_api.latest import (
    ComfyAPI_latest,
    Input as Input_latest,
    InputImpl as InputImpl_latest,
    Types as Types_latest,
)
import importlib
from typing import Type, TYPE_CHECKING
from comfy_api.internal.async_to_sync import create_sync_class
from comfy_api.latest import io, sdk, IO, ComfyExtension  # noqa: F401


class ComfyAPIAdapter_v0_0_3(ComfyAPI_latest):
    VERSION = "0.0.3"
    STABLE = False


class Input(Input_latest):
    pass


class InputImpl(InputImpl_latest):
    pass


class Types(Types_latest):
    pass


ComfyAPI = ComfyAPIAdapter_v0_0_3

# Create a synchronous version of the API
if TYPE_CHECKING:
    from comfy_api.v0_0_3.generated.ComfyAPISyncStub import ComfyAPISyncStub  # type: ignore

    ComfyAPISync: Type[ComfyAPISyncStub]
ComfyAPISync = create_sync_class(ComfyAPIAdapter_v0_0_3)


def __getattr__(name: str):
    if name in {"ui", "UI"}:
        latest = importlib.import_module("comfy_api.latest")
        module = getattr(latest, name)
        globals()["ui"] = module
        globals()["UI"] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ComfyAPI",
    "ComfyAPISync",
    "Input",
    "InputImpl",
    "Types",
    "ComfyExtension",
    "io",
    "IO",
    "ui",
    "UI",
    "sdk",
]
