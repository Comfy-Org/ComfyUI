from comfy_api.latest import (
    ComfyAPI_latest,
    Input as Input_latest,
    InputImpl as InputImpl_latest,
    Types as Types_latest,
)
from typing import Type, TYPE_CHECKING
from comfy_api.internal.async_to_sync import create_sync_class
from comfy_api.latest import io, ui, sdk, IO, UI, ComfyExtension  # noqa: F401


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
