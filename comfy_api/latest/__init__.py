from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from comfy_api.internal import ComfyAPIBase
from comfy_api.internal.singleton import ProxiedSingleton
from comfy_api.internal.async_to_sync import create_sync_class
from ._input import ImageInput, AudioInput, MaskInput, LatentInput, VideoInput
from ._input_impl import VideoFromFile, VideoFromComponents, VideoFromList
from ._util import VideoCodec, VideoContainer, VideoComponents, MESH, VOXEL, SPLAT, File3D
from . import _io_public as io
from . import _ui_public as ui
from . import util
from ._engine import eng
from PIL import Image
import numpy as np


def _preview_tensor(image):
    """A preview as the IMAGE tensor ``eng.progress.update`` takes, from a PIL image or a tensor."""
    if image is None or isinstance(image, ImageInput):
        return image
    import torch
    return torch.from_numpy(np.array(image.convert("RGB"))).float().div(255)


class ComfyAPI_latest(ComfyAPIBase):
    VERSION = "latest"
    STABLE = False

    def __init__(self):
        super().__init__()
        self.node_replacement = self.NodeReplacement()
        self.execution = self.Execution()
        self.caching = self.Caching()

    class NodeReplacement(ProxiedSingleton):
        async def register(self, node_replace: io.NodeReplace) -> None:
            """Register a node replacement mapping."""
            from server import PromptServer
            PromptServer.instance.node_replace_manager.register(node_replace)

    class Execution(ProxiedSingleton):
        async def set_progress(
            self,
            value: float,
            max_value: float,
            node_id: str | None = None,
            preview_image: Image.Image | ImageInput | None = None,
            ignore_size_limit: bool = False,
        ) -> None:
            """
            Update the progress bar displayed in the ComfyUI interface.

            This function allows custom nodes and API calls to report their progress
            back to the user interface, providing visual feedback during long operations.

            Migration from previous API: comfy.utils.PROGRESS_BAR_HOOK
            """
            from . import _runtime
            runtime = _runtime._active_runtime.get()
            if runtime is not None and getattr(runtime.ops, "out_of_process", False):
                from ._engine import eng
                await eng.progress.update(value, max_value, preview=_preview_tensor(preview_image))
                return
            from comfy.cli_args import args
            from comfy_execution.progress import get_progress_state
            from comfy_execution.utils import get_executing_context

            executing_context = get_executing_context()
            if node_id is None and executing_context is not None:
                node_id = executing_context.node_id
            if node_id is None:
                raise ValueError("node_id must be provided if not in executing context")

            # Convert preview_image to PreviewImageTuple if needed
            to_display = preview_image
            if to_display is not None:
                # First convert to PIL Image if needed
                if isinstance(to_display, ImageInput):
                    # Convert ImageInput (torch.Tensor) to PIL Image
                    # Handle tensor shape [B, H, W, C] -> get first image if batch
                    tensor = to_display
                    if len(tensor.shape) == 4:
                        tensor = tensor[0]

                    # Convert to numpy array and scale to 0-255
                    image_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
                    to_display = Image.fromarray(image_np)

                if isinstance(to_display, Image.Image):
                    # Detect image format from PIL Image
                    image_format = to_display.format if to_display.format else "JPEG"
                    # Use None for preview_size if ignore_size_limit is True
                    preview_size = None if ignore_size_limit else args.preview_size
                    to_display = (image_format, to_display, preview_size)

            get_progress_state().update_progress(
                node_id=node_id,
                value=value,
                max_value=max_value,
                image=to_display,
            )

    class Caching(ProxiedSingleton):
        """
        External cache provider API for sharing cached node outputs
        across ComfyUI instances.

        Example::

            from comfy_api.latest import Caching

            class MyCacheProvider(Caching.CacheProvider):
                async def on_lookup(self, context):
                    ...  # check external storage

                async def on_store(self, context, value):
                    ...  # store to external storage

            Caching.register_provider(MyCacheProvider())
        """
        from ._caching import CacheProvider, CacheContext, CacheValue

        async def register_provider(self, provider: "ComfyAPI_latest.Caching.CacheProvider") -> None:
            """Register an external cache provider. Providers are called in registration order."""
            from comfy_execution.cache_provider import register_cache_provider
            register_cache_provider(provider)

        async def unregister_provider(self, provider: "ComfyAPI_latest.Caching.CacheProvider") -> None:
            """Unregister a previously registered cache provider."""
            from comfy_execution.cache_provider import unregister_cache_provider
            unregister_cache_provider(provider)

class ComfyExtension(ABC):
    async def on_load(self) -> None:
        """
        Called when an extension is loaded.
        This should be used to initialize any global resources needed by the extension.
        """

    @abstractmethod
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        """
        Returns a list of nodes that this extension provides.
        """

    async def get_samplers(self) -> dict | None:
        """``{name: fn}``: samplers added to ComfyUI's sampler list. ``fn`` is a sampling loop, as
        ``eng.sampler.custom`` takes: ``fn(model, x, sigmas, extra_args, callback=None, disable=None)``."""
        return None

    async def get_schedulers(self) -> dict | None:
        """``{name: fn}``: schedulers added to ComfyUI's scheduler list. ``fn(model_sampling, steps)``
        returns the sigmas; ``model_sampling`` is ``{"sigma_min", "sigma_max", "sigmas", "sigma",
        "timestep", "percent_to_sigma"}``, the last three functions that call the engine."""
        return None

    async def get_model_folders(self) -> dict | None:
        """``{folder: [extensions] or None}``: model folders the pack's files live in, such as
        ``{"onnx": ["onnx"]}``; ``io.ModelFolder`` lists them."""
        return None

    async def get_node_replacements(self) -> list | None:
        """``io.NodeReplace`` entries: what replaces node ids old workflows use."""
        return None

class Input:
    Image = ImageInput
    Audio = AudioInput
    Mask = MaskInput
    Latent = LatentInput
    Video = VideoInput

class InputImpl:
    VideoFromFile = VideoFromFile
    VideoFromComponents = VideoFromComponents
    VideoFromList = VideoFromList

class Types:
    VideoCodec = VideoCodec
    VideoContainer = VideoContainer
    VideoComponents = VideoComponents
    MESH = MESH
    VOXEL = VOXEL
    SPLAT = SPLAT
    File3D = File3D


Caching = ComfyAPI_latest.Caching

ComfyAPI = ComfyAPI_latest

# Create a synchronous version of the API
if TYPE_CHECKING:
    import comfy_api.latest.generated.ComfyAPISyncStub  # type: ignore

    ComfyAPISync: type[comfy_api.latest.generated.ComfyAPISyncStub.ComfyAPISyncStub]
ComfyAPISync = create_sync_class(ComfyAPI_latest)

# create new aliases for io and ui
IO = io
UI = ui

__all__ = [
    "ComfyAPI",
    "ComfyAPISync",
    "Input",
    "InputImpl",
    "Types",
    "Caching",
    "ComfyExtension",
    "eng",
    "io",
    "IO",
    "ui",
    "UI",
    "util",
]
