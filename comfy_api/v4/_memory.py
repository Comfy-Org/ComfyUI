from __future__ import annotations

from typing import TypedDict

import torch

from comfy_api.internal.singleton import ProxiedSingleton
from comfy_api.latest import io


class ModelLoadOptions(TypedDict, total=False):
    working_memory_bytes: int | float
    full_load: bool


class Memory(ProxiedSingleton):
    async def empty_cache(self) -> None:
        """Release memory PyTorch holds for this process but does not use; keep loaded models.

        Does nothing on the CPU.
        """
        from comfy import model_management

        model_management.soft_empty_cache()

    async def load_models(
        self,
        models: list[io.Model.Type | io.Clip.Type | io.Vae.Type],
        *,
        options: ModelLoadOptions | None = None,
    ) -> None:
        """Load each model onto its own load device, freeing memory as ComfyUI does to run them.

        Load devices default to `torch_device`, `vae_device` and `text_encoder_device` from
        `eng.device.info`. `working_memory_bytes` is the memory the work will need beside the
        models. `full_load` loads each model whole rather than partly, when ComfyUI would
        otherwise split it.
        """
        from comfy import model_management

        request = options or {}
        model_management.load_models_gpu(
            [getattr(model, "patcher", model) for model in models],
            memory_required=request.get("working_memory_bytes", 0),
            force_full_load=request.get("full_load", False),
        )

    async def unload_all(self) -> None:
        """Do nothing: ComfyUI alone decides which models stay loaded."""

    async def free_bytes(self, device: str | None = None) -> int:
        """Return ComfyUI's estimate of free bytes on `device`, or on its own device if None.

        On CUDA it is the driver's free memory plus PyTorch's unused cached blocks; on the CPU and
        MPS it is the system's available RAM. It reserves nothing
        and does not count loaded models that could be evicted. Raises `RuntimeError` for a string
        that is not a device name, and `KeyError` for a device this machine does not have, such as
        `"cuda"` on a Mac.
        """
        from comfy import model_management

        target = (
            torch.device(device)
            if device is not None
            else model_management.get_torch_device()
        )
        return int(model_management.get_free_memory(target))
