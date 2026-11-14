from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import copy
from typing import TypedDict, TypeVar

import torch

from comfy_api.latest import io

from ._model_hooks import ModelHooks


_PatchKey = TypeVar("_PatchKey", bound=str | tuple[object, ...])


class ModelInfo(TypedDict):
    model_type: str
    latent_scale_factor: float
    latent_channels: int
    zsnr: bool
    dtype: str
    sigma_min: float
    sigma_max: float


class Model(ModelHooks):
    async def clone(self, model: io.Model.Type) -> io.Model.Type:
        """Copy model settings and patch registrations while sharing its weights."""
        return model.clone()

    async def add_patches(
        self,
        model: io.Model.Type,
        patches: Mapping[_PatchKey, Sequence[object]],
        strength_patch: float = 1.0,
        strength_model: float = 1.0,
    ) -> io.Model.Type:
        """Return a patched clone; leave the input model unchanged.

        Keys are names from `weight_keys`, or `(name, (dim, start, length))` to patch one slice.
        A one-item sequence holding a tensor of the weight's shape makes the weight
        `weight * strength_model + diff * strength_patch`, after any patches the model already
        has. Keys the model does not have are ignored; a difference of another shape is skipped
        with a logged warning.
        """
        patched = model.clone()
        patched.add_patches(
            {key: tuple(value) for key, value in patches.items()},
            strength_patch,
            strength_model,
        )
        return patched

    async def read_weight(self, model: io.Model.Type, key: str) -> torch.Tensor:
        """Read a weight with this model's patches applied, as ComfyUI computes it.

        Without patches the result shares the model's storage: clone it before changing it.
        While this model or a clone of it is loaded, the stored weight can already hold the
        loaded clone's patches, and this model's are applied on top of it. Raises
        `AttributeError` for a name not in `weight_keys`.
        """
        return model.patch_weight_to_device(key, return_weight=True).detach()

    async def load_lora(
        self,
        model: io.Model.Type | None,
        clip: io.Clip.Type | None,
        lora: dict[str, torch.Tensor],
        strength_model: float = 1.0,
        strength_clip: float = 1.0,
        lora_metadata: Mapping[str, object] | None = None,
    ) -> tuple[io.Model.Type | None, io.Clip.Type | None]:
        """Apply LoRA weights to clones of the model and text encoder; a None input gives None.

        `lora` holds a LoRA file's tensors, in any key format LoraLoader accepts; tensors that
        match neither input are skipped with a logged warning. A non-empty `lora_metadata` is
        attached to the resulting copies; the caller's weight and metadata mappings are not
        changed.
        """
        from comfy.sd import load_lora_for_models

        return load_lora_for_models(
            model,
            clip,
            copy(lora),
            strength_model,
            strength_clip,
            lora_metadata=dict(lora_metadata) if lora_metadata is not None else None,
        )

    async def latent_process_in(
        self, model: io.Model.Type, samples: torch.Tensor
    ) -> torch.Tensor:
        """Convert a latent from its VAE's range to the range the model samples in.

        Depending on the model this scales, shifts and scales, or normalizes each channel, and for a
        few it also changes the latent's shape; for some models it returns `samples` itself, so
        clone the result before changing it in place.
        """
        return model.get_model_object("latent_format").process_in(samples)

    async def latent_process_out(
        self, model: io.Model.Type, samples: torch.Tensor
    ) -> torch.Tensor:
        """Convert a latent from the range the model samples in back to its VAE's range.

        It undoes `latent_process_in`; for some models it returns `samples` itself, so clone the
        result before changing it in place.
        """
        return model.get_model_object("latent_format").process_out(samples)

    async def percent_to_sigma(self, model: io.Model.Type, percent: float) -> float:
        """Return the sigma at a fraction of sampling, as conditioning's `start_percent` uses it.

        0.0 is the start and gives a value at or above `sigma_max`; 1.0 is the end and gives 0.0.
        """
        sampling = model.get_model_object("model_sampling")
        return float(sampling.percent_to_sigma(percent))

    async def timestep(self, model: io.Model.Type, sigma: torch.Tensor) -> torch.Tensor:
        """Map sigma values to the timesteps the model is conditioned on, elementwise.

        The scale is the model's own: for example SD1.x and SDXL give the nearest integer step
        from 0 to 999, and Flux gives the sigma itself.
        """
        sampling = model.get_model_object("model_sampling")
        return sampling.timestep(sigma)

    async def sigma(self, model: io.Model.Type, timestep: torch.Tensor) -> torch.Tensor:
        """Map timesteps on the model's own scale (see `timestep`) to sigma values, elementwise.

        For SD1.x and SDXL, fractional steps are interpolated and steps outside 0 to 999 are
        clamped.
        """
        sampling = model.get_model_object("model_sampling")
        return sampling.sigma(timestep)

    async def info(self, model: io.Model.Type) -> ModelInfo:
        """Describe the model's prediction type, latent format and sigma range as plain data.

        `model_type` is the prediction type the model was loaded with, such as `"EPS"` or
        `"FLOW"`, not its architecture; nodes such as ModelSamplingDiscrete do not change it,
        though `zsnr` and the sigma range follow them. `dtype` is the diffusion model's weight
        dtype. `latent_scale_factor` is the format's scale alone; some formats also shift.
        """
        latent = model.get_model_object("latent_format")
        sampling = model.get_model_object("model_sampling")
        return {
            "model_type": model.get_model_object("model_type").name,
            "latent_scale_factor": latent.scale_factor,
            "latent_channels": latent.latent_channels,
            "zsnr": bool(getattr(sampling, "zsnr", False)),
            "dtype": str(model.model_dtype()).removeprefix("torch."),
            "sigma_min": float(sampling.sigma_min),
            "sigma_max": float(sampling.sigma_max),
        }

    async def weight_keys(self, model: io.Model.Type) -> list[str]:
        """List the model's state names, sorted: the names `add_patches` and `read_weight` take.

        Besides weights they include buffers such as `model_sampling.sigmas`.
        """
        return sorted(model.model_state_dict())
