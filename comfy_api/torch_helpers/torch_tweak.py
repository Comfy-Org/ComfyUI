from __future__ import annotations

import torch

import comfy.model_management
import comfy.utils
from comfy.patcher_extension import CallbacksMP, WrappersMP
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher
    from comfy.patcher_extension import WrapperExecutor


TORCH_TWEAK_KEY = "torch_tweak"


class _DiffusionModelCall(torch.nn.Module):
    '''
    Calls the diffusion model with tensors only, so every Torch Tweak backend can trace and replay it.
    '''
    def __init__(self, diffusion_model: torch.nn.Module):
        super().__init__()
        self.diffusion_model = diffusion_model

    def forward(self, x, timesteps, context, **kwargs):
        if not torch.compiler.is_exporting():
            return self.diffusion_model(x, timesteps, context=context, transformer_options={}, **kwargs)
        # Exporters cannot convert comfy_kitchen kernels. in_training selects their plain torch versions.
        in_training = comfy.model_management.in_training
        comfy.model_management.in_training = True
        try:
            return self.diffusion_model(x, timesteps, context=context, transformer_options={}, **kwargs)
        finally:
            comfy.model_management.in_training = in_training


class _TunedDiffusionModel(torch.nn.Module):
    '''
    Stands in for the diffusion model during apply_model.
    Plain calls go through Torch Tweak. Calls with control, patches or non-tensor inputs run the original model.
    '''
    def __init__(self, diffusion_model: torch.nn.Module, tuned):
        super().__init__()
        self.__dict__["_original"] = diffusion_model
        self.__dict__["_tuned"] = tuned

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.__dict__["_original"], name)

    def forward(self, x, timesteps, context=None, control=None, transformer_options={}, **kwargs):
        patched = transformer_options.get("patches") or transformer_options.get("patches_replace")
        if control is not None or patched or not all(torch.is_tensor(v) for v in kwargs.values()):
            return self._original(x, timesteps, context=context, control=control, transformer_options=transformer_options, **kwargs)
        return self._tuned(x, timesteps, context, **kwargs)


def _xpu_strategy():
    from torch_tweak.torch import HighestThroughputStrategy
    from torch_tweak.torch.backend import (
        OpenVINOBackend,
        OpenVINOBackendConfig,
        TorchEagerBackend,
        TorchInductorBackend,
        XPUGraphBackend,
    )

    strategy = HighestThroughputStrategy(
        backends=[
            XPUGraphBackend(),
            TorchInductorBackend(),
            OpenVINOBackend(OpenVINOBackendConfig(offload_module=False, reuse_ir=True)),
            TorchEagerBackend(),
        ]
    )
    strategy.enable_find_max_batch_size(False)
    return strategy


def set_torch_tweak_wrapper(model: ModelPatcher):
    '''
    Serve the diffusion model of a ModelPatcher through Torch Tweak at sample time.

    Torch Tweak tunes each new input shape on its first step: it measures XPU graphs, Torch Inductor, OpenVINO and eager,
    and keeps the fastest. Later steps with that shape use it.
    Tuned shapes are dropped whenever the weights are loaded again, since LoRA patches and offloading change them.
    The selection of every shape is saved under the Torch Tweak cache directory, keyed by the weights.
    A later run with the same weights builds the saved backend and skips tuning.
    '''
    try:
        from torch_tweak.torch import OnlineTunedModule
    except ImportError as e:
        raise RuntimeError("TorchTweakModel needs the torch-tweak package: pip install torch-tweak") from e

    model.remove_wrappers_with_key(WrappersMP.APPLY_MODEL, TORCH_TWEAK_KEY)
    model.remove_callbacks_with_key(CallbacksMP.ON_LOAD, TORCH_TWEAK_KEY)
    tuned: dict[str, OnlineTunedModule] = {}

    def apply_torch_tweak_wrapper(executor: WrapperExecutor, *args, **kwargs):
        base_model = executor.class_obj
        if base_model.model_lowvram:
            return executor(*args, **kwargs)
        diffusion_model = base_model.diffusion_model
        if "model" not in tuned:
            strategy = _xpu_strategy() if comfy.model_management.is_intel_xpu() else None
            tuned["model"] = OnlineTunedModule(
                _DiffusionModelCall(diffusion_model), f"comfyui_{type(diffusion_model).__name__}", strategy=strategy
            )
        comfy.utils.set_attr(base_model, "diffusion_model", _TunedDiffusionModel(diffusion_model, tuned["model"]))
        try:
            return executor(*args, **kwargs)
        finally:
            comfy.utils.set_attr(base_model, "diffusion_model", diffusion_model)

    def drop_tuned_shapes(*args, **kwargs):
        if "model" in tuned:
            tuned.pop("model").reset()

    model.add_wrapper_with_key(WrappersMP.APPLY_MODEL, TORCH_TWEAK_KEY, apply_torch_tweak_wrapper)
    model.add_callback_with_key(CallbacksMP.ON_LOAD, TORCH_TWEAK_KEY, drop_tuned_shapes)
