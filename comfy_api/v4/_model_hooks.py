from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, Protocol, TypedDict

import torch

from comfy_api.internal.singleton import ProxiedSingleton
from comfy_api.latest import io


class CfgArgs(TypedDict):
    """What a CFG function gets. `cond` and `uncond` are noise predictions, `input - *_denoised`."""

    cond: torch.Tensor
    uncond: torch.Tensor
    cond_denoised: torch.Tensor
    uncond_denoised: torch.Tensor
    cond_scale: float
    sigma: torch.Tensor
    input: torch.Tensor


class PreCfgArgs(TypedDict):
    """What a pre-CFG function gets: the denoised prediction of each prompt, positive first."""

    conds_out: list[torch.Tensor]
    cond_scale: float
    sigma: torch.Tensor
    input: torch.Tensor


class PostCfgArgs(TypedDict):
    """What a post-CFG function gets: `denoised` is the result so far."""

    denoised: torch.Tensor
    cond_denoised: torch.Tensor
    uncond_denoised: torch.Tensor
    cond_scale: float
    sigma: torch.Tensor
    input: torch.Tensor


class DenoiseMaskInfo(TypedDict):
    """`sigmas` is the whole schedule of the run."""

    sigmas: torch.Tensor


class UnetArgs(TypedDict):
    """What a model wrapper gets. `timestep` holds sigmas.

    `cond_or_uncond` says which prompt each chunk of the batch is, by its place in the guider's
    list of prompts: with KSampler, `eng.guider.cfg` and `eng.guider.basic`, 0 is positive and 1
    negative; other guiders, such as DualCFGGuider and PerpNegGuider, number their prompts their
    own way. `c` is ComfyUI's conditioning for the model, passed through as it is: only passing
    it on to `apply_model` as `**c` is part of the API.
    """

    input: torch.Tensor
    timestep: torch.Tensor
    c: dict[str, Any]
    cond_or_uncond: list[int]


class BlockInfo(TypedDict):
    """Where a UNet patch runs: `block` is `("input" | "middle" | "output", index)`.

    `sigmas` is the current sigma; `cond_or_uncond` says which prompt each chunk of the
    batch is, as in `UnetArgs`.
    """

    block: tuple[str, int]
    cond_or_uncond: list[int]
    sigmas: torch.Tensor


class AttentionInfo(BlockInfo):
    """`block_index` is the transformer in the block; `original_shape` is the model input's."""

    block_index: int
    original_shape: list[int]
    n_heads: int


class DenoiseMaskFunction(Protocol):
    def __call__(
        self,
        sigma: torch.Tensor,
        denoise_mask: torch.Tensor,
        extra_options: DenoiseMaskInfo,
    ) -> torch.Tensor: ...


class AttentionOverride(Protocol):
    """`options` are passed through as ComfyUI gave them; only passing them on is part of the API."""

    def __call__(
        self,
        attention: Callable[..., torch.Tensor],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        **options: Any,
    ) -> torch.Tensor: ...


ApplyModel = Callable[..., torch.Tensor]
AttentionPatch = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, AttentionInfo],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]


class _Hook:
    """What ComfyUI calls: `adapt` calls the pack's function with V4's arguments.

    `inner` is a model wrapper this one replaced; ComfyUI reaches it only through this hook.
    """

    def __init__(
        self, function: Callable[..., Any], adapt: Callable[..., Any], inner: Any = None
    ):
        self.function = function
        self.adapt = adapt
        self.inner = inner

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.adapt(self, *args, **kwargs)

    def to(self, device_or_dtype: torch.device | torch.dtype) -> _Hook:
        # ComfyUI moves UNet patches and the model wrapper with the model, so those may hold
        # weights; it does not move CFG or denoise-mask functions.
        if not hasattr(self.function, "to") and not hasattr(self.inner, "to"):
            return self
        function, inner = (
            part.to(device_or_dtype) if hasattr(part, "to") else part
            for part in (self.function, self.inner)
        )
        return _Hook(function, self.adapt, inner)

    def models(self) -> list[Any]:
        # ComfyUI loads the models a hook object names before sampling.
        return [
            model
            for part in (self.function, self.inner)
            if hasattr(part, "models")
            for model in part.models()
        ]

    def cleanup(self) -> None:
        # ComfyUI calls this on each hook object when sampling ends.
        for part in (self.function, self.inner):
            if hasattr(part, "cleanup"):
                part.cleanup()


def _pick(contract: Any, options: Mapping[str, Any]) -> Any:
    return {key: options[key] for key in contract.__required_keys__}


def _last_cut(contract: Any, hook: _Hook, *args: Any) -> Any:
    """Call the hook's `function` with ComfyUI's arguments, the last of them (a dict) cut down to `contract`."""
    *values, options = args
    return hook.function(*values, _pick(contract, options))


def _denoise_mask(
    hook: _Hook,
    sigma: torch.Tensor,
    denoise_mask: torch.Tensor,
    extra_options: dict[str, Any],
) -> torch.Tensor:
    return hook.function(
        sigma, denoise_mask, extra_options=_pick(DenoiseMaskInfo, extra_options)
    )


def _through(
    wrapper: Callable[..., torch.Tensor],
    apply_model: ApplyModel,
    cond_or_uncond: list[int],
    input: torch.Tensor,
    timestep: torch.Tensor,
    **c: Any,
) -> torch.Tensor:
    args = {
        "input": input,
        "timestep": timestep,
        "c": c,
        "cond_or_uncond": cond_or_uncond,
    }
    return wrapper(apply_model, args)


def _unet_wrapper(
    hook: _Hook, apply_model: ApplyModel, args: dict[str, Any]
) -> torch.Tensor:
    if hook.inner is not None:
        apply_model = partial(_through, hook.inner, apply_model, args["cond_or_uncond"])
    return hook.function(apply_model, _pick(UnetArgs, args))


# The options of ComfyUI's attention functions that a model may pass by position, in their order.
_ATTENTION_OPTIONS = ("mask", "attn_precision", "skip_reshape", "skip_output_reshape")


def _attention_override(
    function: AttentionOverride,
    previous: Callable[..., torch.Tensor] | None,
    func: Callable[..., torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    *positional: Any,
    **options: Any,
) -> torch.Tensor:
    attention = func if previous is None else partial(previous, func)
    options.update(zip(_ATTENTION_OPTIONS, positional, strict=False))
    return function(attention, q, k, v, heads, **options)


def _unet_patch(model: io.Model.Type, name: str, hook: _Hook) -> io.Model.Type:
    from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel

    patched = model.clone()
    if isinstance(patched.get_model_object("diffusion_model"), UNetModel):
        patched.set_model_patch(hook, name)
    else:
        logging.warning(
            f"{name} runs only on UNet models; it is not installed on this model."
        )
    return patched


class ModelHooks(ProxiedSingleton):
    async def set_cfg(
        self,
        model: io.Model.Type,
        function: Callable[[CfgArgs], torch.Tensor],
        *,
        disable_cfg1_optimization: bool = False,
    ) -> io.Model.Type:
        """Return a clone whose CFG step is `function(args)`, replacing any already set.

        `function` returns the guided noise prediction; stock CFG is
        `uncond + (cond - uncond) * cond_scale`. With KSampler or `eng.guider.cfg` at `cond_scale`
        1.0 the negative prompt is not run (`uncond_denoised` is zeros) unless
        `disable_cfg1_optimization` is set by this call, an earlier patch of the model, or the
        sampler, as the `*_cfg_pp` samplers do. A guider that does its own guidance decides this
        itself, and may not call `function`: PerpNegGuider does not.
        """
        patched = model.clone()
        patched.set_model_sampler_cfg_function(
            _Hook(function, partial(_last_cut, CfgArgs)), disable_cfg1_optimization
        )
        return patched

    async def set_pre_cfg(
        self,
        model: io.Model.Type,
        function: Callable[[PreCfgArgs], list[torch.Tensor]],
        *,
        disable_cfg1_optimization: bool = False,
    ) -> io.Model.Type:
        """Return a clone that runs `function(args)` before CFG; it returns the new `conds_out`.

        `conds_out` holds the denoised predictions, positive first. Functions run in the order
        added. With KSampler or `eng.guider.cfg` at `cond_scale` 1.0 `conds_out[1]` is zeros unless
        `disable_cfg1_optimization` is set by this call, an earlier patch of the model, or the
        sampler, as the `*_cfg_pp` samplers do. A guider that does its own guidance decides this
        itself, and may not run `function`: DualCFGGuider does not.
        """
        patched = model.clone()
        patched.set_model_sampler_pre_cfg_function(
            _Hook(function, partial(_last_cut, PreCfgArgs)), disable_cfg1_optimization
        )
        return patched

    async def set_post_cfg(
        self,
        model: io.Model.Type,
        function: Callable[[PostCfgArgs], torch.Tensor],
        *,
        disable_cfg1_optimization: bool = False,
    ) -> io.Model.Type:
        """Return a clone that runs `function(args)` after CFG; it returns the new `denoised`.

        Functions run in the order added. With KSampler or `eng.guider.cfg` at `cond_scale` 1.0 the
        negative prompt is not run (`uncond_denoised` is zeros) unless `disable_cfg1_optimization`
        is set by this call, an earlier patch of the model, or the sampler, as the `*_cfg_pp`
        samplers do. A guider that does its own guidance decides this itself.
        """
        patched = model.clone()
        patched.set_model_sampler_post_cfg_function(
            _Hook(function, partial(_last_cut, PostCfgArgs)), disable_cfg1_optimization
        )
        return patched

    async def set_denoise_mask(
        self, model: io.Model.Type, function: DenoiseMaskFunction
    ) -> io.Model.Type:
        """Return a clone whose inpaint mask is `function(sigma, denoise_mask, extra_options=...)`.

        It replaces any mask function already set, and runs only when the latent has a noise
        mask, at each model evaluation: more than once a step for samplers such as Heun.
        `extra_options["sigmas"]` is the whole schedule.
        """
        patched = model.clone()
        patched.set_model_denoise_mask_function(_Hook(function, _denoise_mask))
        return patched

    async def set_unet_wrapper(
        self,
        model: io.Model.Type,
        function: Callable[[ApplyModel, UnetArgs], torch.Tensor],
    ) -> io.Model.Type:
        """Return a clone whose model calls go through `function(apply_model, args)`.

        `function` returns the denoised prediction for the batch, usually
        `apply_model(args["input"], args["timestep"], **args["c"])`; `timestep` holds sigmas. A
        wrapper already on the model runs inside, as `apply_model`.
        """
        patched = model.clone()
        inner = patched.model_options.get("model_function_wrapper")
        patched.set_model_unet_function_wrapper(_Hook(function, _unet_wrapper, inner))
        return patched

    async def set_attn1_patch(
        self, model: io.Model.Type, function: AttentionPatch
    ) -> io.Model.Type:
        """Return a clone that runs `function(x, context, value, extra_options)` before each self-attention.

        It returns the three inputs of the q, k and v projections. Functions run in the order
        added, each on the previous one's result. Only UNet models (such as SD1.x, SD2.x, SDXL
        and SVD) run it; on other models it is not installed, and a warning is logged.
        """
        return _unet_patch(
            model, "attn1_patch", _Hook(function, partial(_last_cut, AttentionInfo))
        )

    async def set_attn2_patch(
        self, model: io.Model.Type, function: AttentionPatch
    ) -> io.Model.Type:
        """Return a clone that runs `function(x, context, value, extra_options)` before each cross-attention.

        It returns the three inputs of the q, k and v projections. Functions run in the order
        added, each on the previous one's result. Only UNet models (such as SD1.x, SD2.x, SDXL
        and SVD) run it; on other models it is not installed, and a warning is logged.
        """
        return _unet_patch(
            model, "attn2_patch", _Hook(function, partial(_last_cut, AttentionInfo))
        )

    async def set_attn1_output_patch(
        self,
        model: io.Model.Type,
        function: Callable[[torch.Tensor, AttentionInfo], torch.Tensor],
    ) -> io.Model.Type:
        """Return a clone that runs `function(out, extra_options)` on each self-attention output.

        Functions run in the order added, each on the previous one's result. Only UNet models
        (such as SD1.x, SD2.x, SDXL and SVD) run it; on other models it is not installed,
        and a warning is logged.
        """
        return _unet_patch(
            model,
            "attn1_output_patch",
            _Hook(function, partial(_last_cut, AttentionInfo)),
        )

    async def set_attn2_output_patch(
        self,
        model: io.Model.Type,
        function: Callable[[torch.Tensor, AttentionInfo], torch.Tensor],
    ) -> io.Model.Type:
        """Return a clone that runs `function(out, extra_options)` on each cross-attention output.

        Functions run in the order added, each on the previous one's result. Only UNet models
        (such as SD1.x, SD2.x, SDXL and SVD) run it; on other models it is not installed,
        and a warning is logged.
        """
        return _unet_patch(
            model,
            "attn2_output_patch",
            _Hook(function, partial(_last_cut, AttentionInfo)),
        )

    async def set_input_block_patch(
        self,
        model: io.Model.Type,
        function: Callable[[torch.Tensor, BlockInfo], torch.Tensor],
    ) -> io.Model.Type:
        """Return a clone that runs `function(h, extra_options)` on each UNet input block's output.

        Functions run in the order added, each on the previous one's result. Only UNet models
        (such as SD1.x, SD2.x, SDXL and SVD) run it; on other models it is not installed,
        and a warning is logged.
        """
        return _unet_patch(
            model, "input_block_patch", _Hook(function, partial(_last_cut, BlockInfo))
        )

    async def set_output_block_patch(
        self,
        model: io.Model.Type,
        function: Callable[
            [torch.Tensor, torch.Tensor, BlockInfo], tuple[torch.Tensor, torch.Tensor]
        ],
    ) -> io.Model.Type:
        """Return a clone that runs `function(h, skip, extra_options)` before each UNet output block.

        It returns `(h, skip)`; `skip` is about to be concatenated to `h`. Functions run in the
        order added, each on the previous one's result. Only UNet models (such as SD1.x, SD2.x,
        SDXL and SVD) run it; on other models it is not installed, and a warning is logged.
        """
        return _unet_patch(
            model, "output_block_patch", _Hook(function, partial(_last_cut, BlockInfo))
        )

    async def set_attention_override(
        self, model: io.Model.Type, function: AttentionOverride
    ) -> io.Model.Type:
        """Return a clone whose attention goes through `function(attention, q, k, v, heads, **options)`.

        `options` are the options ComfyUI passed, always as keywords here, which differ by model
        (such as `mask`, `attn_precision`, `skip_reshape`, `transformer_options`): pass them on to
        `attention` and do not rely on any one being present. An override already on the model
        runs inside, as `attention`. `function` is called only where the model's code passes ComfyUI's
        transformer options to attention, which not every model does (PixArt does not); elsewhere
        nothing calls it and nothing is logged.
        """
        patched = model.clone()
        options = patched.model_options["transformer_options"]
        options["optimized_attention_override"] = partial(
            _attention_override, function, options.get("optimized_attention_override")
        )
        return patched
