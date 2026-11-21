from __future__ import annotations
import torch

import comfy.utils
import comfy.quant_ops
from comfy.patcher_extension import WrappersMP
from typing import TYPE_CHECKING, Callable, Optional
if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher
    from comfy.patcher_extension import WrapperExecutor


COMPILE_KEY = "torch.compile"
TORCH_COMPILE_KWARGS = "torch_compile_kwargs"


def check_torch_compile_compatible(model: ModelPatcher, keys: list[str]):
    '''
    Fail fast when installing a torch.compile wrapper on modules with quantized weights.

    QuantizedTensor kernels (weight requantize, input quantize, quantized GEMM) are plain
    custom kernels, not opaque custom ops, so on CUDA devices Dynamo cannot trace them:
    torch.compile would install successfully but fail with a torch._dynamo tracing error
    on the first sampling step. Raising here makes the failure visible when the wrapper is
    installed instead of silently deferring it. On non-CUDA devices the kernels are pure
    torch ops that Dynamo can trace, so quantized weights are allowed there.
    '''
    load_device = getattr(model, "load_device", None)
    if load_device is None or load_device.type != "cuda":
        return
    for key in keys:
        module = model.get_model_object(key)
        quantized = [name for name, param in module.named_parameters() if isinstance(param, comfy.quant_ops.QuantizedTensor)]
        if quantized:
            examples = ", ".join(f"'{key}.{name}'" for name in quantized[:3])
            more = f" (and {len(quantized) - 3} more)" if len(quantized) > 3 else ""
            raise ValueError(
                f"torch.compile cannot be applied to '{key}': it has quantized (QuantizedTensor) "
                f"weights, e.g. {examples}{more}. Quantized weight kernels are not opaque to "
                "torch.compile on CUDA devices, so the wrapper would install successfully but fail "
                "with a torch._dynamo tracing error on the first sampling step. Do not apply "
                "torch.compile to this model."
            )


def apply_torch_compile_factory(compiled_module_dict: dict[str, Callable]) -> Callable:
    '''
    Create a wrapper that will refer to the compiled_diffusion_model.
    '''
    def apply_torch_compile_wrapper(executor: WrapperExecutor, *args, **kwargs):
        try:
            orig_modules = {}
            for key, value in compiled_module_dict.items():
                orig_modules[key] = comfy.utils.get_attr(executor.class_obj, key)
                comfy.utils.set_attr(executor.class_obj, key, value)
            return executor(*args, **kwargs)
        finally:
            for key, value in orig_modules.items():
                comfy.utils.set_attr(executor.class_obj, key, value)
    return apply_torch_compile_wrapper


def set_torch_compile_wrapper(model: ModelPatcher, backend: str, options: Optional[dict[str,str]]=None,
                              mode: Optional[str]=None, fullgraph=False, dynamic: Optional[bool]=None,
                              keys: list[str]=["diffusion_model"], *args, **kwargs):
    '''
    Perform torch.compile that will be applied at sample time for either the whole model or specific params of the BaseModel instance.

    When keys is None, it will default to using ["diffusion_model"], compiling the whole diffusion_model.
    When a list of keys is provided, it will perform torch.compile on only the selected modules.
    '''
    # if no keys, default to 'diffusion_model'
    if not keys:
        keys = ["diffusion_model"]
    # raise before touching any state if the modules cannot be compiled
    check_torch_compile_compatible(model, keys)
    # clear out any other torch.compile wrappers
    model.remove_wrappers_with_key(WrappersMP.APPLY_MODEL, COMPILE_KEY)
    # create kwargs dict that can be referenced later
    compile_kwargs = {
        "backend": backend,
        "options": options,
        "mode": mode,
        "fullgraph": fullgraph,
        "dynamic": dynamic,
    }
    # get a dict of compiled keys
    compiled_modules = {}
    for key in keys:
        compiled_modules[key] = torch.compile(
                model=model.get_model_object(key),
                **compile_kwargs,
            )
    # add torch.compile wrapper
    wrapper_func = apply_torch_compile_factory(
        compiled_module_dict=compiled_modules,
    )
    # store wrapper to run on BaseModel's apply_model function
    model.add_wrapper_with_key(WrappersMP.APPLY_MODEL, COMPILE_KEY, wrapper_func)
    # keep compile kwargs for reference
    model.model_options[TORCH_COMPILE_KWARGS] = compile_kwargs
