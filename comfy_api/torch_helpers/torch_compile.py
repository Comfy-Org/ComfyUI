from __future__ import annotations
import torch

import comfy.utils
from comfy.patcher_extension import CallbacksMP, WrappersMP
from typing import TYPE_CHECKING, Callable, Optional
if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher
    from comfy.patcher_extension import WrapperExecutor


COMPILE_KEY = "torch.compile"
TORCH_COMPILE_KWARGS = "torch_compile_kwargs"
TORCH_COMPILE_KEYS = "torch_compile_keys"


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
    # clear out any other torch.compile wrappers
    model.remove_wrappers_with_key(WrappersMP.APPLY_MODEL, COMPILE_KEY)
    model.remove_callbacks_with_key(
        CallbacksMP.ON_DEEPCLONE_MULTIGPU, COMPILE_KEY)
    # if no keys, default to 'diffusion_model'
    if not keys:
        keys = ["diffusion_model"]
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
    model.model_options[TORCH_COMPILE_KEYS] = list(keys)

    def compile_multigpu_clone(_source, clone):
        refresh_torch_compile_wrapper(clone)

    model.add_callback_with_key(
        CallbacksMP.ON_DEEPCLONE_MULTIGPU,
        COMPILE_KEY,
        compile_multigpu_clone,
    )


def refresh_torch_compile_wrapper(model: ModelPatcher) -> None:
    compile_kwargs = model.model_options.get(TORCH_COMPILE_KWARGS)
    if compile_kwargs is None:
        return
    set_torch_compile_wrapper(
        model,
        keys=model.model_options.get(TORCH_COMPILE_KEYS),
        **compile_kwargs,
    )
