from __future__ import annotations
import torch
import uuid
import math
import collections
import comfy.model_management
import comfy.conds
import comfy.model_patcher
import comfy.utils
import comfy.hooks
import comfy.patcher_extension
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from comfy.model_base import BaseModel
    from comfy.model_patcher import ModelPatcher
    from comfy.controlnet import ControlBase

def prepare_mask(noise_mask, shape, device):
    return comfy.utils.reshape_mask(noise_mask, shape).to(device)

def get_models_from_cond(cond, model_type):
    models = []
    for c in cond:
        if model_type in c:
            if isinstance(c[model_type], list):
                models += c[model_type]
            else:
                models += [c[model_type]]
    return models

def get_hooks_from_cond(cond, full_hooks: comfy.hooks.HookGroup):
    # get hooks from conds, and collect cnets so they can be checked for extra_hooks
    cnets: list[ControlBase] = []
    for c in cond:
        if 'hooks' in c:
            for hook in c['hooks'].hooks:
                full_hooks.add(hook)
        if 'control' in c:
            cnets.append(c['control'])

    def get_extra_hooks_from_cnet(cnet: ControlBase, _list: list):
        if cnet.extra_hooks is not None:
            _list.append(cnet.extra_hooks)
        if cnet.previous_controlnet is None:
            return _list
        return get_extra_hooks_from_cnet(cnet.previous_controlnet, _list)

    hooks_list = []
    cnets = set(cnets)
    for base_cnet in cnets:
        get_extra_hooks_from_cnet(base_cnet, hooks_list)
    extra_hooks = comfy.hooks.HookGroup.combine_all_hooks(hooks_list)
    if extra_hooks is not None:
        for hook in extra_hooks.hooks:
            full_hooks.add(hook)

    return full_hooks

def convert_cond(cond):
    out = []
    for c in cond:
        temp = c[1].copy()
        model_conds = temp.get("model_conds", {})
        if c[0] is not None:
            temp["cross_attn"] = c[0]
        temp["model_conds"] = model_conds
        temp["uuid"] = uuid.uuid4()
        out.append(temp)
    return out

def cond_has_hooks(cond):
    for c in cond:
        temp = c[1]
        if "hooks" in temp:
            return True
        if "control" in temp:
            control = temp["control"]
            extra_hooks = control.get_extra_hooks()
            if len(extra_hooks) > 0:
                return True
    return False

def get_additional_models(conds, dtype):
    """loads additional models in conditioning"""
    cnets: list[ControlBase] = []
    gligen = []
    add_models = []

    for k in conds:
        cnets += get_models_from_cond(conds[k], "control")
        gligen += get_models_from_cond(conds[k], "gligen")
        add_models += get_models_from_cond(conds[k], "additional_models")

    # Order-preserving dedup. A plain set() would randomize iteration order across runs
    control_nets = list(dict.fromkeys(cnets))

    inference_memory = 0
    control_models = []
    for m in control_nets:
        control_models += m.get_models()
        inference_memory += m.inference_memory_requirements(dtype)

    gligen = [x[1] for x in gligen]
    models = control_models + gligen + add_models

    return models, inference_memory

def get_additional_models_from_model_options(model_options: dict[str]=None):
    """loads additional models from registered AddModels hooks"""
    models = []
    if model_options is not None and "registered_hooks" in model_options:
        registered: comfy.hooks.HookGroup = model_options["registered_hooks"]
        for hook in registered.get_type(comfy.hooks.EnumHookType.AdditionalModels):
            hook: comfy.hooks.AdditionalModelsHook
            models.extend(hook.models)
    return models

def cleanup_additional_models(models):
    """cleanup additional models that were loaded"""
    for m in models:
        if hasattr(m, 'cleanup'):
            m.cleanup()

def preprocess_multigpu_conds(conds: dict[str, list[dict[str]]], model: ModelPatcher, model_options: dict[str]):
    '''If multigpu acceleration required, creates deepclones of ControlNets and GLIGEN per device.'''
    multigpu_models: list[ModelPatcher] = model.get_additional_models_with_key("multigpu")
    if len(multigpu_models) == 0:
        return
    extra_devices = [x.load_device for x in multigpu_models]
    # handle controlnets
    controlnets: set[ControlBase] = set()
    for k in conds:
        for kk in conds[k]:
            if 'control' in kk:
                controlnets.add(kk['control'])
    if len(controlnets) > 0:
        # first, unload all controlnet clones
        for cnet in list(controlnets):
            cnet_models = cnet.get_models()
            for cm in cnet_models:
                comfy.model_management.unload_model_and_clones(cm, unload_additional_models=True)

        # next, make sure each controlnet has a deepclone for all relevant devices
        for cnet in controlnets:
            curr_cnet = cnet
            while curr_cnet is not None:
                for device in extra_devices:
                    if device not in curr_cnet.multigpu_clones:
                        curr_cnet.deepclone_multigpu(device, autoregister=True)
                curr_cnet = curr_cnet.previous_controlnet
        # since all device clones are now present, recreate the linked list for cloned cnets per device
        for cnet in controlnets:
            curr_cnet = cnet
            while curr_cnet is not None:
                prev_cnet = curr_cnet.previous_controlnet
                for device in extra_devices:
                    device_cnet = curr_cnet.get_instance_for_device(device)
                    prev_device_cnet = None
                    if prev_cnet is not None:
                        prev_device_cnet = prev_cnet.get_instance_for_device(device)
                    device_cnet.set_previous_controlnet(prev_device_cnet)
                curr_cnet = prev_cnet
    # potentially handle gligen - since not widely used, ignored for now

def estimate_memory(model, noise_shape, conds):
    cond_shapes = collections.defaultdict(list)
    cond_shapes_min = {}
    for _, cs in conds.items():
        for cond in cs:
            for k, v in model.model.extra_conds_shapes(**cond).items():
                cond_shapes[k].append(v)
                if cond_shapes_min.get(k, None) is None:
                    cond_shapes_min[k] = [v]
                elif math.prod(v) > math.prod(cond_shapes_min[k][0]):
                    cond_shapes_min[k] = [v]

    memory_required = model.model.memory_required([noise_shape[0] * 2] + list(noise_shape[1:]), cond_shapes=cond_shapes)
    minimum_memory_required = model.model.memory_required([noise_shape[0]] + list(noise_shape[1:]), cond_shapes=cond_shapes_min)
    return memory_required, minimum_memory_required

def prepare_sampling(model: ModelPatcher, noise_shape, conds, model_options=None, force_full_load=False, force_offload=False):
    executor = comfy.patcher_extension.WrapperExecutor.new_executor(
        _prepare_sampling,
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING, model_options, is_model_options=True)
    )
    return executor.execute(model, noise_shape, conds, model_options=model_options, force_full_load=force_full_load, force_offload=force_offload)

def _prepare_sampling(model: ModelPatcher, noise_shape, conds, model_options=None, force_full_load=False, force_offload=False):
    model.match_multigpu_clones()
    preprocess_multigpu_conds(conds, model, model_options)
    models, inference_memory = get_additional_models(conds, model.model_dtype())
    models += get_additional_models_from_model_options(model_options)
    models += model.get_nested_additional_models()  # TODO: does this require inference_memory update?
    if force_offload: # In training + offload enabled, we want to force prepare sampling to trigger partial load
        memory_required = 1e20
        minimum_memory_required = None
    else:
        memory_required, minimum_memory_required = estimate_memory(model, noise_shape, conds)
        memory_required += inference_memory
        minimum_memory_required += inference_memory
    comfy.model_management.load_models_gpu([model] + models, memory_required=memory_required, minimum_memory_required=minimum_memory_required, force_full_load=force_full_load)
    real_model: BaseModel = model.model

    return real_model, conds, models

def cleanup_models(conds, models):
    cleanup_additional_models(models)

    control_cleanup = []
    for k in conds:
        control_cleanup += get_models_from_cond(conds[k], "control")

    cleanup_additional_models(set(control_cleanup))

def prepare_model_patcher(model: ModelPatcher, conds, model_options: dict):
    '''
    Registers hooks from conds.
    '''
    # check for hooks in conds - if not registered, see if can be applied
    hooks = comfy.hooks.HookGroup()
    for k in conds:
        get_hooks_from_cond(conds[k], hooks)
    # add wrappers and callbacks from ModelPatcher to transformer_options
    comfy.patcher_extension.merge_nested_dicts(model_options["transformer_options"].setdefault("wrappers", {}), model.wrappers, copy_dict1=False)
    comfy.patcher_extension.merge_nested_dicts(model_options["transformer_options"].setdefault("callbacks", {}), model.callbacks, copy_dict1=False)
    # begin registering hooks
    registered = comfy.hooks.HookGroup()
    target_dict = comfy.hooks.create_target_dict(comfy.hooks.EnumWeightTarget.Model)
    # handle all TransformerOptionsHooks
    for hook in hooks.get_type(comfy.hooks.EnumHookType.TransformerOptions):
        hook: comfy.hooks.TransformerOptionsHook
        hook.add_hook_patches(model, model_options, target_dict, registered)
    # handle all AddModelsHooks
    for hook in hooks.get_type(comfy.hooks.EnumHookType.AdditionalModels):
        hook: comfy.hooks.AdditionalModelsHook
        hook.add_hook_patches(model, model_options, target_dict, registered)
    # handle all WeightHooks by registering on ModelPatcher
    model.register_all_hook_patches(hooks, target_dict, model_options, registered)
    # add registered_hooks onto model_options for further reference
    if len(registered) > 0:
        model_options["registered_hooks"] = registered
    # merge original wrappers and callbacks with hooked wrappers and callbacks
    to_load_options: dict[str] = model_options.setdefault("to_load_options", {})
    for wc_name in ["wrappers", "callbacks"]:
        comfy.patcher_extension.merge_nested_dicts(to_load_options.setdefault(wc_name, {}), model_options["transformer_options"][wc_name],
                                                    copy_dict1=False)
    return to_load_options

def prepare_model_patcher_multigpu_clones(model_patcher: ModelPatcher, loaded_models: list[ModelPatcher], model_options: dict):
    '''
    In case multigpu acceleration is enabled, prep ModelPatchers for each device.
    '''
    multigpu_patchers: list[ModelPatcher] = [x for x in loaded_models if x.is_multigpu_base_clone]
    if len(multigpu_patchers) > 0:
        multigpu_dict: dict[torch.device, ModelPatcher] = {}
        multigpu_dict[model_patcher.load_device] = model_patcher
        for x in multigpu_patchers:
            x.hook_patches = comfy.model_patcher.create_hook_patches_clone(model_patcher.hook_patches, copy_tuples=True)
            x.hook_mode = model_patcher.hook_mode # match main model's hook_mode
            multigpu_dict[x.load_device] = x
        model_options["multigpu_clones"] = multigpu_dict
    return multigpu_patchers
