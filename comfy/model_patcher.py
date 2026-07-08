"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Comfy

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import collections
import inspect
import logging
import math
import uuid
from typing import Callable, Optional

import torch
import tqdm

import comfy.float
import comfy.hooks
import comfy.lora
import comfy.model_management
import comfy.ops
import comfy.patcher_extension
import comfy.utils
import comfy_aimdo.host_buffer
from comfy.comfy_types import UnetWrapperFunction
from comfy.quant_ops import QuantizedTensor
from comfy.patcher_extension import CallbacksMP, PatcherInjection, WrappersMP

import comfy_aimdo.model_vbar

def set_model_options_patch_replace(model_options, patch, name, block_name, number, transformer_index=None):
    to = model_options["transformer_options"].copy()

    if "patches_replace" not in to:
        to["patches_replace"] = {}
    else:
        to["patches_replace"] = to["patches_replace"].copy()

    if name not in to["patches_replace"]:
        to["patches_replace"][name] = {}
    else:
        to["patches_replace"][name] = to["patches_replace"][name].copy()

    if transformer_index is not None:
        block = (block_name, number, transformer_index)
    else:
        block = (block_name, number)
    to["patches_replace"][name][block] = patch
    model_options["transformer_options"] = to
    return model_options

def set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=False):
    model_options["sampler_post_cfg_function"] = model_options.get("sampler_post_cfg_function", []) + [post_cfg_function]
    if disable_cfg1_optimization:
        model_options["disable_cfg1_optimization"] = True
    return model_options

def set_model_options_pre_cfg_function(model_options, pre_cfg_function, disable_cfg1_optimization=False):
    model_options["sampler_pre_cfg_function"] = model_options.get("sampler_pre_cfg_function", []) + [pre_cfg_function]
    if disable_cfg1_optimization:
        model_options["disable_cfg1_optimization"] = True
    return model_options

def create_model_options_clone(orig_model_options: dict):
    return comfy.patcher_extension.copy_nested_dicts(orig_model_options)

def create_hook_patches_clone(orig_hook_patches, copy_tuples=False):
    new_hook_patches = {}
    for hook_ref in orig_hook_patches:
        new_hook_patches[hook_ref] = {}
        for k in orig_hook_patches[hook_ref]:
            new_hook_patches[hook_ref][k] = orig_hook_patches[hook_ref][k][:]
            if copy_tuples:
                for i in range(len(new_hook_patches[hook_ref][k])):
                    new_hook_patches[hook_ref][k][i] = tuple(new_hook_patches[hook_ref][k][i])
    return new_hook_patches

def wipe_lowvram_weight(m):
    if hasattr(m, "prev_comfy_cast_weights"):
        m.comfy_cast_weights = m.prev_comfy_cast_weights
        del m.prev_comfy_cast_weights

    if hasattr(m, "weight_function"):
        m.weight_function = []

    if hasattr(m, "bias_function"):
        m.bias_function = []

def move_weight_functions(m, device):
    if device is None:
        return 0

    memory = 0
    if hasattr(m, "weight_function"):
        for f in m.weight_function:
            if hasattr(f, "move_to"):
                memory += f.move_to(device=device)

    if hasattr(m, "bias_function"):
        for f in m.bias_function:
            if hasattr(f, "move_to"):
                memory += f.move_to(device=device)
    return memory

def string_to_seed(data):
    logging.warning("WARNING: string_to_seed has moved from comfy.model_patcher to comfy.utils")
    return comfy.utils.string_to_seed(data)

class LowVramPatch:
    is_lowvram_patch = True

    def __init__(self, key, patches, convert_func=None, set_func=None):
        self.key = key
        self.patches = patches
        self.convert_func = convert_func # TODO: remove
        self.set_func = set_func
        self.prepared_patches = None

    def memory_required(self):
        counter = [0]
        for patch in self.patches[self.key]:
            comfy.lora.prefetch_prepared_value(patch[1], counter, None, None, False)
        return counter[0]

    def prepare(self, destination, stream, copy=True, commit=True):
        counter = [0]
        prepared_patches = [
            (patch[0], comfy.lora.prefetch_prepared_value(patch[1], counter, destination, stream, copy), patch[2], patch[3], patch[4])
            for patch in self.patches[self.key]
        ]
        if commit:
            self.prepared_patches = prepared_patches
        return prepared_patches

    def clear_prepared(self):
        self.prepared_patches = None

    def __call__(self, weight):
        patches = self.prepared_patches if self.prepared_patches is not None else self.patches[self.key]
        return comfy.lora.calculate_weight(patches, weight, self.key, intermediate_dtype=weight.dtype)

LOWVRAM_PATCH_ESTIMATE_MATH_FACTOR = 2

def low_vram_patch_estimate_vram(model, key):
    weight, set_func, convert_func = get_key_weight(model, key)
    if weight is None:
        return 0
    model_dtype = getattr(model, "manual_cast_dtype", torch.float32)
    if model_dtype is None:
        model_dtype = weight.dtype

    return weight.numel() * model_dtype.itemsize * LOWVRAM_PATCH_ESTIMATE_MATH_FACTOR

def get_key_weight(model, key):
    set_func = None
    convert_func = None
    op_keys = key.rsplit('.', 1)
    if len(op_keys) < 2:
        weight = comfy.utils.get_attr(model, key)
    else:
        op = comfy.utils.get_attr(model, op_keys[0])
        try:
            set_func = getattr(op, "set_{}".format(op_keys[1]))
        except AttributeError:
            pass

        try:
            convert_func = getattr(op, "convert_{}".format(op_keys[1]))
        except AttributeError:
            pass

        weight = getattr(op, op_keys[1])
        if convert_func is not None:
            weight = comfy.utils.get_attr(model, key)

    return weight, set_func, convert_func

def key_param_name_to_key(key, param):
    if len(key) == 0:
        return param
    return "{}.{}".format(key, param)

class AutoPatcherEjector:
    def __init__(self, model: 'ModelPatcher', skip_and_inject_on_exit_only=False):
        self.model = model
        self.was_injected = False
        self.prev_skip_injection = False
        self.skip_and_inject_on_exit_only = skip_and_inject_on_exit_only

    def __enter__(self):
        self.was_injected = False
        self.prev_skip_injection = self.model.skip_injection
        if self.skip_and_inject_on_exit_only:
            self.model.skip_injection = True
        if self.model.is_injected:
            self.model.eject_model()
            self.was_injected = True

    def __exit__(self, *args):
        if self.skip_and_inject_on_exit_only:
            self.model.skip_injection = self.prev_skip_injection
            self.model.inject_model()
        if self.was_injected and not self.model.skip_injection:
            self.model.inject_model()
        self.model.skip_injection = self.prev_skip_injection

class MemoryCounter:
    def __init__(self, initial: int, minimum=0):
        self.value = initial
        self.minimum = minimum
        # TODO: add a safe limit besides 0

    def use(self, weight: torch.Tensor):
        weight_size = weight.nelement() * weight.element_size()
        if self.is_useable(weight_size):
            self.decrement(weight_size)
            return True
        return False

    def is_useable(self, used: int):
        return self.value - used > self.minimum

    def decrement(self, used: int):
        self.value -= used

CustomTorchDevice = collections.namedtuple("FakeDevice", ["type", "index"])("comfy-lazy-caster", 0)

class LazyCastingParam(torch.nn.Parameter):
    def __new__(cls, model, key, tensor):
        return super().__new__(cls, tensor)

    def __init__(self, model, key, tensor):
        self.model = model
        self.key = key

    @property
    def device(self):
        return CustomTorchDevice

    #safetensors will .to() us to the cpu which we catch here to cast on demand. The returned tensor is
    #then just a short lived thing in the safetensors serialization logic inside its big for loop over
    #all weights getting garbage collected per-weight
    def to(self, *args, **kwargs):
        return self.model.patch_weight_to_device(self.key, device_to=self.model.load_device, return_weight=True).to("cpu")


class LazyCastingQuantizedParam:
    def __init__(self, model, key):
        self.model = model
        self.key = key
        self.cpu_state_dict = None

    def state_dict_tensor(self, state_dict_key):
        if self.cpu_state_dict is None:
            weight = self.model.patch_weight_to_device(self.key, device_to=self.model.load_device, return_weight=True)
            self.cpu_state_dict = {k: v.to("cpu") for k, v in weight.state_dict(self.key).items()}
        return self.cpu_state_dict[state_dict_key]


class LazyCastingParamPiece(torch.nn.Parameter):
    def __new__(cls, caster, state_dict_key, tensor):
        return super().__new__(cls, tensor)

    def __init__(self, caster, state_dict_key, tensor):
        self.caster = caster
        self.state_dict_key = state_dict_key

    @property
    def device(self):
        return CustomTorchDevice

    def to(self, *args, **kwargs):
        caster = self.caster
        del self.caster
        return caster.state_dict_tensor(self.state_dict_key)


class ModelPatcher:
    def __init__(self, model, load_device, offload_device, size=0, weight_inplace_update=False):
        self.size = size
        self.model = model
        if not hasattr(self.model, 'device'):
            logging.debug("Model doesn't have a device attribute.")
            self.model.device = offload_device
        elif self.model.device is None:
            self.model.device = offload_device

        self.patches = {}
        self.backup = {}
        self.backup_buffers = {}
        self.object_patches = {}
        self.object_patches_backup = {}
        self.weight_wrapper_patches = {}
        self.model_options = {"transformer_options":{}}
        self.load_device = load_device
        self.offload_device = offload_device
        self.weight_inplace_update = weight_inplace_update
        self.force_cast_weights = False
        self.patches_uuid = uuid.uuid4()
        self.parent = None
        self.pinned = set()

        self.attachments: dict[str] = {}
        self.additional_models: dict[str, list[ModelPatcher]] = {}
        self.callbacks: dict[str, dict[str, list[Callable]]] = CallbacksMP.init_callbacks()
        self.wrappers: dict[str, dict[str, list[Callable]]] = WrappersMP.init_wrappers()

        self.is_injected = False
        self.skip_injection = False
        self.injections: dict[str, list[PatcherInjection]] = {}

        self.hook_patches: dict[comfy.hooks._HookRef] = {}
        self.hook_patches_backup: dict[comfy.hooks._HookRef] = None
        self.hook_backup: dict[str, tuple[torch.Tensor, torch.device]] = {}
        self.cached_hook_patches: dict[comfy.hooks.HookGroup, dict[str, torch.Tensor]] = {}
        self.current_hooks: Optional[comfy.hooks.HookGroup] = None
        self.forced_hooks: Optional[comfy.hooks.HookGroup] = None  # NOTE: only used for CLIP at this time
        self.is_clip = False
        self.hook_mode = comfy.hooks.EnumHookMode.MaxSpeed

        self.cached_patcher_init: tuple[Callable, tuple] | tuple[Callable, tuple, int] | None = None
        self.is_multigpu_base_clone = False
        self.clone_base_uuid = uuid.uuid4()

        if not hasattr(self.model, 'model_loaded_weight_memory'):
            self.model.model_loaded_weight_memory = 0

        if not hasattr(self.model, 'lowvram_patch_counter'):
            self.model.lowvram_patch_counter = 0

        if not hasattr(self.model, 'model_lowvram'):
            self.model.model_lowvram = False

        if not hasattr(self.model, 'current_weight_patches_uuid'):
            self.model.current_weight_patches_uuid = None

        if not hasattr(self.model, 'model_offload_buffer_memory'):
            self.model.model_offload_buffer_memory = 0

    def is_dynamic(self):
        return False

    def model_size(self):
        if self.size > 0:
            return self.size
        self.size = comfy.model_management.module_size(self.model)
        return self.size

    def loaded_size(self):
        return self.model.model_loaded_weight_memory

    def lowvram_patch_counter(self):
        return self.model.lowvram_patch_counter

    def get_free_memory(self, device):
        #Prioritize batching (incl. CFG/conds etc) over keeping the model resident. In
        #the vast majority of setups a little bit of offloading on the giant model more
        #than pays for CFG. So return everything both torch and Aimdo could give us
        aimdo_mem = 0
        if comfy.memory_management.aimdo_enabled:
            aimdo_device = device.index if getattr(device, "type", None) == "cuda" else None
            aimdo_mem = comfy_aimdo.model_vbar.vbars_analyze(aimdo_device)
        return comfy.model_management.get_free_memory(device) + aimdo_mem

    def get_clone_model_override(self):
        return self.model, (self.backup, self.backup_buffers, self.object_patches_backup, self.pinned)

    def clone(self, disable_dynamic=False, model_override=None, force_deepcopy=False):
        class_ = self.__class__
        if self.is_dynamic() and disable_dynamic or force_deepcopy:
            if self.is_dynamic() and disable_dynamic:
                class_ = ModelPatcher
            if model_override is None:
                if self.cached_patcher_init is None:
                    raise RuntimeError("Cannot create non-dynamic delegate: cached_patcher_init is not initialized.")
                temp_model_patcher = self.cached_patcher_init[0](*self.cached_patcher_init[1], disable_dynamic=True)
                if len(self.cached_patcher_init) > 2:
                    temp_model_patcher = temp_model_patcher[self.cached_patcher_init[2]]
                model_override = temp_model_patcher.get_clone_model_override()
        if model_override is None:
            model_override = self.get_clone_model_override()

        n = class_(model_override[0], self.load_device, self.offload_device, self.model_size(), weight_inplace_update=self.weight_inplace_update)
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]
        n.patches_uuid = self.patches_uuid

        n.object_patches = self.object_patches.copy()
        n.weight_wrapper_patches = self.weight_wrapper_patches.copy()
        n.model_options = comfy.utils.deepcopy_list_dict(self.model_options)
        n.parent = self

        n.force_cast_weights = self.force_cast_weights

        n.backup, n.backup_buffers, n.object_patches_backup, n.pinned = model_override[1]

        # attachments
        n.attachments = {}
        for k in self.attachments:
            if hasattr(self.attachments[k], "on_model_patcher_clone"):
                n.attachments[k] = self.attachments[k].on_model_patcher_clone()
            else:
                n.attachments[k] = self.attachments[k]
        # additional models
        for k, c in self.additional_models.items():
            n.additional_models[k] = [x.clone() for x in c]
        # callbacks
        for k, c in self.callbacks.items():
            n.callbacks[k] = {}
            for k1, c1 in c.items():
                n.callbacks[k][k1] = c1.copy()
        # sample wrappers
        for k, w in self.wrappers.items():
            n.wrappers[k] = {}
            for k1, w1 in w.items():
                n.wrappers[k][k1] = w1.copy()
        # injection
        n.is_injected = self.is_injected
        n.skip_injection = self.skip_injection
        for k, i in self.injections.items():
            n.injections[k] = i.copy()
        # hooks
        n.hook_patches = create_hook_patches_clone(self.hook_patches)
        n.hook_patches_backup = create_hook_patches_clone(self.hook_patches_backup) if self.hook_patches_backup else self.hook_patches_backup
        for group in self.cached_hook_patches:
            n.cached_hook_patches[group] = {}
            for k in self.cached_hook_patches[group]:
                n.cached_hook_patches[group][k] = self.cached_hook_patches[group][k]
        n.hook_backup = self.hook_backup
        n.current_hooks = self.current_hooks.clone() if self.current_hooks else self.current_hooks
        n.forced_hooks = self.forced_hooks.clone() if self.forced_hooks else self.forced_hooks
        n.is_clip = self.is_clip
        n.hook_mode = self.hook_mode

        n.cached_patcher_init = self.cached_patcher_init
        n.is_multigpu_base_clone = self.is_multigpu_base_clone
        n.clone_base_uuid = self.clone_base_uuid

        for callback in self.get_all_callbacks(CallbacksMP.ON_CLONE):
            callback(self, n)
        return n

    def deepclone_multigpu(self, new_load_device=None, models_cache: dict[uuid.UUID,ModelPatcher]=None):
        logging.info(f"Creating deepclone of {self.model.__class__.__name__} for {new_load_device if new_load_device else self.load_device}.")
        if self.cached_patcher_init is None:
            raise RuntimeError(
                f"Cannot create multigpu deepclone of {self.model.__class__.__name__}: "
                "the loader that produced this model does not support multigpu "
                "(cached_patcher_init is not initialized). Use a core loader "
                "(CheckpointLoaderSimple, UNETLoader, CLIPLoader/DualCLIPLoader, VAELoader), "
                "or have the custom loader register a cached_patcher_init factory."
            )
        comfy.model_management.unload_model_and_clones(self)
        # Produce a freshly-loaded patcher from the loader factory so the multigpu
        # clone owns its own untainted model weights (rather than relying on
        # copy.deepcopy of an already-patched/already-loaded module).
        temp_model_patcher: ModelPatcher | list[ModelPatcher] = self.cached_patcher_init[0](*self.cached_patcher_init[1])
        if len(self.cached_patcher_init) > 2:
            temp_model_patcher = temp_model_patcher[self.cached_patcher_init[2]]
        # Override clone()'s normal "share self.model + share backup containers" with
        # the pristine model from temp_model_patcher plus empty backup containers --
        # the fresh model has no patches applied, so any deepcopy of self's stale
        # backup/object_patches_backup/pinned would just propagate dead state that
        # no longer corresponds to anything in n.model.
        model_override = (temp_model_patcher.model, ({}, {}, {}, set()))
        n = self.clone(model_override=model_override)
        # clone() copies hook_backup by reference from self; reset since model is pristine.
        n.hook_backup = {}
        # set load device, if present
        if new_load_device is not None:
            n.load_device = new_load_device
        # Ensure any per-device bookkeeping (e.g. ModelPatcherDynamic.dynamic_pins)
        # has an entry for n.load_device on the freshly-loaded n.model. temp_model_patcher's
        # __init__ only registered its own (default) load_device.
        if hasattr(n, "register_load_device"):
            n.register_load_device(n.load_device)
        # multigpu clone should not have multigpu additional_models entry
        n.remove_additional_models("multigpu")
        # multigpu_clone all stored additional_models; make sure circular references are properly handled
        if models_cache is None:
            models_cache = {}
        for key, model_list in n.additional_models.items():
            for i in range(len(model_list)):
                add_model = n.additional_models[key][i]
                if add_model.clone_base_uuid not in models_cache:
                    models_cache[add_model.clone_base_uuid] = add_model.deepclone_multigpu(new_load_device=new_load_device, models_cache=models_cache)
                n.additional_models[key][i] = models_cache[add_model.clone_base_uuid]
        for callback in self.get_all_callbacks(CallbacksMP.ON_DEEPCLONE_MULTIGPU):
            callback(self, n)
        return n

    def match_multigpu_clones(self):
        multigpu_models = self.get_additional_models_with_key("multigpu")
        if len(multigpu_models) > 0:
            new_multigpu_models = []
            for mm in multigpu_models:
                # clone main model, but bring over relevant props from existing multigpu clone
                n = self.clone()
                n.load_device = mm.load_device
                n.backup = mm.backup
                n.object_patches_backup = mm.object_patches_backup
                n.hook_backup = mm.hook_backup
                n.model = mm.model
                n.is_multigpu_base_clone = mm.is_multigpu_base_clone
                n.remove_additional_models("multigpu")
                orig_additional_models: dict[str, list[ModelPatcher]] = comfy.patcher_extension.copy_nested_dicts(n.additional_models)
                n.additional_models = comfy.patcher_extension.copy_nested_dicts(mm.additional_models)
                # figure out which additional models are not present in multigpu clone
                models_cache = {}
                for mm_add_model in mm.get_additional_models():
                    models_cache[mm_add_model.clone_base_uuid] = mm_add_model
                remove_models_uuids = set(list(models_cache.keys()))
                for key, model_list in orig_additional_models.items():
                    for orig_add_model in model_list:
                        if orig_add_model.clone_base_uuid not in models_cache:
                            models_cache[orig_add_model.clone_base_uuid] = orig_add_model.deepclone_multigpu(new_load_device=n.load_device, models_cache=models_cache)
                            existing_list = n.get_additional_models_with_key(key)
                            existing_list.append(models_cache[orig_add_model.clone_base_uuid])
                            n.set_additional_models(key, existing_list)
                        if orig_add_model.clone_base_uuid in remove_models_uuids:
                            remove_models_uuids.remove(orig_add_model.clone_base_uuid)
                # remove duplicate additional models
                for key, model_list in n.additional_models.items():
                    new_model_list = [x for x in model_list if x.clone_base_uuid not in remove_models_uuids]
                    n.set_additional_models(key, new_model_list)
                for callback in self.get_all_callbacks(CallbacksMP.ON_MATCH_MULTIGPU_CLONES):
                    callback(self, n)
                new_multigpu_models.append(n)
            self.set_additional_models("multigpu", new_multigpu_models)

    def is_clone(self, other):
        if hasattr(other, 'model') and self.model is other.model:
            return True
        return False

    def clone_has_same_weights(self, clone: ModelPatcher, allow_multigpu=False):
        if allow_multigpu:
            if self.clone_base_uuid != clone.clone_base_uuid:
                return False
        else:
            if not self.is_clone(clone):
                return False

        if self.current_hooks != clone.current_hooks:
            return False
        if self.forced_hooks != clone.forced_hooks:
            return False
        if self.hook_patches.keys() != clone.hook_patches.keys():
            return False
        if self.attachments.keys() != clone.attachments.keys():
            return False
        if self.additional_models.keys() != clone.additional_models.keys():
            return False
        for key in self.callbacks:
            if len(self.callbacks[key]) != len(clone.callbacks[key]):
                return False
        for key in self.wrappers:
            if len(self.wrappers[key]) != len(clone.wrappers[key]):
                return False
        if self.injections.keys() != clone.injections.keys():
            return False

        if len(self.patches) == 0 and len(clone.patches) == 0:
            return True

        if self.patches_uuid == clone.patches_uuid:
            if len(self.patches) != len(clone.patches):
                logging.warning("WARNING: something went wrong, same patch uuid but different length of patches.")
            else:
                return True

    def memory_required(self, input_shape):
        return self.model.memory_required(input_shape=input_shape)

    def disable_model_cfg1_optimization(self):
        self.model_options["disable_cfg1_optimization"] = True

    def set_model_sampler_cfg_function(self, sampler_cfg_function, disable_cfg1_optimization=False):
        if len(inspect.signature(sampler_cfg_function).parameters) == 3:
            self.model_options["sampler_cfg_function"] = lambda args: sampler_cfg_function(args["cond"], args["uncond"], args["cond_scale"]) #Old way
        else:
            self.model_options["sampler_cfg_function"] = sampler_cfg_function
        if disable_cfg1_optimization:
            self.disable_model_cfg1_optimization()

    def set_model_sampler_post_cfg_function(self, post_cfg_function, disable_cfg1_optimization=False):
        self.model_options = set_model_options_post_cfg_function(self.model_options, post_cfg_function, disable_cfg1_optimization)

    def set_model_sampler_pre_cfg_function(self, pre_cfg_function, disable_cfg1_optimization=False):
        self.model_options = set_model_options_pre_cfg_function(self.model_options, pre_cfg_function, disable_cfg1_optimization)

    def set_model_sampler_calc_cond_batch_function(self, sampler_calc_cond_batch_function):
        self.model_options["sampler_calc_cond_batch_function"] = sampler_calc_cond_batch_function

    def set_model_unet_function_wrapper(self, unet_wrapper_function: UnetWrapperFunction):
        self.model_options["model_function_wrapper"] = unet_wrapper_function

    def set_model_denoise_mask_function(self, denoise_mask_function):
        self.model_options["denoise_mask_function"] = denoise_mask_function

    def set_model_patch(self, patch, name):
        to = self.model_options["transformer_options"]
        if "patches" not in to:
            to["patches"] = {}
        to["patches"][name] = to["patches"].get(name, []) + [patch]

    def set_model_patch_replace(self, patch, name, block_name, number, transformer_index=None):
        self.model_options = set_model_options_patch_replace(self.model_options, patch, name, block_name, number, transformer_index=transformer_index)

    def set_model_attn1_patch(self, patch):
        self.set_model_patch(patch, "attn1_patch")

    def set_model_attn2_patch(self, patch):
        self.set_model_patch(patch, "attn2_patch")

    def set_model_attn1_replace(self, patch, block_name, number, transformer_index=None):
        self.set_model_patch_replace(patch, "attn1", block_name, number, transformer_index)

    def set_model_attn2_replace(self, patch, block_name, number, transformer_index=None):
        self.set_model_patch_replace(patch, "attn2", block_name, number, transformer_index)

    def set_model_attn1_output_patch(self, patch):
        self.set_model_patch(patch, "attn1_output_patch")

    def set_model_attn2_output_patch(self, patch):
        self.set_model_patch(patch, "attn2_output_patch")

    def set_model_input_block_patch(self, patch):
        self.set_model_patch(patch, "input_block_patch")

    def set_model_input_block_patch_after_skip(self, patch):
        self.set_model_patch(patch, "input_block_patch_after_skip")

    def set_model_output_block_patch(self, patch):
        self.set_model_patch(patch, "output_block_patch")

    def set_model_emb_patch(self, patch):
        self.set_model_patch(patch, "emb_patch")

    def set_model_forward_timestep_embed_patch(self, patch):
        self.set_model_patch(patch, "forward_timestep_embed_patch")

    def set_model_double_block_patch(self, patch):
        self.set_model_patch(patch, "double_block")

    def set_model_post_input_patch(self, patch):
        self.set_model_patch(patch, "post_input")

    def set_model_noise_refiner_patch(self, patch):
        self.set_model_patch(patch, "noise_refiner")

    def set_model_middle_block_after_patch(self, patch):
        self.set_model_patch(patch, "middle_block_after_patch")


    def set_model_rope_options(self, scale_x, shift_x, scale_y, shift_y, scale_t, shift_t, **kwargs):
        rope_options = self.model_options["transformer_options"].get("rope_options", {})
        rope_options["scale_x"] = scale_x
        rope_options["scale_y"] = scale_y
        rope_options["scale_t"] = scale_t

        rope_options["shift_x"] = shift_x
        rope_options["shift_y"] = shift_y
        rope_options["shift_t"] = shift_t

        self.model_options["transformer_options"]["rope_options"] = rope_options


    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj

    def set_model_compute_dtype(self, dtype):
        self.add_object_patch("manual_cast_dtype", dtype)
        if dtype is not None:
            self.force_cast_weights = True
        self.patches_uuid = uuid.uuid4() #TODO: optimize by preventing a full model reload for this

    def add_weight_wrapper(self, name, function):
        self.weight_wrapper_patches[name] = self.weight_wrapper_patches.get(name, []) + [function]
        self.patches_uuid = uuid.uuid4()

    def get_model_object(self, name: str) -> torch.nn.Module:
        """Retrieves a nested attribute from an object using dot notation considering
        object patches.

        Args:
            name (str): The attribute path using dot notation (e.g. "model.layer.weight")

        Returns:
            The value of the requested attribute

        Example:
            patcher = ModelPatcher()
            weight = patcher.get_model_object("layer1.conv.weight")
        """
        if name in self.object_patches:
            return self.object_patches[name]
        else:
            if name in self.object_patches_backup:
                return self.object_patches_backup[name]
            else:
                return comfy.utils.get_attr(self.model, name)

    def model_patches_to(self, device):
        to = self.model_options["transformer_options"]
        if "patches" in to:
            patches = to["patches"]
            for name in patches:
                patch_list = patches[name]
                for i in range(len(patch_list)):
                    if hasattr(patch_list[i], "to"):
                        patch_list[i] = patch_list[i].to(device)
        if "patches_replace" in to:
            patches = to["patches_replace"]
            for name in patches:
                patch_list = patches[name]
                for k in patch_list:
                    if hasattr(patch_list[k], "to"):
                        patch_list[k] = patch_list[k].to(device)
        if "model_function_wrapper" in self.model_options:
            wrap_func = self.model_options["model_function_wrapper"]
            if hasattr(wrap_func, "to"):
                self.model_options["model_function_wrapper"] = wrap_func.to(device)

    def model_patches_models(self):
        to = self.model_options["transformer_options"]
        models = []
        if "patches" in to:
            patches = to["patches"]
            for name in patches:
                patch_list = patches[name]
                for i in range(len(patch_list)):
                    if hasattr(patch_list[i], "models"):
                        models += patch_list[i].models()
        if "patches_replace" in to:
            patches = to["patches_replace"]
            for name in patches:
                patch_list = patches[name]
                for k in patch_list:
                    if hasattr(patch_list[k], "models"):
                        models += patch_list[k].models()
        if "model_function_wrapper" in self.model_options:
            wrap_func = self.model_options["model_function_wrapper"]
            if hasattr(wrap_func, "models"):
                models += wrap_func.models()

        return models

    def model_patches_call_function(self, function_name="cleanup", arguments={}):
        to = self.model_options["transformer_options"]
        if "patches" in to:
            patches = to["patches"]
            for name in patches:
                patch_list = patches[name]
                for i in range(len(patch_list)):
                    if hasattr(patch_list[i], function_name):
                        getattr(patch_list[i], function_name)(**arguments)
        if "patches_replace" in to:
            patches = to["patches_replace"]
            for name in patches:
                patch_list = patches[name]
                for k in patch_list:
                    if hasattr(patch_list[k], function_name):
                        getattr(patch_list[k], function_name)(**arguments)
        if "model_function_wrapper" in self.model_options:
            wrap_func = self.model_options["model_function_wrapper"]
            if hasattr(wrap_func, function_name):
                getattr(wrap_func, function_name)(**arguments)

    def model_dtype(self):
        if hasattr(self.model, "get_dtype"):
            return self.model.get_dtype()

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        with self.use_ejected():
            p = set()
            model_sd = self.model.state_dict()
            for k in patches:
                offset = None
                function = None
                if isinstance(k, str):
                    key = k
                else:
                    offset = k[1]
                    key = k[0]
                    if len(k) > 2:
                        function = k[2]

                if key in model_sd:
                    p.add(k)
                    current_patches = self.patches.get(key, [])
                    current_patches.append((strength_patch, patches[k], strength_model, offset, function))
                    self.patches[key] = current_patches

            self.patches_uuid = uuid.uuid4()
            return list(p)

    def get_key_patches(self, filter_prefix=None):
        model_sd = self.model_state_dict()
        p = {}
        for k in model_sd:
            if filter_prefix is not None:
                if not k.startswith(filter_prefix):
                    continue
            bk = self.backup.get(k, None)
            hbk = self.hook_backup.get(k, None)
            weight, set_func, convert_func = get_key_weight(self.model, k)
            if bk is not None:
                weight = bk.weight
            if hbk is not None:
                weight = hbk[0]
            if convert_func is None:
                convert_func = lambda a, **kwargs: a

            if k in self.patches:
                p[k] = [(weight, convert_func)] + self.patches[k]
            else:
                p[k] = [(weight, convert_func)]
        return p

    def model_state_dict(self, filter_prefix=None):
        with self.use_ejected():
            sd = self.model.state_dict()
            keys = list(sd.keys())
            if filter_prefix is not None:
                for k in keys:
                    if not k.startswith(filter_prefix):
                        sd.pop(k)
            return sd

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False, return_weight=False, force_cast=False):
        weight, set_func, convert_func = get_key_weight(self.model, key)
        if key not in self.patches and not force_cast:
            return weight

        inplace_update = self.weight_inplace_update or inplace_update

        if key not in self.backup and not return_weight:
            self.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(weight.to(device=self.offload_device, copy=inplace_update), inplace_update)

        temp_dtype = comfy.model_management.lora_compute_dtype(device_to) if key in self.patches else None
        if device_to is not None:
            temp_weight = comfy.model_management.cast_to_device(weight, device_to, temp_dtype, copy=True)
        else:
            temp_weight = weight.to(temp_dtype, copy=True)
        if convert_func is not None:
            temp_weight = convert_func(temp_weight, inplace=True)

        out_weight = comfy.lora.calculate_weight(self.patches[key], temp_weight, key) if key in self.patches else temp_weight
        if set_func is None:
            if key in self.patches:
                out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype, seed=comfy.utils.string_to_seed(key))
            if return_weight:
                return out_weight
            elif inplace_update:
                comfy.utils.copy_to_param(self.model, key, out_weight)
            else:
                comfy.utils.set_attr_param(self.model, key, out_weight)
        else:
            return set_func(out_weight, inplace_update=inplace_update, seed=comfy.utils.string_to_seed(key), return_weight=return_weight)

    def pin_weight_to_device(self, key):
        weight, set_func, convert_func = get_key_weight(self.model, key)
        if comfy.model_management.pin_memory(weight):
            self.pinned.add(key)

    def unpin_weight(self, key):
        if key in self.pinned:
            weight, set_func, convert_func = get_key_weight(self.model, key)
            comfy.model_management.unpin_memory(weight)
            self.pinned.remove(key)

    def unpin_all_weights(self):
        for key in list(self.pinned):
            self.unpin_weight(key)

    def _load_list(self, for_dynamic=False, default_device=None):
        loading = []
        for n, m in self.model.named_modules():
            default = False
            params = { name: param for name, param in m.named_parameters(recurse=False) }
            for name, param in m.named_parameters(recurse=True):
                if name not in params:
                    default = True # default random weights in non leaf modules
                    break
            if default and default_device is not None:
                for param_name, param in params.items():
                    param.data = param.data.to(device=default_device, dtype=getattr(m, param_name + "_comfy_model_dtype", None))
            if not default and (hasattr(m, "comfy_cast_weights") or len(params) > 0):
                module_mem = comfy.model_management.module_size(m)
                module_offload_mem = module_mem
                if hasattr(m, "comfy_cast_weights"):
                    def check_module_offload_mem(key):
                        if key in self.patches:
                            return low_vram_patch_estimate_vram(self.model, key)
                        model_dtype = getattr(self.model, "manual_cast_dtype", None)
                        weight, _, _ = get_key_weight(self.model, key)
                        if model_dtype is None or weight is None:
                            return 0
                        if (weight.dtype != model_dtype or isinstance(weight, QuantizedTensor)):
                            return weight.numel() * model_dtype.itemsize
                        return 0
                    module_offload_mem += check_module_offload_mem("{}.weight".format(n))
                    module_offload_mem += check_module_offload_mem("{}.bias".format(n))
                # Dynamic: small weights (<64KB) first, then larger weights prioritized by size.
                # Non-dynamic: prioritize by module offload cost.
                if for_dynamic:
                    sort_criteria = (module_offload_mem >= 64 * 1024, -module_offload_mem)
                else:
                    sort_criteria = (module_offload_mem,)
                loading.append(sort_criteria + (module_mem, n, m, params))
        return loading

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        with self.use_ejected():
            self.unpatch_hooks()
            mem_counter = 0
            patch_counter = 0
            lowvram_counter = 0
            lowvram_mem_counter = 0
            loading = self._load_list()

            load_completely = []
            offloaded = []
            offload_buffer = 0
            loading.sort(reverse=True)
            for i, x in enumerate(loading):
                module_offload_mem, module_mem, n, m, params = x

                lowvram_weight = False

                potential_offload = max(offload_buffer, module_offload_mem + sum([ x1[1] for x1 in loading[i+1:i+1+comfy.model_management.NUM_STREAMS]]))
                lowvram_fits = mem_counter + module_mem + potential_offload < lowvram_model_memory

                weight_key = "{}.weight".format(n)
                bias_key = "{}.bias".format(n)

                if not full_load and hasattr(m, "comfy_cast_weights"):
                    if not lowvram_fits:
                        offload_buffer = potential_offload
                        lowvram_weight = True
                        lowvram_counter += 1
                        lowvram_mem_counter += module_mem
                        if hasattr(m, "prev_comfy_cast_weights"): #Already lowvramed
                            continue

                cast_weight = self.force_cast_weights
                m.comfy_force_cast_weights = self.force_cast_weights
                if lowvram_weight:
                    if hasattr(m, "comfy_cast_weights"):
                        m.weight_function = []
                        m.bias_function = []

                    if weight_key in self.patches:
                        if force_patch_weights:
                            self.patch_weight_to_device(weight_key)
                        else:
                            _, set_func, convert_func = get_key_weight(self.model, weight_key)
                            m.weight_function = [LowVramPatch(weight_key, self.patches, convert_func, set_func)]
                            patch_counter += 1
                    if bias_key in self.patches:
                        if force_patch_weights:
                            self.patch_weight_to_device(bias_key)
                        else:
                            _, set_func, convert_func = get_key_weight(self.model, bias_key)
                            m.bias_function = [LowVramPatch(bias_key, self.patches, convert_func, set_func)]
                            patch_counter += 1

                    cast_weight = True
                    offloaded.append((module_mem, n, m, params))
                else:
                    if hasattr(m, "comfy_cast_weights"):
                        wipe_lowvram_weight(m)

                    if full_load or lowvram_fits:
                        mem_counter += module_mem
                        load_completely.append((module_mem, n, m, params))
                    else:
                        offload_buffer = potential_offload

                if cast_weight and hasattr(m, "comfy_cast_weights"):
                    m.prev_comfy_cast_weights = m.comfy_cast_weights
                    m.comfy_cast_weights = True

                if weight_key in self.weight_wrapper_patches:
                    m.weight_function.extend(self.weight_wrapper_patches[weight_key])

                if bias_key in self.weight_wrapper_patches:
                    m.bias_function.extend(self.weight_wrapper_patches[bias_key])

                mem_counter += move_weight_functions(m, device_to)

            load_completely.sort(reverse=True)
            for x in load_completely:
                n = x[1]
                m = x[2]
                params = x[3]
                if hasattr(m, "comfy_patched_weights"):
                    if m.comfy_patched_weights == True:
                        continue

                for param, param_value in params.items():
                    if hasattr(m, "comfy_cast_weights") and getattr(param_value, "is_meta", False):
                        comfy.ops.disable_weight_init._zero_init_parameter(m, param)
                    key = key_param_name_to_key(n, param)
                    self.unpin_weight(key)
                    self.patch_weight_to_device(key, device_to=device_to)
                if comfy.model_management.is_device_cuda(device_to):
                    torch.cuda.synchronize()

                logging.debug("lowvram: loaded module regularly {} {}".format(n, m))
                m.comfy_patched_weights = True

            for x in load_completely:
                x[2].to(device_to)

            for x in offloaded:
                n = x[1]
                params = x[3]
                for param in params:
                    self.pin_weight_to_device(key_param_name_to_key(n, param))

            usable_stat = "{:.2f} MB usable,".format(lowvram_model_memory / (1024 * 1024)) if lowvram_model_memory < 1e32 else ""
            if lowvram_counter > 0:
                logging.info("loaded partially; {} {:.2f} MB loaded, {:.2f} MB offloaded, {:.2f} MB buffer reserved, lowvram patches: {}".format(usable_stat, mem_counter / (1024 * 1024), lowvram_mem_counter / (1024 * 1024), offload_buffer / (1024 * 1024), patch_counter))
                self.model.model_lowvram = True
            else:
                logging.info("loaded completely; {} {:.2f} MB loaded, full load: {}".format(usable_stat, mem_counter / (1024 * 1024), full_load))
                self.model.model_lowvram = False
                if full_load:
                    self.model.to(device_to)
                    mem_counter = self.model_size()

            self.model.lowvram_patch_counter += patch_counter
            self.model.device = device_to
            self.model.model_loaded_weight_memory = mem_counter
            self.model.model_offload_buffer_memory = offload_buffer
            self.model.current_weight_patches_uuid = self.patches_uuid

            for callback in self.get_all_callbacks(CallbacksMP.ON_LOAD):
                callback(self, device_to, lowvram_model_memory, force_patch_weights, full_load)

            self.apply_hooks(self.forced_hooks, force_apply=True)

    def patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True, force_patch_weights=False):
        with self.use_ejected():
            for k in self.object_patches:
                old = comfy.utils.set_attr(self.model, k, self.object_patches[k])
                if k not in self.object_patches_backup:
                    self.object_patches_backup[k] = old

            if lowvram_model_memory == 0:
                full_load = True
            else:
                full_load = False

            if load_weights:
                self.load(device_to, lowvram_model_memory=lowvram_model_memory, force_patch_weights=force_patch_weights, full_load=full_load)
        self.inject_model()
        return self.model

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        self.eject_model()
        if unpatch_weights:
            self.unpatch_hooks()
            self.unpin_all_weights()
            if self.model.model_lowvram:
                for m in self.model.modules():
                    move_weight_functions(m, device_to)
                    wipe_lowvram_weight(m)

                self.model.model_lowvram = False
                self.model.lowvram_patch_counter = 0

            keys = list(self.backup.keys())

            for k in keys:
                bk = self.backup[k]
                if bk.inplace_update:
                    comfy.utils.copy_to_param(self.model, k, bk.weight)
                else:
                    comfy.utils.set_attr_param(self.model, k, bk.weight)

            self.model.current_weight_patches_uuid = None
            self.backup.clear()

            if device_to is not None:
                self.model.to(device_to)
                self.model.device = device_to
            self.model.model_loaded_weight_memory = 0
            self.model.model_offload_buffer_memory = 0

            for m in self.model.modules():
                if hasattr(m, "comfy_patched_weights"):
                    del m.comfy_patched_weights

        keys = list(self.object_patches_backup.keys())
        for k in keys:
            comfy.utils.set_attr(self.model, k, self.object_patches_backup[k])

        self.object_patches_backup.clear()

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        with self.use_ejected():
            hooks_unpatched = False
            memory_freed = 0
            patch_counter = 0
            unload_list = self._load_list()
            unload_list.sort()

            offload_buffer = self.model.model_offload_buffer_memory
            if len(unload_list) > 0:
                NS = comfy.model_management.NUM_STREAMS
                offload_weight_factor = [ min(offload_buffer / (NS + 1), unload_list[0][1]) ] * NS

            for unload in unload_list:
                if memory_to_free + offload_buffer - self.model.model_offload_buffer_memory < memory_freed:
                    break
                module_offload_mem, module_mem, n, m, params = unload

                potential_offload = module_offload_mem + sum(offload_weight_factor)

                lowvram_possible = hasattr(m, "comfy_cast_weights")
                if hasattr(m, "comfy_patched_weights") and m.comfy_patched_weights == True:
                    move_weight = True
                    for param in params:
                        key = key_param_name_to_key(n, param)
                        bk = self.backup.get(key, None)
                        if bk is not None:
                            if not lowvram_possible:
                                move_weight = False
                                break

                            if not hooks_unpatched:
                                self.unpatch_hooks()
                                hooks_unpatched = True

                            if bk.inplace_update:
                                comfy.utils.copy_to_param(self.model, key, bk.weight)
                            else:
                                comfy.utils.set_attr_param(self.model, key, bk.weight)
                            self.backup.pop(key)

                    weight_key = "{}.weight".format(n)
                    bias_key = "{}.bias".format(n)
                    if move_weight:
                        cast_weight = self.force_cast_weights
                        m.to(device_to)
                        module_mem += move_weight_functions(m, device_to)
                        if lowvram_possible:
                            if weight_key in self.patches:
                                if force_patch_weights:
                                    self.patch_weight_to_device(weight_key)
                                else:
                                    _, set_func, convert_func = get_key_weight(self.model, weight_key)
                                    m.weight_function.append(LowVramPatch(weight_key, self.patches, convert_func, set_func))
                                    patch_counter += 1
                            if bias_key in self.patches:
                                if force_patch_weights:
                                    self.patch_weight_to_device(bias_key)
                                else:
                                    _, set_func, convert_func = get_key_weight(self.model, bias_key)
                                    m.bias_function.append(LowVramPatch(bias_key, self.patches, convert_func, set_func))
                                    patch_counter += 1
                            cast_weight = True

                        if cast_weight and hasattr(m, "comfy_cast_weights"):
                            m.prev_comfy_cast_weights = m.comfy_cast_weights
                            m.comfy_cast_weights = True
                        m.comfy_patched_weights = False
                        memory_freed += module_mem
                        offload_buffer = max(offload_buffer, potential_offload)
                        offload_weight_factor.append(module_mem)
                        offload_weight_factor.pop(0)
                        logging.debug("freed {}".format(n))

                        for param in params:
                            self.pin_weight_to_device(key_param_name_to_key(n, param))


            self.model.model_lowvram = True
            self.model.lowvram_patch_counter += patch_counter
            self.model.model_loaded_weight_memory -= memory_freed
            self.model.model_offload_buffer_memory = offload_buffer
            logging.info("Unloaded partially: {:.2f} MB freed, {:.2f} MB remains loaded, {:.2f} MB buffer reserved, lowvram patches: {}".format(memory_freed / (1024 * 1024), self.model.model_loaded_weight_memory / (1024 * 1024), offload_buffer / (1024 * 1024), self.model.lowvram_patch_counter))
            return memory_freed

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        with self.use_ejected(skip_and_inject_on_exit_only=True):
            unpatch_weights = self.model.current_weight_patches_uuid is not None and (self.model.current_weight_patches_uuid != self.patches_uuid or force_patch_weights)
            # TODO: force_patch_weights should not unload + reload full model
            used = self.model.model_loaded_weight_memory
            self.unpatch_model(self.offload_device, unpatch_weights=unpatch_weights)
            if unpatch_weights:
                extra_memory += (used - self.model.model_loaded_weight_memory)

            self.patch_model(load_weights=False)
            if extra_memory < 0 and not unpatch_weights:
                self.partially_unload(self.offload_device, -extra_memory, force_patch_weights=force_patch_weights)
                return 0
            full_load = False
            if self.model.model_lowvram == False and self.model.model_loaded_weight_memory > 0:
                self.apply_hooks(self.forced_hooks, force_apply=True)
                return 0
            if self.model.model_loaded_weight_memory + extra_memory > self.model_size():
                full_load = True
            current_used = self.model.model_loaded_weight_memory
            try:
                self.load(device_to, lowvram_model_memory=current_used + extra_memory, force_patch_weights=force_patch_weights, full_load=full_load)
            except Exception as e:
                self.detach()
                raise e

            return self.model.model_loaded_weight_memory - current_used

    def pinned_memory_size(self):
        # Pinned memory pressure tracking is only implemented for DynamicVram loading
        return 0

    def loaded_ram_size(self):
        # Loaded RAM pressure tracking is only implemented for DynamicVram loading
        return 0

    def partially_unload_ram(self, ram_to_unload):
        return 0

    def detach(self, unpatch_all=True):
        self.eject_model()
        self.model_patches_to(self.offload_device)
        if unpatch_all:
            self.unpatch_model(self.offload_device, unpatch_weights=unpatch_all)
        for callback in self.get_all_callbacks(CallbacksMP.ON_DETACH):
            callback(self, unpatch_all)
        return self.model

    def current_loaded_device(self):
        return self.model.device

    def calculate_weight(self, patches, weight, key, intermediate_dtype=torch.float32):
        logging.warning("The ModelPatcher.calculate_weight function is deprecated, please use: comfy.lora.calculate_weight instead")
        return comfy.lora.calculate_weight(patches, weight, key, intermediate_dtype=intermediate_dtype)

    def cleanup(self):
        self.model_patches_call_function(function_name="cleanup")
        self.clean_hooks()
        if hasattr(self.model, "current_patcher"):
            self.model.current_patcher = None
        for callback in self.get_all_callbacks(CallbacksMP.ON_CLEANUP):
            callback(self)

    def add_callback(self, call_type: str, callback: Callable):
        self.add_callback_with_key(call_type, None, callback)

    def add_callback_with_key(self, call_type: str, key: str, callback: Callable):
        c = self.callbacks.setdefault(call_type, {}).setdefault(key, [])
        c.append(callback)

    def remove_callbacks_with_key(self, call_type: str, key: str):
        c = self.callbacks.get(call_type, {})
        if key in c:
            c.pop(key)

    def get_callbacks(self, call_type: str, key: str):
        return self.callbacks.get(call_type, {}).get(key, [])

    def get_all_callbacks(self, call_type: str):
        c_list = []
        for c in self.callbacks.get(call_type, {}).values():
            c_list.extend(c)
        return c_list

    def add_wrapper(self, wrapper_type: str, wrapper: Callable):
        self.add_wrapper_with_key(wrapper_type, None, wrapper)

    def add_wrapper_with_key(self, wrapper_type: str, key: str, wrapper: Callable):
        w = self.wrappers.setdefault(wrapper_type, {}).setdefault(key, [])
        w.append(wrapper)

    def remove_wrappers_with_key(self, wrapper_type: str, key: str):
        w = self.wrappers.get(wrapper_type, {})
        if key in w:
            w.pop(key)

    def get_wrappers(self, wrapper_type: str, key: str):
        return self.wrappers.get(wrapper_type, {}).get(key, [])

    def get_all_wrappers(self, wrapper_type: str):
        w_list = []
        for w in self.wrappers.get(wrapper_type, {}).values():
            w_list.extend(w)
        return w_list

    def set_attachments(self, key: str, attachment):
        self.attachments[key] = attachment

    def remove_attachments(self, key: str):
        if key in self.attachments:
            self.attachments.pop(key)

    def get_attachment(self, key: str):
        return self.attachments.get(key, None)

    def set_injections(self, key: str, injections: list[PatcherInjection]):
        self.injections[key] = injections

    def remove_injections(self, key: str):
        if key in self.injections:
            self.injections.pop(key)

    def get_injections(self, key: str):
        return self.injections.get(key, None)

    def set_additional_models(self, key: str, models: list['ModelPatcher']):
        self.additional_models[key] = models

    def remove_additional_models(self, key: str):
        if key in self.additional_models:
            self.additional_models.pop(key)

    def get_additional_models_with_key(self, key: str):
        return self.additional_models.get(key, [])

    def get_additional_models(self):
        all_models: list[ModelPatcher] = []
        for models in self.additional_models.values():
            all_models.extend(models)
        return all_models

    def get_nested_additional_models(self):
        def _evaluate_sub_additional_models(prev_models: list[ModelPatcher], cache_set: set[ModelPatcher]):
            '''Make sure circular references do not cause infinite recursion.'''
            next_models = []
            for model in prev_models:
                candidates = model.get_additional_models()
                for c in candidates:
                    if c not in cache_set:
                        next_models.append(c)
                        cache_set.add(c)
            if len(next_models) == 0:
                return prev_models
            return prev_models + _evaluate_sub_additional_models(next_models, cache_set)

        all_models = self.get_additional_models()
        models_set = set(all_models)
        real_all_models = _evaluate_sub_additional_models(prev_models=all_models, cache_set=models_set)
        return real_all_models

    def use_ejected(self, skip_and_inject_on_exit_only=False):
        return AutoPatcherEjector(self, skip_and_inject_on_exit_only=skip_and_inject_on_exit_only)

    def inject_model(self):
        if self.is_injected or self.skip_injection:
            return
        for injections in self.injections.values():
            for inj in injections:
                inj.inject(self)
                self.is_injected = True
        if self.is_injected:
            for callback in self.get_all_callbacks(CallbacksMP.ON_INJECT_MODEL):
                callback(self)

    def eject_model(self):
        if not self.is_injected:
            return
        for injections in self.injections.values():
            for inj in injections:
                inj.eject(self)
        self.is_injected = False
        for callback in self.get_all_callbacks(CallbacksMP.ON_EJECT_MODEL):
            callback(self)

    def pre_run(self):
        if hasattr(self.model, "current_patcher"):
            self.model.current_patcher = self
        for callback in self.get_all_callbacks(CallbacksMP.ON_PRE_RUN):
            callback(self)

    def prepare_state(self, timestep, model_options):
        ignore_multigpu = model_options.get("ignore_multigpu", False)
        for callback in self.get_all_callbacks(CallbacksMP.ON_PREPARE_STATE):
            callback(self, timestep, model_options)
        if not ignore_multigpu and "multigpu_clones" in model_options:
            model_options["ignore_multigpu"] = True
            try:
                for p in model_options["multigpu_clones"].values():
                    p: ModelPatcher
                    p.prepare_state(timestep, model_options)
            finally:
                model_options.pop("ignore_multigpu", None)

    def restore_hook_patches(self):
        if self.hook_patches_backup is not None:
            self.hook_patches = self.hook_patches_backup
            self.hook_patches_backup = None

    def set_hook_mode(self, hook_mode: comfy.hooks.EnumHookMode):
        self.hook_mode = hook_mode

    def prepare_hook_patches_current_keyframe(self, t: torch.Tensor, hook_group: comfy.hooks.HookGroup, model_options: dict[str]):
        curr_t = t[0]
        reset_current_hooks = False
        multigpu_kf_changed_cache = None
        transformer_options = model_options.get("transformer_options", {})
        for hook in hook_group.hooks:
            changed = hook.hook_keyframe.prepare_current_keyframe(curr_t=curr_t, transformer_options=transformer_options)
            # if keyframe changed, remove any cached HookGroups that contain hook with the same hook_ref;
            # this will cause the weights to be recalculated when sampling
            if changed:
                # cache changed for multigpu usage
                if "multigpu_clones" in model_options:
                    if multigpu_kf_changed_cache is None:
                        multigpu_kf_changed_cache = []
                    multigpu_kf_changed_cache.append(hook)
                # reset current_hooks if contains hook that changed
                if self.current_hooks is not None:
                    for current_hook in self.current_hooks.hooks:
                        if current_hook == hook:
                            reset_current_hooks = True
                            break
                for cached_group in list(self.cached_hook_patches.keys()):
                    if cached_group.contains(hook):
                        self.cached_hook_patches.pop(cached_group)
        if reset_current_hooks:
            self.patch_hooks(None)
        if "multigpu_clones" in model_options:
            for p in model_options["multigpu_clones"].values():
                p: ModelPatcher
                p._handle_changed_hook_keyframes(multigpu_kf_changed_cache)

    def _handle_changed_hook_keyframes(self, kf_changed_cache: list[comfy.hooks.Hook]):
        'Used to handle multigpu behavior inside prepare_hook_patches_current_keyframe.'
        if kf_changed_cache is None:
            return
        reset_current_hooks = False
        # reset current_hooks if contains hook that changed
        for hook in kf_changed_cache:
            if self.current_hooks is not None:
                for current_hook in self.current_hooks.hooks:
                    if current_hook == hook:
                        reset_current_hooks = True
                        break
            for cached_group in list(self.cached_hook_patches.keys()):
                if cached_group.contains(hook):
                    self.cached_hook_patches.pop(cached_group)
        if reset_current_hooks:
            self.patch_hooks(None)

    def register_all_hook_patches(self, hooks: comfy.hooks.HookGroup, target_dict: dict[str], model_options: dict=None,
                                  registered: comfy.hooks.HookGroup = None):
        self.restore_hook_patches()
        if registered is None:
            registered = comfy.hooks.HookGroup()
        # handle WeightHooks
        weight_hooks_to_register: list[comfy.hooks.WeightHook] = []
        for hook in hooks.get_type(comfy.hooks.EnumHookType.Weight):
            if hook.hook_ref not in self.hook_patches:
                weight_hooks_to_register.append(hook)
            else:
                registered.add(hook)
        if len(weight_hooks_to_register) > 0:
            # clone hook_patches to become backup so that any non-dynamic hooks will return to their original state
            self.hook_patches_backup = create_hook_patches_clone(self.hook_patches)
            for hook in weight_hooks_to_register:
                hook.add_hook_patches(self, model_options, target_dict, registered)
        for callback in self.get_all_callbacks(CallbacksMP.ON_REGISTER_ALL_HOOK_PATCHES):
            callback(self, hooks, target_dict, model_options, registered)
        return registered

    def add_hook_patches(self, hook: comfy.hooks.WeightHook, patches, strength_patch=1.0, strength_model=1.0):
        with self.use_ejected():
            # NOTE: this mirrors behavior of add_patches func
            current_hook_patches: dict[str,list] = self.hook_patches.get(hook.hook_ref, {})
            p = set()
            model_sd = self.model.state_dict()
            for k in patches:
                offset = None
                function = None
                if isinstance(k, str):
                    key = k
                else:
                    offset = k[1]
                    key = k[0]
                    if len(k) > 2:
                        function = k[2]

                if key in model_sd:
                    p.add(k)
                    current_patches: list[tuple] = current_hook_patches.get(key, [])
                    current_patches.append((strength_patch, patches[k], strength_model, offset, function))
                    current_hook_patches[key] = current_patches
            self.hook_patches[hook.hook_ref] = current_hook_patches
            # since should care about these patches too to determine if same model, reroll patches_uuid
            self.patches_uuid = uuid.uuid4()
            return list(p)

    def get_combined_hook_patches(self, hooks: comfy.hooks.HookGroup):
        # combined_patches will contain  weights of all relevant hooks, per key
        combined_patches = {}
        if hooks is not None:
            for hook in hooks.hooks:
                hook_patches: dict = self.hook_patches.get(hook.hook_ref, {})
                for key in hook_patches.keys():
                    current_patches: list[tuple] = combined_patches.get(key, [])
                    if math.isclose(hook.strength, 1.0):
                        current_patches.extend(hook_patches[key])
                    else:
                        # patches are stored as tuples: (strength_patch, (tuple_with_weights,), strength_model)
                        for patch in hook_patches[key]:
                            new_patch = list(patch)
                            new_patch[0] *= hook.strength
                            current_patches.append(tuple(new_patch))
                    combined_patches[key] = current_patches
        return combined_patches

    def apply_hooks(self, hooks: comfy.hooks.HookGroup, transformer_options: dict=None, force_apply=False):
        # TODO: return transformer_options dict with any additions from hooks
        if self.current_hooks == hooks and (not force_apply or (not self.is_clip and hooks is None)):
            return comfy.hooks.create_transformer_options_from_hooks(self, hooks, transformer_options)
        self.patch_hooks(hooks=hooks)
        for callback in self.get_all_callbacks(CallbacksMP.ON_APPLY_HOOKS):
            callback(self, hooks)
        return comfy.hooks.create_transformer_options_from_hooks(self, hooks, transformer_options)

    def patch_hooks(self, hooks: comfy.hooks.HookGroup):
        with self.use_ejected():
            if hooks is not None:
                model_sd_keys = list(self.model_state_dict().keys())
                memory_counter = None
                if self.hook_mode == comfy.hooks.EnumHookMode.MaxSpeed:
                    # TODO: minimum_counter should have a minimum that conforms to loaded model requirements
                    memory_counter = MemoryCounter(initial=comfy.model_management.get_free_memory(self.load_device),
                                                minimum=comfy.model_management.minimum_inference_memory()*2)
                # if have cached weights for hooks, use it
                cached_weights = self.cached_hook_patches.get(hooks, None)
                if cached_weights is not None:
                    model_sd_keys_set = set(model_sd_keys)
                    for key in cached_weights:
                        if key not in model_sd_keys:
                            logging.warning(f"Cached hook could not patch. Key does not exist in model: {key}")
                            continue
                        self.patch_cached_hook_weights(cached_weights=cached_weights, key=key, memory_counter=memory_counter)
                        model_sd_keys_set.remove(key)
                    self.unpatch_hooks(model_sd_keys_set)
                else:
                    self.unpatch_hooks()
                    relevant_patches = self.get_combined_hook_patches(hooks=hooks)
                    original_weights = None
                    if len(relevant_patches) > 0:
                        original_weights = self.get_key_patches()
                    for key in relevant_patches:
                        if key not in model_sd_keys:
                            logging.warning(f"Cached hook would not patch. Key does not exist in model: {key}")
                            continue
                        self.patch_hook_weight_to_device(hooks=hooks, combined_patches=relevant_patches, key=key, original_weights=original_weights,
                                                            memory_counter=memory_counter)
            else:
                self.unpatch_hooks()
            self.current_hooks = hooks

    def patch_cached_hook_weights(self, cached_weights: dict, key: str, memory_counter: MemoryCounter):
        if key not in self.hook_backup:
            weight: torch.Tensor = comfy.utils.get_attr(self.model, key)
            target_device = self.offload_device
            if self.hook_mode == comfy.hooks.EnumHookMode.MaxSpeed:
                used = memory_counter.use(weight)
                if used:
                    target_device = weight.device
            self.hook_backup[key] = (weight.to(device=target_device, copy=True), weight.device)
        comfy.utils.copy_to_param(self.model, key, cached_weights[key][0].to(device=cached_weights[key][1]))

    def clear_cached_hook_weights(self):
        self.cached_hook_patches.clear()
        self.patch_hooks(None)

    def patch_hook_weight_to_device(self, hooks: comfy.hooks.HookGroup, combined_patches: dict, key: str, original_weights: dict, memory_counter: MemoryCounter):
        if key not in combined_patches:
            return

        weight, set_func, convert_func = get_key_weight(self.model, key)
        weight: torch.Tensor
        if key not in self.hook_backup:
            target_device = self.offload_device
            if self.hook_mode == comfy.hooks.EnumHookMode.MaxSpeed:
                used = memory_counter.use(weight)
                if used:
                    target_device = weight.device
            self.hook_backup[key] = (weight.to(device=target_device, copy=True), weight.device)
        # TODO: properly handle LowVramPatch, if it ends up an issue
        temp_weight = comfy.model_management.cast_to_device(weight, weight.device, torch.float32, copy=True)
        if convert_func is not None:
            temp_weight = convert_func(temp_weight, inplace=True)

        out_weight = comfy.lora.calculate_weight(combined_patches[key],
                                                 temp_weight,
                                                 key, original_weights=original_weights)
        del original_weights[key]
        if set_func is None:
            out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype, seed=comfy.utils.string_to_seed(key))
            comfy.utils.copy_to_param(self.model, key, out_weight)
        else:
            set_func(out_weight, inplace_update=True, seed=comfy.utils.string_to_seed(key))
        if self.hook_mode == comfy.hooks.EnumHookMode.MaxSpeed:
            # TODO: disable caching if not enough system RAM to do so
            target_device = self.offload_device
            used = memory_counter.use(weight)
            if used:
                target_device = weight.device
            self.cached_hook_patches.setdefault(hooks, {})
            self.cached_hook_patches[hooks][key] = (out_weight.to(device=target_device, copy=False), weight.device)
        del temp_weight
        del out_weight
        del weight

    def unpatch_hooks(self, whitelist_keys_set: set[str]=None) -> None:
        with self.use_ejected():
            if len(self.hook_backup) == 0:
                self.current_hooks = None
                return
            keys = list(self.hook_backup.keys())
            if whitelist_keys_set:
                for k in keys:
                    if k in whitelist_keys_set:
                        comfy.utils.copy_to_param(self.model, k, self.hook_backup[k][0].to(device=self.hook_backup[k][1]))
                        self.hook_backup.pop(k)
            else:
                for k in keys:
                    comfy.utils.copy_to_param(self.model, k, self.hook_backup[k][0].to(device=self.hook_backup[k][1]))

                self.hook_backup.clear()
                self.current_hooks = None

    def clean_hooks(self):
        self.unpatch_hooks()
        self.clear_cached_hook_weights()

    def model_state_dict_for_saving(self, model=None, prefix=""):
        if model is None:
            model = self.model

        original_state_dict = model.state_dict()
        output_state_dict = {}
        keys = list(original_state_dict)
        while len(keys) > 0:
            k = keys.pop(0)
            v = original_state_dict[k]
            op_keys = k.rsplit('.', 1)
            if (len(op_keys) < 2) or op_keys[1] not in ["weight", "bias"]:
                output_state_dict[k] = v
                continue
            try:
                op = comfy.utils.get_attr(model, op_keys[0])
            except:
                output_state_dict[k] = v
                continue
            if not op or not hasattr(op, "comfy_cast_weights") or \
                (hasattr(op, "comfy_patched_weights") and op.comfy_patched_weights == True):
                output_state_dict[k] = v
                continue
            key = prefix + k
            weight = comfy.utils.get_attr(self.model, key)
            if isinstance(weight, QuantizedTensor) and k in original_state_dict:
                qt_state_dict = weight.state_dict(k)
                caster = LazyCastingQuantizedParam(self, key)
                for group_key in (x for x in qt_state_dict if x in original_state_dict):
                    if group_key in keys:
                        keys.remove(group_key)
                    output_state_dict.pop(group_key, "")
                    output_state_dict[group_key] = LazyCastingParamPiece(caster, prefix + group_key, original_state_dict[group_key])
                continue
            output_state_dict[k] = LazyCastingParam(self, key, weight)
        return output_state_dict

    def state_dict_for_saving(self, clip_state_dict=None, vae_state_dict=None, clip_vision_state_dict=None):
        unet_state_dict = self.model_state_dict_for_saving(self.model.diffusion_model, "diffusion_model.")
        return self.model.state_dict_for_saving(unet_state_dict, clip_state_dict=clip_state_dict, vae_state_dict=vae_state_dict, clip_vision_state_dict=clip_vision_state_dict)

    def __del__(self):
        self.unpin_all_weights()
        self.detach(unpatch_all=False)

class ModelPatcherDynamic(ModelPatcher):

    def __new__(cls, model=None, load_device=None, offload_device=None, size=0, weight_inplace_update=False):
        if load_device is not None and comfy.model_management.is_device_cpu(load_device):
            #reroute to default MP for CPUs
            return ModelPatcher(model, load_device, offload_device, size, weight_inplace_update)
        return super().__new__(cls)

    def __init__(self, model, load_device, offload_device, size=0, weight_inplace_update=False):
        super().__init__(model, load_device, offload_device, size, weight_inplace_update)
        if not hasattr(self.model, "dynamic_vbars"):
            self.model.dynamic_vbars = {}
        if not hasattr(self.model, "dynamic_pins"):
            self.model.dynamic_pins = {}
        self.register_load_device(self.load_device)
        self.non_dynamic_delegate_model = None
        assert load_device is not None

    def register_load_device(self, device):
        """Ensure dynamic_pins has an entry for *device*.

        Called from __init__ and also from any code that retargets an
        already-constructed patcher to a new load_device (e.g. the
        Select{Model,CLIP,VAE}Device selector nodes); without this entry
        partially_unload_ram() raises KeyError when it tries to read the
        per-device pin state.
        """
        if device not in self.model.dynamic_pins:
            self.model.dynamic_pins[device] = {
                "weights": (comfy_aimdo.host_buffer.HostBuffer(0, 0, 0), [], [-1], [0], [0], {}),
                "patches": (comfy_aimdo.host_buffer.HostBuffer(0, 0, 0), [], [-1], [0], [0], {}),
                "hostbufs_initialized": False,
                "failed": False,
                "active": False,
            }

    def is_dynamic(self):
        return True

    def _vbar_get(self, create=False):
        if self.load_device == torch.device("cpu"):
            return None
        vbar = self.model.dynamic_vbars.get(self.load_device, None)
        if create and vbar is None:
            # x10. We dont know what model defined type casts we have in the vbar, but virtual address
            # space is pretty free. This will cover someone casting an entire model from FP4 to FP32
            # with some left over.
            vbar = comfy_aimdo.model_vbar.ModelVBAR(self.model_size() * 10, self.load_device.index)
            self.model.dynamic_vbars[self.load_device] = vbar
        return vbar

    def loaded_size(self):
        vbar = self._vbar_get()
        return (vbar.loaded_size() if vbar is not None else 0) + self.model.model_loaded_weight_memory

    #Pinning is deferred to ops time. Assert against this API to avoid pin leaks.

    def pin_weight_to_device(self, key):
        raise RuntimeError("pin_weight_to_device invalid for dymamic weight loading")

    def unpin_weight(self, key):
        raise RuntimeError("unpin_weight invalid for dymamic weight loading")

    def unpin_all_weights(self):
        self.partially_unload_ram(1e32)

    def memory_required(self, input_shape):
        #Pad this significantly. We are trying to get away from precise estimates. This
        #estimate is only used when using the ModelPatcherDynamic after ModelPatcher. If you
        #use all ModelPatcherDynamic this is ignored and its all done dynamically.
        return super().memory_required(input_shape=input_shape) * 1.3 + (1024 ** 3)

    def restore_loaded_backups(self):
        restored = self.model.model_loaded_weight_memory
        for key in list(self.backup.keys()):
            bk = self.backup.pop(key)
            comfy.utils.set_attr_param(self.model, key, bk.weight)
        for key in list(self.backup_buffers.keys()):
            comfy.utils.set_attr_buffer(self.model, key, self.backup_buffers.pop(key))
        self.model.model_loaded_weight_memory = 0
        return restored


    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False, dirty=False):

        #Force patching doesn't make sense in Dynamic loading, as you dont know what does and
        #doesn't need to be forced at this stage. The only thing you could do would be patch
        #it all on CPU which consumes huge RAM.
        assert not force_patch_weights

        #Full load doesn't make sense as we dont actually have any loader capability here and
        #now.
        assert not full_load

        assert device_to == self.load_device

        num_patches = 0
        allocated_size = 0
        self.restore_loaded_backups()

        with self.use_ejected():
            self.unpatch_hooks()

            vbar = self._vbar_get(create=True)
            pin_state = self.model.dynamic_pins[self.load_device]
            if not pin_state["hostbufs_initialized"]:
                hostbuf_size = comfy.model_management.pinned_hostbuf_size(self.model_size())
                pin_state["weights"] = (comfy_aimdo.host_buffer.HostBuffer(0, 64 * 1024 * 1024, hostbuf_size), [], [-1], [0], [0], {})
                pin_state["patches"] = (comfy_aimdo.host_buffer.HostBuffer(0, 8 * 1024 * 1024, hostbuf_size), [], [-1], [0], [0], {})
                pin_state["hostbufs_initialized"] = True
            pin_state["failed"] = False
            pin_state["active"] = True
            if vbar is not None:
                vbar.prioritize()

            loading = self._load_list(for_dynamic=True, default_device=device_to)
            loading.sort()

            for x in loading:
                *_, module_mem, n, m, params = x

                def set_dirty(item, dirty):
                    if dirty or not hasattr(item, "_v_signature"):
                        item._v_signature = None

                def setup_param(self, m, n, param_key):
                    nonlocal num_patches
                    key = key_param_name_to_key(n, param_key)

                    weight_function = []

                    weight, _, _ = get_key_weight(self.model, key)
                    if weight is None:
                        return (False, 0)
                    if key in self.patches:
                        if comfy.lora.calculate_shape(self.patches[key], weight, key) != weight.shape:
                            return (True, 0)
                        lowvram_patch = LowVramPatch(key, self.patches)
                        lowvram_patch._pin_state = pin_state
                        setattr(m, param_key + "_lowvram_function", lowvram_patch)
                        num_patches += 1
                    else:
                        setattr(m, param_key + "_lowvram_function", None)

                    if key in self.weight_wrapper_patches:
                        weight_function.extend(self.weight_wrapper_patches[key])
                    setattr(m, param_key + "_function", weight_function)
                    geometry = weight
                    if not isinstance(weight, QuantizedTensor):
                        model_dtype = getattr(m, param_key + "_comfy_model_dtype", None) or weight.dtype
                        weight._model_dtype = model_dtype
                        geometry = comfy.memory_management.TensorGeometry(shape=weight.shape, dtype=model_dtype)
                    return (False, comfy.memory_management.vram_aligned_size(geometry))

                def force_load_param(self, param_key, device_to):
                    key = key_param_name_to_key(n, param_key)
                    weight, _, _ = get_key_weight(self.model, key)
                    if weight is None:
                        return
                    if key in self.backup:
                        comfy.utils.set_attr_param(self.model, key, self.backup[key].weight)
                    self.patch_weight_to_device(key, device_to=device_to, force_cast=True)
                    weight, _, _ = get_key_weight(self.model, key)
                    if weight is not None:
                        self.model.model_loaded_weight_memory += weight.numel() * weight.element_size()

                if hasattr(m, "comfy_cast_weights"):
                    m.comfy_cast_weights = True
                    m.seed_key = n
                    m._pin_state = pin_state
                    set_dirty(m, dirty)

                    #Models that mix tiny and giant weights can causing lopsided stream buffer
                    #rotations and stall. force the tinys over.
                    if module_mem > 16 * 1024:
                        force_load, v_weight_size = setup_param(self, m, n, "weight")
                        force_load_bias, v_weight_bias = setup_param(self, m, n, "bias")
                        force_load = force_load or force_load_bias
                        v_weight_size += v_weight_bias
                        if force_load:
                            logging.info(f"Module {n} has resizing Lora - force loading")
                    else:
                        force_load=True

                    if force_load:
                        if hasattr(m, "_v"):
                            comfy_aimdo.model_vbar.vbar_unpin(m._v)
                            delattr(m, "_v")
                        force_load_param(self, "weight", device_to)
                        force_load_param(self, "bias", device_to)
                    else:
                        if vbar is not None and not hasattr(m, "_v"):
                            m._v = vbar.alloc(v_weight_size)
                        allocated_size += v_weight_size

                    for param in params:
                        if param not in ("weight", "bias"):
                            force_load_param(self, param, device_to)

                else:
                    for param in params:
                        key = key_param_name_to_key(n, param)
                        weight, _, _ = get_key_weight(self.model, key)
                        if key not in self.backup:
                            self.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(weight, False)
                        model_dtype = getattr(m, param + "_comfy_model_dtype", None)
                        casted_weight = weight.to(dtype=model_dtype, device=device_to)
                        comfy.utils.set_attr_param(self.model, key, casted_weight)
                        self.model.model_loaded_weight_memory += casted_weight.numel() * casted_weight.element_size()

                move_weight_functions(m, device_to)

            for key, buf in self.model.named_buffers(recurse=True):
                if key not in self.backup_buffers:
                    self.backup_buffers[key] = buf
                module, buf_name = comfy.utils.resolve_attr(self.model, key)
                model_dtype = getattr(module, buf_name + "_comfy_model_dtype", None)
                casted_buf = buf.to(dtype=model_dtype, device=device_to)
                comfy.utils.set_attr_buffer(self.model, key, casted_buf)
                self.model.model_loaded_weight_memory += casted_buf.numel() * casted_buf.element_size()

            force_load_stat = f" Force pre-loaded {len(self.backup)} weights: {self.model.model_loaded_weight_memory // 1024} KB." if len(self.backup) > 0 else ""
            log_key = (self.patches_uuid, allocated_size, num_patches, len(self.backup), self.model.model_loaded_weight_memory)
            in_loop = bool(getattr(tqdm.tqdm, "_instances", None))
            level = logging.DEBUG if in_loop and getattr(self, "_last_prepare_log_key", None) == log_key else logging.INFO
            self._last_prepare_log_key = log_key
            logging.log(level, f"Model {self.model.__class__.__name__} prepared for dynamic VRAM loading. {allocated_size // (1024 ** 2)}MB Staged. {num_patches} patches attached.{force_load_stat}")

            self.model.device = device_to
            self.model.current_weight_patches_uuid = self.patches_uuid

            for callback in self.get_all_callbacks(CallbacksMP.ON_LOAD):
                #These are all super dangerous. Who knows what the custom nodes actually do here...
                callback(self, device_to, lowvram_model_memory, force_patch_weights, full_load)

            self.apply_hooks(self.forced_hooks, force_apply=True)

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        assert not force_patch_weights #See above
        assert self.load_device != torch.device("cpu")

        vbar = self._vbar_get()
        freed = 0 if vbar is None else vbar.free_memory(memory_to_free)

        if freed < memory_to_free:
            freed += self.restore_loaded_backups()

        return freed

    def loaded_ram_size(self):
        return (self.model.dynamic_pins[self.load_device]["weights"][0].size)

    def pinned_memory_size(self):
        return (self.model.dynamic_pins[self.load_device]["weights"][3][0])

    def unregister_inactive_pins(self, ram_to_unload, subsets=[ "weights", "patches" ]):
        freed = 0
        pin_state = self.model.dynamic_pins[self.load_device]
        for subset in subsets:
            hostbuf, stack, stack_split, pinned_size, *_ = pin_state[subset]
            split = stack_split[0]
            while split >= 0:
                module, offset = stack[split]
                split -= 1
                stack_split[0] = split
                if not module._pin_registered:
                    continue
                size = module._pin.numel() * module._pin.element_size()
                if torch.cuda.cudart().cudaHostUnregister(module._pin.data_ptr()) != 0:
                    comfy.model_management.discard_cuda_async_error()
                    continue
                module._pin_registered = False
                comfy.model_management.TOTAL_PINNED_MEMORY = max(0, comfy.model_management.TOTAL_PINNED_MEMORY - size)
                pinned_size[0] = max(0, pinned_size[0] - size)
                freed += size
                ram_to_unload -= size
                if ram_to_unload <= 0:
                    return freed
        return freed

    def partially_unload_ram(self, ram_to_unload, subsets=[ "weights", "patches" ]):
        freed = 0
        pin_state = self.model.dynamic_pins[self.load_device]
        for subset in subsets:
            hostbuf, stack, stack_split, pinned_size, *_ = pin_state[subset]
            while len(stack) > 0:
                module, offset = stack.pop()
                size = module._pin.numel() * module._pin.element_size()
                module._pin_balancer_entry[-1] = None
                del module._pin_balancer_entry
                del module._pin
                hostbuf.truncate(offset, do_unregister=module._pin_registered)
                stack_split[0] = min(stack_split[0], len(stack) - 1)
                if module._pin_registered:
                    comfy.model_management.TOTAL_PINNED_MEMORY = max(0, comfy.model_management.TOTAL_PINNED_MEMORY - size)
                    pinned_size[0] = max(0, pinned_size[0] - size)
                freed += size
                ram_to_unload -= size
                if ram_to_unload <= 0:
                    return freed
        return freed

    def patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True, force_patch_weights=False):
        #This isn't used by the core at all and can only be to load a model out of
        #the control of proper model_managment. If you are a custom node author reading
        #this, the correct pattern is to call load_models_gpu() to get a proper
        #managed load of your model.
        assert not load_weights
        return super().patch_model(load_weights=load_weights, force_patch_weights=force_patch_weights)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        super().unpatch_model(device_to=None, unpatch_weights=False)

        if unpatch_weights:
            self.partially_unload_ram(1e32)
            self.partially_unload(None, 1e32)
            for m in self.model.modules():
                move_weight_functions(m, device_to)

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        assert not force_patch_weights #See above
        with self.use_ejected(skip_and_inject_on_exit_only=True):
            dirty = self.model.current_weight_patches_uuid is not None and (self.model.current_weight_patches_uuid != self.patches_uuid)

            self.unpatch_model(self.offload_device, unpatch_weights=False)
            self.patch_model(load_weights=False)

            try:
                self.load(device_to, dirty=dirty)
            except Exception as e:
                self.detach()
                raise e
            #ModelPatcher::partially_load returns a number on what got loaded but
            #nothing in core uses this and we have no data in the Dynamic world. Hit
            #the custom node devs with a None rather than a 0 that would mislead any
            #logic they might have.
            return None

    def patch_cached_hook_weights(self, cached_weights: dict, key: str, memory_counter: MemoryCounter):
        assert False #Should be unreachable - we dont ever cache in the new implementation

    def patch_hook_weight_to_device(self, hooks: comfy.hooks.HookGroup, combined_patches: dict, key: str, original_weights: dict, memory_counter: MemoryCounter):
        if key not in combined_patches:
            return

        raise RuntimeError("Hooks not implemented in ModelPatcherDynamic. Please remove --fast arguments form ComfyUI startup")

    def unpatch_hooks(self, whitelist_keys_set: set[str]=None) -> None:
        pass

    def get_non_dynamic_delegate(self):
        model_patcher = self.clone(disable_dynamic=True, model_override=self.non_dynamic_delegate_model)
        self.non_dynamic_delegate_model = model_patcher.get_clone_model_override()
        return model_patcher


CoreModelPatcher = ModelPatcher
