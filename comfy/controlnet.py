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

import torch
from enum import Enum
import math
import os
import logging
import copy
import comfy.utils
import comfy.model_management
import comfy.model_detection
import comfy.model_patcher
import comfy.ops
import comfy.latent_formats
import comfy.model_base

import comfy.cldm.cldm
import comfy.t2i_adapter.adapter
import comfy.ldm.cascade.controlnet
import comfy.cldm.mmdit
import comfy.ldm.hydit.controlnet
import comfy.ldm.flux.controlnet
import comfy.ldm.qwen_image.controlnet
import comfy.cldm.dit_embedder
from typing import TYPE_CHECKING, Union
if TYPE_CHECKING:
    from comfy.hooks import HookGroup


def broadcast_image_to(tensor, target_batch_size, batched_number):
    current_batch_size = tensor.shape[0]
    if current_batch_size == 1:
        return tensor

    per_batch = target_batch_size // batched_number
    tensor = tensor[:per_batch]

    if per_batch > tensor.shape[0]:
        tensor = torch.cat([tensor] * (per_batch // tensor.shape[0]) + [tensor[:(per_batch % tensor.shape[0])]], dim=0)

    current_batch_size = tensor.shape[0]
    if current_batch_size == target_batch_size:
        return tensor
    else:
        return torch.cat([tensor] * batched_number, dim=0)

class StrengthType(Enum):
    CONSTANT = 1
    LINEAR_UP = 2

class ControlIsolation:
    '''Temporarily set a ControlBase object's previous_controlnet to None to prevent cascading calls.'''
    def __init__(self, control: ControlBase):
        self.control = control
        self.orig_previous_controlnet = control.previous_controlnet

    def __enter__(self):
        self.control.previous_controlnet = None

    def __exit__(self, *args):
        self.control.previous_controlnet = self.orig_previous_controlnet

class ControlBase:
    def __init__(self):
        self.cond_hint_original = None
        self.cond_hint = None
        self.strength = 1.0
        self.timestep_percent_range = (0.0, 1.0)
        self.latent_format = None
        self.vae = None
        self.global_average_pooling = False
        self.timestep_range = None
        self.compression_ratio = 8
        self.upscale_algorithm = 'nearest-exact'
        self.extra_args = {}
        self.previous_controlnet: Union[ControlBase, None] = None
        self.extra_conds = []
        self.strength_type = StrengthType.CONSTANT
        self.concat_mask = False
        self.extra_concat_orig = []
        self.extra_concat = None
        self.extra_hooks: HookGroup = None
        self.preprocess_image = lambda a: a
        self.multigpu_clones: dict[torch.device, ControlBase] = {}

    def set_cond_hint(self, cond_hint, strength=1.0, timestep_percent_range=(0.0, 1.0), vae=None, extra_concat=[]):
        self.cond_hint_original = cond_hint
        self.strength = strength
        self.timestep_percent_range = timestep_percent_range
        if self.latent_format is not None:
            if vae is None:
                logging.warning("WARNING: no VAE provided to the controlnet apply node when this controlnet requires one.")
            self.vae = vae
        self.extra_concat_orig = extra_concat.copy()
        if self.concat_mask and len(self.extra_concat_orig) == 0:
            self.extra_concat_orig.append(torch.tensor([[[[1.0]]]]))
        return self

    def pre_run(self, model, percent_to_timestep_function):
        self.timestep_range = (percent_to_timestep_function(self.timestep_percent_range[0]), percent_to_timestep_function(self.timestep_percent_range[1]))
        if self.previous_controlnet is not None:
            self.previous_controlnet.pre_run(model, percent_to_timestep_function)

    def set_previous_controlnet(self, controlnet):
        self.previous_controlnet = controlnet
        return self

    def cleanup(self):
        if self.previous_controlnet is not None:
            self.previous_controlnet.cleanup()
        for device_cnet in self.multigpu_clones.values():
            with ControlIsolation(device_cnet):
                device_cnet.cleanup()
        self.cond_hint = None
        self.extra_concat = None
        self.timestep_range = None

    def get_models(self):
        out = []
        for device_cnet in self.multigpu_clones.values():
            out += device_cnet.get_models_only_self()
        if self.previous_controlnet is not None:
            out += self.previous_controlnet.get_models()
        return out

    def get_models_only_self(self):
        'Calls get_models, but temporarily sets previous_controlnet to None.'
        with ControlIsolation(self):
            return self.get_models()

    def get_instance_for_device(self, device):
        'Returns instance of this Control object intended for selected device.'
        return self.multigpu_clones.get(device, self)

    def deepclone_multigpu(self, load_device, autoregister=False):
        '''
        Create deep clone of Control object where model(s) is set to other devices.

        When autoregister is set to True, the deep clone is also added to multigpu_clones dict.
        '''
        raise NotImplementedError("Classes inheriting from ControlBase should define their own deepclone_multigpu funtion.")

    def get_extra_hooks(self):
        out = []
        if self.extra_hooks is not None:
            out.append(self.extra_hooks)
        if self.previous_controlnet is not None:
            out += self.previous_controlnet.get_extra_hooks()
        return out

    def copy_to(self, c: ControlBase):
        c.cond_hint_original = self.cond_hint_original
        c.strength = self.strength
        c.timestep_percent_range = self.timestep_percent_range
        c.global_average_pooling = self.global_average_pooling
        c.compression_ratio = self.compression_ratio
        c.upscale_algorithm = self.upscale_algorithm
        c.latent_format = self.latent_format
        c.extra_args = self.extra_args.copy()
        c.vae = self.vae
        c.extra_conds = self.extra_conds.copy()
        c.strength_type = self.strength_type
        c.concat_mask = self.concat_mask
        c.extra_concat_orig = self.extra_concat_orig.copy()
        c.extra_hooks = self.extra_hooks.clone() if self.extra_hooks else None
        c.preprocess_image = self.preprocess_image

    def inference_memory_requirements(self, dtype):
        if self.previous_controlnet is not None:
            return self.previous_controlnet.inference_memory_requirements(dtype)
        return 0

    def control_merge(self, control, control_prev, output_dtype):
        out = {'input':[], 'middle':[], 'output': []}

        for key in control:
            control_output = control[key]
            applied_to = set()
            for i in range(len(control_output)):
                x = control_output[i]
                if x is not None:
                    if self.global_average_pooling:
                        x = torch.mean(x, dim=(2, 3), keepdim=True).repeat(1, 1, x.shape[2], x.shape[3])

                    if x not in applied_to: #memory saving strategy, allow shared tensors and only apply strength to shared tensors once
                        applied_to.add(x)
                        if self.strength_type == StrengthType.CONSTANT:
                            x *= self.strength
                        elif self.strength_type == StrengthType.LINEAR_UP:
                            x *= (self.strength ** float(len(control_output) - i))

                    if output_dtype is not None and x.dtype != output_dtype:
                        x = x.to(output_dtype)

                out[key].append(x)

        if control_prev is not None:
            for x in ['input', 'middle', 'output']:
                o = out[x]
                for i in range(len(control_prev[x])):
                    prev_val = control_prev[x][i]
                    if i >= len(o):
                        o.append(prev_val)
                    elif prev_val is not None:
                        if o[i] is None:
                            o[i] = prev_val
                        else:
                            if o[i].shape[0] < prev_val.shape[0]:
                                o[i] = prev_val + o[i]
                            else:
                                o[i] = prev_val + o[i] #TODO: change back to inplace add if shared tensors stop being an issue
        return out

    def set_extra_arg(self, argument, value=None):
        self.extra_args[argument] = value


class ControlNet(ControlBase):
    def __init__(self, control_model=None, global_average_pooling=False, compression_ratio=8, latent_format=None, load_device=None, manual_cast_dtype=None, extra_conds=["y"], strength_type=StrengthType.CONSTANT, concat_mask=False, preprocess_image=lambda a: a):
        super().__init__()
        self.control_model = control_model
        self.load_device = load_device
        if control_model is not None:
            self.control_model_wrapped = comfy.model_patcher.CoreModelPatcher(self.control_model, load_device=load_device, offload_device=comfy.model_management.unet_offload_device())

        self.compression_ratio = compression_ratio
        self.global_average_pooling = global_average_pooling
        self.model_sampling_current = None
        self.manual_cast_dtype = manual_cast_dtype
        self.latent_format = latent_format
        self.extra_conds += extra_conds
        self.strength_type = strength_type
        self.concat_mask = concat_mask
        self.preprocess_image = preprocess_image

    def get_control(self, x_noisy, t, cond, batched_number, transformer_options):
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number, transformer_options)

        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                if control_prev is not None:
                    return control_prev
                else:
                    return None

        dtype = self.control_model.dtype
        if self.manual_cast_dtype is not None:
            dtype = self.manual_cast_dtype

        if self.cond_hint is None or x_noisy.shape[2] * self.compression_ratio != self.cond_hint.shape[2] or x_noisy.shape[3] * self.compression_ratio != self.cond_hint.shape[3]:
            if self.cond_hint is not None:
                del self.cond_hint
            self.cond_hint = None
            compression_ratio = self.compression_ratio
            if self.vae is not None:
                compression_ratio *= self.vae.spacial_compression_encode()
            else:
                if self.latent_format is not None:
                    raise ValueError("This Controlnet needs a VAE but none was provided, please use a ControlNetApply node with a VAE input and connect it.")
            self.cond_hint = comfy.utils.common_upscale(self.cond_hint_original, x_noisy.shape[-1] * compression_ratio, x_noisy.shape[-2] * compression_ratio, self.upscale_algorithm, "center")
            self.cond_hint = self.preprocess_image(self.cond_hint)
            if self.vae is not None:
                loaded_models = comfy.model_management.loaded_models(only_currently_used=True)
                self.cond_hint = self.vae.encode(self.cond_hint.movedim(1, -1))
                comfy.model_management.load_models_gpu(loaded_models)
            if self.latent_format is not None:
                self.cond_hint = self.latent_format.process_in(self.cond_hint)
            if len(self.extra_concat_orig) > 0:
                to_concat = []
                for c in self.extra_concat_orig:
                    c = c.to(self.cond_hint.device)
                    c = comfy.utils.common_upscale(c, self.cond_hint.shape[-1], self.cond_hint.shape[-2], self.upscale_algorithm, "center")
                    if c.ndim < self.cond_hint.ndim:
                        c = c.unsqueeze(2)
                        c = comfy.utils.repeat_to_batch_size(c, self.cond_hint.shape[2], dim=2)
                    to_concat.append(comfy.utils.repeat_to_batch_size(c, self.cond_hint.shape[0]))
                self.cond_hint = torch.cat([self.cond_hint] + to_concat, dim=1)

            self.cond_hint = self.cond_hint.to(device=x_noisy.device, dtype=dtype)
        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = broadcast_image_to(self.cond_hint, x_noisy.shape[0], batched_number)

        context = cond.get('crossattn_controlnet', cond['c_crossattn'])
        extra = self.extra_args.copy()
        for c in self.extra_conds:
            temp = cond.get(c, None)
            if temp is not None:
                extra[c] = comfy.model_base.convert_tensor(temp, dtype, x_noisy.device)

        timestep = self.model_sampling_current.timestep(t)
        x_noisy = self.model_sampling_current.calculate_input(t, x_noisy)

        control = self.control_model(x=x_noisy.to(dtype), hint=self.cond_hint, timesteps=timestep.to(dtype), context=comfy.model_management.cast_to_device(context, x_noisy.device, dtype), **extra)
        return self.control_merge(control, control_prev, output_dtype=None)

    def copy(self):
        c = ControlNet(None, global_average_pooling=self.global_average_pooling, load_device=self.load_device, manual_cast_dtype=self.manual_cast_dtype)
        c.control_model = self.control_model
        c.control_model_wrapped = self.control_model_wrapped
        self.copy_to(c)
        return c

    def deepclone_multigpu(self, load_device, autoregister=False):
        c = self.copy()
        c.control_model = copy.deepcopy(c.control_model)
        c.control_model_wrapped = comfy.model_patcher.ModelPatcher(c.control_model, load_device=load_device, offload_device=comfy.model_management.unet_offload_device())
        if autoregister:
            self.multigpu_clones[load_device] = c
        return c

    def get_models(self):
        out = super().get_models()
        out.append(self.control_model_wrapped)
        return out

    def pre_run(self, model, percent_to_timestep_function):
        super().pre_run(model, percent_to_timestep_function)
        self.model_sampling_current = model.model_sampling

    def cleanup(self):
        self.model_sampling_current = None
        super().cleanup()


class QwenFunControlNet(ControlNet):
    def get_control(self, x_noisy, t, cond, batched_number, transformer_options):
        # Fun checkpoints are more sensitive to high strengths in the generic
        # ControlNet merge path. Use a soft response curve so strength=1.0 stays
        # unchanged while >1 grows more gently.
        original_strength = self.strength
        self.strength = math.sqrt(max(self.strength, 0.0))
        try:
            return super().get_control(x_noisy, t, cond, batched_number, transformer_options)
        finally:
            self.strength = original_strength

    def pre_run(self, model, percent_to_timestep_function):
        super().pre_run(model, percent_to_timestep_function)
        self.set_extra_arg("base_model", model.diffusion_model)

    def cleanup(self):
        self.extra_args.pop("base_model", None)
        super().cleanup()

    def copy(self):
        c = QwenFunControlNet(None, global_average_pooling=self.global_average_pooling, load_device=self.load_device, manual_cast_dtype=self.manual_cast_dtype)
        c.control_model = self.control_model
        c.control_model_wrapped = self.control_model_wrapped
        self.copy_to(c)
        return c

class ControlLoraOps:
    class Linear(torch.nn.Module, comfy.ops.CastWeightBiasOp):
        def __init__(self, in_features: int, out_features: int, bias: bool = True,
                    device=None, dtype=None) -> None:
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.up = None
            self.down = None
            self.bias = None

        def forward(self, input):
            weight, bias, offload_stream = comfy.ops.cast_bias_weight(self, input, offloadable=True)
            if self.up is not None:
                x = torch.nn.functional.linear(input, weight + (torch.mm(self.up.flatten(start_dim=1), self.down.flatten(start_dim=1))).reshape(self.weight.shape).type(input.dtype), bias)
            else:
                x = torch.nn.functional.linear(input, weight, bias)
            comfy.ops.uncast_bias_weight(self, weight, bias, offload_stream)
            return x

    class Conv2d(torch.nn.Module, comfy.ops.CastWeightBiasOp):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros',
            device=None,
            dtype=None
        ):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.transposed = False
            self.output_padding = 0
            self.groups = groups
            self.padding_mode = padding_mode

            self.weight = None
            self.bias = None
            self.up = None
            self.down = None


        def forward(self, input):
            weight, bias, offload_stream = comfy.ops.cast_bias_weight(self, input, offloadable=True)
            if self.up is not None:
                x = torch.nn.functional.conv2d(input, weight + (torch.mm(self.up.flatten(start_dim=1), self.down.flatten(start_dim=1))).reshape(self.weight.shape).type(input.dtype), bias, self.stride, self.padding, self.dilation, self.groups)
            else:
                x = torch.nn.functional.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
            comfy.ops.uncast_bias_weight(self, weight, bias, offload_stream)
            return x

class ControlLora(ControlNet):
    def __init__(self, control_weights, global_average_pooling=False, model_options={}): #TODO? model_options
        ControlBase.__init__(self)
        self.control_weights = control_weights
        self.global_average_pooling = global_average_pooling
        self.extra_conds += ["y"]

    def pre_run(self, model, percent_to_timestep_function):
        super().pre_run(model, percent_to_timestep_function)
        controlnet_config = model.model_config.unet_config.copy()
        controlnet_config.pop("out_channels")
        controlnet_config["hint_channels"] = self.control_weights["input_hint_block.0.weight"].shape[1]
        self.manual_cast_dtype = model.manual_cast_dtype
        dtype = model.get_dtype()
        if self.manual_cast_dtype is None:
            class control_lora_ops(ControlLoraOps, comfy.ops.disable_weight_init):
                pass
        else:
            class control_lora_ops(ControlLoraOps, comfy.ops.manual_cast):
                pass
            dtype = self.manual_cast_dtype

        controlnet_config["operations"] = control_lora_ops
        controlnet_config["dtype"] = dtype
        self.control_model = comfy.cldm.cldm.ControlNet(**controlnet_config)
        self.control_model.to(comfy.model_management.get_torch_device())
        diffusion_model = model.diffusion_model
        sd = diffusion_model.state_dict()

        for k in sd:
            weight = sd[k]
            try:
                comfy.utils.set_attr_param(self.control_model, k, weight)
            except:
                pass

        for k in self.control_weights:
            if (k not in {"lora_controlnet"}):
                if (k.endswith(".up") or k.endswith(".down") or k.endswith(".weight") or k.endswith(".bias")) and ("__" not in k):
                    comfy.utils.set_attr_param(self.control_model, k, self.control_weights[k].to(dtype).to(comfy.model_management.get_torch_device()))

    def copy(self):
        c = ControlLora(self.control_weights, global_average_pooling=self.global_average_pooling)
        self.copy_to(c)
        return c

    def cleanup(self):
        del self.control_model
        self.control_model = None
        super().cleanup()

    def get_models(self):
        out = ControlBase.get_models(self)
        return out

    def inference_memory_requirements(self, dtype):
        return comfy.utils.calculate_parameters(self.control_weights) * comfy.model_management.dtype_size(dtype) + ControlBase.inference_memory_requirements(self, dtype)

def controlnet_config(sd, model_options={}):
    model_config = comfy.model_detection.model_config_from_unet(sd, "", True)

    unet_dtype = model_options.get("dtype", None)
    if unet_dtype is None:
        weight_dtype = comfy.utils.weight_dtype(sd)

        supported_inference_dtypes = list(model_config.supported_inference_dtypes)
        unet_dtype = comfy.model_management.unet_dtype(model_params=-1, supported_dtypes=supported_inference_dtypes, weight_dtype=weight_dtype)

    load_device = comfy.model_management.get_torch_device()
    manual_cast_dtype = comfy.model_management.unet_manual_cast(unet_dtype, load_device)

    operations = model_options.get("custom_operations", None)
    if operations is None:
        operations = comfy.ops.pick_operations(unet_dtype, manual_cast_dtype, disable_fast_fp8=True)

    offload_device = comfy.model_management.unet_offload_device()
    return model_config, operations, load_device, unet_dtype, manual_cast_dtype, offload_device

def controlnet_load_state_dict(control_model, sd):
    missing, unexpected = control_model.load_state_dict(sd, strict=False)

    if len(missing) > 0:
        logging.warning("missing controlnet keys: {}".format(missing))

    if len(unexpected) > 0:
        logging.debug("unexpected controlnet keys: {}".format(unexpected))
    return control_model


def load_controlnet_mmdit(sd, model_options={}):
    new_sd = comfy.model_detection.convert_diffusers_mmdit(sd, "")
    model_config, operations, load_device, unet_dtype, manual_cast_dtype, offload_device = controlnet_config(new_sd, model_options=model_options)
    num_blocks = comfy.model_detection.count_blocks(new_sd, 'joint_blocks.{}.')
    for k in sd:
        new_sd[k] = sd[k]

    concat_mask = False
    control_latent_channels = new_sd.get("pos_embed_input.proj.weight").shape[1]
    if control_latent_channels == 17: #inpaint controlnet
        concat_mask = True

    control_model = comfy.cldm.mmdit.ControlNet(num_blocks=num_blocks, control_latent_channels=control_latent_channels, operations=operations, device=offload_device, dtype=unet_dtype, **model_config.unet_config)
    control_model = controlnet_load_state_dict(control_model, new_sd)

    latent_format = comfy.latent_formats.SD3()
    latent_format.shift_factor = 0 #SD3 controlnet weirdness
    control = ControlNet(control_model, compression_ratio=1, latent_format=latent_format, concat_mask=concat_mask, load_device=load_device, manual_cast_dtype=manual_cast_dtype)
    return control


class ControlNetSD35(ControlNet):
    def pre_run(self, model, percent_to_timestep_function):
        if self.control_model.double_y_emb:
            missing, unexpected = self.control_model.orig_y_embedder.load_state_dict(model.diffusion_model.y_embedder.state_dict(), strict=False)
        else:
            missing, unexpected = self.control_model.x_embedder.load_state_dict(model.diffusion_model.x_embedder.state_dict(), strict=False)
        super().pre_run(model, percent_to_timestep_function)

    def copy(self):
        c = ControlNetSD35(None, global_average_pooling=self.global_average_pooling, load_device=self.load_device, manual_cast_dtype=self.manual_cast_dtype)
        c.control_model = self.control_model
        c.control_model_wrapped = self.control_model_wrapped
        self.copy_to(c)
        return c

def load_controlnet_sd35(sd, model_options={}):
    control_type = -1
    if "control_type" in sd:
        control_type = round(sd.pop("control_type").item())

    # blur_cnet = control_type == 0
    canny_cnet = control_type == 1
    depth_cnet = control_type == 2

    new_sd = {}
    for k in comfy.utils.MMDIT_MAP_BASIC:
        if k[1] in sd:
            new_sd[k[0]] = sd.pop(k[1])
    for k in sd:
        new_sd[k] = sd[k]
    sd = new_sd

    y_emb_shape = sd["y_embedder.mlp.0.weight"].shape
    depth = y_emb_shape[0] // 64
    hidden_size = 64 * depth
    num_heads = depth
    head_dim = hidden_size // num_heads
    num_blocks = comfy.model_detection.count_blocks(new_sd, 'transformer_blocks.{}.')

    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    unet_dtype = comfy.model_management.unet_dtype(model_params=-1)

    manual_cast_dtype = comfy.model_management.unet_manual_cast(unet_dtype, load_device)

    operations = model_options.get("custom_operations", None)
    if operations is None:
        operations = comfy.ops.pick_operations(unet_dtype, manual_cast_dtype, disable_fast_fp8=True)

    control_model = comfy.cldm.dit_embedder.ControlNetEmbedder(img_size=None,
                                                               patch_size=2,
                                                               in_chans=16,
                                                               num_layers=num_blocks,
                                                               main_model_double=depth,
                                                               double_y_emb=y_emb_shape[0] == y_emb_shape[1],
                                                               attention_head_dim=head_dim,
                                                               num_attention_heads=num_heads,
                                                               adm_in_channels=2048,
                                                               device=offload_device,
                                                               dtype=unet_dtype,
                                                               operations=operations)

    control_model = controlnet_load_state_dict(control_model, sd)

    latent_format = comfy.latent_formats.SD3()
    preprocess_image = lambda a: a
    if canny_cnet:
        preprocess_image = lambda a: (a * 255 * 0.5 + 0.5)
    elif depth_cnet:
        preprocess_image = lambda a: 1.0 - a

    control = ControlNetSD35(control_model, compression_ratio=1, latent_format=latent_format, load_device=load_device, manual_cast_dtype=manual_cast_dtype, preprocess_image=preprocess_image)
    return control



def load_controlnet_hunyuandit(controlnet_data, model_options={}):
    model_config, operations, load_device, unet_dtype, manual_cast_dtype, offload_device = controlnet_config(controlnet_data, model_options=model_options)

    control_model = comfy.ldm.hydit.controlnet.HunYuanControlNet(operations=operations, device=offload_device, dtype=unet_dtype)
    control_model = controlnet_load_state_dict(control_model, controlnet_data)

    latent_format = comfy.latent_formats.SDXL()
    extra_conds = ['text_embedding_mask', 'encoder_hidden_states_t5', 'text_embedding_mask_t5', 'image_meta_size', 'style', 'cos_cis_img', 'sin_cis_img']
    control = ControlNet(control_model, compression_ratio=1, latent_format=latent_format, load_device=load_device, manual_cast_dtype=manual_cast_dtype, extra_conds=extra_conds, strength_type=StrengthType.CONSTANT)
    return control

def load_controlnet_flux_xlabs_mistoline(sd, mistoline=False, model_options={}):
    model_config, operations, load_device, unet_dtype, manual_cast_dtype, offload_device = controlnet_config(sd, model_options=model_options)
    control_model = comfy.ldm.flux.controlnet.ControlNetFlux(mistoline=mistoline, operations=operations, device=offload_device, dtype=unet_dtype, **model_config.unet_config)
    sd = model_config.process_unet_state_dict(sd)
    control_model = controlnet_load_state_dict(control_model, sd)
    extra_conds = ['y', 'guidance']
    control = ControlNet(control_model, load_device=load_device, manual_cast_dtype=manual_cast_dtype, extra_conds=extra_conds)
    return control

def load_controlnet_flux_instantx(sd, model_options={}):
    new_sd = comfy.model_detection.convert_diffusers_mmdit(sd, "")
    model_config, operations, load_device, unet_dtype, manual_cast_dtype, offload_device = controlnet_config(new_sd, model_options=model_options)
    for k in sd:
        new_sd[k] = sd[k]

    num_union_modes = 0
    union_cnet = "controlnet_mode_embedder.weight"
    if union_cnet in new_sd:
        num_union_modes = new_sd[union_cnet].shape[0]

    control_latent_channels = new_sd.get("pos_embed_input.weight").shape[1] // 4
    concat_mask = False
    if control_latent_channels == 17:
        concat_mask = True

    control_model = comfy.ldm.flux.controlnet.ControlNetFlux(latent_input=True, num_union_modes=num_union_modes, control_latent_channels=control_latent_channels, operations=operations, device=offload_device, dtype=unet_dtype, **model_config.unet_config)
    control_model = controlnet_load_state_dict(control_model, new_sd)

    latent_format = comfy.latent_formats.Flux()
    extra_conds = ['y', 'guidance']
    control = ControlNet(control_model, compression_ratio=1, latent_format=latent_format, concat_mask=concat_mask, load_device=load_device, manual_cast_dtype=manual_cast_dtype, extra_conds=extra_conds)
    return control

def load_controlnet_qwen_instantx(sd, model_options={}):
    model_config, operations, load_device, unet_dtype, manual_cast_dtype, offload_device = controlnet_config(sd, model_options=model_options)
    control_latent_channels = sd.get("controlnet_x_embedder.weight").shape[1]

    extra_condition_channels = 0
    concat_mask = False
    if control_latent_channels == 68: #inpaint controlnet
        extra_condition_channels = control_latent_channels - 64
        concat_mask = True
    control_model = comfy.ldm.qwen_image.controlnet.QwenImageControlNetModel(extra_condition_channels=extra_condition_channels, operations=operations, device=offload_device, dtype=unet_dtype, **model_config.unet_config)
    control_model = controlnet_load_state_dict(control_model, sd)
    latent_format = comfy.latent_formats.Wan21()
    extra_conds = []
    control = ControlNet(control_model, compression_ratio=1, latent_format=latent_format, concat_mask=concat_mask, load_device=load_device, manual_cast_dtype=manual_cast_dtype, extra_conds=extra_conds)
    return control


def load_controlnet_qwen_fun(sd, model_options={}):
    load_device = comfy.model_management.get_torch_device()
    weight_dtype = comfy.utils.weight_dtype(sd)
    unet_dtype = model_options.get("dtype", weight_dtype)
    manual_cast_dtype = comfy.model_management.unet_manual_cast(unet_dtype, load_device)

    operations = model_options.get("custom_operations", None)
    if operations is None:
        operations = comfy.ops.pick_operations(unet_dtype, manual_cast_dtype, disable_fast_fp8=True)

    in_features = sd["control_img_in.weight"].shape[1]
    inner_dim = sd["control_img_in.weight"].shape[0]

    block_weight = sd["control_blocks.0.attn.to_q.weight"]
    attention_head_dim = sd["control_blocks.0.attn.norm_q.weight"].shape[0]
    num_attention_heads = max(1, block_weight.shape[0] // max(1, attention_head_dim))

    model = comfy.ldm.qwen_image.controlnet.QwenImageFunControlNetModel(
        control_in_features=in_features,
        inner_dim=inner_dim,
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        num_control_blocks=5,
        main_model_double=60,
        injection_layers=(0, 12, 24, 36, 48),
        operations=operations,
        device=comfy.model_management.unet_offload_device(),
        dtype=unet_dtype,
    )
    model = controlnet_load_state_dict(model, sd)

    latent_format = comfy.latent_formats.Wan21()
    control = QwenFunControlNet(
        model,
        compression_ratio=1,
        latent_format=latent_format,
        # Fun checkpoints already expect their own 33-channel context handling.
        # Enabling generic concat_mask injects an extra mask channel at apply-time
        # and breaks the intended fallback packing path.
        concat_mask=False,
        load_device=load_device,
        manual_cast_dtype=manual_cast_dtype,
        extra_conds=[],
    )
    return control

def convert_mistoline(sd):
    return comfy.utils.state_dict_prefix_replace(sd, {"single_controlnet_blocks.": "controlnet_single_blocks."})


def load_controlnet_state_dict(state_dict, model=None, model_options={}):
    controlnet_data = state_dict
    if 'after_proj_list.18.bias' in controlnet_data.keys(): #Hunyuan DiT
        return load_controlnet_hunyuandit(controlnet_data, model_options=model_options)

    if "lora_controlnet" in controlnet_data:
        return ControlLora(controlnet_data, model_options=model_options)

    controlnet_config = None
    supported_inference_dtypes = None

    if "controlnet_cond_embedding.conv_in.weight" in controlnet_data: #diffusers format
        controlnet_config = comfy.model_detection.unet_config_from_diffusers_unet(controlnet_data)
        diffusers_keys = comfy.utils.unet_to_diffusers(controlnet_config)
        diffusers_keys["controlnet_mid_block.weight"] = "middle_block_out.0.weight"
        diffusers_keys["controlnet_mid_block.bias"] = "middle_block_out.0.bias"

        count = 0
        loop = True
        while loop:
            suffix = [".weight", ".bias"]
            for s in suffix:
                k_in = "controlnet_down_blocks.{}{}".format(count, s)
                k_out = "zero_convs.{}.0{}".format(count, s)
                if k_in not in controlnet_data:
                    loop = False
                    break
                diffusers_keys[k_in] = k_out
            count += 1

        count = 0
        loop = True
        while loop:
            suffix = [".weight", ".bias"]
            for s in suffix:
                if count == 0:
                    k_in = "controlnet_cond_embedding.conv_in{}".format(s)
                else:
                    k_in = "controlnet_cond_embedding.blocks.{}{}".format(count - 1, s)
                k_out = "input_hint_block.{}{}".format(count * 2, s)
                if k_in not in controlnet_data:
                    k_in = "controlnet_cond_embedding.conv_out{}".format(s)
                    loop = False
                diffusers_keys[k_in] = k_out
            count += 1

        new_sd = {}
        for k in diffusers_keys:
            if k in controlnet_data:
                new_sd[diffusers_keys[k]] = controlnet_data.pop(k)

        if "control_add_embedding.linear_1.bias" in controlnet_data: #Union Controlnet
            controlnet_config["union_controlnet_num_control_type"] = controlnet_data["task_embedding"].shape[0]
            for k in list(controlnet_data.keys()):
                new_k = k.replace('.attn.in_proj_', '.attn.in_proj.')
                new_sd[new_k] = controlnet_data.pop(k)

        leftover_keys = controlnet_data.keys()
        if len(leftover_keys) > 0:
            logging.warning("leftover keys: {}".format(leftover_keys))
        controlnet_data = new_sd
    elif "controlnet_blocks.0.weight" in controlnet_data:
        if "double_blocks.0.img_attn.norm.key_norm.scale" in controlnet_data:
            return load_controlnet_flux_xlabs_mistoline(controlnet_data, model_options=model_options)
        elif "pos_embed_input.proj.weight" in controlnet_data:
            if "transformer_blocks.0.adaLN_modulation.1.bias" in controlnet_data:
                return load_controlnet_sd35(controlnet_data, model_options=model_options) #Stability sd3.5 format
            else:
                return load_controlnet_mmdit(controlnet_data, model_options=model_options) #SD3 diffusers controlnet
        elif "transformer_blocks.0.img_mlp.net.0.proj.weight" in controlnet_data:
            return load_controlnet_qwen_instantx(controlnet_data, model_options=model_options)
        elif "controlnet_x_embedder.weight" in controlnet_data:
            return load_controlnet_flux_instantx(controlnet_data, model_options=model_options)
    elif "control_blocks.0.after_proj.weight" in controlnet_data and "control_img_in.weight" in controlnet_data:
        return load_controlnet_qwen_fun(controlnet_data, model_options=model_options)

    elif "controlnet_blocks.0.linear.weight" in controlnet_data: #mistoline flux
        return load_controlnet_flux_xlabs_mistoline(convert_mistoline(controlnet_data), mistoline=True, model_options=model_options)

    pth_key = 'control_model.zero_convs.0.0.weight'
    pth = False
    key = 'zero_convs.0.0.weight'
    if pth_key in controlnet_data:
        pth = True
        key = pth_key
        prefix = "control_model."
    elif key in controlnet_data:
        prefix = ""
    else:
        net = load_t2i_adapter(controlnet_data, model_options=model_options)
        if net is None:
            logging.error("error could not detect control model type.")
        return net

    if controlnet_config is None:
        model_config = comfy.model_detection.model_config_from_unet(controlnet_data, prefix, True)
        supported_inference_dtypes = list(model_config.supported_inference_dtypes)
        controlnet_config = model_config.unet_config

    unet_dtype = model_options.get("dtype", None)
    if unet_dtype is None:
        weight_dtype = comfy.utils.weight_dtype(controlnet_data)

        if supported_inference_dtypes is None:
            supported_inference_dtypes = [comfy.model_management.unet_dtype()]

        unet_dtype = comfy.model_management.unet_dtype(model_params=-1, supported_dtypes=supported_inference_dtypes, weight_dtype=weight_dtype)

    load_device = comfy.model_management.get_torch_device()

    manual_cast_dtype = comfy.model_management.unet_manual_cast(unet_dtype, load_device)
    operations = model_options.get("custom_operations", None)
    if operations is None:
        operations = comfy.ops.pick_operations(unet_dtype, manual_cast_dtype)

    controlnet_config["operations"] = operations
    controlnet_config["dtype"] = unet_dtype
    controlnet_config["device"] = comfy.model_management.unet_offload_device()
    controlnet_config.pop("out_channels")
    controlnet_config["hint_channels"] = controlnet_data["{}input_hint_block.0.weight".format(prefix)].shape[1]
    control_model = comfy.cldm.cldm.ControlNet(**controlnet_config)

    if pth:
        if 'difference' in controlnet_data:
            if model is not None:
                comfy.model_management.load_models_gpu([model])
                model_sd = model.model_state_dict()
                for x in controlnet_data:
                    c_m = "control_model."
                    if x.startswith(c_m):
                        sd_key = "diffusion_model.{}".format(x[len(c_m):])
                        if sd_key in model_sd:
                            cd = controlnet_data[x]
                            cd += model_sd[sd_key].type(cd.dtype).to(cd.device)
            else:
                logging.warning("WARNING: Loaded a diff controlnet without a model. It will very likely not work.")

        class WeightsLoader(torch.nn.Module):
            pass
        w = WeightsLoader()
        w.control_model = control_model
        missing, unexpected = w.load_state_dict(controlnet_data, strict=False)
    else:
        missing, unexpected = control_model.load_state_dict(controlnet_data, strict=False)

    if len(missing) > 0:
        logging.warning("missing controlnet keys: {}".format(missing))

    if len(unexpected) > 0:
        logging.debug("unexpected controlnet keys: {}".format(unexpected))

    global_average_pooling = model_options.get("global_average_pooling", False)
    control = ControlNet(control_model, global_average_pooling=global_average_pooling, load_device=load_device, manual_cast_dtype=manual_cast_dtype)
    return control

def load_controlnet(ckpt_path, model=None, model_options={}):
    model_options = model_options.copy()
    if "global_average_pooling" not in model_options:
        filename = os.path.splitext(ckpt_path)[0]
        if filename.endswith("_shuffle") or filename.endswith("_shuffle_fp16"): #TODO: smarter way of enabling global_average_pooling
            model_options["global_average_pooling"] = True

    cnet = load_controlnet_state_dict(comfy.utils.load_torch_file(ckpt_path, safe_load=True), model=model, model_options=model_options)
    if cnet is None:
        logging.error("error checkpoint does not contain controlnet or t2i adapter data {}".format(ckpt_path))
    return cnet

class T2IAdapter(ControlBase):
    def __init__(self, t2i_model, channels_in, compression_ratio, upscale_algorithm, device=None):
        super().__init__()
        self.t2i_model = t2i_model
        self.channels_in = channels_in
        self.control_input = None
        self.compression_ratio = compression_ratio
        self.upscale_algorithm = upscale_algorithm
        if device is None:
            device = comfy.model_management.get_torch_device()
        self.device = device

    def scale_image_to(self, width, height):
        unshuffle_amount = self.t2i_model.unshuffle_amount
        width = math.ceil(width / unshuffle_amount) * unshuffle_amount
        height = math.ceil(height / unshuffle_amount) * unshuffle_amount
        return width, height

    def get_control(self, x_noisy, t, cond, batched_number, transformer_options):
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number, transformer_options)

        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                if control_prev is not None:
                    return control_prev
                else:
                    return None

        if self.cond_hint is None or x_noisy.shape[2] * self.compression_ratio != self.cond_hint.shape[2] or x_noisy.shape[3] * self.compression_ratio != self.cond_hint.shape[3]:
            if self.cond_hint is not None:
                del self.cond_hint
            self.control_input = None
            self.cond_hint = None
            width, height = self.scale_image_to(x_noisy.shape[3] * self.compression_ratio, x_noisy.shape[2] * self.compression_ratio)
            self.cond_hint = comfy.utils.common_upscale(self.cond_hint_original, width, height, self.upscale_algorithm, "center").float().to(self.device)
            if self.channels_in == 1 and self.cond_hint.shape[1] > 1:
                self.cond_hint = torch.mean(self.cond_hint, 1, keepdim=True)
        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = broadcast_image_to(self.cond_hint, x_noisy.shape[0], batched_number)
        if self.control_input is None:
            self.t2i_model.to(x_noisy.dtype)
            self.t2i_model.to(self.device)
            self.control_input = self.t2i_model(self.cond_hint.to(x_noisy.dtype))
            self.t2i_model.cpu()

        control_input = {}
        for k in self.control_input:
            control_input[k] = list(map(lambda a: None if a is None else a.clone(), self.control_input[k]))

        return self.control_merge(control_input, control_prev, x_noisy.dtype)

    def copy(self):
        c = T2IAdapter(self.t2i_model, self.channels_in, self.compression_ratio, self.upscale_algorithm)
        self.copy_to(c)
        return c

    def deepclone_multigpu(self, load_device, autoregister=False):
        c = self.copy()
        c.t2i_model = copy.deepcopy(c.t2i_model)
        c.device = load_device
        if autoregister:
            self.multigpu_clones[load_device] = c
        return c

def load_t2i_adapter(t2i_data, model_options={}): #TODO: model_options
    compression_ratio = 8
    upscale_algorithm = 'nearest-exact'

    if 'adapter' in t2i_data:
        t2i_data = t2i_data['adapter']
    if 'adapter.body.0.resnets.0.block1.weight' in t2i_data: #diffusers format
        prefix_replace = {}
        for i in range(4):
            for j in range(2):
                prefix_replace["adapter.body.{}.resnets.{}.".format(i, j)] = "body.{}.".format(i * 2 + j)
            prefix_replace["adapter.body.{}.".format(i, )] = "body.{}.".format(i * 2)
        prefix_replace["adapter."] = ""
        t2i_data = comfy.utils.state_dict_prefix_replace(t2i_data, prefix_replace)
    keys = t2i_data.keys()

    if "body.0.in_conv.weight" in keys:
        cin = t2i_data['body.0.in_conv.weight'].shape[1]
        model_ad = comfy.t2i_adapter.adapter.Adapter_light(cin=cin, channels=[320, 640, 1280, 1280], nums_rb=4)
    elif 'conv_in.weight' in keys:
        cin = t2i_data['conv_in.weight'].shape[1]
        channel = t2i_data['conv_in.weight'].shape[0]
        ksize = t2i_data['body.0.block2.weight'].shape[2]
        use_conv = False
        down_opts = list(filter(lambda a: a.endswith("down_opt.op.weight"), keys))
        if len(down_opts) > 0:
            use_conv = True
        xl = False
        if cin == 256 or cin == 768:
            xl = True
        model_ad = comfy.t2i_adapter.adapter.Adapter(cin=cin, channels=[channel, channel*2, channel*4, channel*4][:4], nums_rb=2, ksize=ksize, sk=True, use_conv=use_conv, xl=xl)
    elif "backbone.0.0.weight" in keys:
        model_ad = comfy.ldm.cascade.controlnet.ControlNet(c_in=t2i_data['backbone.0.0.weight'].shape[1], proj_blocks=[0, 4, 8, 12, 51, 55, 59, 63])
        compression_ratio = 32
        upscale_algorithm = 'bilinear'
    elif "backbone.10.blocks.0.weight" in keys:
        model_ad = comfy.ldm.cascade.controlnet.ControlNet(c_in=t2i_data['backbone.0.weight'].shape[1], bottleneck_mode="large", proj_blocks=[0, 4, 8, 12, 51, 55, 59, 63])
        compression_ratio = 1
        upscale_algorithm = 'nearest-exact'
    else:
        return None

    missing, unexpected = model_ad.load_state_dict(t2i_data)
    if len(missing) > 0:
        logging.warning("t2i missing {}".format(missing))

    if len(unexpected) > 0:
        logging.debug("t2i unexpected {}".format(unexpected))

    return T2IAdapter(model_ad, model_ad.input_channels, compression_ratio, upscale_algorithm)
