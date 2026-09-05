"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Stability AI

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

import torch
import logging
import contextlib
import inspect
import comfy.model_management
from comfy.cli_args import args, PerformanceFeature
import comfy.float
import json
import comfy.memory_management
import comfy.pinned_memory
import comfy.utils

import comfy_aimdo.model_vbar
import comfy_aimdo.torch

def run_every_op():
    if torch.compiler.is_compiling():
        return

    comfy.model_management.throw_exception_if_processing_interrupted()

def gqa_repeat_factor(query_heads, key_heads, value_heads):
    if key_heads != value_heads:
        raise ValueError(f"Key/value head count mismatch for GQA: {key_heads} != {value_heads}")
    if query_heads == key_heads:
        return 1
    if query_heads % key_heads != 0:
        raise ValueError(f"Query heads must be divisible by key/value heads for GQA: {query_heads} vs {key_heads}")
    return query_heads // key_heads

def repeat_kv_for_gqa(k, v, query_heads, head_dim):
    n_rep = gqa_repeat_factor(query_heads, k.shape[head_dim], v.shape[head_dim])
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=head_dim)
        v = v.repeat_interleave(n_rep, dim=head_dim)
    return k, v

def scaled_dot_product_attention(q, k, v, *args, **kwargs):
    attn_mask = args[0] if len(args) > 0 else kwargs.get("attn_mask")
    if kwargs.get("enable_gqa", False) and attn_mask is not None:
        k, v = repeat_kv_for_gqa(k, v, q.shape[-3], -3)
        kwargs["enable_gqa"] = False
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, *args, **kwargs)


try:
    if torch.cuda.is_available():
        from torch.nn.attention import SDPBackend, sdpa_kernel
        if "set_priority" in inspect.signature(sdpa_kernel).parameters:
            SDPA_BACKEND_PRIORITY = [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]

            def scaled_dot_product_attention(q, k, v, *args, **kwargs):
                if q.nelement() < 1024 * 128:  # arbitrary number, for small inputs cudnn attention seems slower
                    return torch.nn.functional.scaled_dot_product_attention(q, k, v, *args, **kwargs)
                attn_mask = args[0] if len(args) > 0 else kwargs.get("attn_mask")
                if kwargs.get("enable_gqa", False) and attn_mask is not None and not comfy.model_management.is_nvidia():
                    k, v = repeat_kv_for_gqa(k, v, q.shape[-3], -3)
                    kwargs["enable_gqa"] = False
                with sdpa_kernel(SDPA_BACKEND_PRIORITY, set_priority=True):
                    if kwargs.get("enable_gqa", False) and attn_mask is not None and q.shape[-3] != k.shape[-3]:
                        dropout_p = args[1] if len(args) > 1 else kwargs.get("dropout_p", 0.0)
                        is_causal = args[2] if len(args) > 2 else kwargs.get("is_causal", False)
                        params = torch.backends.cuda.SDPAParams(q, k, v, attn_mask, dropout_p, is_causal, True)
                        supports_native_gqa = (
                            torch.backends.cuda.can_use_flash_attention(params)
                            or torch.backends.cuda.can_use_cudnn_attention(params)
                            or torch.backends.cuda.can_use_efficient_attention(params)
                        )
                        if not supports_native_gqa:
                            k, v = repeat_kv_for_gqa(k, v, q.shape[-3], -3)
                            kwargs["enable_gqa"] = False
                    return torch.nn.functional.scaled_dot_product_attention(q, k, v, *args, **kwargs)
        else:
            logging.warning("Torch version too old to set sdpa backend priority.")
except (ModuleNotFoundError, TypeError):
    logging.warning("Could not set sdpa backend priority.")

NVIDIA_MEMORY_CONV_BUG_WORKAROUND = False
try:
    if comfy.model_management.is_nvidia():
        cudnn_version = torch.backends.cudnn.version()
        if (cudnn_version >= 91002 and cudnn_version < 91500) and comfy.model_management.torch_version_numeric >= (2, 9) and comfy.model_management.torch_version_numeric <= (2, 10):
            #TODO: change upper bound version once it's fixed'
            NVIDIA_MEMORY_CONV_BUG_WORKAROUND = True
            logging.info("working around nvidia conv3d memory bug.")
except:
    pass

cast_to = comfy.model_management.cast_to #TODO: remove once no more references

def cast_to_input(weight, input, non_blocking=False, copy=True):
    return comfy.model_management.cast_to(weight, input.dtype, input.device, non_blocking=non_blocking, copy=copy)


def materialize_meta_param(s, param_keys):
    for param_key in param_keys:
        param = getattr(s, param_key, None)
        if param is not None and getattr(param, "is_meta", False):
            setattr(s, param_key, torch.nn.Parameter(torch.zeros(param.shape, dtype=param.dtype), requires_grad=param.requires_grad))


# FIXME: add n=1 cache hit fast path
def cast_modules_with_vbar(comfy_modules, dtype, device, bias_dtype, non_blocking, return_faulted=False):
    offload_stream = None
    cast_buffer = None
    cast_buffer_offset = 0
    if return_faulted:
        fully_faulted = all(not getattr(s, param_key + "_function", []) for s in comfy_modules for param_key in ("weight", "bias"))

    def ensure_offload_stream(module, required_size, check_largest):
        nonlocal offload_stream
        nonlocal cast_buffer

        if offload_stream is None:
            offload_stream = comfy.model_management.get_offload_stream(device)
        if offload_stream is None or not check_largest or len(comfy_modules) != 1:
            return

        current_size = 0 if cast_buffer is None else cast_buffer.size()
        if current_size < required_size and module is comfy.model_management.LARGEST_AIMDO_CASTED_WEIGHT[0]:
            offload_stream = comfy.model_management.get_offload_stream(device)
            cast_buffer = None
        if required_size > comfy.model_management.LARGEST_AIMDO_CASTED_WEIGHT[1]:
            comfy.model_management.LARGEST_AIMDO_CASTED_WEIGHT = (module, required_size)

    def get_cast_buffer(buffer_size):
        nonlocal offload_stream
        nonlocal cast_buffer
        nonlocal cast_buffer_offset

        if buffer_size == 0:
            return None

        if offload_stream is None:
            return torch.empty((buffer_size,), dtype=torch.uint8, device=device)

        cast_buffer = comfy.model_management.get_aimdo_cast_buffer(offload_stream, device)
        buffer = comfy_aimdo.torch.aimdo_to_tensor(cast_buffer.get(buffer_size, cast_buffer_offset), device)
        cast_buffer_offset += buffer_size
        return buffer

    for s in comfy_modules:
        signature = comfy_aimdo.model_vbar.vbar_fault(s._v)
        resident = comfy_aimdo.model_vbar.vbar_signature_compare(signature, s._v_signature)
        if return_faulted and (signature is None or not resident):
            fully_faulted = False
        prefetch = {
            "signature": signature,
            "resident": resident,
        }

        if resident:
            s._prefetch = prefetch
            continue

        materialize_meta_param(s, ["weight", "bias"])
        xfer_dest = comfy_aimdo.torch.aimdo_to_tensor(s._v, device) if signature is not None else None
        cast_geometry = comfy.memory_management.tensors_to_geometries([ s.weight, s.bias ])
        cast_dest = None
        needs_cast = False

        xfer_source = [ s.weight, s.bias ]
        subset = "weights"
        pin = comfy.pinned_memory.get_pin(s, subset=subset)
        if pin is None and not args.fast_disk:
            loaded_pin = comfy.pinned_memory.get_pin(s, subset="weights-loaded")
            if loaded_pin is not None or signature is not None:
                subset = "weights-loaded"
                pin = loaded_pin
        if pin is not None:
            xfer_source = [ pin ]

        for data, geometry in zip([ s.weight, s.bias ], cast_geometry):
            if data is None:
                continue
            if data.dtype != geometry.dtype:
                needs_cast = True
                cast_dest = xfer_dest
                xfer_dest = None
                break

        dest_size = comfy.memory_management.vram_aligned_size(xfer_source)
        ensure_offload_stream(s, dest_size if xfer_dest is None else 0, True)
        if xfer_dest is None:
            xfer_dest = get_cast_buffer(dest_size)

        def cast_maybe_lowvram_patch(xfer_source, xfer_dest, stream, xfer_dest2=None):
            if xfer_source is not None:
                if getattr(xfer_source, "is_lowvram_patch", False):
                    if xfer_dest is not None:
                        xfer_source.prepare(xfer_dest, stream, copy=True, commit=False)
                        xfer_source = [ xfer_dest ]
                        xfer_dest = xfer_dest2
                        xfer_dest2 = None
                    elif xfer_dest2 is not None:
                        xfer_source.prepare(xfer_dest2, stream, copy=True, commit=False)
                        return
                    else:
                        return
                comfy.model_management.cast_to_gathered(xfer_source, xfer_dest, non_blocking=non_blocking, stream=stream, r2=xfer_dest2)

        def handle_pin(m, pin, source, dest, subset="weights", size=None):
            if pin is not None:
                cast_maybe_lowvram_patch([pin], dest, offload_stream)
                return
            if signature is None or not args.fast_disk or args.high_ram:
                comfy.pinned_memory.pin_memory(m, subset=subset, size=size)
                pin = comfy.pinned_memory.get_pin(m, subset=subset)
            cast_maybe_lowvram_patch(source, pin, offload_stream, xfer_dest2=dest)

        handle_pin(s, pin, xfer_source, xfer_dest, subset=subset, size=dest_size)

        for param_key in ("weight", "bias"):
            lowvram_source = getattr(s, param_key + "_lowvram_function", None)
            if lowvram_source is not None:
                ensure_offload_stream(s, cast_buffer_offset, False)
                lowvram_size = lowvram_source.memory_required()
                lowvram_dest = get_cast_buffer(lowvram_size)
                lowvram_source.prepare(lowvram_dest, None, copy=False, commit=True)

                subset = "patches"
                pin = comfy.pinned_memory.get_pin(lowvram_source, subset=subset)
                if pin is None:
                    loaded_pin = comfy.pinned_memory.get_pin(lowvram_source, subset="patches-loaded")
                    if loaded_pin is not None:
                        subset = "patches-loaded"
                        pin = loaded_pin
                    elif signature is not None and not args.fast_disk:
                        subset = "patches-loaded"
                handle_pin(lowvram_source, pin, lowvram_source, lowvram_dest, subset=subset, size=lowvram_size)


        prefetch["xfer_dest"] = xfer_dest
        prefetch["cast_dest"] = cast_dest
        prefetch["cast_geometry"] = cast_geometry
        prefetch["needs_cast"] = needs_cast
        s._prefetch = prefetch

    if return_faulted:
        return offload_stream, fully_faulted
    return offload_stream


def resolve_cast_module_with_vbar(s, dtype, device, bias_dtype, compute_dtype, want_requant, return_weights=True):

    prefetch = getattr(s, "_prefetch", None)

    if prefetch["resident"]:
        weight = s._v_weight
        bias = s._v_bias
    else:
        xfer_dest = prefetch["xfer_dest"]
        if prefetch["needs_cast"]:
            cast_dest = prefetch["cast_dest"] if prefetch["cast_dest"] is not None else torch.empty((comfy.memory_management.vram_aligned_size(prefetch["cast_geometry"]),), dtype=torch.uint8, device=device)
            for pre_cast, post_cast in zip(comfy.memory_management.interpret_gathered_like([s.weight, s.bias ], xfer_dest),
                                           comfy.memory_management.interpret_gathered_like(prefetch["cast_geometry"], cast_dest)):
                if post_cast is not None:
                    post_cast.copy_(pre_cast)
            xfer_dest = cast_dest

        params = comfy.memory_management.interpret_gathered_like(prefetch["cast_geometry"], xfer_dest)
        weight = params[0]
        bias = params[1]
        if prefetch["signature"] is not None:
            s._v_weight = weight
            s._v_bias = bias
        s._v_signature = prefetch["signature"]

    def post_cast(s, param_key, x, dtype, resident, update_weight):
        lowvram_fn = getattr(s, param_key + "_lowvram_function", None)
        fns = getattr(s, param_key + "_function", [])

        if x is None:
            return None

        orig = x

        def to_dequant(tensor, dtype):
            tensor = tensor.to(dtype=dtype)
            if isinstance(tensor, QuantizedTensor):
                tensor = tensor.dequantize()
            return tensor

        if (return_weights and orig.dtype != dtype) or len(fns) > 0:
            x = to_dequant(x, dtype)
        if not resident and lowvram_fn is not None:
            x = to_dequant(x, dtype if compute_dtype is None else compute_dtype)
            x = lowvram_fn(x)
            if (want_requant and len(fns) == 0 or update_weight):
                seed = comfy.utils.string_to_seed(s.seed_key)
                if isinstance(orig, QuantizedTensor):
                    y = orig.requantize_from_float(x, scale="recalculate", stochastic_rounding=seed)
                else:
                    y = comfy.float.stochastic_rounding(x, orig.dtype, seed=seed)
            if want_requant and len(fns) == 0:
                x = y
            if update_weight:
                orig.copy_(y)
        for f in fns:
            x = f(x)
        return x

    update_weight = prefetch["signature"] is not None
    weight = post_cast(s, "weight", weight, dtype, prefetch["resident"], update_weight)
    if bias is not None:
        bias = post_cast(s, "bias", bias, bias_dtype, prefetch["resident"], update_weight)

    if prefetch["signature"] is not None:
        prefetch["resident"] = True

    return (weight, bias) if return_weights else None


def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None, offloadable=False, compute_dtype=None, want_requant=False):
    # NOTE: offloadable=False is a legacy mode and if you are a custom node author reading this please pass
    # offloadable=True and call uncast_bias_weight() after your last usage of the weight/bias. This
    # will add async-offload support to your cast and improve performance.
    if input is not None:
        if dtype is None:
            if isinstance(input, QuantizedTensor):
                dtype = input.params.orig_dtype
            else:
                dtype = input.dtype
        if bias_dtype is None:
            bias_dtype = dtype
        if device is None:
            device = input.device

    def format_return(result, offloadable):
        weight, bias, offload_stream = result
        return (weight, bias, offload_stream) if offloadable else (weight, bias)

    non_blocking = comfy.model_management.device_supports_non_blocking(device)

    if hasattr(s, "_v") and comfy.model_management.is_device_cpu(device):

        #vbar doesn't support CPU weights, but some custom nodes have weird paths
        #that might switch the layer to the CPU and expect it to work. We have to take
        #a clone conservatively as we are mmapped and some SFT files are packed misaligned
        #If you are a custom node author reading this, please move your layer to the GPU
        #or declare your ModelPatcher as CPU in the first place.
        materialize_meta_param(s, ["weight", "bias"])
        weight = s.weight.to(dtype=dtype, copy=True)
        if isinstance(weight, QuantizedTensor):
            weight = weight.dequantize()
        bias = s.bias.to(dtype=bias_dtype, copy=True) if s.bias is not None else None
        return format_return((weight, bias, (None, None, None)), offloadable)

    elif hasattr(s, "_v") and s.weight.device != device:
        prefetched = hasattr(s, "_prefetch")
        offload_stream = None
        offload_device = None
        if not prefetched:
            offload_stream = cast_modules_with_vbar([s], dtype, device, bias_dtype, non_blocking)
            comfy.model_management.sync_stream(device, offload_stream)

        weight, bias = resolve_cast_module_with_vbar(s, dtype, device, bias_dtype, compute_dtype, want_requant)

        if not prefetched:
            if getattr(s, "_prefetch")["signature"] is not None:
                offload_device = device
            for param_key in ("weight", "bias"):
                lowvram_fn = getattr(s, param_key + "_lowvram_function", None)
                if lowvram_fn is not None:
                    lowvram_fn.clear_prepared()
            delattr(s, "_prefetch")
        return format_return((weight, bias, (offload_stream, offload_device, None)), offloadable)


    if offloadable and (device != s.weight.device or
                        (s.bias is not None and device != s.bias.device)):
        offload_stream = comfy.model_management.get_offload_stream(device)
    else:
        offload_stream = None

    bias = None
    weight = None

    if offload_stream is not None and not args.cuda_malloc:
        cast_buffer_size = comfy.memory_management.vram_aligned_size([ s.weight, s.bias ])
        cast_buffer = comfy.model_management.get_cast_buffer(offload_stream, device, cast_buffer_size, s)
        #The streams can be uneven in buffer capability and reject us. Retry to get the other stream
        if cast_buffer is None:
            offload_stream = comfy.model_management.get_offload_stream(device)
            cast_buffer = comfy.model_management.get_cast_buffer(offload_stream, device, cast_buffer_size, s)
        params = comfy.memory_management.interpret_gathered_like([ s.weight, s.bias ], cast_buffer)
        weight = params[0]
        bias = params[1]

    weight_has_function = len(s.weight_function) > 0
    bias_has_function = len(s.bias_function) > 0

    weight = comfy.model_management.cast_to(s.weight, None, device, non_blocking=non_blocking, copy=weight_has_function, stream=offload_stream, r=weight)

    if s.bias is not None:
        bias = comfy.model_management.cast_to(s.bias, None, device, non_blocking=non_blocking, copy=bias_has_function, stream=offload_stream, r=bias)

    comfy.model_management.sync_stream(device, offload_stream)

    bias_a = bias
    weight_a = weight

    if s.bias is not None:
        bias = bias.to(dtype=bias_dtype)
        for f in s.bias_function:
            bias = f(bias)

    if weight_has_function or weight.dtype != dtype:
        weight = weight.to(dtype=dtype)
        if isinstance(weight, QuantizedTensor):
            weight = weight.dequantize()
        for f in s.weight_function:
            weight = f(weight)

    return format_return((weight, bias, (offload_stream, weight_a, bias_a)), offloadable)


def uncast_bias_weight(s, weight, bias, offload_stream):
    if offload_stream is None:
        return
    os, weight_a, bias_a = offload_stream
    device=None
    #FIXME: This is really bad RTTI
    if weight_a is not None and not isinstance(weight_a, torch.Tensor):
        comfy_aimdo.model_vbar.vbar_unpin(s._v)
        device = weight_a
    if os is None:
        return
    if device is None:
        if weight_a is not None:
            device = weight_a.device
        else:
            if bias_a is None:
                return
            device = bias_a.device
    os.wait_stream(comfy.model_management.current_stream(device))

class CastBiasWeightContext:
    # When initialized with no arguments or the first is None, the context
    # will return the tuple (None, None).
    def __init__(self, *args, **kwargs):
        self.slf = args[0] if len(args) else None
        self.state = (None, None) if self.slf is None else cast_bias_weight(*args, **kwargs)

    def __enter__(self):
        result = self.state
        if len(result) < 3 or result[2] is None:
            # Not offloaded, immediately drop references.
            self.state = self.slf = None
        return result[:2]

    def __exit__(self, *_args) -> None:
        if self.slf is None:
            return
        slf, state = self.slf, self.state
        self.state = self.slf = None
        uncast_bias_weight(slf, *state)

class CastWeightBiasOp:
    comfy_cast_weights = False
    weight_function = []
    bias_function = []

class disable_weight_init:
    @staticmethod
    def _zero_init_parameter(module, name):
        param = getattr(module, name)
        device = None if getattr(param, "is_meta", False) else param.device
        setattr(module, name, torch.nn.Parameter(torch.zeros(param.shape, device=device, dtype=param.dtype), requires_grad=False))

    @staticmethod
    def _lazy_load_from_state_dict(module, state_dict, prefix, local_metadata,
                                   missing_keys, unexpected_keys, weight_shape,
                                   bias_shape=None):
        assign_to_params_buffers = local_metadata.get("assign_to_params_buffers", False)
        prefix_len = len(prefix)
        for k, v in state_dict.items():
            key = k[prefix_len:]
            if key == "weight":
                if not assign_to_params_buffers:
                    v = v.clone()
                module.weight = torch.nn.Parameter(v, requires_grad=False)
            elif bias_shape is not None and key == "bias" and v is not None:
                if not assign_to_params_buffers:
                    v = v.clone()
                module.bias = torch.nn.Parameter(v, requires_grad=False)
            else:
                unexpected_keys.append(k)

        if module.weight is None:
            module.weight = torch.nn.Parameter(torch.zeros(weight_shape), requires_grad=False)
            missing_keys.append(prefix + "weight")

        if bias_shape is not None and module.bias is None and getattr(module, "comfy_need_lazy_init_bias", False):
            module.bias = torch.nn.Parameter(torch.zeros(bias_shape), requires_grad=False)
            missing_keys.append(prefix + "bias")

    class Linear(torch.nn.Linear, CastWeightBiasOp):

        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            # don't trust subclasses that BYO state dict loader to call us.
            if (not comfy.memory_management.aimdo_enabled
                or type(self)._load_from_state_dict is not disable_weight_init.Linear._load_from_state_dict):
                super().__init__(in_features, out_features, bias, device, dtype)
                return

            # Issue is with `torch.empty` still reserving the full memory for the layer.
            # Windows doesn't over-commit memory so without this, We are momentarily commit
            # charged for the weight even though we might zero-copy it when we load the
            # state dict. If the commit charge exceeds the ceiling we can destabilize the
            # system.
            torch.nn.Module.__init__(self)
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.bias = None
            self.comfy_need_lazy_init_bias=bias
            self.weight_comfy_model_dtype = dtype
            self.bias_comfy_model_dtype = dtype

        def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                                strict, missing_keys, unexpected_keys, error_msgs):

            if (not comfy.memory_management.aimdo_enabled
                or type(self)._load_from_state_dict is not disable_weight_init.Linear._load_from_state_dict):
                return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                     missing_keys, unexpected_keys, error_msgs)
            disable_weight_init._lazy_load_from_state_dict(
                self,
                state_dict,
                prefix,
                local_metadata,
                missing_keys,
                unexpected_keys,
                weight_shape=(self.in_features, self.out_features),
                bias_shape=(self.out_features,),
            )


        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            with CastBiasWeightContext(self, input, offloadable=True) as (weight, bias):
                return torch.nn.functional.linear(input, weight, bias)

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv1d(torch.nn.Conv1d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            with CastBiasWeightContext(self, input, offloadable=True) as (weight, bias):
                return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv2d(torch.nn.Conv2d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            with CastBiasWeightContext(self, input, offloadable=True) as (weight, bias):
                return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv3d(torch.nn.Conv3d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def _conv_forward(self, input, weight, bias, autopad=None, *args, **kwargs):
            if autopad == "causal_zero":
                weight = weight[:, :, -input.shape[2]:, :, :]
            if NVIDIA_MEMORY_CONV_BUG_WORKAROUND and weight.dtype in (torch.float16, torch.bfloat16):
                out = torch.cudnn_convolution(input, weight, self.padding, self.stride, self.dilation, self.groups, benchmark=False, deterministic=False, allow_tf32=True)
                if bias is not None:
                    out += bias.reshape((1, -1) + (1,) * (out.ndim - 2))
                return out
            else:
                return super()._conv_forward(input, weight, bias, *args, **kwargs)

        def forward_comfy_cast_weights(self, input, autopad=None):
            with CastBiasWeightContext(self, input, offloadable=True) as (weight, bias):
                return self._conv_forward(input, weight, bias, autopad=autopad)

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0 or "autopad" in kwargs:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class GroupNorm(torch.nn.GroupNorm, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            with CastBiasWeightContext(self, input, offloadable=True) as (weight, bias):
                return torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class BatchNorm2d(torch.nn.BatchNorm2d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            with CastBiasWeightContext(self, input, offloadable=True) as (weight, bias):
                running_mean = self.running_mean.to(device=input.device, dtype=weight.dtype) if self.running_mean is not None else None
                running_var = self.running_var.to(device=input.device, dtype=weight.dtype) if self.running_var is not None else None
                return torch.nn.functional.batch_norm(input, running_mean, running_var, weight, bias, self.training, self.momentum, self.eps)

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class LayerNorm(torch.nn.LayerNorm, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            with CastBiasWeightContext(self if self.weight is not None else None, input, offloadable=True) as (weight, bias):
                return torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class RMSNorm(torch.nn.RMSNorm, CastWeightBiasOp):
        def reset_parameters(self):
            self.bias = None
            return None

        def forward_comfy_cast_weights(self, input):
            with CastBiasWeightContext(self if self.weight is not None else None, input, offloadable=True) as (weight, bias):
                return torch.nn.functional.rms_norm(input, self.normalized_shape, weight, self.eps)

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class ConvTranspose2d(torch.nn.ConvTranspose2d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input, output_size=None):
            num_spatial_dims = 2
            output_padding = self._output_padding(
                input, output_size, self.stride, self.padding, self.kernel_size,
                num_spatial_dims, self.dilation)

            with CastBiasWeightContext(self, input, offloadable=True) as (weight, bias):
                return torch.nn.functional.conv_transpose2d(
                       input, weight, bias, self.stride, self.padding,
                       output_padding, self.groups, self.dilation)

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class ConvTranspose1d(torch.nn.ConvTranspose1d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input, output_size=None):
            num_spatial_dims = 1
            output_padding = self._output_padding(
                input, output_size, self.stride, self.padding, self.kernel_size,
                num_spatial_dims, self.dilation)

            with CastBiasWeightContext(self, input, offloadable=True) as (weight, bias):
                return torch.nn.functional.conv_transpose1d(
                       input, weight, bias, self.stride, self.padding,
                       output_padding, self.groups, self.dilation)

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Embedding(torch.nn.Embedding, CastWeightBiasOp):
        def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None,
                     norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None,
                     _freeze=False, device=None, dtype=None):
            # don't trust subclasses that BYO state dict loader to call us.
            if (not comfy.memory_management.aimdo_enabled
                or type(self)._load_from_state_dict is not disable_weight_init.Embedding._load_from_state_dict):
                super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm,
                                 norm_type, scale_grad_by_freq, sparse, _weight,
                                 _freeze, device, dtype)
                return

            torch.nn.Module.__init__(self)
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.padding_idx = padding_idx
            self.max_norm = max_norm
            self.norm_type = norm_type
            self.scale_grad_by_freq = scale_grad_by_freq
            self.sparse = sparse
            # Keep shape/dtype visible for module introspection without reserving storage.
            embedding_dtype = dtype if dtype is not None else torch.get_default_dtype()
            self.weight = torch.nn.Parameter(
                torch.empty((num_embeddings, embedding_dim), device="meta", dtype=embedding_dtype),
                requires_grad=False,
            )
            self.bias = None
            self.weight_comfy_model_dtype = dtype

        def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                                strict, missing_keys, unexpected_keys, error_msgs):

            if (not comfy.memory_management.aimdo_enabled
                or type(self)._load_from_state_dict is not disable_weight_init.Embedding._load_from_state_dict):
                return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                     missing_keys, unexpected_keys, error_msgs)
            disable_weight_init._lazy_load_from_state_dict(
                self,
                state_dict,
                prefix,
                local_metadata,
                missing_keys,
                unexpected_keys,
                weight_shape=(self.num_embeddings, self.embedding_dim),
            )

        def reset_parameters(self):
            self.bias = None
            return None

        def forward_comfy_cast_weights(self, input, out_dtype=None):
            output_dtype = out_dtype
            if self.weight.dtype == torch.float16 or self.weight.dtype == torch.bfloat16:
                out_dtype = None
            with CastBiasWeightContext(self, device=input.device, dtype=out_dtype, offloadable=True) as (weight, bias):
                return torch.nn.functional.embedding(input, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse).to(dtype=output_dtype)


        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                if "out_dtype" in kwargs:
                    kwargs.pop("out_dtype")
                return super().forward(*args, **kwargs)

    @classmethod
    def conv_nd(s, dims, *args, **kwargs):
        if dims == 2:
            return s.Conv2d(*args, **kwargs)
        elif dims == 3:
            return s.Conv3d(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")


class manual_cast(disable_weight_init):
    class Linear(disable_weight_init.Linear):
        comfy_cast_weights = True

    class Conv1d(disable_weight_init.Conv1d):
        comfy_cast_weights = True

    class Conv2d(disable_weight_init.Conv2d):
        comfy_cast_weights = True

    class Conv3d(disable_weight_init.Conv3d):
        comfy_cast_weights = True

    class BatchNorm2d(disable_weight_init.BatchNorm2d):
        comfy_cast_weights = True

    class GroupNorm(disable_weight_init.GroupNorm):
        comfy_cast_weights = True

    class LayerNorm(disable_weight_init.LayerNorm):
        comfy_cast_weights = True

    class ConvTranspose2d(disable_weight_init.ConvTranspose2d):
        comfy_cast_weights = True

    class ConvTranspose1d(disable_weight_init.ConvTranspose1d):
        comfy_cast_weights = True

    class RMSNorm(disable_weight_init.RMSNorm):
        comfy_cast_weights = True

    class Embedding(disable_weight_init.Embedding):
        comfy_cast_weights = True


def fp8_linear(self, input):
    """
    Legacy FP8 linear function for backward compatibility.
    Uses QuantizedTensor subclass for dispatch.
    """
    dtype = self.weight.dtype
    if dtype not in [torch.float8_e4m3fn]:
        return None

    input_dtype = input.dtype
    input_shape = input.shape
    tensor_3d = input.ndim == 3

    if tensor_3d:
        input = input.reshape(-1, input_shape[2])

    if input.ndim != 2:
        return None
    lora_compute_dtype=comfy.model_management.lora_compute_dtype(input.device)
    scale_weight = torch.ones((), device=input.device, dtype=torch.float32)

    scale_input = torch.ones((), device=input.device, dtype=torch.float32)
    input = torch.clamp(input, min=-448, max=448, out=input)
    input_fp8 = input.to(dtype).contiguous()
    layout_params_input = TensorCoreFP8Layout.Params(scale=scale_input, orig_dtype=input_dtype, orig_shape=tuple(input_fp8.shape))
    quantized_input = QuantizedTensor(input_fp8, "TensorCoreFP8Layout", layout_params_input)

    with CastBiasWeightContext(self, input, dtype=dtype, bias_dtype=input_dtype, offloadable=True, compute_dtype=lora_compute_dtype, want_requant=True) as (w, bias):
        # Wrap weight in QuantizedTensor - this enables unified dispatch
        # Call F.linear - __torch_dispatch__ routes to fp8_linear handler in quant_ops.py!
        w_shape = tuple(w.shape)
        layout_params_weight = TensorCoreFP8Layout.Params(scale=scale_weight, orig_dtype=input_dtype, orig_shape=w_shape)
        quantized_weight = QuantizedTensor(w, "TensorCoreFP8Layout", layout_params_weight)
        o = torch.nn.functional.linear(quantized_input, quantized_weight, bias)

    if tensor_3d:
        o = o.reshape((input_shape[0], input_shape[1], w_shape[0]))

    return o

class fp8_ops(manual_cast):
    class Linear(manual_cast.Linear):
        def reset_parameters(self):
            self.scale_weight = None
            self.scale_input = None
            return None

        def forward_comfy_cast_weights(self, input):
            if len(self.weight_function) == 0 and len(self.bias_function) == 0:
                try:
                    out = fp8_linear(self, input)
                    if out is not None:
                        return out
                except Exception as e:
                    logging.info("Exception during fp8 op: {}".format(e))

            with CastBiasWeightContext(self, input, offloadable=True) as (weight, bias):
                return torch.nn.functional.linear(input, weight, bias)

CUBLAS_IS_AVAILABLE = False
try:
    from cublas_ops import CublasLinear, cublas_half_matmul
    CUBLAS_IS_AVAILABLE = True
except ImportError:
    pass

if CUBLAS_IS_AVAILABLE:
    class cublas_ops(manual_cast):
        class Linear(CublasLinear, manual_cast.Linear):
            def reset_parameters(self):
                return None

            def forward_comfy_cast_weights(self, input):
                with CastBiasWeightContext(self, input, offloadable=True) as (weight, bias):
                    return cublas_half_matmul(input, weight, bias, self._epilogue_str, self.has_bias)

            def forward(self, *args, **kwargs):
                run_every_op()
                if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                    return self.forward_comfy_cast_weights(*args, **kwargs)
                else:
                    return super().forward(*args, **kwargs)

# ==============================================================================
# Mixed Precision Operations
# ==============================================================================
from . import quant_ops
from .quant_ops import (
    QuantizedTensor,
    QUANT_ALGOS,
    TensorCoreFP8Layout,
    TensorWiseINT8Layout,
    get_layout_class,
)

def _swiglu_eager(x):
    gate, up = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate).mul_(up)


INPUT_ACT_EAGER = {
    "gelu_tanh": lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
    "swiglu": _swiglu_eager,
}


def linear_input_act(linear, x, input_act):
    """``linear(act(x))``, with ``act`` folded into an INT8 activation quantizer.

    An INT8 linear quantizes its input anyway, so an elementwise activation can
    ride along inside that kernel instead of writing a full-size intermediate to
    HBM and reading it straight back. Worth it for an MLP's down-projection,
    where the intermediate is several times the hidden size.

    """
    weight = linear.weight
    if (comfy.model_management.in_training
            or not isinstance(weight, QuantizedTensor)
            or weight._layout_cls != "TensorWiseINT8Layout"
            or getattr(weight._params, "transposed", False)):
        return linear(INPUT_ACT_EAGER[input_act](x))

    # want_requant keeps a vbar-streamed layer on the INT8 path when a LoRA is
    # patched in on the fly; without it the cast hands back a dequantized weight.
    weight, bias, offload_stream = cast_bias_weight(
        linear, x, offloadable=True, compute_dtype=x.dtype, want_requant=True)
    try:
        if not isinstance(weight, QuantizedTensor):
            # A LoRA weight_function, or activations whose dtype differs from the
            # weight's, make the cast hand back a dequantized tensor.
            return torch.nn.functional.linear(INPUT_ACT_EAGER[input_act](x), weight, bias)
        qdata, scale = TensorWiseINT8Layout.get_plain_tensors(weight)
        return torch.ops.comfy_kitchen.int8_linear(
            x, qdata, scale, bias, quant_ops.ck.DTYPE_TO_CODE[x.dtype],
            getattr(weight._params, "convrot", False),
            getattr(weight._params, "convrot_groupsize", 256),
            input_act,
        )
    finally:
        uncast_bias_weight(linear, weight, bias, offload_stream)


class QuantLinearFunc(torch.autograd.Function):
    """Custom autograd function for quantized linear: quantized forward, optionally FP8 backward.

    When training_fp8_bwd is enabled:
      - Forward: quantize input per layout (FP8/NVFP4), use quantized matmul
      - Backward: all matmuls use FP8 tensor cores via torch.mm dispatch
      - Cached input is FP8 (half the memory of bf16)

    When training_fp8_bwd is disabled:
      - Forward: quantize input per layout, use quantized matmul
      - Backward: dequantize weight to compute_dtype, use standard matmul
    """

    @staticmethod
    def forward(ctx, input_float, weight, bias, layout_type, input_scale, compute_dtype):
        input_shape = input_float.shape
        inp = input_float.detach().flatten(0, -2)  # zero-cost view to 2D

        # Quantize input for forward (same layout as weight)
        if layout_type is not None:
            q_input = QuantizedTensor.from_float(inp, layout_type, scale=input_scale)
        else:
            q_input = inp

        w = weight.detach() if weight.requires_grad else weight
        b = bias.detach() if bias is not None and bias.requires_grad else bias

        output = torch.nn.functional.linear(q_input, w, b)

        # Unflatten output to match original input shape
        if len(input_shape) > 2:
            output = output.unflatten(0, input_shape[:-1])

        # Save for backward
        ctx.input_shape = input_shape
        ctx.has_bias = bias is not None
        ctx.compute_dtype = compute_dtype
        ctx.weight_requires_grad = weight.requires_grad
        ctx.fp8_bwd = comfy.model_management.training_fp8_bwd

        if ctx.fp8_bwd:
            # Cache FP8 quantized input — half the memory of bf16
            if isinstance(q_input, QuantizedTensor) and layout_type.startswith('TensorCoreFP8'):
                ctx.q_input = q_input  # already FP8, reuse
            else:
                # NVFP4 or other layout — quantize input to FP8 for backward
                ctx.q_input = QuantizedTensor.from_float(inp, "TensorCoreFP8E4M3Layout")
            ctx.save_for_backward(weight)
        else:
            ctx.q_input = None
            ctx.save_for_backward(input_float, weight)

        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        compute_dtype = ctx.compute_dtype
        grad_2d = grad_output.flatten(0, -2).to(compute_dtype)

        # Value casting — only difference between fp8 and non-fp8 paths
        if ctx.fp8_bwd:
            weight, = ctx.saved_tensors
            # Wrap as FP8 QuantizedTensors → torch.mm dispatches to _scaled_mm
            grad_mm = QuantizedTensor.from_float(grad_2d, "TensorCoreFP8E5M2Layout")
            if isinstance(weight, QuantizedTensor) and weight._layout_cls.startswith("TensorCoreFP8"):
                weight_mm = weight
            elif isinstance(weight, QuantizedTensor):
                weight_mm = QuantizedTensor.from_float(weight.dequantize().to(compute_dtype), "TensorCoreFP8E4M3Layout")
            else:
                weight_mm = QuantizedTensor.from_float(weight.to(compute_dtype), "TensorCoreFP8E4M3Layout")
            input_mm = ctx.q_input
        else:
            input_float, weight = ctx.saved_tensors
            # Standard tensors → torch.mm does regular matmul
            grad_mm = grad_2d
            if isinstance(weight, QuantizedTensor):
                weight_mm = weight.dequantize().to(compute_dtype)
            else:
                weight_mm = weight.to(compute_dtype)
            input_mm = input_float.flatten(0, -2).to(compute_dtype) if ctx.weight_requires_grad else None

        # Computation — same for both paths, dispatch handles the rest
        grad_input = torch.mm(grad_mm, weight_mm)
        if len(ctx.input_shape) > 2:
            grad_input = grad_input.unflatten(0, ctx.input_shape[:-1])

        grad_weight = None
        if ctx.weight_requires_grad:
            grad_weight = torch.mm(grad_mm.t(), input_mm)

        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_2d.sum(dim=0)

        return grad_input, grad_weight, grad_bias, None, None, None

# Quantized-weight module helpers

def _quantized_apply(module, fn, recurse=True):
    """Re-wrap Parameters after fn so .to()/.cuda() propagate through QuantizedTensor weights."""
    if recurse:
        for child in module.children():
            child._apply(fn)
    for key, param in module._parameters.items():
        if param is None:
            continue
        p = fn(param)
        if (not torch.is_inference_mode_enabled()) and p.is_inference():
            p = p.clone()
        module.register_parameter(key, torch.nn.Parameter(p, requires_grad=False))
    for key, buf in module._buffers.items():
        if buf is not None:
            module._buffers[key] = fn(buf)
    return module


def _load_quantized_module(module, super_load, state_dict, prefix, local_metadata, strict,
                            missing_keys, unexpected_keys, error_msgs, load_extra_params=False):
    """Shared _load_from_state_dict body for quantized-weight modules.

    Pops weight (+ scales, +/- extras), populates module.weight as a Parameter
    or Parameter-wrapped QuantizedTensor, then calls super_load and strips
    consumed keys from missing_keys. Reads compute_dtype from factory_kwargs
    and disabled formats from module._disabled_formats.
    """
    device = module.factory_kwargs["device"]
    compute_dtype = module.factory_kwargs["dtype"]
    disabled_formats = module._disabled_formats
    layer_name = prefix.rstrip('.')

    weight = state_dict.pop(f"{prefix}weight", None)
    if weight is None:
        logging.warning(f"Missing weight for layer {layer_name}")
        module.weight = None
        return
    manually_loaded_keys = [f"{prefix}weight"]

    def pop_scale(name, dtype=None):
        key = f"{prefix}{name}"
        v = state_dict.pop(key, None)
        if v is not None:
            v = v.to(device=device)
            if dtype is not None:
                v = v.view(dtype=dtype)
            manually_loaded_keys.append(key)
        return v

    layer_conf = state_dict.pop(f"{prefix}comfy_quant", None)
    if layer_conf is not None:
        layer_conf = json.loads(layer_conf.numpy().tobytes())

    if layer_conf is None:
        module.weight = torch.nn.Parameter(weight.to(device=device, dtype=compute_dtype), requires_grad=False)
    else:
        module.quant_format = layer_conf.get("format", None)
        module._full_precision_mm_config = layer_conf.get("full_precision_matrix_mult", False)
        if not module._full_precision_mm:
            module._full_precision_mm = module._full_precision_mm_config
        if module.quant_format in disabled_formats:
            module._full_precision_mm = True
        if module.quant_format is None:
            raise ValueError(f"Unknown quantization format for layer {layer_name}")

        qconfig = QUANT_ALGOS[module.quant_format]
        module.layout_type = qconfig["comfy_tensor_layout"]
        layout_cls = get_layout_class(module.layout_type)

        # Per-format scales; fp8 dtype views handle both legacy uint8-on-disk and native fp8.
        if module.quant_format in ("float8_e4m3fn", "float8_e5m2"):
            scales = {"scale": pop_scale("weight_scale")}
        elif module.quant_format == "mxfp8":
            bs = pop_scale("weight_scale", torch.float8_e8m0fnu)
            if bs is None:
                raise ValueError(f"Missing MXFP8 block scales for layer {layer_name}")
            scales = {"scale": bs}
        elif module.quant_format == "nvfp4":
            ts = pop_scale("weight_scale_2")
            bs = pop_scale("weight_scale", torch.float8_e4m3fn)
            if ts is None or bs is None:
                raise ValueError(f"Missing NVFP4 scales for layer {layer_name}")
            scales = {"scale": ts, "block_scale": bs}
        elif module.quant_format == "int8_tensorwise":
            scale = pop_scale("weight_scale")
            if scale is None:
                raise ValueError(f"Missing INT8 weight scale for layer {layer_name}")
            scales = {"scale": scale}
            params_conf = layer_conf.get("params", {})
            if not isinstance(params_conf, dict):
                params_conf = {}
            if layer_conf.get("convrot", params_conf.get("convrot", False)):
                scales["convrot"] = True
                scales["convrot_groupsize"] = int(
                    layer_conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256))
                )
        elif module.quant_format == "convrot_w4a4":
            scale = pop_scale("weight_scale")
            if scale is None:
                raise ValueError(f"Missing ConvRot W4A4 weight scale for layer {layer_name}")
            params_conf = layer_conf.get("params", {})
            if not isinstance(params_conf, dict):
                params_conf = {}
            scales = {
                "scale": scale,
                "convrot_groupsize": int(
                    layer_conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256))
                ),
                "quant_group_size": 64,
                "linear_dtype": layer_conf.get("linear_dtype", params_conf.get("linear_dtype", "int4")),
            }
        elif module.quant_format == "asym_w4a8_int8":
            # int4 weight (packed int8 [N,K/2]) + fp8 per-group scale (weight_s_rel),
            # fp32 per-channel scale (weight_s_channel) + optional Lloyd-Max codebook.
            scale = pop_scale("weight_s_rel")
            if scale is None:
                raise ValueError(f"Missing W4A8 group scale (weight_s_rel) for layer {layer_name}")
            if scale.dtype == torch.uint8:
                scale = scale.view(torch.float8_e4m3fn)
            params_conf = layer_conf.get("params", {})
            if not isinstance(params_conf, dict):
                params_conf = {}
            scales = {
                "scale": scale,
                "s_channel": pop_scale("weight_s_channel"),
                "codebook": pop_scale("weight_codebook"),
                "group_size": int(layer_conf.get("group_size", params_conf.get("group_size", 16))),
                "convrot_groupsize": int(
                    layer_conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256))
                ),
            }
        else:
            raise ValueError(f"Unsupported quantization format: {module.quant_format}")

        params = layout_cls.Params(**scales, orig_dtype=compute_dtype, orig_shape=module._orig_shape)
        module.weight = torch.nn.Parameter(
            QuantizedTensor(weight.to(device=device, dtype=qconfig["storage_t"]), module.layout_type, params),
            requires_grad=False,
        )

        if load_extra_params:
            for param_name in qconfig["parameters"]:
                if param_name in {"weight_scale", "weight_scale_2"}:
                    continue
                param_key = f"{prefix}{param_name}"
                _v = state_dict.pop(param_key, None)
                if _v is None:
                    continue
                module.register_parameter(param_name, torch.nn.Parameter(_v.to(device=device), requires_grad=False))
                manually_loaded_keys.append(param_key)

    super_load(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
    for key in manually_loaded_keys:
        if key in missing_keys:
            missing_keys.remove(key)


def _quantized_weight_state_dict(module, sd, prefix, extra_quant_conf=None, extra_quant_params=()):
    """Shared state_dict body. extra_quant_conf merges into the comfy_quant JSON;
    extra_quant_params names attributes written as additional top-level keys."""
    if not hasattr(module, 'weight'):
        logging.warning(f"Warning: state dict on uninitialized op {prefix}")
        return sd
    bias = getattr(module, 'bias', None)
    if bias is not None:
        sd[f"{prefix}bias"] = bias
    if module.weight is None:
        return sd
    if isinstance(module.weight, QuantizedTensor):
        sd.update(module.weight.state_dict(f"{prefix}weight"))
        quant_conf = {"format": module.quant_format}
        if getattr(module, '_full_precision_mm_config', False):
            quant_conf["full_precision_matrix_mult"] = True
        params = getattr(module.weight, "_params", None)
        if module.quant_format == "int8_tensorwise" and getattr(params, "convrot", False):
            quant_conf["convrot"] = True
            quant_conf["convrot_groupsize"] = getattr(params, "convrot_groupsize", 256)
        elif module.quant_format == "convrot_w4a4":
            quant_conf["convrot_groupsize"] = getattr(params, "convrot_groupsize", 256)
            linear_dtype = getattr(params, "linear_dtype", "int4")
            if linear_dtype != "int4":
                quant_conf["linear_dtype"] = linear_dtype
        elif module.quant_format == "asym_w4a8_int8":
            quant_conf["group_size"] = getattr(params, "group_size", 16)
            quant_conf["convrot_groupsize"] = getattr(params, "convrot_groupsize", 256)
        if extra_quant_conf:
            quant_conf.update(extra_quant_conf)
        sd[f"{prefix}comfy_quant"] = torch.tensor(list(json.dumps(quant_conf).encode("utf-8")), dtype=torch.uint8)
        for name in extra_quant_params:
            value = getattr(module, name, None)
            if value is not None:
                sd[f"{prefix}{name}"] = value
    else:
        sd[f"{prefix}weight"] = module.weight
    return sd


def mixed_precision_ops(quant_config={}, compute_dtype=torch.bfloat16, full_precision_mm=False, disabled=[]):
    class MixedPrecisionOps(manual_cast):
        _quant_config = quant_config
        _compute_dtype = compute_dtype
        _full_precision_mm = full_precision_mm
        _disabled = disabled

        class Linear(torch.nn.Module, CastWeightBiasOp):
            _disabled_formats = disabled

            def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
                super().__init__()

                self.factory_kwargs = {"device": device, "dtype": MixedPrecisionOps._compute_dtype}

                self.in_features = in_features
                self.out_features = out_features
                self._orig_shape = (out_features, in_features)
                if bias:
                    self.bias = torch.nn.Parameter(torch.empty(out_features, **self.factory_kwargs))
                else:
                    self.register_parameter("bias", None)

                self.tensor_class = None
                self._full_precision_mm = MixedPrecisionOps._full_precision_mm
                self._full_precision_mm_config = False

            def reset_parameters(self):
                return None

            def _load_from_state_dict(self, *args):
                _load_quantized_module(self, super()._load_from_state_dict, *args, load_extra_params=True)

            def state_dict(self, *args, destination=None, prefix="", **kwargs):
                sd = destination if destination is not None else {}
                return _quantized_weight_state_dict(self, sd, prefix, extra_quant_params=("input_scale", "pre_quant_scale"))

            def _forward(self, input, weight, bias):
                return torch.nn.functional.linear(input, weight, bias)

            def forward_comfy_cast_weights(
                self,
                input,
                compute_dtype=None,
                want_requant=False,
                weight_only_quant=False,
            ):
                if not weight_only_quant:
                    with CastBiasWeightContext(
                        self,
                        input,
                        offloadable=True,
                        compute_dtype=compute_dtype,
                        want_requant=want_requant,
                    ) as (weight, bias):
                        return self._forward(input, weight, bias)

                with CastBiasWeightContext(
                    self,
                    input=None,
                    dtype=self.weight.dtype,
                    device=input.device,
                    bias_dtype=input.dtype,
                    offloadable=True,
                    compute_dtype=compute_dtype,
                    want_requant=True,
                ) as (weight, bias):
                    weight = weight.to(dtype=input.dtype)
                    return self._forward(input, weight, bias)

            def forward(self, input, *args, **kwargs):
                run_every_op()

                # ModelOpt AWQ-style smoothing
                pre_quant_scale = getattr(self, 'pre_quant_scale', None)
                if pre_quant_scale is not None:
                    input = input * comfy.model_management.cast_to_device(pre_quant_scale, input.device, input.dtype)

                input_shape = input.shape
                reshaped_nd = False
                #If cast needs to apply lora, it should be done in the compute dtype
                compute_dtype = input.dtype

                _use_quantized = (
                    getattr(self, 'layout_type', None) is not None and
                    not isinstance(input, QuantizedTensor) and not self._full_precision_mm and
                    not getattr(self, 'comfy_force_cast_weights', False) and
                    len(self.weight_function) == 0 and len(self.bias_function) == 0
                )
                quantize_input = QUANT_ALGOS.get(getattr(self, 'quant_format', None), {}).get("quantize_input", True)

                # Training path: quantized forward with compute_dtype backward via autograd function
                if (input.requires_grad and _use_quantized and quantize_input):
                    with CastBiasWeightContext(
                        self,
                        input,
                        offloadable=True,
                        compute_dtype=compute_dtype,
                        want_requant=True
                    ) as (weight, bias):
                        scale = getattr(self, 'input_scale', None)
                        if scale is not None:
                            scale = comfy.model_management.cast_to_device(scale, input.device, None)

                        return QuantLinearFunc.apply(
                            input, weight, bias, self.layout_type, scale, compute_dtype
                        )

                # Inference path (unchanged)
                if _use_quantized and quantize_input:

                    # Reshape >=3D tensors to 2D for quantization (needed for NVFP4 and others)
                    input_reshaped = input.reshape(-1, input_shape[-1]) if input.ndim >= 3 else input

                    # Fall back to non-quantized for non-2D tensors
                    if input_reshaped.ndim == 2:
                        reshaped_nd = input.ndim >= 3
                        # dtype is now implicit in the layout class
                        scale = getattr(self, 'input_scale', None)
                        if scale is not None:
                            scale = comfy.model_management.cast_to_device(scale, input.device, None)
                        input = QuantizedTensor.from_float(input_reshaped, self.layout_type, scale=scale)

                weight_only_quant = _use_quantized and not quantize_input and isinstance(self.weight, QuantizedTensor)
                output = self.forward_comfy_cast_weights(
                    input,
                    compute_dtype,
                    want_requant=isinstance(input, QuantizedTensor),
                    weight_only_quant=weight_only_quant,
                )

                # Reshape output back to original rank if input was >2D
                if reshaped_nd:
                    output = output.reshape((*input_shape[:-1], self.weight.shape[0]))

                return output

            def convert_weight(self, weight, inplace=False, **kwargs):
                if isinstance(weight, QuantizedTensor):
                    return weight.dequantize()
                else:
                    return weight

            def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
                if getattr(self, 'layout_type', None) is not None:
                    weight = self.weight.requantize_from_float(weight, scale="recalculate", stochastic_rounding=seed, inplace_ops=True).to(self.weight.dtype)
                else:
                    weight = weight.to(self.weight.dtype)
                if return_weight:
                    return weight

                assert inplace_update is False  # TODO: eventually remove the inplace_update stuff
                self.weight = torch.nn.Parameter(weight, requires_grad=False)

            def _apply(self, fn, recurse=True):  # This is to get torch.compile + moving weights to another device working
                return _quantized_apply(self, fn, recurse)

        class MoEExperts(torch.nn.Module, CastWeightBiasOp):
            """Container for E quantized expert weights, indexed via expert_weight(i).

            The bank lives on self.weight as a single 3D tensor — either a
            compute_dtype Parameter or a Parameter wrapping a QuantizedTensor
            with leading expert dim.

            State-dict layout matches mixed_precision_ops.Linear with a leading
            expert dim:
                {prefix}.weight          quant data (storage_t), leading dim = E
                {prefix}.weight_scale    block / per-tensor scale
                {prefix}.weight_scale_2  [E] or scalar           NVFP4 only
                {prefix}.bias            [E, out_features]       optional, compute_dtype
                {prefix}.comfy_quant     json -> {{"format": "...", "num_experts": E}}

            Without comfy_quant the weight loads as a plain compute_dtype 3D Parameter [E, out, in].
            """

            _disabled_formats = disabled

            def __init__(self, num_experts: int, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
                super().__init__()
                self.num_experts = num_experts
                self.in_features = in_features
                self.out_features = out_features
                self._orig_shape = (num_experts, out_features, in_features)
                self.factory_kwargs = {"device": device, "dtype": MixedPrecisionOps._compute_dtype}
                if bias:
                    self.bias = torch.nn.Parameter(torch.empty(num_experts, out_features, **self.factory_kwargs))
                else:
                    self.register_parameter("bias", None)

                # Populated by _load_from_state_dict:
                self.weight = None
                self.quant_format = None
                self.layout_type = None
                self._full_precision_mm = MixedPrecisionOps._full_precision_mm
                self._full_precision_mm_config = False
                self._resident_bank = None

            def reset_parameters(self):
                return None

            def _apply(self, fn, recurse=True):
                return _quantized_apply(self, fn, recurse)

            def _load_from_state_dict(self, *args):
                _load_quantized_module(self, super()._load_from_state_dict, *args, load_extra_params=False)

            def expert_weight(self, i: int):
                """Expert i's weight (Tensor or per-expert QuantizedTensor view)."""
                if isinstance(self.weight, QuantizedTensor):
                    return self._expert_qt_from(self.weight, i)
                return self.weight[i]

            @contextlib.contextmanager
            def bank_resident(self, input):
                """Cast the whole bank once; expert_linear inside reuses the cast.
                Not re-entrant — do not nest calls on the same instance.
                """
                with CastBiasWeightContext(self, input, offloadable=True) as self._resident_bank:
                    try:
                        yield self
                    finally:
                        self._resident_bank = None

            def expert_linear(self, input: torch.Tensor, i: int) -> torch.Tensor:
                """Linear against expert i's weight (with optional bias)."""
                resident = getattr(self, "_resident_bank", None)
                if resident is not None:
                    weight, bias = resident
                    return self._expert_linear_impl(input, weight, bias, i)
                with CastBiasWeightContext(self, input, offloadable=True) as (weight, bias):
                    return self._expert_linear_impl(input, weight, bias, i)

            def _expert_linear_impl(self, input, weight, bias, i):
                if isinstance(weight, QuantizedTensor):
                    qw = self._expert_qt_from(weight, i)
                else:
                    qw = weight[i]
                b = cast_to_input(bias[i], input, copy=False) if bias is not None else None

                if isinstance(qw, QuantizedTensor):
                    use_fast = (
                        not self._full_precision_mm
                        and qw.layout_cls.supports_fast_matmul()
                        and input.dim() == 2
                    )
                    if use_fast:
                        qin = QuantizedTensor.from_float(input, self.layout_type)
                        return torch.nn.functional.linear(qin, qw, b)
                    out = input @ qw.dequantize().t()
                    return out + b if b is not None else out
                return torch.nn.functional.linear(input, qw, b)

            def _expert_qt_from(self, weight: QuantizedTensor, i: int) -> QuantizedTensor:
                """Build a per-expert QuantizedTensor by indexing into a resident bank."""
                params = weight._params
                kwargs = {
                    "scale": params.scale[i] if params.scale.dim() else params.scale,
                    "orig_dtype": params.orig_dtype,
                    "orig_shape": (self.out_features, self.in_features),
                }
                if hasattr(params, "block_scale"): # NVFP4
                    kwargs["block_scale"] = params.block_scale[i]
                if hasattr(params, "quant_group_size"):
                    kwargs["quant_group_size"] = params.quant_group_size
                if hasattr(params, "convrot_groupsize"):
                    kwargs["convrot_groupsize"] = params.convrot_groupsize
                if hasattr(params, "linear_dtype"):
                    kwargs["linear_dtype"] = params.linear_dtype
                return QuantizedTensor(weight._qdata[i], weight._layout_cls, type(params)(**kwargs))

            def state_dict(self, *args, destination=None, prefix="", **kwargs):
                sd = destination if destination is not None else {}
                return _quantized_weight_state_dict(self, sd, prefix, extra_quant_conf={"num_experts": self.num_experts})

        class Embedding(manual_cast.Embedding):
            def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
                weight_key = f"{prefix}weight"
                layer_conf = state_dict.pop(f"{prefix}comfy_quant", None)
                if layer_conf is not None:
                    layer_conf = json.loads(layer_conf.numpy().tobytes())

                # Only fp8 and int8_tensorwise support per-row dequant via index select.
                # Block-scaled formats (NVFP4, MXFP8) can't do per-row lookup efficiently.
                quant_format = layer_conf.get("format") if layer_conf is not None else None
                manually_loaded_keys = []

                if quant_format in ("float8_e4m3fn", "float8_e5m2", "int8_tensorwise") and weight_key in state_dict:
                    self.quant_format = quant_format
                    qconfig = QUANT_ALGOS[quant_format]
                    self.layout_type = qconfig["comfy_tensor_layout"]
                    layout_cls = get_layout_class(self.layout_type)
                    weight = state_dict.pop(weight_key)
                    manually_loaded_keys.append(weight_key)

                    scale_key = f"{prefix}weight_scale"
                    scale = state_dict.pop(scale_key, None)
                    if scale is not None:
                        scale = scale.float()
                        manually_loaded_keys.append(scale_key)

                    extra = {}
                    if quant_format == "int8_tensorwise" and layer_conf.get("convrot", False):
                        # rotated embedding table: record it so the forward un-rotates after lookup
                        extra["convrot"] = True
                        extra["convrot_groupsize"] = int(layer_conf.get("convrot_groupsize", 256))
                    params = layout_cls.Params(
                        scale=scale if scale is not None else torch.ones((), dtype=torch.float32),
                        orig_dtype=MixedPrecisionOps._compute_dtype,
                        orig_shape=(self.num_embeddings, self.embedding_dim),
                        **extra,
                    )
                    self.weight = torch.nn.Parameter(
                        QuantizedTensor(weight.to(dtype=qconfig["storage_t"]), qconfig["comfy_tensor_layout"], params),
                        requires_grad=False)
                elif layer_conf is not None:
                    # Unsupported format — restore the marker so it round-trips; fall through to default load.
                    state_dict[f"{prefix}comfy_quant"] = torch.tensor(
                        list(json.dumps(layer_conf).encode('utf-8')), dtype=torch.uint8)

                super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
                for k in manually_loaded_keys:
                    if k in missing_keys:
                        missing_keys.remove(k)

            def state_dict(self, *args, destination=None, prefix="", **kwargs):
                sd = destination if destination is not None else {}
                return _quantized_weight_state_dict(self, sd, prefix)

            def forward_comfy_cast_weights(self, input, out_dtype=None):
                weight = self.weight

                # Optimized path: lookup in fp8/int8, dequantize only the selected rows.
                if isinstance(weight, QuantizedTensor) and len(self.weight_function) == 0:
                    with CastBiasWeightContext(self, device=input.device, dtype=weight.dtype, offloadable=True) as (qdata, _bias):
                        if isinstance(qdata, QuantizedTensor):
                            params = qdata._params
                            scale = params.scale
                            qdata = qdata._qdata
                        else:
                            params = weight._params
                            scale = None

                        # int8: per-row scale possible ConvRot, so let the layout do the gather
                        if self.quant_format == "int8_tensorwise":
                            x = get_layout_class(self.layout_type).dequantize_embedding(qdata, params, input)
                            return x if out_dtype is None else x.to(dtype=out_dtype)

                        x = torch.nn.functional.embedding(
                            input, qdata, self.padding_idx, self.max_norm,
                            self.norm_type, self.scale_grad_by_freq, self.sparse)
                    target_dtype = out_dtype if out_dtype is not None else weight._params.orig_dtype
                    x = x.to(dtype=target_dtype)
                    if scale is not None:
                        x = x * scale.to(dtype=target_dtype)
                    return x

                # Fallback for non-quantized or weight_function (LoRA) case
                return super().forward_comfy_cast_weights(input, out_dtype=out_dtype)

    return MixedPrecisionOps

def pick_operations(weight_dtype, compute_dtype, load_device=None, disable_fast_fp8=False, fp8_optimizations=False, model_config=None):
    fp8_compute = comfy.model_management.supports_fp8_compute(load_device) # TODO: if we support more ops this needs to be more granular
    nvfp4_compute = comfy.model_management.supports_nvfp4_compute(load_device)
    mxfp8_compute = comfy.model_management.supports_mxfp8_compute(load_device)

    if model_config and hasattr(model_config, 'quant_config') and model_config.quant_config:
        logging.info("Using mixed precision operations")
        disabled = set()
        if not nvfp4_compute:
            disabled.add("nvfp4")
        if not mxfp8_compute:
            disabled.add("mxfp8")
        if not fp8_compute:
            disabled.add("float8_e4m3fn")
            disabled.add("float8_e5m2")
        logging.info("Native ops: {} {}".format(", ".join(QUANT_ALGOS.keys() - disabled), ", emulated ops: {}".format(", ".join(disabled)) if len(disabled) > 0 else ""))
        return mixed_precision_ops(model_config.quant_config, compute_dtype, disabled=disabled)

    if (
        fp8_compute and
        (fp8_optimizations or PerformanceFeature.Fp8MatrixMultiplication in args.fast) and
        not disable_fast_fp8
    ):
        return fp8_ops

    if (
        PerformanceFeature.CublasOps in args.fast and
        CUBLAS_IS_AVAILABLE and
        weight_dtype == torch.float16 and
        (compute_dtype == torch.float16 or compute_dtype is None)
    ):
        logging.info("Using cublas ops")
        return cublas_ops

    if compute_dtype is None or weight_dtype == compute_dtype:
        return disable_weight_init

    return manual_cast
