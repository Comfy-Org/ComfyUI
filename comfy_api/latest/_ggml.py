"""Host GGML math for the explicitly supported first format set.

No custom-node implementation is imported here. Packed storage remains packed
until a layer requests its forward-time compute weight.
"""
import torch

from ._gguf_tensor import GGMLType, GGUFTensor, _LAYOUTS


def is_quantized(tensor):
    return tensor is not None and getattr(tensor, 'tensor_type', None) not in {
        None, GGMLType.F32, GGMLType.F16}


def require_supported(state):
    from ._ggml_a2 import TYPES

    allowed = {None, GGMLType.F32, GGMLType.F16, GGMLType.BF16,
               GGMLType.Q8_0, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K} | TYPES
    for name, tensor in state.items():
        qtype = getattr(tensor, 'tensor_type', None)
        if qtype not in allowed:
            raise NotImplementedError(f'unsupported GGML tensor type {qtype!r} for {name}')


def _plain(tensor):
    return tensor.as_subclass(torch.Tensor) if isinstance(tensor, GGUFTensor) else tensor


def _scale_min(packed):
    low_scale, low_min, high = packed.split(4, dim=1)
    scales = torch.cat((low_scale & 63, (high & 15) | ((low_scale >> 2) & 48)), dim=1)
    minima = torch.cat((low_min & 63, (high >> 4) | ((low_min >> 2) & 48)), dim=1)
    return scales, minima


def dequantize(tensor, dtype=None, dequant_dtype=None):
    qtype = getattr(tensor, 'tensor_type', None)
    if not is_quantized(tensor):
        return _plain(tensor).to(dtype=dtype)
    from ._ggml_a2 import TYPES, dequantize_blocks

    if qtype not in {GGMLType.BF16, GGMLType.Q8_0, GGMLType.Q4_K,
                     GGMLType.Q5_K, GGMLType.Q6_K} | TYPES:
        raise NotImplementedError(f'unsupported GGML forward tensor type {qtype!r}')
    shape = tensor.tensor_shape
    raw = _plain(tensor).view(torch.uint8)
    _, block_bytes, _ = _LAYOUTS[qtype]
    blocks = raw.reshape(-1, block_bytes)
    compute = dtype if dequant_dtype == 'target' else dequant_dtype
    if qtype in TYPES:
        weights = dequantize_blocks(qtype, blocks, compute)
    elif qtype == GGMLType.BF16:
        weights = (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)
    elif qtype == GGMLType.Q8_0:
        scales = blocks[:, :2].contiguous().view(torch.float16).to(dtype=compute)
        weights = blocks[:, 2:].view(torch.int8) * scales
    elif qtype in {GGMLType.Q4_K, GGMLType.Q5_K}:
        scale = blocks[:, :2].contiguous().view(torch.float16).to(dtype=compute)
        minimum = blocks[:, 2:4].contiguous().view(torch.float16).to(dtype=compute)
        local_scale, local_minimum = _scale_min(blocks[:, 4:16])
        # Keep the separate products in the selected precision: combining
        # these expressions or promoting them changes native fp16 rounding.
        scale = (scale * local_scale).unsqueeze(-1)
        minimum = (minimum * local_minimum).unsqueeze(-1)
        low_offset = 48 if qtype == GGMLType.Q5_K else 16
        packed = blocks[:, low_offset:].reshape(-1, 4, 32)
        values = torch.stack((packed & 15, packed >> 4), dim=2).reshape(-1, 8, 32)
        if qtype == GGMLType.Q5_K:
            high = blocks[:, 16:48].unsqueeze(1)
            shifts = torch.arange(8, dtype=torch.uint8, device=raw.device).view(1, 8, 1)
            values = values | (((high >> shifts) & 1) << 4)
        weights = scale * values - minimum
    else:  # Q6_K: sixteen signed scales for groups of sixteen values.
        low = blocks[:, :128].reshape(-1, 2, 64)
        low = torch.cat((low & 15, low >> 4), dim=2).reshape(-1, 8, 32)
        high = blocks[:, 128:192].reshape(-1, 2, 32)
        high = torch.cat(tuple((high >> shift) & 3 for shift in (0, 2, 4, 6)), dim=2)
        high = high.reshape(-1, 8, 32)
        values = (low | (high << 4)).to(torch.int8) - 32
        scale = blocks[:, 208:].contiguous().view(torch.float16).to(dtype=compute)
        local = blocks[:, 192:208].view(torch.int8).to(dtype=compute)
        scale = (scale * local).reshape(-1, 16, 1)
        weights = scale * values.reshape(-1, 16, 16)
    return weights.reshape(shape).to(dtype=dtype)


def patches_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(patches_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [patches_to_device(item, device) for item in value]
    return value


def make_ops(dequant_dtype=None, patch_dtype=None):
    import comfy.model_management
    import comfy.ops

    allowed = {None, 'target', torch.float16, torch.bfloat16, torch.float32}
    if dequant_dtype not in allowed or patch_dtype not in allowed:
        raise ValueError('GGML precision must be default, target, float16, bfloat16 or float32')
    dequant_policy, patch_policy = dequant_dtype, patch_dtype

    class Layer:
        comfy_cast_weights = True
        dequant_dtype = None
        patch_dtype = None
        largest_layer = False

        def _quantized(self):
            return is_quantized(self.weight) or is_quantized(self.bias)

        def _load_from_state_dict(self, state, prefix, local_metadata, strict,
                                  missing_keys, unexpected_keys, error_msgs):
            weight, bias = state.get(prefix + 'weight'), state.get(prefix + 'bias')
            if not (is_quantized(weight) or is_quantized(bias) or isinstance(self, torch.nn.Linear)):
                return super()._load_from_state_dict(state, prefix, local_metadata, strict,
                                                     missing_keys, unexpected_keys, error_msgs)
            for key, value in state.items():
                suffix = key[len(prefix):]
                if suffix == 'weight':
                    self.weight = torch.nn.Parameter(value, requires_grad=False)
                elif suffix == 'bias' and value is not None:
                    self.bias = torch.nn.Parameter(value, requires_grad=False)
                else:
                    missing_keys.append(key)
            if self.weight is None and isinstance(self, torch.nn.Linear):
                # The native loader fabricates a transposed zero weight here.
                # This unsupported missing-checkpoint path must not silently
                # produce a different initialized model in the trusted host.
                raise NotImplementedError('GGML Linear requires an actual checkpoint weight')
            self.largest_layer = bool(getattr(self.weight, 'is_largest_weight', False))

        def _save_to_state_dict(self, destination, prefix, keep_vars):
            if not self._quantized():
                return super()._save_to_state_dict(destination, prefix, keep_vars)
            destination[prefix + 'weight'] = torch.empty_like(self.weight, device='meta')
            if self.bias is not None:
                destination[prefix + 'bias'] = torch.empty_like(self.bias, device='meta')
            if self.largest_layer:
                dtype = torch.float16 if self.dequant_dtype in {None, 'target'} else self.dequant_dtype
                destination[prefix + 'temp.weight'] = torch.empty(
                    self.weight.tensor_shape, dtype=dtype, device='meta')

        def get_weight(self, tensor, dtype):
            if tensor is None:
                return None
            patches = []
            function = key = None
            for function, entries, key in getattr(tensor, 'patches', []):
                patches.extend(patches_to_device(entries, tensor.device))
            weight = dequantize(tensor, dtype, self.dequant_dtype)
            if patches:
                if self.patch_dtype is None:
                    weight = function(patches, weight, key)
                else:
                    patch_compute = dtype if self.patch_dtype == 'target' else self.patch_dtype
                    weight = function(patches, weight, key, patch_compute)
            return weight

        @torch.compiler.disable
        def cast_bias_weight(self, input=None, dtype=None, device=None, bias_dtype=None):
            dtype = dtype if dtype is not None else getattr(input, 'dtype', torch.float32)
            bias_dtype = dtype if bias_dtype is None else bias_dtype
            device = input.device if device is None else device
            non_blocking = comfy.model_management.device_supports_non_blocking(device)

            def weight_on(tensor, output_dtype):
                if tensor is None:
                    return None
                try:
                    weight = self.get_weight(tensor.to(device), dtype)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    weight = self.get_weight(tensor.to('cpu'), dtype)
                return comfy.ops.cast_to(weight, output_dtype, device,
                                         non_blocking=non_blocking, copy=False)

            bias = weight_on(self.bias, bias_dtype)
            return weight_on(self.weight, dtype), bias

        def forward_comfy_cast_weights(self, input, *args, **kwargs):
            if self._quantized():
                result = self.forward_ggml(input, *args, **kwargs)
            else:
                result = super().forward_comfy_cast_weights(input, *args, **kwargs)
            return _plain(result)

    class Ops(comfy.ops.manual_cast):
        class Linear(Layer, comfy.ops.manual_cast.Linear):
            dequant_dtype = dequant_policy
            patch_dtype = patch_policy

            def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
                torch.nn.Module.__init__(self)
                self.in_features, self.out_features = in_features, out_features
                self.weight = self.bias = None

            def forward_ggml(self, input):
                return torch.nn.functional.linear(input, *self.cast_bias_weight(input))

        class Conv2d(Layer, comfy.ops.manual_cast.Conv2d):
            def forward_ggml(self, input):
                return self._conv_forward(input, *self.cast_bias_weight(input))

        class Embedding(Layer, comfy.ops.manual_cast.Embedding):
            def forward_ggml(self, input, out_dtype=None):
                requested = out_dtype
                if self.weight.dtype in {torch.float16, torch.bfloat16}:
                    out_dtype = None
                weight, _ = self.cast_bias_weight(self, device=input.device, dtype=out_dtype)
                return torch.nn.functional.embedding(
                    input, weight, self.padding_idx, self.max_norm, self.norm_type,
                    self.scale_grad_by_freq, self.sparse).to(dtype=requested)

        class LayerNorm(Layer, comfy.ops.manual_cast.LayerNorm):
            def forward_ggml(self, input):
                if self.weight is None:
                    return super().forward_comfy_cast_weights(input)
                weight, bias = self.cast_bias_weight(input)
                return torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)

        class GroupNorm(Layer, comfy.ops.manual_cast.GroupNorm):
            def forward_ggml(self, input):
                weight, bias = self.cast_bias_weight(input)
                return torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)

        class RMSNorm(Layer, comfy.ops.manual_cast.RMSNorm):
            def forward_ggml(self, input):
                if self.weight is None:
                    return super().forward_comfy_cast_weights(input)
                weight, _ = self.cast_bias_weight(input)
                return torch.nn.functional.rms_norm(input, self.normalized_shape, weight, self.eps)

    return Ops
