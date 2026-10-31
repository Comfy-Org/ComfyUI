"""Model-owned lazy GGML patches; no custom-node modules or guest callbacks."""
from collections import namedtuple

import torch
import comfy.float
import comfy.lora
import comfy.model_management
import comfy.model_patcher
import comfy.utils

from ._ggml import is_quantized, patches_to_device


class GGMLModelPatcher(comfy.model_patcher.ModelPatcher):
    patch_on_device = False
    mmap_released = False

    @classmethod
    def from_patcher(cls, patcher):
        if patcher.is_dynamic():
            raise NotImplementedError('GGML patching requires the ordinary non-dynamic ModelPatcher')
        result = patcher.clone()
        result.__class__ = cls
        result.patch_on_device = False
        result.mmap_released = False
        return result

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False,
                               return_weight=False, force_cast=False):
        if return_weight or force_cast:
            raise NotImplementedError('GGML return_weight/force_cast patching is unsupported')
        if key not in self.patches:
            return None
        weight = comfy.utils.get_attr(self.model, key)
        if is_quantized(weight):
            out = weight.to(device_to)
            device = self.load_device if self.patch_on_device else self.offload_device
            out.patches = [(comfy.lora.calculate_weight,
                            patches_to_device(self.patches[key], device), key)]
        else:
            inplace_update = self.weight_inplace_update or inplace_update
            if key not in self.backup:
                backup = namedtuple('GGMLBackup', ['weight', 'inplace_update'])
                self.backup[key] = backup(weight.to(device=self.offload_device,
                                                    copy=inplace_update), inplace_update)
            if device_to is None:
                working = weight.to(torch.float32, copy=True)
            else:
                working = comfy.model_management.cast_to_device(weight, device_to,
                                                                 torch.float32, copy=True)
            out = comfy.lora.calculate_weight(self.patches[key], working, key)
            out = comfy.float.stochastic_rounding(out, weight.dtype)
        if inplace_update:
            comfy.utils.copy_to_param(self.model, key, out)
        else:
            comfy.utils.set_attr_param(self.model, key, out)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for weight in self.model.parameters():
                if is_quantized(weight):
                    weight.patches = []
        return super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights)

    def load(self, *args, force_patch_weights=False, **kwargs):
        result = super().load(*args, force_patch_weights=True, **kwargs)
        if not self.mmap_released:
            if kwargs.get('lowvram_model_memory', 0) > 0:
                for module in self.model.modules():
                    weights = (getattr(module, 'weight', None), getattr(module, 'bias', None))
                    if any(getattr(weight, 'device', None) == self.offload_device for weight in weights):
                        module.to(self.load_device).to(self.offload_device)
            # This is the native one-time migration marker; on CPU->CPU it
            # does not certify that an underlying file mapping was released.
            self.mmap_released = True
        return result

    def clone(self, *args, **kwargs):
        result = super().clone(*args, **kwargs)
        result.patch_on_device = self.patch_on_device
        return result
