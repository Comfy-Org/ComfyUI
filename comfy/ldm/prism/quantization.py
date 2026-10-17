"""Prism legacy row-scaled FP8 storage with native casting/offload.

Existing native FP8 layouts are tensorwise, not this file's [N,1] row scale.
Use native CastBiasWeightContext for placement and cleanup, then reconstruct
only the active Linear weight in FP32 before BF16/FP32 matrix multiplication.
No activation quantization, no model-level cache, no shared kernel patch.
"""
import torch
import comfy.ops
import comfy.model_management


class RowScaledFP8Ops(comfy.ops.manual_cast):
    class Linear(comfy.ops.manual_cast.Linear):
        def __init__(self, *args, **kwargs):
            """Register the row scale separately from the compressed FP8 parameter."""
            super().__init__(*args, **kwargs)
            self.register_buffer('prism_scale', None, persistent=False)

        def forward(self, input):
            """Reconstruct only the active linear weight through native cast/offload hooks."""
            if self.prism_scale is None:
                return super().forward(input)
            comfy.ops.run_every_op()
            with comfy.ops.CastBiasWeightContext(self, input,
                    dtype=torch.float32, bias_dtype=input.dtype,
                    offloadable=True) as (weight, bias):
                scale = comfy.model_management.cast_to_device(
                    self.prism_scale, input.device, torch.float32)
                weight = (weight * scale).to(input.dtype)
                return torch.nn.functional.linear(input, weight, bias)

        def _save_to_state_dict(self, destination, prefix, keep_vars):
            """Export the compressed weight and its row scale without requantization."""
            super()._save_to_state_dict(destination, prefix, keep_vars)
            if self.prism_scale is not None:
                destination[prefix + 'weight.prism_scale'] = (
                    self.prism_scale if keep_vars else self.prism_scale.detach())
