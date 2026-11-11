import torch
import comfy.model_management

RMSNorm = torch.nn.RMSNorm

# Note: torch's fused F.rms_norm is faster but produces slightly different output than manual implementations (rsqrt/reduction rounding).
def rms_norm(x, weight=None, eps=1e-6):
    if weight is None:
        return torch.nn.functional.rms_norm(x, (x.shape[-1],), eps=eps)
    else:
        return torch.nn.functional.rms_norm(x, weight.shape, weight=comfy.model_management.cast_to(weight, dtype=x.dtype, device=x.device), eps=eps)
