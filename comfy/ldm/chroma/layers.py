import torch
from torch import Tensor, nn

from comfy.ldm.flux.layers import (
    MLPEmbedder,
    ModulationOut,
    modulated_norm,
)

# TODO: remove this in a few months
SingleStreamBlock = None
DoubleStreamBlock = None


class ChromaModulationOut(ModulationOut):
    @classmethod
    def from_offset(cls, tensor: torch.Tensor, offset: int = 0) -> ModulationOut:
        return cls(
            shift=tensor[:, offset : offset + 1, :],
            scale=tensor[:, offset + 1 : offset + 2, :],
            gate=tensor[:, offset + 2 : offset + 3, :],
        )




class Approximator(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_layers = 5, dtype=None, device=None, operations=None):
        super().__init__()
        self.in_proj = operations.Linear(in_dim, hidden_dim, bias=True, dtype=dtype, device=device)
        self.layers = nn.ModuleList([MLPEmbedder(hidden_dim, hidden_dim, dtype=dtype, device=device, operations=operations) for x in range( n_layers)])
        self.norms = nn.ModuleList([operations.RMSNorm(hidden_dim, dtype=dtype, device=device) for x in range( n_layers)])
        self.out_proj = operations.Linear(hidden_dim, out_dim, dtype=dtype, device=device)

    @property
    def device(self):
        # Get the device of the module (assumes all parameters are on the same device)
        return next(self.parameters()).device

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)

        for layer, norms in zip(self.layers, self.norms):
            x = x + layer(norms(x))

        x = self.out_proj(x)

        return x


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm_final = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.linear = operations.Linear(hidden_size, out_channels, bias=True, dtype=dtype, device=device)

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = vec
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        x = modulated_norm(x, self.norm_final, scale[:, None, :], shift[:, None, :])
        x = self.linear(x)
        return x
