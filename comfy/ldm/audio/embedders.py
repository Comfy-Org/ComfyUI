# code adapted from: https://github.com/Stability-AI/stable-audio-tools

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Union
from einops import rearrange
import math
import comfy.ops

class LearnedPositionalEmbedding(nn.Module):
    """Used for continuous time"""

    def __init__(self, dim: int):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.empty(half_dim))

    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

def TimePositionalEmbedding(dim: int, out_features: int) -> nn.Module:
    return nn.Sequential(
        LearnedPositionalEmbedding(dim),
        comfy.ops.manual_cast.Linear(in_features=dim + 1, out_features=out_features),
    )


class ExpoFourierFeatures(nn.Module):
    """Exponentially-spaced Fourier features (no learnable parameters)."""
    def __init__(self, dim, min_freq=0.5, max_freq=10000.0):
        super().__init__()
        self.dim = dim
        self.min_freq = min_freq
        self.max_freq = max_freq

    def forward(self, t):
        in_dtype = t.dtype
        t = t.float()
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        half_dim = self.dim // 2
        ramp = torch.linspace(0, 1, half_dim, device=t.device, dtype=torch.float32)
        freqs = torch.exp(ramp * (math.log(self.max_freq) - math.log(self.min_freq)) + math.log(self.min_freq))
        args = t * freqs * 2 * math.pi
        return torch.cat([args.cos(), args.sin()], dim=-1).to(in_dtype)


class NumberEmbedder(nn.Module):
    def __init__(
        self,
        features: int,
        dim: int = 256,
        fourier_features_type="learned",
    ):
        super().__init__()
        self.features = features
        if fourier_features_type == "expo":
            self.embedding = nn.Sequential(ExpoFourierFeatures(dim=dim), comfy.ops.manual_cast.Linear(in_features=dim, out_features=features))
        else:
            self.embedding = TimePositionalEmbedding(dim=dim, out_features=features)

    def forward(self, x: Union[List[float], Tensor]) -> Tensor:
        if not torch.is_tensor(x):
            device = next(self.embedding.parameters()).device
            x = torch.tensor(x, device=device)
        assert isinstance(x, Tensor)
        shape = x.shape
        x = rearrange(x, "... -> (...)")
        embedding = self.embedding(x)
        x = embedding.view(*shape, self.features)
        return x  # type: ignore


class Conditioner(nn.Module):
    def __init__(
            self,
            dim: int,
            output_dim: int,
            project_out: bool = False
            ):

        super().__init__()

        self.dim = dim
        self.output_dim = output_dim
        self.proj_out = nn.Linear(dim, output_dim) if (dim != output_dim or project_out) else nn.Identity()

    def forward(self, x):
        raise NotImplementedError()

class NumberConditioner(Conditioner):
    '''
        Conditioner that takes a list of floats, normalizes them for a given range, and returns a list of embeddings
    '''
    def __init__(self,
                output_dim: int,
                min_val: float=0,
                max_val: float=1,
                fourier_features_type: str = "learned",
                ):
        super().__init__(output_dim, output_dim)

        self.min_val = min_val
        self.max_val = max_val

        self.embedder = NumberEmbedder(features=output_dim, fourier_features_type=fourier_features_type)

    def forward(self, floats, device=None):
            # Cast the inputs to floats
            floats = [float(x) for x in floats]

            if device is None:
                device = next(self.embedder.parameters()).device

            floats = torch.tensor(floats).to(device)

            floats = floats.clamp(self.min_val, self.max_val)

            normalized_floats = (floats - self.min_val) / (self.max_val - self.min_val)

            # Cast floats to same type as embedder
            embedder_dtype = next(self.embedder.parameters()).dtype
            normalized_floats = normalized_floats.to(embedder_dtype)

            float_embeds = self.embedder(normalized_floats).unsqueeze(1)

            return [float_embeds, torch.ones(float_embeds.shape[0], 1).to(device)]
