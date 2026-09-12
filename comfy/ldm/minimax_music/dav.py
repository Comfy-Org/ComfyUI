import math

import torch
from torch import nn

import comfy.ops


def snake(x, alpha):
    shape = x.shape
    flat = x.reshape(shape[0], shape[1], -1)
    alpha = comfy.ops.cast_to_input(alpha, flat)
    flat = flat + (alpha + 1e-9).reciprocal() * torch.sin(alpha * flat).pow(2)
    return flat.reshape(shape)


class Snake1d(nn.Module):
    def __init__(self, channels, dtype, device):
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(1, channels, 1, dtype=dtype, device=device))

    def forward(self, x):
        return snake(x, self.alpha)


def _weight_norm_conv(operations, *args, **kwargs):
    return nn.utils.parametrizations.weight_norm(operations.Conv1d(*args, **kwargs))


def _weight_norm_conv_transpose(operations, *args, **kwargs):
    return nn.utils.parametrizations.weight_norm(operations.ConvTranspose1d(*args, **kwargs))


class ResidualUnit(nn.Module):
    def __init__(self, dim, dilation, dtype, device, operations):
        super().__init__()
        padding = 3 * dilation
        self.block = nn.Sequential(
            Snake1d(dim, dtype, device),
            _weight_norm_conv(
                operations,
                dim,
                dim,
                kernel_size=7,
                dilation=dilation,
                padding=padding,
                dtype=dtype,
                device=device,
            ),
            Snake1d(dim, dtype, device),
            _weight_norm_conv(operations, dim, dim, kernel_size=1, dtype=dtype, device=device),
        )

    def forward(self, x):
        residual = self.block(x)
        if residual.shape[-1] != x.shape[-1]:
            padding = (x.shape[-1] - residual.shape[-1]) // 2
            x = x[..., padding:x.shape[-1] - padding]
        return x + residual


class DecoderBlock(nn.Module):
    def __init__(self, input_dim, output_dim, stride, dtype, device, operations):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim, dtype, device),
            _weight_norm_conv_transpose(
                operations,
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                dtype=dtype,
                device=device,
            ),
            ResidualUnit(output_dim, 1, dtype, device, operations),
            ResidualUnit(output_dim, 3, dtype, device, operations),
            ResidualUnit(output_dim, 9, dtype, device, operations),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(self, dtype, device, operations):
        super().__init__()
        layers = [
            _weight_norm_conv(
                operations,
                1024,
                1536,
                kernel_size=7,
                padding=3,
                dtype=dtype,
                device=device,
            )
        ]
        channels = 1536
        output_dim = channels
        for index, stride in enumerate((8, 8, 4, 2)):
            input_dim = channels // (2 ** index)
            output_dim = channels // (2 ** (index + 1))
            layers.append(DecoderBlock(input_dim, output_dim, stride, dtype, device, operations))
        layers.extend((
            Snake1d(output_dim, dtype, device),
            _weight_norm_conv(
                operations,
                output_dim,
                1,
                kernel_size=7,
                padding=3,
                dtype=dtype,
                device=device,
            ),
            nn.Tanh(),
        ))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MiniMaxMusic3DAV(nn.Module):
    def __init__(self, dtype=None, device=None, operations=None):
        super().__init__()
        self.dec_in_proj = operations.Conv1d(64, 1024, kernel_size=1, dtype=dtype, device=device)
        self.decoder = Decoder(dtype, device, operations)

    def decode(self, latent):
        batch, _, frames = latent.shape
        folded = latent.reshape(batch * 2, 64, frames)
        waveform = self.decoder(self.dec_in_proj(folded))
        return waveform.reshape(batch, 2, -1)

    forward = decode
