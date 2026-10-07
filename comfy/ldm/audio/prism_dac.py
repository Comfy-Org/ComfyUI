# Continuous DAC architecture adapted from Tencent-Hunyuan/Prism (MIT)
# and descript-audio-codec (MIT). See prism_dac.LICENSE.
import math

import torch
from torch import nn
import comfy.ops
from comfy.ldm.minimax.audio_vae import Snake1d


class ResidualUnit(nn.Module):
    def __init__(self, channels, dilation, operations):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(channels),
            operations.Conv1d(channels, channels, 7, dilation=dilation, padding=3 * dilation),
            Snake1d(channels),
            operations.Conv1d(channels, channels, 1),
        )

    def forward(self, x):
        y = self.block(x)
        padding = (x.shape[-1] - y.shape[-1]) // 2
        if padding > 0:
            x = x[..., padding:-padding]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, channels, stride, operations):
        super().__init__()
        self.block = nn.Sequential(
            *(ResidualUnit(channels // 2, dilation, operations) for dilation in (1, 3, 9)),
            Snake1d(channels // 2),
            operations.Conv1d(channels // 2, channels, 2 * stride,
                              stride=stride, padding=math.ceil(stride / 2)),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, operations):
        super().__init__()
        channels = 128
        layers = [operations.Conv1d(1, channels, 7, padding=3)]
        for stride in (2, 3, 4, 5, 8):
            channels *= 2
            layers.append(EncoderBlock(channels, stride, operations))
        layers.extend((Snake1d(channels), operations.Conv1d(channels, 128, 3, padding=1)))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, channels, output_channels, stride, operations):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(channels),
            operations.ConvTranspose1d(channels, output_channels, 2 * stride,
                stride=stride, padding=math.ceil(stride / 2), output_padding=stride % 2),
            *(ResidualUnit(output_channels, dilation, operations) for dilation in (1, 3, 9)),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(self, operations):
        super().__init__()
        channels = 2048
        layers = [operations.Conv1d(128, channels, 7, padding=3)]
        for stride in (8, 5, 4, 3, 2):
            layers.append(DecoderBlock(channels, channels // 2, stride, operations))
            channels //= 2
        layers.extend((Snake1d(channels), operations.Conv1d(channels, 1, 7, padding=3), nn.Tanh()))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PrismDAC(nn.Module):
    """48 kHz mono continuous DAC; latents are [B, 128, T] with hop 960."""
    def __init__(self, operations):
        super().__init__()
        self.encoder = Encoder(operations)
        self.quant_conv = operations.Conv1d(128, 256, 1)
        self.post_quant_conv = operations.Conv1d(128, 128, 1)
        self.decoder = Decoder(operations)

    def encode(self, waveform):
        waveform = nn.functional.pad(waveform, (0, (-waveform.shape[-1]) % 960))
        # Official continuous posterior mode; not a sampled or discrete codebook.
        return self.quant_conv(self.encoder(waveform)).chunk(2, dim=1)[0]

    def decode(self, latent):
        return self.decoder(self.post_quant_conv(latent))
