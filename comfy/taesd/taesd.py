#!/usr/bin/env python3
"""
Tiny AutoEncoder for Stable Diffusion
(DNN for encoding / decoding SD's latent space)
"""
import torch
import torch.nn as nn

import comfy.utils
import comfy.ops

def conv(n_in, n_out, **kwargs):
    return comfy.ops.disable_weight_init.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3

class Block(nn.Module):
    def __init__(self, n_in: int, n_out: int, use_midblock_gn: bool = False):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = comfy.ops.disable_weight_init.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
        if not use_midblock_gn:
            self.pool = None
            return
        n_gn = n_in * 4
        self.pool = nn.Sequential(
            comfy.ops.disable_weight_init.Conv2d(n_in, n_gn, 1, bias=False),
            comfy.ops.disable_weight_init.GroupNorm(4, n_gn),
            nn.ReLU(inplace=True),
            comfy.ops.disable_weight_init.Conv2d(n_gn, n_in, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool is not None:
            x = x + self.pool(x)
        return self.fuse(self.conv(x) + self.skip(x))

class Encoder(nn.Sequential):
    def __init__(self, latent_channels: int = 4, use_gn: bool = False):
        super().__init__(
            conv(3, 64), Block(64, 64),
            conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
            conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
            conv(64, 64, stride=2, bias=False), Block(64, 64, use_gn), Block(64, 64, use_gn), Block(64, 64, use_gn),
            conv(64, latent_channels),
        )

class Decoder(nn.Sequential):
    def __init__(self, latent_channels: int = 4, use_gn: bool = False):
        super().__init__(
            Clamp(), conv(latent_channels, 64), nn.ReLU(),
            Block(64, 64, use_gn), Block(64, 64, use_gn), Block(64, 64, use_gn), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
            Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
            Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
            Block(64, 64), conv(64, 3),
        )

class DecoderFlux2(Decoder):
    def __init__(self, latent_channels: int = 128, use_gn: bool = True):
        if latent_channels != 128 or not use_gn:
            raise ValueError("Unexpected parameters for Flux2 TAE module")
        super().__init__(latent_channels=32, use_gn=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = (
            x
            .reshape(B, 32, 2, 2, H, W)
            .permute(0, 1, 4, 2, 5, 3)
            .reshape(B, 32, H * 2, W * 2)
        )
        return super().forward(x)

class EncoderFlux2(Encoder):
    def __init__(self, latent_channels: int = 128, use_gn: bool = True):
        if latent_channels != 128 or not use_gn:
            raise ValueError("Unexpected parameters for Flux2 TAE module")
        super().__init__(latent_channels=32, use_gn=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = super().forward(x)
        B, C, H, W = result.shape
        return (
            result
            .reshape(B, C, H // 2, 2, W // 2, 2)
            .permute(0, 1, 3, 5, 2, 4)
            .reshape(B, 128, H // 2, W // 2)
        )


class TAESD(nn.Module):
    latent_magnitude = 3
    latent_shift = 0.5

    def __init__(self, encoder_path=None, decoder_path=None, latent_channels=4):
        """Initialize pretrained TAESD on the given device from the given checkpoints."""
        super().__init__()
        if latent_channels == 128:
            encoder_class = EncoderFlux2
            decoder_class = DecoderFlux2
        else:
            encoder_class = Encoder
            decoder_class = Decoder
        self.taesd_encoder = encoder_class(latent_channels=latent_channels)
        self.taesd_decoder = decoder_class(latent_channels=latent_channels)

        self.vae_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.vae_shift = torch.nn.Parameter(torch.tensor(0.0))
        if encoder_path is not None:
            self.taesd_encoder.load_state_dict(comfy.utils.load_torch_file(encoder_path, safe_load=True))
        if decoder_path is not None:
            self.taesd_decoder.load_state_dict(comfy.utils.load_torch_file(decoder_path, safe_load=True))

    @staticmethod
    def scale_latents(x: torch.Tensor) -> torch.Tensor:
        """raw latents -> [0, 1]"""
        return x.div(2 * TAESD.latent_magnitude).add(TAESD.latent_shift).clamp(0, 1)

    @staticmethod
    def unscale_latents(x: torch.Tensor) -> torch.Tensor:
        """[0, 1] -> raw latents"""
        return x.sub(TAESD.latent_shift).mul(2 * TAESD.latent_magnitude)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x_sample = self.taesd_decoder((x - self.vae_shift) * self.vae_scale)
        x_sample = x_sample.sub(0.5).mul(2)
        return x_sample

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (self.taesd_encoder(x * 0.5 + 0.5) / self.vae_scale) + self.vae_shift
