# Audio feature / VAE facade.
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .ext.autoencoder import AutoEncoderModule
from .ext.autoencoder.distributions import DiagonalGaussianDistribution
from .ext.mel_converter import get_mel_converter


class FeaturesUtils(nn.Module):
    def __init__(
        self,
        *,
        tod_vae_ckpt: str | None = None,
        bigvgan_vocoder_ckpt: str | None = None,
        vocoder_config: dict | None = None,
        mode: Literal["44k"] = "44k",
        need_vae_encoder: bool = True,
        need_vae_decoder: bool = True,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.mel_converter = get_mel_converter(mode)
        self.tod = AutoEncoderModule(
            vae_ckpt_path=tod_vae_ckpt,
            vocoder_ckpt_path=bigvgan_vocoder_ckpt,
            vocoder_config=vocoder_config,
            mode=mode,
            need_vae_encoder=need_vae_encoder,
            need_vae_decoder=need_vae_decoder,
        )
        self.scaling_factor = scaling_factor
        self.downsample_factor = 1024

    def compile(self):
        """Compile the decode and vocode methods with ``torch.compile``."""
        self.decode = torch.compile(self.decode)
        self.vocode = torch.compile(self.vocode)

    def train(self, mode: bool = True) -> FeaturesUtils:
        """Keep the inference-only audio component in evaluation mode."""
        return super().train(False)

    @torch.inference_mode()
    def encode_audio(self, x) -> DiagonalGaussianDistribution:
        """Encode a waveform into an audio latent distribution."""
        assert self.tod is not None, "VAE is not loaded"
        mel = self.mel_converter(x)
        return self.tod.encode(mel)

    @torch.inference_mode()
    def vocode(self, mel: torch.Tensor) -> torch.Tensor:
        """Convert a decoded mel-spectrogram into a waveform."""
        assert self.tod is not None, "VAE is not loaded"
        return self.tod.vocode(mel)

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode audio latents into a mel-spectrogram."""
        assert self.tod is not None, "VAE is not loaded"
        return self.tod.decode(z)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @torch.no_grad()
    def wrapped_decode(self, z):
        """Decode latents and vocode them through the Diffusers forward hook."""
        mel_decoded = self.decode(z.to(dtype=self.dtype))
        return self.vocode(mel_decoded)

    @torch.no_grad()
    def wrapped_encode(self, audio):
        """Encode audio and return the mean latent through the forward hook."""
        dist = self.encode_audio(audio.to(dtype=self.dtype))
        return dist.mean
