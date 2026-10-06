"""Kandinsky 6 audio VAE: TOD-VAE decoder + BigVGAN, loaded by Comfy's stock VAE path."""
import torch.nn as nn

from .core_contract import AUDIO_DEFAULTS
from .mmaudio.ext.autoencoder.autoencoder import build_bigvgan_v2
from .mmaudio.ext.autoencoder.vae import VAE


class Kandinsky6AudioVAE(nn.Module):
    def __init__(self, bigvgan_config, scaling_factor=float(AUDIO_DEFAULTS["scaling_factor"]), mean_value=0.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.mean_value = mean_value
        self.vae = VAE(data_dim=128, embed_dim=40, hidden_dim=512)
        del self.vae.encoder
        self.vae.remove_weight_norm()
        self.vocoder = build_bigvgan_v2(bigvgan_config)

    def decode(self, samples, **kwargs):
        # samples: [B, T, 40] latent. Returns Comfy's [B, channels, samples] audio layout.
        z = samples / self.scaling_factor + self.mean_value
        mel = self.vae.decode(z.transpose(1, 2))
        return self.vocoder(mel)
