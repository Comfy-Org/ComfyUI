"""Kandinsky 6 audio VAE: TOD-VAE + BigVGAN, loaded by Comfy's stock VAE path."""
import torch
import torch.nn as nn

from comfy.ldm.mmaudio.vae44k.bigvgan import build_bigvgan
from comfy.ldm.mmaudio.vae44k.vae import VAE
from .core_contract import AUDIO_DEFAULTS


class MelConverter(nn.Module):
    """44.1 kHz mel-spectrogram converter of the Kandinsky 6 audio VAE.

    The mel basis is owned by the checkpoint (``mel_converter.mel_basis``);
    the load path only builds this converter for files that carry it.
    """

    def __init__(self):
        super().__init__()
        self.n_fft = 2048
        self.hop_size = 512
        self.win_size = 2048

        self.register_buffer('mel_basis', torch.empty((128, 1 + self.n_fft // 2)))
        self.register_buffer('hann_window', torch.hann_window(self.win_size))

    @property
    def device(self):
        return self.mel_basis.device

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform.clamp(min=-1., max=1.).to(self.device)

        pad = int((self.n_fft - self.hop_size) / 2)
        waveform = torch.nn.functional.pad(
            waveform.unsqueeze(1),
            [pad, pad],
            mode='reflect')
        waveform = waveform.squeeze(1)

        spec = torch.stft(waveform,
                          self.n_fft,
                          hop_length=self.hop_size,
                          win_length=self.win_size,
                          window=self.hann_window,
                          center=False,
                          pad_mode='reflect',
                          normalized=False,
                          onesided=True,
                          return_complex=True)

        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9)).float()
        spec = torch.matmul(self.mel_basis, spec)
        spec = torch.log(torch.clamp(spec, min=1e-5))

        return spec


class Kandinsky6AudioVAE(nn.Module):
    def __init__(self, bigvgan_config, scaling_factor=float(AUDIO_DEFAULTS["scaling_factor"]), mean_value=0.0, with_encoder=False):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.mean_value = mean_value
        self.vae = VAE(data_dim=128, embed_dim=40, hidden_dim=512, with_encoder=with_encoder)
        self.vocoder = build_bigvgan(bigvgan_config)
        if with_encoder:
            self.mel_converter = MelConverter()

    def decode(self, samples, **kwargs):
        # samples: [B, T, 40] latent. Returns Comfy's [B, channels, samples] audio layout.
        z = samples / self.scaling_factor + self.mean_value
        mel = self.vae.decode(z.transpose(1, 2))
        return self.vocoder(mel)

    def encode(self, samples, **kwargs):
        # samples: [B, channels, samples] waveform at 44.1 kHz. Returns the [B, T, 40] latent.
        mel = self.mel_converter(samples.mean(dim=1))
        z = self.vae.encode(mel)
        latent = (z - self.mean_value) * self.scaling_factor
        return latent.movedim(1, -1)
