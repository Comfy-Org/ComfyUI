from typing import Literal

import torch
import torch.nn as nn

from .distributions import DiagonalGaussianDistribution
from .vae import VAE_16k
from .bigvgan import BigVGANVocoder
import comfy.audio

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5, *, norm_fn):
    return norm_fn(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes, norm_fn):
    output = dynamic_range_compression_torch(magnitudes, norm_fn=norm_fn)
    return output

class MelConverter(nn.Module):

    def __init__(
        self,
        *,
        sampling_rate: float,
        n_fft: int,
        num_mels: int,
        hop_size: int,
        win_size: int,
        fmin: float,
        fmax: float,
        norm_fn,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.norm_fn = norm_fn

        # mel = librosa_mel_fn(sr=self.sampling_rate,
        #                      n_fft=self.n_fft,
        #                      n_mels=self.num_mels,
        #                      fmin=self.fmin,
        #                      fmax=self.fmax)
        # mel_basis = torch.from_numpy(mel).float()
        mel_basis = torch.empty((num_mels, 1 + n_fft // 2))
        hann_window = torch.hann_window(self.win_size)

        self.register_buffer('mel_basis', mel_basis)
        self.register_buffer('hann_window', hann_window)

    @property
    def device(self):
        return self.mel_basis.device

    def forward(self, waveform: torch.Tensor, center: bool = False) -> torch.Tensor:
        waveform = waveform.clamp(min=-1., max=1.).to(self.device)

        waveform = torch.nn.functional.pad(
            waveform.unsqueeze(1),
            [int((self.n_fft - self.hop_size) / 2),
             int((self.n_fft - self.hop_size) / 2)],
            mode='reflect')
        waveform = waveform.squeeze(1)

        spec = torch.stft(waveform,
                          self.n_fft,
                          hop_length=self.hop_size,
                          win_length=self.win_size,
                          window=self.hann_window,
                          center=center,
                          pad_mode='reflect',
                          normalized=False,
                          onesided=True,
                          return_complex=True)

        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
        spec = torch.matmul(self.mel_basis, spec)
        spec = spectral_normalize_torch(spec, self.norm_fn)

        return spec

class AudioAutoencoder(nn.Module):

    def __init__(
        self,
        *,
        # ckpt_path: str,
        mode=Literal['16k', '44k'],
        need_vae_encoder: bool = True,
    ):
        super().__init__()

        assert mode == "16k", "Only 16k mode is supported currently."
        self.mel_converter = MelConverter(sampling_rate=16_000,
                            n_fft=1024,
                            num_mels=80,
                            hop_size=256,
                            win_size=1024,
                            fmin=0,
                            fmax=8_000,
                            norm_fn=torch.log10)

        self.vae = VAE_16k().eval()

        bigvgan_config = {
            "resblock": "1",
            "num_mels": 80,
            "upsample_rates": [4, 4, 2, 2, 2, 2],
            "upsample_kernel_sizes": [8, 8, 4, 4, 4, 4],
            "upsample_initial_channel": 1536,
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [
                [1, 3, 5],
                [1, 3, 5],
                [1, 3, 5],
            ],
            "activation": "snakebeta",
            "snake_logscale": True,
        }

        self.vocoder = BigVGANVocoder(
            bigvgan_config
        ).eval()

    @torch.inference_mode()
    def encode_audio(self, x) -> DiagonalGaussianDistribution:
        # x: (B * L)
        mel = self.mel_converter(x)
        dist = self.vae.encode(mel)

        return dist

    @torch.no_grad()
    def decode(self, z):
        mel_decoded = self.vae.decode(z)
        audio = self.vocoder(mel_decoded)

        audio = comfy.audio.resample(audio, 16000, 44100)
        return audio

    @torch.no_grad()
    def encode(self, audio):
        audio = audio.mean(dim=1)
        audio = comfy.audio.resample(audio, 44100, 16000)
        dist = self.encode_audio(audio)
        return dist.mean
