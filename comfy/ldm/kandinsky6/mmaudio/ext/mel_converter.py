# Reference: # https://github.com/bytedance/Make-An-Audio-2
from typing import Literal

import torch
import torch.nn as nn


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

        try:
            from librosa.filters import mel as librosa_mel_fn
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "MMAudio audio features require librosa. Install it with `pip install librosa`."
            ) from exc

        mel = librosa_mel_fn(
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            n_mels=self.num_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        mel_basis = torch.from_numpy(mel).float()
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
        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9)).float()
        spec = torch.matmul(self.mel_basis, spec)
        spec = spectral_normalize_torch(spec, self.norm_fn)

        return spec


def get_mel_converter(mode: Literal['44k']) -> MelConverter:
    """Create the 44.1 kHz mel-spectrogram converter used by MMAudio."""
    return MelConverter(sampling_rate=44_100,
                        n_fft=2048,
                        num_mels=128,
                        hop_size=512,
                        win_size=2048,
                        fmin=0,
                        fmax=44100 / 2,
                        norm_fn=torch.log)
