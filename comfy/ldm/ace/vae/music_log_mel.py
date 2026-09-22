# Original from: https://github.com/ace-step/ACE-Step/blob/main/music_dcae/music_log_mel.py
import torch
import torch.nn as nn
from torch import Tensor
from comfy.audio import MelScale

import comfy.model_management

class LinearSpectrogram(nn.Module):
    def __init__(
        self,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        center=False,
        mode="pow2_sqrt",
    ):
        super().__init__()

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.mode = mode

        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, y: Tensor) -> Tensor:
        if y.ndim == 3:
            y = y.squeeze(1)

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                (self.win_length - self.hop_length) // 2,
                (self.win_length - self.hop_length + 1) // 2,
            ),
            mode="reflect",
        ).squeeze(1)
        dtype = y.dtype
        spec = torch.stft(
            y.float(),
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=comfy.model_management.cast_to(self.window, dtype=torch.float32, device=y.device),
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.view_as_real(spec)

        if self.mode == "pow2_sqrt":
            spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
        spec = spec.to(dtype)
        return spec


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate=44100,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        n_mels=128,
        center=False,
        f_min=0.0,
        f_max=None,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate // 2

        self.spectrogram = LinearSpectrogram(n_fft, win_length, hop_length, center)
        self.mel_scale = MelScale(
            self.n_mels,
            self.sample_rate,
            self.f_min,
            self.f_max,
            self.n_fft // 2 + 1,
        )

    def compress(self, x: Tensor) -> Tensor:
        return torch.log(torch.clamp(x, min=1e-5))

    def decompress(self, x: Tensor) -> Tensor:
        return torch.exp(x)

    def forward(self, x: Tensor, return_linear: bool = False) -> Tensor:
        linear = self.spectrogram(x)
        x = self.mel_scale(linear)
        x = self.compress(x)
        # print(x.shape)
        if return_linear:
            return x, self.compress(linear)

        return x
