# Audio DSP adapted from TorchAudio (BSD-2-Clause).
# BSD 2-Clause License
#
# Copyright (c) 2017 Facebook Inc. (Soumith Chintala),
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math

import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F


def resample(waveform, orig_freq, new_freq, lowpass_filter_width=6, rolloff=0.99, resampling_method="sinc_interp_hann", beta=None):
    """Bandlimited sinc resampling along the last dimension, on the input device."""
    if orig_freq <= 0 or new_freq <= 0:
        raise ValueError("Sample rates must be positive")
    if orig_freq == new_freq:
        return waveform
    if int(orig_freq) != orig_freq or int(new_freq) != new_freq:
        raise ValueError("Sample rates must be integers")
    if not waveform.is_floating_point():
        raise TypeError("Audio waveforms must be floating point")
    if lowpass_filter_width <= 0:
        raise ValueError("Lowpass filter width must be positive")

    divisor = math.gcd(int(orig_freq), int(new_freq))
    orig_freq = int(orig_freq) // divisor
    new_freq = int(new_freq) // divisor
    base_freq = min(orig_freq, new_freq) * rolloff
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    idx = torch.arange(-width, width + orig_freq, dtype=waveform.dtype, device=waveform.device)[None, None] / orig_freq
    t = torch.arange(0, -new_freq, -1, dtype=waveform.dtype, device=waveform.device)[:, None, None] / new_freq + idx
    t = (t * base_freq).clamp_(-lowpass_filter_width, lowpass_filter_width)
    if resampling_method == "sinc_interp_hann":
        window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    elif resampling_method == "sinc_interp_kaiser":
        beta = torch.tensor(14.769656459379492 if beta is None else beta, device=waveform.device)
        window = torch.i0(beta * torch.sqrt(1 - (t / lowpass_filter_width) ** 2)) / torch.i0(beta)
    else:
        raise ValueError(f"Unknown resampling method: {resampling_method}")
    t *= math.pi
    kernel = torch.where(t == 0, 1.0, t.sin() / t)
    kernel *= window * (base_freq / orig_freq)

    shape = waveform.shape
    length = shape[-1]
    waveform = waveform.reshape(-1, length)
    waveform = F.pad(waveform, (width, width + orig_freq))
    output = F.conv1d(waveform[:, None], kernel, stride=orig_freq)
    output = output.transpose(1, 2).reshape(waveform.shape[0], -1)
    # Match TorchAudio's float32 rounding to preserve existing output lengths.
    target_length = math.ceil(np.float32(new_freq * length / orig_freq))
    output = output[..., :target_length]
    return output.reshape(*shape[:-1], output.shape[-1])


def _hz_to_mel(freq):
    if freq >= 1000.0:
        return 15.0 + math.log(freq / 1000.0) / (math.log(6.4) / 27.0)
    return freq / (200.0 / 3)


class MelScale(torch.nn.Module):
    """Slaney mel filterbank with area normalization."""

    def __init__(self, n_mels, sample_rate, f_min, f_max, n_stft):
        super().__init__()
        if f_max is None:
            f_max = sample_rate // 2
        all_freqs = torch.linspace(0, sample_rate // 2, n_stft)
        mels = torch.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2)
        freqs = (200.0 / 3) * mels
        log_region = mels >= 15.0
        freqs[log_region] = 1000.0 * torch.exp((math.log(6.4) / 27.0) * (mels[log_region] - 15.0))
        diff = freqs[1:] - freqs[:-1]
        slopes = freqs.unsqueeze(0) - all_freqs.unsqueeze(1)
        fb = torch.minimum(-slopes[:, :-2] / diff[:-1], slopes[:, 2:] / diff[1:]).clamp_min(0)
        fb *= (2.0 / (freqs[2:] - freqs[:-2])).unsqueeze(0)
        self.register_buffer("fb", fb)

    def forward(self, spectrogram):
        return (spectrogram.transpose(-1, -2) @ self.fb).transpose(-1, -2)


class _Spectrogram(torch.nn.Module):
    def __init__(self, n_fft, win_length, hop_length, power):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.power = power
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, waveform):
        shape = waveform.shape
        spec = torch.stft(waveform.reshape(-1, shape[-1]), self.n_fft, self.hop_length, self.win_length, self.window, center=True, pad_mode="reflect", normalized=False, onesided=True, return_complex=True)
        spec = spec.reshape(*shape[:-1], *spec.shape[-2:]).abs()
        return spec if self.power == 1.0 else spec.pow(self.power)


class MelSpectrogram(torch.nn.Module):
    """Hann-windowed magnitude/power spectrogram with Slaney mel normalization."""

    def __init__(self, sample_rate, n_fft, hop_length, n_mels, f_min=0.0, f_max=None, win_length=None, power=2.0):
        super().__init__()
        self.spectrogram = _Spectrogram(n_fft, n_fft if win_length is None else win_length, hop_length, power)
        self.mel_scale = MelScale(n_mels, sample_rate, f_min, f_max, n_fft // 2 + 1)

    def forward(self, waveform):
        return self.mel_scale(self.spectrogram(waveform))


class _LFilter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, waveform, a, b):
        ctx.a = a
        ctx.b = b
        # SciPy supplies the compiled IIR loop; audio nodes normally run on CPU.
        output = scipy.signal.lfilter(b, a, waveform.detach().cpu().numpy())
        return torch.from_numpy(output).to(waveform)

    @staticmethod
    def backward(ctx, grad):
        return _LFilter.apply(grad.flip(-1), ctx.a, ctx.b).flip(-1), None, None


def _biquad(waveform, b0, b1, b2, a0, a1, a2):
    a = torch.stack((a0, a1, a2)).detach().cpu().numpy()
    b = torch.stack((b0, b1, b2)).detach().cpu().numpy()
    return _LFilter.apply(waveform, a, b).clamp(-1, 1)


def bass_biquad(waveform, sample_rate, gain, central_freq=100, Q=0.707):
    dtype = waveform.dtype
    device = waveform.device
    central_freq = torch.as_tensor(central_freq, dtype=dtype, device=device)
    Q = torch.as_tensor(Q, dtype=dtype, device=device)
    gain = torch.as_tensor(gain, dtype=dtype, device=device)
    w0 = 2 * math.pi * central_freq / sample_rate
    alpha = torch.sin(w0) / 2 / Q
    A = torch.exp(gain / 40 * math.log(10))
    temp1 = 2 * torch.sqrt(A) * alpha
    temp2 = (A - 1) * torch.cos(w0)
    temp3 = (A + 1) * torch.cos(w0)
    b0 = A * (A + 1 - temp2 + temp1)
    b1 = 2 * A * (A - 1 - temp3)
    b2 = A * (A + 1 - temp2 - temp1)
    a0 = A + 1 + temp2 + temp1
    a1 = -2 * (A - 1 + temp3)
    a2 = A + 1 + temp2 - temp1
    return _biquad(waveform, b0 / a0, b1 / a0, b2 / a0, a0 / a0, a1 / a0, a2 / a0)


def equalizer_biquad(waveform, sample_rate, center_freq, gain, Q=0.707):
    dtype = waveform.dtype
    device = waveform.device
    center_freq = torch.as_tensor(center_freq, dtype=dtype, device=device)
    Q = torch.as_tensor(Q, dtype=dtype, device=device)
    gain = torch.as_tensor(gain, dtype=dtype, device=device)
    w0 = 2 * math.pi * center_freq / sample_rate
    A = torch.exp(gain / 40.0 * math.log(10))
    alpha = torch.sin(w0) / 2 / Q
    b0 = 1 + alpha * A
    b1 = -2 * torch.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * torch.cos(w0)
    a2 = 1 - alpha / A
    return _biquad(waveform, b0, b1, b2, a0, a1, a2)


def treble_biquad(waveform, sample_rate, gain, central_freq=3000, Q=0.707):
    dtype = waveform.dtype
    device = waveform.device
    central_freq = torch.as_tensor(central_freq, dtype=dtype, device=device)
    Q = torch.as_tensor(Q, dtype=dtype, device=device)
    gain = torch.as_tensor(gain, dtype=dtype, device=device)
    w0 = 2 * math.pi * central_freq / sample_rate
    alpha = torch.sin(w0) / 2 / Q
    A = torch.exp(gain / 40 * math.log(10))
    temp1 = 2 * torch.sqrt(A) * alpha
    temp2 = (A - 1) * torch.cos(w0)
    temp3 = (A + 1) * torch.cos(w0)
    b0 = A * (A + 1 + temp2 + temp1)
    b1 = -2 * A * (A - 1 + temp3)
    b2 = A * (A + 1 + temp2 - temp1)
    a0 = A + 1 - temp2 + temp1
    a1 = 2 * (A - 1 - temp3)
    a2 = A + 1 - temp2 - temp1
    return _biquad(waveform, b0, b1, b2, a0, a1, a2)
