import math
import nodes
import node_helpers
import torch
import torchaudio
import comfy.model_management
import comfy.utils
import numpy as np
import logging
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

import scipy.signal
import scipy.ndimage
import scipy.fft
import scipy.sparse

# Audio Processing Functions - Derived from librosa (https://github.com/librosa/librosa)
# Copyright (c) 2013--2023, librosa development team.

def mel_to_hz(mels, htk=False):
    """Convert mel to Hz (slaney)"""
    mels = np.asanyarray(mels)
    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    if mels.ndim:
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))
    return freqs

def hz_to_mel(frequencies, htk=False):
    """Convert Hz to mel (slaney)"""
    frequencies = np.asanyarray(frequencies)
    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)
    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (frequencies - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    if frequencies.ndim:
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep
    return mels

def compute_cqt(y, sr=22050, hop_length=512, fmin=None, n_bins=84, bins_per_octave=12, tuning=0.0):
    """Compute Constant-Q Transform (CQT) spectrogram."""

    def _relative_bandwidth(freqs):
        bpo = np.empty_like(freqs)
        logf = np.log2(freqs)
        bpo[0] = 1.0 / (logf[1] - logf[0])
        bpo[-1] = 1.0 / (logf[-1] - logf[-2])
        bpo[1:-1] = 2.0 / (logf[2:] - logf[:-2])
        return (2.0 ** (2.0 / bpo) - 1.0) / (2.0 ** (2.0 / bpo) + 1.0)

    def _wavelet_lengths(freqs, sr, filter_scale, alpha):
        Q = float(filter_scale) / alpha
        return Q * sr / freqs  # shape (n_bins,) floats

    def _build_wavelet(freqs_oct, sr, filter_scale, alpha_oct):
        lengths = _wavelet_lengths(freqs_oct, sr, filter_scale, alpha_oct)
        filters = []
        for ilen, freq in zip(lengths, freqs_oct):
            t = np.arange(int(-ilen // 2), int(ilen // 2), dtype=float)
            sig = (np.cos(t * 2 * np.pi * freq / sr)
                   + 1j * np.sin(t * 2 * np.pi * freq / sr)).astype(np.complex64)
            sig *= scipy.signal.get_window('hann', len(sig), fftbins=True)
            l1 = np.sum(np.abs(sig))
            tiny = np.finfo(np.float32).tiny
            sig /= max(l1, tiny)
            filters.append(sig)
        max_len = max(lengths)
        n_fft = int(2.0 ** np.ceil(np.log2(max_len)))
        out = np.zeros((len(filters), n_fft), dtype=np.complex64)
        for k, f in enumerate(filters):
            lpad = int((n_fft - len(f)) // 2)
            out[k, lpad: lpad + len(f)] = f
        return out, lengths

    def _resample_half(y):
        ratio = 0.5
        n_samples = int(np.ceil(len(y) * ratio))
        # Kaiser-windowed FIR matches librosa/soxr more closely than scipy's default Hamming filter
        L = 2
        h = scipy.signal.firwin(160 * L + 1, 0.96 / L, window=('kaiser', 6.5))
        y_hat = scipy.signal.resample_poly(y.astype(np.float32), 1, 2, window=h)
        if len(y_hat) > n_samples:
            y_hat = y_hat[:n_samples]
        elif len(y_hat) < n_samples:
            y_hat = np.pad(y_hat, (0, n_samples - len(y_hat)))
        y_hat /= np.sqrt(ratio)
        return y_hat.astype(np.float32)

    def _sparsify_rows(x, quantile=0.01):
        mags = np.abs(x)
        norms = np.sum(mags, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        mag_sort = np.sort(mags, axis=1)
        cumulative_mag = np.cumsum(mag_sort / norms, axis=1)
        threshold_idx = np.argmin(cumulative_mag < quantile, axis=1)
        x_sparse = scipy.sparse.lil_matrix(x.shape, dtype=x.dtype)
        for i, j in enumerate(threshold_idx):
            idx = np.where(mags[i] >= mag_sort[i, j])
            x_sparse[i, idx] = x[i, idx]
        return x_sparse.tocsr()

    if fmin is None:
        fmin = 32.70319566257483  # C1 note frequency

    fmin = fmin * (2.0 ** (tuning / bins_per_octave))
    freqs = fmin * (2.0 ** (np.arange(n_bins) / bins_per_octave))

    alpha = _relative_bandwidth(freqs)
    lengths = _wavelet_lengths(freqs, float(sr), 1, alpha)

    n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
    n_filters = min(bins_per_octave, n_bins)

    cqt_resp = []
    my_y = y.astype(np.float32)
    my_sr = float(sr)
    my_hop = int(hop_length)

    for i in range(n_octaves):
        if i == 0:
            sl = slice(-n_filters, None)
        else:
            sl = slice(-n_filters * (i + 1), -n_filters * i)

        freqs_oct = freqs[sl]
        alpha_oct = alpha[sl]

        basis, basis_lengths = _build_wavelet(freqs_oct, my_sr, 1, alpha_oct)
        n_fft_oct = basis.shape[1]

        # Frequency-domain normalisation
        basis = basis.astype(np.complex64)
        basis *= basis_lengths[:, np.newaxis] / float(n_fft_oct)
        fft_basis = scipy.fft.fft(basis, n=n_fft_oct, axis=1)[:, :(n_fft_oct // 2) + 1]
        fft_basis = _sparsify_rows(fft_basis, quantile=0.01)
        fft_basis = fft_basis * np.sqrt(sr / my_sr)

        y_pad = np.pad(my_y, int(n_fft_oct // 2), mode='constant')
        n_frames = 1 + (len(y_pad) - n_fft_oct) // my_hop
        frames = np.lib.stride_tricks.as_strided(
            y_pad,
            shape=(n_fft_oct, n_frames),
            strides=(y_pad.strides[0], y_pad.strides[0] * my_hop),
        )
        stft_result = scipy.fft.rfft(frames, axis=0)
        cqt_resp.append(fft_basis.dot(stft_result))

        if my_hop % 2 == 0:
            my_hop //= 2
            my_sr /= 2.0
            my_y = _resample_half(my_y)

    max_col = min(c.shape[-1] for c in cqt_resp)
    cqt_out = np.empty((n_bins, max_col), dtype=np.complex64)
    end = n_bins
    for c_i in cqt_resp:
        n_oct = c_i.shape[0]
        if end < n_oct:
            cqt_out[:end, :] = c_i[-end:, :max_col]
        else:
            cqt_out[end - n_oct:end, :] = c_i[:, :max_col]
        end -= n_oct

    cqt_out /= np.sqrt(lengths)[:, np.newaxis]
    return np.abs(cqt_out).astype(np.float32)


def cq_to_chroma_mapping(n_input, bins_per_octave=12, n_chroma=12, fmin=None):
    """Map CQT bins to chroma bins."""

    if fmin is None:
        fmin = 32.70319566257483  # C1 note frequency

    n_merge = bins_per_octave / n_chroma
    cq_to_ch = np.repeat(np.eye(n_chroma), int(n_merge), axis=1)
    cq_to_ch = np.roll(cq_to_ch, -int(n_merge // 2), axis=1)
    n_octaves = int(np.ceil(n_input / bins_per_octave))
    cq_to_ch = np.tile(cq_to_ch, n_octaves)[:, :n_input]

    midi_0 = np.mod(12 * np.log2(fmin / 440.0) + 69, 12)
    roll = int(np.round(midi_0 * (n_chroma / 12.0)))
    cq_to_ch = np.roll(cq_to_ch, roll, axis=0)

    return cq_to_ch.astype(np.float32)


def _parabolic_interpolation(S, axis=-2):
    """Compute parabolic interpolation shift for peak refinement."""
    S_next = np.roll(S, -1, axis=axis)
    S_prev = np.roll(S, 1, axis=axis)

    a = S_next + S_prev - 2 * S
    b = (S_next - S_prev) / 2.0

    shifts = np.zeros_like(S)
    valid = np.abs(b) < np.abs(a)
    shifts[valid] = -b[valid] / a[valid]

    if axis == -2 or axis == S.ndim - 2:
        shifts[0, :] = 0
        shifts[-1, :] = 0
    elif axis == 0:
        shifts[0, ...] = 0
        shifts[-1, ...] = 0

    return shifts


def _localmax(S, axis=-2):
    """Find local maxima along an axis."""

    S_prev = np.roll(S, 1, axis=axis)
    S_next = np.roll(S, -1, axis=axis)

    local_max = (S > S_prev) & (S >= S_next)

    if axis == -2 or axis == S.ndim - 2:
        local_max[-1, :] = S[-1, :] > S[-2, :]
        # First element is never a local max (strict inequality with previous)
        local_max[0, :] = False
    elif axis == 0:
        local_max[-1, ...] = S[-1, ...] > S[-2, ...]
        local_max[0, ...] = False

    return local_max


def piptrack(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
             fmin=150.0, fmax=4000.0, threshold=0.1):
    """Pitch tracking on thresholded parabolically-interpolated STFT."""

    # Compute STFT if not provided
    if S is None:
        if y is None:
            raise ValueError("Either y or S must be provided")

        fft_window = scipy.signal.get_window('hann', n_fft, fftbins=True)
        if len(fft_window) < n_fft:
            lpad = int((n_fft - len(fft_window)) // 2)
            fft_window = np.pad(fft_window, (lpad, int(n_fft - len(fft_window) - lpad)), mode='constant')
        fft_window = fft_window.reshape((-1, 1))

        y_pad = np.pad(y, int(n_fft // 2), mode='constant')
        n_frames = 1 + (len(y_pad) - n_fft) // hop_length
        frames = np.lib.stride_tricks.as_strided(
            y_pad,
            shape=(n_fft, n_frames),
            strides=(y_pad.strides[0], y_pad.strides[0] * hop_length)
        )

        S = scipy.fft.rfft((fft_window * frames).astype(np.float32), axis=0)

    S = np.abs(S)

    fmin = max(fmin, 0)
    fmax = min(fmax, float(sr) / 2)

    fft_freqs = np.fft.rfftfreq(S.shape[0] * 2 - 2, 1.0 / sr)
    if len(fft_freqs) > S.shape[0]:
        fft_freqs = fft_freqs[:S.shape[0]]

    shift = _parabolic_interpolation(S, axis=0)
    avg = np.gradient(S, axis=0)
    dskew = 0.5 * avg * shift

    pitches = np.zeros_like(S)
    mags = np.zeros_like(S)

    freq_mask = (fmin <= fft_freqs) & (fft_freqs < fmax)
    freq_mask = freq_mask.reshape(-1, 1)

    ref_value = threshold * np.max(S, axis=0, keepdims=True)
    local_max = _localmax(S * (S > ref_value), axis=0)
    idx = np.nonzero(freq_mask & local_max)

    pitches[idx] = (idx[0] + shift[idx]) * float(sr) / (S.shape[0] * 2 - 2)
    mags[idx] = S[idx] + dskew[idx]

    return pitches, mags


def hz_to_octs(frequencies, tuning=0.0, bins_per_octave=12):
    """Convert frequencies (Hz) to octave numbers."""

    A440 = 440.0 * 2.0 ** (tuning / bins_per_octave)
    octs = np.log2(np.asanyarray(frequencies) / (float(A440) / 16))
    return octs


def pitch_tuning(frequencies, resolution=0.01, bins_per_octave=12):
    """Estimate tuning offset from a collection of pitches."""

    frequencies = np.atleast_1d(frequencies)
    frequencies = frequencies[frequencies > 0]

    if not np.any(frequencies):
        return 0.0

    residual = np.mod(bins_per_octave * hz_to_octs(frequencies, tuning=0.0,
                                                     bins_per_octave=bins_per_octave), 1.0)
    residual[residual >= 0.5] -= 1.0

    bins = np.linspace(-0.5, 0.5, int(np.ceil(1.0 / resolution)) + 1)
    counts, tuning = np.histogram(residual, bins)
    tuning_est = tuning[np.argmax(counts)]
    return tuning_est


def estimate_tuning(y, sr=22050, bins_per_octave=12):
    """Estimate global tuning deviation from 12-TET."""
    n_fft = 2048
    hop_length = 512

    if len(y) < n_fft:
        return 0.0

    pitch, mag = piptrack(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                          fmin=150.0, fmax=4000.0, threshold=0.1)

    pitch_mask = pitch > 0

    if not pitch_mask.any():
        return 0.0

    threshold = np.median(mag[pitch_mask])
    valid_pitches = pitch[(mag >= threshold) & pitch_mask]

    if len(valid_pitches) == 0:
        return 0.0

    tuning = pitch_tuning(valid_pitches, resolution=0.01, bins_per_octave=bins_per_octave)

    return float(tuning)


def compute_chroma_cens(y, sr=22050, hop_length=512, n_chroma=12,
                       n_octaves=7, bins_per_octave=36,
                       win_len_smooth=41, norm=2):
    """Compute Chroma Energy Normalized Statistics (CENS) features."""

    tuning = estimate_tuning(y, sr, bins_per_octave=bins_per_octave)

    fmin = 32.70319566257483  # C1 note frequency
    n_bins = n_octaves * bins_per_octave
    cqt_mag = compute_cqt(y, sr=sr, hop_length=hop_length,
                         fmin=fmin, n_bins=n_bins,
                         bins_per_octave=bins_per_octave,
                         tuning=tuning)

    chroma_map = cq_to_chroma_mapping(n_bins, bins_per_octave=bins_per_octave,
                                     n_chroma=n_chroma, fmin=fmin)
    chroma = np.dot(chroma_map, cqt_mag)

    threshold = np.finfo(chroma.dtype).tiny
    chroma_sum = np.sum(np.abs(chroma), axis=0, keepdims=True)
    chroma_sum = np.maximum(chroma_sum, threshold)
    chroma = chroma / chroma_sum

    quant_steps = [0.4, 0.2, 0.1, 0.05]
    quant_weights = [0.25, 0.25, 0.25, 0.25]
    chroma_quant = np.zeros_like(chroma)
    for step, weight in zip(quant_steps, quant_weights):
        chroma_quant += (chroma > step) * weight

    if win_len_smooth is not None and win_len_smooth > 0:
        win = scipy.signal.get_window('hann', win_len_smooth + 2, fftbins=False)
        win /= np.sum(win)
        win = win.reshape(1, -1)
        chroma_smooth = scipy.ndimage.convolve(chroma_quant, win, mode='constant')
    else:
        chroma_smooth = chroma_quant

    if norm == 2:
        threshold = np.finfo(chroma_smooth.dtype).tiny
        chroma_norm = np.sqrt(np.sum(chroma_smooth ** 2, axis=0, keepdims=True))
        chroma_norm = np.maximum(chroma_norm, threshold)
        chroma_smooth = chroma_smooth / chroma_norm
    elif norm == np.inf:
        threshold = np.finfo(chroma_smooth.dtype).tiny
        chroma_norm = np.max(np.abs(chroma_smooth), axis=0, keepdims=True)
        chroma_norm = np.maximum(chroma_norm, threshold)
        chroma_smooth = chroma_smooth / chroma_norm

    return chroma_smooth


def _create_mel_filterbank(sr, n_fft, n_mels=128, fmin=0.0, fmax=None):
    """Create mel-scale filterbank matrix."""
    if fmax is None:
        fmax = sr / 2.0
    mel_basis = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=np.float32)
    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)
    min_mel = hz_to_mel(fmin)
    max_mel = hz_to_mel(fmax)
    mels = np.linspace(min_mel, max_mel, n_mels + 2)
    mel_f = mel_to_hz(mels)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        mel_basis[i] = np.maximum(0, np.minimum(lower, upper))

    enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
    mel_basis *= enorm[:, np.newaxis]
    return mel_basis


def _compute_mel_spectrogram(data, sr, n_fft=2048, hop_length=512, n_mels=128):
    """Compute mel spectrogram from audio signal."""
    fft_window = scipy.signal.get_window('hann', n_fft, fftbins=True)
    if len(fft_window) < n_fft:
        lpad = int((n_fft - len(fft_window)) // 2)
        fft_window = np.pad(fft_window, (lpad, int(n_fft - len(fft_window) - lpad)), mode='constant')

    fft_window = fft_window.reshape((-1, 1))
    data_padded = np.pad(data, int(n_fft // 2), mode='constant')
    n_frames = 1 + (len(data_padded) - n_fft) // hop_length
    shape = (n_fft, n_frames)
    strides = (data_padded.strides[0], data_padded.strides[0] * hop_length)
    frames = np.lib.stride_tricks.as_strided(data_padded, shape=shape, strides=strides)

    stft_result = scipy.fft.rfft(fft_window * frames, axis=0).astype(np.complex64)
    power_spec = np.abs(stft_result) ** 2

    mel_basis = _create_mel_filterbank(sr, n_fft, n_mels=n_mels, fmin=0.0, fmax=sr / 2.0)
    mel_spec = np.dot(mel_basis, power_spec)
    return mel_spec.astype(np.float32)


def quick_tempo_estimate(audio_np, sr, start_bpm=120.0, std_bpm=1.0, hop_length=512):
    """Estimate tempo using autocorrelation tempogram."""

    if len(audio_np) < hop_length * 10:
        logging.warning("Audio too short for tempo estimation, returning default BPM of 120.0")
        return 120.0

    n_fft = 2048
    mel_S = _compute_mel_spectrogram(audio_np, sr, n_fft=n_fft, hop_length=hop_length, n_mels=128)
    log_mel_S = 10.0 * np.log10(np.maximum(1e-10, mel_S))

    lag = 1
    S_diff = log_mel_S[:, lag:] - log_mel_S[:, :-lag]
    S_onset = np.maximum(0.0, S_diff)
    onset_env_pre = np.mean(S_onset, axis=0)
    pad_width = lag + n_fft // (2 * hop_length)
    onset_env = np.pad(onset_env_pre, (pad_width, 0), mode='constant')
    onset_env = onset_env[:mel_S.shape[1]]

    return estimate_tempo_from_onset(onset_env, sr, hop_length, start_bpm, std_bpm, max_tempo=320.0)


def estimate_tempo_from_onset(onset_env, sr, hop_length, start_bpm=120.0, std_bpm=1.0, max_tempo=320.0):
    """Estimate tempo from onset strength envelope using autocorrelation tempogram."""
    if len(onset_env) < 20:
        return 120.0

    ac_size = 8.0
    win_length = int(np.round(ac_size * sr / hop_length))
    win_length = min(win_length, len(onset_env))

    pad_width = win_length // 2
    onset_padded = np.pad(onset_env, (pad_width, pad_width), mode='linear_ramp', end_values=(0, 0))

    n_frames = len(onset_env)
    shape = (win_length, n_frames)
    strides = (onset_padded.strides[0], onset_padded.strides[0])
    frames = np.lib.stride_tricks.as_strided(onset_padded, shape=shape, strides=strides)

    hann_window = scipy.signal.get_window('hann', win_length, fftbins=True)
    windowed_frames = frames * hann_window[:, np.newaxis]

    tempogram = np.zeros((win_length, n_frames))
    for i in range(n_frames):
        frame = windowed_frames[:, i]
        n_pad = scipy.fft.next_fast_len(2 * len(frame) - 1)
        fft_result = scipy.fft.rfft(frame, n=n_pad)
        powspec = np.abs(fft_result) ** 2
        ac = scipy.fft.irfft(powspec, n=n_pad)
        tempogram[:, i] = ac[:win_length]

    ac_max = np.max(np.abs(tempogram), axis=0)
    mask = ac_max > 0
    tempogram[:, mask] /= ac_max[mask]

    tempogram_mean = np.mean(tempogram, axis=1)
    tempogram_mean = np.maximum(tempogram_mean, 0)

    bpms = np.zeros(win_length, dtype=np.float64)
    bpms[0] = np.inf
    bpms[1:] = 60.0 * sr / (hop_length * np.arange(1.0, win_length))

    logprior = -0.5 * ((np.log2(bpms) - np.log2(start_bpm)) / std_bpm) ** 2

    if max_tempo is not None:
        max_idx = int(np.argmax(bpms < max_tempo))
        if max_idx > 0:
            logprior[:max_idx] = -np.inf

    weighted = np.log1p(1e6 * tempogram_mean) + logprior
    best_idx = int(np.argmax(weighted[1:])) + 1
    tempo = bpms[best_idx]

    return tempo


def detect_onset_peaks(onset_env, sr=22050, hop_length=512, pre_max=0.03, post_max=0.0,
                      pre_avg=0.10, post_avg=0.10, wait=0.03, delta=0.07):
    """Detect onset peaks using peak picking algorithm."""

    onset_normalized = onset_env - np.min(onset_env)
    onset_max = np.max(onset_normalized)
    if onset_max > 0:
        onset_normalized = onset_normalized / onset_max

    pre_max_frames = int(pre_max * sr / hop_length)
    post_max_frames = int(post_max * sr / hop_length) + 1
    pre_avg_frames = int(pre_avg * sr / hop_length)
    post_avg_frames = int(post_avg * sr / hop_length) + 1
    wait_frames = int(wait * sr / hop_length)

    peaks = np.zeros(len(onset_normalized), dtype=bool)
    peaks[0] = (onset_normalized[0] >= np.max(onset_normalized[:min(post_max_frames, len(onset_normalized))]))
    peaks[0] &= (onset_normalized[0] >= np.mean(onset_normalized[:min(post_avg_frames, len(onset_normalized))]) + delta)

    if peaks[0]:
        n = wait_frames + 1
    else:
        n = 1

    while n < len(onset_normalized):
        maxn = np.max(onset_normalized[max(0, n - pre_max_frames):min(n + post_max_frames, len(onset_normalized))])
        peaks[n] = (onset_normalized[n] == maxn)

        if not peaks[n]:
            n += 1
            continue

        avgn = np.mean(onset_normalized[max(0, n - pre_avg_frames):min(n + post_avg_frames, len(onset_normalized))])
        peaks[n] &= (onset_normalized[n] >= avgn + delta)

        if not peaks[n]:
            n += 1
            continue

        n += wait_frames + 1

    return np.flatnonzero(peaks).astype(np.int32)


def track_beats(onset_env, tempo, sr, hop_length, tightness=100, trim=True):
    """Track beats using dynamic programming."""

    frame_rate = sr / hop_length
    frames_per_beat = np.round(frame_rate * 60.0 / tempo)

    if frames_per_beat <= 0 or len(onset_env) < 2:
        return np.array([], dtype=np.int32)

    onset_std = np.std(onset_env, ddof=1)
    if onset_std > 0:
        onset_normalized = onset_env / onset_std
    else:
        onset_normalized = onset_env

    window_range = np.arange(-frames_per_beat, frames_per_beat + 1)
    window = np.exp(-0.5 * (window_range * 32.0 / frames_per_beat) ** 2)

    localscore = scipy.signal.convolve(onset_normalized, window, mode='same')

    backlink = np.full(len(localscore), -1, dtype=np.int32)
    cumscore = np.zeros(len(localscore), dtype=np.float64)

    score_thresh = 0.01 * localscore.max()
    first_beat = True

    backlink[0] = -1
    cumscore[0] = localscore[0]

    fpb = int(frames_per_beat)

    for i in range(1, len(localscore)):
        score_i = localscore[i]
        best_score = -np.inf
        beat_location = -1

        search_start = int(i - np.round(fpb / 2.0))
        search_end = int(i - 2 * fpb - 1)

        for loc in range(search_start, search_end, -1):
            if loc < 0:
                break

            score = cumscore[loc] - tightness * (np.log(i - loc) - np.log(fpb)) ** 2

            if score > best_score:
                best_score = score
                beat_location = loc

        if beat_location >= 0:
            cumscore[i] = score_i + best_score
        else:
            cumscore[i] = score_i

        if first_beat and score_i < score_thresh:
            backlink[i] = -1
        else:
            backlink[i] = beat_location
            first_beat = False

    local_max_mask = np.zeros(len(cumscore), dtype=bool)

    local_max_mask[0] = False

    for i in range(1, len(cumscore) - 1):
        local_max_mask[i] = (cumscore[i] > cumscore[i-1]) and (cumscore[i] >= cumscore[i+1])

    if len(cumscore) > 1:
        local_max_mask[-1] = cumscore[-1] > cumscore[-2]

    if np.any(local_max_mask):
        median_max = np.median(cumscore[local_max_mask])
        threshold = 0.5 * median_max

        tail = -1
        for i in range(len(cumscore) - 1, -1, -1):
            if local_max_mask[i] and cumscore[i] >= threshold:
                tail = i
                break
    else:
        tail = len(cumscore) - 1

    beats = np.zeros(len(localscore), dtype=bool)
    n = tail
    visited = set()
    while n >= 0 and n not in visited:
        beats[n] = True
        visited.add(n)
        n = backlink[n]

    if trim and np.any(beats):
        beat_positions = np.flatnonzero(beats)

        beat_localscores = localscore[beat_positions]

        w = np.hanning(5)
        smooth_boe_full = np.convolve(beat_localscores, w)
        smooth_boe = smooth_boe_full[len(w)//2 : len(localscore) + len(w)//2]

        threshold = 0.5 * np.sqrt(np.mean(smooth_boe ** 2))

        start_frame = 0
        while start_frame < len(localscore) and localscore[start_frame] <= threshold:
            beats[start_frame] = False
            start_frame += 1

        end_frame = len(localscore) - 1
        while end_frame >= 0 and localscore[end_frame] <= threshold:
            beats[end_frame] = False
            end_frame -= 1

    return np.flatnonzero(beats).astype(np.int32)

def compute_onset_envelope(mel_spec_db, n_fft=2048, hop_length=512):
    """Compute onset strength envelope from a log-mel spectrogram (dB)."""
    lag = 1
    onset_diff = mel_spec_db[:, lag:] - mel_spec_db[:, :-lag]
    onset_diff = np.maximum(0.0, onset_diff)
    envelope_pre_pad = np.mean(onset_diff, axis=0)

    pad_width = lag + n_fft // (2 * hop_length)
    envelope = np.pad(envelope_pre_pad, (pad_width, 0), mode='constant')
    envelope = envelope[:mel_spec_db.shape[1]]

    return envelope

def compute_mfcc(mel_spec_db, n_mfcc=20):
    """Compute MFCC features from a log-mel spectrogram (dB)."""
    mfcc = scipy.fft.dct(mel_spec_db, axis=0, type=2, norm='ortho')[:n_mfcc].T
    return mfcc.astype(np.float32)


def power_to_db(S, amin=1e-10, top_db=80.0, ref=1.0):
    """Convert a power spectrogram (amplitude squared) to decibel (dB) units"""
    S = np.asarray(S)
    log_spec = 10.0 * np.log10(np.maximum(amin, S))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref))
    if top_db is not None:
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)
    return log_spec


class WanDancerEncodeAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanDancerEncodeAudio",
            category="model/conditioning/wan/dancer",
            inputs=[
                io.Audio.Input("audio"),
                io.Int.Input("video_frames", default=149, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Float.Input("audio_inject_scale", default=1.0, min=0.0, max=10.0, step=0.01, tooltip="The scale for the audio features when injected into the video model."),
            ],
            outputs=[
                io.AudioEncoderOutput.Output(display_name="audio_encoder_output"),
                io.String.Output(display_name="fps_string", tooltip="The calculated fps based on the audio length and the number of video frames. Used in the prompt."),
            ],
        )

    @classmethod
    def execute(cls, video_frames, audio_inject_scale, audio) -> io.NodeOutput:
        waveform = audio["waveform"][0]
        sample_rate = audio["sample_rate"]
        base_fps = 30
        hop_length = 512
        model_sr = 22050
        n_fft = 2048

        # start tempo from original audio (not the resampled one) to match the reference pipeline
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=False)

        start_bpm = quick_tempo_estimate(waveform.squeeze().cpu().numpy(), sample_rate, hop_length=hop_length)

        # resample to the sample rate used for feature extraction
        resample_sr = base_fps * hop_length
        waveform = torchaudio.functional.resample(waveform, sample_rate, resample_sr)

        waveform_np = waveform.cpu().numpy().squeeze()
        mel_spec = _compute_mel_spectrogram(waveform_np, model_sr, n_fft, hop_length, n_mels=128)
        mel_spec_db = power_to_db(mel_spec, amin=1e-10, top_db=80.0, ref=1.0)
        envelope = compute_onset_envelope(mel_spec_db, n_fft, hop_length)
        mfcc = compute_mfcc(mel_spec_db, n_mfcc=20)
        chroma = compute_chroma_cens(y=waveform_np, sr=model_sr, hop_length=hop_length).T
        # detect peaks
        peak_idxs = detect_onset_peaks(envelope, sr=model_sr, hop_length=hop_length)
        peak_onehot = np.zeros_like(envelope, dtype=np.float32)
        peak_onehot[peak_idxs] = 1.0
        # detect beats
        beat_tracking_tempo = estimate_tempo_from_onset(envelope, sr=model_sr, hop_length=hop_length, start_bpm=start_bpm)
        beat_idxs = track_beats(envelope, beat_tracking_tempo, model_sr, hop_length, tightness=100, trim=True)
        beat_onehot = np.zeros_like(envelope, dtype=np.float32)
        beat_onehot[beat_idxs] = 1.0

        audio_feature = np.concatenate(
            [envelope[:, None], mfcc, chroma, peak_onehot[:, None], beat_onehot[:, None]],
            axis=-1,
        )
        audio_feature = torch.from_numpy(audio_feature).unsqueeze(0).to(comfy.model_management.intermediate_device())

        fps = float(base_fps / int(audio_feature.shape[1] / video_frames + 0.5))

        audio_encoder_output = {
            "audio_feature": audio_feature,
            "fps": fps,
            "audio_inject_scale": audio_inject_scale,
        }

        if int(fps + 0.5) != 30:
            fps_string = " 帧率是{:.4f}".format(fps) # "frame rate is" in Chinese, as it was in the original pipeline
        else:
            fps_string = ", 帧率是30fps。" # to match the reference pipeline when the fps is 30

        return io.NodeOutput(audio_encoder_output, fps_string)


class WanDancerVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanDancerVideo",
            category="model/conditioning/wan/dancer",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=832, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=149, min=1, max=nodes.MAX_RESOLUTION, step=4, tooltip="The number of frames in the generated video. Should stay 149 for WanDancer."),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True, tooltip="The CLIP vision embeds for the first frame."),
                io.ClipVisionOutput.Input("clip_vision_output_ref", optional=True, tooltip="The CLIP vision embeds for the reference image."),
                io.Image.Input("start_image", optional=True, tooltip="The initial image(s) to be encoded, can be any number of frames."),
                io.Mask.Input("mask", optional=True, tooltip="Image conditioning mask for the start image(s). White is kept, black is generated. Used for the local generations."),
                io.AudioEncoderOutput.Input("audio_encoder_output", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent", tooltip="Empty latent."),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, start_image=None, mask=None, clip_vision_output=None, clip_vision_output_ref=None, audio_encoder_output=None) -> io.NodeOutput:
        latent = torch.zeros([1, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            image = torch.zeros((length, height, width, start_image.shape[-1]), device=start_image.device, dtype=start_image.dtype)
            image[:start_image.shape[0]] = start_image

            concat_latent_image = vae.encode(image[:, :, :, :3])
            if mask is None:
                concat_mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=start_image.device, dtype=start_image.dtype)
                concat_mask[:, :, :((start_image.shape[0] - 1) // 4) + 1] = 0.0
            else:
                concat_mask = 1 - mask[:length].unsqueeze(0)
                concat_mask = comfy.utils.common_upscale(concat_mask, concat_latent_image.shape[-2], concat_latent_image.shape[-1], "nearest-exact", "disabled")
                concat_mask = torch.cat([torch.repeat_interleave(concat_mask[:, 0:1], repeats=4, dim=1), concat_mask[:, 1:]], dim=1)
                concat_mask = concat_mask.view(1, concat_mask.shape[1] // 4, 4, concat_latent_image.shape[-2], concat_latent_image.shape[-1]).transpose(1, 2)

            positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": concat_mask})
            negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": concat_mask})

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output, "clip_vision_output_ref": clip_vision_output_ref})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output, "clip_vision_output_ref": clip_vision_output_ref})

        if audio_encoder_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"audio_embed": audio_encoder_output["audio_feature"], "fps": audio_encoder_output["fps"], "audio_inject_scale": audio_encoder_output.get("audio_inject_scale", 1.0)})
            negative = node_helpers.conditioning_set_values(negative, {"audio_embed": audio_encoder_output["audio_feature"], "fps": audio_encoder_output["fps"], "audio_inject_scale": audio_encoder_output.get("audio_inject_scale", 1.0)})

        out_latent = {}
        out_latent["samples"] = latent
        return io.NodeOutput(positive, negative, out_latent)


class WanDancerPadKeyframes(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanDancerPadKeyframes",
            category="image/video",
            inputs=[
                io.Image.Input("images",),
                io.Int.Input("segment_length", default=149, min=1, max=10000, tooltip="Length of this segment (usually 149 frames)"),
                io.Int.Input("segment_index", default=0, min=0, max=100, tooltip="Which segment this is (0 for first, 1 for second, etc.)"),
                io.Audio.Input("audio", tooltip="Audio to calculate total output frames from and extract segment audio."),
            ],
            outputs=[
                io.Image.Output(display_name="keyframes_sequence", tooltip="Padded keyframe sequence"),
                io.Mask.Output(display_name="keyframes_mask", tooltip="Mask indicating valid frames"),
                io.Audio.Output(display_name="audio_segment", tooltip="Audio segment for this video segment"),
            ],
        )

    @classmethod
    def do_execute(cls, images, segment_length, segment_index, audio):
        B, H, W, C = images.shape
        fps = 30

        # calculate total frames
        audio_duration = audio["waveform"].shape[-1] / audio["sample_rate"]
        segment_duration = segment_length / fps
        buffer = 0.2
        num_segments = int((audio_duration - buffer) / segment_duration) + 1 if audio_duration > buffer else 0
        total_frames = num_segments * segment_length

        mask = torch.zeros((segment_length, H, W), device=images.device, dtype=images.dtype)
        keyframes = torch.zeros((segment_length, H, W, C), dtype=images.dtype, device=images.device)

        # guard: with no audio or no images, nothing to place — leave keyframes/mask zeroed
        if total_frames > 0 and B > 0:
            frame_interval = float(total_frames) / B
            seg_num = int(math.ceil(total_frames / segment_length))
            is_last_segment = (segment_index == seg_num - 1)

            positions = []
            images_before_this_segment = 0

            # count images consumed by previous segments
            for seg_idx in range(segment_index):
                end_idx = (total_frames - segment_length * seg_idx - 1) if seg_idx == seg_num - 1 else (segment_length - 1)
                cnt = 0
                while cnt * frame_interval < end_idx - frame_interval:
                    cnt += 1
                images_before_this_segment += cnt

            # positions for current segment
            end_index = (total_frames - segment_length * segment_index - 1) if is_last_segment else (segment_length - 1)
            cnt = 0
            while cnt * frame_interval < end_index - frame_interval:
                pos = int(math.ceil(frame_interval * cnt))
                positions.append((pos, images_before_this_segment + cnt))
                cnt += 1
            positions.append((end_index, images_before_this_segment + cnt))

            valid_positions = [(pos, idx) for pos, idx in positions if idx < B and pos < segment_length]

            if valid_positions:
                seg_positions, img_indices = zip(*valid_positions)
                seg_positions = torch.tensor(seg_positions, dtype=torch.long, device=images.device)
                img_indices = torch.tensor(img_indices, dtype=torch.long, device=images.device)
                mask[seg_positions] = 1
                keyframes[seg_positions] = images[img_indices]

        # extract audio segment
        segment_duration = segment_length / fps
        start_time = segment_index * segment_duration
        end_time = min(start_time + segment_duration, audio_duration)

        sample_rate = audio["sample_rate"]
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)

        audio_segment_waveform = audio["waveform"][:, :, start_sample:end_sample]
        audio_segment = {
            "waveform": audio_segment_waveform,
            "sample_rate": sample_rate
        }

        return keyframes, mask, audio_segment

    @classmethod
    def execute(cls, images, segment_length, segment_index, audio=None) -> io.NodeOutput:
        return io.NodeOutput(*cls.do_execute(images, segment_length, segment_index, audio))

class WanDancerPadKeyframesList(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanDancerPadKeyframesList",
            category="image/video",
            inputs=[
                io.Image.Input("images"),
                io.Int.Input("segment_length", default=149, min=1, max=10000, tooltip="Length of each segment (usually 149 frames)"),
                io.Int.Input("num_segments", default=1, min=1, max=100, tooltip="How many padded segments to emit as lists."),
                io.Audio.Input("audio", tooltip="Audio to slice for each emitted segment."),
            ],
            outputs=[
                io.Image.Output(display_name="keyframes_sequence", tooltip="Padded keyframe sequences", is_output_list=True),
                io.Mask.Output(display_name="keyframes_mask", tooltip="Masks indicating valid frames", is_output_list=True),
                io.Audio.Output(display_name="audio_segment", tooltip="Audio segment for each video segment", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, images, segment_length, num_segments, audio=None) -> io.NodeOutput:
        outputs = [WanDancerPadKeyframes.do_execute(images, segment_length, i, audio) for i in range(num_segments)]
        keyframes, masks, audio_segments = zip(*outputs)
        return io.NodeOutput(list(keyframes), list(masks), list(audio_segments))

class WanDancerExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            WanDancerVideo,
            WanDancerEncodeAudio,
            WanDancerPadKeyframes,
            WanDancerPadKeyframesList,
        ]

async def comfy_entrypoint() -> WanDancerExtension:
    return WanDancerExtension()
