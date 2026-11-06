import pytest
import torch

from comfy.audio import resample


@pytest.mark.parametrize(
    "shape, source_rate, target_rate, dtype, method",
    [
        ((0,), 44100, 48000, torch.float32, "sinc_interp_hann"),
        ((1, 0), 48000, 44100, torch.float64, "sinc_interp_kaiser"),
        ((1, 2, 0), 44100, 48000, torch.float64, "sinc_interp_hann"),
        ((2, 1, 0), 48000, 44100, torch.float32, "sinc_interp_kaiser"),
    ],
)
def test_resample_empty_audio(shape, source_rate, target_rate, dtype, method):
    """Empty audio keeps its dimensions and gradient connection across rate changes."""
    waveform = torch.empty(shape, dtype=dtype, requires_grad=True)

    result = resample(waveform, source_rate, target_rate, resampling_method=method)

    assert result.shape == waveform.shape
    assert result.dtype == waveform.dtype
    assert result.device == waveform.device
    result.sum().backward()
    torch.testing.assert_close(waveform.grad, torch.empty_like(waveform))


def test_resample_nonempty_audio():
    """Ordinary downsampling preserves a constant signal away from the boundaries."""
    waveform = torch.ones(1, 2, 480, dtype=torch.float64, requires_grad=True)

    result = resample(waveform, 48000, 24000)

    assert result.shape == (1, 2, 240)
    torch.testing.assert_close(result[..., 6:-6], torch.ones_like(result[..., 6:-6]), rtol=1e-3, atol=1e-3)
    result.sum().backward()
    assert torch.isfinite(waveform.grad).all()
