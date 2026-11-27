import pytest
import torch

from comfy.audio import resample


@pytest.mark.parametrize("orig_freq,new_freq", [(44100, 48000), (48000, 44100)])
def test_resample_empty_waveform(orig_freq, new_freq):
    waveform = torch.empty(2, 3, 0, dtype=torch.float64)

    output = resample(waveform, orig_freq, new_freq)

    assert output.shape == waveform.shape
    assert output.dtype == waveform.dtype
    assert output.device == waveform.device
