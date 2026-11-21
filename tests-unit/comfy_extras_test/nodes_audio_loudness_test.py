import math
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

import comfy.audio

with patch.dict(sys.modules, {"server": MagicMock(), "comfy.model_management": MagicMock()}):
    from comfy_extras.nodes_audio import AudioNormalizeLoudness

SAMPLE_RATE = 48000


def sine(db_peak, seconds=5.0, sample_rate=SAMPLE_RATE, channels=2, freq=997.0):
    t = torch.arange(int(seconds * sample_rate), dtype=torch.float64) / sample_rate
    wave = 10 ** (db_peak / 20) * torch.sin(2 * math.pi * freq * t)
    return wave.repeat(channels, 1).float()


@pytest.mark.parametrize("sample_rate", [16000, 44100, 48000])
def test_reference_sine_is_minus_23_lufs(sample_rate):
    wave = sine(-23.0, sample_rate=sample_rate)
    assert comfy.audio.integrated_loudness(wave, sample_rate) == pytest.approx(-23.0, abs=0.1)


def test_mono_sine_loudness():
    assert comfy.audio.integrated_loudness(sine(-20.0, channels=1), SAMPLE_RATE) == pytest.approx(-23.0, abs=0.1)


def test_quiet_section_is_relative_gated():
    loud = sine(-23.0, seconds=4.0)
    quiet = sine(-50.0, seconds=4.0)
    assert comfy.audio.integrated_loudness(torch.cat((loud, quiet), dim=-1), SAMPLE_RATE) == pytest.approx(-23.0, abs=0.5)


def test_unmeasurable_audio_is_negative_infinity():
    assert comfy.audio.integrated_loudness(torch.zeros(2, SAMPLE_RATE), SAMPLE_RATE) == -math.inf
    assert comfy.audio.integrated_loudness(sine(-10.0, seconds=0.3), SAMPLE_RATE) == -math.inf
    assert comfy.audio.integrated_loudness(torch.zeros(2, 0), SAMPLE_RATE) == -math.inf


@pytest.mark.parametrize("target", [-23.0, -16.0, -30.0])
def test_node_reaches_target(target):
    audio = {"waveform": sine(-10.0).unsqueeze(0), "sample_rate": SAMPLE_RATE}
    out = AudioNormalizeLoudness.execute(audio, target).result[0]
    assert out["sample_rate"] == SAMPLE_RATE
    assert out["waveform"].shape == audio["waveform"].shape
    assert comfy.audio.integrated_loudness(out["waveform"][0], SAMPLE_RATE) == pytest.approx(target, abs=0.1)


def test_node_normalizes_each_batch_item():
    waveform = torch.stack((sine(-10.0), sine(-35.0)))
    out = AudioNormalizeLoudness.execute({"waveform": waveform, "sample_rate": SAMPLE_RATE}, -23.0).result[0]["waveform"]
    for item in out:
        assert comfy.audio.integrated_loudness(item, SAMPLE_RATE) == pytest.approx(-23.0, abs=0.1)


def test_node_passes_through_unmeasurable_items():
    waveform = torch.stack((torch.zeros(2, SAMPLE_RATE), sine(-10.0, seconds=1.0)))
    out = AudioNormalizeLoudness.execute({"waveform": waveform, "sample_rate": SAMPLE_RATE}, -23.0).result[0]["waveform"]
    assert torch.equal(out[0], waveform[0])
    assert comfy.audio.integrated_loudness(out[1], SAMPLE_RATE) == pytest.approx(-23.0, abs=0.1)


def test_node_leaves_input_untouched():
    waveform = sine(-10.0).unsqueeze(0)
    original = waveform.clone()
    AudioNormalizeLoudness.execute({"waveform": waveform, "sample_rate": SAMPLE_RATE}, -23.0)
    assert torch.equal(waveform, original)


def test_very_low_sample_rate_is_unmeasurable():
    assert comfy.audio.integrated_loudness(torch.ones(1, 4), 1) == -math.inf
