import math

import av
import numpy as np
import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy_api_nodes.util.conversions import audio_input_to_mp3  # noqa: E402

SAMPLE_RATE = 48000
DURATION = 2.0
LEFT_HZ = 440.0
RIGHT_HZ = 880.0


def tone(freq, duration=DURATION, sample_rate=SAMPLE_RATE):
    t = torch.arange(int(sample_rate * duration), dtype=torch.float32) / sample_rate
    return 0.5 * torch.sin(2 * math.pi * freq * t)


@pytest.fixture
def stereo_audio():
    """Comfy AUDIO with two tones that stay distinguishable through mp3."""
    waveform = torch.stack([tone(LEFT_HZ), tone(RIGHT_HZ)]).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": SAMPLE_RATE}


@pytest.fixture
def mono_audio():
    return {"waveform": tone(LEFT_HZ).unsqueeze(0).unsqueeze(0), "sample_rate": SAMPLE_RATE}


def decode(buffer):
    """(planes[C, N], sample_rate, channels) of an encoded mp3 buffer"""
    buffer.seek(0)
    with av.open(buffer, mode="r") as container:
        stream = container.streams.audio[0]
        planes = []
        for frame in container.decode(audio=0):
            array = frame.to_ndarray()
            if frame.format.is_planar:
                planes.append(array)
            else:
                planes.append(array.reshape(-1, len(frame.layout.channels)).T)
        return np.concatenate(planes, axis=1), stream.codec_context.sample_rate, len(stream.layout.channels)


def dominant_hz(signal, sample_rate):
    """Peak frequency, ignoring the encoder's padding at either edge"""
    edge = int(0.2 * sample_rate)
    window = signal[edge:-edge]
    spectrum = np.abs(np.fft.rfft(window * np.hanning(window.size)))
    return np.fft.rfftfreq(window.size, 1.0 / sample_rate)[int(np.argmax(spectrum))]


def test_stereo_duration_is_preserved(stereo_audio):
    planes, sample_rate, channels = decode(audio_input_to_mp3(stereo_audio))
    assert channels == 2
    assert sample_rate == SAMPLE_RATE
    assert planes.shape[1] / sample_rate == pytest.approx(DURATION, abs=0.15)


def test_stereo_channels_are_not_concatenated(stereo_audio):
    """The channels must be interleaved; concatenating them plays the clip twice."""
    planes, sample_rate, _ = decode(audio_input_to_mp3(stereo_audio))
    assert dominant_hz(planes[0], sample_rate) == pytest.approx(LEFT_HZ, abs=15)
    assert dominant_hz(planes[1], sample_rate) == pytest.approx(RIGHT_HZ, abs=15)


def test_mono_duration_is_preserved(mono_audio):
    planes, sample_rate, _ = decode(audio_input_to_mp3(mono_audio))
    assert planes.shape[1] / sample_rate == pytest.approx(DURATION, abs=0.15)
    assert dominant_hz(planes[0], sample_rate) == pytest.approx(LEFT_HZ, abs=15)
