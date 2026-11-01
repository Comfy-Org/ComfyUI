import wave

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import folder_paths
from comfy_extras.nodes_audio import LoadAudio, f32_pcm, load


def test_unsigned_pcm_samples():
    """Unsigned PCM maps its midpoint to silence without modifying the input."""
    waveform = torch.tensor([[0, 64, 128], [192, 254, 255]], dtype=torch.uint8)
    original = waveform.clone()

    result = f32_pcm(waveform)

    expected = torch.tensor([[-1.0, -0.5, 0.0], [0.5, 0.984375, 0.9921875]], dtype=torch.float32)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(waveform, original)
    assert result.device == waveform.device


@pytest.mark.parametrize(
    "dtype, values, expected",
    [
        (torch.int16, [-32768, 0, 32767], [-1.0, 0.0, 0.999969482421875]),
        (torch.int32, [-2147483648, 0, 2147483647], [-1.0, 0.0, 1.0]),
    ],
)
def test_signed_pcm_samples(dtype, values, expected):
    """Existing signed PCM scaling is preserved."""
    torch.testing.assert_close(f32_pcm(torch.tensor(values, dtype=dtype)), torch.tensor(expected, dtype=torch.float32))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_float_pcm_passthrough(dtype):
    """Floating waveforms retain their existing passthrough behavior."""
    waveform = torch.tensor([-0.5, 0.0, 0.5], dtype=dtype)
    assert f32_pcm(waveform) is waveform


def test_unsupported_pcm_dtype():
    """Other unsupported integer formats still raise an error."""
    with pytest.raises(ValueError, match="Unsupported wav dtype"):
        f32_pcm(torch.tensor([0], dtype=torch.int64))


@pytest.mark.parametrize("channels", [1, 2])
def test_load_unsigned_pcm_wav(tmp_path, monkeypatch, channels):
    """Real mono/stereo files load with normalized samples and unchanged metadata."""
    samples = [[0, 64, 128, 192, 255]]
    expected = [[-1.0, -0.5, 0.0, 0.5, 0.9921875]]
    if channels == 2:
        samples.append([255, 192, 128, 64, 0])
        expected.append([0.9921875, 0.5, 0.0, -0.5, -1.0])
    path = tmp_path / "pcm-u8.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(1)
        output.setframerate(8000)
        output.writeframes(bytes(value for frame in zip(*samples) for value in frame) * 1024)

    waveform, sample_rate = load(str(path))
    reference = torch.tensor(expected, dtype=torch.float32).repeat(1, 1024)
    torch.testing.assert_close(waveform, reference)
    assert sample_rate == 8000

    monkeypatch.setattr(folder_paths, "input_directory", str(tmp_path))
    result = LoadAudio.execute(path.name)[0]
    torch.testing.assert_close(result["waveform"], reference.unsqueeze(0))
    assert result["sample_rate"] == sample_rate
