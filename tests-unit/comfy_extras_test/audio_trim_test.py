import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy_extras.nodes_audio import TrimAudioDuration  # noqa: E402


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("start, first_frame", [(0.2, 200), (-0.4, 600)])
def test_trim_releases_source_storage(channels, start, first_frame):
    waveform = torch.arange(channels * 1000, dtype=torch.float64).reshape(1, channels, 1000)
    audio = {"waveform": waveform, "sample_rate": 1000}

    result = TrimAudioDuration.execute(audio, start, 0.15)[0]
    trimmed = result["waveform"]

    torch.testing.assert_close(trimmed, waveform[..., first_frame:first_frame + 150])
    assert result["sample_rate"] == audio["sample_rate"]
    assert trimmed.dtype == waveform.dtype
    assert trimmed.device == waveform.device
    assert trimmed.untyped_storage().nbytes() == trimmed.numel() * trimmed.element_size()


def test_trim_preserves_gradients():
    waveform = torch.arange(20, dtype=torch.float32).reshape(1, 1, 20).requires_grad_()
    result = TrimAudioDuration.execute({"waveform": waveform, "sample_rate": 10}, 0.2, 0.3)[0]

    result["waveform"].sum().backward()
    expected = torch.zeros_like(waveform)
    expected[..., 2:5] = 1
    torch.testing.assert_close(waveform.grad, expected)


def test_trim_clamps_to_audio_bounds():
    waveform = torch.arange(10, dtype=torch.float32).reshape(1, 1, 10)
    audio = {"waveform": waveform, "sample_rate": 10}

    result = TrimAudioDuration.execute(audio, -10.0, 0.2)[0]
    torch.testing.assert_close(result["waveform"], waveform[..., :2])
    result = TrimAudioDuration.execute(audio, 0.8, 10.0)[0]
    torch.testing.assert_close(result["waveform"], waveform[..., 8:])


def test_trim_none_and_empty_audio():
    assert TrimAudioDuration.execute(None, 0.0, 1.0)[0] is None
    audio = {"waveform": torch.empty(1, 1, 0), "sample_rate": 10}
    assert TrimAudioDuration.execute(audio, 0.0, 1.0)[0] is audio
