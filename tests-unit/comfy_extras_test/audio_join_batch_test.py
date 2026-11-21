import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy_extras.nodes_audio import JoinAudioChannels


@pytest.mark.parametrize("left_batch,right_batch", [(1, 3), (3, 1), (2, 3), (3, 2), (3, 3)])
def test_join_audio_channel_batches(left_batch, right_batch):
    left = torch.arange(left_batch * 8, dtype=torch.float64).reshape(left_batch, 1, 8) / 100
    right = -torch.arange(right_batch * 12, dtype=torch.float64).reshape(right_batch, 1, 12) / 100
    left_before, right_before = left.clone(), right.clone()
    result = JoinAudioChannels.execute(
        {"waveform": left, "sample_rate": 8000},
        {"waveform": right, "sample_rate": 8000},
    )[0]

    expected = torch.stack([
        torch.stack([left[i % left_batch, 0], right[i % right_batch, 0, :8]])
        for i in range(max(left_batch, right_batch))
    ])
    torch.testing.assert_close(result["waveform"], expected)
    assert result["sample_rate"] == 8000
    assert result["waveform"].device == left.device
    torch.testing.assert_close(left, left_before)
    torch.testing.assert_close(right, right_before)


@pytest.mark.parametrize("left_rate,right_rate", [(8000, 16000), (16000, 8000)])
def test_join_batches_with_resampling(left_rate, right_rate):
    left = torch.linspace(-0.5, 0.5, 80).reshape(1, 1, 80)
    right = torch.linspace(0.5, -0.5, 96).reshape(1, 1, 96).repeat(3, 1, 1)
    result = JoinAudioChannels.execute(
        {"waveform": left, "sample_rate": left_rate},
        {"waveform": right, "sample_rate": right_rate},
    )[0]
    expected = torch.cat([
        JoinAudioChannels.execute(
            {"waveform": left, "sample_rate": left_rate},
            {"waveform": right[i:i + 1], "sample_rate": right_rate},
        )[0]["waveform"] for i in range(3)
    ])
    torch.testing.assert_close(result["waveform"], expected)
    assert result["sample_rate"] == 16000


def test_join_audio_trims_longer_left():
    left = torch.arange(12, dtype=torch.float32).reshape(1, 1, 12)
    right = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
    result = JoinAudioChannels.execute(
        {"waveform": left, "sample_rate": 8000},
        {"waveform": right, "sample_rate": 8000},
    )[0]
    torch.testing.assert_close(result["waveform"], torch.cat([left[..., :8], right], dim=1))


def test_join_audio_missing_input_passthrough():
    audio = {"waveform": torch.zeros(2, 1, 8), "sample_rate": 8000}
    assert JoinAudioChannels.execute(audio, None)[0] is audio
    assert JoinAudioChannels.execute(None, audio)[0] is audio
    assert JoinAudioChannels.execute(None, None)[0] is None


def test_join_audio_still_rejects_stereo_input():
    audio = {"waveform": torch.zeros(1, 2, 8), "sample_rate": 8000}
    with pytest.raises(ValueError, match="must be mono"):
        JoinAudioChannels.execute(audio, audio)
