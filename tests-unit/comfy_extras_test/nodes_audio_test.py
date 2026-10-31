import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy_extras.nodes_audio import AudioConcat


@pytest.mark.parametrize("batches", [(1, 3), (3, 1), (2, 3), (3, 3)])
@pytest.mark.parametrize("direction", ["after", "before"])
def test_audio_concat_repeats_shorter_batch(batches, direction):
    left = torch.arange(batches[0] * 8, dtype=torch.float32).reshape(-1, 2, 4) / 100
    right = torch.arange(batches[1] * 6, dtype=torch.float32).reshape(-1, 2, 3) / 100 + 0.5
    original_left, original_right = left.clone(), right.clone()
    audio1 = {"waveform": left, "sample_rate": 44100}
    audio2 = {"waveform": right, "sample_rate": 44100}

    result = AudioConcat.execute(audio1, audio2, direction)[0]

    assert result["sample_rate"] == 44100
    assert result["waveform"].shape == (max(batches), 2, 7)
    for i, waveform in enumerate(result["waveform"]):
        first, second = (left[i % batches[0]], right[i % batches[1]])
        if direction == "before":
            first, second = second, first
        torch.testing.assert_close(waveform[:, : first.shape[-1]], first)
        torch.testing.assert_close(waveform[:, first.shape[-1] :], second)
    torch.testing.assert_close(left, original_left)
    torch.testing.assert_close(right, original_right)
