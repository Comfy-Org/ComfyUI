import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy_extras.nodes_audio import AudioMerge


@pytest.mark.parametrize("batches", [(1, 3), (3, 1), (2, 3), (3, 2), (3, 3)])
@pytest.mark.parametrize("merge_method", ["add", "mean", "subtract", "multiply"])
def test_audio_merge_repeats_shorter_batch(batches, merge_method):
    batch_1, batch_2 = batches
    waveform_1 = torch.linspace(-0.25, 0.25, batch_1 * 2 * 8, dtype=torch.float32).reshape(batch_1, 2, 8)
    waveform_2 = torch.linspace(0.25, -0.25, batch_2 * 2 * 8, dtype=torch.float32).reshape(batch_2, 2, 8)
    waveform_1_before, waveform_2_before = waveform_1.clone(), waveform_2.clone()

    result = AudioMerge.execute(
        {"waveform": waveform_1, "sample_rate": 44100},
        {"waveform": waveform_2, "sample_rate": 44100},
        merge_method,
    )[0]

    output_batch = max(batch_1, batch_2)
    merged_1 = torch.stack([waveform_1[i % batch_1] for i in range(output_batch)])
    merged_2 = torch.stack([waveform_2[i % batch_2] for i in range(output_batch)])
    expected = {
        "add": merged_1 + merged_2,
        "mean": (merged_1 + merged_2) / 2,
        "subtract": merged_1 - merged_2,
        "multiply": merged_1 * merged_2,
    }[merge_method]

    assert result["sample_rate"] == 44100
    assert result["waveform"].shape == (output_batch, 2, 8)
    torch.testing.assert_close(result["waveform"], expected)
    torch.testing.assert_close(waveform_1, waveform_1_before)
    torch.testing.assert_close(waveform_2, waveform_2_before)


@pytest.mark.parametrize("length_1,length_2", [(10, 6), (6, 10)])
def test_audio_merge_aligns_batch_with_length_mismatch(length_1, length_2):
    # Audio1 keeps its role as the length reference (audio2 is trimmed/padded to
    # length_1), while the output batch is the maximum of both input batches.
    batch_1, batch_2 = 2, 3
    waveform_1 = torch.linspace(-0.25, 0.25, batch_1 * 1 * length_1, dtype=torch.float32).reshape(batch_1, 1, length_1)
    waveform_2 = torch.linspace(0.25, -0.25, batch_2 * 1 * length_2, dtype=torch.float32).reshape(batch_2, 1, length_2)

    result = AudioMerge.execute(
        {"waveform": waveform_1, "sample_rate": 44100},
        {"waveform": waveform_2, "sample_rate": 44100},
        "add",
    )[0]

    if length_2 > length_1:
        aligned_2 = waveform_2[..., :length_1]
    else:
        aligned_2 = torch.nn.functional.pad(waveform_2, (0, length_1 - length_2))
    expected = torch.stack([waveform_1[i % batch_1] + aligned_2[i % batch_2] for i in range(batch_2)])

    assert result["sample_rate"] == 44100
    assert result["waveform"].shape == (batch_2, 1, length_1)
    torch.testing.assert_close(result["waveform"], expected)
