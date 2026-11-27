from copy import deepcopy

import pytest
import torch

from comfy.cli_args import args

_original_cpu = args.cpu
if not torch.cuda.is_available():
    args.cpu = True
try:
    import nodes
    from comfy.sampler_helpers import prepare_mask
finally:
    args.cpu = _original_cpu


def repeat_latent(latent, amount):
    original = deepcopy(latent)
    repeated = nodes.RepeatLatentBatch().repeat(latent, amount)[0]

    assert latent.keys() == original.keys() == repeated.keys()
    for key, value in original.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(latent[key], value, rtol=0, atol=0)
        else:
            assert latent[key] == value
    torch.testing.assert_close(
        repeated["samples"],
        original["samples"].repeat((amount,) + (1,) * (original["samples"].ndim - 1)),
        rtol=0, atol=0,
    )
    if "batch_index" in original:
        assert repeated["batch_index"] == [4, 6, 4, 7, 9, 7, 10, 12, 10][:3 * amount]
    if "metadata" in original:
        assert repeated["metadata"] == original["metadata"]
    return repeated


@pytest.mark.parametrize(
    "batch_size,mask_values,effective_values,amount,video",
    [
        pytest.param(3, [0, 1], [0, 1, 0], 2, False, id="short"),
        pytest.param(3, [0.25, 0.75], [0.25, 0.75, 0.25], 3, False, id="short-three-copies"),
        pytest.param(3, [0, 0.25, 0.75, 1], [0, 0.25, 0.75], 2, False, id="long"),
        pytest.param(3, [0, 0.25, 1], [0, 0.25, 1], 2, False, id="matching"),
        pytest.param(4, [0, 1], [0, 1, 0, 1], 2, False, id="divisible"),
        pytest.param(3, [0.75], [0.75, 0.75, 0.75], 2, False, id="singleton"),
        pytest.param(3, [0, 1], [0, 1, 0], 1, False, id="identity-short"),
        pytest.param(3, [0, 0.25, 0.75, 1], [0, 0.25, 0.75], 1, False, id="identity-long"),
        pytest.param(3, [0, 0.25, 1], [0, 0.25, 1], 1, False, id="identity-matching"),
        pytest.param(3, [0.75], [0.75, 0.75, 0.75], 1, False, id="identity-singleton"),
        pytest.param(3, [0, 1], [0, 1, 0], 2, True, id="video-short"),
        pytest.param(3, [0, 0.25, 1], [0, 0.25, 1], 2, True, id="video-matching"),
    ],
)
def test_repeat_preserves_effective_mask_batch(batch_size, mask_values, effective_values, amount, video):
    shape = (batch_size, 2, 2, 3, 4) if video else (batch_size, 2, 3, 4)
    samples = torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape)
    masks = torch.tensor(mask_values, dtype=torch.float32).reshape((-1,) + (1,) * (samples.ndim - 1))
    masks = masks.expand((-1, 1) + shape[2:]).clone()
    latent = {"samples": samples, "noise_mask": masks, "metadata": {"name": "original"}}
    if batch_size == 3:
        latent["batch_index"] = [4, 6, 4]

    original_mask = prepare_mask(masks, samples.shape, samples.device)
    expected_mask = torch.tensor(effective_values, dtype=masks.dtype).reshape((-1,) + (1,) * (samples.ndim - 1)).expand_as(samples)
    torch.testing.assert_close(original_mask, expected_mask, rtol=0, atol=0)

    repeated = repeat_latent(latent, amount)
    actual = prepare_mask(repeated["noise_mask"], repeated["samples"].shape, samples.device)
    expected_values = torch.tensor(effective_values * amount, dtype=masks.dtype)
    expected = expected_values.reshape((-1,) + (1,) * (samples.ndim - 1)).expand_as(repeated["samples"])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, original_mask.repeat((amount,) + (1,) * (samples.ndim - 1)), rtol=0, atol=0)
    assert repeated["noise_mask"].shape[1:] == masks.shape[1:]
    assert repeated["noise_mask"].dtype == masks.dtype
    assert repeated["noise_mask"].device == masks.device
    if len(mask_values) == 1:
        assert repeated["noise_mask"].shape == masks.shape
        torch.testing.assert_close(repeated["noise_mask"], masks, rtol=0, atol=0)


@pytest.mark.parametrize("amount", [1, 2])
def test_repeat_without_mask(amount):
    latent = {
        "samples": torch.arange(72, dtype=torch.float32).reshape(3, 2, 3, 4),
        "batch_index": [4, 6, 4],
    }
    repeated = repeat_latent(latent, amount)
    assert "noise_mask" not in repeated


def test_repeat_matches_pre_aligned_mask_input():
    samples = torch.arange(72, dtype=torch.float32).reshape(3, 2, 3, 4)
    masks = torch.tensor([0.25, 0.75]).reshape(2, 1, 1, 1).expand(2, 1, 3, 4).clone()
    aligned_masks = masks[[0, 1, 0]]
    original = prepare_mask(masks, samples.shape, samples.device)
    aligned = prepare_mask(aligned_masks, samples.shape, samples.device)
    torch.testing.assert_close(original, aligned, rtol=0, atol=0)

    repeated = repeat_latent({"samples": samples, "noise_mask": masks}, 2)
    repeated_aligned = repeat_latent({"samples": samples, "noise_mask": aligned_masks}, 2)
    actual = prepare_mask(repeated["noise_mask"], repeated["samples"].shape, samples.device)
    expected = prepare_mask(repeated_aligned["noise_mask"], repeated_aligned["samples"].shape, samples.device)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_repeat_noncontiguous_mask_with_spatial_resize():
    samples = torch.arange(96, dtype=torch.float32).reshape(3, 2, 4, 4)
    masks = torch.tensor(
        [[[[0, 0.5], [0.25, 0.75]]], [[[1, 0.5], [0.75, 0.25]]]],
        dtype=torch.float64,
    ).transpose(-2, -1)
    assert not masks.is_contiguous()
    strides = masks.stride()
    original = prepare_mask(masks, samples.shape, samples.device)

    repeated = repeat_latent({"samples": samples, "noise_mask": masks}, 2)
    actual = prepare_mask(repeated["noise_mask"], repeated["samples"].shape, samples.device)
    torch.testing.assert_close(actual, original.repeat(2, 1, 1, 1), rtol=0, atol=0)
    assert masks.stride() == strides
    assert repeated["noise_mask"].shape[1:] == masks.shape[1:]
    assert repeated["noise_mask"].dtype == masks.dtype
    assert repeated["noise_mask"].device == masks.device


@pytest.mark.parametrize("amount,frames", [(1, [0, 1]), (2, [0.5, 0.5])])
def test_repeat_preserves_lower_rank_temporal_mask_behavior(amount, frames):
    samples = torch.arange(48, dtype=torch.float32).reshape(1, 2, 2, 3, 4)
    # Lower-rank video masks pack time along their leading dimension.
    masks = torch.tensor([0.0, 1.0]).reshape(2, 1, 1, 1).expand(2, 1, 3, 4).clone()
    original = prepare_mask(masks, samples.shape, samples.device)
    expected_original = torch.tensor([0.0, 1.0]).reshape(1, 1, 2, 1, 1).expand_as(samples)
    torch.testing.assert_close(original, expected_original, rtol=0, atol=0)

    repeated = repeat_latent({"samples": samples, "noise_mask": masks}, amount)
    torch.testing.assert_close(repeated["noise_mask"], masks.repeat(amount, 1, 1, 1), rtol=0, atol=0)
    actual = prepare_mask(repeated["noise_mask"], repeated["samples"].shape, samples.device)
    # Preserve the existing temporal resampling when this representation is repeated.
    expected = torch.tensor(frames, dtype=masks.dtype).reshape(1, 1, 2, 1, 1).expand_as(repeated["samples"])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
