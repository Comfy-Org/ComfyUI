from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args import args

_original_cpu = args.cpu
if not torch.cuda.is_available():
    args.cpu = True
try:
    from comfy.model_base import HunyuanVideo15_SR_Distilled
    from comfy_extras.nodes_hunyuan import HunyuanVideo15SuperResolution
finally:
    args.cpu = _original_cpu


class EncodedVAE:
    """Replace only neural encoding with a known single-frame latent."""

    def __init__(self, encoded):
        self.encoded = encoded
        self.inputs = []

    def encode(self, image):
        self.inputs.append(image.clone())
        return self.encoded


def sr_inputs(batch_size):
    samples = torch.arange(batch_size * 32 * 3 * 2 * 3, dtype=torch.float32).reshape(batch_size, 32, 3, 2, 3) / 100 + 2
    encoded = torch.arange(32 * 2 * 3, dtype=torch.float32).reshape(1, 32, 1, 2, 3) / 50 - 0.75
    start_image = torch.linspace(0, 1, 32 * 48 * 3).reshape(1, 32, 48, 3)
    return samples, encoded, start_image


def expected_condition(samples, encoded, with_reference):
    reference = torch.cat((encoded.expand(samples.shape[0], -1, -1, -1, -1), torch.zeros_like(samples[:, :, 1:])), dim=2)
    if not with_reference:
        reference = torch.zeros_like(samples)
    reference_mask = samples.new_tensor([int(with_reference), 0, 0]).reshape(1, 1, 3, 1, 1)
    reference_mask = reference_mask.expand(samples.shape[0], 1, 3, 2, 3)
    return torch.cat((reference, reference_mask, samples, torch.ones_like(reference_mask)), dim=1)


@pytest.mark.parametrize(
    "batch_size,with_reference,noise_augmentation,with_vision",
    [(1, True, 0.0, False), (2, True, 0.7, True),
     (1, False, 0.0, False), (2, False, 0.7, True)],
)
def test_sr_node_keeps_reference_mask_and_low_quality_latent_separate(
    monkeypatch, batch_size, with_reference, noise_augmentation, with_vision,
):
    monkeypatch.setattr(args, "gpu_only", False)
    samples, encoded, start_image = sr_inputs(batch_size)
    latent = {"samples": samples, "batch_index": list(range(batch_size)), "metadata": {"source": "low-quality"}}
    positive = [[torch.ones(1, 2, 3), {"label": "positive", "pooled_output": torch.tensor([0.25, 0.75])}]]
    negative = [[torch.zeros(1, 2, 3), {"label": "negative", "pooled_output": torch.tensor([-0.25, -0.75])}]]
    original = deepcopy((latent, positive, negative, encoded, start_image))
    vision = SimpleNamespace(last_hidden_state=torch.arange(6).reshape(1, 2, 3)) if with_vision else None
    vae = EncodedVAE(encoded)

    positive_out, negative_out, latent_out = HunyuanVideo15SuperResolution.execute(
        positive, negative, latent, noise_augmentation,
        vae=vae if with_reference else None,
        start_image=start_image if with_reference else None,
        clip_vision_output=vision,
    ).result

    expected = expected_condition(samples, encoded, with_reference)
    for output, original_conditioning in ((positive_out, positive), (negative_out, negative)):
        torch.testing.assert_close(output[0][1]["concat_latent_image"], expected, rtol=0, atol=0)
        assert output[0][1]["noise_augmentation"] == noise_augmentation
        assert output[0][0] is original_conditioning[0][0]
        assert output[0][1]["label"] == original_conditioning[0][1]["label"]
        torch.testing.assert_close(output[0][1]["pooled_output"], original_conditioning[0][1]["pooled_output"], rtol=0, atol=0)
        if with_vision:
            assert output[0][1]["clip_vision_output"] is vision
        else:
            assert "clip_vision_output" not in output[0][1]

    assert latent_out is latent
    assert latent.keys() == original[0].keys()
    torch.testing.assert_close(latent["samples"], original[0]["samples"], rtol=0, atol=0)
    assert latent["batch_index"] == original[0]["batch_index"]
    assert latent["metadata"] == original[0]["metadata"]
    for actual, before in ((positive, original[1]), (negative, original[2])):
        assert actual[0][1].keys() == before[0][1].keys()
        assert actual[0][1]["label"] == before[0][1]["label"]
        torch.testing.assert_close(actual[0][0], before[0][0], rtol=0, atol=0)
        torch.testing.assert_close(actual[0][1]["pooled_output"], before[0][1]["pooled_output"], rtol=0, atol=0)
    torch.testing.assert_close(encoded, original[3], rtol=0, atol=0)
    torch.testing.assert_close(start_image, original[4], rtol=0, atol=0)
    assert len(vae.inputs) == int(with_reference)
    if with_reference:
        torch.testing.assert_close(vae.inputs[0], start_image, rtol=0, atol=0)


@pytest.mark.parametrize("with_reference", [False, True])
@pytest.mark.parametrize("noise_augmentation", [0.0, 0.7])
def test_sr_conditioning_reaches_real_concat_consumer(monkeypatch, with_reference, noise_augmentation):
    monkeypatch.setattr(args, "gpu_only", False)
    samples, encoded, start_image = sr_inputs(2)
    conditioning = [[torch.zeros(1, 2, 3), {}]]
    positive, negative, _ = HunyuanVideo15SuperResolution.execute(
        conditioning, conditioning, {"samples": samples}, noise_augmentation,
        vae=EncodedVAE(encoded) if with_reference else None,
        start_image=start_image if with_reference else None,
    ).result
    original_condition = positive[0][1]["concat_latent_image"].clone()
    seed = 1234
    if noise_augmentation == 0:
        expected_low_quality = 0.75 * samples
    else:
        generator = torch.Generator(device="cpu").manual_seed(seed - 10)
        expected_low_quality = 0.7 * torch.randn(samples.shape, generator=generator) + 0.3 * samples
    reference, reference_mask, _, low_quality_mask = expected_condition(samples, encoded, with_reference).split([32, 1, 32, 1], dim=1)
    expected = torch.cat((reference, reference_mask, expected_low_quality, low_quality_mask), dim=1)

    # concat_cond assembles tensors without accessing diffusion weights.
    for metadata in (positive[0][1], negative[0][1]):
        actual = HunyuanVideo15_SR_Distilled.concat_cond(
            SimpleNamespace(), noise=torch.zeros_like(samples), device=torch.device("cpu"), seed=seed, **metadata,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(metadata["concat_latent_image"], original_condition, rtol=0, atol=0)
