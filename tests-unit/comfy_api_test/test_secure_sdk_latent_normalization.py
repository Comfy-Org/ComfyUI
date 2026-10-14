import asyncio

import pytest
import torch

from comfy import latent_formats, model_patcher, nested_tensor, sample
from comfy_api.latest import _sdk


def _model():
    model = torch.nn.Module()
    model.latent_format = latent_formats.SD15()
    return model_patcher.ModelPatcher(
        model, torch.device("cpu"), torch.device("cpu"))


@pytest.mark.parametrize("channels,fill", [(16, 0.0), (4, 0.0), (16, 0.25)])
def test_model_normalization_matches_native_and_preserves_latent_metadata(channels, fill):
    model = _model()
    samples = torch.full((1, channels, 8, 8), fill, dtype=torch.float64)
    mask = torch.ones((1, 8, 8))
    value = {"samples": samples, "noise_mask": mask, "batch_index": [3]}
    expected = sample.fix_empty_latent_channels(model, samples)

    async def run():
        refs = _sdk.InProcessRefResolver()
        model_ref = _sdk.ModelRef._wrap(await refs.create("MODEL", model))
        latent = _sdk.LatentRef._wrap(await refs.create("LATENT", value))
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            normalized = await model_ref.fix_empty_latent_channels(latent)
        return normalized, await refs.resolve(normalized), await refs.resolve(latent)

    normalized, result, original = asyncio.run(run())
    assert isinstance(normalized, _sdk.LatentRef)
    assert result is not original
    assert original["samples"] is samples
    assert original["samples"].shape == (1, channels, 8, 8)
    assert result["samples"].dtype == expected.dtype
    assert result["samples"].device == expected.device
    torch.testing.assert_close(result["samples"], expected, rtol=0, atol=0)
    assert result["noise_mask"] is mask
    assert result["batch_index"] == [3]


def test_normalization_precedes_actual_latent_noise_generation():
    async def run():
        refs = _sdk.InProcessRefResolver()
        model_ref = _sdk.ModelRef._wrap(await refs.create("MODEL", _model()))
        latent = _sdk.LatentRef._wrap(await refs.create(
            "LATENT", {"samples": torch.zeros((1, 16, 8, 8))}))
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            normalized = await model_ref.fix_empty_latent_channels(latent)
            rng_state = torch.random.get_rng_state().clone()
            noise = await normalized.random_noise(17)
            assert torch.equal(torch.random.get_rng_state(), rng_state)
        return await refs.resolve(normalized), await refs.resolve(noise)

    normalized, noise = asyncio.run(run())
    assert normalized["samples"].shape == noise.shape == (1, 4, 8, 8)
    assert torch.count_nonzero(noise) > 0


def test_native_nested_latent_passthrough_is_retained():
    samples = nested_tensor.NestedTensor([
        torch.zeros((1, 16, 8, 8)), torch.ones((1, 16, 4, 4))])

    async def run():
        refs = _sdk.InProcessRefResolver()
        model_ref = _sdk.ModelRef._wrap(await refs.create("MODEL", _model()))
        latent = _sdk.LatentRef._wrap(await refs.create("LATENT", {"samples": samples}))
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            normalized = await model_ref.fix_empty_latent_channels(latent)
        return await refs.resolve(normalized)

    assert asyncio.run(run())["samples"] is samples


@pytest.mark.parametrize("model_kind,latent_kind", [("IMAGE", "LATENT"), ("MODEL", "TENSOR")])
def test_normalization_rejects_wrong_ref_kinds(model_kind, latent_kind):
    async def run():
        refs = _sdk.InProcessRefResolver()
        model_ref = await refs.create(model_kind, _model())
        latent = await refs.create(latent_kind, {"samples": torch.zeros((1, 4, 8, 8))})
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            await _sdk.current_runtime().ops.apply(
                "sampling.fix_empty_latent_channels", model_ref, {"latent": latent})

    with pytest.raises(TypeError, match="requires a .* ref"):
        asyncio.run(run())


@pytest.mark.parametrize("value", [{}, {"samples": "not a tensor"}, {"samples": torch.zeros(4)}])
def test_normalization_rejects_malformed_latent(value):
    async def run():
        refs = _sdk.InProcessRefResolver()
        model_ref = _sdk.ModelRef._wrap(await refs.create("MODEL", _model()))
        latent = _sdk.LatentRef._wrap(await refs.create("LATENT", value))
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            await model_ref.fix_empty_latent_channels(latent)

    with pytest.raises(TypeError, match="LATENT"):
        asyncio.run(run())
