from collections import OrderedDict
import json

import pytest
import torch

from comfy.cli_args import args
from comfy.sd import VAE
from comfy.pixel_space_convert import PixelspaceConversionVAE
from comfy_api import v4
from comfy_api.internal.singleton import SingletonMetaclass
from comfy_api.v4._vae import Vae


@pytest.fixture(autouse=True)
def isolate_api_and_cpu_loader(monkeypatch):
    instances = dict(SingletonMetaclass._instances)
    instances.pop(v4.ComfyAPI, None)
    instances.pop(Vae, None)
    monkeypatch.setattr(SingletonMetaclass, "_instances", instances)
    monkeypatch.setattr(args, "cpu_vae", True)
    monkeypatch.setattr(args, "fp32_vae", True)


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_constructed_vae_runs_native_encode_and_decode(synchronous):
    weights = {"pixel_space_vae": torch.tensor(1.0)}
    original_weight = weights["pixel_space_vae"]
    pixels = torch.linspace(0, 1, 8 * 12 * 3).reshape(1, 8, 12, 3)
    api = v4.ComfyAPISync().vae if synchronous else v4.eng.vae

    with torch.inference_mode():
        if synchronous:
            vae = api.from_state_dict(weights)
            samples = api.encode(vae, pixels)
            restored = api.decode(vae, samples)
        else:
            vae = await api.from_state_dict(weights)
            samples = await api.encode(vae, pixels)
            restored = await api.decode(vae, samples)

    assert vae.device == torch.device("cpu")
    assert weights["pixel_space_vae"] is original_weight
    torch.testing.assert_close(samples, pixels.movedim(-1, 1) * 2 - 1)
    torch.testing.assert_close(restored, pixels)


@pytest.mark.asyncio
async def test_weight_metadata_reaches_the_native_module_loader(monkeypatch):
    weights = OrderedDict(pixel_space_vae=torch.tensor(1.0))
    weights._metadata = {"": {"version": 7}}
    native_load = PixelspaceConversionVAE._load_from_state_dict
    observed = []

    def observe_load(
        self, state_dict, prefix, local_metadata, strict, missing, unexpected, errors
    ):
        observed.append(local_metadata.copy())
        return native_load(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing,
            unexpected,
            errors,
        )

    monkeypatch.setattr(PixelspaceConversionVAE, "_load_from_state_dict", observe_load)
    await v4.eng.vae.from_state_dict(weights)
    v4.ComfyAPISync().vae.from_state_dict(weights)

    assert observed == [{"version": 7}, {"version": 7}]
    assert weights._metadata == {"": {"version": 7}}


@pytest.mark.asyncio
@pytest.mark.parametrize("weights", [{}, {"unrecognized.weight": torch.empty(0)}])
async def test_invalid_weights_fail_before_returning_a_vae(weights):
    with pytest.raises(RuntimeError, match="VAE is invalid"):
        await v4.eng.vae.from_state_dict(weights)
    with pytest.raises(RuntimeError, match="VAE is invalid"):
        v4.ComfyAPISync().vae.from_state_dict(weights)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("downscale", "upscale", "expected_factors"),
    [
        (2.5, 4.5, (2.5, 4.5, None, 2.5, 4.5)),
        (
            (lambda length: (length + 3) // 4, 8, 8),
            (lambda length: length * 4, 16, 16),
            (8, 16, 4, None, None),
        ),
    ],
)
async def test_info_preserves_legacy_fields_and_detaches_metadata(
    downscale, upscale, expected_factors
):
    vae = VAE(no_init=True)
    vae.latent_channels = 16
    vae.vae_dtype = torch.bfloat16
    vae.downscale_ratio = downscale
    vae.upscale_ratio = upscale
    encode, decode, temporal, downscale_value, upscale_value = expected_factors
    expected = {
        "latent_channels": 16,
        "dtype": "bfloat16",
        "spacial_compression_decode": decode,
        "spacial_compression_encode": encode,
        "temporal_compression_decode": temporal,
        "downscale_ratio": downscale_value,
        "upscale_ratio": upscale_value,
    }

    asynchronous = await v4.eng.vae.info(vae)
    synchronous = v4.ComfyAPISync().vae.info(vae)

    assert asynchronous == synchronous == expected
    assert type(asynchronous) is type(synchronous) is dict
    assert all(
        type(value) in (int, float, str, type(None)) for value in asynchronous.values()
    )
    asynchronous["latent_channels"] = 99
    assert synchronous == expected
    assert await v4.eng.vae.info(vae) == expected
    assert vae.latent_channels == 16


@pytest.mark.asyncio
async def test_info_remains_available_without_initialized_weights():
    vae = VAE(sd={})

    asynchronous = await v4.eng.vae.info(vae)
    synchronous = v4.ComfyAPISync().vae.info(vae)

    assert asynchronous == synchronous
    assert asynchronous["dtype"] is None
    assert asynchronous["latent_channels"] == vae.latent_channels


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata", "error_type"),
    [
        ({"config": "malformed JSON"}, json.JSONDecodeError),
        ({"config": {"audio_vae": {}}}, AssertionError),
    ],
)
async def test_failed_native_loading_preserves_the_callers_weight_mapping(
    metadata, error_type
):
    weights = {
        "vocoder.resblocks.0.convs1.0.weight": torch.empty(0),
        "audio_vae.test": torch.empty(0),
    }
    original = weights.copy()
    native_weights = weights.copy()
    with pytest.raises(error_type) as native_error:
        VAE(sd=native_weights, metadata=metadata)
    assert "audio_vae.test" not in native_weights
    assert native_weights["autoencoder.test"] is original["audio_vae.test"]

    with pytest.raises(error_type) as async_error:
        await v4.eng.vae.from_state_dict(weights, metadata=metadata)
    assert weights.keys() == original.keys()
    assert all(weights[key] is value for key, value in original.items())

    with pytest.raises(error_type) as sync_error:
        v4.ComfyAPISync().vae.from_state_dict(weights, metadata=metadata)
    assert weights.keys() == original.keys()
    assert all(weights[key] is value for key, value in original.items())
    assert str(async_error.value) == str(sync_error.value) == str(native_error.value)
    if error_type is AssertionError:
        assert str(native_error.value) == "Vocoder config is required for audio VAE"


@pytest.mark.asyncio
async def test_injected_loader_receives_metadata_and_its_refusal_propagates():
    weights = {"pixel_space_vae": torch.tensor(1.0)}
    metadata = {"config": {"format": "provider-owned"}}
    calls = []
    refusal = PermissionError("provider refused this model")

    class RefusingVae(Vae):
        async def from_state_dict(self, sd, *, metadata=None):
            calls.append((sd, metadata))
            raise refusal

    Vae.inject_instance(RefusingVae())
    with pytest.raises(PermissionError) as async_error:
        await v4.eng.vae.from_state_dict(weights, metadata=metadata)
    with pytest.raises(PermissionError) as sync_error:
        v4.ComfyAPISync().vae.from_state_dict(weights, metadata=metadata)

    assert async_error.value is sync_error.value is refusal
    assert len(calls) == 2
    assert all(sd is weights and supplied is metadata for sd, supplied in calls)
