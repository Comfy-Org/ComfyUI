import asyncio
import threading

import pytest
import torch

from comfy.sd import VAE
from comfy_api import v4
from comfy_api.internal.singleton import SingletonMetaclass
from comfy_api.v4._vae import Vae


@pytest.fixture(autouse=True)
def isolate_api_instances(monkeypatch):
    instances = dict(SingletonMetaclass._instances)
    instances.pop(v4.ComfyAPI, None)
    instances.pop(Vae, None)
    monkeypatch.setattr(SingletonMetaclass, "_instances", instances)


@pytest.fixture
def vae():
    return VAE(
        sd={"pixel_space_vae": torch.tensor(1.0)},
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


@pytest.fixture
def pixels():
    return torch.linspace(0, 1, 2 * 12 * 16 * 3).reshape(2, 12, 16, 3).transpose(1, 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("tiled", [False, True])
@pytest.mark.parametrize("synchronous", [False, True])
async def test_native_round_trip_preserves_pixels_and_tensor_layout(
    vae, pixels, tiled, synchronous
):
    original = pixels.clone()
    api = v4.ComfyAPISync().vae if synchronous else v4.eng.vae
    options = {"tile_x": 8, "tile_y": 12, "overlap": 2} if tiled else {}
    encode = api.encode_tiled if tiled else api.encode
    decode = api.decode_tiled if tiled else api.decode

    with torch.inference_mode():
        if synchronous:
            samples = encode(vae, pixels, **options)
            restored = decode(vae, samples, **options)
        else:
            samples = await encode(vae, pixels, **options)
            restored = await decode(vae, samples, **options)

    torch.testing.assert_close(samples, original.movedim(-1, 1) * 2 - 1)
    torch.testing.assert_close(restored, original)
    torch.testing.assert_close(pixels, original, rtol=0, atol=0)


@pytest.mark.asyncio
@pytest.mark.parametrize("inference", [False, True])
async def test_both_api_forms_preserve_caller_pytorch_context(
    vae, pixels, monkeypatch, inference
):
    native_encode = vae.encode
    observed = []

    def observe_encode(value):
        observed.append(
            (
                threading.get_ident(),
                torch.is_inference_mode_enabled(),
                torch.is_grad_enabled(),
                torch.is_autocast_enabled("cpu"),
                torch.get_autocast_dtype("cpu"),
            )
        )
        return native_encode(value)

    monkeypatch.setattr(vae, "encode", observe_encode)
    with (
        torch.enable_grad(),
        torch.inference_mode(inference),
        torch.autocast("cpu", dtype=torch.bfloat16),
    ):
        asynchronous = await v4.eng.vae.encode(vae, pixels)
        synchronous = v4.ComfyAPISync().vae.encode(vae, pixels)

    assert (
        observed
        == [(threading.get_ident(), inference, not inference, True, torch.bfloat16)] * 2
    )
    torch.testing.assert_close(asynchronous, synchronous)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["encode_tiled", "decode_tiled"])
async def test_tiling_options_reach_the_native_vae_without_layout_changes(
    vae, pixels, monkeypatch, operation
):
    native_operation = getattr(vae, operation)
    value = pixels if operation == "encode_tiled" else pixels.movedim(-1, 1) * 2 - 1
    observed = []

    def observe_tiling(
        value, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None
    ):
        observed.append((value, (tile_x, tile_y, overlap, tile_t, overlap_t)))
        return native_operation(
            value,
            tile_x=tile_x,
            tile_y=tile_y,
            overlap=overlap,
            tile_t=tile_t,
            overlap_t=overlap_t,
        )

    monkeypatch.setattr(vae, operation, observe_tiling)
    with torch.inference_mode():
        default_result = await getattr(v4.eng.vae, operation)(vae, value)
        explicit_result = getattr(v4.ComfyAPISync().vae, operation)(
            vae,
            value,
            tile_x=8,
            tile_y=12,
            overlap=2,
            tile_t=3,
            overlap_t=1,
        )

    assert all(item[0] is value for item in observed)
    assert [item[1] for item in observed] == [
        (None, None, None, None, None),
        (8, 12, 2, 3, 1),
    ]
    torch.testing.assert_close(default_result, explicit_result)


@pytest.mark.asyncio
async def test_sync_and_async_use_the_injected_group(vae, pixels):
    calls = []

    class OffsetVae(Vae):
        async def encode(self, vae, pixels):
            calls.append(vae)
            return pixels + 1

    Vae.inject_instance(OffsetVae())

    asynchronous = await v4.eng.vae.encode(vae, pixels)
    synchronous = v4.ComfyAPISync().vae.encode(vae, pixels)

    assert calls == [vae, vae]
    torch.testing.assert_close(asynchronous, pixels + 1)
    torch.testing.assert_close(synchronous, asynchronous)


def test_sync_call_refuses_work_that_needs_the_event_loop(vae, pixels):
    class SuspendingVae(Vae):
        async def encode(self, vae, pixels):
            await asyncio.sleep(0)
            return pixels

    Vae.inject_instance(SuspendingVae())

    with pytest.raises(RuntimeError, match="needs an event loop"):
        v4.ComfyAPISync().vae.encode(vae, pixels)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation", ["encode", "decode", "encode_tiled", "decode_tiled"]
)
async def test_invalid_vae_errors_are_not_hidden(operation):
    invalid_vae = VAE(sd={})
    values = torch.empty(1, 3, 8, 8)

    with pytest.raises(RuntimeError, match="VAE is invalid"):
        await getattr(v4.eng.vae, operation)(invalid_vae, values)
    with pytest.raises(RuntimeError, match="VAE is invalid"):
        getattr(v4.ComfyAPISync().vae, operation)(invalid_vae, values)


def audio_vae(vae):
    vae.latent_dim = 1
    vae.memory_used_encode = vae.memory_used_decode = lambda shape, dtype: shape[-1]
    return vae


def audio_vae_with_extra_channel(vae):
    vae.extra_1d_channel = 16
    vae.first_stage_model.decode = lambda samples: samples.flatten(1, 2)
    return vae


def video_vae(vae):
    vae.latent_dim = 3
    vae.upscale_ratio = (lambda frames: frames, 8, 8)
    vae.first_stage_model.decode = lambda samples: samples.repeat_interleave(
        8, -1
    ).repeat_interleave(8, -2)
    return vae


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "layout, latent_shape, options",
    [
        (audio_vae, (1, 3, 4096), {}),
        (audio_vae, (1, 3, 4096), {"tile_x": 1024, "overlap": 128}),
        (audio_vae_with_extra_channel, (1, 3, 1, 4096), {}),
        (video_vae, (1, 3, 4, 80, 80), {}),
        (video_vae, (1, 3, 4, 80, 80), {"tile_x": 8}),
        (video_vae, (1, 3, 4, 80, 80), {"tile_x": 16, "tile_y": 8}),
    ],
)
async def test_decode_tiled_with_unset_sizes_decodes_every_kind_of_vae(
    vae, layout, latent_shape, options
):
    vae = layout(vae)
    samples = torch.linspace(-1, 1, torch.Size(latent_shape).numel()).reshape(
        latent_shape
    )

    with torch.inference_mode():
        tiled = await v4.eng.vae.decode_tiled(vae, samples, **options)
        synced = v4.ComfyAPISync().vae.decode_tiled(vae, samples, **options)

    torch.testing.assert_close(tiled, vae.decode(samples))
    torch.testing.assert_close(synced, tiled, rtol=0, atol=0)


@pytest.mark.asyncio
@pytest.mark.parametrize("options", [{}, {"tile_x": 1024, "overlap": 128}])
async def test_encode_tiled_with_unset_sizes_encodes_audio(vae, options):
    vae = audio_vae(vae)
    audio = torch.linspace(0, 1, 3 * 4096).reshape(1, 4096, 3)

    with torch.inference_mode():
        tiled = await v4.eng.vae.encode_tiled(vae, audio, **options)
        synced = v4.ComfyAPISync().vae.encode_tiled(vae, audio, **options)

    torch.testing.assert_close(tiled, vae.encode(audio))
    torch.testing.assert_close(synced, tiled, rtol=0, atol=0)


@pytest.mark.asyncio
@pytest.mark.parametrize("tile_size", [64, 512])
async def test_video_decode_tiled_without_overlap_overlaps_as_vae_decode_tiled(
    vae, monkeypatch, tile_size
):
    import nodes

    vae = video_vae(vae)
    native_decode_tiled = vae.decode_tiled
    overlaps = []

    def observe(samples, overlap=None, **options):
        overlaps.append(overlap)
        return native_decode_tiled(samples, overlap=overlap, **options)

    monkeypatch.setattr(vae, "decode_tiled", observe)
    samples = torch.zeros(1, 3, 4, 80, 80)
    with torch.inference_mode():
        nodes.VAEDecodeTiled().decode(vae, {"samples": samples}, tile_size, 64)
        await v4.eng.vae.decode_tiled(
            vae, samples, tile_x=tile_size // 8, tile_y=tile_size // 8
        )

    assert overlaps[0] == overlaps[1]


@pytest.mark.asyncio
async def test_unset_sizes_reach_a_vae_that_tiles_video_itself(vae, monkeypatch):
    vae = video_vae(vae)
    vae.handles_tiling = True
    received = []
    monkeypatch.setattr(
        vae, "decode_tiled", lambda samples, **options: received.append(options)
    )

    await v4.eng.vae.decode_tiled(vae, torch.zeros(1, 3, 4, 80, 80))

    assert received == [
        dict.fromkeys(("tile_x", "tile_y", "overlap", "tile_t", "overlap_t"))
    ]


@pytest.mark.asyncio
async def test_video_overlap_default_is_at_most_a_quarter_of_the_smaller_tile(
    vae, monkeypatch
):
    vae = video_vae(vae)
    native_decode_tiled = vae.decode_tiled
    overlaps = []

    def observe(samples, overlap=None, **options):
        overlaps.append(overlap)
        return native_decode_tiled(samples, overlap=overlap, **options)

    monkeypatch.setattr(vae, "decode_tiled", observe)
    with torch.inference_mode():
        await v4.eng.vae.decode_tiled(
            vae, torch.zeros(1, 3, 4, 80, 80), tile_x=16, tile_y=8
        )

    assert overlaps == [2]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["encode_tiled", "decode_tiled"])
@pytest.mark.parametrize(
    "sizes",
    [{"tile_x": 0}, {"tile_y": -8}, {"tile_t": 0}, {"overlap": -1}, {"overlap_t": -1}],
)
async def test_a_tile_size_that_is_not_positive_or_a_negative_overlap_is_refused(
    vae, pixels, monkeypatch, operation, sizes
):
    monkeypatch.setattr(
        vae, operation, lambda *args, **kwargs: pytest.fail("reached the VAE")
    )

    with pytest.raises(ValueError, match="tile sizes must be positive"):
        await getattr(v4.ComfyAPI_v4().vae, operation)(vae, pixels, **sizes)
