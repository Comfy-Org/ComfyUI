import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import gc
import threading
from types import SimpleNamespace
import weakref

import pytest
import torch

from comfy.sd import VAE
from comfy import model_management
from comfy_api.latest import _sdk


class _Decoder(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, padding=1, groups=3, bias=False, dtype=dtype)
        self.conv.weight.data.copy_(torch.arange(27, dtype=dtype).reshape(3, 1, 3, 3) / 27)
        self.other = torch.nn.Conv1d(1, 1, 3, padding=1, padding_mode="replicate")
        self.calls = []
        self.fail = False
        self.oom_once = False

    def decode(self, samples, **kwargs):
        self.calls.append(("decode", self.conv.padding_mode, kwargs))
        if self.fail:
            raise RuntimeError("decoder failed")
        if self.oom_once:
            self.oom_once = False
            raise torch.OutOfMemoryError("fixture OOM")
        return self.conv(samples)

    def encode(self, pixels):
        self.calls.append(("encode", self.conv.padding_mode, {}))
        return self.conv(pixels)


@pytest.fixture
def make_vae(monkeypatch):
    monkeypatch.setattr(model_management, "load_models_gpu", lambda *a, **kw: None)
    monkeypatch.setattr(model_management, "soft_empty_cache", lambda *a, **kw: None)

    def make(model=None, dtype=torch.float32):
        vae = object.__new__(VAE)
        vae.first_stage_model = model if model is not None else _Decoder(dtype)
        vae.device = vae.output_device = torch.device("cpu")
        vae.vae_dtype = dtype
        vae.vae_output_dtype = lambda: dtype
        vae.patcher = SimpleNamespace(get_free_memory=lambda device: 4096)
        vae.memory_used_decode = vae.memory_used_encode = lambda *args: 1
        vae.disable_offload = False
        vae.latent_dim = 2
        vae.latent_channels = 3
        vae.output_channels = 3
        vae.extra_1d_channel = None
        vae.handles_tiling = False
        vae.format_encoded = None
        vae.crop_input = False
        vae.pad_channel_value = None
        vae.upscale_ratio = vae.downscale_ratio = 1
        vae.upscale_index_formula = None
        vae.process_output = vae.process_input = lambda value: value
        return vae
    return make


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tiled", [False, True])
def test_native_decode_matches_permanent_source_padding_then_restores(make_vae, dtype, tiled):
    vae = make_vae(dtype=dtype)
    model = vae.first_stage_model
    samples = torch.arange(105, dtype=dtype).reshape(1, 3, 5, 7) / 105
    decode = (lambda: vae.decode_tiled(samples, tile_x=64, tile_y=64)) if tiled else (lambda: vae.decode(samples))
    ordinary = decode()
    model.conv.padding_mode = "circular"
    expected = decode()
    model.conv.padding_mode = "zeros"
    with vae.decode_policy("circular"):
        actual = decode()
    assert torch.equal(actual, expected)
    assert not torch.equal(ordinary, actual)
    assert model.conv.padding_mode == "zeros"
    assert model.other.padding_mode == "replicate"
    assert torch.equal(decode(), ordinary)


@pytest.mark.parametrize("tiled", [False, True])
def test_decode_failure_restores_distinct_padding_modes(make_vae, tiled):
    vae = make_vae()
    model = vae.first_stage_model
    model.conv.padding_mode = "replicate"
    model.extra = torch.nn.Conv2d(1, 1, 3, padding=1, padding_mode="reflect")
    model.fail = True
    with pytest.raises(RuntimeError, match="decoder failed"):
        with vae.decode_policy("circular"):
            if tiled:
                vae.decode_tiled(torch.zeros(1, 3, 5, 7), tile_x=64, tile_y=64)
            else:
                vae.decode(torch.zeros(1, 3, 5, 7))
    assert model.conv.padding_mode == "replicate"
    assert model.extra.padding_mode == "reflect"


def test_oom_fallback_keeps_scope_and_restores(make_vae):
    vae = make_vae()
    samples = torch.arange(105, dtype=torch.float32).reshape(1, 3, 5, 7)
    vae.first_stage_model.conv.padding_mode = "circular"
    expected = vae.decode_tiled_(samples).movedim(1, -1)
    vae.first_stage_model.conv.padding_mode = "zeros"
    vae.first_stage_model.calls.clear()
    vae.first_stage_model.oom_once = True
    with vae.decode_policy("circular"):
        actual = vae.decode(samples)
    assert torch.equal(expected, actual)
    assert {mode for _, mode, _ in vae.first_stage_model.calls} == {"circular"}
    assert vae.first_stage_model.conv.padding_mode == "zeros"


@pytest.mark.parametrize("operation", ["decode", "decode_tiled", "encode", "encode_tiled"])
def test_shared_model_serializes_other_canonical_operations(make_vae, operation):
    first = make_vae()
    second = make_vae(first.first_stage_model)
    samples = torch.ones(1, 3, 5, 7)
    started = threading.Event()

    @torch.inference_mode()
    def competing():
        started.set()
        if operation == "decode":
            return second.decode(samples)
        if operation == "decode_tiled":
            return second.decode_tiled(samples, tile_x=64, tile_y=64)
        if operation == "encode":
            return second.encode(samples.movedim(1, -1))
        return second.encode_tiled(samples.movedim(1, -1), tile_x=64, tile_y=64, overlap=8)

    with ThreadPoolExecutor(max_workers=1) as pool:
        with first.decode_policy("circular"):
            future = pool.submit(competing)
            assert started.wait(5)
            with pytest.raises(TimeoutError):
                future.result(timeout=0.05)
            assert first.first_stage_model.calls == []
        assert future.result(timeout=5).numel() == samples.numel()
    assert {mode for _, mode, _ in first.first_stage_model.calls} == {"zeros"}


def test_independent_model_operations_are_not_serialized(make_vae):
    first, second = make_vae(), make_vae()
    with ThreadPoolExecutor(max_workers=1) as pool:
        with first.decode_policy("circular"):
            future = pool.submit(second.decode, torch.ones(1, 3, 5, 7))
            assert future.result(timeout=5).shape == (1, 5, 7, 3)
    assert {mode for _, mode, _ in second.first_stage_model.calls} == {"zeros"}


@pytest.mark.parametrize("operation", ["decode", "decode_tiled", "encode", "encode_tiled"])
def test_shallow_roots_with_shared_children_serialize(make_vae, operation):
    first = make_vae()
    stage = copy.copy(first.first_stage_model)
    stage._modules = first.first_stage_model._modules.copy()
    second = make_vae(stage)
    assert stage is not first.first_stage_model
    assert stage.conv is first.first_stage_model.conv
    samples = torch.ones(1, 3, 5, 7)
    started = threading.Event()
    @torch.inference_mode()
    def competing():
        started.set()
        if operation == "decode":
            return second.decode(samples)
        if operation == "decode_tiled":
            return second.decode_tiled(samples, tile_x=64, tile_y=64)
        if operation == "encode":
            return second.encode(samples.movedim(1, -1))
        return second.encode_tiled(samples.movedim(1, -1), tile_x=64, tile_y=64, overlap=8)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with first.decode_policy("circular"):
            future = pool.submit(competing)
            assert started.wait(5)
            with pytest.raises(TimeoutError):
                future.result(timeout=0.05)
            assert first.first_stage_model.calls == []
        future.result(timeout=5)
    assert {mode for _, mode, _ in stage.calls} == {"zeros"}
    assert stage.conv.padding_mode == "zeros"


@pytest.mark.parametrize("operation", ["encode", "encode_tiled"])
def test_actual_compile_selector_shared_encoder_is_serialized(make_vae, operation):
    class SplitStage(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Conv2d(3, 3, 3, padding=1)
            self.decoder = torch.nn.Conv2d(3, 3, 3, padding=1)
            self.calls = []
        def encode(self, pixels):
            self.calls.append(self.encoder.padding_mode)
            return self.encoder(pixels)
    source = make_vae(SplitStage())
    source.patcher.clone = lambda: copy.copy(source.patcher)
    async def compile_ref():
        refs = _sdk.InProcessRefResolver()
        ref = _sdk.VaeRef._wrap(await refs.create("VAE", source))
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            result = await ref.compile(encoder=False, decoder=True)
        return await refs.resolve(result)
    compiled = asyncio.run(compile_ref())
    assert compiled.first_stage_model is not source.first_stage_model
    assert compiled.first_stage_model.encoder is source.first_stage_model.encoder
    assert compiled.first_stage_model.decoder._orig_mod is source.first_stage_model.decoder
    assert compiled.first_stage_model.encoder.weight is source.first_stage_model.encoder.weight
    started = threading.Event()
    @torch.inference_mode()
    def competing():
        started.set()
        pixels = torch.ones(1, 5, 7, 3)
        if operation == "encode":
            return source.encode(pixels)
        return source.encode_tiled(pixels, tile_x=64, tile_y=64, overlap=8)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(RuntimeError, match="scope failure"):
            with compiled.decode_policy("circular"):
                future = pool.submit(competing)
                assert started.wait(5)
                with pytest.raises(TimeoutError):
                    future.result(timeout=0.05)
                assert source.first_stage_model.calls == []
                raise RuntimeError("scope failure")
        future.result(timeout=5)
    assert source.first_stage_model.calls == ["zeros"] * (1 if operation == "encode" else 3)
    assert source.first_stage_model.encoder.padding_mode == "zeros"
    assert source.first_stage_model.decoder.padding_mode == "zeros"


def test_invalid_vae_keeps_native_error():
    value = object.__new__(VAE)
    value.first_stage_model = None
    with pytest.raises(RuntimeError, match="VAE is invalid"):
        value.decode(torch.zeros(1, 3, 5, 7))
    with pytest.raises(RuntimeError, match="VAE is invalid"):
        value.decode_policy("circular")


def test_nested_scope_and_model_lifetime(make_vae):
    vae = make_vae()
    model = vae.first_stage_model
    with vae.decode_policy("circular"):
        with vae.decode_policy("default"):
            assert model.conv.padding_mode == "circular"
        with vae.decode_policy("circular"):
            assert model.conv.padding_mode == "circular"
        assert model.conv.padding_mode == "circular"
    assert model.conv.padding_mode == "zeros"
    ref = weakref.ref(model)
    del model, vae
    gc.collect()
    assert ref() is None


@pytest.mark.parametrize("method", ["decode", "decode_tensor", "decode_tiled", "decode_tensor_tiled"])
def test_public_policy_reaches_native_math(make_vae, method):
    async def run():
        value = make_vae()
        refs = _sdk.InProcessRefResolver()
        vae = _sdk.VaeRef._wrap(await refs.create("VAE", value))
        sample = torch.arange(105, dtype=torch.float32).reshape(1, 3, 5, 7)
        latent = _sdk.LatentRef._wrap(await refs.create("LATENT", {"samples": sample}))
        kwargs = {"tile_size": 64, "overlap": 16} if "tiled" in method else {}
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            actual = await getattr(vae, method)(latent, padding_mode="circular", **kwargs)
        assert value.first_stage_model.conv.padding_mode == "zeros"
        assert {mode for _, mode, _ in value.first_stage_model.calls} == {"circular"}
        with value.decode_policy("circular"):
            expected = value.decode_tiled(sample, tile_x=64, tile_y=64, overlap=16) if "tiled" in method else value.decode(sample)
        assert torch.equal(await refs.resolve(actual), expected)
    asyncio.run(run())


@pytest.mark.parametrize("method", ["decode", "decode_tensor", "decode_tiled", "decode_tensor_tiled"])
def test_default_request_keeps_existing_provider_signature(method):
    async def run():
        refs = _sdk.InProcessRefResolver()
        vae = _sdk.VaeRef._wrap(await refs.create("VAE", object()))
        latent = _sdk.LatentRef._wrap(await refs.create("LATENT", {}))
        ops = _sdk.InProcessOps()
        calls = []
        async def provider(ref, latent, **kwargs):
            calls.append(kwargs)
        ops.register_op("vae." + method, provider)
        with _sdk.bind_runtime(refs, None, ops):
            await getattr(vae, method)(latent)
        expected = {"tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8} if "tiled" in method else {}
        assert calls == [expected]
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["reflect", "zeros", None, True, [], {}])
def test_invalid_policy_fails_before_provider_decode(make_vae, mode):
    async def run():
        value = make_vae()
        refs = _sdk.InProcessRefResolver()
        vae = _sdk.VaeRef._wrap(await refs.create("VAE", value))
        latent = _sdk.LatentRef._wrap(await refs.create("LATENT", {"samples": torch.ones(1, 3, 5, 7)}))
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            with pytest.raises(ValueError, match="padding_mode"):
                await vae.decode(latent, padding_mode=mode)
        assert value.first_stage_model.calls == []
    asyncio.run(run())


def test_external_provider_without_policy_fails_explicitly():
    provider = SimpleNamespace(decode=lambda _: pytest.fail("must not decode"))
    with pytest.raises(ValueError, match="does not support scoped circular"):
        _sdk.InProcessOps._vae_decode_policy(provider, "circular")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_native_tiles_preserve_omission_and_source_math(make_vae, dtype):
    async def run():
        value = make_vae(dtype=dtype)
        sample = torch.arange(105, dtype=dtype).reshape(1, 3, 5, 7) / 105
        refs = _sdk.InProcessRefResolver()
        vae = _sdk.VaeRef._wrap(await refs.create("VAE", value))
        latent = _sdk.LatentRef._wrap(await refs.create("LATENT", {"samples": sample}))
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            actual = await vae.decode_tiled_native(latent, tile_x=512, tile_y=512, padding_mode="circular")
        assert value.first_stage_model.conv.padding_mode == "zeros"
        with value.decode_policy("circular"):
            expected = value.decode_tiled(sample, tile_x=512, tile_y=512)
        assert torch.equal(expected, await refs.resolve(actual))
    asyncio.run(run())


@pytest.mark.parametrize("shape", [(1, 2, 3, 3), (2, 4, 2, 3, 3)])
@pytest.mark.parametrize("options", [{}, {"tile_x": 512, "tile_y": 512}, {"tile_x": 7, "tile_y": 5, "overlap": 0, "tile_t": 9, "overlap_t": 0}])
def test_native_tiles_pass_exact_kwargs_and_preserve_video_batch(shape, options):
    async def run():
        refs = _sdk.InProcessRefResolver()
        calls = []
        pixels = torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape)
        def decode(samples, **kwargs):
            calls.append((samples, kwargs))
            return pixels
        value = SimpleNamespace(decode_tiled=decode)
        vae = _sdk.VaeRef._wrap(await refs.create("VAE", value))
        samples = torch.zeros(1, 3, 2, 3)
        latent = _sdk.LatentRef._wrap(await refs.create("LATENT", {"samples": samples}))
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            output = await vae.decode_tiled_native(latent, **options)
        assert calls == [(samples, options)]
        assert await refs.resolve(output) is pixels
    asyncio.run(run())


@pytest.mark.parametrize("name,bad", [("tile_x", 0), ("tile_y", True), ("tile_t", 4097), ("overlap", -1), ("overlap_t", 1.5)])
def test_invalid_native_tiles_do_not_decode(name, bad):
    async def run():
        refs = _sdk.InProcessRefResolver()
        value = SimpleNamespace(decode_tiled=lambda *a, **kw: pytest.fail("must not decode"))
        vae = _sdk.VaeRef._wrap(await refs.create("VAE", value))
        latent = _sdk.LatentRef._wrap(await refs.create("LATENT", {}))
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            with pytest.raises(ValueError, match=name):
                await vae.decode_tiled_native(latent, **{name: bad})
    asyncio.run(run())


@pytest.mark.parametrize("tile", [512, 2])
def test_native_video_omission_preserves_success_and_errors(make_vae, tile):
    async def run():
        value = make_vae()
        value.latent_dim = 3
        value.upscale_ratio = (1, 1, 1)
        value.first_stage_model.conv = torch.nn.Conv3d(3, 3, 3, padding=1, groups=3, bias=False)
        samples = torch.ones(1, 3, 2, 5, 7)
        if tile == 512:
            expected = value.decode_tiled(samples, tile_x=tile, tile_y=tile)
        else:
            with pytest.raises(TypeError, match="NoneType") as native:
                value.decode_tiled(samples, tile_x=tile, tile_y=tile)
        refs = _sdk.InProcessRefResolver()
        vae = _sdk.VaeRef._wrap(await refs.create("VAE", value))
        latent = _sdk.LatentRef._wrap(await refs.create("LATENT", {"samples": samples}))
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            if tile == 512:
                out = await vae.decode_tiled_native(latent, tile_x=tile, tile_y=tile, padding_mode="circular")
                assert torch.equal(await refs.resolve(out), expected)
                assert expected.shape == (1, 2, 5, 7, 3)
            else:
                with pytest.raises(type(native.value)) as public:
                    await vae.decode_tiled_native(latent, tile_x=tile, tile_y=tile, padding_mode="circular")
                assert str(public.value) == str(native.value)
        assert value.first_stage_model.conv.padding_mode == "zeros"
    asyncio.run(run())
