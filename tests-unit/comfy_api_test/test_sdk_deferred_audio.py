import asyncio
import gc
import io
import os
import struct
import sys
import threading
import wave

import pytest

from comfy_api.latest import io as node_io
from comfy_api.latest._caching import allows_external_cache
from comfy_api.latest._deferred_audio import AudioWindow, DeferredAudio, DeferredAudioPool
from comfy_api.latest._sdk import (
    AssetRef, AudioDecodeError, AudioRef, ExecutionPlan, InProcessCtxProvider, InProcessOps,
    InProcessRefResolver, OpNotSupported, Ref, bind_runtime, unwrap_outputs, wrap_inputs,
)
from comfy_execution.caching import BasicCache, CacheKeySetID, RAMPressureCache


pytestmark = pytest.mark.skipif(
    not all(hasattr(os, name) for name in ("O_NOFOLLOW", "O_NONBLOCK")),
    reason="managed audio capture requires no-follow regular-file support",
)


def runtime():
    refs = InProcessRefResolver()
    context = InProcessCtxProvider().build(ExecutionPlan(
        prompt_id="deferred-audio", node_id="1", node_type="deferred-audio",
        prompt={"1": {"class_type": "deferred-audio"}},
    ))
    return refs, context, InProcessOps(allow_deferred_audio=True)


def source(tmp_path):
    path = tmp_path / "source.wav"
    encoded = io.BytesIO()
    with wave.open(encoded, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(struct.pack("<hhhh", 0, 8192, -8192, 16384))
    path.write_bytes(encoded.getvalue())
    return path


def test_real_decoder_first_stream_window_and_float32_samples(tmp_path):
    import torch

    value = DeferredAudioPool().capture(str(source(tmp_path)), AudioWindow(1 / 16000, 2 / 16000, 1024, 2), 4096)
    result = value.materialize()
    assert result["sample_rate"] == 16000
    assert torch.equal(result["waveform"], torch.tensor([[[0.25, -0.25]]]))


def test_actual_codec_failure_has_public_decode_error_without_renaming_limits(tmp_path):
    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"not an encoded audio container")
    value = DeferredAudioPool().capture(str(corrupt), AudioWindow(0, None, 1024, 2), 4096)
    with pytest.raises(AudioDecodeError):
        value.materialize()
    bounded = DeferredAudioPool().capture(str(source(tmp_path)), AudioWindow(0, None, 4, 2), 4096)
    with pytest.raises(ValueError, match="decoded byte budget"):
        bounded.materialize()


def test_actual_container_without_audio_has_decode_error(tmp_path):
    from PIL import Image

    path = tmp_path / "image.png"
    Image.new("RGB", (1, 1)).save(path)
    value = DeferredAudioPool().capture(str(path), AudioWindow(0, None, 1024, 2), 4096)
    with pytest.raises(AudioDecodeError, match="no audio stream"):
        value.materialize()


def test_existing_eager_audio_value_and_kind_are_preserved():
    async def run():
        refs, context, ops = runtime()
        value = {"waveform": "existing samples", "sample_rate": 16000}
        with bind_runtime(refs, context, ops):
            audio = await AudioRef.from_value(value)
            assert await audio.value() is value
            assert audio.kind == "AUDIO"
        refs.clear()

    asyncio.run(run())


@pytest.mark.parametrize("name,is_source_error", [
    ("InvalidDataError", True), ("EOFError", True),
    ("PermissionError", False), ("MemoryError", False),
    ("TimeoutError", False), ("NotImplementedError", False),
    ("DecoderNotFoundError", False), ("OSError", False),
    ("ArgumentError", False), ("PyAVCallbackError", False),
])
def test_actual_pyav_error_hierarchy_does_not_turn_resource_errors_into_decode_errors(monkeypatch, name, is_source_error):
    from comfy_api.latest import _audio_decode

    error_type = getattr(_audio_decode.av.error, name)
    original = error_type(1, "controlled PyAV boundary failure")

    def unavailable(*args, **kwargs):
        raise original

    monkeypatch.setattr(_audio_decode.av, "open", unavailable)
    with pytest.raises(AudioDecodeError if is_source_error else error_type) as caught:
        _audio_decode.decode_audio(b"encoded", AudioWindow(0, None, 1024, 2))
    if is_source_error:
        assert caught.value.__cause__ is original
    else:
        assert caught.value is original


def test_unused_output_survives_producer_and_rewrap_without_consumption(tmp_path, monkeypatch):
    from comfy_api.latest import _audio_decode

    calls = []
    monkeypatch.setattr(_audio_decode, "decode_audio", lambda data, window: calls.append(data) or {"waveform": "samples", "sample_rate": 16000})

    async def run():
        path = source(tmp_path)
        refs, context, ops = runtime()
        with bind_runtime(refs, context, ops):
            asset = AssetRef._wrap(await refs.create("ASSET", str(path)))
            audio = await AudioRef.from_asset(asset, max_source_bytes=4096, max_decoded_bytes=1024, timeout_seconds=2)
            output = await unwrap_outputs(refs, node_io.NodeOutput(audio))
        refs.clear()
        path.write_bytes(b"replaced source")
        assert calls == []
        consumer, context, ops = runtime()
        with bind_runtime(consumer, context, ops):
            wrapped = await wrap_inputs(consumer, {"audio": output.result[0]})
            assert isinstance(wrapped["audio"], AudioRef)
            assert calls == []
            assert (await wrapped["audio"].value())["sample_rate"] == 16000
            assert calls[0] != b"replaced source"
        consumer.clear()

    asyncio.run(run())


@pytest.mark.parametrize("read", [lambda value: value["sample_rate"], len, list])
def test_every_ordinary_mapping_access_consumes_and_caches(tmp_path, monkeypatch, read):
    from comfy_api.latest import _audio_decode

    calls = []
    monkeypatch.setattr(_audio_decode, "decode_audio", lambda data, window: calls.append(data) or {"waveform": "samples", "sample_rate": 16000})
    value = DeferredAudioPool().capture(str(source(tmp_path)), AudioWindow(0, None, 1024, 2), 4096)
    read(value)
    read(value)
    assert len(calls) == 1


def test_concurrent_waiters_share_failure_and_later_access_retries(tmp_path, monkeypatch):
    from comfy_api.latest import _audio_decode

    entered, release = threading.Event(), threading.Event()
    calls = []

    def decode(data, window):
        calls.append(data)
        entered.set()
        assert release.wait(2)
        if len(calls) == 1:
            raise ValueError("decode failed")
        return {"waveform": "samples", "sample_rate": 16000}

    monkeypatch.setattr(_audio_decode, "decode_audio", decode)
    value = DeferredAudioPool().capture(str(source(tmp_path)), AudioWindow(0, None, 1024, 2), 4096)

    async def run():
        one = asyncio.create_task(value.materialize_async())
        two = asyncio.create_task(value.materialize_async())
        assert await asyncio.to_thread(entered.wait, 2)
        release.set()
        results = await asyncio.gather(one, two, return_exceptions=True)
        assert all(isinstance(result, ValueError) for result in results)
        assert len(calls) == 1
        assert (await value.materialize_async())["sample_rate"] == 16000
        assert len(calls) == 2

    asyncio.run(run())


def test_cancelled_waiter_keeps_shared_attempt_and_live_budget(tmp_path, monkeypatch):
    from comfy_api.latest import _audio_decode

    entered, release = threading.Event(), threading.Event()

    def decode(data, window):
        entered.set()
        assert release.wait(2)
        return {"waveform": "samples", "sample_rate": 16000}

    monkeypatch.setattr(_audio_decode, "decode_audio", decode)
    pool = DeferredAudioPool(max_values=1)
    path = source(tmp_path)
    value = pool.capture(str(path), AudioWindow(0, None, 1024, 2), 4096)

    async def run():
        waiter = asyncio.create_task(value.materialize_async())
        assert await asyncio.to_thread(entered.wait, 2)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        with pytest.raises(ValueError, match="live-value"):
            pool.capture(str(path), AudioWindow(0, None, 1024, 2), 4096)
        release.set()
        assert (await value.materialize_async())["sample_rate"] == 16000

    asyncio.run(run())


def test_excess_waiter_is_refused_without_duplicate_decode_work(tmp_path, monkeypatch):
    from comfy_api.latest import _audio_decode

    release = threading.Event()
    calls = []

    def decode(data, window):
        calls.append(data)
        assert release.wait(2)
        return {"waveform": "samples", "sample_rate": 16000}

    monkeypatch.setattr(_audio_decode, "decode_audio", decode)
    value = DeferredAudioPool().capture(str(source(tmp_path)), AudioWindow(0, None, 1024, 2), 4096)

    async def run():
        tasks = [asyncio.create_task(value.materialize_async()) for _ in range(17)]
        try:
            finished, waiting = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            assert len(finished) == 1 and len(waiting) == 16
            with pytest.raises(ValueError, match="waiter limit"):
                await next(iter(finished))
        finally:
            release.set()
        results = await asyncio.gather(*waiting)
        assert all(result["sample_rate"] == 16000 for result in results)
        assert len(calls) == 1

    asyncio.run(run())


def test_decoder_import_and_executor_failure_do_not_poison_later_attempt(tmp_path, monkeypatch):
    value = DeferredAudioPool().capture(str(source(tmp_path)), AudioWindow(0, None, 1024, 2), 4096)

    async def run():
        loop = asyncio.get_running_loop()
        with monkeypatch.context() as local:
            def unavailable(*args):
                raise RuntimeError("executor unavailable")
            local.setattr(loop, "run_in_executor", unavailable)
            with pytest.raises(RuntimeError, match="executor unavailable"):
                await value.materialize_async()
        with monkeypatch.context() as local:
            local.setitem(sys.modules, "comfy_api.latest._audio_decode", None)
            with pytest.raises(ImportError):
                await value.materialize_async()
        assert (await value.materialize_async())["sample_rate"] == 16000

    asyncio.run(run())


def test_cache_inspection_does_not_consume_nested_managed_audio(tmp_path, monkeypatch):
    from comfy_api.latest import _audio_decode

    def forbidden(*args):
        raise AssertionError("cache inspection decoded audio")

    monkeypatch.setattr(_audio_decode, "decode_audio", forbidden)
    value = DeferredAudioPool().capture(str(source(tmp_path)), AudioWindow(0, None, 1024, 2), 4096)
    assert value._comfy_cache_tensors() == ()
    assert value._comfy_cache_extra_bytes() > 0
    assert not allows_external_cache([{"audio": value}])
    cache = BasicCache(CacheKeySetID)
    entry = type("Entry", (), {"outputs": [[value]], "ui": {}})()
    assert not cache._is_external_cacheable_value(entry)
    from comfy_execution import caching

    monkeypatch.setattr(caching, "virtual_memory_available", lambda: 0)
    pressure = RAMPressureCache(CacheKeySetID)
    pressure.cache["audio"] = entry
    pressure.used_generation["audio"] = pressure.generation - 1
    pressure.timestamps["audio"] = 0
    assert pressure.ram_release(1) >= value._comfy_cache_extra_bytes()
    assert "audio" not in pressure.cache


def test_old_provider_opt_in_and_authoritative_asset_kind(tmp_path):
    async def run():
        refs, context, ops = runtime()
        with bind_runtime(refs, context, InProcessOps()):
            with pytest.raises(OpNotSupported):
                await AudioRef.from_asset(Ref("ASSET", "unknown"), max_source_bytes=4096, max_decoded_bytes=1024, timeout_seconds=2)
        with bind_runtime(refs, context, ops):
            wrong = await refs.create("VALUE", str(source(tmp_path)))
            with pytest.raises(TypeError):
                await AudioRef.from_asset(Ref("ASSET", wrong.id), max_source_bytes=4096, max_decoded_bytes=1024, timeout_seconds=2)

    asyncio.run(run())


def test_pool_releases_weak_slot_after_last_output_owner(tmp_path):
    pool = DeferredAudioPool(max_values=1)
    path = source(tmp_path)
    value = pool.capture(str(path), AudioWindow(0, None, 1024, 2), 4096)
    with pytest.raises(ValueError, match="live-value"):
        pool.capture(str(path), AudioWindow(0, None, 1024, 2), 4096)
    del value
    gc.collect()
    assert type(pool.capture(str(path), AudioWindow(0, None, 1024, 2), 4096)) is DeferredAudio
