"""Native ref, token, pixel and selected model-policy checks; no trained weights."""
import asyncio

import numpy as np
import pytest
import torch

import comfy.clip_vision
import comfy.model_management as mm
from comfy_api.latest import _sdk


CONFIG = {"model_type": "clip_vision_model", "num_hidden_layers": 2,
          "hidden_size": 8, "num_attention_heads": 2, "intermediate_size": 16,
          "hidden_act": "quick_gelu", "num_channels": 3, "patch_size": 2,
          "image_size": 4, "projection_dim": 4}


def native_output(batch=3):
    output = comfy.clip_vision.Output()
    output.last_hidden_state = torch.arange(batch * 5 * 4, dtype=torch.float64).reshape(batch, 5, 4)
    output.penultimate_hidden_states = -output.last_hidden_state - 3
    output.image_embeds = torch.full((batch, 4), 99.)
    return output


@pytest.mark.parametrize("field", ["last_hidden_state", "penultimate_hidden_states"])
@pytest.mark.parametrize("start,count", [(0, None), (1, None), (1, 1), (2, 1)])
def test_native_tokens_keep_values_dtype_order_and_parent_release(field, start, count):
    async def run():
        refs = _sdk.InProcessRefResolver()
        native = native_output()
        source = _sdk.ClipVisionOutputRef._wrap(await refs.create("CLIP_VISION_OUTPUT", native))
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            result = await getattr(source, field)(batch_start=start, batch_count=count)
            await refs.release(source)
            value = await result.raw()
            stop = 3 if count is None else start + count
            expected = getattr(native, field)[start:stop]
            assert value.dtype == expected.dtype and value.device == expected.device
            assert value.shape == expected.shape and torch.equal(value, expected)
            assert value.data_ptr() == expected.data_ptr()
            refs.clear()
            with pytest.raises(KeyError):
                await result.raw()
    asyncio.run(run())


@pytest.mark.parametrize("start,count,error", [
    (True, 1, TypeError), (-1, 1, ValueError), (3, 1, ValueError),
    (0.5, 1, TypeError), (0, True, TypeError), (0, 0, ValueError),
    (0, -1, ValueError), (2, 2, ValueError), (0, 65, ValueError),
    (0, 1.5, TypeError),
])
def test_token_selection_refuses_invalid_bounds(start, count, error):
    async def run():
        refs = _sdk.InProcessRefResolver()
        source = _sdk.ClipVisionOutputRef._wrap(await refs.create("CLIP_VISION_OUTPUT", native_output()))
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            with pytest.raises(error):
                await source.last_hidden_state(batch_start=start, batch_count=count)
    asyncio.run(run())


def test_remaining_rows_are_bounded_and_projection_honors_byte_limit(monkeypatch):
    async def run():
        refs = _sdk.InProcessRefResolver()
        source = _sdk.ClipVisionOutputRef._wrap(await refs.create("CLIP_VISION_OUTPUT", native_output(65)))
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            with pytest.raises(ValueError, match="1..64"):
                await source.last_hidden_state()
            assert (await (await source.last_hidden_state(batch_start=1)).raw()).shape[0] == 64
            monkeypatch.setenv("COMFY_SECURE_MAX_STORE_BYTES", "8")
            with pytest.raises(ValueError, match="buffer limit"):
                await source.last_hidden_state(batch_count=1)
    asyncio.run(run())


@pytest.mark.parametrize("case", ["rank", "missing", "object", "wrong-kind", "stale", "meta"])
def test_token_projection_refuses_invalid_native_inputs(case):
    async def run():
        refs = _sdk.InProcessRefResolver()
        output = native_output()
        if case == "rank":
            output.last_hidden_state = torch.zeros(3, 4)
        if case == "missing":
            del output.last_hidden_state
        if case == "meta":
            output.last_hidden_state = torch.empty(3, 5, 4, device="meta")
        if case == "object":
            output = object()
        ref = await refs.create("CLIP_VISION_OUTPUT", output)
        source = _sdk.TensorRef._wrap(ref) if case == "wrong-kind" else _sdk.ClipVisionOutputRef._wrap(ref)
        if case == "stale":
            await refs.release(ref)
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            with pytest.raises((TypeError, ValueError, KeyError)):
                await _sdk.current_runtime().ops.apply("clip_vision_output.last_hidden_state", source, {})
    asyncio.run(run())


@pytest.mark.parametrize("array", [False, True])
@pytest.mark.parametrize("channels", [1, 2, 3, 4])
def test_public_image_constructor_preserves_pixels_dtype_and_bhwc(array, channels):
    pixels = torch.linspace(-2., 2., 2 * 4 * 5 * channels, dtype=torch.float64).reshape(2, 4, 5, channels)
    original = pixels.numpy() if array else pixels
    async def run():
        refs = _sdk.InProcessRefResolver()
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            result = await _sdk.ImageRef.from_value(original)
            assert type(result) is _sdk.ImageRef and result.kind == "IMAGE"
            actual = await result.raw()
            assert actual.dtype == pixels.dtype and actual.shape == pixels.shape
            assert torch.equal(actual, pixels) and actual.data_ptr() == pixels.data_ptr()
    asyncio.run(run())


@pytest.mark.parametrize("case", ["rank", "empty", "channels", "bool", "complex", "nan", "inf", "object", "sparse", "meta"])
def test_image_constructor_refuses_invalid_pixels(case):
    pixels = {
        "rank": torch.zeros(2, 3, 4), "empty": torch.zeros(0, 2, 3, 3),
        "channels": torch.zeros(1, 2, 3, 5), "bool": torch.zeros(1, 2, 3, 3, dtype=torch.bool),
        "complex": torch.zeros(1, 2, 3, 3, dtype=torch.complex64),
        "nan": torch.full((1, 2, 3, 3), float("nan")),
        "inf": np.full((1, 2, 3, 3), float("inf")), "object": {"model": object()},
        "sparse": torch.zeros(1, 2, 3, 3).to_sparse(),
        "meta": torch.empty(1, 2, 3, 3, device="meta"),
    }[case]
    async def run():
        refs = _sdk.InProcessRefResolver()
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            with pytest.raises((TypeError, ValueError)):
                await _sdk.ImageRef.from_value(pixels)
    asyncio.run(run())


def test_image_constructor_checks_existing_byte_ceiling_and_fixed_kind(monkeypatch):
    async def run():
        refs = _sdk.InProcessRefResolver()
        class OtherKind(_sdk.ImageRef):
            KIND = "MODEL"
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            assert (await OtherKind.from_value(torch.zeros(1, 2, 3, 3))).kind == "IMAGE"
            monkeypatch.setenv("COMFY_SECURE_MAX_STORE_BYTES", "8")
            with pytest.raises(ValueError, match="8-byte"):
                await _sdk.ImageRef.from_value(torch.zeros(1, 2, 3, 3))
    asyncio.run(run())


@pytest.mark.parametrize("enumerated_devices", ["native", "empty"])
def test_offload_uses_native_clone_family_and_keeps_unrelated_model_loaded(
    monkeypatch, enumerated_devices,
):
    monkeypatch.setattr(mm, "text_encoder_device", lambda: torch.device("cpu"))
    if enumerated_devices == "empty":
        monkeypatch.setattr(mm, "get_all_torch_devices", lambda: [])
    vision = comfy.clip_vision.ClipVisionModel(CONFIG)
    other = comfy.clip_vision.ClipVisionModel(CONFIG)
    clone = vision.patcher.clone()
    before = list(mm.current_loaded_models)
    try:
        mm.load_models_gpu([clone, other.patcher], force_full_load=True)
        assert any(model.model is clone for model in mm.current_loaded_models)
        assert any(model.model is other.patcher for model in mm.current_loaded_models)
        async def run():
            refs = _sdk.InProcessRefResolver()
            selected = _sdk.ClipVisionRef._wrap(await refs.create("CLIP_VISION", vision))
            with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
                assert await selected.offload() is None
                assert not any(model.model is clone for model in mm.current_loaded_models)
                assert any(model.model is other.patcher for model in mm.current_loaded_models)
                mm.load_model_gpu(vision.patcher)
                assert any(model.model is vision.patcher for model in mm.current_loaded_models)
                await selected.offload()
                await refs.release(selected)
                with pytest.raises(KeyError):
                    await selected.offload()
        asyncio.run(run())
    finally:
        mm.unload_model_and_clones(vision.patcher, unload_additional_models=False, all_devices=True)
        mm.unload_model_and_clones(other.patcher, unload_additional_models=False, all_devices=True)
        assert mm.current_loaded_models == before


def test_offload_refuses_arbitrary_model_values_and_ref_kinds():
    async def run():
        refs = _sdk.InProcessRefResolver()
        selected = _sdk.ClipVisionRef._wrap(await refs.create("CLIP_VISION", torch.nn.Linear(2, 2)))
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            with pytest.raises(TypeError, match="canonical"):
                await selected.offload()
            with pytest.raises(TypeError, match="CLIP_VISION ref"):
                await _sdk.current_runtime().ops.apply("clip_vision.offload", _sdk.ModelRef._wrap(selected), {})
    asyncio.run(run())
