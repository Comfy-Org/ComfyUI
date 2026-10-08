"""Vision token access preserves numeric buffers and denies object escape."""
import asyncio
from types import SimpleNamespace

import pytest
import torch

from comfy_api.latest import _sdk
import comfy.clip_vision


class _Projection:
    def __init__(self):
        self.calls = []

    def __call__(self, pixel_values, intermediate_output):
        self.calls.append((pixel_values, intermediate_output))
        tokens = pixel_values.flatten(2).transpose(1, 2)[:, :5]
        hidden = torch.stack((tokens, tokens + 1, tokens + 2), dim=1) if intermediate_output == "all" else tokens + 1
        return tokens, hidden, pixel_values.mean((2, 3)), {"projection": "kept"}


def _encoder(size=224, all_states=False):
    encoder = comfy.clip_vision.ClipVisionModel.__new__(comfy.clip_vision.ClipVisionModel)
    encoder.image_size = size
    encoder.image_mean = [0.48145466, 0.4578275, 0.40821073]
    encoder.image_std = [0.26862954, 0.26130258, 0.27577711]
    encoder.model_type = "siglip_vision_model" if all_states else "clip_vision_model"
    encoder.return_all_hidden_states = all_states
    encoder.load_device = torch.device("cpu")
    encoder.patcher = object()
    encoder.model = _Projection()
    return encoder


async def _encode(encoder, pixels):
    refs = _sdk.InProcessRefResolver()
    model = _sdk.ClipVisionRef._wrap(await refs.create("CLIP_VISION", encoder))
    pixel_ref = _sdk.TensorRef._wrap(await refs.create("TENSOR", pixels))
    with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
        size = await model.input_size()
        output = await model.encode_pixels(pixel_ref)
        tokens = await output.penultimate_hidden_states()
        return size, await refs.resolve(output), await refs.resolve(tokens)


async def _tokens(value, field):
    refs = _sdk.InProcessRefResolver()
    output = _sdk.ClipVisionOutputRef._wrap(await refs.create(
        "CLIP_VISION_OUTPUT", SimpleNamespace(**{field: value})))
    with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
        token = await getattr(output, field)()
        assert isinstance(token, _sdk.TensorRef) and token.kind == "TENSOR"
        return await refs.resolve(token)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("field", ["penultimate_hidden_states", "last_hidden_state"])
def test_tokens_keep_shape_dtype_layout_and_values(dtype, field):
    original = torch.arange(2*7*11, dtype=dtype).reshape(2,7,11).transpose(1,2)
    result = asyncio.run(_tokens(original, field))
    assert result is original
    assert result.dtype == dtype and result.shape == (2,11,7)
    assert torch.equal(result,original)


@pytest.mark.parametrize("value", [None, {"encoder": "not a tensor"},
    torch.ones(1,2), torch.ones(1,2,3,dtype=torch.int64)])
@pytest.mark.parametrize("field", ["penultimate_hidden_states", "last_hidden_state"])
def test_tokens_do_not_return_host_objects_or_wrong_layout(value, field):
    with pytest.raises(TypeError,match="floating BTD"):
        asyncio.run(_tokens(value, field))


@pytest.mark.parametrize("shape", [(0,2,3),(65,2,3),(1,16384,16384)])
@pytest.mark.parametrize("field", ["penultimate_hidden_states", "last_hidden_state"])
def test_tokens_size_denial_without_allocating_payload(shape, field):
    value = torch.empty(shape,device="meta")
    with pytest.raises(ValueError,match="bounded"):
        asyncio.run(_tokens(value, field))


def test_final_and_penultimate_tokens_remain_distinct():
    async def run():
        refs = _sdk.InProcessRefResolver()
        last = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        penultimate = last + 100
        output = _sdk.ClipVisionOutputRef._wrap(await refs.create("CLIP_VISION_OUTPUT",
            SimpleNamespace(last_hidden_state=last, penultimate_hidden_states=penultimate)))
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            assert await refs.resolve(await output.last_hidden_state()) is last
            assert await refs.resolve(await output.penultimate_hidden_states()) is penultimate
    asyncio.run(run())


@pytest.mark.parametrize("size", [224, 336])
@pytest.mark.parametrize("all_states", [False, True])
def test_normalized_pixels_preserve_canonical_encoder_outputs_and_load_policy(monkeypatch, size, all_states):
    encoder = _encoder(size, all_states)
    loads = []
    monkeypatch.setattr(comfy.clip_vision.comfy.model_management, "load_model_gpu", loads.append)
    monkeypatch.setattr(comfy.clip_vision.comfy.model_management, "intermediate_device", lambda: torch.device("cpu"))
    image = torch.linspace(0, 1, 2 * 33 * 49 * 3).reshape(2, 33, 49, 3)
    expected = encoder.encode_image(image, crop=False)
    pixels = comfy.clip_vision.clip_preprocess(image, size=size, mean=encoder.image_mean, std=encoder.image_std, crop=False)
    normalized = pixels * torch.tensor([[[[1.]], [[0.]], [[0.5]]]])
    actual_size, actual, tokens = asyncio.run(_encode(encoder, normalized.to(torch.float64)))
    assert actual_size == size and loads == [encoder.patcher, encoder.patcher]
    forwarded, mode = encoder.model.calls[-1]
    assert forwarded.dtype == torch.float32 and torch.equal(forwarded, normalized)
    assert mode == ("all" if all_states else -2)
    assert torch.equal(actual.image_embeds, normalized.mean((2, 3)))
    assert torch.equal(tokens, normalized.flatten(2).transpose(1, 2)[:, :5] + 1)
    assert actual.image_sizes == [(3, size, size)] * 2
    assert actual.mm_projected == expected.mm_projected
    if all_states:
        assert torch.equal(actual.all_hidden_states[:, -2], tokens)
    unmasked = asyncio.run(_encode(encoder, pixels))[1]
    for name in ("image_embeds", "last_hidden_state", "penultimate_hidden_states"):
        assert torch.equal(getattr(unmasked, name), getattr(expected, name))


@pytest.mark.parametrize("pixels,exception", [
    (torch.ones(1, 3, 224, 224, dtype=torch.int64), TypeError),
    (torch.ones(1, 224, 224, 3), ValueError),
    (torch.ones(1, 3, 336, 336), ValueError),
    (torch.empty(0, 3, 224, 224), ValueError),
    (torch.empty(65, 3, 224, 224, device="meta"), ValueError),
    (torch.empty(32, 3, 224, 224, dtype=torch.float64, device="meta"), ValueError),
    (torch.full((1, 3, 224, 224), float("nan")), ValueError),
    (torch.full((1, 3, 224, 224), float("inf")), ValueError),
])
def test_pixel_rejection_precedes_model_load(pixels, exception):
    encoder = _encoder()
    with pytest.raises(exception):
        asyncio.run(_encode(encoder, pixels))
    assert encoder.model.calls == []


@pytest.mark.parametrize("size,model_type", [(512, "clip_vision_model"), (224, "siglip2_vision_model")])
def test_packed_or_unadmitted_vision_layout_is_not_treated_as_bchw(size, model_type):
    encoder = _encoder(size)
    encoder.model_type = model_type
    with pytest.raises(ValueError, match="admitted"):
        asyncio.run(_encode(encoder, torch.ones(1, 3, size, size)))
    assert encoder.model.calls == []


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int64])
def test_public_tensor_constructor_keeps_dense_buffer_without_object_escape(dtype):
    async def run():
        refs = _sdk.InProcessRefResolver()
        original = torch.arange(12, dtype=dtype).reshape(3, 4).T
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            result = await _sdk.TensorRef.from_value(original)
        assert result.kind == "TENSOR" and await refs.resolve(result) is original
    asyncio.run(run())


@pytest.mark.parametrize("value,exception", [(object(), TypeError), (torch.empty(134217729, device="meta"), ValueError)])
def test_public_tensor_constructor_denies_object_and_oversized_payload(value, exception):
    async def run():
        refs = _sdk.InProcessRefResolver()
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            with pytest.raises(exception):
                await _sdk.TensorRef.from_value(value)
        assert refs._table == {}
    asyncio.run(run())
