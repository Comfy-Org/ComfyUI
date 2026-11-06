import asyncio

import numpy as np
import pytest
import torch

from comfy_api.latest import sdk
from comfy_api.latest import _sdk


@pytest.mark.parametrize("shape", [(8, 9), (2, 8, 9), (1, 1, 8, 9), (1, 3, 8, 9)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64, torch.bfloat16, torch.bool])
def test_public_mask_constructor_preserves_native_shape_dtype_and_storage(shape, dtype):
    async def run():
        refs = _sdk.InProcessRefResolver()
        value = torch.ones(shape, dtype=dtype)
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            result = await sdk.MaskRef.from_value(value)
        assert type(result) is sdk.MaskRef and result.kind == "MASK"
        assert await refs.resolve(result) is value
        refs.clear()
        with pytest.raises(KeyError, match="unknown ref"):
            await refs.resolve(result)
    asyncio.run(run())


@pytest.mark.parametrize("shape", [(2, 8, 9), (2, 8, 9, 2)])
def test_public_native_image_constructor_keeps_layout_and_unclamped_values(shape):
    async def run():
        refs = _sdk.InProcessRefResolver()
        value = torch.linspace(-2, 3, int(np.prod(shape)), dtype=torch.float64).reshape(shape)
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            result = await sdk.ImageRef.from_tensor(value)
            if len(shape) == 3:
                with pytest.raises(ValueError, match="BHWC"):
                    await sdk.ImageRef.from_value(value)
        assert type(result) is sdk.ImageRef and result.kind == "IMAGE"
        assert await refs.resolve(result) is value
    asyncio.run(run())


def test_numpy_mask_uses_native_shape_and_dtype():
    async def run():
        refs = _sdk.InProcessRefResolver()
        value = np.arange(72, dtype=np.uint8).reshape(8, 9)
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            result = await sdk.MaskRef.from_value(value)
        actual = await refs.resolve(result)
        assert actual.shape == value.shape and actual.dtype == torch.uint8
        np.testing.assert_array_equal(actual.numpy(), value)
    asyncio.run(run())


@pytest.mark.parametrize("constructor,value,error", [
    (sdk.MaskRef.from_value, torch.ones(8), ValueError),
    (sdk.MaskRef.from_value, torch.ones(1, 5, 8, 9), ValueError),
    (sdk.MaskRef.from_value, torch.empty(1, 0, 9), ValueError),
    (sdk.MaskRef.from_value, torch.full((8, 9), float("nan")), ValueError),
    (sdk.ImageRef.from_tensor, torch.ones(8, 9), ValueError),
    (sdk.ImageRef.from_tensor, torch.ones(1, 8, 9, 5), ValueError),
    (sdk.ImageRef.from_tensor, torch.ones(1, 8, 9, dtype=torch.bool), TypeError),
    (sdk.ImageRef.from_tensor, torch.full((1, 8, 9), float("inf")), ValueError),
    (sdk.MaskRef.from_value, np.empty((8, 9), dtype=object), TypeError),
])
def test_public_constructors_refuse_invalid_buffers_before_ref_creation(constructor, value, error):
    async def run():
        refs = _sdk.InProcessRefResolver()
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            with pytest.raises(error):
                await constructor(value)
        assert refs._table == {}
    asyncio.run(run())


def test_public_mask_and_image_constructors_keep_existing_allocation_bounds(monkeypatch):
    monkeypatch.setenv("COMFY_SECURE_MAX_STORE_BYTES", "16")
    async def run():
        refs = _sdk.InProcessRefResolver()
        with _sdk.bind_runtime(refs, None, _sdk.InProcessOps()):
            for constructor, shape in [(sdk.MaskRef.from_value, (8, 9)),
                                       (sdk.ImageRef.from_tensor, (1, 8, 9))]:
                with pytest.raises(ValueError, match="16-byte"):
                    await constructor(torch.ones(shape))
        assert refs._table == {}
    asyncio.run(run())

