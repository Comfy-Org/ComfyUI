import asyncio

import numpy as np
import pytest

from comfy_api.latest import _sdk


@pytest.mark.parametrize("shape,dtype", [((24, 3), "float32"), ((3,), "int16"), ((), "float64"), ((0, 3), "float32"), ((2, 3), "complex64"), ((2, 3), "bool")])
def test_numeric_arrays_are_data_refs_without_image_or_sigmas_inference(shape, dtype):
    async def run():
        array = np.zeros(shape, dtype=dtype)
        refs = _sdk.InProcessRefResolver()
        result = await _sdk.wrap_inputs(refs, {"motion": {"mean": array}}, {"motion": "MTVCRAFTERMOTION"})
        ref = result["motion"]["mean"]
        assert type(ref) is _sdk.TensorRef and ref.kind == "TENSOR"
        assert await refs.resolve(ref) is array
    asyncio.run(run())


@pytest.mark.parametrize("value", [np.array([object()], dtype=object), np.array(["text"]), np.zeros(2, dtype=[("x", "f4")]), object()])
def test_non_numeric_arrays_and_host_objects_remain_opaque(value):
    assert _sdk._ref_type_for(value) == (_sdk.OpaqueRef, "OPAQUE")


def test_projected_array_bytes_are_bounded_before_ref_creation():
    value = np.lib.stride_tricks.as_strided(np.zeros(1, dtype=np.float32), shape=(134217729,), strides=(0,))
    with pytest.raises(ValueError, match="512 MiB"):
        _sdk._ref_type_for(value)
