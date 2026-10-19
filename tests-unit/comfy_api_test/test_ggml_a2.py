"""Auditable zero-index blocks for the selected native compatibility profile."""
import pytest
import torch

from comfy_api.latest import _ggml
from comfy_api.latest._ggml_a2 import TYPES
from comfy_api.latest._gguf_tensor import GGMLType, GGUFTensor, _LAYOUTS


@pytest.mark.parametrize('qtype', sorted(TYPES))
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32, torch.bfloat16])
def test_zero_index_codebook_vectors(qtype, dtype):
    count, size, _ = _LAYOUTS[qtype]
    raw = torch.zeros((1, size), dtype=torch.uint8)
    one = torch.tensor([1.], dtype=torch.float16).view(torch.uint8)
    expected = 1.0
    if qtype in {GGMLType.TQ1_0, GGMLType.TQ2_0}:
        raw[:, -2:] = one
        # Native uint8 subtraction wraps zero to255. This deliberately checks
        # compatibility, and must not be described as signed ternary spec math.
        expected = 255.0
    elif qtype == GGMLType.IQ1_M:
        raw[:, 48:] = torch.tensor([0, 0, 0xC000, 0x3000], dtype=torch.uint16).view(torch.uint8)
        expected = -0.875
    elif qtype in {GGMLType.MXFP4, GGMLType.NVFP4}:
        expected = 0.0
    else:
        raw[:, :2] = one
        if qtype == GGMLType.IQ1_S:
            expected = -0.875
        if qtype == GGMLType.IQ4_NL:
            expected = -127.0
        if qtype == GGMLType.IQ4_XS:
            expected = 4064.0
    tensor = GGUFTensor(raw, tensor_type=qtype, tensor_shape=(1, count))
    actual = _ggml.dequantize(tensor, dtype)
    assert torch.equal(actual, torch.full((1, count), expected, dtype=dtype))
    _ggml.require_supported({'weight': tensor})


def test_iq_lookups_are_owned_by_each_call():
    from comfy_api.latest import _ggml_a2
    first = _ggml_a2._grid('iq2_xxs', torch.device('cpu'))
    second = _ggml_a2._grid('iq2_xxs', torch.device('cpu'))
    assert first.data_ptr() != second.data_ptr()
    assert torch.equal(first, second)
    assert not any(isinstance(value, torch.Tensor) for value in vars(_ggml_a2).values())
