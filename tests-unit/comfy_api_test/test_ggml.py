"""Small hand-auditable GGML blocks and ordinary Core operation behavior."""
import pytest
import torch

from comfy_api.latest._gguf_tensor import GGMLType, GGUFTensor
from comfy_api.latest._ggml import dequantize, make_ops


def block(qtype):
    half = torch.tensor([0.5], dtype=torch.float16).view(torch.uint8)
    if qtype == GGMLType.Q8_0:
        return torch.cat((half, torch.arange(32, dtype=torch.uint8))), [n * 0.5 for n in range(32)]
    if qtype in {GGMLType.Q4_K, GGMLType.Q5_K}:
        # Eight scale1/min0 groups. Packed low nibbles1, high nibbles2.
        header = torch.cat((half, torch.zeros(2, dtype=torch.uint8),
                            torch.tensor([1] * 4 + [0] * 4 + [1] * 4, dtype=torch.uint8)))
        if qtype == GGMLType.Q5_K:
            header = torch.cat((header, torch.full((32,), 0x55, dtype=torch.uint8)))
        raw = torch.cat((header, torch.full((128,), 0x21, dtype=torch.uint8)))
        pair = (8.5, 1.) if qtype == GGMLType.Q5_K else (0.5, 1.)
        return raw, [value for _ in range(4) for value in ([pair[0]] * 32 + [pair[1]] * 32)]
    raw = torch.cat((torch.full((128,), 0x21, dtype=torch.uint8),
                     torch.full((64,), 0xE4, dtype=torch.uint8),
                     torch.ones(16, dtype=torch.uint8), half))
    return raw, [value for _ in range(2) for group in (-15.5, -7.5, 1., 9.) for value in [group] * 32]


@pytest.mark.parametrize('qtype', [GGMLType.Q8_0, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K])
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32, torch.bfloat16])
def test_known_quantized_block(qtype, dtype):
    raw, values = block(qtype)
    tensor = GGUFTensor(raw.unsqueeze(0), tensor_type=qtype, tensor_shape=(1, len(values)))
    actual = dequantize(tensor, dtype)
    assert torch.equal(actual, torch.tensor([values], dtype=dtype))
    assert torch.equal(tensor.as_subclass(torch.Tensor).flatten(), raw)


@pytest.mark.parametrize('qtype,dtype', [(GGMLType.F16, torch.float16), (GGMLType.F32, torch.float32)])
def test_plain_weights_retain_storage(qtype, dtype):
    raw = torch.arange(8, dtype=dtype)
    tensor = GGUFTensor(raw, tensor_type=qtype, tensor_shape=(8,))
    actual = dequantize(tensor, dtype)
    assert type(actual) is torch.Tensor and actual.data_ptr() == raw.data_ptr()
    assert torch.equal(actual, raw)


def test_bfloat_bits():
    bits = torch.tensor([0x3F80, 0xC000, 0x0000], dtype=torch.uint16)
    tensor = GGUFTensor(bits.view(torch.uint8), tensor_type=GGMLType.BF16, tensor_shape=(3,))
    assert torch.equal(dequantize(tensor, torch.float32), torch.tensor([1., -2., 0.]))


def test_unknown_forward_type_refuses():
    tensor = GGUFTensor(torch.zeros(32, dtype=torch.uint8), tensor_type=10, tensor_shape=(32,))
    with pytest.raises(NotImplementedError, match='tensor type 10'):
        dequantize(tensor)


def test_precision_selection_is_per_model_and_linear_only():
    first = make_ops(torch.float32, torch.float16)
    second = make_ops()
    assert first.Linear.dequant_dtype is torch.float32
    assert first.Linear.patch_dtype is torch.float16
    assert second.Linear.dequant_dtype is second.Linear.patch_dtype is None
    for name in ('Conv2d', 'Embedding', 'LayerNorm', 'GroupNorm', 'RMSNorm'):
        assert getattr(first, name).dequant_dtype is None


def test_linear_uses_quantized_checkpoint_until_each_forward():
    raw, values = block(GGMLType.Q8_0)
    model = make_ops().Linear(32, 2, bias=False)
    packed = raw.expand(2, -1).clone()
    state = GGUFTensor(packed, tensor_type=GGMLType.Q8_0, tensor_shape=(2, 32))
    model.load_state_dict({'weight': state})
    expected = torch.tensor([values, values])
    input = torch.linspace(-1, 1, 32).unsqueeze(0)
    assert torch.equal(model(input), torch.nn.functional.linear(input, expected))
    assert isinstance(model.weight, GGUFTensor) and model.weight.dtype is torch.uint8
    assert torch.equal(model.weight.as_subclass(torch.Tensor), packed)
