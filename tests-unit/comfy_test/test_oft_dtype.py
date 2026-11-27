import pytest
import torch

from comfy.weight_adapter.boft import BOFTAdapter
from comfy.weight_adapter.oft import OFTAdapter


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_oft_calculate_weight_dtypes(dtype):
    torch.manual_seed(0)
    weight = torch.randn(32, 16, dtype=dtype)
    blocks = torch.randn(4, 8, 8, dtype=dtype) * 0.05
    adapter = OFTAdapter({"w.oft_blocks"}, (blocks, None, 0.0, None))

    out = adapter.calculate_weight(
        weight.clone(),
        "w",
        strength=1.0,
        strength_model=1.0,
        offset=None,
        function=lambda x: x,
        intermediate_dtype=dtype,
    )
    assert out.dtype == dtype
    assert not torch.allclose(out, weight)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_oft_bypass_and_train_dtypes(dtype):
    torch.manual_seed(0)
    blocks = torch.randn(4, 8, 8, dtype=dtype) * 0.05
    adapter = OFTAdapter({"w.oft_blocks"}, (blocks, None, 0.0, None))
    adapter.is_conv = False
    y = torch.randn(2, 4, 32, dtype=dtype)

    out_g = adapter.g(y)
    assert out_g.dtype == dtype
    assert not torch.allclose(out_g, y)

    diff = adapter.to_train()
    diff.is_conv = False
    assert diff.g(y).dtype == dtype
    assert diff(torch.randn(32, 16, dtype=dtype)).dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_boft_calculate_weight_and_bypass_dtypes(dtype):
    torch.manual_seed(0)
    weight = torch.randn(32, 16, dtype=dtype)
    blocks = torch.randn(2, 4, 8, 8, dtype=dtype) * 0.05
    adapter = BOFTAdapter({"w.oft_blocks"}, (blocks, None, 0.0, None))
    adapter.is_conv = False

    out_w = adapter.calculate_weight(
        weight.clone(),
        "w",
        strength=1.0,
        strength_model=1.0,
        offset=None,
        function=lambda x: x,
        intermediate_dtype=dtype,
    )
    assert out_w.dtype == dtype
    assert not torch.allclose(out_w, weight)

    y = torch.randn(2, 4, 32, dtype=dtype)
    out_g = adapter.g(y)
    assert out_g.dtype == dtype
    assert not torch.allclose(out_g, y)
