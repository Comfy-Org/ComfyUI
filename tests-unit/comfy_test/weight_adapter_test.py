import torch

from comfy.weight_adapter import BOFTAdapter, OFTAdapter


def apply_adapter(adapter_cls, blocks, alpha):
    adapter = adapter_cls.load("x", {"x.oft_blocks": blocks}, alpha, None)
    weight = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    return adapter.calculate_weight(weight, "x", 1.0, 1.0, None, lambda w: w)


# 4 blocks of 2x2 gives out_dim 8, so lycoris clamps at alpha * 8 and ||Q|| is 0.283.
def test_oft_constraint_is_scaled_by_out_dim():
    blocks = torch.zeros(4, 2, 2)
    blocks[:, 0, 1] = 0.1

    unclamped = apply_adapter(OFTAdapter, blocks, 0.0)
    assert torch.allclose(apply_adapter(OFTAdapter, blocks, 0.1), unclamped)
    assert not torch.allclose(apply_adapter(OFTAdapter, blocks, 0.01), unclamped)


# BOFT(3, 4 blocks of 2x2) also gives out_dim 8, with ||Q|| of 0.490.
def test_boft_constraint_is_scaled_by_out_dim():
    blocks = torch.zeros(3, 4, 2, 2)
    blocks[:, :, 0, 1] = 0.1

    unclamped = apply_adapter(BOFTAdapter, blocks, 0.0)
    assert torch.allclose(apply_adapter(BOFTAdapter, blocks, 0.1), unclamped)
    assert not torch.allclose(apply_adapter(BOFTAdapter, blocks, 0.01), unclamped)
