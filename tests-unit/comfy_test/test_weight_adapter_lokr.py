import torch

from comfy.weight_adapter.lokr import LoKrAdapter


def make_adapter(weights):
    return LoKrAdapter.load("layer", weights, 1.0, None)


def test_direct_matrices_apply_alpha_rank_scale_when_calculating_weight():
    w1 = torch.ones((8, 8))
    w2 = torch.ones((256, 128))
    adapter = make_adapter({"layer.lokr_w1": w1, "layer.lokr_w2": w2})
    weight = torch.zeros((2048, 1024))

    result = adapter.calculate_weight(
        weight, "layer.weight", 1.0, 1.0, None, lambda update: update
    )

    torch.testing.assert_close(result, torch.full_like(weight, 1 / 32))


def test_direct_matrices_apply_alpha_rank_scale_in_bypass():
    adapter = make_adapter(
        {"layer.lokr_w1": torch.ones((8, 8)), "layer.lokr_w2": torch.ones((256, 128))}
    )
    x = torch.ones((1, 1024))

    result = adapter.h(x, torch.zeros((1, 2048)))

    torch.testing.assert_close(result, torch.full((1, 2048), 32.0))


def test_reconstructed_matrices_keep_factor_rank_scaling():
    adapter = make_adapter(
        {
            "layer.lokr_w1_a": torch.ones((8, 32)),
            "layer.lokr_w1_b": torch.ones((32, 8)),
            "layer.lokr_w2_a": torch.ones((256, 32)),
            "layer.lokr_w2_b": torch.ones((32, 128)),
        }
    )
    weight = torch.zeros((2048, 1024))

    result = adapter.calculate_weight(
        weight, "layer.weight", 1.0, 1.0, None, lambda update: update
    )

    torch.testing.assert_close(result, torch.full_like(weight, 32.0))
