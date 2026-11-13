import pytest
import torch

import comfy.model_management
import comfy.ops
from comfy.ldm.flux.layers import DoubleStreamBlock, EmbedND, ModulationOut, SingleStreamBlock
from comfy_extras.nodes_train import patch


@pytest.mark.parametrize("block_type", [DoubleStreamBlock, SingleStreamBlock])
def test_checkpointed_flux_block_gradient_matches_finite_difference(block_type, monkeypatch):
    monkeypatch.setattr(comfy.model_management, "in_training", True)
    generator = torch.Generator().manual_seed(1)
    if block_type is DoubleStreamBlock:
        block = block_type(32, 4, mlp_ratio=2.0, modulation=False, operations=comfy.ops.manual_cast)
        inputs = [torch.randn(1, 4, 32, generator=generator, dtype=torch.float64),
                  torch.randn(1, 2, 32, generator=generator, dtype=torch.float64)]
    else:
        block = block_type(32, 4, mlp_ratio=2.0, modulation=False, operations=comfy.ops.manual_cast)
        inputs = [torch.randn(1, 6, 32, generator=generator, dtype=torch.float64)]
    block = block.to(torch.float64)
    with torch.no_grad():
        for param in block.parameters():
            param.copy_(torch.randn(param.shape, generator=generator, dtype=torch.float64) * 0.05)

    ids = torch.zeros(1, 6, 3)
    pe = EmbedND(dim=8, theta=10000, axes_dim=[2, 2, 4])(ids).to(torch.float64)

    def modulation():
        return ModulationOut(
            shift=torch.randn(1, 1, 32, generator=generator, dtype=torch.float64) * 0.1,
            scale=torch.randn(1, 1, 32, generator=generator, dtype=torch.float64) * 0.1,
            gate=torch.randn(1, 1, 32, generator=generator, dtype=torch.float64) * 0.5 + 1,
        )

    if block_type is DoubleStreamBlock:
        vec = ((modulation(), modulation()), (modulation(), modulation()))

        def loss():
            img, txt = block(inputs[0] * 1.0, inputs[1] * 1.0, vec, pe)
            return img.square().sum() + txt.square().sum()
    else:
        vec = modulation()

        def loss():
            return block(inputs[0] * 1.0, vec, pe).square().sum()

    tensors = list(block.parameters()) + inputs
    directions = [torch.randn(t.shape, generator=generator, dtype=torch.float64) for t in tensors]
    epsilon = 1e-6
    with torch.no_grad():
        for tensor, direction in zip(tensors, directions):
            tensor.add_(epsilon * direction)
        positive = loss().item()
        for tensor, direction in zip(tensors, directions):
            tensor.add_(-2 * epsilon * direction)
        negative = loss().item()
        for tensor, direction in zip(tensors, directions):
            tensor.add_(epsilon * direction)

    patch(block)
    for tensor in inputs:
        tensor.requires_grad_(True)
    loss().backward()
    derivative = sum((tensor.grad * direction).sum().item() for tensor, direction in zip(tensors, directions))
    assert derivative == pytest.approx((positive - negative) / (2 * epsilon), rel=1e-5, abs=1e-5)
