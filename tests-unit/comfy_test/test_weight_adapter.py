"""Weight-adapter regression tests."""

from __future__ import annotations

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy.weight_adapter.base import weight_decompose  # noqa: E402


@pytest.mark.parametrize(
    "weight_shape",
    [
        pytest.param((4, 6), id="linear"),
        pytest.param((4, 3, 2, 2), id="conv2d"),
    ],
)
def test_weight_decompose_output_axis_uses_adapted_weight_norm(weight_shape):
    generator = torch.Generator(device="cpu").manual_seed(42)
    weight = torch.randn(weight_shape, generator=generator, dtype=torch.float32)
    lora_diff = torch.randn(weight_shape, generator=generator, dtype=torch.float32)

    alpha = 0.625
    strength = 1.0
    adapted_weight = weight + alpha * lora_diff

    output_axis_shape = (weight_shape[0], *[1] * (len(weight_shape) - 1))
    adapted_norm = (
        adapted_weight.reshape(weight_shape[0], -1)
        .norm(dim=1, keepdim=True)
        .reshape(output_axis_shape)
    )
    target_scale = torch.linspace(
        0.75,
        1.25,
        steps=weight_shape[0],
        dtype=weight.dtype,
    ).reshape(output_axis_shape)
    dora_scale = adapted_norm * target_scale

    actual = weight_decompose(
        dora_scale=dora_scale,
        weight=weight.clone(),
        lora_diff=lora_diff.clone(),
        alpha=alpha,
        strength=strength,
        intermediate_dtype=torch.float32,
        function=lambda tensor: tensor,
    )

    expected = adapted_weight * target_scale
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
