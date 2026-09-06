import math

import torch

from comfy.cli_args import args

args.cpu = True

import comfy.ldm.modules.attention
import comfy.model_base


class _StubModel:
    """Minimal stand-in for BaseModel: just the attributes memory_required reads."""
    memory_usage_factor_conds = ()
    memory_usage_shape_process = {}
    memory_usage_factor = 2.0

    def get_dtype_inference(self):
        return torch.bfloat16


INPUT_SHAPE = (1, 16, 1, 180, 320)
AREA = INPUT_SHAPE[0] * math.prod(INPUT_SHAPE[2:])
DTYPE_SIZE = 2  # bf16
EFFICIENT = AREA * DTYPE_SIZE * 0.01 * _StubModel.memory_usage_factor * (1024 * 1024)
CONSERVATIVE = AREA * 0.15 * _StubModel.memory_usage_factor * (1024 * 1024)


def _estimate(monkeypatch, memory_efficient):
    monkeypatch.setattr(
        comfy.ldm.modules.attention, "optimized_attention_memory_efficient", memory_efficient
    )
    return comfy.model_base.BaseModel.memory_required(_StubModel(), INPUT_SHAPE)


def test_efficient_backend_uses_efficient_estimate(monkeypatch):
    assert _estimate(monkeypatch, True) == EFFICIENT


def test_quadratic_backend_uses_conservative_estimate(monkeypatch):
    assert _estimate(monkeypatch, False) == CONSERVATIVE
