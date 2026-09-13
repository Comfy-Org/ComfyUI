import math

import pytest
import torch

from comfy.cli_args import args

args.cpu = True

import comfy.ldm.modules.attention
import comfy.model_base
import comfy.sampler_helpers
from comfy.model_patcher import ModelPatcher


class _StubModel:
    """Minimal stand-in for BaseModel: just the attributes memory_required reads."""
    memory_usage_factor_conds = ()
    memory_usage_shape_process = {}
    memory_usage_factor = 2.0

    def get_dtype_inference(self):
        return torch.bfloat16

    memory_required = comfy.model_base.BaseModel.memory_required

    def extra_conds_shapes(self, **kwargs):
        return {}


class _StubYuE2(comfy.model_base.YuE2):
    memory_usage_factor_conds = ()
    memory_usage_shape_process = {}
    memory_usage_factor = 2.0

    def __init__(self):
        pass

    def get_dtype_inference(self):
        return torch.bfloat16


class _LegacyMemoryModel:
    def memory_required(self, input_shape, cond_shapes={}):
        return input_shape[0] + len(cond_shapes)

    def extra_conds_shapes(self, **kwargs):
        return {}


INPUT_SHAPE = (1, 16, 1, 180, 320)
AREA = INPUT_SHAPE[0] * math.prod(INPUT_SHAPE[2:])
DTYPE_SIZE = 2  # bf16
EFFICIENT = AREA * DTYPE_SIZE * 0.01 * _StubModel.memory_usage_factor * (1024 * 1024)
CONSERVATIVE = AREA * 0.15 * _StubModel.memory_usage_factor * (1024 * 1024)


BACKEND_CASES = (
    ("sage", "sage_attention_enabled", "attention_sage", False),
    ("flash", "flash_attention_enabled", "attention_flash", False),
    ("xformers", "xformers_enabled", "attention_xformers", True),
    ("pytorch", "pytorch_attention_enabled", "attention_pytorch", False),
    ("comfy_kitchen_int8", "comfy_kitchen_attention_enabled", "attention_comfy_kitchen_int8", False),
)

SELECTORS = tuple(case[1] for case in BACKEND_CASES)
PYTORCH_FALLBACK_CASES = tuple(case for case in BACKEND_CASES if case[0] != "xformers")


def _select_backend(monkeypatch, enabled_selector=None, flash_attention=False, use_split=False):
    attention = comfy.ldm.modules.attention
    for selector in SELECTORS:
        monkeypatch.setattr(comfy.model_management, selector, lambda: False)
    if enabled_selector is not None:
        monkeypatch.setattr(comfy.model_management, enabled_selector, lambda: True)
    monkeypatch.setattr(comfy.model_management, "pytorch_attention_flash_attention", lambda: flash_attention)
    monkeypatch.setattr(attention, "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE", True)
    monkeypatch.setattr(attention.args, "use_split_cross_attention", use_split)
    return attention._select_optimized_attention()


def _estimate(profile):
    return comfy.model_base.BaseModel.memory_required(
        _StubModel(), INPUT_SHAPE, memory_efficient_attention=profile
    )


@pytest.mark.parametrize("backend, selector, function_name, efficient", BACKEND_CASES)
def test_selected_backend_uses_its_memory_profile(monkeypatch, backend, selector, function_name, efficient):
    selected, profile = _select_backend(monkeypatch, selector)

    assert selected is getattr(comfy.ldm.modules.attention, function_name)
    assert profile is efficient
    assert _estimate(profile) == (EFFICIENT if efficient else CONSERVATIVE)


@pytest.mark.parametrize("backend, selector, function_name, efficient", PYTORCH_FALLBACK_CASES)
def test_pytorch_flash_fallback_makes_selected_backend_memory_efficient(monkeypatch, backend, selector, function_name, efficient):
    selected, profile = _select_backend(monkeypatch, selector, flash_attention=True)

    assert selected is getattr(comfy.ldm.modules.attention, function_name)
    assert profile is True
    assert _estimate(profile) == EFFICIENT


def test_split_attention_uses_conservative_memory_profile(monkeypatch):
    selected, profile = _select_backend(monkeypatch, use_split=True)

    assert selected is comfy.ldm.modules.attention.attention_split
    assert profile is False
    assert _estimate(profile) == CONSERVATIVE


def test_sub_quadratic_attention_uses_conservative_memory_profile(monkeypatch):
    selected, profile = _select_backend(monkeypatch)

    assert selected is comfy.ldm.modules.attention.attention_sub_quad
    assert profile is False
    assert _estimate(profile) == CONSERVATIVE


def test_yue2_memory_required_uses_explicit_attention_profile(monkeypatch):
    monkeypatch.setattr(comfy.ldm.modules.attention, "optimized_attention_memory_efficient", True)

    memory = comfy.model_base.YuE2.memory_required(
        _StubYuE2(),
        INPUT_SHAPE,
        {"c_crossattn": [(1, 2, 3)]},
        memory_efficient_attention=False,
    )

    assert memory == CONSERVATIVE + 12


def test_model_attention_override_uses_conservative_memory_profile(monkeypatch):
    monkeypatch.setattr(comfy.ldm.modules.attention, "optimized_attention_memory_efficient", True)
    model = ModelPatcher(_StubModel(), torch.device("cpu"), torch.device("cpu"))

    model.set_model_optimized_attention(lambda *args, **kwargs: None)
    memory_required, minimum_memory_required = comfy.sampler_helpers.estimate_memory(model, INPUT_SHAPE, {})

    assert memory_required == CONSERVATIVE * 2
    assert minimum_memory_required == CONSERVATIVE


def test_model_attention_override_uses_efficient_memory_profile(monkeypatch):
    monkeypatch.setattr(comfy.ldm.modules.attention, "optimized_attention_memory_efficient", False)
    model = ModelPatcher(_StubModel(), torch.device("cpu"), torch.device("cpu"))

    model.set_model_optimized_attention(lambda *args, **kwargs: None, memory_efficient=True)
    memory_required, minimum_memory_required = comfy.sampler_helpers.estimate_memory(model, INPUT_SHAPE, {})

    assert memory_required == EFFICIENT * 2
    assert minimum_memory_required == EFFICIENT


def test_model_attention_override_defaults_to_conservative_memory_profile():
    model = ModelPatcher(_StubModel(), torch.device("cpu"), torch.device("cpu"))

    model.set_model_optimized_attention(lambda *args, **kwargs: None)

    assert model.model_options["optimized_attention_memory_efficient"] is False


def test_model_patcher_memory_required_omits_absent_attention_profile():
    model = ModelPatcher(_LegacyMemoryModel(), torch.device("cpu"), torch.device("cpu"))

    assert "optimized_attention_memory_efficient" not in model.model_options
    assert model.memory_required(INPUT_SHAPE) == 1


def test_estimate_memory_omits_absent_attention_profile():
    model = ModelPatcher(_LegacyMemoryModel(), torch.device("cpu"), torch.device("cpu"))

    assert "optimized_attention_memory_efficient" not in model.model_options
    assert comfy.sampler_helpers.estimate_memory(model, INPUT_SHAPE, {}) == (2, 1)
