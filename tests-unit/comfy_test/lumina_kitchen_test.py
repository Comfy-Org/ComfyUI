"""Lumina Comfy Kitchen integration regression tests."""

from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args import args

original_cpu = args.cpu
if not torch.cuda.is_available():
    args.cpu = True

import comfy.ldm.lumina.model as lumina_model  # noqa: E402
import comfy.ldm.modules.attention as attention  # noqa: E402
import comfy.model_base as model_base  # noqa: E402

args.cpu = original_cpu


@pytest.mark.parametrize("length", [160, 4096, 4256])
def test_kitchen_bf16_routes_every_supported_length(monkeypatch, length):
    """Supported BF16 calls of every model length must use Kitchen HIP."""
    calls = []
    kitchen = SimpleNamespace(
        hip_attention_is_supported=lambda q, k, v: True,
        hip_attention=lambda q, k, v, scale: calls.append((q.shape, scale)) or q,
    )
    monkeypatch.setattr(attention.comfy.quant_ops, "ck", kitchen)
    monkeypatch.setattr(attention.model_management, "is_amd", lambda: True)

    q = torch.empty((1, 1, length, 128), dtype=torch.bfloat16)
    out = attention.attention_kitchen_bf16(
        q, q, q, 1, skip_reshape=True, skip_output_reshape=True,
    )

    assert out is q
    assert calls == [(q.shape, None)]


def test_kitchen_int8_respects_explicit_low_precision_opt_out(monkeypatch):
    """Explicit low-precision opt-out must select Kitchen BF16, not INT8."""
    calls = []
    kitchen = SimpleNamespace(
        hip_attention_is_supported=lambda q, k, v: True,
        hip_int8_attention_is_supported=lambda q, k, v: True,
        hip_attention=lambda q, k, v, scale: calls.append("bf16") or q,
        hip_int8_attention=lambda q, k, v, scale: calls.append("int8") or q,
    )
    monkeypatch.setattr(attention.comfy.quant_ops, "ck", kitchen)
    monkeypatch.setattr(attention.model_management, "is_amd", lambda: True)

    q = torch.empty((1, 1, 1024, 128), dtype=torch.bfloat16)
    assert attention.attention_kitchen_int8(
        q, q, q, 1, skip_reshape=True, skip_output_reshape=True,
    ) is q
    assert attention.attention_kitchen_int8(
        q, q, q, 1, skip_reshape=True, skip_output_reshape=True,
        low_precision_attention=False,
    ) is q

    assert calls == ["int8", "bf16"]


def test_unsupported_kitchen_attention_uses_pytorch(monkeypatch):
    """Unsupported Kitchen calls must retain the PyTorch fallback."""
    fallback = object()
    kitchen = SimpleNamespace(
        hip_attention_is_supported=lambda q, k, v: True,
        hip_attention=lambda q, k, v, scale: pytest.fail("HIP kernel called"),
    )
    monkeypatch.setattr(attention.comfy.quant_ops, "ck", kitchen)
    monkeypatch.setattr(attention.model_management, "is_amd", lambda: True)
    monkeypatch.setattr(attention, "attention_pytorch", lambda *args, **kwargs: fallback)

    q = torch.empty((1, 1, 160, 128), dtype=torch.bfloat16)
    mask = torch.ones((1, 1, 160, 160), dtype=torch.bool)

    assert attention.attention_kitchen_bf16(
        q, q, q, 1, mask=mask, skip_reshape=True, skip_output_reshape=True,
    ) is fallback


def test_missing_layout_fusions_fall_back(monkeypatch):
    """Missing Kitchen layout methods must retain the unfused paths."""
    monkeypatch.setattr(lumina_model, "_FUSED_RMS_MODULATED", None)
    monkeypatch.setattr(lumina_model, "_FUSED_SWIGLU_FFN", None)

    linear = SimpleNamespace(weight=None)
    ffn_layer = SimpleNamespace(weight=None, bias=None)
    feed_forward = SimpleNamespace(w1=ffn_layer, w2=ffn_layer, w3=ffn_layer)

    assert lumina_model._fused_rms_modulated_linear(None, linear, None, None) is None
    assert lumina_model._fused_swiglu_ffn_postnorm(None, feed_forward, None) is None
    assert lumina_model._fused_swiglu_ffn(None, feed_forward, None, None) is None


@pytest.mark.parametrize("wrapper_type", [model_base.Lumina2, model_base.ZImagePixelSpace])
def test_lumina_wrapper_delegates_dynamic_vram_units(wrapper_type):
    """Lumina wrappers must expose their diffusion model's unit ordering."""
    expected = ([object(), object()], [object()])

    class DynamicUnitsModel(torch.nn.Module):
        """Minimal diffusion model exposing Dynamic VRAM units."""

        def get_dynamic_vram__units(self):
            """Return the sentinel execution order."""
            return expected

    wrapper = wrapper_type.__new__(wrapper_type)
    torch.nn.Module.__init__(wrapper)
    wrapper.diffusion_model = DynamicUnitsModel()

    assert wrapper.get_dynamic_vram__units() is expected
