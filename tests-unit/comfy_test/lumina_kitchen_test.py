"""Lumina Comfy Kitchen integration regression tests."""

from types import SimpleNamespace

import torch

from comfy.cli_args import args

original_cpu = args.cpu
if not torch.cuda.is_available():
    args.cpu = True

import comfy.ldm.lumina.model as lumina_model  # noqa: E402

args.cpu = original_cpu


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


def test_legacy_offloaded_weights_skip_fusion():
    """Weights streamed without a VBAR cannot be cast together and must not fuse."""
    x = torch.empty(1, 4, device="meta")
    offloaded = SimpleNamespace(weight=torch.empty(4, 4))

    with lumina_model._cast_together((offloaded,), x) as supported:
        assert not supported
