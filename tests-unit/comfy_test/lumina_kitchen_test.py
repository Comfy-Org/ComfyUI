"""Lumina Comfy Kitchen integration regression tests."""

import contextlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from comfy.cli_args import args

original_cpu = args.cpu
if not torch.cuda.is_available():
    args.cpu = True

import comfy.ldm.lumina.model as lumina_model  # noqa: E402
import comfy.ops  # noqa: E402
import comfy.quant_ops  # noqa: E402
from comfy.ldm.flux.layers import EmbedND  # noqa: E402

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


def test_fusions_cast_norm_with_weights(monkeypatch):
    """Each fused op casts its norm in the same group as its INT8 weights."""
    groups = []

    @contextlib.contextmanager
    def cast_together(modules, x):
        groups.append(modules)
        yield False

    monkeypatch.setattr(lumina_model, "_int8_convrot_linear", lambda linear: True)
    monkeypatch.setattr(lumina_model, "_cast_together", cast_together)
    monkeypatch.setattr(lumina_model, "_FUSED_RMS_MODULATED", _reference_rms_modulated)
    monkeypatch.setattr(lumina_model, "_FUSED_SWIGLU_FFN", _reference_swiglu_ffn)
    block = _make_block(True)
    x = torch.randn(1, 8, block.dim, dtype=torch.bfloat16)
    scale = torch.zeros(1, block.dim, dtype=torch.bfloat16)

    assert lumina_model._fused_rms_modulated_linear(x, block.attention.qkv, block.attention_norm1, scale) is None
    assert lumina_model._fused_swiglu_ffn(x, block.feed_forward, block.ffn_norm1, scale) is None
    assert lumina_model._fused_swiglu_ffn_postnorm(x, block.feed_forward, block.ffn_norm1) is None
    assert [any(m is block.attention_norm1 or m is block.ffn_norm1 for m in group) for group in groups] == [True] * 3


def _vbar_module():
    return SimpleNamespace(_v=object(), _v_signature=b"resident", weight=torch.empty(4, 4, device="meta"))


def test_streamed_vbar_weights_skip_fusion(monkeypatch):
    """Weights that are not already in VRAM keep the unfused, overlapped streaming path."""
    unpinned = []
    monkeypatch.setattr(lumina_model.comfy_aimdo.model_vbar, "vbar_fault", lambda alloc: b"streamed")
    monkeypatch.setattr(lumina_model.comfy_aimdo.model_vbar, "vbar_unpin", unpinned.append)
    monkeypatch.setattr(lumina_model.comfy.model_prefetch, "pin_modules", pytest.fail)
    modules = (_vbar_module(), _vbar_module())

    with lumina_model._cast_together(modules, torch.empty(1, 4, device="meta")) as supported:
        assert not supported
    assert unpinned == [modules[0]._v]


def test_resident_vbar_weights_fuse(monkeypatch):
    """Resident weights are pinned together for the fused op and released afterwards."""
    pinned, cleaned = [], []
    monkeypatch.setattr(lumina_model.comfy_aimdo.model_vbar, "vbar_fault", lambda alloc: b"resident")
    monkeypatch.setattr(lumina_model.comfy_aimdo.model_vbar, "vbar_unpin", lambda alloc: None)
    monkeypatch.setattr(
        lumina_model.comfy.model_prefetch, "pin_modules",
        lambda modules, device: (pinned.extend(modules), (None, True))[1],
    )
    monkeypatch.setattr(
        lumina_model.comfy.model_prefetch, "cleanup_prefetched_modules",
        lambda module, modules: cleaned.extend(modules),
    )
    modules = (_vbar_module(), _vbar_module())

    with lumina_model._cast_together(modules, torch.empty(1, 4, device="meta")) as supported:
        assert supported
    assert pinned == list(modules)
    assert cleaned == list(modules)


def _reference_rms_modulated(x, weight, bias, norm_weight, eps, modulation_scale):
    """Unfused composition of the Kitchen fused RMS modulation + linear op."""
    if modulation_scale.numel() != x.shape[-1]:
        return NotImplemented
    normed = F.rms_norm(x, (x.shape[-1],), norm_weight, eps)
    return F.linear(normed * (1 + modulation_scale.unsqueeze(1)), weight, bias)


def _reference_swiglu_ffn(x, w1, w3, w2, b1, b3, b2, norm_weight=None, norm_eps=None, modulation_scale=None):
    """Unfused composition of the Kitchen fused SwiGLU FFN op."""
    if norm_weight is not None:
        if modulation_scale.numel() != x.shape[-1]:
            return NotImplemented
        x = F.rms_norm(x, (x.shape[-1],), norm_weight, norm_eps) * (1 + modulation_scale.unsqueeze(1))
    hidden = lumina_model.clamp_fp16(F.silu(F.linear(x, w1, b1)) * F.linear(x, w3, b3))
    return F.linear(hidden, w2, b2)


def _reference_rms_gated_residual(activation, norm_weight, residual, gate, eps):
    """Unfused composition of the Kitchen fused RMS + gate + residual op."""
    return residual + gate * F.rms_norm(activation, (activation.shape[-1],), norm_weight, eps)


def _make_block(modulation):
    torch.manual_seed(0)
    dim, n_heads = 64, 4
    block = lumina_model.JointTransformerBlock(
        0, dim, n_heads, n_heads, multiple_of=16, ffn_dim_multiplier=None,
        norm_eps=1e-5, qk_norm=False, modulation=modulation, z_image_modulation=True,
        operation_settings={"operations": comfy.ops.disable_weight_init, "device": "cpu", "dtype": torch.bfloat16},
    )
    with torch.no_grad():
        for param in block.parameters():
            param.copy_(torch.randn_like(param) * 0.1 + (1.0 if param.ndim == 1 else 0.0))
    return block


def _block_inputs(batch, block):
    seq = 8
    x = torch.randn(batch, seq, block.dim, dtype=torch.bfloat16)
    pos_ids = torch.arange(seq, dtype=torch.float32).view(1, seq, 1).expand(batch, seq, 1)
    freqs_cis = EmbedND(dim=block.head_dim, theta=10000, axes_dim=[block.head_dim])(pos_ids).movedim(1, 2)
    adaln_input = torch.randn(batch, min(block.dim, 256), dtype=torch.bfloat16)
    return x, freqs_cis, adaln_input


def _enable_reference_fusions(monkeypatch, calls):
    def counted(name, fn):
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            if result is not NotImplemented:
                calls.append(name)
            return result
        return wrapper

    @contextlib.contextmanager
    def cast_together(modules, x):
        yield True

    monkeypatch.setattr(lumina_model, "_int8_convrot_linear", lambda linear: True)
    monkeypatch.setattr(lumina_model, "_cast_together", cast_together)
    monkeypatch.setattr(lumina_model, "_FUSED_RMS_MODULATED", counted("rms_modulated", _reference_rms_modulated))
    monkeypatch.setattr(lumina_model, "_FUSED_SWIGLU_FFN", counted("swiglu_ffn", _reference_swiglu_ffn))
    monkeypatch.setattr(
        comfy.quant_ops.ck, "rms_gated_residual",
        counted("rms_gated_residual", _reference_rms_gated_residual), raising=False,
    )


@pytest.mark.parametrize(
    ("modulation", "batch", "expected_calls"),
    [
        (True, 1, ["rms_modulated", "rms_gated_residual", "swiglu_ffn", "rms_gated_residual"]),
        (True, 2, []),
        (False, 1, ["swiglu_ffn"]),
    ],
)
def test_fused_block_matches_unfused(monkeypatch, modulation, batch, expected_calls):
    """The fused block wiring must produce the same output as the unfused path."""
    block = _make_block(modulation)
    x, freqs_cis, adaln_input = _block_inputs(batch, block)
    if not modulation:
        adaln_input = None

    with torch.no_grad():
        unfused = block(x.clone(), None, freqs_cis, adaln_input)
        calls = []
        _enable_reference_fusions(monkeypatch, calls)
        fused = block(x.clone(), None, freqs_cis, adaln_input)

    assert calls == expected_calls
    torch.testing.assert_close(fused, unfused, rtol=2e-2, atol=2e-2)
