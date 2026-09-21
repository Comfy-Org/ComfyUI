"""SeedVR2 internals regression tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.ldm.seedvr.model as seedvr_model  # noqa: E402
import comfy.ldm.seedvr.vae as vae_mod  # noqa: E402
import comfy.ldm.modules.attention as attention  # noqa: E402
import comfy.ops as comfy_ops  # noqa: E402
from comfy.ldm.seedvr.vae import (  # noqa: E402
    causal_norm_wrapper,
    set_norm_limit,
)
from comfy.ldm.seedvr.attention import var_attention_optimized_split  # noqa: E402


_NUM_CHANNELS = 8
_NUM_GROUPS = 4
_TENSOR_SHAPE = (1, 8, 2, 4, 4)

_GROUPNORM_SUBCLASSES = [
    pytest.param(comfy_ops.disable_weight_init.GroupNorm, id="disable_weight_init"),
    pytest.param(comfy_ops.manual_cast.GroupNorm, id="manual_cast"),
]


@pytest.mark.parametrize("groupnorm_cls", _GROUPNORM_SUBCLASSES)
def test_seedvr_groupnorm_low_limit_uses_chunked_groupnorm_path(groupnorm_cls):
    real_group_norm = vae_mod.F.group_norm
    set_norm_limit(1e-9)
    try:
        gn = groupnorm_cls(num_channels=_NUM_CHANNELS, num_groups=_NUM_GROUPS)
        gn.eval()

        forward_hook_calls = []

        def _hook(module, inputs, output):
            forward_hook_calls.append(tuple(inputs[0].shape))

        spy_calls = []

        def _group_norm_spy(input_tensor, num_groups_arg, *args, **kwargs):
            spy_calls.append({"num_groups": int(num_groups_arg)})
            return real_group_norm(input_tensor, num_groups_arg, *args, **kwargs)

        handle = gn.register_forward_hook(_hook)
        try:
            with patch.object(vae_mod.F, "group_norm", side_effect=_group_norm_spy):
                out_tensor = causal_norm_wrapper(gn, torch.randn(*_TENSOR_SHAPE))
        finally:
            handle.remove()

        full_calls = len(forward_hook_calls)
        chunked_calls = sum(1 for entry in spy_calls if entry["num_groups"] < _NUM_GROUPS)

        assert tuple(int(s) for s in out_tensor.shape) == _TENSOR_SHAPE
        assert full_calls == 0, (
            f"low-limit GroupNorm gate must NOT take the full-forward path; got full_calls={full_calls}"
        )
        assert chunked_calls > 0, (
            f"low-limit GroupNorm gate must take the chunked path; got chunked_calls={chunked_calls}"
        )
    finally:
        set_norm_limit(None)


def test_seedvr2_7b_swin_attention_forward_uses_optimized_var_attention(monkeypatch):
    dim = 8
    heads = 2
    head_dim = 4
    attn = seedvr_model.NaSwinAttention(
        vid_dim=dim,
        txt_dim=dim,
        heads=heads,
        head_dim=head_dim,
        qk_bias=False,
        qk_norm=comfy_ops.disable_weight_init.RMSNorm,
        qk_norm_eps=1e-6,
        rope_type=None,
        rope_dim=head_dim,
        shared_weights=False,
        window=(2, 1, 1),
        window_method="720pwin_by_size_bysize",
        version=True,
        device="cpu",
        dtype=torch.float32,
        operations=comfy_ops.disable_weight_init,
    )
    generator = torch.Generator(device="cpu").manual_seed(11)
    vid = torch.randn(8, dim, generator=generator)
    txt = torch.randn(3, dim, generator=generator)
    vid_shape = torch.tensor([[2, 2, 2]], dtype=torch.long)
    txt_shape = torch.tensor([[3]], dtype=torch.long)
    calls = []

    def fake_optimized_var_attention(**kwargs):
        calls.append(kwargs)
        return kwargs["q"]

    monkeypatch.setattr(seedvr_model, "optimized_var_attention", fake_optimized_var_attention)

    vid_out, txt_out = attn(vid, txt, vid_shape, txt_shape, seedvr_model.Cache(disable=True))

    assert tuple(vid_out.shape) == (8, dim)
    assert tuple(txt_out.shape) == (3, dim)
    assert len(calls) == 1
    call = calls[0]
    assert tuple(call["q"].shape) == (14, heads, head_dim)
    assert tuple(call["k"].shape) == (14, heads, head_dim)
    assert tuple(call["v"].shape) == (14, heads, head_dim)
    assert call["heads"] == heads
    assert call["skip_reshape"] is True
    assert call["skip_output_reshape"] is True
    assert call["cu_seqlens_q"] == [0, 7, 14]
    assert call["cu_seqlens_k"] == [0, 7, 14]


def test_var_attention_optimized_split_calls_dense_backend_per_window(monkeypatch):
    heads = 2
    head_dim = 3
    q = torch.arange(30, dtype=torch.float32).reshape(5, heads, head_dim)
    k = q + 100
    v = q + 200
    cu = [0, 2, 5]
    calls = []

    def fake_optimized_attention(q_arg, k_arg, v_arg, heads_arg, **kwargs):
        calls.append(
            {
                "q_shape": tuple(q_arg.shape),
                "k_shape": tuple(k_arg.shape),
                "v_shape": tuple(v_arg.shape),
                "heads": heads_arg,
                "kwargs": kwargs,
            }
        )
        return q_arg + v_arg

    monkeypatch.setattr(attention, "optimized_attention", fake_optimized_attention)

    out = var_attention_optimized_split(
        q,
        k,
        v,
        heads,
        cu,
        cu,
        skip_reshape=True,
        skip_output_reshape=True,
    )

    assert tuple(out.shape) == (5, heads, head_dim)
    assert len(calls) == 2
    assert calls[0]["q_shape"] == (1, heads, 2, head_dim)
    assert calls[1]["q_shape"] == (1, heads, 3, head_dim)
    assert all(call["heads"] == heads for call in calls)
    assert all(call["kwargs"]["skip_reshape"] is True for call in calls)
    assert all(call["kwargs"]["skip_output_reshape"] is True for call in calls)
    torch.testing.assert_close(out, q + v, rtol=0, atol=0)

