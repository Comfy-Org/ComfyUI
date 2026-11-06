"""Regression test for Wan reference-latent/rope token-count mismatch (issue #16181)."""

from __future__ import annotations

import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.ops  # noqa: E402
from comfy.ldm.wan.model import WanModel  # noqa: E402


def _make_model():
    return WanModel(
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=8,
        in_dim=4,
        dim=32,
        ffn_dim=16,
        freq_dim=8,
        text_dim=8,
        out_dim=4,
        num_heads=2,
        num_layers=1,
        in_dim_ref_conv=4,
        operations=comfy.ops.disable_weight_init,
    )


def test_wan_model_accepts_odd_sized_reference_latent():
    # A reference latent whose spatial size isn't an even multiple of the
    # (1, 2, 2) patch size used to make ref_conv emit fewer tokens than the
    # rope embedding allocated for it, crashing apply_rope with a
    # "freqs shape is not broadcastable to input" error.
    torch.manual_seed(0)
    model = _make_model()

    x = torch.randn(1, 4, 2, 5, 6)
    timestep = torch.tensor([1.0])
    context = torch.randn(1, 8, 8)
    reference_latent = torch.randn(1, 4, 5, 6)

    out = model(x, timestep, context, reference_latent=reference_latent)

    assert out.shape == x.shape
