"""Regression test for the MiniMax H3 VAE attention's qk_norm_scale buffer.

ModelPatcher's lowvram loader only moves modules that own parameters or
implement comfy_cast_weights (see _load_list in comfy/model_patcher.py), so
the Attention module's own qk_norm_scale buffer never gets moved along with
the rest of the model. comfy_kitchen only ships specialized rms_rope kernels
for CUDA; other backends (e.g. XPU) fall back to its eager reference path,
which calls torch.nn.functional.rms_norm and therefore enforces that the
scale tensor is on the same device as query/key, raising a "not on the same
device" RuntimeError.
"""

import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ldm.minimax.vae as vae_module  # noqa: E402


class _PreNorm:
    def __init__(self, dim, device):
        self.weight = torch.ones(dim, device=device)
        self.eps = 1e-5


def _fake_rms_rope_split_half_(q, k, freqs_cis, q_scale, k_scale=None, epsilon=1e-6, rot_dim=0):
    # Mirrors comfy_kitchen's eager reference kernel closely enough to
    # reproduce its device check without needing XPU/CUDA hardware.
    torch.nn.functional.rms_norm(q, (q.shape[-1],), weight=q_scale, eps=epsilon)
    return q, k


def _fake_optimized_attention(q, k, v, heads, skip_reshape=False):
    return v.transpose(1, 2).reshape(v.shape[0], v.shape[2], -1)


def _attention_with_stranded_buffer(heads, dim_head):
    """An Attention module whose weights moved to another device but whose
    qk_norm_scale buffer was left behind on cpu, as _load_list would do."""
    attn = vae_module.Attention(heads=heads, dim_head=dim_head)
    attn.to_qkv = attn.to_qkv.to("meta")
    attn.to_out = attn.to_out.to("meta")
    attn.norm_q = attn.norm_q.to("meta")
    attn.norm_k = attn.norm_k.to("meta")
    return attn


def test_attention_moves_qk_norm_scale_buffer_to_query_device(monkeypatch):
    heads, dim_head = 2, 4
    dim = heads * dim_head
    attn = _attention_with_stranded_buffer(heads, dim_head)

    monkeypatch.setattr(
        vae_module.comfy.quant_ops.ck, "rms_rope_split_half_", _fake_rms_rope_split_half_
    )
    monkeypatch.setattr(vae_module, "optimized_attention", _fake_optimized_attention)

    x = torch.randn(1, 3, dim, device="meta")
    rotary_pos_emb = torch.zeros(1, 3, dim_head // 2, 2, device="meta")

    with torch.no_grad():
        out = attn.forward(
            x, rotary_pos_emb, pre_norm=_PreNorm(dim, "meta"),
            residual=x, residual_scale=torch.ones_like(x),
        )

    assert out.shape == (1, 3, dim)
    assert out.device.type == "meta"
