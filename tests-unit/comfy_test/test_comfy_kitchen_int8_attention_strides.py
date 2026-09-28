import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy.ldm.modules.attention import _comfy_kitchen_int8_inputs


def _fused_qkv_views(seq_len, heads, head_dim):
    """Mimics MiniMax H3's Attention.forward: q/k/v come from splitting the
    output of a single fused qkv_proj, so each is a view whose per-token
    stride spans all three of q, k, and v instead of just its own slice."""
    inner = heads * head_dim
    buf = torch.randn(seq_len, 3 * inner)
    q, k, v = buf.split(inner, dim=-1)
    q, k, v = (t.view(seq_len, heads, head_dim).transpose(0, 1).unsqueeze(0) for t in (q, k, v))
    return q, k, v


def test_skip_reshape_materializes_contiguous_qkv():
    heads, head_dim, seq_len = 4, 8, 16
    q, k, v = _fused_qkv_views(seq_len, heads, head_dim)
    assert q.stride(2) == 3 * heads * head_dim

    q_out, k_out, v_out, mask, b, dim_head = _comfy_kitchen_int8_inputs(
        q, k, v, heads, mask=None, skip_reshape=True, enable_gqa=False
    )

    for name, original, materialized in (("q", q, q_out), ("k", k, k_out), ("v", v, v_out)):
        assert materialized.is_contiguous(), f"{name} should be contiguous"
        assert materialized.stride(2) == head_dim, f"{name} stride should not include the qkv fusion factor"
        torch.testing.assert_close(materialized, original)
