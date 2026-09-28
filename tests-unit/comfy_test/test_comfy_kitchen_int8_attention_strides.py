import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.ops
import comfy.ldm.minimax.model as minimax_model
from comfy.ldm.minimax.model import Attention, rope_rotation_table


def test_attention_materializes_contiguous_qkv_for_kitchen_backend():
    """MiniMax H3's qkv_proj is a single fused Linear, so q/k/v start as
    views whose per-token stride spans all three of q, k, and v instead of
    just their own slice. The in-place rope path (taken whenever rope_freqs
    is given, i.e. always in production) never reallocates q/k, so that
    inflated stride would otherwise reach the comfy_kitchen int8 backend and
    overflow its int32 stride indexing."""
    heads, head_dim, seq_len = 2, 8, 6
    attn = Attention(hidden=heads * head_dim, heads=heads, head_dim=head_dim, eps=1e-6, operations=comfy.ops.disable_weight_init)
    attn.requires_grad_(False)

    x = torch.randn(seq_len, heads * head_dim)
    rope_freqs = rope_rotation_table(torch.zeros(seq_len, head_dim), torch.float32)

    captured = {}

    def fake_optimized_attention(q, k, v, heads_, **kwargs):
        captured["q"], captured["k"], captured["v"] = q.peek(), k.peek(), v.peek()
        return torch.zeros(v.peek().shape[0], v.peek().shape[2], heads_ * head_dim)

    original = minimax_model.optimized_attention
    minimax_model.optimized_attention = fake_optimized_attention
    try:
        with torch.no_grad():
            attn.forward(x, rope_freqs=rope_freqs)
    finally:
        minimax_model.optimized_attention = original

    for name, tensor in captured.items():
        assert tensor.is_contiguous(), f"{name} should be contiguous"
        assert tensor.stride(2) == head_dim, f"{name} stride should not include the qkv fusion factor"
