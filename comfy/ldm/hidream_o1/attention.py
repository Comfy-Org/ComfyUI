"""HiDream-O1 two-pass attention: tokens [0, ar_len) are causal, [ar_len, T)
attend full K/V. Splitting Q at the boundary avoids the (B, 1, T, T) additive
mask the general-purpose path would build (~500 MB at T~16K) and lets the
gen half hit the user's preferred backend via optimized_attention.
"""

import torch

import comfy.ops
from comfy.ldm.modules.attention import optimized_attention


def make_two_pass_attention(ar_len: int, transformer_options=None):
    """Build a two-pass attention callable. AR pass uses SDPA-causal directly, gen pass routes through optimized_attention.
    The AR pass goes through SDPA directand bypasses wrappers, it is only ~1% of T at typical edit sizes.
    """

    def two_pass_attention(q, k, v, heads, **kwargs):
        B, H, T, D = q.shape

        if T < k.shape[2]: # KV-cache hot path: Q is shorter than K/V (cached AR prefix is in K/V only), all fresh Q positions are in the gen region, single full-attention call
            out = optimized_attention(q, k, v, heads, mask=None, skip_reshape=True, skip_output_reshape=True, transformer_options=transformer_options)
        elif ar_len >= T:
            out = comfy.ops.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        elif ar_len <= 0:
            out = optimized_attention(q, k, v, heads, mask=None, skip_reshape=True, skip_output_reshape=True, transformer_options=transformer_options)
        else:
            out_ar = comfy.ops.scaled_dot_product_attention(
                q[:, :, :ar_len], k[:, :, :ar_len], v[:, :, :ar_len],
                attn_mask=None, dropout_p=0.0, is_causal=True,
            )
            out_gen = optimized_attention(
                q[:, :, ar_len:], k, v, heads,
                mask=None, skip_reshape=True, skip_output_reshape=True,
                transformer_options=transformer_options,
            )
            out = torch.cat([out_ar, out_gen], dim=2)

        return out.transpose(1, 2).reshape(B, T, H * D)

    return two_pass_attention
