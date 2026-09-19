from typing import Optional

import torch
import torch.nn as nn
from comfy.ldm.modules.attention import optimized_attention


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act_layer=nn.ReLU, device=None, dtype=None, operations=None):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            operations.Linear(dims[i], dims[i + 1], device=device, dtype=dtype)
            for i in range(num_layers)
        )
        self.act = act_layer()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x

class Attention(nn.Module):
    def __init__(self, embed_dims, num_heads, query_dims=None, key_dims=None, value_dims=None, qkv_bias=True, proj_bias=True,
                 device=None, dtype=None, operations=None):
        super().__init__()
        self.query_dims = query_dims or embed_dims
        self.key_dims = key_dims or embed_dims
        self.value_dims = value_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads

        lin = lambda i, o, b: operations.Linear(i, o, bias=b, device=device, dtype=dtype)
        self.q_proj = lin(self.query_dims, embed_dims, qkv_bias)
        self.k_proj = lin(self.key_dims,   embed_dims, qkv_bias)
        self.v_proj = lin(self.value_dims, embed_dims, qkv_bias)
        self.proj   = lin(embed_dims, self.query_dims, proj_bias)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.reshape(b, n, self.num_heads, self.head_dims).transpose(1, 2)

    def forward(self, q, k, v, attn_mask: Optional[torch.Tensor] = None):
        q, k, v = self._split(self.q_proj(q)), self._split(self.k_proj(k)), self._split(self.v_proj(v))
        x = optimized_attention(q, k, v, self.num_heads, mask=attn_mask, skip_reshape=True, low_precision_attention=False)
        return self.proj(x)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, token_dims, context_dims, num_heads=8, head_dims=64, mlp_dims=1024,
                 repeat_pe=False, skip_first_pe=False, device=None, dtype=None, operations=None):
        super().__init__()
        self.repeat_pe = repeat_pe
        self.skip_first_pe = skip_first_pe

        ln = lambda d: operations.LayerNorm(d, eps=1e-6, device=device, dtype=dtype)
        attn_dim = num_heads * head_dims
        attn_kwargs = dict(embed_dims=attn_dim, num_heads=num_heads, device=device, dtype=dtype, operations=operations)

        if repeat_pe:
            self.ln_pe_1, self.ln_pe_2 = ln(token_dims), ln(context_dims)

        self.ln1 = ln(token_dims)
        self.self_attn = Attention(query_dims=token_dims, key_dims=token_dims, value_dims=token_dims, **attn_kwargs)

        self.ln2_1, self.ln2_2 = ln(token_dims), ln(context_dims)
        self.cross_attn = Attention(query_dims=token_dims, key_dims=context_dims, value_dims=context_dims, **attn_kwargs)

        self.ln3 = ln(token_dims)
        self.ffn = MLP(token_dims, mlp_dims, token_dims, num_layers=2, act_layer=nn.GELU, device=device, dtype=dtype, operations=operations)

    def forward(self, x, context, x_pe=None, context_pe=None, x_mask=None):
        """x: [B, N_tokens, C], context: [B, N_ctx, C], x_mask: [B, N_tokens] or None."""
        # LaPE-style PE re-norm per layer.
        if self.repeat_pe and context_pe is not None:
            x_pe = self.ln_pe_1(x_pe)
            context_pe = self.ln_pe_2(context_pe)

        # Self-attn over tokens.
        if self.repeat_pe and not self.skip_first_pe and x_pe is not None:
            q = k = self.ln1(x) + x_pe
            v = self.ln1(x)
        else:
            q = k = v = self.ln1(x)

        attn_mask = None
        if x_mask is not None:
            attn_mask = x_mask[:, :, None] @ x_mask[:, None, :]
            attn_mask.diagonal(dim1=1, dim2=2).fill_(1)  # avoid all-invalid rows -> nan
            attn_mask = attn_mask > 0
        x = x + self.self_attn(q, k, v, attn_mask=attn_mask)

        # Cross-attn: tokens attend to image context.
        if self.repeat_pe and context_pe is not None:
            q = self.ln2_1(x) + x_pe
            k = self.ln2_2(context) + context_pe
            v = self.ln2_2(context)
        else:
            q = self.ln2_1(x)
            k = v = self.ln2_2(context)
        x = x + self.cross_attn(q, k, v)

        x = x + self.ffn(self.ln3(x))
        return x, context
