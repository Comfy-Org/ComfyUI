#original code from https://github.com/genmoai/models under apache 2.0 license
#adapted to ComfyUI

from typing import Optional

import torch
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def pool_tokens(x: torch.Tensor, mask: torch.Tensor, *, keepdim=False) -> torch.Tensor:
    """
    Pool tokens in x using mask.

    NOTE: We assume x does not require gradients.

    Args:
        x: (B, L, D) tensor of tokens.
        mask: (B, L) boolean tensor indicating which tokens are not padding.

    Returns:
        pooled: (B, D) tensor of pooled tokens.
    """
    assert x.size(1) == mask.size(1)  # Expected mask to have same length as tokens.
    assert x.size(0) == mask.size(0)  # Expected mask to have same batch size as tokens.
    mask = mask[:, :, None].to(dtype=x.dtype)
    mask = mask / mask.sum(dim=1, keepdim=True).clamp(min=1)
    pooled = (x * mask).sum(dim=1, keepdim=keepdim)
    return pooled


class AttentionPool(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        output_dim: int = None,
        device: Optional[torch.device] = None,
        dtype=None,
        operations=None,
    ):
        """
        Args:
            spatial_dim (int): Number of tokens in sequence length.
            embed_dim (int): Dimensionality of input tokens.
            num_heads (int): Number of attention heads.
            output_dim (int): Dimensionality of output tokens. Defaults to embed_dim.
        """
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.num_heads = num_heads
        self.to_kv = operations.Linear(embed_dim, 2 * embed_dim, device=device, dtype=dtype)
        self.to_q = operations.Linear(embed_dim, embed_dim, device=device, dtype=dtype)
        self.to_out = operations.Linear(embed_dim, output_dim or embed_dim, device=device, dtype=dtype)

    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): (B, L, D) tensor of input tokens.
            mask (torch.Tensor): (B, L) boolean tensor indicating which tokens are not padding.

        NOTE: We assume x does not require gradients.

        Returns:
            x (torch.Tensor): (B, D) tensor of pooled tokens.
        """
        D = x.size(2)

        # Construct attention mask, shape: (B, 1, num_queries=1, num_keys=1+L).
        attn_mask = mask[:, None, None, :].bool()  # (B, 1, 1, L).
        attn_mask = F.pad(attn_mask, (1, 0), value=True)  # (B, 1, 1, 1+L).

        # Average non-padding token features. These will be used as the query.
        x_pool = pool_tokens(x, mask, keepdim=True)  # (B, 1, D)

        # Concat pooled features to input sequence.
        x = torch.cat([x_pool, x], dim=1)  # (B, L+1, D)

        # Compute queries, keys, values. Only the mean token is used to create a query.
        kv = self.to_kv(x)  # (B, L+1, 2 * D)
        q = self.to_q(x[:, 0])  # (B, D)

        # Extract heads.
        head_dim = D // self.num_heads
        kv = kv.unflatten(2, (2, self.num_heads, head_dim))  # (B, 1+L, 2, H, head_dim)
        kv = kv.transpose(1, 3)  # (B, H, 2, 1+L, head_dim)
        k, v = kv.unbind(2)  # (B, H, 1+L, head_dim)
        q = q.unflatten(1, (self.num_heads, head_dim))  # (B, H, head_dim)
        q = q.unsqueeze(2)  # (B, H, 1, head_dim)

        # Compute attention.
        del kv
        q, k, v = AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v)
        x = optimized_attention(q, k, v, self.num_heads, mask=attn_mask, skip_reshape=True, skip_output_reshape=True, preferred_attention=self.comfy_attention)

        # Concatenate heads and run output.
        x = x.squeeze(2).flatten(1, 2)  # (B, D = H * head_dim)
        x = self.to_out(x)
        return x
