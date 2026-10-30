import torch
import torch.nn as nn
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention
import comfy.ops

class AttentionPool(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None, dtype=None, device=None, operations=None):
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.positional_embedding = nn.Parameter(torch.empty(spacial_dim + 1, embed_dim, dtype=dtype, device=device))
        self.k_proj = operations.Linear(embed_dim, embed_dim, dtype=dtype, device=device)
        self.q_proj = operations.Linear(embed_dim, embed_dim, dtype=dtype, device=device)
        self.v_proj = operations.Linear(embed_dim, embed_dim, dtype=dtype, device=device)
        self.c_proj = operations.Linear(embed_dim, output_dim or embed_dim, dtype=dtype, device=device)
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def forward(self, x):
        x = x[:,:self.positional_embedding.shape[0] - 1]
        x = x.permute(1, 0, 2)  # NLC -> LNC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (L+1)NC
        x = x + comfy.ops.cast_to_input(self.positional_embedding[:, None, :], x) # (L+1)NC

        q = self.q_proj(x[:1])
        k = self.k_proj(x)
        v = self.v_proj(x)

        batch_size = q.shape[1]
        head_dim = self.embed_dim // self.num_heads
        q = q.view(1, batch_size * self.num_heads, head_dim).transpose(0, 1).view(batch_size, self.num_heads, -1, head_dim)
        k = k.view(k.shape[0], batch_size * self.num_heads, head_dim).transpose(0, 1).view(batch_size, self.num_heads, -1, head_dim)
        v = v.view(v.shape[0], batch_size * self.num_heads, head_dim).transpose(0, 1).view(batch_size, self.num_heads, -1, head_dim)

        q, k, v = AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v)
        attn_output = optimized_attention(q, k, v, self.num_heads, skip_reshape=True, preferred_attention=self.comfy_attention).transpose(0, 1)

        attn_output = self.c_proj(attn_output)
        return attn_output.squeeze(0)
