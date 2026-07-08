import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy.ldm.modules.attention import optimized_attention
import comfy.model_management
import comfy.quant_ops

def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0
    if not comfy.model_management.supports_fp64(pos.device):
        device = torch.device("cpu")
    else:
        device = pos.device

    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.to(device), omega)
    out = torch.stack([torch.cos(out), torch.sin(out)], dim=0)
    return out.to(dtype=torch.float32, device=pos.device)

class ErnieImageEmbedND3(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: tuple):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = list(axes_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        emb = torch.cat([rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(3)], dim=-1)
        cos_ = emb[0]
        sin_ = emb[1]
        N = cos_.shape[-1]
        half = N // 2
        cos_top = cos_[..., :half].repeat_interleave(2, dim=-1)
        sin_top = sin_[..., :half].repeat_interleave(2, dim=-1)
        cos_bot = cos_[..., half:].repeat_interleave(2, dim=-1)
        sin_bot = sin_[..., half:].repeat_interleave(2, dim=-1)
        rot = torch.stack([cos_top, -sin_top, sin_bot, cos_bot], dim=-1)
        return rot.reshape(*rot.shape[:-1], 2, 2).unsqueeze(2)

class ErnieImagePatchEmbedDynamic(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int, operations, device=None, dtype=None):
        super().__init__()
        self.patch_size = patch_size
        self.proj = operations.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        batch_size, dim, height, width = x.shape
        return x.reshape(batch_size, dim, height * width).transpose(1, 2).contiguous()

class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool = False):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        if self.flip_sin_to_cos:
            emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        else:
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int, operations, device=None, dtype=None):
        super().__init__()
        Linear = operations.Linear
        self.linear_1 = Linear(in_channels, time_embed_dim, bias=True, device=device, dtype=dtype)
        self.act = nn.SiLU()
        self.linear_2 = Linear(time_embed_dim, time_embed_dim, bias=True, device=device, dtype=dtype)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample

class ErnieImageAttention(nn.Module):
    def __init__(self, query_dim: int, heads: int, dim_head: int, eps: float = 1e-6, operations=None, device=None, dtype=None):
        super().__init__()
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = heads * dim_head

        Linear = operations.Linear
        RMSNorm = operations.RMSNorm

        self.to_q = Linear(query_dim, self.inner_dim, bias=False, device=device, dtype=dtype)
        self.to_k = Linear(query_dim, self.inner_dim, bias=False, device=device, dtype=dtype)
        self.to_v = Linear(query_dim, self.inner_dim, bias=False, device=device, dtype=dtype)

        self.norm_q = RMSNorm(dim_head, eps=eps, elementwise_affine=True, device=device, dtype=dtype)
        self.norm_k = RMSNorm(dim_head, eps=eps, elementwise_affine=True, device=device, dtype=dtype)

        self.to_out = nn.ModuleList([Linear(self.inner_dim, query_dim, bias=False, device=device, dtype=dtype)])

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None, image_rotary_emb: torch.Tensor = None) -> torch.Tensor:
        B, S, _ = x.shape

        q_flat = self.to_q(x)
        k_flat = self.to_k(x)
        v_flat = self.to_v(x)

        query = q_flat.view(B, S, self.heads, self.head_dim)
        key = k_flat.view(B, S, self.heads, self.head_dim)

        query = self.norm_q(query)
        key = self.norm_k(key)

        if image_rotary_emb is not None:
            query, key = comfy.quant_ops.ck.apply_rope_split_half(query, key, image_rotary_emb)

        q_flat = query.reshape(B, S, -1)
        k_flat = key.reshape(B, S, -1)

        hidden_states = optimized_attention(q_flat, k_flat, v_flat, self.heads, mask=attention_mask)

        return self.to_out[0](hidden_states)

class ErnieImageFeedForward(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int, operations, device=None, dtype=None):
        super().__init__()
        Linear = operations.Linear
        self.gate_proj = Linear(hidden_size, ffn_hidden_size, bias=False, device=device, dtype=dtype)
        self.up_proj = Linear(hidden_size, ffn_hidden_size, bias=False, device=device, dtype=dtype)
        self.linear_fc2 = Linear(ffn_hidden_size, hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.up_proj(x) * F.gelu(self.gate_proj(x)))

class ErnieImageSharedAdaLNBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ffn_hidden_size: int, eps: float = 1e-6, operations=None, device=None, dtype=None):
        super().__init__()
        RMSNorm = operations.RMSNorm

        self.adaLN_sa_ln = RMSNorm(hidden_size, eps=eps, device=device, dtype=dtype)
        self.self_attention = ErnieImageAttention(
            query_dim=hidden_size,
            dim_head=hidden_size // num_heads,
            heads=num_heads,
            eps=eps,
            operations=operations,
            device=device,
            dtype=dtype
        )
        self.adaLN_mlp_ln = RMSNorm(hidden_size, eps=eps, device=device, dtype=dtype)
        self.mlp = ErnieImageFeedForward(hidden_size, ffn_hidden_size, operations=operations, device=device, dtype=dtype)

    def forward(self, x, rotary_pos_emb, temb, attention_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = temb

        residual = x
        x_norm = self.adaLN_sa_ln(x)
        x_norm = x_norm * (1 + scale_msa) + shift_msa

        attn_out = self.self_attention(x_norm, attention_mask=attention_mask, image_rotary_emb=rotary_pos_emb)
        x = residual + gate_msa * attn_out

        residual = x
        x_norm = self.adaLN_mlp_ln(x)
        x_norm = x_norm * (1 + scale_mlp) + shift_mlp

        return residual + gate_mlp * self.mlp(x_norm)

class ErnieImageAdaLNContinuous(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, operations=None, device=None, dtype=None):
        super().__init__()
        LayerNorm = operations.LayerNorm
        Linear = operations.Linear
        self.norm = LayerNorm(hidden_size, elementwise_affine=False, eps=eps, device=device, dtype=dtype)
        self.linear = Linear(hidden_size, hidden_size * 2, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(conditioning).chunk(2, dim=-1)
        x = self.norm(x)
        x = torch.addcmul(shift.unsqueeze(1), x, 1 + scale.unsqueeze(1))
        return x

class ErnieImageModel(nn.Module):
    def __init__(
        self,
        hidden_size: int = 4096,
        num_attention_heads: int = 32,
        num_layers: int = 36,
        ffn_hidden_size: int = 12288,
        in_channels: int = 128,
        out_channels: int = 128,
        patch_size: int = 1,
        text_in_dim: int = 3072,
        rope_theta: int = 256,
        rope_axes_dim: tuple = (32, 48, 48),
        eps: float = 1e-6,
        qk_layernorm: bool = True,
        device=None,
        dtype=None,
        operations=None,
        **kwargs
    ):
        super().__init__()
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.patch_size = patch_size
        self.out_channels = out_channels

        Linear = operations.Linear

        self.x_embedder = ErnieImagePatchEmbedDynamic(in_channels, hidden_size, patch_size, operations, device, dtype)
        self.text_proj = Linear(text_in_dim, hidden_size, bias=False, device=device, dtype=dtype) if text_in_dim != hidden_size else None

        self.time_proj = Timesteps(hidden_size, flip_sin_to_cos=False)
        self.time_embedding = TimestepEmbedding(hidden_size, hidden_size, operations, device, dtype)

        self.pos_embed = ErnieImageEmbedND3(dim=self.head_dim, theta=rope_theta, axes_dim=rope_axes_dim)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            Linear(hidden_size, 6 * hidden_size, device=device, dtype=dtype)
        )

        self.layers = nn.ModuleList([
            ErnieImageSharedAdaLNBlock(hidden_size, num_attention_heads, ffn_hidden_size, eps, operations, device, dtype)
            for _ in range(num_layers)
        ])

        self.final_norm = ErnieImageAdaLNContinuous(hidden_size, eps, operations, device, dtype)
        self.final_linear = Linear(hidden_size, patch_size * patch_size * out_channels, device=device, dtype=dtype)

    def forward(self, x, timesteps, context, **kwargs):
        device, dtype = x.device, x.dtype
        B, C, H, W = x.shape
        p, Hp, Wp = self.patch_size, H // self.patch_size, W // self.patch_size
        N_img = Hp * Wp

        img_bsh = self.x_embedder(x)

        text_bth = context
        if self.text_proj is not None and text_bth.numel() > 0:
            text_bth = self.text_proj(text_bth)
        Tmax = text_bth.shape[1]

        hidden_states = torch.cat([img_bsh, text_bth], dim=1)

        text_ids = torch.zeros((B, Tmax, 3), device=device, dtype=torch.float32)
        text_ids[:, :, 0] = torch.linspace(0, Tmax - 1, steps=Tmax, device=x.device, dtype=torch.float32)
        index = float(Tmax)

        transformer_options = kwargs.get("transformer_options", {})
        rope_options = transformer_options.get("rope_options", None)

        h_len, w_len = float(Hp), float(Wp)
        h_offset, w_offset = 0.0, 0.0

        if rope_options is not None:
            h_len = (h_len - 1.0) * rope_options.get("scale_y", 1.0) + 1.0
            w_len = (w_len - 1.0) * rope_options.get("scale_x", 1.0) + 1.0
            index += rope_options.get("shift_t", 0.0)
            h_offset += rope_options.get("shift_y", 0.0)
            w_offset += rope_options.get("shift_x", 0.0)

        image_ids = torch.zeros((Hp, Wp, 3), device=device, dtype=torch.float32)
        image_ids[:, :, 0] = image_ids[:, :, 1] + index
        image_ids[:, :, 1] = image_ids[:, :, 1] + torch.linspace(h_offset, h_len - 1 + h_offset, steps=Hp, device=device, dtype=torch.float32).unsqueeze(1)
        image_ids[:, :, 2] = image_ids[:, :, 2] + torch.linspace(w_offset, w_len - 1 + w_offset, steps=Wp, device=device, dtype=torch.float32).unsqueeze(0)

        image_ids = image_ids.view(1, N_img, 3).expand(B, -1, -1)

        rotary_pos_emb = self.pos_embed(torch.cat([image_ids, text_ids], dim=1))
        del image_ids, text_ids

        sample = self.time_proj(timesteps).to(dtype)
        c = self.time_embedding(sample)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
            t.unsqueeze(1).contiguous() for t in self.adaLN_modulation(c).chunk(6, dim=-1)
        ]

        temb = [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
        for layer in self.layers:
            hidden_states = layer(hidden_states, rotary_pos_emb, temb)

        hidden_states = self.final_norm(hidden_states, c).type_as(hidden_states)

        patches = self.final_linear(hidden_states)[:, :N_img, :]
        output = (
            patches.view(B, Hp, Wp, p, p, self.out_channels)
            .permute(0, 5, 1, 3, 2, 4)
            .contiguous()
            .view(B, self.out_channels, H, W)
        )

        return output
