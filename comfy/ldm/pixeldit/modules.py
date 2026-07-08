import torch
import torch.nn as nn

from comfy.ldm.flux.math import apply_rope, rope
from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.modules.diffusionmodules.mmdit import Mlp, get_1d_sincos_pos_embed_from_grid_torch


def apply_adaln_(x, shift, scale):
    return x.addcmul_(x, scale).add_(shift)


def precompute_freqs_cis_2d(dim, height, width, theta=10000.0, scale=16.0,
                            ref_grid_h=None, ref_grid_w=None,
                            scale_x=1.0, scale_y=1.0, shift_x=0.0, shift_y=0.0,
                            device=None, dtype=torch.float32, **kwargs):
    """2D RoPE with x/y axis frequencies interleaved at stride 2 across head dim.

    rope_options:
      scale_x / scale_y multiply the position range (RoPE extrapolation).
      shift_x / shift_y offset the position origin (tiled / regional inference).
    With ref_grid_h/w set, also applies NTK-aware per-axis theta scaling
    (rope_mode='ntk_aware'): theta_axis = theta * (current/ref)^(dim_axis/(dim_axis-2)).
    Returns Flux-format rotation matrices of shape [H*W, dim/2, 2, 2].
    Layout of head-dim pairs: [x_0, y_0, x_1, y_1, ..., x_{dim/4-1}, y_{dim/4-1}].
    """
    dim_axis = dim // 2
    if ref_grid_h is not None and dim_axis > 2:
        h_ntk = (height / ref_grid_h) ** (dim_axis / (dim_axis - 2))
        w_ntk = (width / ref_grid_w) ** (dim_axis / (dim_axis - 2))
    else:
        h_ntk = w_ntk = 1.0

    x_lin = torch.linspace(shift_x, scale * scale_x + shift_x, width, device=device)
    y_lin = torch.linspace(shift_y, scale * scale_y + shift_y, height, device=device)
    y_grid, x_grid = torch.meshgrid(y_lin, x_lin, indexing="ij")
    x_rope = rope(x_grid.reshape(1, -1), dim_axis, theta * w_ntk).squeeze(0)
    y_rope = rope(y_grid.reshape(1, -1), dim_axis, theta * h_ntk).squeeze(0)
    out = torch.stack([x_rope, y_rope], dim=2).reshape(height * width, dim // 2, 2, 2)
    return out.to(dtype=dtype)


def get_2d_sincos_pos_embed(embed_dim, height, width, device=None, dtype=torch.float32):
    """Standard 2D sin/cos absolute positional embedding (ViT-style).

    first half encodes W-coordinates, second half H.
    """
    assert embed_dim % 4 == 0
    grid_h = torch.arange(height, dtype=torch.float32, device=device)
    grid_w = torch.arange(width, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(grid_h, grid_w, indexing="ij")
    emb_w = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid_x.reshape(-1), device=device)
    emb_h = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid_y.reshape(-1), device=device)
    return torch.cat([emb_w, emb_h], dim=1).to(dtype=dtype)


class RotaryAttention(nn.Module):
    """Single-stream self-attention with rotary positional encoding (used inside PiTBlock)."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, dtype=None, device=None, operations=None):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = operations.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        self.q_norm = operations.RMSNorm(self.head_dim, eps=1e-6, dtype=dtype, device=device)
        self.k_norm = operations.RMSNorm(self.head_dim, eps=1e-6, dtype=dtype, device=device)
        self.proj = operations.Linear(dim, dim, dtype=dtype, device=device)

    def forward(self, x, pos, mask=None, transformer_options={}):
        B, N, C = x.shape
        H = self.num_heads
        D = self.head_dim
        qkv = self.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = apply_rope(self.q_norm(q), self.k_norm(k), pos[None, None])
        x = optimized_attention(q, k, v, H, mask=mask, skip_reshape=True, transformer_options=transformer_options)
        return self.proj(x)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm = operations.RMSNorm(hidden_size, eps=1e-6, dtype=dtype, device=device)
        self.linear = operations.Linear(hidden_size, out_channels, bias=True, dtype=dtype, device=device)

    def forward(self, x):
        return self.linear(self.norm(x))


class PatchTokenEmbedder(nn.Module):
    """Linear projection used both for patchified-image tokens and text-feature tokens."""
    def __init__(self, in_chans, embed_dim, use_norm=False, bias=True, dtype=None, device=None, operations=None):
        super().__init__()
        self.proj = operations.Linear(in_chans, embed_dim, bias=bias, dtype=dtype, device=device)
        self.norm = operations.RMSNorm(embed_dim, eps=1e-6, dtype=dtype, device=device) if use_norm else nn.Identity()

    def forward(self, x):
        return self.norm(self.proj(x))


class PixelTokenEmbedder(nn.Module):
    """Pixel-level embedder: lifts each RGB pixel to hidden_size and packs into per-patch sequences."""
    def __init__(self, in_channels, hidden_size_output, dtype=None, device=None, operations=None):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size_output = hidden_size_output
        self.proj = operations.Linear(self.in_channels, self.hidden_size_output, bias=True, dtype=dtype, device=device)

    def forward(self, inputs, patch_size):
        B, _, H, W = inputs.shape
        Hs, Ws = H // patch_size, W // patch_size
        P2 = patch_size * patch_size
        x = inputs.permute(0, 2, 3, 1).contiguous()
        x = self.proj(x)
        pos_full = get_2d_sincos_pos_embed(self.hidden_size_output, H, W, device=x.device, dtype=x.dtype).view(H, W, self.hidden_size_output)
        x = x + pos_full.unsqueeze(0)
        x = x.view(B, Hs, patch_size, Ws, patch_size, self.hidden_size_output)
        return x.permute(0, 1, 3, 2, 4, 5).reshape(B * Hs * Ws, P2, self.hidden_size_output)


class PiTBlock(nn.Module):
    """Pixel-level transformer block.

    Compresses each patch's P^2 pixel tokens → 1 attention token via a linear,
    runs global self-attention across patches with 2D RoPE, then expands back to P^2 tokens.
    Conditioning is per-pixel adaLN from the patch-level features.
    """
    def __init__(self, pixel_hidden_size, patch_hidden_size, patch_size, num_heads, mlp_ratio=4.0,
                 attn_hidden_size=None, attn_num_heads=None, dtype=None, device=None, operations=None, mlp_chunks=1):
        super().__init__()
        self.pixel_dim = pixel_hidden_size
        self.context_dim = patch_hidden_size
        self.attn_dim = attn_hidden_size if attn_hidden_size is not None else patch_hidden_size
        self.num_heads = attn_num_heads if attn_num_heads is not None else num_heads
        assert self.attn_dim % self.num_heads == 0

        p2 = patch_size * patch_size
        self.compress_to_attn = operations.Linear(p2 * self.pixel_dim, self.attn_dim, bias=True, dtype=dtype, device=device)
        self.expand_from_attn = operations.Linear(self.attn_dim, p2 * self.pixel_dim, bias=True, dtype=dtype, device=device)

        self.norm1 = operations.RMSNorm(self.pixel_dim, eps=1e-6, dtype=dtype, device=device)
        self.attn = RotaryAttention(self.attn_dim, num_heads=self.num_heads, qkv_bias=False, dtype=dtype, device=device, operations=operations)
        self.norm2 = operations.RMSNorm(self.pixel_dim, eps=1e-6, dtype=dtype, device=device)
        self.mlp = Mlp(self.pixel_dim, hidden_features=int(self.pixel_dim * mlp_ratio), dtype=dtype, device=device, operations=operations)

        self.adaLN_modulation_msa = operations.Linear(self.context_dim, 3 * self.pixel_dim * p2, bias=True, dtype=dtype, device=device)
        self.adaLN_modulation_mlp = operations.Linear(self.context_dim, 3 * self.pixel_dim * p2, bias=True, dtype=dtype, device=device)

        self._rope_fn = precompute_freqs_cis_2d
        self.mlp_chunks = max(1, int(mlp_chunks))

    def _fetch_pos(self, height, width, device, dtype, **rope_opts):
        return self._rope_fn(self.attn_dim // self.num_heads, height, width, device=device, dtype=dtype, **rope_opts)

    def forward(self, x, s_cond, image_height, image_width, patch_size, mask=None, transformer_options={}):
        BL, P2, _ = x.shape
        Hs, Ws = image_height // patch_size, image_width // patch_size
        L = Hs * Ws
        B = BL // L

        # Attention path uses only msa params; compute, use, free before mlp params allocate.
        msa_params = self.adaLN_modulation_msa(s_cond).view(BL, P2, 3 * self.pixel_dim)
        shift_msa, scale_msa, gate_msa = msa_params.chunk(3, dim=-1)

        x_norm = apply_adaln_(self.norm1(x), shift_msa, scale_msa)
        x_flat = x_norm.view(BL, P2 * self.pixel_dim)

        x_comp = self.compress_to_attn(x_flat).view(B, L, self.attn_dim)
        pos_comp = self._fetch_pos(Hs, Ws, x.device, x.dtype, **(transformer_options.get("rope_options") or {}))
        attn_out = self.attn(x_comp, pos_comp, mask=mask, transformer_options=transformer_options)
        attn_flat = self.expand_from_attn(attn_out.view(B * L, self.attn_dim))
        attn_exp = attn_flat.view(BL, P2, self.pixel_dim)
        x = torch.addcmul(x, gate_msa, attn_exp)
        del msa_params, shift_msa, scale_msa, gate_msa

        mlp_params = self.adaLN_modulation_mlp(s_cond).view(BL, P2, 3 * self.pixel_dim)
        shift_mlp, scale_mlp, gate_mlp = mlp_params.chunk(3, dim=-1)
        gate_mlp = gate_mlp.contiguous()  # detach from mlp_params so the del below frees shift+scale storage before the MLP
        mlp_input = apply_adaln_(self.norm2(x), shift_mlp, scale_mlp)
        del mlp_params, shift_mlp, scale_mlp

        # MLP in chunks since the peak memory usage is huge here
        chunk_size = (BL + self.mlp_chunks - 1) // self.mlp_chunks
        for s in range(0, BL, chunk_size):
            e = min(s + chunk_size, BL)
            x[s:e].addcmul_(gate_mlp[s:e], self.mlp(mlp_input[s:e]))
        return x
