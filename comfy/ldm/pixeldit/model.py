import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ldm.common_dit
import comfy.patcher_extension
from comfy.ldm.flux.math import apply_rope, rope
from comfy.ldm.hidream.model import FeedForwardSwiGLU
from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.modules.diffusionmodules.mmdit import TimestepEmbedder

from .modules import (
    FinalLayer,
    PatchTokenEmbedder,
    PiTBlock,
    PixelTokenEmbedder,
    apply_adaln_,
    precompute_freqs_cis_2d,
)


class MMDiTJointAttention(nn.Module):
    """Joint MMDiT attention with separate Q/K/V/proj for image and text streams.

    RoPE is applied to each stream before concatenation so each stream uses its own
    2D/1D positional encoding. Concat order is [text, image] (text first).
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, dtype=None, device=None, operations=None):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv_x = operations.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        self.qkv_y = operations.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)

        self.q_norm_x = operations.RMSNorm(self.head_dim, eps=1e-6, dtype=dtype, device=device)
        self.k_norm_x = operations.RMSNorm(self.head_dim, eps=1e-6, dtype=dtype, device=device)
        self.q_norm_y = operations.RMSNorm(self.head_dim, eps=1e-6, dtype=dtype, device=device)
        self.k_norm_y = operations.RMSNorm(self.head_dim, eps=1e-6, dtype=dtype, device=device)

        self.proj_x = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.proj_y = operations.Linear(dim, dim, dtype=dtype, device=device)

    def forward(self, x, y, pos_img, pos_txt=None, attn_mask=None, transformer_options={}):
        B, Nx, _ = x.shape
        _, Ny, _ = y.shape
        H = self.num_heads
        D = self.head_dim

        qkv_x = self.qkv_x(x).reshape(B, Nx, 3, H, D).permute(2, 0, 3, 1, 4)
        qx, kx, vx = qkv_x.unbind(0)
        qx = self.q_norm_x(qx)
        kx = self.k_norm_x(kx)

        qkv_y = self.qkv_y(y).reshape(B, Ny, 3, H, D).permute(2, 0, 3, 1, 4)
        qy, ky, vy = qkv_y.unbind(0)
        qy = self.q_norm_y(qy)
        ky = self.k_norm_y(ky)

        qx, kx = apply_rope(qx, kx, pos_img[None, None])
        if pos_txt is not None:
            qy, ky = apply_rope(qy, ky, pos_txt[None, None])

        q_joint = torch.cat([qy, qx], dim=2)
        k_joint = torch.cat([ky, kx], dim=2)
        v_joint = torch.cat([vy, vx], dim=2)

        out_joint = optimized_attention(
            q_joint, k_joint, v_joint, H,
            mask=attn_mask, skip_reshape=True, skip_output_reshape=True,
            transformer_options=transformer_options,
        )

        out_y = out_joint[:, :, :Ny, :].transpose(1, 2).reshape(B, Ny, H * D)
        out_x = out_joint[:, :, Ny:, :].transpose(1, 2).reshape(B, Nx, H * D)

        return self.proj_x(out_x), self.proj_y(out_y)


class MMDiTBlockT2I(nn.Module):
    def __init__(self, hidden_size, groups, mlp_ratio=4.0, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm_x1 = operations.RMSNorm(hidden_size, eps=1e-6, dtype=dtype, device=device)
        self.norm_y1 = operations.RMSNorm(hidden_size, eps=1e-6, dtype=dtype, device=device)
        self.attn = MMDiTJointAttention(hidden_size, num_heads=groups, qkv_bias=False, dtype=dtype, device=device, operations=operations)
        self.norm_x2 = operations.RMSNorm(hidden_size, eps=1e-6, dtype=dtype, device=device)
        self.norm_y2 = operations.RMSNorm(hidden_size, eps=1e-6, dtype=dtype, device=device)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_x = FeedForwardSwiGLU(hidden_size, mlp_hidden_dim, multiple_of=1, dtype=dtype, device=device, operations=operations)
        self.mlp_y = FeedForwardSwiGLU(hidden_size, mlp_hidden_dim, multiple_of=1, dtype=dtype, device=device, operations=operations)
        self.adaLN_modulation_img = nn.Sequential(operations.Linear(hidden_size, 6 * hidden_size, bias=True, dtype=dtype, device=device))
        self.adaLN_modulation_txt = nn.Sequential(operations.Linear(hidden_size, 6 * hidden_size, bias=True, dtype=dtype, device=device))

    def forward(self, x, y, c, pos_img, pos_txt=None, attn_mask=None, transformer_options={}):
        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = self.adaLN_modulation_img(c).chunk(6, dim=-1)
        shift_msa_y, scale_msa_y, gate_msa_y, shift_mlp_y, scale_mlp_y, gate_mlp_y = self.adaLN_modulation_txt(c).chunk(6, dim=-1)

        x_norm = apply_adaln_(self.norm_x1(x), shift_msa_x, scale_msa_x)
        y_norm = apply_adaln_(self.norm_y1(y), shift_msa_y, scale_msa_y)
        attn_x, attn_y = self.attn(x_norm, y_norm, pos_img, pos_txt, attn_mask, transformer_options=transformer_options)
        x = torch.addcmul(x, gate_msa_x, attn_x)
        y = torch.addcmul(y, gate_msa_y, attn_y)

        x = torch.addcmul(x, gate_mlp_x, self.mlp_x(apply_adaln_(self.norm_x2(x), shift_mlp_x, scale_mlp_x)))
        y = torch.addcmul(y, gate_mlp_y, self.mlp_y(apply_adaln_(self.norm_y2(y), shift_mlp_y, scale_mlp_y)))
        return x, y


class PixDiT_T2I(nn.Module):
    """PixelDiT T2I model. Hardcoded for the released 1024px Stage-3 checkpoint
    (also runs at 512px when fed the appropriate latent size and flow_shift).

    Forward:
      x:        [B, 3, H, W] pixel-space input (no VAE)
      timesteps:[B] in [0, 1000] (ComfyUI flow sampling convention)
      context:  [B, Ltxt, 2304] Gemma-2-2b-it hidden states (chi_prompt prepended)
    Returns flow-matching velocity [B, 3, H, W].
    """
    def __init__(
        self,
        in_channels=3,
        num_groups=24,
        hidden_size=1536,
        pixel_hidden_size=16,
        pixel_attn_hidden_size=1152,
        pixel_num_groups=16,
        patch_depth=14,
        pixel_depth=2,
        patch_size=16,
        txt_embed_dim=2304,
        txt_max_length=300,
        use_text_rope=True,
        text_rope_theta=10000.0,
        image_model=None,
        dtype=None,
        device=None,
        operations=None,
        pixel_mlp_chunks=2,
    ):
        super().__init__()
        self.dtype = dtype
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.patch_depth = patch_depth
        self.pixel_depth = pixel_depth
        self.patch_size = patch_size
        self.pixel_hidden_size = pixel_hidden_size
        self.pixel_attn_hidden_size = pixel_attn_hidden_size
        self.pixel_num_groups = pixel_num_groups
        self.txt_embed_dim = txt_embed_dim
        self.txt_max_length = txt_max_length
        self.use_text_rope = use_text_rope
        self.text_rope_theta = text_rope_theta

        self.pixel_embedder = PixelTokenEmbedder(self.in_channels, self.pixel_hidden_size, dtype=dtype, device=device, operations=operations)
        self.s_embedder = PatchTokenEmbedder(self.in_channels * self.patch_size ** 2, self.hidden_size, bias=True, dtype=dtype, device=device, operations=operations)
        self.t_embedder = TimestepEmbedder(self.hidden_size, dtype=dtype, device=device, operations=operations, max_period=10)
        self.y_embedder = PatchTokenEmbedder(self.txt_embed_dim, self.hidden_size, bias=True, use_norm=True, dtype=dtype, device=device, operations=operations)
        self.y_pos_embedding = nn.Parameter(torch.empty(1, self.txt_max_length, self.hidden_size, dtype=dtype, device=device))

        self.patch_blocks = nn.ModuleList([
            MMDiTBlockT2I(self.hidden_size, self.num_groups,
                          dtype=dtype, device=device, operations=operations)
            for _ in range(self.patch_depth)
        ])
        self.pixel_blocks = nn.ModuleList([
            PiTBlock(
                self.pixel_hidden_size,
                self.hidden_size,
                patch_size=self.patch_size,
                num_heads=self.num_groups,
                attn_hidden_size=self.pixel_attn_hidden_size,
                attn_num_heads=self.pixel_num_groups,
                dtype=dtype, device=device, operations=operations,
                mlp_chunks=pixel_mlp_chunks,
            )
            for _ in range(self.pixel_depth)
        ])

        self.final_layer = FinalLayer(self.pixel_hidden_size, self.out_channels, dtype=dtype, device=device, operations=operations)

    def _fetch_patch_pos(self, height, width, device, dtype, **rope_opts):
        return precompute_freqs_cis_2d(self.hidden_size // self.num_groups, height, width, device=device, dtype=dtype, **rope_opts)

    def _fetch_text_pos(self, length, device, dtype):
        return rope(torch.arange(length, dtype=torch.float32, device=device).reshape(1, -1), self.hidden_size // self.num_groups, self.text_rope_theta).squeeze(0).to(dtype=dtype)

    def forward(self, x, timesteps, context=None, attention_mask=None, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward, self, comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options),
        ).execute(x, timesteps, context, attention_mask, transformer_options, **kwargs)

    def _pre_patch_block(self, s, i, **kwargs):
        """Hook for subclasses to inject per-block state into the patch stream (e.g. PiD's LQ gate)."""
        return s

    def _forward(self, x, timesteps, context=None, attention_mask=None, transformer_options={}, **kwargs):
        H_orig, W_orig = x.shape[2], x.shape[3]
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))
        B, _, H, W = x.shape
        Hs = H // self.patch_size
        Ws = W // self.patch_size
        L = Hs * Ws

        pos_img = self._fetch_patch_pos(Hs, Ws, x.device, x.dtype, **(transformer_options.get("rope_options") or {}))
        x_patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)

        t_emb = self.t_embedder(timesteps.view(-1), x.dtype).view(B, -1, self.hidden_size)

        if context is None or context.dim() != 3:
            raise ValueError("PixDiT_T2I requires context (text embeddings) of shape [B, L, D]")
        Ltxt = min(context.shape[1], self.txt_max_length)
        y = context[:, :Ltxt, :]
        y_emb = self.y_embedder(y).view(B, Ltxt, self.hidden_size)
        y_emb = y_emb + self.y_pos_embedding[:, :Ltxt, :].to(y_emb) # y_pos_embedding is a raw nn.Parameter

        condition = F.silu(t_emb)
        pos_txt = self._fetch_text_pos(Ltxt, x.device, x.dtype) if self.use_text_rope else None

        s = self.s_embedder(x_patches)
        for i, blk in enumerate(self.patch_blocks):
            s = self._pre_patch_block(s, i, **kwargs)
            s, y_emb = blk(s, y_emb, condition, pos_img, pos_txt, None, transformer_options=transformer_options)
        s = F.silu(t_emb + s)

        s_cond = s.view(B * L, self.hidden_size)
        x_pixels = self.pixel_embedder(x, patch_size=self.patch_size)
        for blk in self.pixel_blocks:
            x_pixels = blk(x_pixels, s_cond, H, W, self.patch_size, mask=None, transformer_options=transformer_options)

        x_pixels = self.final_layer(x_pixels)
        C_out = self.out_channels
        P2 = self.patch_size * self.patch_size
        x_pixels = x_pixels.view(B, L, P2, C_out).permute(0, 3, 2, 1).reshape(B, C_out * P2, L)
        out = F.fold(x_pixels, (H, W), kernel_size=self.patch_size, stride=self.patch_size)
        return out[:, :, :H_orig, :W_orig]
