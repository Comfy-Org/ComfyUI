# Uni3C controlnet for Wan 2.1: https://github.com/ewrfcas/Uni3C
# Converted from the original diffusers based implementation.
import torch
import torch.nn as nn

from comfy.ldm.flux.layers import EmbedND
from .model import WanFeedForward, WanSelfAttention


class Uni3CLayerNormZero(nn.Module):
    def __init__(
        self,
        conditioning_dim,
        embedding_dim,
        eps=1e-5,
        device=None, dtype=None, operations=None
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = operations.Linear(conditioning_dim, 3 * embedding_dim, device=device, dtype=dtype)
        self.norm = operations.LayerNorm(embedding_dim, eps=eps, elementwise_affine=True, device=device, dtype=dtype)

    def forward(self, x, temb):
        shift, scale, gate = self.linear(self.silu(temb)).chunk(3, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x, gate[:, None, :]


class Uni3CAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        time_embed_dim=5120,
        eps=1e-6,
        device=None, dtype=None, operations=None
    ):
        super().__init__()
        operation_settings = {"operations": operations, "device": device, "dtype": dtype}
        self.norm1 = Uni3CLayerNormZero(time_embed_dim, dim, device=device, dtype=dtype, operations=operations)
        self.self_attn = WanSelfAttention(dim, num_heads, qk_norm=True, eps=eps, operation_settings=operation_settings)
        self.norm2 = Uni3CLayerNormZero(time_embed_dim, dim, device=device, dtype=dtype, operations=operations)
        self.ffn = WanFeedForward(
            operations.Linear(dim, ffn_dim, device=device, dtype=dtype), nn.GELU(approximate='tanh'),
            operations.Linear(ffn_dim, dim, device=device, dtype=dtype))

    def forward(self, x, temb, freqs):
        norm_x, gate_msa = self.norm1(x, temb)
        x = x + gate_msa * self.self_attn(norm_x, freqs)
        norm_x, gate_ff = self.norm2(x, temb)
        x = x + gate_ff * self.ffn(norm_x)
        return x


class MaskCamEmbed(nn.Module):
    def __init__(
        self,
        add_channels=7,
        mid_channels=256,
        conv_out_dim=5120,
        device=None, dtype=None, operations=None
    ):
        super().__init__()
        self.mask_padding = [0, 0, 0, 0, 3, 0]  # first frame conditioning
        self.mask_proj = nn.Sequential(
            operations.Conv3d(add_channels, mid_channels, kernel_size=(4, 8, 8), stride=(4, 8, 8), device=device, dtype=dtype),
            operations.GroupNorm(mid_channels // 8, mid_channels, device=device, dtype=dtype),
            nn.SiLU())
        self.mask_zero_proj = operations.Conv3d(mid_channels, conv_out_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2), device=device, dtype=dtype)

    def forward(self, add_inputs):
        add_padded = torch.nn.functional.pad(add_inputs, self.mask_padding, mode="constant", value=0)
        add_embeds = self.mask_proj(add_padded)
        add_embeds = self.mask_zero_proj(add_embeds)
        add_embeds = add_embeds.flatten(2).transpose(1, 2)
        return add_embeds


class WanUni3CControlnet(nn.Module):
    def __init__(
        self,
        in_channels=36,
        conv_out_dim=5120,
        dim=1024,
        ffn_dim=8192,
        num_heads=16,
        num_layers=20,
        time_embed_dim=5120,
        out_proj_dim=5120,
        add_channels=7,
        mid_channels=256,
        device=None, dtype=None, operations=None
    ):
        super().__init__()
        patch_size = (1, 2, 2)
        self.num_layers = num_layers

        self.controlnet_patch_embedding = operations.Conv3d(
            in_channels, conv_out_dim, kernel_size=patch_size, stride=patch_size, device=device, dtype=torch.float32)
        self.controlnet_mask_embedding = MaskCamEmbed(add_channels, mid_channels, conv_out_dim, device=device, dtype=dtype, operations=operations)

        if conv_out_dim != dim:
            self.proj_in = operations.Linear(conv_out_dim, dim, device=device, dtype=dtype)
        else:
            self.proj_in = nn.Identity()

        self.controlnet_blocks = nn.ModuleList([
            Uni3CAttentionBlock(dim, ffn_dim, num_heads, time_embed_dim, device=device, dtype=dtype, operations=operations)
            for _ in range(num_layers)])
        self.proj_out = nn.ModuleList([
            operations.Linear(dim, out_proj_dim, device=device, dtype=dtype)
            for _ in range(num_layers)])

        head_dim = dim // num_heads
        self.rope_embedder = EmbedND(dim=head_dim, theta=10000.0, axes_dim=[head_dim - 4 * (head_dim // 6), 2 * (head_dim // 6), 2 * (head_dim // 6)])

    def rope_encode(self, t_len, h_len, w_len, device=None, dtype=None):
        img_ids = torch.zeros((t_len, h_len, w_len, 3), device=device, dtype=dtype)
        img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.arange(t_len, device=device, dtype=dtype).reshape(-1, 1, 1)
        img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.arange(h_len, device=device, dtype=dtype).reshape(1, -1, 1)
        img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.arange(w_len, device=device, dtype=dtype).reshape(1, 1, -1)
        img_ids = img_ids.reshape(1, -1, img_ids.shape[-1])
        freqs = self.rope_embedder(img_ids).movedim(1, 2)
        return freqs

    def process_input(self, control_input, render_mask=None, camera_embedding=None):
        # render_mask/camera_embedding are the checkpoint's extra conditioning path, not wired up yet
        hidden = self.controlnet_patch_embedding(control_input.float()).to(control_input.dtype)
        t_len, h_len, w_len = hidden.shape[2:]
        freqs = self.rope_encode(t_len, h_len, w_len, device=hidden.device, dtype=hidden.dtype)
        hidden = hidden.flatten(2).transpose(1, 2)

        add_inputs = None
        if camera_embedding is not None and render_mask is not None:
            add_inputs = torch.cat([render_mask, camera_embedding], dim=1)
        elif render_mask is not None:
            add_inputs = render_mask

        if add_inputs is not None:
            hidden = hidden + self.controlnet_mask_embedding(add_inputs.to(hidden.dtype))

        hidden = self.proj_in(hidden)
        return hidden, freqs

    def forward_block(self, block_index, hidden, temb, freqs):
        hidden = self.controlnet_blocks[block_index](hidden, temb, freqs)
        residual = self.proj_out[block_index](hidden)
        return hidden, residual
