# Code from: https://github.com/Alpha-VLLM/Lumina-Image-2.0/blob/main/models/model.py

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import comfy.ldm.common_dit

from comfy.ldm.modules.diffusionmodules.mmdit import TimestepEmbedder
from comfy.ldm.modules.attention import optimized_attention_masked
from comfy.ldm.flux.layers import EmbedND
from comfy.ldm.flux.math import apply_rope
import comfy.patcher_extension
import comfy.utils
from comfy.ldm.chroma_radiance.layers import NerfEmbedder


def invert_slices(slices, length):
    sorted_slices = sorted(slices)
    result = []
    current = 0

    for start, end in sorted_slices:
        if current < start:
            result.append((current, start))
        current = max(current, end)

    if current < length:
        result.append((current, length))

    return result


def modulate(x, scale, timestep_zero_index=None):
    if timestep_zero_index is None:
        return x * (1 + scale.unsqueeze(1))
    else:
        scale = (1 + scale.unsqueeze(1))
        actual_batch = scale.size(0) // 2
        slices = timestep_zero_index
        invert = invert_slices(timestep_zero_index, x.shape[1])
        for s in slices:
            x[:, s[0]:s[1]] *= scale[actual_batch:]
        for s in invert:
            x[:, s[0]:s[1]] *= scale[:actual_batch]
        return x


def apply_gate(gate, x, timestep_zero_index=None):
    if timestep_zero_index is None:
        return gate * x
    else:
        actual_batch = gate.size(0) // 2

        slices = timestep_zero_index
        invert = invert_slices(timestep_zero_index, x.shape[1])
        for s in slices:
            x[:, s[0]:s[1]] *= gate[actual_batch:]
        for s in invert:
            x[:, s[0]:s[1]] *= gate[:actual_batch]
        return x

#############################################################################
#                               Core NextDiT Model                              #
#############################################################################

def clamp_fp16(x):
    if x.dtype == torch.float16:
        return torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
    return x

class JointAttention(nn.Module):
    """Multi-head attention module."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int],
        qk_norm: bool,
        out_bias: bool = False,
        operation_settings={},
    ):
        """
        Initialize the Attention module.

        Args:
            dim (int): Number of input dimensions.
            n_heads (int): Number of heads.
            n_kv_heads (Optional[int]): Number of kv heads, if using GQA.

        """
        super().__init__()
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_local_heads = n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads

        self.qkv = operation_settings.get("operations").Linear(
            dim,
            (n_heads + self.n_kv_heads + self.n_kv_heads) * self.head_dim,
            bias=False,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )
        self.out = operation_settings.get("operations").Linear(
            n_heads * self.head_dim,
            dim,
            bias=out_bias,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )

        if qk_norm:
            self.q_norm = operation_settings.get("operations").RMSNorm(self.head_dim, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
            self.k_norm = operation_settings.get("operations").RMSNorm(self.head_dim, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        else:
            self.q_norm = self.k_norm = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        transformer_options={},
    ) -> torch.Tensor:
        """

        Args:
            x:
            x_mask:
            freqs_cis:

        Returns:

        """
        bsz, seqlen, _ = x.shape

        xq, xk, xv = torch.split(
            self.qkv(x),
            [
                self.n_local_heads * self.head_dim,
                self.n_local_kv_heads * self.head_dim,
                self.n_local_kv_heads * self.head_dim,
            ],
            dim=-1,
        )
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq, xk = apply_rope(xq, xk, freqs_cis)

        n_rep = self.n_local_heads // self.n_local_kv_heads
        if n_rep >= 1:
            xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
        output = optimized_attention_masked(xq.movedim(1, 2), xk.movedim(1, 2), xv.movedim(1, 2), self.n_local_heads, x_mask, skip_reshape=True, transformer_options=transformer_options)

        return self.out(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        operation_settings={},
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple
                of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden
                dimension. Defaults to None.

        """
        super().__init__()
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = operation_settings.get("operations").Linear(
            dim,
            hidden_dim,
            bias=False,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )
        self.w2 = operation_settings.get("operations").Linear(
            hidden_dim,
            dim,
            bias=False,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )
        self.w3 = operation_settings.get("operations").Linear(
            dim,
            hidden_dim,
            bias=False,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )

    # @torch.compile
    def _forward_silu_gating(self, x1, x3):
        return clamp_fp16(F.silu(x1) * x3)

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


class JointTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
        z_image_modulation=False,
        attn_out_bias=False,
        operation_settings={},
    ) -> None:
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            dim (int): Embedding dimension of the input features.
            n_heads (int): Number of attention heads.
            n_kv_heads (Optional[int]): Number of attention heads in key and
                value features (if using GQA), or set to None for the same as
                query.
            multiple_of (int):
            ffn_dim_multiplier (float):
            norm_eps (float):

        """
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = JointAttention(dim, n_heads, n_kv_heads, qk_norm, out_bias=attn_out_bias, operation_settings=operation_settings)
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            operation_settings=operation_settings,
        )
        self.layer_id = layer_id
        self.attention_norm1 = operation_settings.get("operations").RMSNorm(dim, eps=norm_eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.ffn_norm1 = operation_settings.get("operations").RMSNorm(dim, eps=norm_eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))

        self.attention_norm2 = operation_settings.get("operations").RMSNorm(dim, eps=norm_eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.ffn_norm2 = operation_settings.get("operations").RMSNorm(dim, eps=norm_eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))

        self.modulation = modulation
        if modulation:
            if z_image_modulation:
                self.adaLN_modulation = nn.Sequential(
                    operation_settings.get("operations").Linear(
                        min(dim, 256),
                        4 * dim,
                        bias=True,
                        device=operation_settings.get("device"),
                        dtype=operation_settings.get("dtype"),
                    ),
                )
            else:
                self.adaLN_modulation = nn.Sequential(
                    nn.SiLU(),
                    operation_settings.get("operations").Linear(
                        min(dim, 1024),
                        4 * dim,
                        bias=True,
                        device=operation_settings.get("device"),
                        dtype=operation_settings.get("dtype"),
                    ),
                )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor]=None,
        timestep_zero_index=None,
        transformer_options={},
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        Returns:
            torch.Tensor: Output tensor after applying attention and
                feedforward layers.

        """
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(4, dim=1)

            x = x + apply_gate(gate_msa.unsqueeze(1).tanh(), self.attention_norm2(
                clamp_fp16(self.attention(
                    modulate(self.attention_norm1(x), scale_msa, timestep_zero_index=timestep_zero_index),
                    x_mask,
                    freqs_cis,
                    transformer_options=transformer_options,
                ))), timestep_zero_index=timestep_zero_index
            )
            x = x + apply_gate(gate_mlp.unsqueeze(1).tanh(), self.ffn_norm2(
                clamp_fp16(self.feed_forward(
                    modulate(self.ffn_norm1(x), scale_mlp, timestep_zero_index=timestep_zero_index),
                ))), timestep_zero_index=timestep_zero_index
            )
        else:
            assert adaln_input is None
            x = x + self.attention_norm2(
                clamp_fp16(self.attention(
                    self.attention_norm1(x),
                    x_mask,
                    freqs_cis,
                    transformer_options=transformer_options,
                ))
            )
            x = x + self.ffn_norm2(
                self.feed_forward(
                    self.ffn_norm1(x),
                )
            )
        return x


class FinalLayer(nn.Module):
    """
    The final layer of NextDiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels, z_image_modulation=False, operation_settings={}):
        super().__init__()
        self.norm_final = operation_settings.get("operations").LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )
        self.linear = operation_settings.get("operations").Linear(
            hidden_size,
            patch_size * patch_size * out_channels,
            bias=True,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )

        if z_image_modulation:
            min_mod = 256
        else:
            min_mod = 1024

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            operation_settings.get("operations").Linear(
                min(hidden_size, min_mod),
                hidden_size,
                bias=True,
                device=operation_settings.get("device"),
                dtype=operation_settings.get("dtype"),
            ),
        )

    def forward(self, x, c, timestep_zero_index=None):
        scale = self.adaLN_modulation(c)
        x = modulate(self.norm_final(x), scale, timestep_zero_index=timestep_zero_index)
        x = self.linear(x)
        return x


def pad_zimage(feats, pad_token, pad_tokens_multiple):
    pad_extra = (-feats.shape[1]) % pad_tokens_multiple
    return torch.cat((feats, pad_token.to(device=feats.device, dtype=feats.dtype, copy=True).unsqueeze(0).repeat(feats.shape[0], pad_extra, 1)), dim=1), pad_extra


def pos_ids_x(start_t, H_tokens, W_tokens, batch_size, device, transformer_options={}):
    rope_options = transformer_options.get("rope_options", None)
    h_scale = 1.0
    w_scale = 1.0
    h_start = 0
    w_start = 0
    if rope_options is not None:
        h_scale = rope_options.get("scale_y", 1.0)
        w_scale = rope_options.get("scale_x", 1.0)

        h_start = rope_options.get("shift_y", 0.0)
        w_start = rope_options.get("shift_x", 0.0)
    x_pos_ids = torch.zeros((batch_size, H_tokens * W_tokens, 3), dtype=torch.float32, device=device)
    x_pos_ids[:, :, 0] = start_t
    x_pos_ids[:, :, 1] = (torch.arange(H_tokens, dtype=torch.float32, device=device) * h_scale + h_start).view(-1, 1).repeat(1, W_tokens).flatten()
    x_pos_ids[:, :, 2] = (torch.arange(W_tokens, dtype=torch.float32, device=device) * w_scale + w_start).view(1, -1).repeat(H_tokens, 1).flatten()
    return x_pos_ids


class NextDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 4,
        dim: int = 4096,
        n_layers: int = 32,
        n_refiner_layers: int = 2,
        n_heads: int = 32,
        n_kv_heads: Optional[int] = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: float = 4.0,
        norm_eps: float = 1e-5,
        qk_norm: bool = False,
        cap_feat_dim: int = 5120,
        axes_dims: List[int] = (16, 56, 56),
        axes_lens: List[int] = (1, 512, 512),
        rope_theta=10000.0,
        z_image_modulation=False,
        time_scale=1.0,
        pad_tokens_multiple=None,
        clip_text_dim=None,
        siglip_feat_dim=None,
        image_model=None,
        device=None,
        dtype=None,
        operations=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        operation_settings = {"operations": operations, "device": device, "dtype": dtype}
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.time_scale = time_scale
        self.pad_tokens_multiple = pad_tokens_multiple

        self.x_embedder = operation_settings.get("operations").Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=dim,
            bias=True,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )

        self.noise_refiner = nn.ModuleList(
            [
                JointTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                    z_image_modulation=z_image_modulation,
                    operation_settings=operation_settings,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                JointTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                    operation_settings=operation_settings,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )

        self.t_embedder = TimestepEmbedder(min(dim, 1024), output_size=256 if z_image_modulation else None, **operation_settings)
        self.cap_embedder = nn.Sequential(
            operation_settings.get("operations").RMSNorm(cap_feat_dim, eps=norm_eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")),
            operation_settings.get("operations").Linear(
                cap_feat_dim,
                dim,
                bias=True,
                device=operation_settings.get("device"),
                dtype=operation_settings.get("dtype"),
            ),
        )

        self.clip_text_pooled_proj = None

        if clip_text_dim is not None:
            self.clip_text_dim = clip_text_dim
            self.clip_text_pooled_proj = nn.Sequential(
                operation_settings.get("operations").RMSNorm(clip_text_dim, eps=norm_eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")),
                operation_settings.get("operations").Linear(
                    clip_text_dim,
                    clip_text_dim,
                    bias=True,
                    device=operation_settings.get("device"),
                    dtype=operation_settings.get("dtype"),
                ),
            )
            self.time_text_embed = nn.Sequential(
                nn.SiLU(),
                operation_settings.get("operations").Linear(
                    min(dim, 1024) + clip_text_dim,
                    min(dim, 1024),
                    bias=True,
                    device=operation_settings.get("device"),
                    dtype=operation_settings.get("dtype"),
                ),
            )

        self.layers = nn.ModuleList(
            [
                JointTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    z_image_modulation=z_image_modulation,
                    attn_out_bias=False,
                    operation_settings=operation_settings,
                )
                for layer_id in range(n_layers)
            ]
        )

        if siglip_feat_dim is not None:
            self.siglip_embedder = nn.Sequential(
                operation_settings.get("operations").RMSNorm(siglip_feat_dim, eps=norm_eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")),
                operation_settings.get("operations").Linear(
                    siglip_feat_dim,
                    dim,
                    bias=True,
                    device=operation_settings.get("device"),
                    dtype=operation_settings.get("dtype"),
                ),
            )
            self.siglip_refiner = nn.ModuleList(
                [
                    JointTransformerBlock(
                        layer_id,
                        dim,
                        n_heads,
                        n_kv_heads,
                        multiple_of,
                        ffn_dim_multiplier,
                        norm_eps,
                        qk_norm,
                        modulation=False,
                        operation_settings=operation_settings,
                    )
                    for layer_id in range(n_refiner_layers)
                ]
            )
            self.siglip_pad_token = nn.Parameter(torch.empty((1, dim), device=device, dtype=dtype))
        else:
            self.siglip_embedder = None
            self.siglip_refiner = None
            self.siglip_pad_token = None

        # This norm final is in the lumina 2.0 code but isn't actually used for anything.
        # self.norm_final = operation_settings.get("operations").RMSNorm(dim, eps=norm_eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels, z_image_modulation=z_image_modulation, operation_settings=operation_settings)

        if self.pad_tokens_multiple is not None:
            self.x_pad_token = nn.Parameter(torch.empty((1, dim), device=device, dtype=dtype))
            self.cap_pad_token = nn.Parameter(torch.empty((1, dim), device=device, dtype=dtype))

        assert (dim // n_heads) == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.rope_embedder = EmbedND(dim=dim // n_heads, theta=rope_theta, axes_dim=axes_dims)
        self.dim = dim
        self.n_heads = n_heads

    def unpatchify(
        self, x: torch.Tensor, img_size: List[Tuple[int, int]], cap_size: List[int], return_tensor=False
    ) -> List[torch.Tensor]:
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        pH = pW = self.patch_size
        imgs = []
        for i in range(x.size(0)):
            H, W = img_size[i]
            begin = cap_size[i]
            end = begin + (H // pH) * (W // pW)
            imgs.append(
                x[i][begin:end]
                .view(H // pH, W // pW, pH, pW, self.out_channels)
                .permute(4, 0, 2, 1, 3)
                .flatten(3, 4)
                .flatten(1, 2)
            )

        if return_tensor:
            imgs = torch.stack(imgs, dim=0)
        return imgs

    def embed_cap(self, cap_feats=None, offset=0, bsz=1, device=None, dtype=None):
        if cap_feats is not None:
            cap_feats = self.cap_embedder(cap_feats)
            cap_feats_len = cap_feats.shape[1]
            if self.pad_tokens_multiple is not None:
                cap_feats, _ = pad_zimage(cap_feats, self.cap_pad_token, self.pad_tokens_multiple)
        else:
            cap_feats_len = 0
            cap_feats = self.cap_pad_token.to(device=device, dtype=dtype, copy=True).unsqueeze(0).repeat(bsz, self.pad_tokens_multiple, 1)

        cap_pos_ids = torch.zeros(bsz, cap_feats.shape[1], 3, dtype=torch.float32, device=device)
        cap_pos_ids[:, :, 0] = torch.arange(cap_feats.shape[1], dtype=torch.float32, device=device) + 1.0 + offset
        embeds = (cap_feats,)
        freqs_cis = (self.rope_embedder(cap_pos_ids).movedim(1, 2),)
        return embeds, freqs_cis, cap_feats_len

    def embed_all(self, x, cap_feats=None, siglip_feats=None, offset=0, omni=False, transformer_options={}):
        bsz = 1
        pH = pW = self.patch_size
        device = x.device
        embeds, freqs_cis, cap_feats_len = self.embed_cap(cap_feats, offset=offset, bsz=bsz, device=device, dtype=x.dtype)

        if (not omni) or self.siglip_embedder is None:
            cap_feats_len = embeds[0].shape[1] + offset
            embeds += (None,)
            freqs_cis += (None,)
        else:
            cap_feats_len += offset
            if siglip_feats is not None:
                b, h, w, c = siglip_feats.shape
                siglip_feats = siglip_feats.permute(0, 3, 1, 2).reshape(b, h * w, c)
                siglip_feats = self.siglip_embedder(siglip_feats)
                siglip_pos_ids = torch.zeros((bsz, siglip_feats.shape[1], 3), dtype=torch.float32, device=device)
                siglip_pos_ids[:, :, 0] = cap_feats_len + 2
                siglip_pos_ids[:, :, 1] = (torch.linspace(0, h * 8 - 1, steps=h, dtype=torch.float32, device=device).floor()).view(-1, 1).repeat(1, w).flatten()
                siglip_pos_ids[:, :, 2] = (torch.linspace(0, w * 8 - 1, steps=w, dtype=torch.float32, device=device).floor()).view(1, -1).repeat(h, 1).flatten()
                if self.siglip_pad_token is not None:
                    siglip_feats, pad_extra = pad_zimage(siglip_feats, self.siglip_pad_token, self.pad_tokens_multiple)  # TODO: double check
                    siglip_pos_ids = torch.nn.functional.pad(siglip_pos_ids, (0, 0, 0, pad_extra))
            else:
                if self.siglip_pad_token is not None:
                    siglip_feats = self.siglip_pad_token.to(device=device, dtype=x.dtype, copy=True).unsqueeze(0).repeat(bsz, self.pad_tokens_multiple, 1)
                    siglip_pos_ids = torch.zeros((bsz, siglip_feats.shape[1], 3), dtype=torch.float32, device=device)

            if siglip_feats is None:
                embeds += (None,)
                freqs_cis += (None,)
            else:
                embeds += (siglip_feats,)
                freqs_cis += (self.rope_embedder(siglip_pos_ids).movedim(1, 2),)

        B, C, H, W = x.shape
        x = self.x_embedder(x.view(B, C, H // pH, pH, W // pW, pW).permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2))
        x_pos_ids = pos_ids_x(cap_feats_len + 1, H // pH, W // pW, bsz, device, transformer_options=transformer_options)
        if self.pad_tokens_multiple is not None:
            x, pad_extra = pad_zimage(x, self.x_pad_token, self.pad_tokens_multiple)
            x_pos_ids = torch.nn.functional.pad(x_pos_ids, (0, 0, 0, pad_extra))

        embeds += (x,)
        freqs_cis += (self.rope_embedder(x_pos_ids).movedim(1, 2),)
        return embeds, freqs_cis, cap_feats_len + len(freqs_cis) - 1


    def patchify_and_embed(
        self, x: torch.Tensor, cap_feats: torch.Tensor, cap_mask: torch.Tensor, t: torch.Tensor, num_tokens, ref_latents=[], ref_contexts=[], siglip_feats=[], transformer_options={}
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]], List[int], torch.Tensor]:
        bsz = x.shape[0]
        cap_mask = None  # TODO?
        main_siglip = None
        orig_x = x

        embeds = ([], [], [])
        freqs_cis = ([], [], [])
        leftover_cap = []

        start_t = 0
        omni = len(ref_latents) > 0
        if omni:
            for i, ref in enumerate(ref_latents):
                if i < len(ref_contexts):
                    ref_con = ref_contexts[i]
                else:
                    ref_con = None
                if i < len(siglip_feats):
                    sig_feat = siglip_feats[i]
                else:
                    sig_feat = None

                out = self.embed_all(ref, ref_con, sig_feat, offset=start_t, omni=omni, transformer_options=transformer_options)
                for i, e in enumerate(out[0]):
                    if e is not None:
                        embeds[i].append(comfy.utils.repeat_to_batch_size(e, bsz))
                        freqs_cis[i].append(out[1][i])
                start_t = out[2]
            leftover_cap = ref_contexts[len(ref_latents):]

        H, W = x.shape[-2], x.shape[-1]
        img_sizes = [(H, W)] * bsz
        out = self.embed_all(x, cap_feats, main_siglip, offset=start_t, omni=omni, transformer_options=transformer_options)
        img_len = out[0][-1].shape[1]
        cap_len = out[0][0].shape[1]
        for i, e in enumerate(out[0]):
            if e is not None:
                e = comfy.utils.repeat_to_batch_size(e, bsz)
                embeds[i].append(e)
                freqs_cis[i].append(out[1][i])
        start_t = out[2]

        for cap in leftover_cap:
            out = self.embed_cap(cap, offset=start_t, bsz=bsz, device=x.device, dtype=x.dtype)
            cap_len += out[0][0].shape[1]
            embeds[0].append(comfy.utils.repeat_to_batch_size(out[0][0], bsz))
            freqs_cis[0].append(out[1][0])
            start_t += out[2]

        patches = transformer_options.get("patches", {})

        # refine context
        cap_feats = torch.cat(embeds[0], dim=1)
        cap_freqs_cis = torch.cat(freqs_cis[0], dim=1)
        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_mask, cap_freqs_cis, transformer_options=transformer_options)

        feats = (cap_feats,)
        fc = (cap_freqs_cis,)

        if omni and len(embeds[1]) > 0:
            siglip_mask = None
            siglip_feats_combined = torch.cat(embeds[1], dim=1)
            siglip_feats_freqs_cis = torch.cat(freqs_cis[1], dim=1)
            if self.siglip_refiner is not None:
                for layer in self.siglip_refiner:
                    siglip_feats_combined = layer(siglip_feats_combined, siglip_mask, siglip_feats_freqs_cis, transformer_options=transformer_options)
            feats += (siglip_feats_combined,)
            fc += (siglip_feats_freqs_cis,)

        padded_img_mask = None
        x = torch.cat(embeds[-1], dim=1)
        fc_x = torch.cat(freqs_cis[-1], dim=1)
        if omni:
            timestep_zero_index = [(x.shape[1] - img_len, x.shape[1])]
        else:
            timestep_zero_index = None

        x_input = x
        for i, layer in enumerate(self.noise_refiner):
            x = layer(x, padded_img_mask, fc_x, t, timestep_zero_index=timestep_zero_index, transformer_options=transformer_options)
            if "noise_refiner" in patches:
                for p in patches["noise_refiner"]:
                    out = p({"img": x, "img_input": x_input, "txt": cap_feats, "pe": fc_x, "vec": t, "x": orig_x, "block_index": i, "transformer_options": transformer_options, "block_type": "noise_refiner"})
                    if "img" in out:
                        x = out["img"]

        padded_full_embed = torch.cat(feats + (x,), dim=1)
        if timestep_zero_index is not None:
            ind = padded_full_embed.shape[1] - x.shape[1]
            timestep_zero_index = [(ind + x.shape[1] - img_len, ind + x.shape[1])]
            timestep_zero_index.append((feats[0].shape[1] - cap_len, feats[0].shape[1]))

        mask = None
        l_effective_cap_len = [padded_full_embed.shape[1] - img_len] * bsz
        return padded_full_embed, mask, img_sizes, l_effective_cap_len, torch.cat(fc + (fc_x,), dim=1), timestep_zero_index

    def forward(self, x, timesteps, context, num_tokens, attention_mask=None, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, kwargs.get("transformer_options", {}))
        ).execute(x, timesteps, context, num_tokens, attention_mask, **kwargs)

    # def forward(self, x, t, cap_feats, cap_mask):
    def _forward(self, x, timesteps, context, num_tokens, attention_mask=None, ref_latents=[], ref_contexts=[], siglip_feats=[], transformer_options={}, **kwargs):
        omni = len(ref_latents) > 0
        if omni:
            timesteps = torch.cat([timesteps * 0, timesteps], dim=0)

        t = 1.0 - timesteps
        cap_feats = context
        cap_mask = attention_mask
        bs, c, h, w = x.shape
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))
        """
        Forward pass of NextDiT.
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of text tokens/features
        """

        t = self.t_embedder(t * self.time_scale, dtype=x.dtype)  # (N, D)
        adaln_input = t

        if self.clip_text_pooled_proj is not None:
            pooled = kwargs.get("clip_text_pooled", None)
            if pooled is not None:
                pooled = self.clip_text_pooled_proj(pooled)
            else:
                pooled = torch.zeros((x.shape[0], self.clip_text_dim), device=x.device, dtype=x.dtype)

            adaln_input = self.time_text_embed(torch.cat((t, pooled), dim=-1))

        patches = transformer_options.get("patches", {})
        x_is_tensor = isinstance(x, torch.Tensor)
        img, mask, img_size, cap_size, freqs_cis, timestep_zero_index = self.patchify_and_embed(x, cap_feats, cap_mask, adaln_input, num_tokens, ref_latents=ref_latents, ref_contexts=ref_contexts, siglip_feats=siglip_feats, transformer_options=transformer_options)
        freqs_cis = freqs_cis.to(img.device)

        transformer_options["total_blocks"] = len(self.layers)
        transformer_options["block_type"] = "double"
        img_input = img
        for i, layer in enumerate(self.layers):
            transformer_options["block_index"] = i
            img = layer(img, mask, freqs_cis, adaln_input, timestep_zero_index=timestep_zero_index, transformer_options=transformer_options)
            if "double_block" in patches:
                for p in patches["double_block"]:
                    out = p({"img": img[:, cap_size[0]:], "img_input": img_input[:, cap_size[0]:], "txt": img[:, :cap_size[0]], "pe": freqs_cis[:, cap_size[0]:], "vec": adaln_input, "x": x, "block_index": i, "transformer_options": transformer_options})
                    if "img" in out:
                        img[:, cap_size[0]:] = out["img"]
                    if "txt" in out:
                        img[:, :cap_size[0]] = out["txt"]

        img = self.final_layer(img, adaln_input, timestep_zero_index=timestep_zero_index)
        img = self.unpatchify(img, img_size, cap_size, return_tensor=x_is_tensor)[:, :, :h, :w]
        return -img


#############################################################################
#                        Pixel Space Decoder Components                     #
#############################################################################

def _modulate_shift_scale(x, shift, scale):
    return x * (1 + scale) + shift


class PixelResBlock(nn.Module):
    """
    Residual block with AdaLN modulation, zero-initialised so it starts as
    an identity at the beginning of training.
    """

    def __init__(self, channels: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.in_ln = operations.LayerNorm(channels, eps=1e-6, dtype=dtype, device=device)
        self.mlp = nn.Sequential(
            operations.Linear(channels, channels, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Linear(channels, channels, bias=True, dtype=dtype, device=device),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            operations.Linear(channels, 3 * channels, bias=True, dtype=dtype, device=device),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = _modulate_shift_scale(self.in_ln(x), shift, scale)
        h = self.mlp(h)
        return x + gate * h


class DCTFinalLayer(nn.Module):
    """Zero-initialised output projection (adopted from DiT)."""

    def __init__(self, model_channels: int, out_channels: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm_final = operations.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.linear = operations.Linear(model_channels, out_channels, bias=True, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm_final(x))


class SimpleMLPAdaLN(nn.Module):
    """
    Small MLP decoder head for the pixel-space variant.

    Takes per-patch pixel values and a per-patch conditioning vector from the
    transformer backbone and predicts the denoised pixel values.

    x : [B*N, P^2, C]   – noisy pixel values per patch position
    c : [B*N, dim]       – backbone hidden state per patch (conditioning)
    → [B*N, P^2, C]
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        z_channels: int,
        num_res_blocks: int,
        max_freqs: int = 8,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.dtype = dtype

        # Project backbone hidden state → per-patch conditioning
        self.cond_embed = operations.Linear(z_channels, model_channels, dtype=dtype, device=device)

        # Input projection with DCT positional encoding
        self.input_embedder = NerfEmbedder(
            in_channels=in_channels,
            hidden_size_input=model_channels,
            max_freqs=max_freqs,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            PixelResBlock(model_channels, dtype=dtype, device=device, operations=operations) for _ in range(num_res_blocks)
        ])

        # Output projection
        self.final_layer = DCTFinalLayer(model_channels, out_channels, dtype=dtype, device=device, operations=operations)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: [B*N, 1, P^2*C],  c: [B*N, dim]
        original_dtype = x.dtype
        weight_dtype = self.cond_embed.weight.dtype if hasattr(self.cond_embed, "weight") and self.cond_embed.weight is not None else (self.dtype or x.dtype)
        x = self.input_embedder(x)                                   # [B*N, 1, model_channels]
        y = self.cond_embed(c.to(weight_dtype)).unsqueeze(1)         # [B*N, 1, model_channels]
        x = x.to(weight_dtype)
        for block in self.res_blocks:
            x = block(x, y)
        return self.final_layer(x).to(original_dtype)                # [B*N, 1, P^2*C]


#############################################################################
#                          NextDiT – Pixel Space                            #
#############################################################################

class NextDiTPixelSpace(NextDiT):
    """
    Pixel-space variant of NextDiT.

    Identical transformer backbone to NextDiT, but the output head is replaced
    with a small MLP decoder (SimpleMLPAdaLN) that operates on raw pixel values
    per patch rather than a single affine projection.

    Key differences vs NextDiT:
      • ``final_layer`` is removed; ``dec_net`` (SimpleMLPAdaLN) is used instead.
      • ``_forward`` stores the raw patchified pixel values before the backbone
        embedding and feeds them to ``dec_net`` together with the per-patch
        backbone hidden states.
      • Supports optional x0 prediction via ``use_x0``.
    """

    def __init__(
        self,
        # decoder-specific
        decoder_hidden_size: int = 3840,
        decoder_num_res_blocks: int = 4,
        decoder_max_freqs: int = 8,
        decoder_in_channels: int = None,  # full flattened patch size (patch_size^2 * in_channels)
        use_x0: bool = False,
        # all NextDiT args forwarded unchanged
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Remove the latent-space final layer – not used in pixel space
        del self.final_layer

        patch_size = kwargs.get("patch_size", 2)
        in_channels = kwargs.get("in_channels", 4)
        dim = kwargs.get("dim", 4096)

        # decoder_in_channels is the full flattened patch: patch_size^2 * in_channels
        dec_in_ch = decoder_in_channels if decoder_in_channels is not None else patch_size ** 2 * in_channels

        self.dec_net = SimpleMLPAdaLN(
            in_channels=dec_in_ch,
            model_channels=decoder_hidden_size,
            out_channels=dec_in_ch,
            z_channels=dim,
            num_res_blocks=decoder_num_res_blocks,
            max_freqs=decoder_max_freqs,
            dtype=kwargs.get("dtype"),
            device=kwargs.get("device"),
            operations=kwargs.get("operations"),
        )

        if use_x0:
            self.register_buffer("__x0__", torch.tensor([]))

    # ------------------------------------------------------------------
    # Forward — mirrors NextDiT._forward exactly, replacing final_layer
    # with the pixel-space dec_net decoder.
    # ------------------------------------------------------------------
    def _forward(self, x, timesteps, context, num_tokens, attention_mask=None, ref_latents=[], ref_contexts=[], siglip_feats=[], transformer_options={}, **kwargs):
        omni = len(ref_latents) > 0
        if omni:
            timesteps = torch.cat([timesteps * 0, timesteps], dim=0)

        t = 1.0 - timesteps
        cap_feats = context
        cap_mask = attention_mask
        bs, c, h, w = x.shape
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))

        t = self.t_embedder(t * self.time_scale, dtype=x.dtype)
        adaln_input = t

        if self.clip_text_pooled_proj is not None:
            pooled = kwargs.get("clip_text_pooled", None)
            if pooled is not None:
                pooled = self.clip_text_pooled_proj(pooled)
            else:
                pooled = torch.zeros((x.shape[0], self.clip_text_dim), device=x.device, dtype=x.dtype)
            adaln_input = self.time_text_embed(torch.cat((t, pooled), dim=-1))

        # ---- capture raw pixel patches before patchify_and_embed embeds them ----
        pH = pW = self.patch_size
        B, C, H, W = x.shape
        pixel_patches = (
            x.view(B, C, H // pH, pH, W // pW, pW)
             .permute(0, 2, 4, 3, 5, 1)   # [B, Ht, Wt, pH, pW, C]
             .flatten(3)                   # [B, Ht, Wt, pH*pW*C]
             .flatten(1, 2)               # [B, N, pH*pW*C]
        )
        N = pixel_patches.shape[1]
        # decoder sees one token per patch: [B*N, 1, P^2*C]
        pixel_values = pixel_patches.reshape(B * N, 1, pH * pW * C)

        patches = transformer_options.get("patches", {})
        x_is_tensor = isinstance(x, torch.Tensor)
        img, mask, img_size, cap_size, freqs_cis, timestep_zero_index = self.patchify_and_embed(
            x, cap_feats, cap_mask, adaln_input, num_tokens,
            ref_latents=ref_latents, ref_contexts=ref_contexts,
            siglip_feats=siglip_feats, transformer_options=transformer_options
        )
        freqs_cis = freqs_cis.to(img.device)

        transformer_options["total_blocks"] = len(self.layers)
        transformer_options["block_type"] = "double"
        img_input = img
        for i, layer in enumerate(self.layers):
            transformer_options["block_index"] = i
            img = layer(img, mask, freqs_cis, adaln_input, timestep_zero_index=timestep_zero_index, transformer_options=transformer_options)
            if "double_block" in patches:
                for p in patches["double_block"]:
                    out = p({"img": img[:, cap_size[0]:], "img_input": img_input[:, cap_size[0]:], "txt": img[:, :cap_size[0]], "pe": freqs_cis[:, cap_size[0]:], "vec": adaln_input, "x": x, "block_index": i, "transformer_options": transformer_options})
                    if "img" in out:
                        img[:, cap_size[0]:] = out["img"]
                    if "txt" in out:
                        img[:, :cap_size[0]] = out["txt"]

        # ---- pixel-space decoder (replaces final_layer + unpatchify) ----
        # img may have padding tokens beyond N; only the first N are real image patches
        img_hidden = img[:, cap_size[0]:cap_size[0] + N, :]  # [B, N, dim]
        decoder_cond = img_hidden.reshape(B * N, self.dim)    # [B*N, dim]

        output = self.dec_net(pixel_values, decoder_cond)  # [B*N, 1, P^2*C]
        output = output.reshape(B, N, -1)                  # [B, N, P^2*C]

        # prepend zero cap placeholder so unpatchify indexing works unchanged
        cap_placeholder = torch.zeros(
            B, cap_size[0], output.shape[-1], device=output.device, dtype=output.dtype
        )
        img_out = self.unpatchify(
            torch.cat([cap_placeholder, output], dim=1),
            img_size, cap_size, return_tensor=x_is_tensor
        )[:, :, :h, :w]

        return -img_out

    def forward(self, x, timesteps, context, num_tokens, attention_mask=None, **kwargs):
        # _forward returns neg_x0 = -x0 (negated decoder output).
        #
        # Reference inference (working_inference_reference.py):
        #   out = _forward(img, t)          # = -x0
        #   pred = (img - out) / t          # = (img + x0) / t  [_apply_x0_residual]
        #   img += (t_prev - t_curr) * pred # Euler step
        #
        # ComfyUI's Euler sampler does the same:
        #   x_next = x + (sigma_next - sigma) * model_output
        # So model_output must equal pred = (x - neg_x0) / t = (x - (-x0)) / t = (x + x0) / t
        neg_x0 = comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, kwargs.get("transformer_options", {}))
        ).execute(x, timesteps, context, num_tokens, attention_mask, **kwargs)

        return (x - neg_x0) / timesteps.view(-1, 1, 1, 1)
