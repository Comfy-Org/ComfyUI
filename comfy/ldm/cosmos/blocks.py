# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Optional
import logging

import numpy as np
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention
import comfy.model_management
import comfy.quant_ops


def get_normalization(name: str, channels: int, weight_args={}, operations=None):
    if name == "I":
        return nn.Identity()
    elif name == "R":
        return operations.RMSNorm(channels, elementwise_affine=True, eps=1e-6, **weight_args)
    else:
        raise ValueError(f"Normalization {name} not found")


class BaseAttentionOp(nn.Module):
    def __init__(self):
        super().__init__()


class Attention(nn.Module):
    """
    Generalized attention impl.

    Allowing for both self-attention and cross-attention configurations depending on whether a `context_dim` is provided.
    If `context_dim` is None, self-attention is assumed.

    Parameters:
        query_dim (int): Dimension of each query vector.
        context_dim (int, optional): Dimension of each context vector. If None, self-attention is assumed.
        heads (int, optional): Number of attention heads. Defaults to 8.
        dim_head (int, optional): Dimension of each head. Defaults to 64.
        dropout (float, optional): Dropout rate applied to the output of the attention block. Defaults to 0.0.
        attn_op (BaseAttentionOp, optional): Custom attention operation to be used instead of the default.
        qkv_bias (bool, optional): If True, adds a learnable bias to query, key, and value projections. Defaults to False.
        out_bias (bool, optional): If True, adds a learnable bias to the output projection. Defaults to False.
        qkv_norm (str, optional): A string representing normalization strategies for query, key, and value projections.
                                  Defaults to "SSI".
        qkv_norm_mode (str, optional): A string representing normalization mode for query, key, and value projections.
                                        Defaults to 'per_head'. Only support 'per_head'.

    Examples:
        >>> attn = Attention(query_dim=128, context_dim=256, heads=4, dim_head=32, dropout=0.1)
        >>> query = torch.randn(10, 128)  # Batch size of 10
        >>> context = torch.randn(10, 256)  # Batch size of 10
        >>> output = attn(query, context)  # Perform the attention operation

    Note:
        https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    """

    def __init__(
        self,
        query_dim: int,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        attn_op: Optional[BaseAttentionOp] = None,
        qkv_bias: bool = False,
        out_bias: bool = False,
        qkv_norm: str = "SSI",
        qkv_norm_mode: str = "per_head",
        backend: str = "transformer_engine",
        qkv_format: str = "bshd",
        weight_args={},
        operations=None,
    ) -> None:
        super().__init__()
        self.comfy_attention = ComfyAttention()

        self.is_selfattn = context_dim is None  # self attention

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head
        self.qkv_norm_mode = qkv_norm_mode
        self.qkv_format = qkv_format

        if self.qkv_norm_mode == "per_head":
            norm_dim = dim_head
        else:
            raise ValueError(f"Normalization mode {self.qkv_norm_mode} not found, only support 'per_head'")

        self.backend = backend

        self.to_q = nn.Sequential(
            operations.Linear(query_dim, inner_dim, bias=qkv_bias, **weight_args),
            get_normalization(qkv_norm[0], norm_dim, weight_args=weight_args, operations=operations),
        )
        self.to_k = nn.Sequential(
            operations.Linear(context_dim, inner_dim, bias=qkv_bias, **weight_args),
            get_normalization(qkv_norm[1], norm_dim, weight_args=weight_args, operations=operations),
        )
        self.to_v = nn.Sequential(
            operations.Linear(context_dim, inner_dim, bias=qkv_bias, **weight_args),
            get_normalization(qkv_norm[2], norm_dim, weight_args=weight_args, operations=operations),
        )

        self.to_out = nn.Sequential(
            operations.Linear(inner_dim, query_dim, bias=out_bias, **weight_args),
            nn.Dropout(dropout),
        )

    def cal_qkv(
        self, x, context=None, mask=None, rope_emb=None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del kwargs


        """
        self.to_q, self.to_k, self.to_v are nn.Sequential with projection + normalization layers.
        Before 07/24/2024, these modules normalize across all heads.
        After 07/24/2024, to support tensor parallelism and follow the common practice in the community,
        we support to normalize per head.
        To keep the checkpoint copatibility with the previous code,
        we keep the nn.Sequential but call the projection and the normalization layers separately.
        We use a flag `self.qkv_norm_mode` to control the normalization behavior.
        The default value of `self.qkv_norm_mode` is "per_head", which means we normalize per head.
        """
        if self.qkv_norm_mode == "per_head":
            q = self.to_q[0](x)
            context = x if context is None else context
            k = self.to_k[0](context)
            v = self.to_v[0](context)
            q, k, v = map(
                lambda t: rearrange(t, "s b (n c) -> b n s c", n=self.heads, c=self.dim_head),
                (q, k, v),
            )
        else:
            raise ValueError(f"Normalization mode {self.qkv_norm_mode} not found, only support 'per_head'")

        q = self.to_q[1](q)
        k = self.to_k[1](k)
        v = self.to_v[1](v)
        if self.is_selfattn and rope_emb is not None:  # only apply to self-attention!
            if not comfy.model_management.in_training:
                q, k = comfy.quant_ops.ck.apply_rope_split_half(q, k, rope_emb)
            else:
                # apply_rotary_pos_emb inlined
                q_shape = q.shape
                q = q.reshape(*q.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2)
                q = rope_emb[..., 0] * q[..., 0] + rope_emb[..., 1] * q[..., 1]
                q = q.movedim(-1, -2).reshape(*q_shape).to(x.dtype)

                # apply_rotary_pos_emb inlined
                k_shape = k.shape
                k = k.reshape(*k.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2)
                k = rope_emb[..., 0] * k[..., 0] + rope_emb[..., 1] * k[..., 1]
                k = k.movedim(-1, -2).reshape(*k_shape).to(x.dtype)
        return q, k, v

    def forward(
        self,
        x,
        context=None,
        mask=None,
        rope_emb=None,
        transformer_options={},
        **kwargs,
    ):
        """
        Args:
            x (Tensor): The query tensor of shape [B, Mq, K]
            context (Optional[Tensor]): The key tensor of shape [B, Mk, K] or use x as context [self attention] if None
        """
        q, k, v = self.cal_qkv(x, context, mask, rope_emb=rope_emb, **kwargs)
        q, k, v = AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v)
        out = optimized_attention(q, k, v, self.heads, skip_reshape=True, mask=mask, skip_output_reshape=True, preferred_attention=self.comfy_attention, transformer_options=transformer_options)
        del q, k, v
        out = rearrange(out, " b n s c -> s b (n c)")
        return self.to_out(out)


class FeedForward(nn.Module):
    """
    Transformer FFN with optional gating

    Parameters:
        d_model (int): Dimensionality of input features.
        d_ff (int): Dimensionality of the hidden layer.
        dropout (float, optional): Dropout rate applied after the activation function. Defaults to 0.1.
        activation (callable, optional): The activation function applied after the first linear layer.
                                         Defaults to nn.ReLU().
        is_gated (bool, optional): If set to True, incorporates gating mechanism to the feed-forward layer.
                                   Defaults to False.
        bias (bool, optional): If set to True, adds a bias to the linear layers. Defaults to True.

    Example:
        >>> ff = FeedForward(d_model=512, d_ff=2048)
        >>> x = torch.randn(64, 10, 512)  # Example input tensor
        >>> output = ff(x)
        >>> print(output.shape)  # Expected shape: (64, 10, 512)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation=nn.ReLU(),
        is_gated: bool = False,
        bias: bool = False,
        weight_args={},
        operations=None,
    ) -> None:
        super().__init__()

        self.layer1 = operations.Linear(d_model, d_ff, bias=bias, **weight_args)
        self.layer2 = operations.Linear(d_ff, d_model, bias=bias, **weight_args)

        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.is_gated = is_gated
        if is_gated:
            self.linear_gate = operations.Linear(d_model, d_ff, bias=False, **weight_args)

    def forward(self, x: torch.Tensor):
        g = self.activation(self.layer1(x))
        if self.is_gated:
            x = g * self.linear_gate(x)
        else:
            x = g
        assert self.dropout.p == 0.0, "we skip dropout"
        return self.layer2(x)


class GPT2FeedForward(FeedForward):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, bias: bool = False, weight_args={}, operations=None):
        super().__init__(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=nn.GELU(),
            is_gated=False,
            bias=bias,
            weight_args=weight_args,
            operations=operations,
        )

    def forward(self, x: torch.Tensor):
        assert self.dropout.p == 0.0, "we skip dropout"

        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)

        return x


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Timesteps(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps):
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - 0.0)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)

        return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_adaln_lora: bool = False, weight_args={}, operations=None):
        super().__init__()
        logging.debug(
            f"Using AdaLN LoRA Flag:  {use_adaln_lora}. We enable bias if no AdaLN LoRA for backward compatibility."
        )
        self.linear_1 = operations.Linear(in_features, out_features, bias=not use_adaln_lora, **weight_args)
        self.activation = nn.SiLU()
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.linear_2 = operations.Linear(out_features, 3 * out_features, bias=False, **weight_args)
        else:
            self.linear_2 = operations.Linear(out_features, out_features, bias=True, **weight_args)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        emb = self.linear_1(sample)
        emb = self.activation(emb)
        emb = self.linear_2(emb)

        if self.use_adaln_lora:
            adaln_lora_B_3D = emb
            emb_B_D = sample
        else:
            emb_B_D = emb
            adaln_lora_B_3D = None

        return emb_B_D, adaln_lora_B_3D


class FourierFeatures(nn.Module):
    """
    Implements a layer that generates Fourier features from input tensors, based on randomly sampled
    frequencies and phases. This can help in learning high-frequency functions in low-dimensional problems.

    [B] -> [B, D]

    Parameters:
        num_channels (int): The number of Fourier features to generate.
        bandwidth (float, optional): The scaling factor for the frequency of the Fourier features. Defaults to 1.
        normalize (bool, optional): If set to True, the outputs are scaled by sqrt(2), usually to normalize
                                    the variance of the features. Defaults to False.

    Example:
        >>> layer = FourierFeatures(num_channels=256, bandwidth=0.5, normalize=True)
        >>> x = torch.randn(10, 256)  # Example input tensor
        >>> output = layer(x)
        >>> print(output.shape)  # Expected shape: (10, 256)
    """

    def __init__(self, num_channels, bandwidth=1, normalize=False):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * bandwidth * torch.randn(num_channels), persistent=True)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels), persistent=True)
        self.gain = np.sqrt(2) if normalize else 1

    def forward(self, x, gain: float = 1.0):
        """
        Apply the Fourier feature transformation to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
            gain (float, optional): An additional gain factor applied during the forward pass. Defaults to 1.

        Returns:
            torch.Tensor: The transformed tensor, with Fourier features applied.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32).ger(self.freqs.to(torch.float32)).add(self.phases.to(torch.float32))
        x = x.cos().mul(self.gain * gain).to(in_dtype)
        return x


class PatchEmbed(nn.Module):
    """
    PatchEmbed is a module for embedding patches from an input tensor by applying either 3D or 2D convolutional layers,
    depending on the . This module can process inputs with temporal (video) and spatial (image) dimensions,
    making it suitable for video and image processing tasks. It supports dividing the input into patches
    and embedding each patch into a vector of size `out_channels`.

    Parameters:
    - spatial_patch_size (int): The size of each spatial patch.
    - temporal_patch_size (int): The size of each temporal patch.
    - in_channels (int): Number of input channels. Default: 3.
    - out_channels (int): The dimension of the embedding vector for each patch. Default: 768.
    - bias (bool): If True, adds a learnable bias to the output of the convolutional layers. Default: True.
    """

    def __init__(
        self,
        spatial_patch_size,
        temporal_patch_size,
        in_channels=3,
        out_channels=768,
        bias=True,
        weight_args={},
        operations=None,
    ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
            ),
            operations.Linear(
                in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size, out_channels, bias=bias, **weight_args
            ),
        )
        self.out = nn.Identity()

    def forward(self, x):
        """
        Forward pass of the PatchEmbed module.

        Parameters:
        - x (torch.Tensor): The input tensor of shape (B, C, T, H, W) where
            B is the batch size,
            C is the number of channels,
            T is the temporal dimension,
            H is the height, and
            W is the width of the input.

        Returns:
        - torch.Tensor: The embedded patches as a tensor, with shape b t h w c.
        """
        assert x.dim() == 5
        _, _, T, H, W = x.shape
        assert H % self.spatial_patch_size == 0 and W % self.spatial_patch_size == 0
        assert T % self.temporal_patch_size == 0
        x = self.proj(x)
        return self.out(x)


class FinalLayer(nn.Module):
    """
    The final layer of video DiT.
    """

    def __init__(
        self,
        hidden_size,
        spatial_patch_size,
        temporal_patch_size,
        out_channels,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        weight_args={},
        operations=None,
    ):
        super().__init__()
        self.norm_final = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **weight_args)
        self.linear = operations.Linear(
            hidden_size, spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels, bias=False, **weight_args
        )
        self.hidden_size = hidden_size
        self.n_adaln_chunks = 2
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                operations.Linear(hidden_size, adaln_lora_dim, bias=False, **weight_args),
                operations.Linear(adaln_lora_dim, self.n_adaln_chunks * hidden_size, bias=False, **weight_args),
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), operations.Linear(hidden_size, self.n_adaln_chunks * hidden_size, bias=False, **weight_args)
            )

    def forward(
        self,
        x_BT_HW_D,
        emb_B_D,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
    ):
        if self.use_adaln_lora:
            assert adaln_lora_B_3D is not None
            shift_B_D, scale_B_D = (self.adaLN_modulation(emb_B_D) + adaln_lora_B_3D[:, : 2 * self.hidden_size]).chunk(
                2, dim=1
            )
        else:
            shift_B_D, scale_B_D = self.adaLN_modulation(emb_B_D).chunk(2, dim=1)

        B = emb_B_D.shape[0]
        T = x_BT_HW_D.shape[0] // B
        shift_BT_D, scale_BT_D = repeat(shift_B_D, "b d -> (b t) d", t=T), repeat(scale_B_D, "b d -> (b t) d", t=T)
        x_BT_HW_D = modulate(self.norm_final(x_BT_HW_D), shift_BT_D, scale_BT_D)

        x_BT_HW_D = self.linear(x_BT_HW_D)
        return x_BT_HW_D


class VideoAttn(nn.Module):
    """
    Implements video attention with optional cross-attention capabilities.

    This module processes video features while maintaining their spatio-temporal structure. It can perform
    self-attention within the video features or cross-attention with external context features.

    Parameters:
        x_dim (int): Dimension of input feature vectors
        context_dim (Optional[int]): Dimension of context features for cross-attention. None for self-attention
        num_heads (int): Number of attention heads
        bias (bool): Whether to include bias in attention projections. Default: False
        qkv_norm_mode (str): Normalization mode for query/key/value projections. Must be "per_head". Default: "per_head"
        x_format (str): Format of input tensor. Must be "BTHWD". Default: "BTHWD"

    Input shape:
        - x: (T, H, W, B, D) video features
        - context (optional): (M, B, D) context features for cross-attention
        where:
            T: temporal dimension
            H: height
            W: width
            B: batch size
            D: feature dimension
            M: context sequence length
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: Optional[int],
        num_heads: int,
        bias: bool = False,
        qkv_norm_mode: str = "per_head",
        x_format: str = "BTHWD",
        weight_args={},
        operations=None,
    ) -> None:
        super().__init__()
        self.x_format = x_format

        self.attn = Attention(
            x_dim,
            context_dim,
            num_heads,
            x_dim // num_heads,
            qkv_bias=bias,
            qkv_norm="RRI",
            out_bias=bias,
            qkv_norm_mode=qkv_norm_mode,
            qkv_format="sbhd",
            weight_args=weight_args,
            operations=operations,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        crossattn_mask: Optional[torch.Tensor] = None,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        transformer_options: Optional[dict] = {},
    ) -> torch.Tensor:
        """
        Forward pass for video attention.

        Args:
            x (Tensor): Input tensor of shape (B, T, H, W, D) or (T, H, W, B, D) representing batches of video data.
            context (Tensor): Context tensor of shape (B, M, D) or (M, B, D),
            where M is the sequence length of the context.
            crossattn_mask (Optional[Tensor]): An optional mask for cross-attention mechanisms.
            rope_emb_L_1_1_D (Optional[Tensor]):
            Rotary positional embedding tensor of shape (L, 1, 1, D). L == THW for current video training.

        Returns:
            Tensor: The output tensor with applied attention, maintaining the input shape.
        """

        x_T_H_W_B_D = x
        context_M_B_D = context
        T, H, W, B, D = x_T_H_W_B_D.shape
        x_THW_B_D = rearrange(x_T_H_W_B_D, "t h w b d -> (t h w) b d")
        x_THW_B_D = self.attn(
            x_THW_B_D,
            context_M_B_D,
            crossattn_mask,
            rope_emb=rope_emb_L_1_1_D,
            transformer_options=transformer_options,
        )
        x_T_H_W_B_D = rearrange(x_THW_B_D, "(t h w) b d -> t h w b d", h=H, w=W)
        return x_T_H_W_B_D


def adaln_norm_state(norm_state, x, scale, shift):
    normalized = norm_state(x)
    return normalized * (1 + scale) + shift


class DITBuildingBlock(nn.Module):
    """
    A building block for the DiT (Diffusion Transformer) architecture that supports different types of
    attention and MLP operations with adaptive layer normalization.

    Parameters:
        block_type (str): Type of block - one of:
            - "cross_attn"/"ca": Cross-attention
            - "full_attn"/"fa": Full self-attention
            - "mlp"/"ff": MLP/feedforward block
        x_dim (int): Dimension of input features
        context_dim (Optional[int]): Dimension of context features for cross-attention
        num_heads (int): Number of attention heads
        mlp_ratio (float): MLP hidden dimension multiplier. Default: 4.0
        bias (bool): Whether to use bias in layers. Default: False
        mlp_dropout (float): Dropout rate for MLP. Default: 0.0
        qkv_norm_mode (str): QKV normalization mode. Default: "per_head"
        x_format (str): Input tensor format. Default: "BTHWD"
        use_adaln_lora (bool): Whether to use AdaLN-LoRA. Default: False
        adaln_lora_dim (int): Dimension for AdaLN-LoRA. Default: 256
    """

    def __init__(
        self,
        block_type: str,
        x_dim: int,
        context_dim: Optional[int],
        num_heads: int,
        mlp_ratio: float = 4.0,
        bias: bool = False,
        mlp_dropout: float = 0.0,
        qkv_norm_mode: str = "per_head",
        x_format: str = "BTHWD",
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        weight_args={},
        operations=None
    ) -> None:
        block_type = block_type.lower()

        super().__init__()
        self.x_format = x_format
        if block_type in ["cross_attn", "ca"]:
            self.block = VideoAttn(
                x_dim,
                context_dim,
                num_heads,
                bias=bias,
                qkv_norm_mode=qkv_norm_mode,
                x_format=self.x_format,
                weight_args=weight_args,
                operations=operations,
            )
        elif block_type in ["full_attn", "fa"]:
            self.block = VideoAttn(
                x_dim, None, num_heads, bias=bias, qkv_norm_mode=qkv_norm_mode, x_format=self.x_format, weight_args=weight_args, operations=operations
            )
        elif block_type in ["mlp", "ff"]:
            self.block = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio), dropout=mlp_dropout, bias=bias, weight_args=weight_args, operations=operations)
        else:
            raise ValueError(f"Unknown block type: {block_type}")

        self.block_type = block_type
        self.use_adaln_lora = use_adaln_lora

        self.norm_state = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.n_adaln_chunks = 3
        if use_adaln_lora:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                operations.Linear(x_dim, adaln_lora_dim, bias=False, **weight_args),
                operations.Linear(adaln_lora_dim, self.n_adaln_chunks * x_dim, bias=False, **weight_args),
            )
        else:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), operations.Linear(x_dim, self.n_adaln_chunks * x_dim, bias=False, **weight_args))

    def forward(
        self,
        x: torch.Tensor,
        emb_B_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
        transformer_options: Optional[dict] = {},
    ) -> torch.Tensor:
        """
        Forward pass for dynamically configured blocks with adaptive normalization.

        Args:
            x (Tensor): Input tensor of shape (B, T, H, W, D) or (T, H, W, B, D).
            emb_B_D (Tensor): Embedding tensor for adaptive layer normalization modulation.
            crossattn_emb (Tensor): Tensor for cross-attention blocks.
            crossattn_mask (Optional[Tensor]): Optional mask for cross-attention.
            rope_emb_L_1_1_D (Optional[Tensor]):
            Rotary positional embedding tensor of shape (L, 1, 1, D). L == THW for current video training.

        Returns:
            Tensor: The output tensor after processing through the configured block and adaptive normalization.
        """
        if self.use_adaln_lora:
            shift_B_D, scale_B_D, gate_B_D = (self.adaLN_modulation(emb_B_D) + adaln_lora_B_3D).chunk(
                self.n_adaln_chunks, dim=1
            )
        else:
            shift_B_D, scale_B_D, gate_B_D = self.adaLN_modulation(emb_B_D).chunk(self.n_adaln_chunks, dim=1)

        shift_1_1_1_B_D, scale_1_1_1_B_D, gate_1_1_1_B_D = (
            shift_B_D.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            scale_B_D.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            gate_B_D.unsqueeze(0).unsqueeze(0).unsqueeze(0),
        )

        if self.block_type in ["mlp", "ff"]:
            x = x + gate_1_1_1_B_D * self.block(
                adaln_norm_state(self.norm_state, x, scale_1_1_1_B_D, shift_1_1_1_B_D),
            )
        elif self.block_type in ["full_attn", "fa"]:
            x = x + gate_1_1_1_B_D * self.block(
                adaln_norm_state(self.norm_state, x, scale_1_1_1_B_D, shift_1_1_1_B_D),
                context=None,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                transformer_options=transformer_options,
            )
        elif self.block_type in ["cross_attn", "ca"]:
            x = x + gate_1_1_1_B_D * self.block(
                adaln_norm_state(self.norm_state, x, scale_1_1_1_B_D, shift_1_1_1_B_D),
                context=crossattn_emb,
                crossattn_mask=crossattn_mask,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                transformer_options=transformer_options,
            )
        else:
            raise ValueError(f"Unknown block type: {self.block_type}")

        return x


class GeneralDITTransformerBlock(nn.Module):
    """
    A wrapper module that manages a sequence of DITBuildingBlocks to form a complete transformer layer.
    Each block in the sequence is specified by a block configuration string.

    Parameters:
        x_dim (int): Dimension of input features
        context_dim (int): Dimension of context features for cross-attention blocks
        num_heads (int): Number of attention heads
        block_config (str): String specifying block sequence (e.g. "ca-fa-mlp" for cross-attention,
                          full-attention, then MLP)
        mlp_ratio (float): MLP hidden dimension multiplier. Default: 4.0
        x_format (str): Input tensor format. Default: "BTHWD"
        use_adaln_lora (bool): Whether to use AdaLN-LoRA. Default: False
        adaln_lora_dim (int): Dimension for AdaLN-LoRA. Default: 256

    The block_config string uses "-" to separate block types:
        - "ca"/"cross_attn": Cross-attention block
        - "fa"/"full_attn": Full self-attention block
        - "mlp"/"ff": MLP/feedforward block

    Example:
        block_config = "ca-fa-mlp" creates a sequence of:
        1. Cross-attention block
        2. Full self-attention block
        3. MLP block
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        block_config: str,
        mlp_ratio: float = 4.0,
        x_format: str = "BTHWD",
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        weight_args={},
        operations=None
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.x_format = x_format
        for block_type in block_config.split("-"):
            self.blocks.append(
                DITBuildingBlock(
                    block_type,
                    x_dim,
                    context_dim,
                    num_heads,
                    mlp_ratio,
                    x_format=self.x_format,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                    weight_args=weight_args,
                    operations=operations,
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        emb_B_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_3D: Optional[torch.Tensor] = None,
        transformer_options: Optional[dict] = {},
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(
                x,
                emb_B_D,
                crossattn_emb,
                crossattn_mask,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_3D=adaln_lora_B_3D,
                transformer_options=transformer_options,
            )
        return x
