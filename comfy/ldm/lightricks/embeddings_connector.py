import math
from typing import Optional

import comfy.ldm.common_dit
import torch
from comfy.ldm.lightricks.model import (
    CrossAttention,
    FeedForward,
    freqs_cis_matrix,
    generate_freq_grid_np,
)
from torch import nn


class BasicTransformerBlock1D(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:

        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        standardization_norm (`str`, *optional*, defaults to `"layer_norm"`): The type of pre-normalization to use. Can be `"layer_norm"` or `"rms_norm"`.
        norm_eps (`float`, *optional*, defaults to 1e-5): Epsilon value for normalization layers.
        qk_norm (`str`, *optional*, defaults to None):
            Set to 'layer_norm' or `rms_norm` to perform query and key normalization.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*): Dimension of the inner feed-forward layer. If not provided, defaults to `dim * 4`.
        ff_bias (`bool`, *optional*, defaults to `True`): Whether to use bias in the feed-forward layer.
        attention_out_bias (`bool`, *optional*, defaults to `True`): Whether to use bias in the attention output layer.
        use_rope (`bool`, *optional*, defaults to `False`): Whether to use Rotary Position Embeddings (RoPE).
        ffn_dim_mult (`int`, *optional*, defaults to 4): Multiplier for the inner dimension of the feed-forward layer.
    """

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        context_dim=None,
        attn_precision=None,
        apply_gated_attention=False,
        ff_bias=True,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            context_dim=None,
            apply_gated_attention=apply_gated_attention,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        # 3. Feed-forward
        self.ff = FeedForward(
            dim,
            dim_out=dim,
            glu=True,
            ff_bias=ff_bias,
            dtype=dtype,
            device=device,
            operations=operations,
        )

    def forward(self, hidden_states, attention_mask=None, pe=None) -> torch.FloatTensor:

        # Notice that normalization is always applied before the real computation in the following blocks.

        # 1. Normalization Before Self-Attention
        norm_hidden_states = comfy.ldm.common_dit.rms_norm(hidden_states)

        norm_hidden_states = norm_hidden_states.squeeze(1)

        # 2. Self-Attention
        attn_output = self.attn1(norm_hidden_states, mask=attention_mask, pe=pe)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 3. Normalization before Feed-Forward
        norm_hidden_states = comfy.ldm.common_dit.rms_norm(hidden_states)

        # 4. Feed-forward
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class Embeddings1DConnector(nn.Module):
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels=128,
        cross_attention_dim=2048,
        attention_head_dim=128,
        num_attention_heads=30,
        num_layers=2,
        positional_embedding_theta=10000.0,
        positional_embedding_max_pos=[4096],
        causal_temporal_positioning=False,
        num_learnable_registers: Optional[int] = 128,
        apply_gated_attention=False,
        connector_ff_bias=True,
        dtype=None,
        device=None,
        operations=None,
        split_rope=False,
        double_precision_rope=False,
        **kwargs,
    ):
        super().__init__()
        self.dtype = dtype
        self.out_channels = in_channels
        self.num_attention_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim
        self.causal_temporal_positioning = causal_temporal_positioning
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = positional_embedding_max_pos
        self.split_rope = split_rope
        self.double_precision_rope = double_precision_rope
        self.transformer_1d_blocks = nn.ModuleList(
            [
                BasicTransformerBlock1D(
                    self.inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    context_dim=cross_attention_dim,
                    apply_gated_attention=apply_gated_attention,
                    ff_bias=connector_ff_bias,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
                for _ in range(num_layers)
            ]
        )

        inner_dim = num_attention_heads * attention_head_dim
        self.num_learnable_registers = num_learnable_registers
        if self.num_learnable_registers:
            self.learnable_registers = nn.Parameter(
                torch.empty(
                    self.num_learnable_registers, inner_dim, dtype=dtype, device=device
                )
            )

    def get_fractional_positions(self, indices_grid):
        fractional_positions = torch.stack(
            [
                indices_grid[:, i] / self.positional_embedding_max_pos[i]
                for i in range(1)
            ],
            dim=-1,
        )
        return fractional_positions

    def precompute_freqs(self, indices_grid, spacing):
        source_dtype = indices_grid.dtype
        dtype = (
            torch.float32
            if source_dtype in (torch.bfloat16, torch.float16)
            else source_dtype
        )

        fractional_positions = self.get_fractional_positions(indices_grid)
        indices = (
            generate_freq_grid_np(
                self.positional_embedding_theta,
                indices_grid.shape[1],
                self.inner_dim,
            )
            if self.double_precision_rope
            else self.generate_freq_grid(spacing, dtype, fractional_positions.device)
        ).to(device=fractional_positions.device)

        if spacing == "exp_2":
            freqs = (
                (indices * fractional_positions.unsqueeze(-1))
                .transpose(-1, -2)
                .flatten(2)
            )
        else:
            freqs = (
                (indices * (fractional_positions.unsqueeze(-1) * 2 - 1))
                .transpose(-1, -2)
                .flatten(2)
            )
        return freqs

    def generate_freq_grid(self, spacing, dtype, device):
        dim = self.inner_dim
        theta = self.positional_embedding_theta
        n_pos_dims = 1
        n_elem = 2 * n_pos_dims  # 2 for cos and sin e.g. x 3 = 6
        start = 1
        end = theta

        if spacing == "exp":
            indices = theta ** (torch.arange(0, dim, n_elem, device="cpu", dtype=torch.float32) / (dim - n_elem))
            indices = indices.to(dtype=dtype, device=device)
        elif spacing == "exp_2":
            indices = 1.0 / theta ** (torch.arange(0, dim, n_elem, device=device) / dim)
            indices = indices.to(dtype=dtype)
        elif spacing == "linear":
            indices = torch.linspace(
                start, end, dim // n_elem, device=device, dtype=dtype
            )
        elif spacing == "sqrt":
            indices = torch.linspace(
                start**2, end**2, dim // n_elem, device=device, dtype=dtype
            ).sqrt()

        indices = indices * math.pi / 2

        return indices

    def precompute_freqs_cis(self, indices_grid, spacing="exp", out_dtype=None):
        dim = self.inner_dim
        n_elem = 2  # 2 because of cos and sin
        freqs = self.precompute_freqs(indices_grid, spacing)
        if self.split_rope:
            expected_freqs = dim // 2
            current_freqs = freqs.shape[-1]
            pad_size = expected_freqs - current_freqs
        else:
            pad_size = dim % n_elem
        return freqs_cis_matrix(
            freqs,
            pad_size,
            self.split_rope,
            self.num_attention_heads,
            out_dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        The [`Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            indices_grid (`torch.LongTensor` of shape `(batch size, 3, num latent pixels)`):
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        # 1. Input

        if self.num_learnable_registers:
            num_registers_duplications = math.ceil(
                max(1024, hidden_states.shape[1]) / self.num_learnable_registers
            )
            learnable_registers = torch.tile(
                self.learnable_registers.to(hidden_states), (num_registers_duplications, 1)
            )

            hidden_states = torch.cat((hidden_states, learnable_registers[hidden_states.shape[1]:].unsqueeze(0).repeat(hidden_states.shape[0], 1, 1)), dim=1)

            if attention_mask is not None:
                attention_mask = torch.zeros([1, 1, 1, hidden_states.shape[1]], dtype=attention_mask.dtype, device=attention_mask.device)

        indices_grid = torch.arange(
            hidden_states.shape[1], dtype=torch.float32, device=hidden_states.device
        )
        indices_grid = indices_grid[None, None, :]
        freqs_cis = self.precompute_freqs_cis(indices_grid, out_dtype=hidden_states.dtype)

        # 2. Blocks
        for block_idx, block in enumerate(self.transformer_1d_blocks):
            hidden_states = block(
                hidden_states, attention_mask=attention_mask, pe=freqs_cis
            )

        # 3. Output
        # if self.output_scale is not None:
        #     hidden_states = hidden_states / self.output_scale

        hidden_states = comfy.ldm.common_dit.rms_norm(hidden_states)

        return hidden_states, attention_mask
