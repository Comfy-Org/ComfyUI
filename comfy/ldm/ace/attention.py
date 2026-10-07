# Original from: https://github.com/ace-step/ACE-Step/blob/main/models/attention.py
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Tuple, Union, Optional

import torch
import torch.nn.functional as F
from torch import nn

import comfy.model_management
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention

class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        kv_heads: Optional[int] = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        qk_norm: Optional[str] = None,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        processor=None,
        out_dim: int = None,
        out_context_dim: int = None,
        context_pre_only=None,
        pre_only=False,
        elementwise_affine: bool = True,
        is_causal: bool = False,
        dtype=None, device=None, operations=None
    ):
        super().__init__()
        self.comfy_attention = ComfyAttention()

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.out_context_dim = out_context_dim if out_context_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.is_causal = is_causal

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        self.heads = out_dim // dim_head if out_dim is not None else heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads

        self.added_kv_proj_dim = added_kv_proj_dim
        self.only_cross_attention = only_cross_attention

        if self.added_kv_proj_dim is None and self.only_cross_attention:
            raise ValueError(
                "`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None. Make sure to set either `only_cross_attention=False` or define `added_kv_proj_dim`."
            )

        self.group_norm = None
        self.spatial_norm = None

        self.norm_q = None
        self.norm_k = None

        self.norm_cross = None
        self.to_q = operations.Linear(query_dim, self.inner_dim, bias=bias, dtype=dtype, device=device)

        if not self.only_cross_attention:
            # only relevant for the `AddedKVProcessor` classes
            self.to_k = operations.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias, dtype=dtype, device=device)
            self.to_v = operations.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias, dtype=dtype, device=device)
        else:
            self.to_k = None
            self.to_v = None

        self.added_proj_bias = added_proj_bias
        if self.added_kv_proj_dim is not None:
            self.add_k_proj = operations.Linear(added_kv_proj_dim, self.inner_kv_dim, bias=added_proj_bias, dtype=dtype, device=device)
            self.add_v_proj = operations.Linear(added_kv_proj_dim, self.inner_kv_dim, bias=added_proj_bias, dtype=dtype, device=device)
            if self.context_pre_only is not None:
                self.add_q_proj = operations.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias, dtype=dtype, device=device)
        else:
            self.add_q_proj = None
            self.add_k_proj = None
            self.add_v_proj = None

        if not self.pre_only:
            self.to_out = nn.ModuleList([])
            self.to_out.append(operations.Linear(self.inner_dim, self.out_dim, bias=out_bias, dtype=dtype, device=device))
            self.to_out.append(nn.Dropout(dropout))
        else:
            self.to_out = None

        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_add_out = operations.Linear(self.inner_dim, self.out_context_dim, bias=out_bias, dtype=dtype, device=device)
        else:
            self.to_add_out = None

        self.norm_added_q = None
        self.norm_added_k = None
        self.processor = processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        transformer_options={},
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            transformer_options=transformer_options,
            **cross_attention_kwargs,
        )


class CustomLiteLAProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections. add rms norm for query and key and apply RoPE"""

    def __init__(self):
        self.kernel_func = nn.ReLU(inplace=False)
        self.eps = 1e-15
        self.pad_val = 1.0

    def apply_rotary_emb(
        self,
        x: torch.Tensor,
        freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
        to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
        reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
        tensors contain rotary embeddings and are returned as real tensors.

        Args:
            x (`torch.Tensor`):
                Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
            freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
        """
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        rotary_freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        rotary_freqs_cis_cross: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        hidden_states_len = hidden_states.shape[1]

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        if encoder_hidden_states is not None:
            context_input_ndim = encoder_hidden_states.ndim
            if context_input_ndim == 4:
                batch_size, channel, height, width = encoder_hidden_states.shape
                encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        dtype = hidden_states.dtype
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # `context` projections.
        has_encoder_hidden_state_proj = hasattr(attn, "add_q_proj") and hasattr(attn, "add_k_proj") and hasattr(attn, "add_v_proj")
        if encoder_hidden_states is not None and has_encoder_hidden_state_proj:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            # attention
            if not attn.is_cross_attention:
                query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
                key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
                value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)
            else:
                query = hidden_states
                key = encoder_hidden_states
                value = encoder_hidden_states

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.transpose(-1, -2).reshape(batch_size, attn.heads, head_dim, -1)
        key = key.transpose(-1, -2).reshape(batch_size, attn.heads, head_dim, -1).transpose(-1, -2)
        value = value.transpose(-1, -2).reshape(batch_size, attn.heads, head_dim, -1)

        # RoPE需要 [B, H, S, D] 输入
        # 此时 query是 [B, H, D, S], 需要转成 [B, H, S, D] 才能应用RoPE
        query = query.permute(0, 1, 3, 2)  # [B, H, S, D]  (从 [B, H, D, S])

        # Apply query and key normalization if needed
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if rotary_freqs_cis is not None:
            query = self.apply_rotary_emb(query, rotary_freqs_cis)
            if not attn.is_cross_attention:
                key = self.apply_rotary_emb(key, rotary_freqs_cis)
            elif rotary_freqs_cis_cross is not None and has_encoder_hidden_state_proj:
                key = self.apply_rotary_emb(key, rotary_freqs_cis_cross)

        # 此时 query是 [B, H, S, D]，需要还原成 [B, H, D, S]
        query = query.permute(0, 1, 3, 2)  # [B, H, D, S]

        if attention_mask is not None:
            # attention_mask: [B, S] -> [B, 1, S, 1]
            attention_mask = attention_mask[:, None, :, None].to(key.dtype)  # [B, 1, S, 1]
            query = query * attention_mask.permute(0, 1, 3, 2)  # [B, H, S, D] * [B, 1, S, 1]
            if not attn.is_cross_attention:
                key = key * attention_mask  # key: [B, h, S, D] 与 mask [B, 1, S, 1] 相乘
                value = value * attention_mask.permute(0, 1, 3, 2)  # 如果 value 是 [B, h, D, S]，那么需调整mask以匹配S维度

        if attn.is_cross_attention and encoder_attention_mask is not None and has_encoder_hidden_state_proj:
            encoder_attention_mask = encoder_attention_mask[:, None, :, None].to(key.dtype)  # [B, 1, S_enc, 1]
            # 此时 key: [B, h, S_enc, D], value: [B, h, D, S_enc]
            key = key * encoder_attention_mask  # [B, h, S_enc, D] * [B, 1, S_enc, 1]
            value = value * encoder_attention_mask.permute(0, 1, 3, 2)  # [B, h, D, S_enc] * [B, 1, 1, S_enc]

        query = self.kernel_func(query)
        key = self.kernel_func(key)

        query, key, value = query.float(), key.float(), value.float()

        value = F.pad(value, (0, 0, 0, 1), mode="constant", value=self.pad_val)

        vk = torch.matmul(value, key)

        hidden_states = torch.matmul(vk, query)

        if hidden_states.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.float()

        hidden_states = hidden_states[:, :, :-1] / (hidden_states[:, :, -1:] + self.eps)

        hidden_states = hidden_states.view(batch_size, attn.heads * head_dim, -1).permute(0, 2, 1)

        hidden_states = hidden_states.to(dtype)
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.to(dtype)

        # Split the attention outputs.
        if encoder_hidden_states is not None and not attn.is_cross_attention and has_encoder_hidden_state_proj:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : hidden_states_len],
                hidden_states[:, hidden_states_len:],
            )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if encoder_hidden_states is not None and not attn.context_pre_only and not attn.is_cross_attention and hasattr(attn, "to_add_out"):
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if encoder_hidden_states is not None and context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if torch.get_autocast_gpu_dtype() == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
            if encoder_hidden_states is not None:
                encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return hidden_states, encoder_hidden_states


class CustomerAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def apply_rotary_emb(
        self,
        x: torch.Tensor,
        freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
        to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
        reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
        tensors contain rotary embeddings and are returned as real tensors.

        Args:
            x (`torch.Tensor`):
                Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
            freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
        """
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        rotary_freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        rotary_freqs_cis_cross: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        transformer_options={},
        *args,
        **kwargs,
    ) -> torch.Tensor:

        residual = hidden_states
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        has_encoder_hidden_state_proj = hasattr(attn, "add_q_proj") and hasattr(attn, "add_k_proj") and hasattr(attn, "add_v_proj")

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if rotary_freqs_cis is not None:
            query = self.apply_rotary_emb(query, rotary_freqs_cis)
            if not attn.is_cross_attention:
                key = self.apply_rotary_emb(key, rotary_freqs_cis)
            elif rotary_freqs_cis_cross is not None and has_encoder_hidden_state_proj:
                key = self.apply_rotary_emb(key, rotary_freqs_cis_cross)

        if attn.is_cross_attention and encoder_attention_mask is not None and has_encoder_hidden_state_proj:
            # attention_mask: N x S1
            # encoder_attention_mask: N x S2
            # cross attention 整合attention_mask和encoder_attention_mask
            combined_mask = attention_mask[:, :, None] * encoder_attention_mask[:, None, :]
            attention_mask = torch.where(combined_mask == 1, 0.0, -torch.inf)
            attention_mask = attention_mask[:, None, :, :].expand(-1, attn.heads, -1, -1).to(query.dtype)

        elif not attn.is_cross_attention and attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        heads = query.shape[1]
        query, key, value = AttentionTensorContainer(query), AttentionTensorContainer(key), AttentionTensorContainer(value)
        hidden_states = optimized_attention(
            query, key, value, heads=heads, mask=attention_mask, skip_reshape=True, preferred_attention=attn.comfy_attention, transformer_options=transformer_options,
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

def val2list(x: list or tuple or any, repeat_time=1) -> list:  # type: ignore
    """Repeat `val` for `repeat_time` times and return the list or val if list/tuple."""
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: list or tuple or any, min_len: int = 1, idx_repeat: int = -1) -> tuple:  # type: ignore
    """Return tuple with min_len by repeating element at idx_repeat."""
    # convert to list first
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


def get_same_padding(kernel_size: Union[int, Tuple[int, ...]]) -> Union[int, Tuple[int, ...]]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, f"kernel size {kernel_size} should be odd number"
        return kernel_size // 2

class ConvLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        padding: Union[int, None] = None,
        use_bias=False,
        norm=None,
        act=None,
        dtype=None, device=None, operations=None
    ):
        super().__init__()
        if padding is None:
            padding = get_same_padding(kernel_size)
            padding *= dilation

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding
        self.use_bias = use_bias

        self.conv = operations.Conv1d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=use_bias,
            device=device,
            dtype=dtype
        )
        if norm is not None:
            self.norm = operations.RMSNorm(out_dim, elementwise_affine=False, dtype=dtype, device=device)
        else:
            self.norm = None
        if act is not None:
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class GLUMBConv(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_feature=None,
        kernel_size=3,
        stride=1,
        padding: Union[int, None] = None,
        use_bias=False,
        norm=(None, None, None),
        act=("silu", "silu", None),
        dilation=1,
        dtype=None, device=None, operations=None
    ):
        out_feature = out_feature or in_features
        super().__init__()
        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act = val2tuple(act, 3)

        self.glu_act = nn.SiLU(inplace=False)
        self.inverted_conv = ConvLayer(
            in_features,
            hidden_features * 2,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act=act[0],
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.depth_conv = ConvLayer(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size,
            stride=stride,
            groups=hidden_features * 2,
            padding=padding,
            use_bias=use_bias[1],
            norm=norm[1],
            act=None,
            dilation=dilation,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.point_conv = ConvLayer(
            hidden_features,
            out_feature,
            1,
            use_bias=use_bias[2],
            norm=norm[2],
            act=act[2],
            dtype=dtype,
            device=device,
            operations=operations,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.inverted_conv(x)
        x = self.depth_conv(x)

        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate

        x = self.point_conv(x)
        x = x.transpose(1, 2)

        return x


class LinearTransformerBlock(nn.Module):
    """
    A Sana block with global shared adaptive layer norm (adaLN-single) conditioning.
    """
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        use_adaln_single=True,
        cross_attention_dim=None,
        added_kv_proj_dim=None,
        context_pre_only=False,
        mlp_ratio=4.0,
        add_cross_attention=False,
        add_cross_attention_dim=None,
        qk_norm=None,
        dtype=None, device=None, operations=None
    ):
        super().__init__()

        self.norm1 = operations.RMSNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            added_kv_proj_dim=added_kv_proj_dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            qk_norm=qk_norm,
            processor=CustomLiteLAProcessor2_0(),
            dtype=dtype,
            device=device,
            operations=operations,
        )

        self.add_cross_attention = add_cross_attention
        self.context_pre_only = context_pre_only

        if add_cross_attention and add_cross_attention_dim is not None:
            self.cross_attn = Attention(
                query_dim=dim,
                cross_attention_dim=add_cross_attention_dim,
                added_kv_proj_dim=add_cross_attention_dim,
                dim_head=attention_head_dim,
                heads=num_attention_heads,
                out_dim=dim,
                context_pre_only=context_pre_only,
                bias=True,
                qk_norm=qk_norm,
                processor=CustomerAttnProcessor2_0(),
                dtype=dtype,
                device=device,
                operations=operations,
            )

        self.norm2 = operations.RMSNorm(dim, 1e-06, elementwise_affine=False)

        self.ff = GLUMBConv(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            use_bias=(True, True, False),
            norm=(None, None, None),
            act=("silu", "silu", None),
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.use_adaln_single = use_adaln_single
        if use_adaln_single:
            self.scale_shift_table = nn.Parameter(torch.empty(6, dim, dtype=dtype, device=device))

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor = None,
        encoder_attention_mask: torch.FloatTensor = None,
        rotary_freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        rotary_freqs_cis_cross: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        temb: torch.FloatTensor = None,
        transformer_options={},
    ):

        N = hidden_states.shape[0]

        # step 1: AdaLN single
        if self.use_adaln_single:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                comfy.model_management.cast_to(self.scale_shift_table[None], dtype=temb.dtype, device=temb.device) + temb.reshape(N, 6, -1)
            ).chunk(6, dim=1)

        norm_hidden_states = self.norm1(hidden_states)
        if self.use_adaln_single:
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        # step 2: attention
        if not self.add_cross_attention:
            attn_output, encoder_hidden_states = self.attn(
                hidden_states=norm_hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                rotary_freqs_cis=rotary_freqs_cis,
                rotary_freqs_cis_cross=rotary_freqs_cis_cross,
                transformer_options=transformer_options,
            )
        else:
            attn_output, _ = self.attn(
                hidden_states=norm_hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                rotary_freqs_cis=rotary_freqs_cis,
                rotary_freqs_cis_cross=None,
                transformer_options=transformer_options,
            )

        if self.use_adaln_single:
            attn_output = gate_msa * attn_output
        hidden_states = attn_output + hidden_states

        if self.add_cross_attention:
            attn_output = self.cross_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                rotary_freqs_cis=rotary_freqs_cis,
                rotary_freqs_cis_cross=rotary_freqs_cis_cross,
                transformer_options=transformer_options,
            )
            hidden_states = attn_output + hidden_states

        # step 3: add norm
        norm_hidden_states = self.norm2(hidden_states)
        if self.use_adaln_single:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        # step 4: feed forward
        ff_output = self.ff(norm_hidden_states)
        if self.use_adaln_single:
            ff_output = gate_mlp * ff_output

        hidden_states = hidden_states + ff_output

        return hidden_states
