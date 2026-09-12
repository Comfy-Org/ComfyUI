# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import torch
from torch import nn
import torch.nn.functional as F

import comfy.ops
from comfy.ldm.modules.attention import optimized_attention


class QueryAttention(nn.Module):
    def __init__(
        self, hidden_size, num_heads, dtype=None, device=None, operations=None
    ):
        super().__init__()
        self.heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.inner_dim = hidden_size
        self.in_proj_weight = nn.Parameter(
            torch.empty(3 * hidden_size, hidden_size, dtype=dtype, device=device)
        )
        self.in_proj_bias = nn.Parameter(
            torch.empty(3 * hidden_size, dtype=dtype, device=device)
        )
        self.out_proj = operations.Linear(
            hidden_size, hidden_size, bias=True, dtype=dtype, device=device
        )

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        transformer_options={},
    ):
        weight = comfy.ops.cast_to_input(self.in_proj_weight, hidden_states)
        bias = comfy.ops.cast_to_input(self.in_proj_bias, hidden_states)
        query = F.linear(
            hidden_states, weight[: self.inner_dim], bias[: self.inner_dim]
        )
        key = F.linear(
            encoder_hidden_states,
            weight[self.inner_dim : 2 * self.inner_dim],
            bias[self.inner_dim : 2 * self.inner_dim],
        )
        value = F.linear(
            encoder_hidden_states,
            weight[2 * self.inner_dim :],
            bias[2 * self.inner_dim :],
        )
        query = query.unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2)
        key = key.unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2)
        value = value.unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2)
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = torch.where(
                attention_mask,
                torch.zeros((), dtype=query.dtype, device=query.device),
                torch.full(
                    (),
                    -torch.finfo(query.dtype).max,
                    dtype=query.dtype,
                    device=query.device,
                ),
            )
        hidden_states = optimized_attention(
            query,
            key,
            value,
            self.heads,
            mask=attention_mask,
            skip_reshape=True,
            transformer_options=transformer_options,
        )
        return self.out_proj(hidden_states)


class QueryFormerBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        intermediate_size,
        norm_eps,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=norm_eps)
        self.norm_k = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=norm_eps)
        self.cross_attn = QueryAttention(
            hidden_size, num_heads, dtype, device, operations
        )
        self.dropout = nn.Identity()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=norm_eps)
        self.mlp = nn.Module()
        self.mlp.fc1 = operations.Linear(
            hidden_size, intermediate_size, bias=True, dtype=dtype, device=device
        )
        self.mlp.fc2 = operations.Linear(
            intermediate_size, hidden_size, bias=True, dtype=dtype, device=device
        )

    def forward(
        self,
        query_embeds,
        encoder_hidden_states,
        attention_mask=None,
        transformer_options={},
    ):
        query_embeds = self.norm_q(query_embeds)
        encoder_hidden_states = self.norm_k(encoder_hidden_states)
        query_embeds = query_embeds + self.cross_attn(
            query_embeds, encoder_hidden_states, attention_mask, transformer_options
        )
        query_embeds = self.norm1(query_embeds)
        mlp_output = self.mlp.fc2(
            F.gelu(self.mlp.fc1(query_embeds), approximate="tanh")
        )
        return query_embeds + mlp_output


class QueryFormer(nn.Module):
    def __init__(
        self,
        num_queries=256,
        hidden_size=2048,
        num_hidden_layers=1,
        num_attention_heads=16,
        intermediate_size=8192,
        norm_eps=1e-6,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )
        self.meta_queries = nn.Parameter(
            torch.empty(num_queries, hidden_size, dtype=dtype, device=device)
        )
        self.query_blocks = nn.ModuleList(
            [
                QueryFormerBlock(
                    hidden_size,
                    num_attention_heads,
                    intermediate_size,
                    norm_eps,
                    dtype,
                    device,
                    operations,
                )
                for _ in range(num_hidden_layers)
            ]
        )

    def forward(self, inputs_embeds, attention_mask, transformer_options={}):
        query_embeds = comfy.ops.cast_to_input(
            self.meta_queries, inputs_embeds, copy=False
        ).unsqueeze(0).expand(inputs_embeds.shape[0], -1, -1)
        for block in self.query_blocks:
            query_embeds = block(
                query_embeds, inputs_embeds, attention_mask.bool(), transformer_options
            )
        return query_embeds


class TextProjectionAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        norm_eps,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.k_proj = operations.Linear(
            hidden_size, hidden_size, bias=True, dtype=dtype, device=device
        )
        self.v_proj = operations.Linear(
            hidden_size, hidden_size, bias=True, dtype=dtype, device=device
        )
        self.q_proj = operations.Linear(
            hidden_size, hidden_size, bias=True, dtype=dtype, device=device
        )
        self.out_proj = operations.Linear(
            hidden_size, hidden_size, bias=True, dtype=dtype, device=device
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=norm_eps, elementwise_affine=False)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=norm_eps, elementwise_affine=False)

    def forward(self, hidden_states, transformer_options={}):
        query = self.q_norm(
            self.q_proj(hidden_states).unflatten(-1, (self.heads, self.head_dim))
        ).transpose(1, 2)
        key = self.k_norm(
            self.k_proj(hidden_states).unflatten(-1, (self.heads, self.head_dim))
        ).transpose(1, 2)
        value = (
            self.v_proj(hidden_states)
            .unflatten(-1, (self.heads, self.head_dim))
            .transpose(1, 2)
        )
        hidden_states = optimized_attention(
            query,
            key,
            value,
            self.heads,
            skip_reshape=True,
            transformer_options=transformer_options,
        )
        return self.out_proj(hidden_states)


class TextProjectionMLP(nn.Module):
    def __init__(
        self, hidden_size, intermediate_size, dtype=None, device=None, operations=None
    ):
        super().__init__()
        self.fc1 = operations.Linear(
            hidden_size, intermediate_size, bias=True, dtype=dtype, device=device
        )
        self.fc2 = operations.Linear(
            intermediate_size, hidden_size, bias=True, dtype=dtype, device=device
        )

    def forward(self, hidden_states):
        return self.fc2(F.gelu(self.fc1(hidden_states), approximate="tanh"))


class TextProjectionBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        norm_eps,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.self_attn = TextProjectionAttention(
            hidden_size, num_attention_heads, norm_eps, dtype, device, operations
        )
        self.layer_norm1 = nn.RMSNorm(
            hidden_size, eps=norm_eps, elementwise_affine=False
        )
        self.mlp = TextProjectionMLP(
            hidden_size, intermediate_size, dtype, device, operations
        )
        self.layer_norm2 = nn.RMSNorm(
            hidden_size, eps=norm_eps, elementwise_affine=False
        )

    def forward(self, hidden_states, transformer_options={}):
        hidden_states = hidden_states + self.self_attn(
            self.layer_norm1(hidden_states), transformer_options
        )
        return hidden_states + self.mlp(self.layer_norm2(hidden_states))


class TextProjection(nn.Module):
    def __init__(
        self,
        hidden_size=2048,
        intermediate_size=8960,
        num_hidden_layers=6,
        num_attention_heads=32,
        projection_dim=2560,
        norm_eps=1e-6,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )
        self.layers = nn.ModuleList(
            [
                TextProjectionBlock(
                    hidden_size,
                    intermediate_size,
                    num_attention_heads,
                    norm_eps,
                    dtype,
                    device,
                    operations,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.projector = operations.Linear(
            hidden_size, projection_dim, bias=True, dtype=dtype, device=device
        )

    def forward(self, hidden_states, transformer_options={}):
        for layer in self.layers:
            hidden_states = layer(hidden_states, transformer_options)
        return self.projector(hidden_states)


class SigVQAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_bias,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.qkv = operations.Linear(
            hidden_size,
            3 * hidden_size,
            bias=attention_bias,
            dtype=dtype,
            device=device,
        )
        self.proj = operations.Linear(
            hidden_size, hidden_size, bias=attention_bias, dtype=dtype, device=device
        )

    def forward(self, hidden_states, transformer_options={}):
        query, key, value = self.qkv(hidden_states).chunk(3, dim=-1)
        query = query.unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2)
        key = key.unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2)
        value = value.unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2)
        hidden_states = optimized_attention(
            query,
            key,
            value,
            self.heads,
            skip_reshape=True,
            transformer_options=transformer_options,
        )
        return self.proj(hidden_states)


class SigVQMLP(nn.Module):
    def __init__(
        self, hidden_size, intermediate_size, dtype=None, device=None, operations=None
    ):
        super().__init__()
        self.fc1 = operations.Linear(
            hidden_size, intermediate_size, bias=True, dtype=dtype, device=device
        )
        self.fc2 = operations.Linear(
            intermediate_size, hidden_size, bias=True, dtype=dtype, device=device
        )

    def forward(self, hidden_states):
        return self.fc2(F.gelu(self.fc1(hidden_states)))


class SigVQVisionBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        attention_bias,
        norm_eps,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.norm1 = operations.LayerNorm(
            hidden_size, eps=norm_eps, dtype=dtype, device=device
        )
        self.norm2 = operations.LayerNorm(
            hidden_size, eps=norm_eps, dtype=dtype, device=device
        )
        self.attn = SigVQAttention(
            hidden_size, num_attention_heads, attention_bias, dtype, device, operations
        )
        self.mlp = SigVQMLP(hidden_size, intermediate_size, dtype, device, operations)

    def forward(self, hidden_states, transformer_options={}):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), transformer_options
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class SigVQPatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_size,
        patch_size,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = operations.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            dtype=dtype,
            device=device,
        )

    def forward(self, pixel_values):
        return self.proj(pixel_values).flatten(2).transpose(1, 2)


class SigVQEmbeddings(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        hidden_size,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        num_positions = (image_size // patch_size) ** 2
        self.position_embedding = operations.Embedding(
            num_positions, hidden_size, dtype=dtype, device=device
        )

    def forward(self, hidden_states, grid_height, grid_width):
        batch_size = hidden_states.shape[0]
        position_embedding = comfy.ops.cast_to_input(
            self.position_embedding.weight, hidden_states
        )
        hidden_size = position_embedding.shape[1]
        original_size = int(position_embedding.shape[0] ** 0.5)
        position_embedding = position_embedding.reshape(
            original_size, original_size, hidden_size
        )
        position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(0).float()
        height_coordinates = torch.arange(
            grid_height, device=hidden_states.device, dtype=torch.float32
        )
        width_coordinates = torch.arange(
            grid_width, device=hidden_states.device, dtype=torch.float32
        )
        height_coordinates, width_coordinates = torch.meshgrid(
            height_coordinates, width_coordinates, indexing="ij"
        )
        grid = torch.stack(
            (
                ((width_coordinates.flatten() + 0.5) / grid_width) * 2 - 1,
                ((height_coordinates.flatten() + 0.5) / grid_height) * 2 - 1,
            ),
            dim=-1,
        )
        grid = grid.reshape(1, grid_height * grid_width, 1, 2).expand(
            batch_size, -1, -1, -1
        )
        position_embedding = F.grid_sample(
            position_embedding.expand(batch_size, -1, -1, -1),
            grid,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        )
        return hidden_states + position_embedding.squeeze(-1).transpose(1, 2).to(
            hidden_states.dtype
        )


class SigVQQuantizer(nn.Module):
    def __init__(
        self, num_embeddings, embedding_dim, dtype=None, device=None, operations=None
    ):
        super().__init__()
        self.embedding = operations.Embedding(
            num_embeddings, embedding_dim, dtype=dtype, device=device
        )

    def forward(self, hidden_states):
        hidden_states = hidden_states.permute(0, 2, 3, 1).contiguous()
        hidden_states = F.normalize(
            hidden_states.reshape(-1, hidden_states.shape[-1]), p=2, dim=-1
        )
        embedding = comfy.ops.cast_to_input(self.embedding.weight, hidden_states)
        embedding = F.normalize(embedding, p=2, dim=-1)
        distances = (
            torch.sum(hidden_states**2, dim=1, keepdim=True)
            + torch.sum(embedding**2, dim=1)
            - 2 * torch.matmul(hidden_states, embedding.t())
        )
        return torch.argmin(distances, dim=1)


class LinearSilu(nn.Module):
    def __init__(self, dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.proj = operations.Linear(dim, dim, bias=True, dtype=dtype, device=device)

    def forward(self, hidden_states):
        return F.silu(self.proj(hidden_states))


class PriorProjector(nn.Module):
    def __init__(self, dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.net = nn.ModuleList(
            [
                LinearSilu(dim, dtype, device, operations),
                nn.Identity(),
                operations.Linear(dim, dim, bias=True, dtype=dtype, device=device),
            ]
        )

    def forward(self, hidden_states):
        for layer in self.net:
            hidden_states = layer(hidden_states)
        return hidden_states


class SigVQ(nn.Module):
    def __init__(
        self,
        image_size=2048,
        patch_size=16,
        in_channels=3,
        hidden_size=1536,
        intermediate_size=6144,
        num_hidden_layers=40,
        num_attention_heads=16,
        attention_bias=True,
        norm_eps=1e-6,
        codebook_size=16384,
        codebook_embed_dim=2048,
        semantic_embed_dim=4096,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.visual = nn.Module()
        self.visual.patch_embed = SigVQPatchEmbed(
            in_channels, hidden_size, patch_size, dtype, device, operations
        )
        self.visual.embeddings = SigVQEmbeddings(
            image_size, patch_size, hidden_size, dtype, device, operations
        )
        self.visual.blocks = nn.ModuleList(
            [
                SigVQVisionBlock(
                    hidden_size,
                    intermediate_size,
                    num_attention_heads,
                    attention_bias,
                    norm_eps,
                    dtype,
                    device,
                    operations,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.vqmodel = nn.Module()
        self.vqmodel.quant_conv = operations.Conv2d(
            hidden_size, codebook_embed_dim, kernel_size=1, dtype=dtype, device=device
        )
        self.vqmodel.quantize = SigVQQuantizer(
            codebook_size, codebook_embed_dim, dtype, device, operations
        )
        self.prior_token_embedding = operations.Embedding(
            codebook_size, semantic_embed_dim, dtype=dtype, device=device
        )
        self.prior_projector = PriorProjector(
            semantic_embed_dim, dtype, device, operations
        )

    def forward(self, pixel_values=None, token_ids=None, transformer_options={}):
        if (pixel_values is None) == (token_ids is None):
            raise ValueError("provide exactly one of pixel_values or token_ids")
        if pixel_values is not None:
            if pixel_values.ndim != 4:
                raise ValueError(
                    "pixel_values must have 4 dimensions, got shape "
                    f"{tuple(pixel_values.shape)}"
                )
            height, width = pixel_values.shape[-2:]
            if height % self.patch_size != 0 or width % self.patch_size != 0:
                raise ValueError(
                    "image height and width must be divisible by "
                    f"{self.patch_size}, got {height}x{width}"
                )
            grid_height = height // self.patch_size
            grid_width = width // self.patch_size
            hidden_states = self.visual.patch_embed(pixel_values)
            hidden_states = self.visual.embeddings(
                hidden_states, grid_height, grid_width
            )
            for block in self.visual.blocks:
                hidden_states = block(hidden_states, transformer_options)
            hidden_states = hidden_states.transpose(1, 2).reshape(
                pixel_values.shape[0], self.hidden_size, grid_height, grid_width
            )
            hidden_states = self.vqmodel.quant_conv(hidden_states)
            token_ids = self.vqmodel.quantize(hidden_states).reshape(
                pixel_values.shape[0], -1
            )
        elif token_ids.ndim != 2:
            raise ValueError(
                "token_ids must have 2 dimensions, got shape "
                f"{tuple(token_ids.shape)}"
            )
        semantic_features = self.prior_projector(self.prior_token_embedding(token_ids))
        return semantic_features, token_ids
