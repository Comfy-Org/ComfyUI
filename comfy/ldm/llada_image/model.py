# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import comfy.ldm.common_dit
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.modules.attention import optimized_attention


ADALN_EMBED_DIM = 256
SEQUENCE_MULTIPLE = 32


@dataclass
class LLaDAImageSequence:
    features: list[torch.Tensor]
    position_ids: list[torch.Tensor]
    padding_masks: list[torch.Tensor]
    noise_masks: list[list[int]] | None = None


class TimestepEmbedder(nn.Module):
    def __init__(
        self,
        output_dim,
        hidden_dim=1024,
        frequency_embedding_dim=256,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            operations.Linear(
                frequency_embedding_dim,
                hidden_dim,
                bias=True,
                dtype=dtype,
                device=device,
            ),
            nn.SiLU(),
            operations.Linear(
                hidden_dim, output_dim, bias=True, dtype=dtype, device=device
            ),
        )
        self.frequency_embedding_dim = frequency_embedding_dim

    def forward(self, timestep, hidden_dtype):
        embedding = timestep_embedding(
            timestep, self.frequency_embedding_dim, time_factor=1.0
        )
        return self.mlp(embedding.to(dtype=hidden_dtype))


class RopeEmbedder(nn.Module):
    def __init__(self, theta, axes_dims):
        super().__init__()
        self.theta = theta
        self.axes_dims = axes_dims

    def forward(self, position_ids):
        matrices = []
        for axis, axis_dim in enumerate(self.axes_dims):
            scale = (
                torch.arange(
                    0, axis_dim, 2, dtype=torch.float32, device=position_ids.device
                )
                / axis_dim
            )
            omega = 1.0 / self.theta**scale
            angles = position_ids[..., axis].float().unsqueeze(-1) * omega
            matrices.append(
                torch.stack(
                    (
                        torch.cos(angles),
                        -torch.sin(angles),
                        torch.sin(angles),
                        torch.cos(angles),
                    ),
                    dim=-1,
                ).unflatten(-1, (2, 2))
            )
        return torch.cat(matrices, dim=-3).unsqueeze(1)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        norm_eps,
        qk_norm,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.heads = num_heads
        self.head_dim = dim // num_heads
        self.to_q = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.to_k = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.to_v = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.norm_q = (
            nn.RMSNorm(self.head_dim, eps=norm_eps, elementwise_affine=False)
            if qk_norm
            else nn.Identity()
        )
        self.norm_k = (
            nn.RMSNorm(self.head_dim, eps=norm_eps, elementwise_affine=False)
            if qk_norm
            else nn.Identity()
        )
        self.to_out = nn.ModuleList(
            [
                operations.Linear(dim, dim, bias=False, dtype=dtype, device=device),
                nn.Identity(),
            ]
        )

    def forward(self, hidden_states, attention_mask, frequencies, transformer_options):
        query = self.to_q(hidden_states).unflatten(-1, (self.heads, self.head_dim))
        key = self.to_k(hidden_states).unflatten(-1, (self.heads, self.head_dim))
        value = self.to_v(hidden_states).unflatten(-1, (self.heads, self.head_dim))
        query = self.norm_q(query).transpose(1, 2)
        key = self.norm_k(key).transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = apply_rope(query, key, frequencies)
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]
        hidden_states = optimized_attention(
            query,
            key,
            value,
            self.heads,
            mask=attention_mask,
            skip_reshape=True,
            transformer_options=transformer_options,
        )
        return self.to_out[0](hidden_states)


class FeedForward(nn.Module):
    def __init__(self, dim, dtype=None, device=None, operations=None):
        super().__init__()
        hidden_dim = int(dim / 3 * 8)
        self.w1 = operations.Linear(
            dim, hidden_dim, bias=False, dtype=dtype, device=device
        )
        self.w2 = operations.Linear(
            hidden_dim, dim, bias=False, dtype=dtype, device=device
        )
        self.w3 = operations.Linear(
            dim, hidden_dim, bias=False, dtype=dtype, device=device
        )

    def forward(self, hidden_states):
        return self.w2(F.silu(self.w1(hidden_states)) * self.w3(hidden_states))


def select_per_token(noisy_value, clean_value, noise_mask, sequence_length):
    noise_mask = noise_mask.unsqueeze(-1)
    return torch.where(
        noise_mask == 1,
        noisy_value.unsqueeze(1).expand(-1, sequence_length, -1),
        clean_value.unsqueeze(1).expand(-1, sequence_length, -1),
    )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        norm_eps,
        qk_norm,
        modulation,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.modulation = modulation
        self.attention = Attention(
            dim, num_heads, norm_eps, qk_norm, dtype, device, operations
        )
        self.feed_forward = FeedForward(dim, dtype, device, operations)
        self.attention_norm1 = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=False)
        self.ffn_norm1 = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=False)
        self.attention_norm2 = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=False)
        self.ffn_norm2 = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=False)
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                operations.Linear(
                    min(dim, ADALN_EMBED_DIM),
                    4 * dim,
                    bias=True,
                    dtype=dtype,
                    device=device,
                )
            )

    def forward(
        self,
        hidden_states,
        attention_mask,
        frequencies,
        adaln_input=None,
        noise_mask=None,
        adaln_noisy=None,
        adaln_clean=None,
        transformer_options={},
    ):
        if self.modulation:
            sequence_length = hidden_states.shape[1]
            if noise_mask is None:
                scale_msa, gate_msa, scale_mlp, gate_mlp = (
                    self.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
                )
                gate_msa = gate_msa.tanh()
                gate_mlp = gate_mlp.tanh()
                scale_msa = 1.0 + scale_msa
                scale_mlp = 1.0 + scale_mlp
            else:
                noisy_modulation = self.adaLN_modulation(adaln_noisy)
                clean_modulation = self.adaLN_modulation(adaln_clean)
                noisy_scale_msa, noisy_gate_msa, noisy_scale_mlp, noisy_gate_mlp = (
                    noisy_modulation.chunk(4, dim=1)
                )
                clean_scale_msa, clean_gate_msa, clean_scale_mlp, clean_gate_mlp = (
                    clean_modulation.chunk(4, dim=1)
                )
                scale_msa = select_per_token(
                    1.0 + noisy_scale_msa,
                    1.0 + clean_scale_msa,
                    noise_mask,
                    sequence_length,
                )
                scale_mlp = select_per_token(
                    1.0 + noisy_scale_mlp,
                    1.0 + clean_scale_mlp,
                    noise_mask,
                    sequence_length,
                )
                gate_msa = select_per_token(
                    noisy_gate_msa.tanh(),
                    clean_gate_msa.tanh(),
                    noise_mask,
                    sequence_length,
                )
                gate_mlp = select_per_token(
                    noisy_gate_mlp.tanh(),
                    clean_gate_mlp.tanh(),
                    noise_mask,
                    sequence_length,
                )

            attention_output = self.attention(
                self.attention_norm1(hidden_states) * scale_msa,
                attention_mask,
                frequencies,
                transformer_options,
            )
            hidden_states = hidden_states + gate_msa * self.attention_norm2(
                attention_output
            )
            hidden_states = hidden_states + gate_mlp * self.ffn_norm2(
                self.feed_forward(self.ffn_norm1(hidden_states) * scale_mlp)
            )
        else:
            attention_output = self.attention(
                self.attention_norm1(hidden_states),
                attention_mask,
                frequencies,
                transformer_options,
            )
            hidden_states = hidden_states + self.attention_norm2(attention_output)
            hidden_states = hidden_states + self.ffn_norm2(
                self.feed_forward(self.ffn_norm1(hidden_states))
            )
        return hidden_states


class FinalLayer(nn.Module):
    def __init__(self, dim, out_channels, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = operations.Linear(
            dim, out_channels, bias=True, dtype=dtype, device=device
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            operations.Linear(
                min(dim, ADALN_EMBED_DIM), dim, bias=True, dtype=dtype, device=device
            ),
        )

    def forward(
        self,
        hidden_states,
        adaln_input=None,
        noise_mask=None,
        adaln_noisy=None,
        adaln_clean=None,
    ):
        if noise_mask is None:
            scale = 1.0 + self.adaLN_modulation(adaln_input).unsqueeze(1)
        else:
            sequence_length = hidden_states.shape[1]
            noisy_scale = 1.0 + self.adaLN_modulation(adaln_noisy)
            clean_scale = 1.0 + self.adaLN_modulation(adaln_clean)
            scale = select_per_token(
                noisy_scale, clean_scale, noise_mask, sequence_length
            )
        return self.linear(self.norm_final(hidden_states) * scale)


class LLaDAImage(nn.Module):
    def __init__(
        self,
        all_patch_size=(1,),
        all_f_patch_size=(1,),
        in_channels=128,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        semantic_feat_dim=4096,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=(32, 48, 48),
        image_model=None,
        variant=None,
        dtype=None,
        device=None,
        operations=None,
        **kwargs,
    ):
        super().__init__()
        if len(all_patch_size) != len(all_f_patch_size):
            raise ValueError(
                "all_patch_size and all_f_patch_size must have the same length"
            )
        if dim % n_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")
        if dim // n_heads != sum(axes_dims):
            raise ValueError(
                "the attention head dimension must equal the sum of axes_dims"
            )
        self.dtype = dtype
        self.out_channels = in_channels
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.t_scale = t_scale

        self.all_x_embedder = nn.ModuleDict()
        self.all_final_layer = nn.ModuleDict()
        for patch_size, f_patch_size in zip(all_patch_size, all_f_patch_size):
            patch_key = f"{patch_size}-{f_patch_size}"
            patch_dim = f_patch_size * patch_size * patch_size * in_channels
            self.all_x_embedder[patch_key] = operations.Linear(
                patch_dim, dim, bias=True, dtype=dtype, device=device
            )
            self.all_final_layer[patch_key] = FinalLayer(
                dim, patch_dim, dtype, device, operations
            )

        def make_block(modulation):
            return TransformerBlock(
                dim, n_heads, norm_eps, qk_norm, modulation, dtype, device, operations
            )

        self.noise_refiner = nn.ModuleList(
            [make_block(True) for _ in range(n_refiner_layers)]
        )
        self.context_refiner = nn.ModuleList(
            [make_block(False) for _ in range(n_refiner_layers)]
        )
        self.sigvq_refiner = nn.ModuleList(
            [make_block(False) for _ in range(n_refiner_layers)]
        )
        self.layers = nn.ModuleList([make_block(True) for _ in range(n_layers)])
        self.t_embedder = TimestepEmbedder(
            min(dim, ADALN_EMBED_DIM), dtype=dtype, device=device, operations=operations
        )
        self.cap_embedder = nn.Sequential(
            nn.RMSNorm(cap_feat_dim, eps=norm_eps, elementwise_affine=False),
            operations.Linear(cap_feat_dim, dim, bias=True, dtype=dtype, device=device),
        )
        self.semantic_embedder = nn.Sequential(
            nn.RMSNorm(semantic_feat_dim, eps=norm_eps, elementwise_affine=False),
            operations.Linear(
                semantic_feat_dim, dim, bias=True, dtype=dtype, device=device
            ),
        )
        self.sigvq_embedder = nn.Sequential(
            nn.RMSNorm(semantic_feat_dim, eps=norm_eps, elementwise_affine=False),
            operations.Linear(
                semantic_feat_dim, dim, bias=True, dtype=dtype, device=device
            ),
        )
        self.x_pad_token = nn.Parameter(torch.empty(1, dim, dtype=dtype, device=device))
        self.cap_pad_token = nn.Parameter(
            torch.empty(1, dim, dtype=dtype, device=device)
        )
        self.sigvq_pad_token = nn.Parameter(
            torch.empty(1, dim, dtype=dtype, device=device)
        )
        self.rope_embedder = RopeEmbedder(rope_theta, axes_dims)

    @staticmethod
    def create_coordinate_grid(size, start, device):
        axes = [
            torch.arange(
                start_value, start_value + span, dtype=torch.int32, device=device
            )
            for start_value, span in zip(start, size)
        ]
        return torch.stack(torch.meshgrid(axes, indexing="ij"), dim=-1)

    def patchify_image(self, image, patch_size, f_patch_size):
        original_size = tuple(image.shape[1:])
        image = comfy.ldm.common_dit.pad_to_patch_size(
            image.unsqueeze(0), (f_patch_size, patch_size, patch_size)
        ).squeeze(0)
        channels, frames, height, width = image.shape
        frame_tokens = frames // f_patch_size
        height_tokens = height // patch_size
        width_tokens = width // patch_size
        image = image.view(
            channels,
            frame_tokens,
            f_patch_size,
            height_tokens,
            patch_size,
            width_tokens,
            patch_size,
        )
        image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(
            frame_tokens * height_tokens * width_tokens,
            f_patch_size * patch_size * patch_size * channels,
        )
        return (
            image,
            (original_size, (frames, height, width)),
            (frame_tokens, height_tokens, width_tokens),
        )

    def pad_with_ids(
        self, features, position_grid_size, position_start, noise_value=None
    ):
        original_length = len(features)
        padding_length = (-original_length) % SEQUENCE_MULTIPLE
        padded_length = original_length + padding_length
        device = features.device
        position_ids = self.create_coordinate_grid(
            position_grid_size, position_start, device
        ).flatten(0, 2)
        if padding_length > 0:
            padding_position_ids = self.create_coordinate_grid(
                (1, 1, 1), (0, 0, 0), device
            ).flatten(0, 2)
            position_ids = torch.cat(
                (position_ids, padding_position_ids.repeat(padding_length, 1)), dim=0
            )
            features = torch.cat(
                (features, features[-1:].repeat(padding_length, 1)), dim=0
            )
            padding_mask = torch.cat(
                (
                    torch.zeros(original_length, dtype=torch.bool, device=device),
                    torch.ones(padding_length, dtype=torch.bool, device=device),
                )
            )
        else:
            padding_mask = torch.zeros(original_length, dtype=torch.bool, device=device)
        noise_mask = [noise_value] * padded_length if noise_value is not None else None
        return features, position_ids, padding_mask, padded_length, noise_mask

    @staticmethod
    def batch_sequences(
        features, position_ids, inner_padding_masks, pad_token, noise_masks=None
    ):
        sequence_lengths = [len(item) for item in features]
        max_sequence_length = max(sequence_lengths)
        features = torch.cat(features, dim=0)
        inner_padding_mask = torch.cat(inner_padding_masks).unsqueeze(-1)
        features = torch.where(
            inner_padding_mask,
            comfy.ops.cast_to_input(pad_token, features, copy=False),
            features,
        )
        features = pad_sequence(
            list(features.split(sequence_lengths, dim=0)),
            batch_first=True,
            padding_value=0.0,
        )
        position_ids = pad_sequence(position_ids, batch_first=True, padding_value=0)[
            :, : features.shape[1]
        ]

        attention_mask = None
        if not all(length == max_sequence_length for length in sequence_lengths):
            attention_mask = torch.zeros(
                (len(sequence_lengths), max_sequence_length),
                dtype=torch.bool,
                device=features.device,
            )
            for batch_index, sequence_length in enumerate(sequence_lengths):
                attention_mask[batch_index, :sequence_length] = True

        noise_mask = None
        if noise_masks is not None:
            noise_mask = pad_sequence(
                [
                    torch.tensor(mask, dtype=torch.long, device=features.device)
                    for mask in noise_masks
                ],
                batch_first=True,
                padding_value=0,
            )[:, : features.shape[1]]
        return features, position_ids, attention_mask, sequence_lengths, noise_mask

    def unpatchify(
        self, hidden_states, sizes, patch_size, f_patch_size, image_offsets=None
    ):
        outputs = []
        for batch_index, batch_hidden_states in enumerate(hidden_states):
            if image_offsets is None:
                batch_sizes = [sizes[batch_index]]
                image_hidden_states = batch_hidden_states
            else:
                batch_sizes = sizes[batch_index]
                start, end = image_offsets[batch_index]
                image_hidden_states = batch_hidden_states[start:end]

            current_offset = 0
            for original_size, padded_size in batch_sizes:
                frames, height, width = padded_size
                token_count = (
                    frames // f_patch_size
                    * (height // patch_size)
                    * (width // patch_size)
                )
                padding_length = (-token_count) % SEQUENCE_MULTIPLE
                output = (
                    image_hidden_states[
                        current_offset : current_offset + token_count
                    ]
                    .view(
                        frames // f_patch_size,
                        height // patch_size,
                        width // patch_size,
                        f_patch_size,
                        patch_size,
                        patch_size,
                        self.out_channels,
                    )
                    .permute(6, 0, 3, 1, 4, 2, 5)
                    .reshape(self.out_channels, frames, height, width)
                )
                output = output[
                    :, : original_size[0], : original_size[1], : original_size[2]
                ]
                current_offset += token_count + padding_length
            outputs.append(output)
        return outputs

    def prepare_t2i_sequences(
        self, x, cap_feats, glm_features, patch_size, f_patch_size
    ):
        image_sequence = LLaDAImageSequence([], [], [])
        cap_sequence = LLaDAImageSequence([], [], []) if cap_feats is not None else None
        glm_sequence = (
            LLaDAImageSequence([], [], []) if glm_features is not None else None
        )
        image_sizes = []

        for batch_index, latent in enumerate(x):
            position_cursor = 1
            if cap_sequence is not None:
                padded_features, position_ids, padding_mask, sequence_length, _ = (
                    self.pad_with_ids(
                        cap_feats[batch_index],
                        (len(cap_feats[batch_index]), 1, 1),
                        (position_cursor, 0, 0),
                    )
                )
                cap_sequence.features.append(padded_features)
                cap_sequence.position_ids.append(position_ids)
                cap_sequence.padding_masks.append(padding_mask)
                position_cursor += sequence_length

            if glm_sequence is not None:
                padded_features, position_ids, padding_mask, sequence_length, _ = (
                    self.pad_with_ids(
                        glm_features[batch_index],
                        (len(glm_features[batch_index]), 1, 1),
                        (position_cursor, 0, 0),
                    )
                )
                glm_sequence.features.append(padded_features)
                glm_sequence.position_ids.append(position_ids)
                glm_sequence.padding_masks.append(padding_mask)
                position_cursor += sequence_length

            patches, image_size, token_grid_size = self.patchify_image(
                latent, patch_size, f_patch_size
            )
            padded_features, position_ids, padding_mask, _, _ = self.pad_with_ids(
                patches, token_grid_size, (position_cursor, 0, 0)
            )
            image_sequence.features.append(padded_features)
            image_sequence.position_ids.append(position_ids)
            image_sequence.padding_masks.append(padding_mask)
            image_sizes.append(image_size)
        return image_sequence, cap_sequence, glm_sequence, image_sizes

    def prepare_editing_sequences(
        self, x, cap_feats, glm_cap_feats, source_latents, patch_size, f_patch_size
    ):
        image_sequence = LLaDAImageSequence([], [], [], [])
        cap_sequence = LLaDAImageSequence([], [], [], [])
        sigvq_sequence = LLaDAImageSequence([], [], [], [])
        image_sizes = []
        image_offsets = []

        for batch_index, latent in enumerate(x):
            cap_end_positions = []
            position_cursor = 1
            batch_cap_features = []
            batch_cap_positions = []
            batch_cap_padding = []
            batch_cap_noise = []
            for noise_value in (0, 1):
                padded_features, position_ids, padding_mask, _, noise_mask = (
                    self.pad_with_ids(
                        cap_feats[batch_index],
                        (len(cap_feats[batch_index]), 1, 1),
                        (position_cursor, 0, 0),
                        noise_value,
                    )
                )
                batch_cap_features.append(padded_features)
                batch_cap_positions.append(position_ids)
                batch_cap_padding.append(padding_mask)
                batch_cap_noise.extend(noise_mask)
                position_cursor += len(cap_feats[batch_index])
                cap_end_positions.append(position_cursor)
                position_cursor += 2

            batch_image_features = []
            batch_image_sizes = []
            batch_image_positions = []
            batch_image_padding = []
            batch_image_noise = []
            for image, position_start, noise_value in zip(
                (source_latents[batch_index], latent), cap_end_positions, (0, 1)
            ):
                patches, image_size, token_grid_size = self.patchify_image(
                    image, patch_size, f_patch_size
                )
                padded_features, position_ids, padding_mask, _, noise_mask = (
                    self.pad_with_ids(
                        patches, token_grid_size, (position_start, 0, 0), noise_value
                    )
                )
                batch_image_features.append(padded_features)
                batch_image_sizes.append(image_size)
                batch_image_positions.append(position_ids)
                batch_image_padding.append(padding_mask)
                batch_image_noise.extend(noise_mask)

            batch_cap_features = torch.cat(batch_cap_features, dim=0)
            batch_image_features = torch.cat(batch_image_features, dim=0)
            cap_sequence.features.append(batch_cap_features)
            cap_sequence.position_ids.append(torch.cat(batch_cap_positions, dim=0))
            cap_sequence.padding_masks.append(torch.cat(batch_cap_padding, dim=0))
            cap_sequence.noise_masks.append(batch_cap_noise)
            image_sequence.features.append(batch_image_features)
            image_sequence.position_ids.append(torch.cat(batch_image_positions, dim=0))
            image_sequence.padding_masks.append(torch.cat(batch_image_padding, dim=0))
            image_sequence.noise_masks.append(batch_image_noise)
            image_sizes.append(batch_image_sizes)
            image_offsets.append(
                (
                    len(batch_cap_features),
                    len(batch_cap_features) + len(batch_image_features),
                )
            )

            padded_features, position_ids, padding_mask, _, noise_mask = (
                self.pad_with_ids(
                    glm_cap_feats[batch_index],
                    (len(glm_cap_feats[batch_index]), 1, 1),
                    (len(batch_cap_features) + len(batch_image_features) + 1, 0, 0),
                    0,
                )
            )
            sigvq_sequence.features.append(padded_features)
            sigvq_sequence.position_ids.append(position_ids)
            sigvq_sequence.padding_masks.append(padding_mask)
            sigvq_sequence.noise_masks.append(noise_mask)
        return image_sequence, cap_sequence, sigvq_sequence, image_sizes, image_offsets

    @staticmethod
    def merge_padded_sequences(
        feature_groups, position_groups, length_groups, noise_mask_groups=None
    ):
        batch_size = feature_groups[0].shape[0]
        merged_features = []
        merged_positions = []
        merged_noise_masks = [] if noise_mask_groups is not None else None

        for batch_index in range(batch_size):
            merged_features.append(
                torch.cat(
                    [
                        features[batch_index, : lengths[batch_index]]
                        for features, lengths in zip(feature_groups, length_groups)
                    ],
                    dim=0,
                )
            )
            merged_positions.append(
                torch.cat(
                    [
                        positions[batch_index, : lengths[batch_index]]
                        for positions, lengths in zip(position_groups, length_groups)
                    ],
                    dim=0,
                )
            )
            if merged_noise_masks is not None:
                merged_noise_masks.append(
                    torch.cat(
                        [
                            noise_masks[batch_index, : lengths[batch_index]]
                            for noise_masks, lengths in zip(
                                noise_mask_groups, length_groups
                            )
                        ],
                        dim=0,
                    )
                )

        merged_lengths = [len(features) for features in merged_features]
        merged_features = pad_sequence(
            merged_features, batch_first=True, padding_value=0.0
        )
        merged_positions = pad_sequence(
            merged_positions, batch_first=True, padding_value=0
        )
        attention_mask = None
        max_length = max(merged_lengths)
        if not all(length == max_length for length in merged_lengths):
            attention_mask = torch.zeros(
                (batch_size, max_length),
                dtype=torch.bool,
                device=merged_features.device,
            )
            for batch_index, sequence_length in enumerate(merged_lengths):
                attention_mask[batch_index, :sequence_length] = True

        noise_mask = None
        if merged_noise_masks is not None:
            noise_mask = pad_sequence(
                merged_noise_masks, batch_first=True, padding_value=0
            )[:, : merged_features.shape[1]]
        return (
            merged_features,
            merged_positions,
            attention_mask,
            merged_lengths,
            noise_mask,
        )

    @staticmethod
    def feature_list(features, mask=None):
        if features is None:
            return None
        if isinstance(features, list):
            return features
        if mask is None:
            return list(features.unbind(dim=0))
        return [
            batch_features[batch_mask.bool()]
            for batch_features, batch_mask in zip(features, mask)
        ]

    def run_blocks(
        self,
        blocks,
        features,
        attention_mask,
        positions,
        transformer_options,
        adaln_input=None,
        noise_mask=None,
        adaln_noisy=None,
        adaln_clean=None,
    ):
        frequencies = self.rope_embedder(positions)
        for block in blocks:
            features = block(
                features,
                attention_mask,
                frequencies,
                adaln_input,
                noise_mask,
                adaln_noisy,
                adaln_clean,
                transformer_options,
            )
        return features

    def forward(
        self,
        x,
        timestep,
        context=None,
        attention_mask=None,
        semantic_features=None,
        semantic_mask=None,
        source_latents=None,
        transformer_options={},
        **kwargs,
    ):
        if source_latents is None and context is None and semantic_features is None:
            raise ValueError(
                "text-to-image inference requires context or semantic_features"
            )
        if source_latents is not None and (
            context is None or semantic_features is None
        ):
            raise ValueError(
                "editing requires context, semantic_features, and source_latents"
            )
        patch_size = self.all_patch_size[0]
        f_patch_size = self.all_f_patch_size[0]
        patch_key = f"{patch_size}-{f_patch_size}"
        cap_feats = self.feature_list(context, attention_mask)
        glm_cap_feats = self.feature_list(semantic_features, semantic_mask)
        x_list = [latent.unsqueeze(1) for latent in x.unbind(dim=0)]
        source_list = None
        if source_latents is not None:
            if isinstance(source_latents, list):
                source_list = source_latents
            else:
                source_list = list(source_latents.unbind(dim=0))
            source_list = [
                latent.unsqueeze(1) if latent.ndim == 3 else latent
                for latent in source_list
            ]

        batch_size = len(x_list)
        is_editing = source_list is not None
        if is_editing:
            if timestep.shape[0] == 1:
                timestep = timestep.repeat(batch_size)
            dual_timestep = torch.cat((timestep, torch.zeros_like(timestep)), dim=0)
            dual_embedding = self.t_embedder(
                dual_timestep.abs() * self.t_scale, x.dtype
            )
            noisy_embedding = dual_embedding[:batch_size]
            clean_embedding = dual_embedding[batch_size:]
            image_sequence, cap_sequence, sigvq_sequence, image_sizes, image_offsets = (
                self.prepare_editing_sequences(
                    x_list,
                    cap_feats,
                    glm_cap_feats,
                    source_list,
                    patch_size,
                    f_patch_size,
                )
            )
            adaln_input = None
        else:
            adaln_input = self.t_embedder(timestep * self.t_scale, x.dtype)
            noisy_embedding = None
            clean_embedding = None
            image_offsets = None
            glm_features = (
                [
                    self.semantic_embedder(batch_features)
                    for batch_features in glm_cap_feats
                ]
                if glm_cap_feats is not None
                else None
            )
            image_sequence, cap_sequence, glm_sequence, image_sizes = (
                self.prepare_t2i_sequences(
                    x_list, cap_feats, glm_features, patch_size, f_patch_size
                )
            )

        image_lengths = [len(features) for features in image_sequence.features]
        image_features = self.all_x_embedder[patch_key](
            torch.cat(image_sequence.features, dim=0)
        )
        (
            image_features,
            image_positions,
            image_attention_mask,
            image_lengths,
            image_noise_mask,
        ) = self.batch_sequences(
            list(image_features.split(image_lengths, dim=0)),
            image_sequence.position_ids,
            image_sequence.padding_masks,
            self.x_pad_token,
            image_sequence.noise_masks,
        )
        image_features = self.run_blocks(
            self.noise_refiner,
            image_features,
            image_attention_mask,
            image_positions,
            transformer_options,
            adaln_input,
            image_noise_mask,
            noisy_embedding,
            clean_embedding,
        )

        if is_editing:
            cap_lengths = [len(features) for features in cap_sequence.features]
            cap_features = self.cap_embedder(torch.cat(cap_sequence.features, dim=0))
            (
                cap_features,
                cap_positions,
                cap_attention_mask,
                cap_lengths,
                cap_noise_mask,
            ) = self.batch_sequences(
                list(cap_features.split(cap_lengths, dim=0)),
                cap_sequence.position_ids,
                cap_sequence.padding_masks,
                self.cap_pad_token,
                cap_sequence.noise_masks,
            )
            cap_features = self.run_blocks(
                self.context_refiner,
                cap_features,
                cap_attention_mask,
                cap_positions,
                transformer_options,
            )

            sigvq_lengths = [len(features) for features in sigvq_sequence.features]
            sigvq_features = self.sigvq_embedder(
                torch.cat(sigvq_sequence.features, dim=0)
            )
            (
                sigvq_features,
                sigvq_positions,
                sigvq_attention_mask,
                sigvq_lengths,
                sigvq_noise_mask,
            ) = self.batch_sequences(
                list(sigvq_features.split(sigvq_lengths, dim=0)),
                sigvq_sequence.position_ids,
                sigvq_sequence.padding_masks,
                self.sigvq_pad_token,
                sigvq_sequence.noise_masks,
            )
            if sigvq_features.shape[1] > 0:
                sigvq_features = self.run_blocks(
                    self.sigvq_refiner,
                    sigvq_features,
                    sigvq_attention_mask,
                    sigvq_positions,
                    transformer_options,
                )
            (
                unified_features,
                unified_positions,
                unified_attention_mask,
                _,
                unified_noise_mask,
            ) = self.merge_padded_sequences(
                (cap_features, image_features, sigvq_features),
                (cap_positions, image_positions, sigvq_positions),
                (cap_lengths, image_lengths, sigvq_lengths),
                (cap_noise_mask, image_noise_mask, sigvq_noise_mask),
            )
        else:
            condition_feature_groups = []
            condition_position_groups = []
            condition_length_groups = []
            if cap_sequence is not None:
                cap_lengths = [len(features) for features in cap_sequence.features]
                cap_features = self.cap_embedder(
                    torch.cat(cap_sequence.features, dim=0)
                )
                cap_features, cap_positions, _, cap_lengths, _ = self.batch_sequences(
                    list(cap_features.split(cap_lengths, dim=0)),
                    cap_sequence.position_ids,
                    cap_sequence.padding_masks,
                    self.cap_pad_token,
                )
                condition_feature_groups.append(cap_features)
                condition_position_groups.append(cap_positions)
                condition_length_groups.append(cap_lengths)

            if glm_sequence is not None:
                glm_lengths = [len(features) for features in glm_sequence.features]
                glm_features = torch.cat(glm_sequence.features, dim=0)
                glm_features, glm_positions, _, glm_lengths, _ = self.batch_sequences(
                    list(glm_features.split(glm_lengths, dim=0)),
                    glm_sequence.position_ids,
                    glm_sequence.padding_masks,
                    self.cap_pad_token,
                )
                condition_feature_groups.append(glm_features)
                condition_position_groups.append(glm_positions)
                condition_length_groups.append(glm_lengths)

            (
                condition_features,
                condition_positions,
                condition_attention_mask,
                condition_lengths,
                _,
            ) = self.merge_padded_sequences(
                tuple(condition_feature_groups),
                tuple(condition_position_groups),
                tuple(condition_length_groups),
            )
            condition_features = self.run_blocks(
                self.context_refiner,
                condition_features,
                condition_attention_mask,
                condition_positions,
                transformer_options,
            )
            (
                unified_features,
                unified_positions,
                unified_attention_mask,
                _,
                unified_noise_mask,
            ) = self.merge_padded_sequences(
                (image_features, condition_features),
                (image_positions, condition_positions),
                (image_lengths, condition_lengths),
            )

        unified_features = self.run_blocks(
            self.layers,
            unified_features,
            unified_attention_mask,
            unified_positions,
            transformer_options,
            adaln_input,
            unified_noise_mask,
            noisy_embedding,
            clean_embedding,
        )
        if is_editing:
            unified_features = self.all_final_layer[patch_key](
                unified_features,
                noise_mask=unified_noise_mask,
                adaln_noisy=noisy_embedding,
                adaln_clean=clean_embedding,
            )
        else:
            unified_features = self.all_final_layer[patch_key](
                unified_features, adaln_input=adaln_input
            )

        output = self.unpatchify(
            list(unified_features.unbind(dim=0)),
            image_sizes,
            patch_size,
            f_patch_size,
            image_offsets,
        )
        return -torch.stack([batch_output.squeeze(1) for batch_output in output], dim=0)
