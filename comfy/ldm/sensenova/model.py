import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.patcher_extension
import comfy.utils
from comfy.ldm.common_dit import pad_to_patch_size
from comfy.ldm.flux.math import apply_rope1
from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.modules.diffusionmodules.mmdit import TimestepEmbedder

from .sampling import resolution_noise_scale


HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 12288
NUM_LAYERS = 42
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
MERGED_PATCH_SIZE = 32
VOCAB_SIZE = 151936


def _pad_to_merged_patch_size(value):
    height, width = value.shape[-2:]
    height_pad = max(16 - height, 0)
    width_pad = max(16 - width, 0)
    if height_pad or width_pad:
        value = F.pad(
            value,
            (0, width_pad, 0, height_pad),
            mode="replicate" if height > 0 and width > 0 else "constant",
        )
    return pad_to_patch_size(value, (MERGED_PATCH_SIZE, MERGED_PATCH_SIZE))


def _generation_batch_size(total_batch, prefix_batch):
    if prefix_batch < 1 or total_batch < 1 or total_batch % prefix_batch != 0:
        raise ValueError(
            "SenseNova generation batch must be a positive multiple of the prefix batch "
            f"(generation={total_batch}, prefix={prefix_batch})"
        )
    return total_batch // prefix_batch


def _match_prefix_batch(total_batch, text_input_ids, prefix_indexes, prefix_mask):
    prefix_batch = text_input_ids.shape[0]
    if prefix_batch > 0 and total_batch % prefix_batch:
        text_input_ids = comfy.utils.resize_to_batch_size(text_input_ids, total_batch)
        if prefix_indexes is not None:
            prefix_indexes = comfy.utils.resize_to_batch_size(
                prefix_indexes, total_batch
            )
        if prefix_mask is not None:
            prefix_mask = comfy.utils.resize_to_batch_size(prefix_mask, total_batch)
    return text_input_ids, prefix_indexes, prefix_mask


def _expand_prefix_batch(value, generation_batch):
    """Repeat each guidance branch's prefix KV for its generated variants."""
    if generation_batch == 1:
        return value
    prefix_batch = value.shape[0]
    return (
        value.unsqueeze(1)
        .expand(prefix_batch, generation_batch, *value.shape[1:])
        .reshape(prefix_batch * generation_batch, *value.shape[1:])
    )


def _prepare_llm_rope(positions, dim, theta, device, dtype):
    frequencies = theta ** (
        -torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim
    )
    positions = positions.to(device=device, dtype=torch.float32)
    if positions.ndim == 1:
        positions = positions.unsqueeze(0)
    angles = positions.unsqueeze(-1) * frequencies
    embedding = torch.cat((angles, angles), dim=-1).unsqueeze(1)
    return embedding.cos().to(dtype), embedding.sin().to(dtype)


def _prepare_mrope(indexes, device, dtype):
    return (
        _prepare_llm_rope(indexes[0], HEAD_DIM // 2, 5000000.0, device, dtype),
        _prepare_llm_rope(indexes[1], HEAD_DIM // 4, 10000.0, device, dtype),
        _prepare_llm_rope(indexes[2], HEAD_DIM // 4, 10000.0, device, dtype),
    )


def _apply_llm_rope(query, key, rope):
    cosine, sine = rope

    def rotate_half(value):
        first, second = value.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    # Keep this split-half RoPE on the reference PyTorch formula. The
    # comfy-kitchen CUDA kernel is selected automatically on CUDA 13 builds;
    # on Blackwell it can return finite but numerically incorrect values, which
    # corrupts the generated image without raising an execution error.
    return (
        query * cosine + rotate_half(query) * sine,
        key * cosine + rotate_half(key) * sine,
    )


def _apply_interleaved_rope(value, positions, theta):
    dim = value.shape[-1]
    frequencies = theta ** (
        -torch.arange(0, dim, 2, dtype=torch.float32, device=value.device) / dim
    )
    angles = (
        positions.to(device=value.device, dtype=torch.float32).unsqueeze(-1)
        * frequencies
    )
    cosine = angles.cos()
    sine = angles.sin()
    # comfy-kitchen acceleration backends use the canonical four-dimensional
    # input and six-dimensional rotation layout. SenseNova's vision patches
    # have no head axis, so add a singleton one instead of relying on the eager
    # backend's more permissive rank handling.
    rotation = torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(
        1, 1, *angles.shape, 2, 2
    )
    return apply_rope1(value.float().unsqueeze(1), rotation).squeeze(1)


class VisionEmbeddings(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        self.patch_embedding = operations.Conv2d(
            3, 1024, kernel_size=16, stride=16, device=device, dtype=dtype
        )
        self.dense_embedding = operations.Conv2d(
            1024, HIDDEN_SIZE, kernel_size=2, stride=2, device=device, dtype=dtype
        )
        self.gelu = nn.GELU()

    def forward(self, image):
        patches = self.gelu(self.patch_embedding(image))
        batch, channels, height, width = patches.shape
        patches = patches.flatten(2).transpose(1, 2)
        indexes = torch.arange(height * width, device=patches.device)
        x_positions = indexes % width
        y_positions = indexes // width
        first = _apply_interleaved_rope(
            patches[..., : channels // 2], x_positions, 10000.0
        )
        second = _apply_interleaved_rope(
            patches[..., channels // 2 :], y_positions, 10000.0
        )
        patches = torch.cat((first, second), dim=-1).to(image.dtype)
        patches = patches.transpose(1, 2).reshape(batch, channels, height, width)
        patches = self.dense_embedding(patches)
        return patches.flatten(2).transpose(1, 2)


class VisionModel(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        self.embeddings = VisionEmbeddings(
            device=device, dtype=dtype, operations=operations
        )

    def forward(self, image):
        return self.embeddings(image)


class MLP(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        self.gate_proj = operations.Linear(
            HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False, device=device, dtype=dtype
        )
        self.up_proj = operations.Linear(
            HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False, device=device, dtype=dtype
        )
        self.down_proj = operations.Linear(
            INTERMEDIATE_SIZE, HIDDEN_SIZE, bias=False, device=device, dtype=dtype
        )

    def forward(self, hidden_states):
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class Attention(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        self.q_proj = operations.Linear(
            HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, bias=False, device=device, dtype=dtype
        )
        self.q_proj_mot_gen = operations.Linear(
            HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, bias=False, device=device, dtype=dtype
        )
        self.k_proj = operations.Linear(
            HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False, device=device, dtype=dtype
        )
        self.k_proj_mot_gen = operations.Linear(
            HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False, device=device, dtype=dtype
        )
        self.v_proj = operations.Linear(
            HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False, device=device, dtype=dtype
        )
        self.v_proj_mot_gen = operations.Linear(
            HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False, device=device, dtype=dtype
        )
        self.o_proj = operations.Linear(
            NUM_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False, device=device, dtype=dtype
        )
        self.o_proj_mot_gen = operations.Linear(
            NUM_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False, device=device, dtype=dtype
        )

        self.q_norm = operations.RMSNorm(
            HEAD_DIM // 2, eps=1e-6, device=device, dtype=dtype
        )
        self.q_norm_mot_gen = operations.RMSNorm(
            HEAD_DIM // 2, eps=1e-6, device=device, dtype=dtype
        )
        self.q_norm_hw = operations.RMSNorm(
            HEAD_DIM // 2, eps=1e-6, device=device, dtype=dtype
        )
        self.q_norm_hw_mot_gen = operations.RMSNorm(
            HEAD_DIM // 2, eps=1e-6, device=device, dtype=dtype
        )
        self.k_norm = operations.RMSNorm(
            HEAD_DIM // 2, eps=1e-6, device=device, dtype=dtype
        )
        self.k_norm_mot_gen = operations.RMSNorm(
            HEAD_DIM // 2, eps=1e-6, device=device, dtype=dtype
        )
        self.k_norm_hw = operations.RMSNorm(
            HEAD_DIM // 2, eps=1e-6, device=device, dtype=dtype
        )
        self.k_norm_hw_mot_gen = operations.RMSNorm(
            HEAD_DIM // 2, eps=1e-6, device=device, dtype=dtype
        )

    def _project(self, hidden_states, rope, generation):
        batch, length, _ = hidden_states.shape
        if generation:
            query = self.q_proj_mot_gen(hidden_states).view(
                batch, length, NUM_HEADS, HEAD_DIM
            )
            key = self.k_proj_mot_gen(hidden_states).view(
                batch, length, NUM_KV_HEADS, HEAD_DIM
            )
            value = (
                self.v_proj_mot_gen(hidden_states)
                .view(batch, length, NUM_KV_HEADS, HEAD_DIM)
                .transpose(1, 2)
            )
            query_t, query_hw = query.chunk(2, dim=-1)
            key_t, key_hw = key.chunk(2, dim=-1)
            query_t = self.q_norm_mot_gen(query_t).transpose(1, 2)
            query_hw = self.q_norm_hw_mot_gen(query_hw).transpose(1, 2)
            key_t = self.k_norm_mot_gen(key_t).transpose(1, 2)
            key_hw = self.k_norm_hw_mot_gen(key_hw).transpose(1, 2)
        else:
            query = self.q_proj(hidden_states).view(batch, length, NUM_HEADS, HEAD_DIM)
            key = self.k_proj(hidden_states).view(batch, length, NUM_KV_HEADS, HEAD_DIM)
            value = (
                self.v_proj(hidden_states)
                .view(batch, length, NUM_KV_HEADS, HEAD_DIM)
                .transpose(1, 2)
            )
            query_t, query_hw = query.chunk(2, dim=-1)
            key_t, key_hw = key.chunk(2, dim=-1)
            query_t = self.q_norm(query_t).transpose(1, 2)
            query_hw = self.q_norm_hw(query_hw).transpose(1, 2)
            key_t = self.k_norm(key_t).transpose(1, 2)
            key_hw = self.k_norm_hw(key_hw).transpose(1, 2)

        query_h, query_w = query_hw.chunk(2, dim=-1)
        key_h, key_w = key_hw.chunk(2, dim=-1)
        query_t, key_t = _apply_llm_rope(query_t, key_t, rope[0])
        query_h, key_h = _apply_llm_rope(query_h, key_h, rope[1])
        query_w, key_w = _apply_llm_rope(query_w, key_w, rope[2])
        query = torch.cat((query_t, query_h, query_w), dim=-1)
        key = torch.cat((key_t, key_h, key_w), dim=-1)
        return query, key, value

    def forward_prefix(
        self, hidden_states, rope, attention_mask, transformer_options
    ):
        query, key, value = self._project(hidden_states, rope, False)
        output = optimized_attention(
            query,
            key,
            value,
            NUM_HEADS,
            mask=attention_mask,
            skip_reshape=True,
            transformer_options=transformer_options,
            enable_gqa=True,
        )
        return self.o_proj(output), key, value

    def forward_generation(
        self, hidden_states, rope, prefix_key, prefix_value, transformer_options
    ):
        query, key, value = self._project(hidden_states, rope, True)
        key = torch.cat((prefix_key, key), dim=2)
        value = torch.cat((prefix_value, value), dim=2)
        output = optimized_attention(
            query,
            key,
            value,
            NUM_HEADS,
            mask=None,
            skip_reshape=True,
            transformer_options=transformer_options,
            enable_gqa=True,
        )
        return self.o_proj_mot_gen(output)


class DecoderLayer(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        self.self_attn = Attention(device=device, dtype=dtype, operations=operations)
        self.mlp = MLP(device=device, dtype=dtype, operations=operations)
        self.mlp_mot_gen = MLP(device=device, dtype=dtype, operations=operations)
        self.input_layernorm = operations.RMSNorm(
            HIDDEN_SIZE, eps=1e-6, device=device, dtype=dtype
        )
        self.input_layernorm_mot_gen = operations.RMSNorm(
            HIDDEN_SIZE, eps=1e-6, device=device, dtype=dtype
        )
        self.post_attention_layernorm = operations.RMSNorm(
            HIDDEN_SIZE, eps=1e-6, device=device, dtype=dtype
        )
        self.post_attention_layernorm_mot_gen = operations.RMSNorm(
            HIDDEN_SIZE, eps=1e-6, device=device, dtype=dtype
        )

    def forward_prefix(self, prefix, prefix_rope, prefix_mask, transformer_options):
        prefix_attention, prefix_key, prefix_value = self.self_attn.forward_prefix(
            self.input_layernorm(prefix),
            prefix_rope,
            prefix_mask,
            transformer_options,
        )
        prefix = prefix + prefix_attention
        prefix = prefix + self.mlp(self.post_attention_layernorm(prefix))
        return prefix, prefix_key, prefix_value

    def forward_generation(
        self, image, image_rope, prefix_key, prefix_value, transformer_options
    ):
        image_attention = self.self_attn.forward_generation(
            self.input_layernorm_mot_gen(image),
            image_rope,
            prefix_key,
            prefix_value,
            transformer_options,
        )
        image = image + image_attention
        image = image + self.mlp_mot_gen(self.post_attention_layernorm_mot_gen(image))
        return image


class LanguageBackbone(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        self.embed_tokens = operations.Embedding(
            VOCAB_SIZE, HIDDEN_SIZE, padding_idx=151643, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList(
            DecoderLayer(device=device, dtype=dtype, operations=operations)
            for _ in range(NUM_LAYERS)
        )
        self.norm = operations.RMSNorm(
            HIDDEN_SIZE, eps=1e-6, device=device, dtype=dtype
        )
        self.norm_mot_gen = operations.RMSNorm(
            HIDDEN_SIZE, eps=1e-6, device=device, dtype=dtype
        )


class LanguageModel(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        self.model = LanguageBackbone(device=device, dtype=dtype, operations=operations)


class ConvDecoder(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        self.ps1 = nn.PixelShuffle(2)
        self.conv1 = operations.Conv2d(
            1024, 1024, kernel_size=3, padding=1, device=device, dtype=dtype
        )
        self.act1 = nn.GELU()
        self.ps2 = nn.PixelShuffle(2)
        self.conv2 = operations.Conv2d(
            256, 192, kernel_size=3, padding=1, device=device, dtype=dtype
        )
        self.ps3 = nn.PixelShuffle(8)

    def forward(self, hidden_states):
        hidden_states = self.act1(self.conv1(self.ps1(hidden_states)))
        return self.ps3(self.conv2(self.ps2(hidden_states)))


class SenseNovaU15(nn.Module):
    def __init__(
        self, image_model=None, dtype=None, device=None, operations=None, **kwargs
    ):
        super().__init__()
        self.dtype = dtype
        self.vision_model = VisionModel(
            device=device, dtype=dtype, operations=operations
        )
        self.language_model = LanguageModel(
            device=device, dtype=dtype, operations=operations
        )
        self.fm_modules = nn.ModuleDict(
            {
                "vision_model_mot_gen": VisionModel(
                    device=device, dtype=dtype, operations=operations
                ),
                "timestep_embedder": TimestepEmbedder(
                    HIDDEN_SIZE, device=device, dtype=dtype, operations=operations
                ),
                "fm_head": ConvDecoder(
                    device=device, dtype=dtype, operations=operations
                ),
                "noise_scale_embedder": TimestepEmbedder(
                    HIDDEN_SIZE, device=device, dtype=dtype, operations=operations
                ),
            }
        )

    def forward(self, x, timesteps, context=None, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options
            ),
        ).execute(x, timesteps, context, transformer_options, **kwargs)

    def _prepare_prefix(
        self, text_input_ids, reference_images, prefix_indexes, prefix_mask
    ):
        prefix = self.language_model.model.embed_tokens(text_input_ids)
        if reference_images:
            reference_embeds = [
                self.vision_model(_pad_to_merged_patch_size(reference))
                for reference in reference_images
            ]
            selected = text_input_ids == 151669
            prefix = prefix.clone()
            prefix[selected] = torch.cat(reference_embeds, dim=1).reshape(
                -1, HIDDEN_SIZE
            )

        prefix_length = text_input_ids.shape[1]
        if prefix_indexes is None:
            prefix_positions = torch.arange(
                prefix_length, dtype=torch.long, device=prefix.device
            )
            zeros = torch.zeros_like(prefix_positions)
            prefix_indexes = torch.stack((prefix_positions, zeros, zeros))
            prefix_mask = torch.full(
                (prefix_length, prefix_length),
                float("-inf"),
                dtype=prefix.dtype,
                device=prefix.device,
            ).triu(1)
            prefix_time = torch.full(
                (prefix.shape[0],),
                prefix_length,
                dtype=torch.long,
                device=prefix.device,
            )
        else:
            prefix_indexes = prefix_indexes.transpose(0, 1)
            prefix_time = prefix_indexes[0].amax(dim=-1) + 1

        return prefix, prefix_indexes, prefix_mask, prefix_time

    def preprocess_prefix(
        self,
        text_input_ids,
        reference_images=None,
        prefix_indexes=None,
        prefix_mask=None,
    ):
        prefix, prefix_indexes, prefix_mask, prefix_time = self._prepare_prefix(
            text_input_ids, reference_images, prefix_indexes, prefix_mask
        )
        prefix_keys = []
        prefix_values = []
        prefix_rope = _prepare_mrope(prefix_indexes, prefix.device, prefix.dtype)
        transformer_options = {}
        for layer_index, layer in enumerate(self.language_model.model.layers):
            transformer_options["block_index"] = layer_index
            prefix, prefix_key, prefix_value = layer.forward_prefix(
                prefix,
                prefix_rope,
                prefix_mask,
                transformer_options,
            )
            prefix_keys.append(prefix_key)
            prefix_values.append(prefix_value)
        return prefix_keys, prefix_values, prefix_time

    def _forward(
        self,
        x,
        timesteps,
        context=None,
        transformer_options={},
        text_input_ids=None,
        reference_images=None,
        prefix_indexes=None,
        prefix_mask=None,
        prefix_keys=None,
        prefix_values=None,
        prefix_time=None,
        **kwargs,
    ):
        if text_input_ids is None and prefix_keys is None:
            raise ValueError("SenseNova-U1.5 requires text conditioning")

        original_height, original_width = x.shape[-2:]
        x = _pad_to_merged_patch_size(x)
        batch, _, height, width = x.shape
        if prefix_keys is None:
            text_input_ids, prefix_indexes, prefix_mask = _match_prefix_batch(
                batch, text_input_ids, prefix_indexes, prefix_mask
            )
            prefix_batch = text_input_ids.shape[0]
            if reference_images:
                reference_images = [
                    comfy.utils.resize_to_batch_size(reference, prefix_batch)
                    for reference in reference_images
                ]
            else:
                reference_images = None
        else:
            prefix_batch = prefix_keys[0].shape[0]
            if prefix_batch > 0 and batch % prefix_batch:
                prefix_keys = [
                    comfy.utils.resize_to_batch_size(value, batch)
                    for value in prefix_keys
                ]
                prefix_values = [
                    comfy.utils.resize_to_batch_size(value, batch)
                    for value in prefix_values
                ]
                prefix_time = comfy.utils.resize_to_batch_size(prefix_time, batch)
                prefix_batch = batch
        generation_batch = _generation_batch_size(batch, prefix_batch)
        token_height = height // MERGED_PATCH_SIZE
        token_width = width // MERGED_PATCH_SIZE
        image_length = token_height * token_width

        image = self.fm_modules["vision_model_mot_gen"](x)
        time_embedding = self.fm_modules["timestep_embedder"](timesteps, image.dtype)
        noise_scale = resolution_noise_scale(height, width) / 16.0
        scale_timesteps = torch.full_like(timesteps, noise_scale)
        time_embedding = time_embedding + self.fm_modules["noise_scale_embedder"](
            scale_timesteps, image.dtype
        )
        image = image + time_embedding[:, None, :]

        if prefix_keys is None:
            prefix, prefix_indexes, prefix_mask, prefix_time = self._prepare_prefix(
                text_input_ids, reference_images, prefix_indexes, prefix_mask
            )
            prefix_rope = _prepare_mrope(prefix_indexes, prefix.device, prefix.dtype)
        image_time = prefix_time.repeat_interleave(generation_batch)

        image_positions = torch.arange(image_length, dtype=torch.long, device=x.device)
        image_indexes = torch.stack(
            (
                image_time[:, None].expand(batch, image_length),
                (image_positions // token_width)[None].expand(batch, image_length),
                (image_positions % token_width)[None].expand(batch, image_length),
            )
        )
        image_rope = _prepare_mrope(image_indexes, image.device, image.dtype)

        for layer_index, layer in enumerate(self.language_model.model.layers):
            transformer_options["block_index"] = layer_index
            if prefix_keys is None:
                prefix, prefix_key, prefix_value = layer.forward_prefix(
                    prefix,
                    prefix_rope,
                    prefix_mask,
                    transformer_options,
                )
            else:
                prefix_key = prefix_keys[layer_index]
                prefix_value = prefix_values[layer_index]
            generation_prefix_key = _expand_prefix_batch(prefix_key, generation_batch)
            generation_prefix_value = _expand_prefix_batch(
                prefix_value, generation_batch
            )
            image = layer.forward_generation(
                image,
                image_rope,
                generation_prefix_key,
                generation_prefix_value,
                transformer_options,
            )

        image = self.language_model.model.norm_mot_gen(image)
        image = image.view(batch, token_height, token_width, HIDDEN_SIZE).permute(
            0, 3, 1, 2
        )
        predicted = self.fm_modules["fm_head"](image)
        denominator = (1.0 - timesteps).clamp_min(0.02).view(batch, 1, 1, 1)
        velocity = (x - predicted) / denominator
        return velocity[..., :original_height, :original_width]
