import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.patcher_extension
import comfy.model_prefetch
import comfy.ops
import comfy.utils
from comfy.ldm.common_dit import pad_to_patch_size
from comfy.ldm.flux.math import apply_rope1
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention
from comfy.ldm.modules.diffusionmodules.mmdit import TimestepEmbedder
from comfy.text_encoders.llama import BaseGenerate, FixedKV
from comfy.text_encoders.token_stream import report_generated_tokens

from .conditioning import IMAGE_CONTEXT_ID, IM_END_ID, MERGED_PATCH_SIZE, image_token_mask
from .sampling import resolution_noise_scale


HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 12288
NUM_LAYERS = 42
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
VOCAB_SIZE = 151936


def _use_host_rows(embed):
    return (
        hasattr(embed, "_v")
        and not (embed.weight.device.type == "cpu" and embed.weight.dtype in comfy.model_management.FLOAT8_TYPES)
        and embed.weight_lowvram_function is None
        and len(embed.weight_function) == 0
        and comfy.ops.vbar_above_watermark(embed)
    )


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


class MLP(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None, combine_projections=False):
        super().__init__()
        self.gate_proj = operations.Linear(
            HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False, device=device, dtype=dtype
        )
        self.up_proj = operations.Linear(
            HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False, device=device, dtype=dtype
        )
        self.combined_gate_up = None
        if combine_projections:
            self.combined_gate_up = comfy.ops.CombinedLinear((self.gate_proj, self.up_proj))
            self.combined_gate_up.comfy_gemv_optimized = True
        self.down_proj = operations.Linear(
            INTERMEDIATE_SIZE, HIDDEN_SIZE, bias=False, device=device, dtype=dtype
        )

    def forward(self, hidden_states):
        if self.combined_gate_up is not None:
            gate_up = self.combined_gate_up.forward(hidden_states)
            if gate_up is not NotImplemented:
                return comfy.ops.linear_input_act(self.down_proj, gate_up, "swiglu")
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Attention(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None, combine_projections=False):
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.q_proj = operations.Linear(HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, bias=False, device=device, dtype=dtype)
        self.k_proj = operations.Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False, device=device, dtype=dtype)
        self.v_proj = operations.Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False, device=device, dtype=dtype)
        self.combined_qkv = None
        if combine_projections:
            self.combined_qkv = comfy.ops.CombinedLinear((self.q_proj, self.k_proj, self.v_proj))
            self.combined_qkv.comfy_gemv_optimized = True
        self.o_proj = operations.Linear(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False, device=device, dtype=dtype)
        self.q_norm = operations.RMSNorm(HEAD_DIM // 2, eps=1e-6, device=device, dtype=dtype)
        self.q_norm_hw = operations.RMSNorm(HEAD_DIM // 2, eps=1e-6, device=device, dtype=dtype)
        self.k_norm = operations.RMSNorm(HEAD_DIM // 2, eps=1e-6, device=device, dtype=dtype)
        self.k_norm_hw = operations.RMSNorm(HEAD_DIM // 2, eps=1e-6, device=device, dtype=dtype)

    def _project(self, hidden_states, rope):
        batch, length, _ = hidden_states.shape
        qkv = self.combined_qkv.forward(hidden_states) if self.combined_qkv is not None else NotImplemented
        if qkv is NotImplemented:
            query, key, value = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        else:
            kv_size = NUM_KV_HEADS * HEAD_DIM
            query, key, value = qkv.split((NUM_HEADS * HEAD_DIM, kv_size, kv_size), dim=-1)
        query = query.view(batch, length, NUM_HEADS, HEAD_DIM)
        key = key.view(batch, length, NUM_KV_HEADS, HEAD_DIM)
        value = value.view(batch, length, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        query_t, query_hw = query.chunk(2, dim=-1)
        key_t, key_hw = key.chunk(2, dim=-1)
        query_t = self.q_norm(query_t).transpose(1, 2)
        query_hw = self.q_norm_hw(query_hw).transpose(1, 2)
        key_t = self.k_norm(key_t).transpose(1, 2)
        key_hw = self.k_norm_hw(key_hw).transpose(1, 2)

        query_t, key_t = _apply_llm_rope(query_t, key_t, rope[0])
        if len(rope) == 1:
            query = torch.cat((query_t, query_hw), dim=-1)
            key = torch.cat((key_t, key_hw), dim=-1)
        else:
            query_h, query_w = query_hw.chunk(2, dim=-1)
            key_h, key_w = key_hw.chunk(2, dim=-1)
            query_h, key_h = _apply_llm_rope(query_h, key_h, rope[1])
            query_w, key_w = _apply_llm_rope(query_w, key_w, rope[2])
            query = torch.cat((query_t, query_h, query_w), dim=-1)
            key = torch.cat((key_t, key_h, key_w), dim=-1)
        return query, key, value

    def forward_prefix(
        self, hidden_states, rope, attention_mask, transformer_options,
        past_key=None, past_value=None,
    ):
        query, key, value = self._project(hidden_states, rope)
        if past_key is not None:
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)
        query = AttentionTensorContainer(query)
        output = optimized_attention(
            query,
            AttentionTensorContainer(key),
            AttentionTensorContainer(value),
            NUM_HEADS,
            mask=attention_mask,
            skip_reshape=True,
            transformer_options=transformer_options,
            enable_gqa=True, preferred_attention=self.comfy_attention,
        )
        return self.o_proj(output), key, value

    def forward_generation(
        self, hidden_states, rope, prefix_key, prefix_value, transformer_options
    ):
        query, key, value = self._project(hidden_states, rope)
        key = torch.cat((prefix_key, key), dim=2)
        value = torch.cat((prefix_value, value), dim=2)
        query, key, value = AttentionTensorContainer(query), AttentionTensorContainer(key), AttentionTensorContainer(value)
        output = optimized_attention(
            query,
            key,
            value,
            NUM_HEADS,
            mask=None,
            skip_reshape=True,
            transformer_options=transformer_options,
            enable_gqa=True, preferred_attention=self.comfy_attention,
        )
        return self.o_proj(output)

    def forward_decode(
        self,
        hidden_states,
        rope,
        cache,
        attention_mask,
        transformer_options,
    ):
        query, key, value = self._project(hidden_states, rope)
        key, value = key.transpose(1, 2), value.transpose(1, 2)
        cache.key.index_copy_(1, cache.position, key)
        cache.value.index_copy_(1, cache.position, value)
        output = optimized_attention(
            AttentionTensorContainer(query),
            AttentionTensorContainer(cache.key.transpose(1, 2)),
            AttentionTensorContainer(cache.value.transpose(1, 2)),
            NUM_HEADS,
            mask=attention_mask,
            skip_reshape=True,
            transformer_options=transformer_options,
            enable_gqa=True,
            preferred_attention=self.comfy_attention,
        )
        return self.o_proj(output)


class DecoderLayer(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None, combine_projections=False):
        super().__init__()
        self.self_attn = Attention(device=device, dtype=dtype, operations=operations, combine_projections=combine_projections)
        self.mlp = MLP(device=device, dtype=dtype, operations=operations, combine_projections=combine_projections)
        self.input_layernorm = operations.RMSNorm(HIDDEN_SIZE, eps=1e-6, device=device, dtype=dtype)
        self.post_attention_layernorm = operations.RMSNorm(HIDDEN_SIZE, eps=1e-6, device=device, dtype=dtype)

    def forward_prefix(
        self, prefix, prefix_rope, prefix_mask, transformer_options,
        past_key=None, past_value=None,
    ):
        prefix_attention, prefix_key, prefix_value = self.self_attn.forward_prefix(
            self.input_layernorm(prefix),
            prefix_rope,
            prefix_mask,
            transformer_options,
            past_key=past_key,
            past_value=past_value,
        )
        prefix = prefix + prefix_attention
        prefix = prefix + self.mlp(self.post_attention_layernorm(prefix))
        return prefix, prefix_key, prefix_value

    def forward_generation(
        self, image, image_rope, prefix_key, prefix_value, transformer_options
    ):
        image_attention = self.self_attn.forward_generation(
            self.input_layernorm(image),
            image_rope,
            prefix_key,
            prefix_value,
            transformer_options,
        )
        image = image + image_attention
        image = image + self.mlp(self.post_attention_layernorm(image))
        return image

    def forward_decode(
        self,
        hidden_states,
        rope,
        cache,
        attention_mask,
        transformer_options,
    ):
        attention = self.self_attn.forward_decode(
            self.input_layernorm(hidden_states),
            rope,
            cache,
            attention_mask,
            transformer_options,
        )
        hidden_states = hidden_states + attention
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states


class LanguageBackbone(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        self.embed_tokens = operations.Embedding(VOCAB_SIZE, HIDDEN_SIZE, padding_idx=151643, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            DecoderLayer(device=device, dtype=dtype, operations=operations, combine_projections=True)
            for _ in range(NUM_LAYERS)
        )
        self.norm = operations.RMSNorm(HIDDEN_SIZE, eps=1e-6, device=device, dtype=dtype)


class LanguageModel(nn.Module):
    def __init__(self, has_lm_head=False, device=None, dtype=None, operations=None):
        super().__init__()
        self.model = LanguageBackbone(device=device, dtype=dtype, operations=operations)
        if has_lm_head:
            self.lm_head = operations.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False, device=device, dtype=dtype)


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


class SenseNovaTextModel(nn.Module, BaseGenerate):
    def __init__(self, has_lm_head=False, device=None, dtype=None, operations=None):
        super().__init__()
        self.has_lm_head = has_lm_head
        self.vision_model = VisionEmbeddings(device=device, dtype=dtype, operations=operations)
        self.language_model = LanguageModel(has_lm_head=has_lm_head, device=device, dtype=dtype, operations=operations)

    def _prepare_prefix(
        self, text_input_ids, reference_images, prefix_indexes, prefix_mask, text_suffix=None, compute_dtype=None
    ):
        prefix_length = text_input_ids.shape[1]
        all_ids = torch.cat((text_input_ids, text_suffix), dim=1) if text_suffix is not None else text_input_ids
        embed = self.language_model.model.embed_tokens
        if _use_host_rows(embed):
            prefix = embed.host_rows(all_ids, out_dtype=compute_dtype)
        else:
            prefix = embed(all_ids).to(dtype=compute_dtype)
        if reference_images:
            reference_embeds = [
                self.vision_model(_pad_to_merged_patch_size(reference))
                for reference in reference_images
            ]
            starts_in_image = bool(text_input_ids.shape[1] and int(text_input_ids[0, 0]) == IMAGE_CONTEXT_ID)
            selected = image_token_mask(text_input_ids, starts_in_image=starts_in_image)
            prefix = prefix.clone()
            prefix[:, :prefix_length][selected] = torch.cat(reference_embeds, dim=1).reshape(
                -1, HIDDEN_SIZE
            )

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

        if text_suffix is not None:
            # Upstream appends thinking tokens as text, even when their IDs are image markers.
            suffix_length = text_suffix.shape[1]
            positions = prefix_time[:, None] + torch.arange(suffix_length, device=prefix.device)
            zeros = torch.zeros_like(positions)
            suffix_indexes = torch.stack((positions, zeros, zeros))
            if prefix_indexes.ndim == 2:
                suffix_indexes = suffix_indexes[:, 0]
            prefix_indexes = torch.cat((prefix_indexes, suffix_indexes), dim=-1)
            length = prefix_length + suffix_length
            mask = torch.full(
                (*prefix_mask.shape[:-2], length, length), float("-inf"),
                dtype=prefix_mask.dtype, device=prefix.device,
            ).triu(1)
            mask[..., :prefix_length, :prefix_length] = prefix_mask
            prefix_mask = mask
            prefix_time = prefix_time + suffix_length

        return prefix, prefix_indexes, prefix_mask, prefix_time

    def _preprocess_prefix_state(
        self,
        text_input_ids,
        reference_images=None,
        prefix_indexes=None,
        prefix_mask=None,
        transformer_options=None,
        text_suffix=None,
        compute_dtype=None,
        interrupt=None,
        past_keys=None,
        past_values=None,
        output_device=None,
    ):
        output_device = torch.device(output_device) if output_device is not None else None
        prefix, prefix_indexes, prefix_mask, prefix_time = self._prepare_prefix(
            text_input_ids, reference_images, prefix_indexes, prefix_mask, text_suffix, compute_dtype
        )
        prefix_keys = []
        prefix_values = []
        prefix_rope = _prepare_mrope(prefix_indexes, prefix.device, prefix.dtype)
        if transformer_options is None:
            transformer_options = {}
        prefetch = comfy.model_prefetch.make_prefetch_queue(
            list(self.language_model.model.layers),
            prefix.device, transformer_options,
        )
        with comfy.model_prefetch.malloc_graph_scope(prefix.device):
            for layer_index, layer in enumerate(self.language_model.model.layers):
                if interrupt is not None:
                    interrupt()
                comfy.model_prefetch.prefetch_queue_pop(prefetch, prefix.device, layer, prefix.dtype, malloc_scope="prefix")
                transformer_options["block_index"] = layer_index
                if past_keys is not None:
                    with comfy.model_prefetch.pause_malloc_graph():
                        past_key = past_keys[layer_index].to(device=prefix.device, dtype=prefix.dtype, non_blocking=True)
                        past_value = past_values[layer_index].to(device=prefix.device, dtype=prefix.dtype, non_blocking=True)
                    output, prefix_key, prefix_value = layer.forward_prefix(
                        prefix,
                        prefix_rope,
                        prefix_mask,
                        transformer_options,
                        past_key=past_key,
                        past_value=past_value,
                    )
                    del past_key, past_value
                else:
                    output, prefix_key, prefix_value = layer.forward_prefix(
                        prefix,
                        prefix_rope,
                        prefix_mask,
                        transformer_options,
                    )
                with comfy.model_prefetch.pause_malloc_graph():
                    if output_device is not None and output_device != prefix_key.device:
                        if past_keys is not None and past_keys[layer_index].device == output_device:
                            past_length = past_keys[layer_index].shape[2]
                            new_key = prefix_key[:, :, past_length:].to(output_device)
                            new_value = prefix_value[:, :, past_length:].to(output_device)
                            past_key_out = past_keys[layer_index].to(dtype=new_key.dtype)
                            past_value_out = past_values[layer_index].to(dtype=new_value.dtype)
                            prefix_keys.append(torch.cat((past_key_out, new_key), dim=2))
                            prefix_values.append(torch.cat((past_value_out, new_value), dim=2))
                        else:
                            prefix_keys.append(prefix_key.to(output_device))
                            prefix_values.append(prefix_value.to(output_device))
                    else:
                        prefix_keys.append(prefix_key.clone())
                        prefix_values.append(prefix_value.clone())
                prefix.copy_(output)
                del output, prefix_key, prefix_value
            comfy.model_prefetch.prefetch_queue_pop(prefetch, prefix.device, None, malloc_scope="prefix")
        return prefix, prefix_keys, prefix_values, prefix_time

    def preprocess_prefix(
        self,
        text_input_ids,
        reference_images=None,
        prefix_indexes=None,
        prefix_mask=None,
        transformer_options=None,
        text_suffix=None,
        compute_dtype=None,
        past_keys=None,
        past_values=None,
        output_device=None,
    ):
        _, prefix_keys, prefix_values, prefix_time = (
            self._preprocess_prefix_state(
                text_input_ids,
                reference_images,
                prefix_indexes,
                prefix_mask,
                transformer_options,
                text_suffix,
                compute_dtype,
                past_keys=past_keys,
                past_values=past_values,
                output_device=output_device,
            )
        )
        return prefix_keys, prefix_values, prefix_time

    def _next_text_token(self, hidden_states, **sampling):
        hidden_states = self.language_model.model.norm(hidden_states[:, -1:])
        return self.sample_token(self.language_model.lm_head(hidden_states)[:, -1], **sampling).flatten()

    def _decode_text_token(
        self,
        token,
        caches,
        hidden_states,
        rope,
        attention_mask,
        transformer_options=None,
    ):
        embed = self.language_model.model.embed_tokens
        if _use_host_rows(embed):
            hidden_states.copy_(embed.host_rows(token[:, None], out_dtype=hidden_states.dtype))
        else:
            hidden_states.copy_(embed(token[:, None]))
        if caches:
            caches[0].prepare(1)
            attention_mask[..., caches[0].index].zero_()
        if transformer_options is None:
            transformer_options = {}
        prefetch = comfy.model_prefetch.make_prefetch_queue(
            list(self.language_model.model.layers),
            hidden_states.device, transformer_options,
        )
        with comfy.model_prefetch.malloc_graph_scope(hidden_states.device):
            for layer_index, layer in enumerate(self.language_model.model.layers):
                transformer_options["block_index"] = layer_index
                cache = caches[layer_index]

                def core():
                    hidden_states.copy_(layer.forward_decode(hidden_states, rope, cache, attention_mask, transformer_options))

                comfy.model_prefetch.prefetch_queue_pop(
                    prefetch, hidden_states.device, layer, hidden_states.dtype,
                    core=core, enable_graph=True, malloc_scope="decode",
                )
                cache.advance(1)
            comfy.model_prefetch.prefetch_queue_pop(prefetch, hidden_states.device, None, malloc_scope="decode")

    def generate_text_tokens(
        self,
        text_input_ids,
        reference_images=None,
        prefix_indexes=None,
        prefix_mask=None,
        *,
        max_tokens,
        stop_token_ids,
        transformer_options=None,
        progress=None,
        interrupt=None,
        compute_dtype=None,
        do_sample=False,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        seed=None,
    ):
        """Return generated IDs and the stop token (None at the token limit).

        EOS is excluded from the returned IDs; other stop tokens are included.
        KV state is local to this call.
        """

        if not self.has_lm_head:
            raise RuntimeError(
                "This SenseNova checkpoint does not contain language_model.lm_head.weight, "
                "which is required for text generation."
            )
        if text_input_ids.shape[0] != 1:
            raise ValueError("SenseNova text generation currently requires a single prompt batch.")

        hidden_states, prefix_keys, prefix_values, prefix_time = (
            self._preprocess_prefix_state(
                text_input_ids,
                reference_images,
                prefix_indexes,
                prefix_mask,
                transformer_options,
                compute_dtype=compute_dtype,
                interrupt=interrupt,
            )
        )
        generator = torch.Generator(device=hidden_states.device) if do_sample else None
        if generator is not None:
            generator.manual_seed(seed) if seed is not None else generator.seed()
        history = text_input_ids[0].tolist()
        sampling = dict(do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p,
                        repetition_penalty=repetition_penalty, presence_penalty=presence_penalty,
                        token_history=history, generator=generator)
        next_token = self._next_text_token(hidden_states, **sampling)
        decode_hidden = torch.empty_like(hidden_states[:, -1:])
        del hidden_states
        token_ids = []
        for step in range(max_tokens):
            if interrupt is not None:
                interrupt()
            token_id = int(next_token.item())
            if progress is not None:
                progress(step + 1)
            if token_id == IM_END_ID:
                return token_ids, token_id
            token_ids.append(token_id)
            report_generated_tokens(token_ids)
            history.append(token_id)
            if token_id in stop_token_ids:
                return token_ids, token_id
            if step + 1 == max_tokens:
                break
            if step == 0:
                batch = text_input_ids.shape[0]
                position = torch.empty((batch,), device=decode_hidden.device, dtype=torch.int64)
                prefix_length = text_input_ids.shape[1]
                capacity = prefix_length + max_tokens - 1
                seqlen = torch.full((batch,), prefix_length, device=decode_hidden.device, dtype=torch.int32)
                attention_mask = decode_hidden.new_full((batch, 1, 1, capacity), float("-inf"))
                attention_mask[..., :prefix_length].zero_()
                caches = []
                for index in range(len(prefix_keys)):
                    key, value = prefix_keys[index], prefix_values[index]
                    _, heads, length, head_dim = key.shape
                    buffer = key.new_zeros((batch, capacity, heads, head_dim))
                    cache = FixedKV(buffer, torch.zeros_like(buffer), length, position, seqlen)
                    cache.key[:, :length].copy_(key.transpose(1, 2))
                    cache.value[:, :length].copy_(value.transpose(1, 2))
                    caches.append(cache)
                    prefix_keys[index] = prefix_values[index] = None
                    del key, value, buffer, cache
                del prefix_keys, prefix_values
                positions = prefix_time[:, None] + torch.arange(max_tokens - 1, device=prefix_time.device)
                temporal = _prepare_llm_rope(positions, HEAD_DIM // 2, 5000000.0, decode_hidden.device, decode_hidden.dtype)
                rope = (tuple(freq[:, :, :1].clone() for freq in temporal),)
            else:
                for dst, src in zip(rope[0], temporal):
                    dst.copy_(src[:, :, step:step + 1])
            self._decode_text_token(
                next_token, caches, decode_hidden, rope, attention_mask, transformer_options,
            )
            next_token = self._next_text_token(decode_hidden, **sampling)

        return token_ids, None



class SenseNovaU15(nn.Module):
    def __init__(
        self,
        dtype=None,
        device=None,
        operations=None,
        **kwargs,
    ):
        super().__init__()
        self.dtype = dtype
        self.layers = nn.ModuleList(
            DecoderLayer(device=device, dtype=dtype, operations=operations)
            for _ in range(NUM_LAYERS)
        )
        self.norm = operations.RMSNorm(HIDDEN_SIZE, eps=1e-6, device=device, dtype=dtype)
        self.patch_embed = VisionEmbeddings(device=device, dtype=dtype, operations=operations)
        self.timestep_embedder = TimestepEmbedder(HIDDEN_SIZE, device=device, dtype=dtype, operations=operations)
        self.head = ConvDecoder(device=device, dtype=dtype, operations=operations)
        self.noise_scale_embedder = TimestepEmbedder(HIDDEN_SIZE, device=device, dtype=dtype, operations=operations)

    def forward(self, x, timesteps, context=None, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options
            ),
        ).execute(x, timesteps, context, transformer_options, **kwargs)

    def _forward(
        self,
        x,
        timesteps,
        context=None,
        transformer_options={},
        prefix_keys=None,
        prefix_values=None,
        prefix_time=None,
        **kwargs,
    ):
        if prefix_keys is None:
            raise ValueError("SenseNova-U1.5 requires text conditioning")

        original_height, original_width = x.shape[-2:]
        x = _pad_to_merged_patch_size(x)
        batch, _, height, width = x.shape
        prefix_batch = prefix_keys[0].shape[0]
        if prefix_batch > 0 and batch % prefix_batch:
            prefix_keys = [comfy.utils.resize_to_batch_size(value, batch) for value in prefix_keys]
            prefix_values = [comfy.utils.resize_to_batch_size(value, batch) for value in prefix_values]
            prefix_time = comfy.utils.resize_to_batch_size(prefix_time, batch)
            prefix_batch = batch
        generation_batch = _generation_batch_size(batch, prefix_batch)
        token_height = height // MERGED_PATCH_SIZE
        token_width = width // MERGED_PATCH_SIZE
        image_length = token_height * token_width

        image = self.patch_embed(x)
        time_embedding = self.timestep_embedder(timesteps, image.dtype)
        noise_scale = resolution_noise_scale(height, width) / 16.0
        time_embedding = time_embedding + self.noise_scale_embedder(
            torch.full_like(timesteps, noise_scale), image.dtype
        )
        image = image + time_embedding[:, None, :]

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

        prefetch = comfy.model_prefetch.make_prefetch_queue(
            list(self.layers),
            image.device, transformer_options,
        )
        with comfy.model_prefetch.malloc_graph_scope(image.device):
            for layer_index, layer in enumerate(self.layers):
                comfy.model_prefetch.prefetch_queue_pop(prefetch, image.device, layer, image.dtype, malloc_scope="generation")
                transformer_options["block_index"] = layer_index
                image.copy_(layer.forward_generation(
                    image,
                    image_rope,
                    _expand_prefix_batch(prefix_keys[layer_index], generation_batch),
                    _expand_prefix_batch(prefix_values[layer_index], generation_batch),
                    transformer_options,
                ))
            comfy.model_prefetch.prefetch_queue_pop(prefetch, image.device, None, malloc_scope="generation")

        image = self.norm(image)
        image = image.view(batch, token_height, token_width, HIDDEN_SIZE).permute(
            0, 3, 1, 2
        )
        predicted = self.head(image)
        denominator = (1.0 - timesteps).clamp_min(0.02).view(batch, 1, 1, 1)
        velocity = (x - predicted) / denominator
        return velocity[..., :original_height, :original_width]
