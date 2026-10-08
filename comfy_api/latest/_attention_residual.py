"""Attach bounded pack residuals to canonical UNet cross-attention sites."""
from __future__ import annotations

import math
import re

import torch

from comfy.ldm.modules.attention import BasicTransformerBlock, optimized_attention


_SITE = re.compile(r"(input_blocks|output_blocks|middle_block)\.(\d+)\.transformer_blocks\.(\d+)$")
_BLOCK_SITE = re.compile(r"(input_blocks|output_blocks)\.(\d+)\.\d+\.transformer_blocks\.(\d+)$")
MAX_BYTES = 512 * 1024 * 1024


def validate_captures(captures):
    tensors = captures.get("tensors", [])
    masks = captures.get("masks", [])
    if len(tensors) > 512 or len(masks) > 32:
        raise ValueError("attention residual capture count exceeds its bounds")
    values = tensors + masks
    if any(not isinstance(value, torch.Tensor) for value in values):
        raise TypeError("attention residual captures must be tensors")
    if sum(value.numel() * value.element_size() for value in values) > MAX_BYTES:
        raise ValueError("attention residual captures exceed 512 MiB")
    storages = {value.untyped_storage()._cdata: value.untyped_storage() for value in values}
    if sum(storage.nbytes() for storage in storages.values()) > MAX_BYTES:
        raise ValueError("attention residual backing storage exceeds 512 MiB")


def _metadata(options, key):
    heads = options["n_heads"]
    if type(heads) is not int or not 1 <= heads <= 256:
        raise ValueError("attention residual requires bounded head metadata")
    branches = list(options.get("cond_or_uncond", []))
    if len(branches) > 64 or any(type(branch) is not int or branch not in (0, 1, 2) for branch in branches):
        raise ValueError("attention residual branch metadata is invalid")
    shape = list(options.get("original_shape", []))
    if len(shape) > 5 or any(type(size) is not int or not 1 <= size <= 131072 for size in shape):
        raise ValueError("attention residual activation shape is invalid")
    sigma = options.get("sigmas")
    if sigma is not None:
        if not isinstance(sigma, torch.Tensor) or not 1 <= sigma.numel() <= 64:
            raise ValueError("attention residual sigma metadata is invalid")
        sigma = float(sigma.flatten()[0])
        if not math.isfinite(sigma):
            raise ValueError("attention residual sigma must be finite")
    metadata = {
        "block": list(key[:2]), "block_index": key[2],
        "n_heads": heads, "original_shape": shape,
        "cond_or_uncond": branches, "sigma": sigma,
    }
    transformer_index = options.get("transformer_index", 0)
    if type(transformer_index) is not int or not 0 <= transformer_index <= 4096:
        raise ValueError("attention residual transformer index is invalid")
    metadata["transformer_index"] = transformer_index
    animation = options.get("ad_params")
    if animation is not None:
        length = animation.get("full_length")
        indices = animation.get("sub_idxs")
        if length is not None:
            if type(length) is not int or not 1 <= length <= 4096:
                raise ValueError("attention residual temporal length is invalid")
            metadata["full_length"] = length
        if indices is not None:
            indices = list(indices)
            if len(indices) > 4096 or any(type(index) is not int or not -4096 <= index < 4096 for index in indices):
                raise ValueError("attention residual temporal indices are invalid")
            metadata["temporal_indices"] = indices
    return metadata


def attach(model, invoke, captures):
    """Clone the patcher; compose after prior replacements at each real site."""
    validate_captures(captures)
    diffusion = model.get_model_object("diffusion_model")
    sites = []
    for name, module in diffusion.named_modules():
        if not isinstance(module, BasicTransformerBlock) or module.attn2 is None:
            continue
        match = _BLOCK_SITE.fullmatch(name) or _SITE.fullmatch(name)
        if match is not None:
            family, block, index = match.groups()
            sites.append(({"input_blocks": "input", "output_blocks": "output", "middle_block": "middle"}[family], int(block), int(index)))
    if not sites or len(sites) > 512 or len(sites) != len(set(sites)):
        raise ValueError("attention residual requires bounded canonical UNet sites")
    previous = model.model_options.get("transformer_options", {}).get("patches_replace", {}).get("attn2", {})
    patched = model.clone()

    def replacement(key, prior):
        def attention(query, context, value, options):
            base = prior(query, context, value, options) if prior is not None else optimized_attention(
                query, context, value, options["n_heads"], attn_precision=options.get("attn_precision"))
            metadata = _metadata(options, key)
            residual = invoke(base, query, metadata, captures.get("tensors", []), captures.get("masks", []))
            if not isinstance(residual, torch.Tensor) or residual.shape != base.shape or residual.dtype != base.dtype or residual.device != base.device:
                raise TypeError("attention residual must preserve shape, dtype, and device")
            return base + residual
        return attention

    for key in sites:
        prior = previous.get(key, previous.get(key[:2]))
        patched.set_model_attn2_replace(replacement(key, prior), *key)
    return patched
