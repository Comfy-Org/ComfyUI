"""Trusted implementation of WhatDreamsCost temporal prompt relay."""
from __future__ import annotations


def apply_prompt_relay(patcher, latent, token_starts, token_ends, pixel_lengths, epsilon):
    import math
    import types

    import torch
    import comfy.ldm.modules.attention

    if not (1 <= len(token_starts) <= 1024 and len(token_starts) == len(token_ends) == len(pixel_lengths)):
        raise ValueError("prompt_relay ranges and lengths must contain the same 1..1024 entries")
    if any(start < 0 or end <= start or end > 131072 for start, end in zip(token_starts, token_ends)):
        raise ValueError("prompt_relay token ranges are invalid")
    if any(length <= 0 or length > 1000000 for length in pixel_lengths):
        raise ValueError("prompt_relay lengths must be positive")
    if not isinstance(latent, dict) or not isinstance(latent.get("samples"), torch.Tensor):
        raise ValueError("prompt_relay needs a canonical LATENT value")
    samples = latent["samples"]
    if samples.ndim != 5 or samples.shape[2] < 1:
        raise ValueError("prompt_relay needs a five-dimensional video latent")

    diffusion = patcher.model.diffusion_model
    if hasattr(diffusion, "patch_size") and not hasattr(diffusion, "patchifier"):
        architecture, patch_size, temporal_stride = "wan", tuple(diffusion.patch_size), 4
    elif hasattr(diffusion, "patchifier"):
        architecture, patch_size = "ltx", (1, 1, 1)
        temporal_stride = int(diffusion.vae_scale_factors[0])
    else:
        raise ValueError(f"prompt_relay does not support {type(diffusion).__name__}")

    latent_frames = int(samples.shape[2])
    tokens_per_frame = (int(samples.shape[3]) // int(patch_size[1])) * (int(samples.shape[4]) // int(patch_size[2]))
    total_pixels = sum(pixel_lengths)
    target_total = min(latent_frames, max(1, round(total_pixels / temporal_stride)))
    if target_total >= latent_frames - 1:
        target_total = latent_frames
    exact = [length * target_total / total_pixels for length in pixel_lengths]
    effective = [int(value) for value in exact]
    diff = target_total - sum(effective)
    order = sorted(range(len(exact)), key=lambda index: -(exact[index] - int(exact[index])))
    for index in order[:diff]:
        effective[index] += 1
    for index, length in enumerate(effective):
        if length < 1:
            donor = max(range(len(effective)), key=effective.__getitem__)
            if effective[donor] > 1:
                effective[donor] -= 1
                effective[index] = 1

    sigma = 1.0 / math.log(1.0 / epsilon)
    segments, cursor = [], 0
    for start, end, length in zip(token_starts, token_ends, effective):
        if length > 0:
            segments.append({"tokens": torch.arange(start, end), "midpoint": (2 * cursor + length) // 2, "window": max(length // 2 - 2, 0)})
        cursor += length
    maximum_token = max(token_ends)
    cache = {}

    def mask_fn(lq, lk, dtype, device, transformer_options):
        cond = transformer_options.get("cond_or_uncond", [])
        if lq == lk or (1 in cond and 0 not in cond):
            return None
        grid = transformer_options.get("grid_sizes")
        attention_type = transformer_options.get("promptrelay_attn_type", "attn2")
        is_audio = attention_type == "audio_attn2"
        if is_audio:
            mode, actual_tokens_per_frame = "scaled", tokens_per_frame
        else:
            actual_tokens_per_frame = int(grid[1]) * int(grid[2]) if grid is not None else (lq // latent_frames if lq % latent_frames == 0 else tokens_per_frame)
            video_lq = latent_frames * actual_tokens_per_frame
            if lk == video_lq or lk < maximum_token:
                return None
            mode = "video" if lq == video_lq else "scaled"
        key = (lq, lk, mode, str(device), dtype)
        if key not in cache:
            offset = torch.zeros(lq, lk, device=device, dtype=dtype)
            query_frames = torch.arange(lq, device=device).float()
            query_frames = query_frames // actual_tokens_per_frame if mode == "video" else query_frames * latent_frames / lq
            for segment in segments:
                indexes = segment["tokens"].to(device=device)
                distance = (query_frames[:, None] - segment["midpoint"]).abs()
                cost = torch.relu(distance - segment["window"]) ** 2 / (2 * sigma**2)
                offset[:, indexes] = cost.to(dtype)
            cache[key] = -offset
        return cache[key]

    def masked_attention(q, k, v, heads, mask, options):
        return comfy.ldm.modules.attention.attention_pytorch(q, k, v, heads, mask=mask, _inside_attn_wrapper=True, transformer_options=options)

    model = patcher.clone()
    if architecture == "wan":
        from comfy.ldm.wan.model import WanI2VCrossAttention

        for block_index, block in enumerate(diffusion.blocks):
            cross_attention = block.cross_attn
            key = f"diffusion_model.blocks.{block_index}.cross_attn.forward"
            if key in getattr(model, "object_patches", {}):
                raise ValueError(f"prompt_relay cannot stack with an existing patch at {key}")
            if isinstance(cross_attention, WanI2VCrossAttention):
                def implementation(self, x, context, context_img_len, transformer_options=None, **kwargs):
                    options = transformer_options or {}
                    context_img, context_text = context[:, :context_img_len], context[:, context_img_len:]
                    q = self.norm_q(self.q(x))
                    image_out = comfy.ldm.modules.attention.optimized_attention(q, self.norm_k_img(self.k_img(context_img)), self.v_img(context_img), heads=self.num_heads, transformer_options=options)
                    k, v = self.norm_k(self.k(context_text)), self.v(context_text)
                    mask = mask_fn(q.shape[1], k.shape[1], q.dtype, q.device, options)
                    text_out = masked_attention(q, k, v, self.num_heads, mask, options) if mask is not None else comfy.ldm.modules.attention.optimized_attention(q, k, v, heads=self.num_heads, transformer_options=options)
                    return self.o(text_out + image_out)
            else:
                def implementation(self, x, context, transformer_options=None, **kwargs):
                    options = transformer_options or {}
                    q, k, v = self.norm_q(self.q(x)), self.norm_k(self.k(context)), self.v(context)
                    mask = mask_fn(q.shape[1], k.shape[1], q.dtype, q.device, options)
                    output = masked_attention(q, k, v, self.num_heads, mask, options) if mask is not None else comfy.ldm.modules.attention.optimized_attention(q, k, v, heads=self.num_heads, transformer_options=options)
                    return self.o(output)
            model.add_object_patch(key, types.MethodType(implementation, cross_attention))
        return model

    model.model_options.setdefault("transformer_options", {})["promptrelay_mask_fn"] = mask_fn
    for block_index, block in enumerate(diffusion.transformer_blocks):
        for attribute in ("attn2", "audio_attn2"):
            module = getattr(block, attribute, None)
            if module is None:
                continue
            key = f"diffusion_model.transformer_blocks.{block_index}.{attribute}.forward"
            underlying = model.get_model_object(key)

            def wrapped(self, x, context=None, mask=None, pe=None, k_pe=None, transformer_options=None, _underlying=underlying, _attribute=attribute):
                options = transformer_options or {}
                if context is not None:
                    relay_mask = mask_fn(x.shape[1], context.shape[1], x.dtype, x.device, {**options, "promptrelay_attn_type": _attribute})
                    if relay_mask is not None:
                        mask = relay_mask if mask is None else mask + relay_mask
                if mask is not None:
                    previous = options.get("optimized_attention_override")
                    def override(function, *args, **kwargs):
                        if kwargs.get("mask") is not None:
                            return comfy.ldm.modules.attention.attention_pytorch(*args, **kwargs)
                        return previous(function, *args, **kwargs) if previous else function(*args, **kwargs)
                    options = {**options, "optimized_attention_override": override}
                return _underlying(x, context=context, mask=mask, pe=pe, k_pe=k_pe, transformer_options=options)

            model.add_object_patch(key, types.MethodType(wrapped, module))
    return model
