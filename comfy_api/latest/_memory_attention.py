from __future__ import annotations

import types


class MemoryAttentionError(RuntimeError):
    pass


def _require_sage():
    try:
        from sageattention import sageattn
        from sageattention.core import get_cuda_arch_versions
    except Exception as error:
        raise MemoryAttentionError(
            "memory-efficient SageAttention requires a current "
            "sageattention installation") from error
    if not get_cuda_arch_versions():
        raise MemoryAttentionError(
            "memory-efficient SageAttention found no supported CUDA architecture")
    return sageattn


def _sage_nhd(query, key, value):
    sageattn = _require_sage()
    return sageattn(
        query, key, value, tensor_layout="NHD", is_causal=False,
        smooth_k=False)


def _ltx2_forward(
    module, value, context=None, mask=None, pe=None, k_pe=None,
    transformer_options=None,
):
    import torch
    from comfy.ldm.lightricks.model import apply_rotary_emb
    from comfy.ldm.modules import attention

    transformer_options = transformer_options or {}
    context = value if context is None else context
    query = module.q_norm(module.to_q(value))
    key = module.k_norm(module.to_k(context))
    if pe is not None:
        query = apply_rotary_emb(query, pe)
        key = apply_rotary_emb(key, pe if k_pe is None else k_pe)
    result = module.to_v(context)
    if mask is not None:
        try:
            from comfy.ldm.lightricks.model import (
                GuideAttentionMask, _attention_with_guide_mask,
            )
        except ImportError:
            GuideAttentionMask = None
        if GuideAttentionMask is not None and isinstance(
                mask, GuideAttentionMask):
            result = _attention_with_guide_mask(
                query, key, result, module.heads, mask,
                attn_precision=module.attn_precision,
                transformer_options=transformer_options)
        else:
            result = attention.optimized_attention_masked(
                query, key, result, module.heads, mask,
                attn_precision=module.attn_precision,
                transformer_options=transformer_options)
        if module.to_gate_logits is not None:
            gate = module.to_gate_logits(value)
            batch, sequence, _ = result.shape
            result = result.view(
                batch, sequence, module.heads, module.dim_head)
            result.mul_((2.0 * torch.sigmoid(gate)).unsqueeze(-1))
            result = result.view(
                batch, sequence, module.heads * module.dim_head)
        return module.to_out(result)

    batch, sequence, _ = query.shape
    query = query.view(batch, sequence, module.heads, module.dim_head)
    key = key.view(batch, key.shape[1], module.heads, module.dim_head)
    result = result.view(
        batch, result.shape[1], module.heads, module.dim_head)
    result = _sage_nhd(query, key, result)
    if module.to_gate_logits is not None:
        gate = module.to_gate_logits(value)
        result.mul_((2.0 * torch.sigmoid(gate)).unsqueeze(-1))
    return module.to_out(result.view(batch, sequence, -1))


def _wan_self_forward(module, value, frequencies, transformer_options=None):
    from comfy.ldm.flux.math import apply_rope

    del transformer_options
    batch, sequence = value.shape[:2]
    query = module.norm_q(module.q(value)).view(
        batch, sequence, module.num_heads, module.head_dim)
    key = module.norm_k(module.k(value)).view(
        batch, sequence, module.num_heads, module.head_dim)
    query, key = apply_rope(query, key, frequencies)
    result = module.v(value).view(
        batch, sequence, module.num_heads, module.head_dim)
    result = _sage_nhd(query, key, result)
    return module.o(result.view(
        batch, sequence, module.num_heads * module.head_dim))


def _wan_t2v_forward(
    module, value, context, transformer_options=None, **kwargs,
):
    del transformer_options, kwargs
    batch, sequence = value.shape[:2]
    query = module.norm_q(module.q(value)).view(
        batch, sequence, module.num_heads, module.head_dim)
    key = module.norm_k(module.k(context)).view(
        batch, -1, module.num_heads, module.head_dim)
    result = module.v(context).view(
        batch, -1, module.num_heads, module.head_dim)
    result = _sage_nhd(query, key, result)
    return module.o(result.view(
        batch, sequence, module.num_heads * module.head_dim))


def _wan_i2v_forward(
    module, value, context, context_img_len, transformer_options=None,
):
    del transformer_options
    batch, sequence = value.shape[:2]
    image_context = context[:, :context_img_len]
    text_context = context[:, context_img_len:]
    query = module.norm_q(module.q(value)).view(
        batch, sequence, module.num_heads, module.head_dim)
    image_key = module.norm_k_img(module.k_img(image_context)).view(
        batch, -1, module.num_heads, module.head_dim)
    image_value = module.v_img(image_context).view(
        batch, -1, module.num_heads, module.head_dim)
    result = _sage_nhd(query, image_key, image_value)
    key = module.norm_k(module.k(text_context)).view(
        batch, -1, module.num_heads, module.head_dim)
    text_value = module.v(text_context).view(
        batch, -1, module.num_heads, module.head_dim)
    result.add_(_sage_nhd(query, key, text_value))
    return module.o(result.view(
        batch, sequence, module.num_heads * module.head_dim))


def _require_minimax_model():
    try:
        from comfy.ldm.minimax.model import MiniMaxH3Model
    except ImportError as error:
        raise MemoryAttentionError(
            "MiniMax memory-efficient attention requires core MiniMax H3 "
            "support; update ComfyUI") from error
    return MiniMaxH3Model


def _minimax_forward(
    module, value, rope_freqs=None, transformer_options=None,
):
    import torch
    import comfy.model_management as model_management
    import comfy.quant_ops

    transformer_options = transformer_options or {}
    if isinstance(value, list):
        value = value.pop()
    dtype = value.dtype
    device = value.device
    sequence = value.shape[0]
    query, key, result = module.qkv_proj(value).split(
        module.heads * module.head_dim, dim=-1)
    del value
    query = query.view(1, sequence, module.heads, module.head_dim)
    key = key.view(1, sequence, module.heads, module.head_dim)
    result = result.view(1, sequence, module.heads, module.head_dim)
    if rope_freqs is not None:
        query_weight = model_management.cast_to(
            module.q_norm.weight, device=device)
        key_weight = model_management.cast_to(
            module.k_norm.weight, device=device)
        comfy.quant_ops.ck.rms_rope_split_half_(
            query, key, rope_freqs, query_weight, key_weight,
            epsilon=module.q_norm.eps,
            rot_dim=rope_freqs.shape[-3] * 2)
    else:
        query = module.q_norm(query)
        key = module.k_norm(key)

    groups = min(
        transformer_options.get("minimax_head_chunks", 1), module.heads)
    if groups <= 1:
        output = _sage_nhd(query, key, result)
        return module.out_proj(
            output.view(sequence, module.heads * module.head_dim))

    output = torch.empty(
        (sequence, module.heads * module.head_dim),
        dtype=dtype, device=device)
    output_nhd = output.view(1, sequence, module.heads, module.head_dim)
    start = 0
    for index in range(groups):
        stop = (
            start + module.heads // groups
            + (1 if index < module.heads % groups else 0)
        )
        output_nhd[:, :, start:stop] = _sage_nhd(
            query[:, :, start:stop], key[:, :, start:stop],
            result[:, :, start:stop])
        start = stop
    del query, key, result
    return module.out_proj(output)


def apply_ltx2(patcher, triton_kernels: bool):
    _require_sage()
    model = patcher.clone()
    diffusion_model = model.get_model_object("diffusion_model")
    blocks = getattr(diffusion_model, "transformer_blocks", None)
    if blocks is None:
        raise MemoryAttentionError(
            "LTX2 memory-efficient attention needs transformer blocks")
    for index, block in enumerate(blocks):
        prefix = f"diffusion_model.transformer_blocks.{index}.attn1"
        model.add_object_patch(
            f"{prefix}.use_triton_kernels", triton_kernels)
        model.add_object_patch(
            f"{prefix}.forward",
            types.MethodType(_ltx2_forward, block.attn1))
    return model


def apply_wan(patcher):
    _require_sage()
    from comfy.ldm.wan.model import (
        WanI2VCrossAttention, WanT2VCrossAttention,
    )

    model = patcher.clone()
    diffusion_model = model.get_model_object("diffusion_model")
    blocks = getattr(diffusion_model, "blocks", None)
    if blocks is None:
        raise MemoryAttentionError(
            "Wan memory-efficient attention needs transformer blocks")
    for index, block in enumerate(blocks):
        prefix = f"diffusion_model.blocks.{index}"
        model.add_object_patch(
            f"{prefix}.self_attn.forward",
            types.MethodType(_wan_self_forward, block.self_attn))
        cross_attention = getattr(block, "cross_attn", None)
        if type(cross_attention) is WanI2VCrossAttention:
            model.add_object_patch(
                f"{prefix}.cross_attn.forward",
                types.MethodType(_wan_i2v_forward, cross_attention))
        elif type(cross_attention) is WanT2VCrossAttention:
            model.add_object_patch(
                f"{prefix}.cross_attn.forward",
                types.MethodType(_wan_t2v_forward, cross_attention))
    return model


def apply_minimax(patcher):
    _require_sage()
    MiniMaxH3Model = _require_minimax_model()
    model = patcher.clone()
    diffusion_model = model.get_model_object("diffusion_model")
    if not isinstance(diffusion_model, MiniMaxH3Model):
        raise MemoryAttentionError(
            "MiniMax memory-efficient attention requires a MiniMax H3 model")
    blocks = getattr(diffusion_model, "blocks", None)
    if blocks is None:
        raise MemoryAttentionError(
            "MiniMax memory-efficient attention needs transformer blocks")
    for index, block in enumerate(blocks):
        model.add_object_patch(
            f"diffusion_model.blocks.{index}.attn.forward",
            types.MethodType(_minimax_forward, block.attn))
    return model
