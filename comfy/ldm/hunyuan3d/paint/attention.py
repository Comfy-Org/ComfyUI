# Specialized attention for the Hunyuan3D 2.1 paint UNet: per-material projection
# heads, wide-head reference-injection attention, and the 3D rotary embedding
# (voxelized canonical-coordinate maps) used by cross-view attention.

import torch
import torch.nn as nn

from comfy.ldm.modules.attention import CrossAttention, attention_pytorch, optimized_attention


class MaterialAttention(CrossAttention):
    """CrossAttention with additional per-material projections.

    The base projections serve the albedo material; every other material gets its
    own set under ``processor`` (checkpoint layout: ``attn.processor.to_q_<mat>``,
    ``to_out_<mat>.0`` etc.). ``full_qkv=False`` builds only the value/output pair
    used by reference-injection attention.
    """

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, extra_materials=(),
                 full_qkv=True, dtype=None, device=None, operations=None):
        super().__init__(query_dim, context_dim=context_dim, heads=heads, dim_head=dim_head,
                         dtype=dtype, device=device, operations=operations)
        inner_dim = heads * dim_head
        kv_dim = query_dim if context_dim is None else context_dim
        extra = {}
        for mat in extra_materials:
            if full_qkv:
                extra[f"to_q_{mat}"] = operations.Linear(query_dim, inner_dim, bias=False, dtype=dtype, device=device)
                extra[f"to_k_{mat}"] = operations.Linear(kv_dim, inner_dim, bias=False, dtype=dtype, device=device)
            extra[f"to_v_{mat}"] = operations.Linear(kv_dim, inner_dim, bias=False, dtype=dtype, device=device)
            extra[f"to_out_{mat}"] = nn.Sequential(operations.Linear(inner_dim, query_dim, dtype=dtype, device=device))
        self.processor = nn.ModuleDict(extra)

    def forward_per_material(self, tokens, materials, transformer_options={}):
        """Self-attention within each frame, with projections selected by material.

        tokens: ``(B, M, V, L, C)`` grouped per material; returns the same shape.
        """
        b, m, v, l, c = tokens.shape
        outs = []
        for i, mat in enumerate(materials):
            x = tokens[:, i].reshape(b * v, l, c)
            if i == 0:
                q, k, val = self.to_q(x), self.to_k(x), self.to_v(x)
                out = optimized_attention(q, k, val, self.heads, transformer_options=transformer_options)
                outs.append(self.to_out(out))
            else:
                q = self.processor[f"to_q_{mat}"](x)
                k = self.processor[f"to_k_{mat}"](x)
                val = self.processor[f"to_v_{mat}"](x)
                out = optimized_attention(q, k, val, self.heads, transformer_options=transformer_options)
                outs.append(self.processor[f"to_out_{mat}"](out))
        return torch.stack(outs, dim=1).reshape(b, v, m, l, c).transpose(1, 2)

    def forward_reference(self, albedo_tokens, bank_tokens, materials, transformer_options={}):
        """Reference-injection attention with wide-head value packing.

        Queries come from the albedo tokens only (``(B, V*L, C)``); keys from the
        bank tokens; the per-material value projections are channel-concatenated
        and split head-first into ``M*dim_head``-wide slices, so each head attends
        over a contiguous chunk of the concatenated channels (mixing materials),
        then each head's output is split back per material and routed through that
        material's output projection. Returns ``(M, B, V*L, C)``.
        """
        b, tq = albedo_tokens.shape[:2]
        n_mat = len(materials)
        q = self.to_q(albedo_tokens)
        k = self.to_k(bank_tokens)
        vals = [self.to_v(bank_tokens)]
        for mat in materials[1:]:
            vals.append(self.processor[f"to_v_{mat}"](bank_tokens))
        v = torch.cat(vals, dim=-1)

        q = q.view(b, tq, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(b, -1, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(b, -1, self.heads, n_mat * self.dim_head).transpose(1, 2)
        if n_mat == 1:
            out = optimized_attention(q, k, v, self.heads, skip_reshape=True,
                                      transformer_options=transformer_options)
            return self.to_out(out).unsqueeze(0)
        # value head width differs from the query/key head width; fused backends
        # derive it from the query, so this call needs the plain SDPA path
        out = attention_pytorch(q, k, v, self.heads, skip_reshape=True, skip_output_reshape=True)
        out = out.view(b, self.heads, tq, n_mat, self.dim_head)
        out = out.permute(3, 0, 2, 1, 4).reshape(n_mat, b, tq, self.heads * self.dim_head)
        outs = [self.to_out(out[0])]
        for i, mat in enumerate(materials[1:]):
            outs.append(self.processor[f"to_out_{mat}"](out[i + 1]))
        return torch.stack(outs, dim=0)


def cross_view_attention(attn, tokens, rope=None, transformer_options={}):
    """Joint self-attention over all views of one (batch, material) group.

    tokens: ``(B*M, V*L, C)``; ``rope`` is an optional ``(cos, sin)`` pair
    (fp32, ``(B*M, 1, V*L, dim_head // 2)``) applied to Q and K.
    """
    b = tokens.shape[0]
    q = attn.to_q(tokens)
    k = attn.to_k(tokens)
    v = attn.to_v(tokens)
    q = q.view(b, -1, attn.heads, attn.dim_head).transpose(1, 2)
    k = k.view(b, -1, attn.heads, attn.dim_head).transpose(1, 2)
    v = v.view(b, -1, attn.heads, attn.dim_head).transpose(1, 2)
    if rope is not None:
        cos, sin = rope
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
    out = optimized_attention(q, k, v, attn.heads, skip_reshape=True,
                              transformer_options=transformer_options)
    return attn.to_out(out)


def apply_rotary(x, cos, sin):
    """Pairwise rotary rotation of ``(..., T, D)`` by half-length cos/sin tables."""
    x1, x2 = x.float().unflatten(-1, (-1, 2)).unbind(-1)
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2).to(x.dtype)


def voxelize_position_maps(position_maps, grid_h, grid_w, voxel_resolution):
    """Average canonical-coordinate maps into a token grid of integer voxel indices.

    position_maps: ``(B, V, 3, Hp, Wp)`` in ``[0, 1]`` with exact-white background.
    A pixel is background if any channel is exactly 1.0; cells with fewer valid
    pixels than ``cell_area // 16`` quantize to the origin voxel. The masked
    average runs in fp16 (trained rounding behavior). Returns ``(B, V, gh*gw, 3)``
    int64 voxel coordinates.
    """
    b, v, _, hp, wp = position_maps.shape
    ch, cw = hp // grid_h, wp // grid_w
    cells = position_maps.reshape(b, v, 3, grid_h, ch, grid_w, cw).half()
    valid = (cells != 1.0).all(dim=2, keepdim=True).half()
    count = valid.sum(dim=(4, 6))
    mean = (cells * valid).sum(dim=(4, 6)) / count.clamp(min=1.0)
    enough = count >= float((ch * cw) // 16)
    coords = torch.where(enough, mean, torch.zeros_like(mean))
    coords = coords.clamp(0.0, 1.0) * (voxel_resolution - 1)
    coords = coords.round().long()
    return coords.permute(0, 1, 3, 4, 2).reshape(b, v, grid_h * grid_w, 3)


def rotary_tables(voxels, dim_head, voxel_resolution):
    """Per-token cos/sin tables for the 3-axis rotary embedding.

    ``dim_head`` splits 3:3:2 eighths across x/y/z (position-map channel order);
    each axis uses a standard 1-D rotary table with base theta 10000 over the
    voxel range. voxels: ``(B, V, T, 3)`` int64. Returns fp32 ``(cos, sin)`` of
    shape ``(B, V*T, dim_head // 2)`` (half length, pairwise application).
    """
    axis_dims = (3 * dim_head // 8, 3 * dim_head // 8, dim_head // 4)
    b, v, t, _ = voxels.shape
    pos = torch.arange(voxel_resolution, device=voxels.device, dtype=torch.float32)
    cos_parts, sin_parts = [], []
    for axis, dim in enumerate(axis_dims):
        freqs = 10000.0 ** (-torch.arange(0, dim, 2, device=voxels.device, dtype=torch.float32) / dim)
        angles = pos[:, None] * freqs[None]
        per_token = angles[voxels[..., axis].reshape(b, v * t)]
        cos_parts.append(per_token.cos())
        sin_parts.append(per_token.sin())
    return torch.cat(cos_parts, dim=-1), torch.cat(sin_parts, dim=-1)
