# Hunyuan3D 2.1 paint UNet (hunyuan3d-paintpbr-v2-1): a dual-stream SD-2.1-style
# UNet that denoises packed multiview PBR material latents. The generation stream
# (``unet.*``) augments every transformer block with material self-attention,
# reference-injection attention, cross-view attention (3D rotary) and DINO
# cross-attention; the reference stream (``unet_dual.*``) is a plain SD2 UNet run
# once at timestep 0 to harvest the per-block reference feature bank.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops
from comfy.ldm.modules.attention import CrossAttention, FeedForward
from comfy.ldm.hunyuan3d.paint.attention import (
    MaterialAttention,
    cross_view_attention,
    rotary_tables,
    voxelize_position_maps,
)


class PaintReferenceBank:
    """Opaque holder for a precomputed reference feature bank. Identity equality
    keeps ComfyUI cond-batching treating the bank as one constant condition."""

    def __init__(self, embeds):
        self.embeds = embeds

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


class PaintForwardState:
    """Per-forward conditioning bundle threaded through the generation stream's
    blocks; also carries the per-resolution rotary tables for this call only."""

    def __init__(self, materials, batch, views, context, bank=None, ref_scale=None,
                 dino_tokens=None, position_maps=None):
        self.materials = materials
        self.batch = batch
        self.views = views
        self.context = context
        self.bank = bank
        self.ref_scale = ref_scale
        self.dino_tokens = dino_tokens
        self.position_maps = position_maps
        self._rope = {}

    def rope(self, hw, dim_head):
        if self.position_maps is None:
            return None
        if hw not in self._rope:
            gh, gw = hw
            voxel_resolution = 8 * gh
            voxels = voxelize_position_maps(self.position_maps, gh, gw, voxel_resolution)
            self._rope[hw] = rotary_tables(voxels, dim_head, voxel_resolution)
        return self._rope[hw]


class ReferenceCapture:
    """Collects each reference-stream block's pre-attention LayerNorm output, in
    block tree order (the same order the generation stream reads the bank)."""

    def __init__(self):
        self.captures = []


def timestep_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(torch.arange(half, dtype=torch.float32, device=timesteps.device)
                      * (-math.log(10000.0) / half))
    angles = timesteps.float()[:, None] * freqs[None]
    return torch.cat([angles.cos(), angles.sin()], dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_dim, time_embed_dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.linear_1 = operations.Linear(in_dim, time_embed_dim, dtype=dtype, device=device)
        self.act = nn.SiLU()
        self.linear_2 = operations.Linear(time_embed_dim, time_embed_dim, dtype=dtype, device=device)

    def forward(self, emb):
        return self.linear_2(self.act(self.linear_1(emb)))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim, groups, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm1 = operations.GroupNorm(groups, in_channels, eps=1e-5, dtype=dtype, device=device)
        self.conv1 = operations.Conv2d(in_channels, out_channels, 3, padding=1, dtype=dtype, device=device)
        self.time_emb_proj = operations.Linear(time_embed_dim, out_channels, dtype=dtype, device=device)
        self.norm2 = operations.GroupNorm(groups, out_channels, eps=1e-5, dtype=dtype, device=device)
        self.conv2 = operations.Conv2d(out_channels, out_channels, 3, padding=1, dtype=dtype, device=device)
        if in_channels != out_channels:
            self.conv_shortcut = operations.Conv2d(in_channels, out_channels, 1, dtype=dtype, device=device)
        else:
            self.conv_shortcut = None

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_emb_proj(F.silu(emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


class Downsample(nn.Module):
    def __init__(self, channels, dtype=None, device=None, operations=None):
        super().__init__()
        self.conv = operations.Conv2d(channels, channels, 3, stride=2, padding=1, dtype=dtype, device=device)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels, dtype=None, device=None, operations=None):
        super().__init__()
        self.conv = operations.Conv2d(channels, channels, 3, padding=1, dtype=dtype, device=device)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class SDTransformerBlock(nn.Module):
    """Standard SD2 transformer block. With ``materials`` the self-attention gains
    the per-material projections used by the generation stream."""

    def __init__(self, dim, heads, dim_head, cross_dim, materials, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm1 = operations.LayerNorm(dim, dtype=dtype, device=device)
        if materials is not None:
            self.attn1 = MaterialAttention(dim, heads=heads, dim_head=dim_head,
                                           extra_materials=materials[1:], full_qkv=True,
                                           dtype=dtype, device=device, operations=operations)
        else:
            self.attn1 = CrossAttention(dim, heads=heads, dim_head=dim_head,
                                        dtype=dtype, device=device, operations=operations)
        self.norm2 = operations.LayerNorm(dim, dtype=dtype, device=device)
        self.attn2 = CrossAttention(dim, context_dim=cross_dim, heads=heads, dim_head=dim_head,
                                    dtype=dtype, device=device, operations=operations)
        self.norm3 = operations.LayerNorm(dim, dtype=dtype, device=device)
        self.ff = FeedForward(dim, glu=True, dtype=dtype, device=device, operations=operations)


class PaintTransformerBlock(nn.Module):
    """One ``transformer_blocks`` entry: the standard block under ``transformer``
    plus, on the generation stream, the specialized attention paths."""

    def __init__(self, dim, heads, dim_head, cross_dim, materials, dtype=None, device=None, operations=None):
        super().__init__()
        self.transformer = SDTransformerBlock(dim, heads, dim_head, cross_dim, materials,
                                              dtype=dtype, device=device, operations=operations)
        self.specialized = materials is not None
        self.block_index = 0
        if self.specialized:
            self.attn_multiview = CrossAttention(dim, heads=heads, dim_head=dim_head,
                                                 dtype=dtype, device=device, operations=operations)
            self.attn_refview = MaterialAttention(dim, heads=heads, dim_head=dim_head,
                                                  extra_materials=materials[1:], full_qkv=False,
                                                  dtype=dtype, device=device, operations=operations)
            self.attn_dino = CrossAttention(dim, context_dim=cross_dim, heads=heads, dim_head=dim_head,
                                            dtype=dtype, device=device, operations=operations)

    def forward(self, x, context, state, hw, transformer_options={}):
        t = self.transformer
        if not self.specialized:
            n1 = t.norm1(x)
            if isinstance(state, ReferenceCapture):
                state.captures.append(n1)
            x = x + t.attn1(n1, transformer_options=transformer_options)
            x = x + t.attn2(t.norm2(x), context=context, transformer_options=transformer_options)
            return x + t.ff(t.norm3(x))

        b, m, v = state.batch, len(state.materials), state.views
        l, c = x.shape[1], x.shape[2]
        n1 = t.norm1(x)
        grouped_n1 = n1.reshape(b, m, v, l, c)
        x = x + t.attn1.forward_per_material(grouped_n1, state.materials,
                                             transformer_options=transformer_options
                                             ).reshape(b * m * v, l, c)
        if state.bank is not None:
            bank = state.bank[self.block_index]
            if bank.shape[0] != b:
                bank = bank.repeat(b // bank.shape[0], 1, 1)
            ref_out = self.attn_refview.forward_reference(grouped_n1[:, 0].reshape(b, v * l, c), bank,
                                                          state.materials, transformer_options=transformer_options)
            ref_out = ref_out.permute(1, 0, 2, 3).reshape(b, m * v * l, c)
            if state.ref_scale is not None:
                ref_out = ref_out * state.ref_scale.view(b, 1, 1)
            x = x + ref_out.reshape(b * m * v, l, c)
        if v > 1:
            rope = state.rope(hw, self.attn_multiview.dim_head)
            if rope is not None:
                cos, sin = rope
                rope = (cos.repeat_interleave(m, dim=0).unsqueeze(1),
                        sin.repeat_interleave(m, dim=0).unsqueeze(1))
            mv = cross_view_attention(self.attn_multiview, grouped_n1.reshape(b * m, v * l, c),
                                      rope=rope, transformer_options=transformer_options)
            x = x + mv.reshape(b * m * v, l, c)
        n2 = t.norm2(x)
        x = x + t.attn2(n2, context=state.context, transformer_options=transformer_options)
        if state.dino_tokens is not None:
            x = x + self.attn_dino(n2, context=state.dino_tokens, transformer_options=transformer_options)
        return x + t.ff(t.norm3(x))


class SpatialTransformer(nn.Module):
    def __init__(self, channels, heads, depth, cross_dim, materials, groups, dtype=None, device=None, operations=None):
        super().__init__()
        dim_head = channels // heads
        self.norm = operations.GroupNorm(groups, channels, eps=1e-6, dtype=dtype, device=device)
        self.proj_in = operations.Linear(channels, channels, dtype=dtype, device=device)
        self.transformer_blocks = nn.ModuleList(
            PaintTransformerBlock(channels, heads, dim_head, cross_dim, materials,
                                  dtype=dtype, device=device, operations=operations)
            for _ in range(depth))
        self.proj_out = operations.Linear(channels, channels, dtype=dtype, device=device)

    def forward(self, x, context, state, transformer_options={}):
        b, c, h, w = x.shape
        residual = x
        tokens = self.norm(x).permute(0, 2, 3, 1).reshape(b, h * w, c)
        tokens = self.proj_in(tokens)
        for block in self.transformer_blocks:
            tokens = block(tokens, context, state, (h, w), transformer_options=transformer_options)
        tokens = self.proj_out(tokens)
        return tokens.reshape(b, h, w, c).permute(0, 3, 1, 2) + residual


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim, num_layers, heads, depth,
                 cross_dim, materials, groups, has_attention, add_downsample,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.resnets = nn.ModuleList(
            ResnetBlock(in_channels if i == 0 else out_channels, out_channels, time_embed_dim,
                        groups, dtype=dtype, device=device, operations=operations)
            for i in range(num_layers))
        if has_attention:
            self.attentions = nn.ModuleList(
                SpatialTransformer(out_channels, heads, depth, cross_dim, materials, groups,
                                   dtype=dtype, device=device, operations=operations)
                for _ in range(num_layers))
        else:
            self.attentions = None
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample(out_channels, dtype=dtype, device=device,
                                                          operations=operations)])
        else:
            self.downsamplers = None

    def forward(self, x, emb, context, state, transformer_options={}):
        outs = []
        for i, resnet in enumerate(self.resnets):
            x = resnet(x, emb)
            if self.attentions is not None:
                x = self.attentions[i](x, context, state, transformer_options=transformer_options)
            outs.append(x)
        if self.downsamplers is not None:
            x = self.downsamplers[0](x)
            outs.append(x)
        return x, outs


class MidBlock(nn.Module):
    def __init__(self, channels, time_embed_dim, heads, depth, cross_dim, materials, groups,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.resnets = nn.ModuleList(
            ResnetBlock(channels, channels, time_embed_dim, groups,
                        dtype=dtype, device=device, operations=operations)
            for _ in range(2))
        self.attentions = nn.ModuleList([
            SpatialTransformer(channels, heads, depth, cross_dim, materials, groups,
                               dtype=dtype, device=device, operations=operations)])

    def forward(self, x, emb, context, state, transformer_options={}):
        x = self.resnets[0](x, emb)
        x = self.attentions[0](x, context, state, transformer_options=transformer_options)
        return self.resnets[1](x, emb)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, prev_output_channel, time_embed_dim, num_layers,
                 heads, depth, cross_dim, materials, groups, has_attention, add_upsample,
                 dtype=None, device=None, operations=None):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            skip_channels = in_channels if i == num_layers - 1 else out_channels
            resnet_in = prev_output_channel if i == 0 else out_channels
            resnets.append(ResnetBlock(resnet_in + skip_channels, out_channels, time_embed_dim,
                                       groups, dtype=dtype, device=device, operations=operations))
        self.resnets = nn.ModuleList(resnets)
        if has_attention:
            self.attentions = nn.ModuleList(
                SpatialTransformer(out_channels, heads, depth, cross_dim, materials, groups,
                                   dtype=dtype, device=device, operations=operations)
                for _ in range(num_layers))
        else:
            self.attentions = None
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample(out_channels, dtype=dtype, device=device,
                                                      operations=operations)])
        else:
            self.upsamplers = None

    def forward(self, x, skips, emb, context, state, transformer_options={}):
        for i, resnet in enumerate(self.resnets):
            x = resnet(torch.cat([x, skips.pop()], dim=1), emb)
            if self.attentions is not None:
                x = self.attentions[i](x, context, state, transformer_options=transformer_options)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class DinoProjection(nn.Module):
    """Maps each DINO embedding to 4 cross-attention tokens (Linear to
    ``4*cross_dim``, split along the token axis, then LayerNorm)."""

    def __init__(self, dino_dim, cross_dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.proj = operations.Linear(dino_dim, 4 * cross_dim, dtype=dtype, device=device)
        self.norm = operations.LayerNorm(cross_dim, dtype=dtype, device=device)

    def forward(self, tokens):
        b, t, _ = tokens.shape
        return self.norm(self.proj(tokens).reshape(b, t * 4, -1))


class UNetStream(nn.Module):
    """One SD2-style UNet tree. With ``materials`` set this is the generation
    stream, which also owns the learned material/reference embeddings and the
    DINO projection (checkpoint keys live under this submodule)."""

    def __init__(self, in_channels, out_channels, block_out_channels, layers_per_block,
                 cross_dim, num_attention_heads, depth, groups, materials=None,
                 pbr_token_channels=None, dino_dim=None, use_dino=False,
                 dtype=None, device=None, operations=None):
        super().__init__()
        ch0 = block_out_channels[0]
        time_embed_dim = ch0 * 4
        self.sinusoid_dim = ch0
        self.conv_in = operations.Conv2d(in_channels, ch0, 3, padding=1, dtype=dtype, device=device)
        self.time_embedding = TimestepEmbedding(ch0, time_embed_dim, dtype=dtype, device=device,
                                                operations=operations)

        if materials is not None:
            for mat in materials:
                self.register_parameter(
                    f"learned_text_clip_{mat}",
                    nn.Parameter(torch.empty(pbr_token_channels, cross_dim, dtype=dtype, device=device)))
            self.register_parameter(
                "learned_text_clip_ref",
                nn.Parameter(torch.empty(pbr_token_channels, cross_dim, dtype=dtype, device=device)))
            if use_dino:
                self.image_proj_model_dino = DinoProjection(dino_dim, cross_dim, dtype=dtype,
                                                            device=device, operations=operations)

        self.down_blocks = nn.ModuleList()
        output_channel = ch0
        for i, out_ch in enumerate(block_out_channels):
            is_final = i == len(block_out_channels) - 1
            self.down_blocks.append(DownBlock(
                output_channel, out_ch, time_embed_dim, layers_per_block, num_attention_heads[i],
                depth, cross_dim, materials, groups, has_attention=not is_final,
                add_downsample=not is_final, dtype=dtype, device=device, operations=operations))
            output_channel = out_ch

        self.mid_block = MidBlock(block_out_channels[-1], time_embed_dim, num_attention_heads[-1],
                                  depth, cross_dim, materials, groups,
                                  dtype=dtype, device=device, operations=operations)

        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(block_out_channels))
        reversed_heads = list(reversed(num_attention_heads))
        output_channel = reversed_channels[0]
        for i, out_ch in enumerate(reversed_channels):
            prev_output_channel = output_channel
            in_ch = reversed_channels[min(i + 1, len(block_out_channels) - 1)]
            is_final = i == len(block_out_channels) - 1
            self.up_blocks.append(UpBlock(
                in_ch, out_ch, prev_output_channel, time_embed_dim, layers_per_block + 1,
                reversed_heads[i], depth, cross_dim, materials, groups, has_attention=i > 0,
                add_upsample=not is_final, dtype=dtype, device=device, operations=operations))
            output_channel = out_ch

        self.conv_norm_out = operations.GroupNorm(groups, ch0, eps=1e-5, dtype=dtype, device=device)
        self.conv_act = nn.SiLU()
        self.conv_out = operations.Conv2d(ch0, out_channels, 3, padding=1, dtype=dtype, device=device)

        index = 0
        for module in self.modules():
            if isinstance(module, PaintTransformerBlock):
                module.block_index = index
                index += 1

    def forward(self, sample, timesteps, context, state=None, transformer_options={}):
        emb = timestep_embedding(timesteps, self.sinusoid_dim).to(sample.dtype)
        emb = self.time_embedding(emb)
        h = self.conv_in(sample)
        skips = [h]
        for block in self.down_blocks:
            h, outs = block(h, emb, context, state, transformer_options=transformer_options)
            skips.extend(outs)
        h = self.mid_block(h, emb, context, state, transformer_options=transformer_options)
        for block in self.up_blocks:
            h = block(h, skips, emb, context, state, transformer_options=transformer_options)
        return self.conv_out(self.conv_act(self.conv_norm_out(h)))


class UNet2p5DConditionModel(nn.Module):
    def __init__(self, in_channels=12, ref_in_channels=4, out_channels=4,
                 block_out_channels=(320, 640, 1280, 1280), layers_per_block=2,
                 cross_attention_dim=1024, num_attention_heads=(5, 10, 20, 20),
                 transformer_layers_per_block=1, norm_num_groups=32,
                 pbr_setting=("albedo", "mr"), pbr_token_channels=77,
                 dino_embeddings_dim=1536, use_dino=True, image_model=None,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.dtype = dtype
        self.pbr_setting = tuple(pbr_setting)
        self.cross_attention_dim = cross_attention_dim
        self.use_dino = use_dino
        self.unet = UNetStream(in_channels, out_channels, block_out_channels, layers_per_block,
                               cross_attention_dim, num_attention_heads, transformer_layers_per_block,
                               norm_num_groups, materials=self.pbr_setting,
                               pbr_token_channels=pbr_token_channels, dino_dim=dino_embeddings_dim,
                               use_dino=use_dino, dtype=dtype, device=device, operations=operations)
        self.unet_dual = UNetStream(ref_in_channels, out_channels, block_out_channels, layers_per_block,
                                    cross_attention_dim, num_attention_heads, transformer_layers_per_block,
                                    norm_num_groups, dtype=dtype, device=device, operations=operations)

    def material_context(self, batch_size, dtype=None, device=None):
        """Learned per-material embedding tokens, ``(batch_size, M, T, cross_dim)``."""
        embeds = torch.stack([self.unet.get_parameter(f"learned_text_clip_{mat}")
                              for mat in self.pbr_setting])
        embeds = embeds.unsqueeze(0).expand(batch_size, -1, -1, -1)
        if dtype is not None or device is not None:
            embeds = embeds.to(dtype=dtype, device=device)
        return embeds

    def compute_reference_bank(self, ref_latents):
        """Run the reference stream once at timestep 0 on ``(B, R, 4, H, W)``
        reference latents and harvest each block's pre-attention LayerNorm output
        as ``(B, R*L_block, C_block)`` tensors, in block tree order."""
        b, r = ref_latents.shape[:2]
        flat = ref_latents.reshape(b * r, *ref_latents.shape[2:])
        ref_context = comfy.ops.cast_to_input(self.unet.get_parameter("learned_text_clip_ref"), flat)
        ref_context = ref_context.unsqueeze(0).expand(b * r, -1, -1)
        capture = ReferenceCapture()
        timesteps = torch.zeros(b * r, device=flat.device)
        self.unet_dual(flat, timesteps, ref_context, state=capture)
        return [c.reshape(b, r * c.shape[1], c.shape[2]) for c in capture.captures]

    def forward(self, x, timesteps, context=None, ref_bank=None, ref_latents=None,
                dino_features=None, position_maps=None, ref_scale=None,
                control=None, transformer_options={}, **kwargs):
        b, _, frames, h, w = x.shape
        m = len(self.pbr_setting)
        v = frames // m
        sample = x.movedim(2, 1).reshape(b * frames, x.shape[1], h, w)

        ts = timesteps.reshape(-1).to(x.device)
        if ts.shape[0] == 1:
            ts = ts.expand(b * frames)
        else:
            ts = ts.repeat_interleave(frames)

        if context is None:
            context = self.material_context(b, dtype=x.dtype, device=x.device)
        if context.ndim == 3:
            context = context.reshape(b, m, -1, context.shape[-1])
        tokens = context.shape[2]
        context_flat = context.unsqueeze(2).expand(b, m, v, tokens, context.shape[-1])
        context_flat = context_flat.reshape(b * frames, tokens, context.shape[-1])

        bank = None
        if ref_bank is not None:
            bank = ref_bank.embeds
        elif ref_latents is not None:
            bank = self.compute_reference_bank(ref_latents)

        scale = None
        if ref_scale is not None:
            scale = ref_scale.reshape(-1).to(device=x.device, dtype=x.dtype)
            if scale.shape[0] == 1:
                scale = scale.expand(b)

        dino_tokens = None
        if dino_features is not None:
            dino_tokens = self.unet.image_proj_model_dino(dino_features)
            dino_tokens = dino_tokens.unsqueeze(1).expand(b, frames, *dino_tokens.shape[1:])
            dino_tokens = dino_tokens.reshape(b * frames, *dino_tokens.shape[2:])

        state = PaintForwardState(self.pbr_setting, b, v, context_flat, bank=bank,
                                  ref_scale=scale, dino_tokens=dino_tokens,
                                  position_maps=position_maps)
        out = self.unet(sample, ts, context_flat, state=state,
                        transformer_options=transformer_options)
        return out.reshape(b, frames, out.shape[1], h, w).movedim(1, 2)
