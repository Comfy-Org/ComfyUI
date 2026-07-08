import torch
import torch.nn as nn

from comfy.ldm.modules.diffusionmodules.util import timestep_embedding
from comfy.ldm.modules.diffusionmodules.openaimodel import Downsample, TimestepEmbedSequential, ResBlock, SpatialTransformer
from comfy.ldm.modules.attention import optimized_attention


class ZeroSFT(nn.Module):
    def __init__(self, label_nc, norm_nc, concat_channels=0, dtype=None, device=None, operations=None):
        super().__init__()

        ks = 3
        pw = ks // 2

        self.param_free_norm = operations.GroupNorm(32, norm_nc + concat_channels, dtype=dtype, device=device)

        nhidden = 128

        self.mlp_shared = nn.Sequential(
            operations.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw, dtype=dtype, device=device),
            nn.SiLU()
        )
        self.zero_mul = operations.Conv2d(nhidden, norm_nc + concat_channels, kernel_size=ks, padding=pw, dtype=dtype, device=device)
        self.zero_add = operations.Conv2d(nhidden, norm_nc + concat_channels, kernel_size=ks, padding=pw, dtype=dtype, device=device)

        self.zero_conv = operations.Conv2d(label_nc, norm_nc, 1, 1, 0, dtype=dtype, device=device)
        self.pre_concat = bool(concat_channels != 0)

    def forward(self, c, h, h_ori=None, control_scale=1):
        if h_ori is not None and self.pre_concat:
            h_raw = torch.cat([h_ori, h], dim=1)
        else:
            h_raw = h

        h = h + self.zero_conv(c)
        if h_ori is not None and self.pre_concat:
            h = torch.cat([h_ori, h], dim=1)
        actv = self.mlp_shared(c)
        gamma = self.zero_mul(actv)
        beta = self.zero_add(actv)
        h = self.param_free_norm(h)
        h = torch.addcmul(h + beta, h, gamma)
        if h_ori is not None and not self.pre_concat:
            h = torch.cat([h_ori, h], dim=1)
        return torch.lerp(h_raw, h, control_scale)


class _CrossAttnInner(nn.Module):
    """Inner cross-attention module matching the state_dict layout of the original CrossAttention."""
    def __init__(self, query_dim, context_dim, heads, dim_head, dtype=None, device=None, operations=None):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = operations.Linear(query_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_k = operations.Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_v = operations.Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_out = nn.Sequential(
            operations.Linear(inner_dim, query_dim, dtype=dtype, device=device),
        )

    def forward(self, x, context):
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        return self.to_out(optimized_attention(q, k, v, self.heads))


class ZeroCrossAttn(nn.Module):
    def __init__(self, context_dim, query_dim, dtype=None, device=None, operations=None):
        super().__init__()
        heads = query_dim // 64
        dim_head = 64
        self.attn = _CrossAttnInner(query_dim, context_dim, heads, dim_head, dtype=dtype, device=device, operations=operations)
        self.norm1 = operations.GroupNorm(32, query_dim, dtype=dtype, device=device)
        self.norm2 = operations.GroupNorm(32, context_dim, dtype=dtype, device=device)

    def forward(self, context, x, control_scale=1):
        b, c, h, w = x.shape
        x_in = x

        x = self.attn(
            self.norm1(x).flatten(2).transpose(1, 2),
            self.norm2(context).flatten(2).transpose(1, 2),
        ).transpose(1, 2).unflatten(2, (h, w))

        return x_in + x * control_scale


class GLVControl(nn.Module):
    """SUPIR's Guided Latent Vector control encoder. Truncated UNet (input + middle blocks only)."""
    def __init__(
        self,
        in_channels=4,
        model_channels=320,
        num_res_blocks=2,
        attention_resolutions=(4, 2),
        channel_mult=(1, 2, 4),
        num_head_channels=64,
        transformer_depth=(1, 2, 10),
        context_dim=2048,
        adm_in_channels=2816,
        use_linear_in_transformer=True,
        use_checkpoint=False,
        dtype=None,
        device=None,
        operations=None,
        **kwargs,
    ):
        super().__init__()
        self.model_channels = model_channels
        time_embed_dim = model_channels * 4

        self.time_embed = nn.Sequential(
            operations.Linear(model_channels, time_embed_dim, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Linear(time_embed_dim, time_embed_dim, dtype=dtype, device=device),
        )

        self.label_emb = nn.Sequential(
            nn.Sequential(
                operations.Linear(adm_in_channels, time_embed_dim, dtype=dtype, device=device),
                nn.SiLU(),
                operations.Linear(time_embed_dim, time_embed_dim, dtype=dtype, device=device),
            )
        )

        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                operations.Conv2d(in_channels, model_channels, 3, padding=1, dtype=dtype, device=device)
            )
        ])
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(num_res_blocks):
                layers = [
                    ResBlock(ch, time_embed_dim, 0, out_channels=mult * model_channels,
                             dtype=dtype, device=device, operations=operations)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    num_heads = ch // num_head_channels
                    layers.append(
                        SpatialTransformer(ch, num_heads, num_head_channels,
                                           depth=transformer_depth[level], context_dim=context_dim,
                                           use_linear=use_linear_in_transformer,
                                           use_checkpoint=use_checkpoint,
                                           dtype=dtype, device=device, operations=operations)
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(ch, True, out_channels=ch, dtype=dtype, device=device, operations=operations)
                    )
                )
                ds *= 2

        num_heads = ch // num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, 0, dtype=dtype, device=device, operations=operations),
            SpatialTransformer(ch, num_heads, num_head_channels,
                               depth=transformer_depth[-1], context_dim=context_dim,
                               use_linear=use_linear_in_transformer,
                               use_checkpoint=use_checkpoint,
                               dtype=dtype, device=device, operations=operations),
            ResBlock(ch, time_embed_dim, 0, dtype=dtype, device=device, operations=operations),
        )

        self.input_hint_block = TimestepEmbedSequential(
            operations.Conv2d(in_channels, model_channels, 3, padding=1, dtype=dtype, device=device)
        )

    def forward(self, x, timesteps, xt, context=None, y=None, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(x.dtype)
        emb = self.time_embed(t_emb) + self.label_emb(y)

        guided_hint = self.input_hint_block(x, emb, context)

        hs = []
        h = xt
        for module in self.input_blocks:
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        hs.append(h)
        return hs


class SUPIR(nn.Module):
    """
    SUPIR model containing GLVControl (control encoder) and project_modules (adapters).
    State dict keys match the original SUPIR checkpoint layout:
      control_model.*           -> GLVControl
      project_modules.*         -> nn.ModuleList of ZeroSFT/ZeroCrossAttn
    """
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()

        self.control_model = GLVControl(dtype=dtype, device=device, operations=operations)

        project_channel_scale = 2
        cond_output_channels = [320] * 4 + [640] * 3 + [1280] * 3
        project_channels = [int(c * project_channel_scale) for c in [160] * 4 + [320] * 3 + [640] * 3]
        concat_channels = [320] * 2 + [640] * 3 + [1280] * 4 + [0]
        cross_attn_insert_idx = [6, 3]

        self.project_modules = nn.ModuleList()
        for i in range(len(cond_output_channels)):
            self.project_modules.append(ZeroSFT(
                project_channels[i], cond_output_channels[i],
                concat_channels=concat_channels[i],
                dtype=dtype, device=device, operations=operations,
            ))

        for i in cross_attn_insert_idx:
            self.project_modules.insert(i, ZeroCrossAttn(
                cond_output_channels[i], concat_channels[i],
                dtype=dtype, device=device, operations=operations,
            ))
