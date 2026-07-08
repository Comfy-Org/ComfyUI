# TripoSplat flow-matching denoiser (LatentSeqMMFlowModel). Registered as a ModelType.FLOW arch and
# driven by the standard KSampler; jointly denoises the (B, 8192, 16) latent and a (B, 1, 5) camera token
# carried as a 2-element nested latent.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
import comfy.patcher_extension
import comfy.rmsnorm
from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.flux.math import apply_rope


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim, heads, dtype=None, device=None):
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(heads, dim, dtype=dtype, device=device))

    def forward(self, x):
        x = comfy.rmsnorm.rms_norm(x)
        return x * comfy.model_management.cast_to(self.gamma, x.dtype, x.device)


# Positional embeddings

class RePo3DRotaryEmbedding(nn.Module):
    def __init__(self, model_channels, num_heads, head_dim, repo_hidden_ratio=0.125, max_freq=16.0,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        repo_hidden_size = int(model_channels * repo_hidden_ratio)
        self.norm = operations.LayerNorm(model_channels, dtype=dtype, device=device)
        self.gate_map = operations.Linear(model_channels, repo_hidden_size, bias=False, dtype=dtype, device=device)
        self.content_map = operations.Linear(model_channels, repo_hidden_size, bias=False, dtype=dtype, device=device)
        self.act = nn.SiLU()
        self.final_map = operations.Linear(repo_hidden_size, 3 * num_heads, bias=False, dtype=dtype, device=device)
        self.dim_0 = 2 * (head_dim // 6)
        self.dim_1 = 2 * (head_dim // 6)
        self.dim_2 = head_dim - self.dim_0 - self.dim_1
        dims = [self.dim_0, self.dim_1, self.dim_2]
        freqs_list = []
        for d in dims:
            freq_dim = d // 2
            freqs_list.append(torch.linspace(1.0, float(max_freq), steps=freq_dim, dtype=torch.float32))
        self.freqs_0 = nn.Parameter(freqs_list[0])
        self.freqs_1 = nn.Parameter(freqs_list[1])
        self.freqs_2 = nn.Parameter(freqs_list[2])

    def forward(self, hidden_states):
        h = self.norm(hidden_states)
        feat = self.act(self.gate_map(h)) * self.content_map(h)
        out = self.final_map(feat)
        B, L, _ = out.shape
        delta_pos = out.reshape(B, L, self.num_heads, 3)
        f0 = comfy.model_management.cast_to(self.freqs_0, torch.float32, out.device)
        f1 = comfy.model_management.cast_to(self.freqs_1, torch.float32, out.device)
        f2 = comfy.model_management.cast_to(self.freqs_2, torch.float32, out.device)
        ang_0 = delta_pos[..., 0].unsqueeze(-1) * f0 * torch.pi
        ang_1 = delta_pos[..., 1].unsqueeze(-1) * f1 * torch.pi
        ang_2 = delta_pos[..., 2].unsqueeze(-1) * f2 * torch.pi
        ang = torch.cat([ang_0, ang_1, ang_2], dim=-1).float()  # (B, L, heads, head_dim/2)
        cos, sin = ang.cos(), ang.sin()
        return torch.stack([cos, -sin, sin, cos], dim=-1).reshape(*ang.shape, 2, 2)


class PcdAbsolutePositionEmbedder(nn.Module):
    # Sinusoidal absolute position embedding. Two fixed schedules are used in TripoSplat:
    # "pow2" (flow-model latent anchors) and "log2" (octree / gaussian decoders).
    def __init__(self, channels: int, in_channels: int = 3, max_res: int = 16, schedule: str = "pow2"):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.max_res = max_res
        self.schedule = schedule
        self.freq_dim = channels // in_channels // 2

    def _freqs(self, device):
        if self.schedule == "pow2":
            freqs_2exp = torch.arange(self.max_res, dtype=torch.float32, device=device)
            res_dim = max(0, self.freq_dim - self.max_res)
            freqs_res = (torch.arange(res_dim, dtype=torch.float32, device=device) / max(res_dim, 1) * self.max_res
                         if res_dim > 0 else torch.empty(0, device=device))
            freqs = torch.cat([freqs_2exp, freqs_res], dim=0)[:self.freq_dim]
            return torch.pow(2.0, freqs) * 2.0  # *2 folds this schedule's 2*pi into the shared *pi below
        logs = torch.linspace(0.0, float(self.max_res), steps=self.freq_dim, dtype=torch.float32, device=device)
        return torch.pow(2.0, logs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        *dims, D = x.shape
        out = torch.outer(x.reshape(-1), self._freqs(x.device)) * torch.pi
        out = torch.cat([out.sin(), out.cos()], dim=-1).reshape(*dims, -1)
        if out.shape[-1] < self.channels:
            out = torch.cat([out, torch.zeros(*dims, self.channels - out.shape[-1],
                                              device=out.device, dtype=out.dtype)], dim=-1)
        return out.to(orig_dtype)


def attention(q, k, v, transformer_options=None):
    # q, k, v: (B, L, heads, dim) -> (B, L, heads, dim). Shared optimized_attention call convention.
    out = optimized_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), heads=q.shape[2],
                              skip_reshape=True, skip_output_reshape=True, low_precision_attention=False,
                              transformer_options=transformer_options)
    return out.transpose(1, 2)


# Transformer building blocks

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dtype=None, device=None, operations=None):
        super().__init__()
        self.mlp = nn.Sequential(
            operations.Linear(in_channels, hidden_channels, dtype=dtype, device=device),
            nn.GELU(approximate="tanh"),
            operations.Linear(hidden_channels, out_channels, dtype=dtype, device=device),
        )

    def forward(self, x):
        return self.mlp(x)


class RopeMultiHeadAttention(nn.Module):
    def __init__(self, channels, num_heads, qkv_bias=True, qk_rms_norm=False, use_rope=False,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.qk_rms_norm = qk_rms_norm
        self.use_rope = use_rope
        self.qkv = operations.Linear(channels, channels * 3, bias=qkv_bias, dtype=dtype, device=device)
        if self.qk_rms_norm:
            self.q_norm = MultiHeadRMSNorm(self.head_dim, num_heads, dtype=dtype, device=device)
            self.k_norm = MultiHeadRMSNorm(self.head_dim, num_heads, dtype=dtype, device=device)
        self.out = operations.Linear(channels, channels, dtype=dtype, device=device)

    def forward(self, x, rope_emb=None, transformer_options=None):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        if self.use_rope:
            q, k = apply_rope(q, k, rope_emb)
        if self.qk_rms_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        h = attention(q, k, v, transformer_options)  # (B, L, heads, dim)
        return self.out(h.reshape(B, L, C))


class UnifiedTransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, mlp_ratio=4.0,
                 use_rope=False, qk_rms_norm=False, qkv_bias=True,
                 modulation=True, share_mod=False,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.modulation = modulation
        self.share_mod = share_mod
        self.norm1 = operations.LayerNorm(channels, elementwise_affine=not modulation, eps=1e-6, dtype=dtype, device=device)
        self.norm2 = operations.LayerNorm(channels, elementwise_affine=not modulation, eps=1e-6, dtype=dtype, device=device)
        self.attn = RopeMultiHeadAttention(channels, num_heads=num_heads,
                                           qkv_bias=qkv_bias, use_rope=use_rope, qk_rms_norm=qk_rms_norm,
                                           dtype=dtype, device=device, operations=operations)
        self.mlp = MLP(channels, int(channels * mlp_ratio), channels, dtype=dtype, device=device, operations=operations)
        if modulation:
            if not share_mod:
                self.adaLN_modulation = nn.Sequential(
                    nn.SiLU(), operations.Linear(channels, 6 * channels, bias=True, dtype=dtype, device=device))
            self.shift_table = nn.Parameter(torch.empty(1, 6 * channels, dtype=dtype, device=device))

    def forward(self, x, mod=None, rotary_emb=None, transformer_options=None):
        if self.modulation:
            if not self.share_mod:
                mod = self.adaLN_modulation(mod)
            mod = mod + comfy.model_management.cast_to(self.shift_table, mod.dtype, mod.device)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
            h = torch.addcmul(shift_msa.unsqueeze(1), self.norm1(x), 1 + scale_msa.unsqueeze(1))
            x = torch.addcmul(x, self.attn(h, rope_emb=rotary_emb, transformer_options=transformer_options), gate_msa.unsqueeze(1))
            h = torch.addcmul(shift_mlp.unsqueeze(1), self.norm2(x), 1 + scale_mlp.unsqueeze(1))
            x = torch.addcmul(x, self.mlp(h), gate_mlp.unsqueeze(1))
        else:
            x = x + self.attn(self.norm1(x), rope_emb=rotary_emb, transformer_options=transformer_options)
            x = x + self.mlp(self.norm2(x))
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=None, device=None, operations=None):
        super().__init__()
        self.mlp = nn.Sequential(
            operations.Linear(frequency_embedding_size, hidden_size, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        emb = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


class LatentSeqMMFlowModel(nn.Module):
    def __init__(self, image_model=None, q_token_length=8192, in_channels=16, model_channels=1024,
                 cond_channels=1280, out_channels=16, num_blocks=24, num_refiner_blocks=2,
                 num_heads=None, num_head_channels=64, cam_channels=5, cond2_channels=128,
                 mlp_ratio=4, share_mod=True, qk_rms_norm=True,
                 dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        self.q_token_length = q_token_length
        self.in_channels = in_channels
        self.cam_channels = cam_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.cond2_channels = cond2_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_refiner_blocks = num_refiner_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm

        factory_kwargs = dict(dtype=dtype, device=device)
        op_kwargs = dict(operations=operations, **factory_kwargs)

        self.t_embedder = TimestepEmbedder(model_channels, **op_kwargs)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), operations.Linear(model_channels, 6 * model_channels, bias=True, **factory_kwargs))

        self.input_layer = operations.Linear(in_channels, model_channels, **factory_kwargs)
        self.cond_embedder = operations.Linear(cond_channels, model_channels, **factory_kwargs)
        self.cond_embedder2 = operations.Linear(cond2_channels, model_channels, **factory_kwargs) if cond2_channels is not None else None

        # Fixed Sobol (low-discrepancy) 3D anchor positions for the latent tokens, used as positional encoding.
        # The embedder is parameter-free and the anchors are fixed, precompute once.
        sobol_seq = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=123).draw(q_token_length)
        pos_emb = PcdAbsolutePositionEmbedder(model_channels)(sobol_seq.unsqueeze(0))
        self.register_buffer("pos_emb", pos_emb, persistent=False)

        # RePo3DRotaryEmbedding layers for the refiner and main blocks
        repo_kwargs = dict(num_heads=self.num_heads, head_dim=num_head_channels, **op_kwargs)
        self.noise_repo_layers = nn.ModuleList(
            [RePo3DRotaryEmbedding(model_channels, **repo_kwargs) for _ in range(num_refiner_blocks)])
        self.context_repo_layers = nn.ModuleList(
            [RePo3DRotaryEmbedding(model_channels, **repo_kwargs) for _ in range(num_refiner_blocks)])
        self.repo_layers = nn.ModuleList(
            [RePo3DRotaryEmbedding(model_channels, **repo_kwargs) for _ in range(num_blocks)])

        # Refiner blocks
        block_kwargs = dict(num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, use_rope=True, qk_rms_norm=self.qk_rms_norm, **op_kwargs)
        self.noise_refiner = nn.ModuleList(
            [UnifiedTransformerBlock(model_channels, modulation=True, share_mod=self.share_mod, **block_kwargs) for _ in range(num_refiner_blocks)])
        self.context_refiner = nn.ModuleList(
            [UnifiedTransformerBlock(model_channels, modulation=False, **block_kwargs) for _ in range(num_refiner_blocks)])

        self.cam_refiner = MLP(self.cam_channels, model_channels, model_channels, **op_kwargs)

        self.blocks = nn.ModuleList(
            [UnifiedTransformerBlock(model_channels, modulation=True, share_mod=self.share_mod, **block_kwargs) for _ in range(num_blocks)])

        self.shift_table = nn.Parameter(torch.empty(1, 2, model_channels, **factory_kwargs))
        self.out_layer = operations.Linear(model_channels, out_channels, **factory_kwargs)
        self.cam_out_layer = operations.Linear(model_channels, cam_channels, **factory_kwargs)

    def forward(self, x, t, context=None, ref_latents=None, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, t, context, ref_latents, transformer_options, **kwargs)

    def _forward(self, x, t, context=None, ref_latents=None, transformer_options={}, **kwargs):
        # x is the unpacked nested latent: [latent (B,8192,in_channels), camera (B,1,cam_channels)].
        # context == feature1.
        z, camera = x[0], x[1]
        feat1 = context

        h_x = self.input_layer(z)
        h_cond = self.cond_embedder(feat1)
        if ref_latents is not None and self.cond_embedder2 is not None:
            # Flatten the Flux2 VAE latent (B,128,h,w) to a token sequence and front-pad to feat1's length
            # (the pad count = feat1's prefix tokens: DINOv3 cls + registers), then add to the context.
            feat2 = ref_latents[0].flatten(2).transpose(1, 2)
            feat2 = F.pad(feat2, (0, 0, feat1.shape[1] - feat2.shape[1], 0))
            h_cond = h_cond + self.cond_embedder2(feat2.to(h_cond.dtype))
        t_emb = self.t_embedder(t)
        t_mod = self.adaLN_modulation(t_emb) if self.share_mod else t_emb

        h_x = h_x + self.pos_emb.to(z)

        for i, block in enumerate(self.noise_refiner):
            h_x = block(h_x, mod=t_mod, rotary_emb=self.noise_repo_layers[i](h_x), transformer_options=transformer_options)

        for i, block in enumerate(self.context_refiner):
            h_cond = block(h_cond, mod=None, rotary_emb=self.context_repo_layers[i](h_cond), transformer_options=transformer_options)

        cam = camera.to(z)
        h_cam = self.cam_refiner(cam)
        h = torch.cat([h_x, h_cond, h_cam], dim=1)

        for i, block in enumerate(self.blocks):
            h = block(h, mod=t_mod, rotary_emb=self.repo_layers[i](h), transformer_options=transformer_options)

        h_x = F.layer_norm(h[:, :z.shape[1]].float(), h.shape[-1:]).to(z)
        h_cam = F.layer_norm(h[:, -cam.shape[1]:].float(), h.shape[-1:]).to(z)

        shift, scale = (comfy.model_management.cast_to(self.shift_table, t_emb.dtype, t_emb.device) + t_emb.unsqueeze(1)).chunk(2, dim=1)
        scale = 1 + scale
        h_x = torch.addcmul(shift, h_x, scale)
        h_cam = torch.addcmul(shift, h_cam, scale)

        return self.out_layer(h_x), self.cam_out_layer(h_cam)
