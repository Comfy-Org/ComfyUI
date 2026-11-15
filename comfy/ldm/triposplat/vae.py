# TripoSplat gaussian decoder ("VAE"): an octree probability decoder picks point coords, then an
# elastic-gaussian decoder predicts per-point gaussian params. OctreeGaussianDecoder.decode() returns
# a Gaussian. The octree sampler uses the global torch RNG (no generator) like upstream, so seed it for repeatable decodes.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
import comfy.ops
from .gaussian import build_gaussian_models
from .model import MultiHeadRMSNorm, MLP, PcdAbsolutePositionEmbedder, attention


# Quasi-random sampling utilities (pure functions, dtype/device-agnostic)

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]


def radical_inverse(base, n):
    val = 0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val


def halton_sequence(dim, n):
    return [radical_inverse(PRIMES[i], n) for i in range(dim)]


def hammersley_sequence(dim, n, num_samples):
    return [n / num_samples] + halton_sequence(dim - 1, n)


def sample_probs(probs, counts, generator=None):
    # Systematic resampling: distribute counts[r] draws across the P bins of row r
    batch_shape = counts.shape
    R = counts.numel()
    P = probs.size(-1)
    device = probs.device
    probs = probs.reshape(R, P).to(torch.float32).clamp_min(0)
    counts = counts.reshape(R).to(device=device, dtype=torch.long)

    row_sums = probs.sum(1, keepdim=True)
    probs = torch.where(row_sums == 0, probs.new_tensor(1.0 / P), probs / row_sums.clamp_min(1))
    cdf = probs.cumsum(dim=1).clamp(max=1.0 - 1e-12)

    Nmax = int(counts.max())
    if Nmax == 0:
        return counts.new_zeros(*batch_shape, P)
    cnt = counts.clamp_min(1).float().unsqueeze(1)                              # (R, 1)
    grid = torch.arange(Nmax, device=device, dtype=torch.float32).unsqueeze(0)  # (1, Nmax)
    u = (torch.rand(R, 1, generator=generator).to(device) + grid) / cnt         # (R, Nmax) systematic samples (CPU-seeded)
    idx = torch.searchsorted(cdf, u.clamp(max=1.0 - 1e-12)).clamp_max(P - 1)
    weight = (grid < counts.unsqueeze(1)).to(cdf.dtype)                         # mask out j >= counts[r]
    out = torch.zeros(R, P, dtype=torch.float32, device=device)
    out.scatter_add_(1, idx, weight)
    return out.to(torch.long).view(*batch_shape, P)


class MultiHeadAttention(nn.Module):
    def __init__(self, channels, num_heads, ctx_channels=None, type="self", qkv_bias=True, qk_rms_norm=False,
                 dtype=None, device=None, operations=None):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.qk_rms_norm = qk_rms_norm
        if self._type == "self":
            self.to_qkv = operations.Linear(channels, channels * 3, bias=qkv_bias, dtype=dtype, device=device)
        else:
            self.to_q = operations.Linear(channels, channels, bias=qkv_bias, dtype=dtype, device=device)
            self.to_kv = operations.Linear(self.ctx_channels, channels * 2, bias=qkv_bias, dtype=dtype, device=device)
        if self.qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads, dtype=dtype, device=device)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads, dtype=dtype, device=device)
        self.to_out = operations.Linear(channels, channels, dtype=dtype, device=device)

    def forward(self, x, context=None):
        B, L, C = x.shape
        if self._type == "self":
            q, k, v = self.to_qkv(x).reshape(B, L, 3, self.num_heads, -1).unbind(dim=2)
        else:
            Lkv = context.shape[1]
            q = self.to_q(x).reshape(B, L, self.num_heads, -1)
            k, v = self.to_kv(context).reshape(B, Lkv, 2, self.num_heads, -1).unbind(dim=2)
        if self.qk_rms_norm:
            q = self.q_rms_norm(q)
            k = self.k_rms_norm(k)
        h = attention(q, k, v)
        return self.to_out(h.reshape(B, L, -1))


# Octree probability decoder

class LevelEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, max_period=1024,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.mlp = nn.Sequential(
            operations.Linear(frequency_embedding_size, hidden_size, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

    @staticmethod
    def level_embedding(t, dim, max_period=1024):
        half = dim // 2
        freqs = torch.exp(-np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None] * 2 * torch.pi
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        emb = self.level_embedding(t, self.frequency_embedding_size, self.max_period)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


class ModulatedTransformerCrossOnlyBlock(nn.Module):
    def __init__(self, channels, ctx_channels, num_heads, mlp_ratio=4.0, share_mod=False,
                 qk_rms_norm_cross=True, qkv_bias=True, dtype=None, device=None, operations=None):
        super().__init__()
        self.share_mod = share_mod
        self.norm1 = operations.LayerNorm(channels, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.norm2 = operations.LayerNorm(channels, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.cross_attn = MultiHeadAttention(channels, ctx_channels=ctx_channels, num_heads=num_heads,
                                             type="cross", qkv_bias=qkv_bias,
                                             qk_rms_norm=qk_rms_norm_cross, dtype=dtype, device=device, operations=operations)
        self.mlp = MLP(channels, int(channels * mlp_ratio), channels, dtype=dtype, device=device, operations=operations)
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), operations.Linear(channels, 6 * channels, bias=True, dtype=dtype, device=device))

    def forward(self, x, mod, context):
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = torch.addcmul(shift_msa.unsqueeze(1), self.norm1(x), 1 + scale_msa.unsqueeze(1))
        x = torch.addcmul(x, self.cross_attn(h, context), gate_msa.unsqueeze(1))
        h = torch.addcmul(shift_mlp.unsqueeze(1), self.norm2(x), 1 + scale_mlp.unsqueeze(1))
        x = torch.addcmul(x, self.mlp(h), gate_mlp.unsqueeze(1))
        return x


class OctreeProbabilityFixedlenDecoder(nn.Module):
    # Cross-attention transformer over octree coords -> per-node 8-way child occupancy logits.
    def __init__(self, model_channels=1024, cond_channels=16, num_blocks=4, num_heads=16,
                 num_head_channels=64, mlp_ratio=4.0, share_mod=True,
                 qk_rms_norm_cross=True, dtype=None, device=None, operations=None):
        super().__init__()
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.share_mod = share_mod
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.input_layer = operations.Linear(model_channels, model_channels, dtype=dtype, device=device)
        self.l_embedder = LevelEmbedder(model_channels, dtype=dtype, device=device, operations=operations)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), operations.Linear(model_channels, 6 * model_channels, bias=True, dtype=dtype, device=device))
        if cond_channels is not None:
            self.blocks = nn.ModuleList([
                ModulatedTransformerCrossOnlyBlock(
                    model_channels, ctx_channels=cond_channels, num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio, qk_rms_norm_cross=self.qk_rms_norm_cross,
                    share_mod=self.share_mod, dtype=dtype, device=device, operations=operations)
                for _ in range(num_blocks)
            ])
        self.out_proj = operations.Linear(model_channels, 8, dtype=dtype, device=device)
        self.in_proj = operations.Linear(3, model_channels, dtype=dtype, device=device)
        self.pos_embedder = PcdAbsolutePositionEmbedder(channels=model_channels, in_channels=3, max_res=10, schedule="log2")

    def forward(self, x, l, cond):
        d = next(self.parameters()).dtype
        B, L, _ = x.shape
        h = self.in_proj(x.to(d)) + self.pos_embedder(x.reshape(-1, 3)).reshape(B, L, -1).to(d)
        h = self.input_layer(h)
        l_emb = self.l_embedder(l)
        if self.share_mod:
            l_emb = self.adaLN_modulation(l_emb)
        cond = cond.to(d)
        for block in self.blocks:
            h = block(h, l_emb, cond)
        h = F.layer_norm(h.float(), h.shape[-1:]).to(d)
        logits = self.out_proj(h)
        return {"logits": logits, "probs": torch.softmax(logits, dim=-1)}

    @staticmethod
    def sample(model, cond, num_points, level, temperature=1.0, generator=None):
        B = cond.shape[0]
        device = cond.device
        child_offset = torch.tensor([[i, j, k] for k in [0, 1] for j in [0, 1] for i in [0, 1]],
                                    dtype=torch.long, device=device)
        prev_coords_int = torch.zeros(B, 1, 3, dtype=torch.long, device=device)
        prev_counts = torch.full((B, 1), num_points, dtype=torch.long, device=device)
        prev_log_probs = torch.zeros(B, 1, dtype=torch.float32, device=device)
        batch_indices_range = torch.arange(B, device=device).unsqueeze(1)

        for lv in range(1, level + 1):
            res_p = 1 << (lv - 1)
            res = 1 << lv
            parent_coords_norm = (prev_coords_int.to(torch.float32) + 0.5) / res_p
            res_tensor = torch.full((B,), res, dtype=torch.long, device=device)
            pred_logits = model(parent_coords_norm, res_tensor, cond)["logits"] / temperature
            pred_probs = torch.softmax(pred_logits, dim=-1)
            pred_log_probs = torch.log_softmax(pred_logits, dim=-1)
            sampled = sample_probs(pred_probs, prev_counts, generator=generator).flatten(1, 2)
            pred_log_probs = pred_log_probs.flatten(1, 2)
            prev_log_probs_expanded = prev_log_probs.repeat_interleave(8, dim=1)
            child_coords_int = (prev_coords_int[:, :, None, :] * 2 + child_offset[None, None, :, :]).flatten(1, 2)
            mask = sampled > 0
            max_valid = mask.sum(dim=1).max().item()
            scatter_indices = mask.cumsum(dim=1) - 1
            valid_scatter_indices = scatter_indices[mask]
            valid_batch_indices = batch_indices_range.expand_as(mask)[mask]
            next_prev_coords_int = torch.zeros(B, max_valid, 3, dtype=child_coords_int.dtype, device=device)
            next_prev_coords_int[valid_batch_indices, valid_scatter_indices] = child_coords_int[mask]
            next_prev_counts = torch.zeros(B, max_valid, dtype=sampled.dtype, device=device)
            next_prev_counts[valid_batch_indices, valid_scatter_indices] = sampled[mask]
            next_prev_log_probs = torch.zeros(B, max_valid, dtype=prev_log_probs.dtype, device=device)
            next_prev_log_probs[valid_batch_indices, valid_scatter_indices] = (prev_log_probs_expanded + pred_log_probs)[mask]
            prev_coords_int = next_prev_coords_int
            prev_counts = next_prev_counts
            prev_log_probs = next_prev_log_probs

        res = 1 << level
        prev_log_probs = torch.repeat_interleave(prev_log_probs.flatten(0, 1), prev_counts.flatten(0, 1), dim=0).reshape(B, num_points)
        coords_int = torch.repeat_interleave(prev_coords_int.flatten(0, 1), prev_counts.flatten(0, 1), dim=0).reshape(B, num_points, -1)
        rand = torch.rand(coords_int.shape, dtype=torch.float32, generator=generator).to(device)
        coords_norm = (coords_int.to(torch.float32) + rand) / res
        return {"points": coords_norm, "log_probs": prev_log_probs}


# Elastic gaussian decoder

class TransformerCrossBlock(nn.Module):
    def __init__(self, channels, ctx_channels, num_heads, mlp_ratio=4.0,
                 qk_rms_norm=True, qk_rms_norm_cross=True, qkv_bias=True,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.norm1 = operations.LayerNorm(channels, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.norm2 = operations.LayerNorm(channels, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
        self.norm3 = operations.LayerNorm(channels, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.self_attn = MultiHeadAttention(channels, num_heads=num_heads, type="self", qkv_bias=qkv_bias,
                                            qk_rms_norm=qk_rms_norm, dtype=dtype, device=device, operations=operations)
        self.cross_attn = MultiHeadAttention(channels, ctx_channels=ctx_channels, num_heads=num_heads, type="cross",
                                             qkv_bias=qkv_bias, qk_rms_norm=qk_rms_norm_cross, dtype=dtype, device=device, operations=operations)
        self.mlp = MLP(channels, int(channels * mlp_ratio), channels, dtype=dtype, device=device, operations=operations)

    def forward(self, x, context):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), context)
        x = x + self.mlp(self.norm3(x))
        return x


class ElasticGaussianFixedlenDecoder(nn.Module):
    # Cross-attention transformer over sampled octree points -> per-point gaussian params.
    def __init__(self, in_channels=3, model_channels=1024, cond_channels=16, num_blocks=16, num_heads=16,
                 num_head_channels=64, mlp_ratio=4.0, *, representation_config=None,
                 qk_rms_norm=True, qk_rms_norm_cross=True, dtype=None, device=None, operations=None):
        super().__init__()
        self.rep_config = representation_config or dict(
            lr=dict(_xyz=1.0, _features_dc=1.0, _opacity=1.0, _scaling=1.0, _rotation=0.1),
            perturb_offset=True, perturbe_size=1.5, offset_scale=0.05, num_gaussians=32,
            filter_kernel_size_3d=0.0009, scaling_bias=0.004, opacity_bias=0.1,
            scaling_activation="softplus",
        )
        self.out_channels = self._calc_layout()
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.input_layer = operations.Linear(model_channels, model_channels, dtype=dtype, device=device)
        if cond_channels is not None:
            self.blocks = nn.ModuleList([
                TransformerCrossBlock(model_channels, ctx_channels=cond_channels,
                                      num_heads=self.num_heads, mlp_ratio=self.mlp_ratio,
                                      qk_rms_norm=qk_rms_norm, qk_rms_norm_cross=qk_rms_norm_cross,
                                      dtype=dtype, device=device, operations=operations)
                for _ in range(num_blocks)
            ])
        self.in_proj = operations.Linear(in_channels, model_channels, dtype=dtype, device=device)
        self.pos_embedder = PcdAbsolutePositionEmbedder(channels=model_channels, in_channels=3, max_res=10, schedule="log2")
        self.out_proj = operations.Linear(model_channels, self.out_channels, dtype=dtype, device=device)
        self._build_perturbation()

    def _calc_layout(self):
        ng = self.rep_config['num_gaussians']
        self.layout = {
            '_xyz':         {'shape': (ng, 3),    'size': ng * 3},
            '_features_dc': {'shape': (ng, 1, 3), 'size': ng * 3},
            '_scaling':     {'shape': (ng, 3),    'size': ng * 3},
            '_rotation':    {'shape': (ng, 4),    'size': ng * 4},
            '_opacity':     {'shape': (ng, 1),    'size': ng},
        }
        self.layout['_offset_scale'] = {'shape': (ng, 1), 'size': ng}
        start = 0
        for k, v in self.layout.items():
            v['range'] = (start, start + v['size'])
            start += v['size']
        return start

    def _build_perturbation(self):
        ng = self.rep_config['num_gaussians']
        perturbation = torch.tensor([hammersley_sequence(3, i, ng) for i in range(ng)]).float()
        perturbation = torch.atanh((perturbation * 2 - 1) / self.rep_config['perturbe_size'])
        self.register_buffer('points_offset_perturbation', perturbation)
        base = torch.tensor(self.rep_config['offset_scale'])
        self.register_buffer('base_offset_scale', torch.log(torch.exp(base) - 1.0))

    def _get_offset(self, h):
        B = h.shape[0]
        r = self.layout['_offset_scale']['range']
        _offset_scale = F.softplus(
            h[:, :, r[0]:r[1]].reshape(B, -1, *self.layout['_offset_scale']['shape'])
            + comfy.model_management.cast_to(self.base_offset_scale, h.dtype, h.device))

        r = self.layout['_xyz']['range']
        offset = h[:, :, r[0]:r[1]].reshape(B, -1, *self.layout['_xyz']['shape'])
        offset = offset * self.rep_config['lr']['_xyz']
        if self.rep_config['perturb_offset']:
            offset = offset + comfy.model_management.cast_to(self.points_offset_perturbation, offset.dtype, offset.device)
        offset = torch.tanh(offset) * 0.5 * self.rep_config['perturbe_size']
        offset = offset * _offset_scale
        return offset

    def forward(self, x=None, cond=None):
        pcd = x["points"]
        d = next(self.parameters()).dtype
        B, L, _ = pcd.shape
        h = self.in_proj(pcd.to(d)) + self.pos_embedder(pcd.reshape(-1, 3)).reshape(B, L, -1).to(d)
        h = self.input_layer(h)
        cond = cond.to(d)
        for block in self.blocks:
            h = block(h, cond)
        h = F.layer_norm(h.float(), h.shape[-1:]).to(h.dtype)
        return {"features": self.out_proj(h)}


# Combined octree gaussian decoder (comfy first-stage model)

class OctreeGaussianDecoder(nn.Module):
    _MAX_VOXEL_LEVEL = 8

    def __init__(self, dtype=None, device=None, operations=None):
        super().__init__()
        if operations is None:
            operations = comfy.ops.disable_weight_init
        self.octree = OctreeProbabilityFixedlenDecoder(dtype=dtype, device=device, operations=operations)
        self.gs = ElasticGaussianFixedlenDecoder(dtype=dtype, device=device, operations=operations)

    @property
    def gaussians_per_point(self) -> int:
        return self.gs.rep_config['num_gaussians']

    def decode(self, latent: torch.Tensor, num_gaussians: int, level: int = None, generator=None):
        # level defaults to the full octree depth, a lower level is cheaper (coarser) for live previews.
        # generator (a CPU torch.Generator) makes the octree sampling reproducible without touching global RNG.
        level = self._MAX_VOXEL_LEVEL if level is None else level
        num_decoder_tokens = max(1, num_gaussians // self.gaussians_per_point)
        points_pred = OctreeProbabilityFixedlenDecoder.sample(
            self.octree, latent, num_points=num_decoder_tokens, level=level, temperature=1.0, generator=generator,
        )
        pred = self.gs(x=points_pred, cond=latent)
        return build_gaussian_models(self.gs, points_pred, pred)  # one GaussianModel per batch item
