# Prism audio-video transformer, adapted from Tencent-Hunyuan/Prism.
# MIT + third-party attributions: see LICENSE in this directory.
import math

import torch
from torch import nn
import torch.nn.functional as F

import comfy.ops
import comfy.model_prefetch
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention
from comfy.ldm.common_dit import pad_to_patch_size


def timestep_embedding(dim, timestep):
    frequency = torch.pow(10000, -torch.arange(dim // 2, device=timestep.device, dtype=torch.float64) / (dim // 2))
    phase = torch.outer(timestep.double(), frequency)
    return torch.cat((phase.cos(), phase.sin()), dim=-1).to(timestep.dtype)


def rotary_table(dim, length):
    frequency = 1.0 / (10000 ** (torch.arange(0, dim, 2, device='cpu').double() / dim))
    phase = torch.outer(torch.arange(length, device='cpu', dtype=torch.float64), frequency)
    return torch.polar(torch.ones_like(phase), phase)


def apply_rotary(x, freqs, heads):
    # Released Prism rotates adjacent real/imaginary pairs in FP64 and rounds
    # only once. Lower-precision intermediate arithmetic is not equivalent.
    pairs = x.reshape(x.shape[0], x.shape[1], heads, -1, 2).double()
    rotated = torch.view_as_complex(pairs) * freqs
    return torch.view_as_real(rotated).flatten(2).to(x.dtype)


def bridge_rotary(x, positions):
    frequency = 1.0 / (10000 ** (torch.arange(0, 128, 2, device=x.device, dtype=torch.int64).float() / 128))
    with torch.autocast(x.device.type, enabled=False):
        phase = (frequency[None, :, None] @ positions[None, None, :]).transpose(1, 2)
        phase = torch.cat((phase, phase), dim=-1)
    return phase.cos().to(x.dtype), phase.sin().to(x.dtype)


def apply_bridge_rotary(x, freqs, heads):
    x = x.reshape(x.shape[0], x.shape[1], heads, -1)
    cos, sin = (f.unsqueeze(2).to(x) for f in freqs)
    first, second = x.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return (x * cos + rotated * sin).flatten(2)


class AttentionModule(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads
        self.comfy_attention = ComfyAttention()

    def forward(self, q, k, v, transformer_options):
        return optimized_attention(AttentionTensorContainer(q), AttentionTensorContainer(k),
            AttentionTensorContainer(v), self.heads, preferred_attention=self.comfy_attention,
            transformer_options=transformer_options)


class SelfAttention(nn.Module):
    def __init__(self, dim, heads, eps, device, dtype, operations):
        super().__init__()
        self.heads = heads
        self.q = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.k = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.v = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.o = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.norm_q = operations.RMSNorm(dim, eps=eps, device=device, dtype=dtype)
        self.norm_k = operations.RMSNorm(dim, eps=eps, device=device, dtype=dtype)
        self.attn = AttentionModule(heads)

    def forward(self, x, freqs, grid, sparse, options):
        q = apply_rotary(self.norm_q(self.q(x)), freqs, self.heads)
        k = apply_rotary(self.norm_k(self.k(x)), freqs, self.heads)
        v = self.v(x)
        attention = options.get('prism_attention', 'prism_sparse_tail_safe')
        if sparse and attention != 'dense' and grid[0] > 1:
            from .sparse import sparse_attention
            out = sparse_attention(q, k, v, self.heads, grid,
                options.get('prism_sparsity', 0.75), options.get('prism_cdf_threshold', 0.2))
            if attention == 'prism_sparse_tail_safe' and grid[0] % 4:
                start = (grid[0] // 4) * 4 * grid[1] * grid[2]
                tail = self.attn(q[:, start:], k, v, options)
                out = torch.cat((out[:, :start], tail), dim=1)
        else:
            out = self.attn(q, k, v, options)
        return self.o(out)


class CrossAttention(nn.Module):
    def __init__(self, dim, kv_dim, heads, eps, device, dtype, operations):
        super().__init__()
        self.heads = heads
        self.q = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.k = operations.Linear(kv_dim, dim, device=device, dtype=dtype)
        self.v = operations.Linear(kv_dim, dim, device=device, dtype=dtype)
        self.o = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.norm_q = operations.RMSNorm(dim, eps=eps, device=device, dtype=dtype)
        self.norm_k = operations.RMSNorm(dim, eps=eps, device=device, dtype=dtype)
        self.attn = AttentionModule(heads)

    def forward(self, x, context, options, q_freqs=None, k_freqs=None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        v = self.v(context)
        if q_freqs is not None:
            q = apply_bridge_rotary(q, q_freqs, self.heads)
        if k_freqs is not None:
            k = apply_bridge_rotary(k, k_freqs, self.heads)
        return self.o(self.attn(q, k, v, options))


class Conditioner(nn.Module):
    def __init__(self, dim, kv_dim, device, dtype, operations):
        super().__init__()
        self.y_norm = operations.LayerNorm(kv_dim, eps=1e-6, device=device, dtype=dtype)
        self.inner = CrossAttention(dim, kv_dim, dim // 128, 1e-6, device, dtype, operations)

    def forward(self, x, context, x_freqs, context_freqs, options):
        return self.inner(x, self.y_norm(context), options, x_freqs, context_freqs)


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, ffn_dim, eps, device, dtype, operations):
        super().__init__()
        self.self_attn = SelfAttention(dim, heads, eps, device, dtype, operations)
        self.cross_attn = CrossAttention(dim, dim, heads, eps, device, dtype, operations)
        self.norm1 = operations.LayerNorm(dim, eps=eps, elementwise_affine=False, device=device, dtype=dtype)
        self.norm2 = operations.LayerNorm(dim, eps=eps, elementwise_affine=False, device=device, dtype=dtype)
        self.norm3 = operations.LayerNorm(dim, eps=eps, device=device, dtype=dtype)
        self.ffn = nn.Sequential(operations.Linear(dim, ffn_dim, device=device, dtype=dtype),
            nn.GELU(approximate='tanh'), operations.Linear(ffn_dim, dim, device=device, dtype=dtype))
        self.modulation = nn.Parameter(torch.empty(1, 6, dim, device=device, dtype=dtype))

    def forward(self, x, context, modulation, freqs, grid, sparse, options):
        shift, scale, gate, mlp_shift, mlp_scale, mlp_gate = (
            comfy.ops.cast_to_input(self.modulation, modulation) + modulation).chunk(6, dim=1)
        out = self.norm1(x) * (1 + scale) + shift
        x = x + gate * self.self_attn(out, freqs, grid, sparse, options)
        x = x + self.cross_attn(self.norm3(x), context, options)
        out = self.norm2(x) * (1 + mlp_scale) + mlp_shift
        chunks = options.get('prism_ffn_chunks', 1)
        if chunks > 1:
            out = torch.cat([self.ffn(part) for part in out.chunk(chunks, dim=1)], dim=1)
        else:
            out = self.ffn(out)
        return x + mlp_gate * out


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch, device, dtype, operations):
        super().__init__()
        self.norm = operations.LayerNorm(dim, eps=1e-6, elementwise_affine=False, device=device, dtype=dtype)
        self.head = operations.Linear(dim, out_dim * math.prod(patch), device=device, dtype=dtype)
        self.modulation = nn.Parameter(torch.empty(1, 2, dim, device=device, dtype=dtype))

    def forward(self, x, time):
        shift, scale = (comfy.ops.cast_to_input(self.modulation, time) + time.unsqueeze(1)).chunk(2, dim=1)
        return self.head(self.norm(x) * (1 + scale) + shift)


class Tower(nn.Module):
    def __init__(self, dim, heads, ffn_dim, layers, audio, device, dtype, operations):
        super().__init__()
        self.dim = dim
        self.audio = audio
        self.patch = (1,) if audio else (1, 2, 2)
        if audio:
            self.patch_embedding = operations.Conv1d(128, dim, 1, device=device, dtype=dtype)
        else:
            self.patch_embedding = operations.Conv3d(36, dim, self.patch, stride=self.patch, device=device, dtype=dtype)
        self.text_embedding = nn.Sequential(operations.Linear(4096, dim, device=device, dtype=dtype),
            nn.GELU(approximate='tanh'), operations.Linear(dim, dim, device=device, dtype=dtype))
        self.time_embedding = nn.Sequential(operations.Linear(256, dim, device=device, dtype=dtype),
            nn.SiLU(), operations.Linear(dim, dim, device=device, dtype=dtype))
        self.time_projection = nn.Sequential(nn.SiLU(), operations.Linear(dim, dim * 6, device=device, dtype=dtype))
        self.blocks = nn.ModuleList(DiTBlock(dim, heads, ffn_dim, 1e-6, device, dtype, operations) for _ in range(layers))
        self.head = Head(dim, 128 if audio else 16, self.patch, device, dtype, operations)
        head_dim = dim // heads
        if audio:
            self.freqs = rotary_table(head_dim, 16384).chunk(3, dim=-1)
        else:
            self.freqs = tuple(rotary_table(d, 1024) for d in (head_dim - 2 * (head_dim // 3), head_dim // 3, head_dim // 3))

    def embed_time(self, timestep, dtype):
        with torch.autocast(timestep.device.type, enabled=False):
            time = self.time_embedding(timestep_embedding(256, timestep.float()))
            modulation = self.time_projection(time).unflatten(1, (6, self.dim))
        return time.to(dtype), modulation.to(dtype)

    def patchify(self, x):
        x = self.patch_embedding(x)
        grid = x.shape[2:]
        return x.flatten(2).transpose(1, 2).contiguous(), grid

    def assemble_freqs(self, grid, device):
        if self.audio:
            return torch.cat(tuple(f[:grid[0]] for f in self.freqs), dim=-1).reshape(grid[0], 1, -1).to(device)
        t, h, w = grid
        f = tuple(freq.to(device) for freq in self.freqs)
        return torch.cat((f[0][:t].view(t, 1, 1, -1).expand(t, h, w, -1),
            f[1][:h].view(1, h, 1, -1).expand(t, h, w, -1),
            f[2][:w].view(1, 1, w, -1).expand(t, h, w, -1)), dim=-1).reshape(t * h * w, 1, -1)

    def unpatchify(self, x, grid):
        if self.audio:
            return x.transpose(1, 2)
        t, h, w = grid
        return x.reshape(x.shape[0], t, h, w, 1, 2, 2, 16).permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(x.shape[0], 16, t, h * 2, w * 2)


class FusedBlock(nn.Module):
    def __init__(self, video_block, audio_block, device, dtype, operations):
        super().__init__()
        self.video_block = video_block
        self.audio_block = audio_block
        self.a2v_conditioner = Conditioner(5120, 1536, device, dtype, operations)
        self.v2a_conditioner = Conditioner(1536, 5120, device, dtype, operations)


class Prism(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None, image_model=None,
                 video_layers=40, audio_layers=30, boundary_ratio=0.9):
        super().__init__()
        self.dtype = dtype
        self.boundary_ratio = boundary_ratio
        self.video_dit = Tower(5120, 40, 13824, video_layers, False, device, dtype, operations)
        self.video_dit_2 = Tower(5120, 40, 13824, video_layers, False, device, dtype, operations)
        self.audio_dit = Tower(1536, 12, 8960, audio_layers, True, device, dtype, operations)
        count = min(video_layers, audio_layers)
        self.fusion_blocks = nn.ModuleList(FusedBlock(self.video_dit.blocks[i], self.audio_dit.blocks[i],
            device, dtype, operations) for i in range(count))
        self.remaining_video_blocks = nn.ModuleList(self.video_dit.blocks[count:])
        self.video_dit.blocks = nn.ModuleList()
        self.audio_dit.blocks = nn.ModuleList()

    def get_dynamic_units(self):
        units = []
        for fused in self.fusion_blocks:
            units.extend((fused.a2v_conditioner, fused.v2a_conditioner, fused.video_block, fused.audio_block))
        units.extend(self.remaining_video_blocks)
        units.extend(self.video_dit_2.blocks)
        return units

    def forward(self, video, audio, reference, context, audio_context, video_time, audio_time,
                fps, low_noise, transformer_options):
        original_shape = video.shape[2:]
        video = pad_to_patch_size(video, (1, 2, 2))
        reference = pad_to_patch_size(reference, (1, 2, 2))
        tower = self.video_dit_2 if low_noise else self.video_dit
        vt, vm = tower.embed_time(video_time, self.dtype)
        at, am = self.audio_dit.embed_time(audio_time, self.dtype)
        vc = tower.text_embedding(context)
        ac = self.audio_dit.text_embedding(audio_context)
        vx, grid = tower.patchify(torch.cat((video, reference), dim=1).to(self.dtype))
        ax, audio_grid = self.audio_dit.patchify(audio.to(self.dtype))
        vf = tower.assemble_freqs(grid, vx.device)
        af = self.audio_dit.assemble_freqs(audio_grid, ax.device)
        video_position = torch.arange(grid[0], device=vx.device, dtype=torch.float32) * (50.0 / (fps / 4.0))
        video_position = video_position.repeat_interleave(grid[1] * grid[2])
        audio_position = torch.arange(audio_grid[0], device=ax.device, dtype=torch.float32)
        vr = bridge_rotary(vx, video_position)
        ar = bridge_rotary(ax, audio_position)
        sequence = []
        for i, fused in enumerate(self.fusion_blocks):
            vb = self.video_dit_2.blocks[i] if low_noise else fused.video_block
            sequence.extend((fused.a2v_conditioner, fused.v2a_conditioner, vb, fused.audio_block))
        remaining = list(self.video_dit_2.blocks[len(self.fusion_blocks):]) if low_noise else list(self.remaining_video_blocks)
        sequence.extend(remaining)
        queue = comfy.model_prefetch.make_prefetch_queue(sequence, vx.device, transformer_options)
        for i, fused in enumerate(self.fusion_blocks):
            before_video = vx
            comfy.model_prefetch.prefetch_queue_pop(queue, vx.device, fused.a2v_conditioner)
            vx = vx + fused.a2v_conditioner(vx, ax, vr, ar, transformer_options)
            comfy.model_prefetch.prefetch_queue_pop(queue, ax.device, fused.v2a_conditioner)
            ax = ax + fused.v2a_conditioner(ax, before_video, ar, vr, transformer_options)
            vb = self.video_dit_2.blocks[i] if low_noise else fused.video_block
            comfy.model_prefetch.prefetch_queue_pop(queue, vx.device, vb)
            vx = vb(vx, vc, vm, vf, grid, True, transformer_options)
            comfy.model_prefetch.prefetch_queue_pop(queue, ax.device, fused.audio_block)
            ax = fused.audio_block(ax, ac, am, af, audio_grid, False, transformer_options)
        for block in remaining:
            comfy.model_prefetch.prefetch_queue_pop(queue, vx.device, block)
            vx = block(vx, vc, vm, vf, grid, True, transformer_options)
        comfy.model_prefetch.prefetch_queue_pop(queue, vx.device, None)
        video_out = tower.unpatchify(tower.head(vx, vt), grid)
        audio_out = self.audio_dit.unpatchify(self.audio_dit.head(ax, at), audio_grid)
        return video_out[:, :, :original_shape[0], :original_shape[1], :original_shape[2]], audio_out
