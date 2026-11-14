import torch
import torch.nn.functional as F
from torch import nn

import comfy.ldm.common_dit
import comfy.ldm.flux.math
import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy.patcher_extension
import comfy.quant_ops
from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.lightricks.model import FeedForward
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention

FFN_CHUNK = 8192


def modulated_norm(x, norm, scale, shift):
    if comfy.model_management.in_training:
        return torch.addcmul(shift, norm(x), 1 + scale)
    return comfy.quant_ops.ck.adaln(x, scale, shift, norm.eps)


def gated_residual(x, y, gate):
    # in place inside the memory compiler region, out of place keeps x for backward when training
    if comfy.model_management.in_training:
        return torch.addcmul(x, y, gate)
    return x.addcmul_(y, gate)


def rms_rope(q, k, q_norm, k_norm, pe):
    if comfy.model_management.in_training:
        return comfy.ldm.flux.math.apply_rope(q_norm(q), k_norm(k), pe)
    q_scale, _, q_stream = comfy.ops.cast_bias_weight(q_norm, q, offloadable=True)
    k_scale, _, k_stream = comfy.ops.cast_bias_weight(k_norm, k, offloadable=True)
    q, k = comfy.quant_ops.ck.rms_rope(q, k, pe, q_scale, k_scale, q_norm.eps)
    comfy.ops.uncast_bias_weight(q_norm, q_scale, None, q_stream)
    comfy.ops.uncast_bias_weight(k_norm, k_scale, None, k_stream)
    return q, k


def attention(q, k, v, heads, comfy_attention, transformer_options):
    q, k, v = AttentionTensorContainer(q.flatten(2)), AttentionTensorContainer(k.flatten(2)), AttentionTensorContainer(v)
    return optimized_attention(q, k, v, heads, transformer_options=transformer_options, preferred_attention=comfy_attention)


class SwiGLU(nn.Module):
    def __init__(self, dim_in, dim_out, dtype=None, device=None, operations=None):
        super().__init__()
        self.proj = operations.Linear(dim_in, 2 * dim_out, bias=False, dtype=dtype, device=device)


class RefinerFeedForward(nn.Module):
    def __init__(self, dim, inner_dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.net = nn.Sequential(SwiGLU(dim, inner_dim, dtype=dtype, device=device, operations=operations), nn.Identity(),
                                 operations.Linear(inner_dim, dim, bias=False, dtype=dtype, device=device))

    def forward(self, x):
        # the projection is laid out [value | gate]
        value, gate = self.net[0].proj(x).chunk(2, dim=-1)
        return self.net[2](value * F.silu(gate))


class RefinerAttention(nn.Module):
    def __init__(self, dim, num_heads, dtype=None, device=None, operations=None):
        super().__init__()
        self.heads = num_heads
        self.to_q = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.to_k = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.to_v = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.norm_q = operations.RMSNorm(dim // num_heads, eps=1e-5, dtype=dtype, device=device)
        self.norm_k = operations.RMSNorm(dim // num_heads, eps=1e-5, dtype=dtype, device=device)
        self.to_out = nn.ModuleList([operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)])


class RefinerBlock(nn.Module):
    def __init__(self, dim, num_heads, dtype=None, device=None, operations=None):
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.norm1 = operations.RMSNorm(dim, eps=1e-5, dtype=dtype, device=device)
        self.attn = RefinerAttention(dim, num_heads, dtype=dtype, device=device, operations=operations)
        self.norm2 = operations.RMSNorm(dim, eps=1e-5, dtype=dtype, device=device)
        self.ff = RefinerFeedForward(dim, 4 * dim, dtype=dtype, device=device, operations=operations)

    def forward(self, x, transformer_options={}):
        B, N, _ = x.shape
        y = self.norm1(x)
        q = self.attn.norm_q(self.attn.to_q(y).view(B, N, self.attn.heads, -1))
        k = self.attn.norm_k(self.attn.to_k(y).view(B, N, self.attn.heads, -1))
        v = self.attn.to_v(y)
        del y
        x = x + self.attn.to_out[0](attention(q, k, v, self.attn.heads, self.comfy_attention, transformer_options))
        return x + self.ff(self.norm2(x))


class JointAttention(nn.Module):
    def __init__(self, dim, num_heads, dtype=None, device=None, operations=None):
        super().__init__()
        self.heads = num_heads
        head_dim = dim // num_heads
        self.to_q = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.to_k = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.to_v = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.add_q_proj = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.add_k_proj = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.add_v_proj = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.norm_q = operations.RMSNorm(head_dim, eps=1e-6, dtype=dtype, device=device)
        self.norm_k = operations.RMSNorm(head_dim, eps=1e-6, dtype=dtype, device=device)
        self.norm_added_q = operations.RMSNorm(head_dim, eps=1e-6, dtype=dtype, device=device)
        self.norm_added_k = operations.RMSNorm(head_dim, eps=1e-6, dtype=dtype, device=device)
        self.to_out = nn.ModuleList([operations.Linear(dim, dim, dtype=dtype, device=device)])
        self.to_add_out = operations.Linear(dim, dim, dtype=dtype, device=device)

    def qkv(self, img, txt, pe):
        # video q/k get the 3D RoPE, the caption q/k only their norm
        B, N, _ = img.shape
        q, k = rms_rope(self.to_q(img).view(B, N, self.heads, -1), self.to_k(img).view(B, N, self.heads, -1), self.norm_q, self.norm_k, pe)
        B, N, _ = txt.shape
        q_txt = self.norm_added_q(self.add_q_proj(txt).view(B, N, self.heads, -1))
        k_txt = self.norm_added_k(self.add_k_proj(txt).view(B, N, self.heads, -1))
        return torch.cat((q, q_txt), dim=1), torch.cat((k, k_txt), dim=1), torch.cat((self.to_v(img), self.add_v_proj(txt)), dim=1)


class Modulation(nn.Module):
    def __init__(self, dim, chunks, adaln_dim, adaln_dtype=None, device=None, operations=None):
        super().__init__()
        self.chunks = chunks
        self.linear = operations.Linear(adaln_dim, chunks * dim, dtype=adaln_dtype, device=device)

    def forward(self, vec, dtype):
        return self.linear(vec).to(dtype).unsqueeze(1).chunk(self.chunks, dim=-1)


class JointBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, context_mlp_ratio, adaln_dim, adaln_dtype=None, dtype=None, device=None, operations=None):
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.norm1 = Modulation(dim, 6, adaln_dim, adaln_dtype=adaln_dtype, device=device, operations=operations)
        # the caption modulation is a learned constant, the same at every timestep
        self.scale_shift_table_context = nn.Parameter(torch.empty(6 * dim, dtype=dtype, device=device))
        # affine-free, shared by video and caption tokens
        self.norm = operations.LayerNorm(dim, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.attn = JointAttention(dim, num_heads, dtype=dtype, device=device, operations=operations)
        self.ff = FeedForward(dim, dim, mult=mlp_ratio, dtype=dtype, device=device, operations=operations)
        self.ff_context = FeedForward(dim, dim, mult=context_mlp_ratio, dtype=dtype, device=device, operations=operations)

    def forward(self, img, txt, vec, pe, transformer_options={}):
        # (shift, scale, gate) for attention, then for the feed-forward
        mod = self.norm1(vec, img.dtype)
        mod_txt = comfy.ops.cast_to_input(self.scale_shift_table_context, txt).chunk(6)
        q, k, v = self.attn.qkv(modulated_norm(img, self.norm, mod[1], mod[0]), modulated_norm(txt, self.norm, mod_txt[1], mod_txt[0]), pe)
        out = attention(q, k, v, self.attn.heads, self.comfy_attention, transformer_options)
        del q, k, v
        n = img.shape[1]
        img = gated_residual(img, self.attn.to_out[0](out[:, :n]), mod[2])
        txt = gated_residual(txt, self.attn.to_add_out(out[:, n:]), mod_txt[2])
        del out
        if comfy.model_management.in_training:
            img = gated_residual(img, self.ff(modulated_norm(img, self.norm, mod[4], mod[3])), mod[5])
        else:
            # token chunks bound the 8x wide intermediate; activations are quantized per token, so the result is unchanged
            for x in img.split(FFN_CHUNK, dim=1):
                gated_residual(x, self.ff(modulated_norm(x, self.norm, mod[4], mod[3])), mod[5])
        txt = gated_residual(txt, self.ff_context(modulated_norm(txt, self.norm, mod_txt[4], mod_txt[3])), mod_txt[5])
        return img, txt


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.linear_1 = operations.Linear(256, dim, dtype=dtype, device=device)
        self.linear_2 = operations.Linear(dim, dim, dtype=dtype, device=device)

    def forward(self, t, dtype):
        return self.linear_2(F.silu(self.linear_1(timestep_embedding(t, 256).to(dtype))))


class OutputNorm(nn.Module):
    def __init__(self, dim, adaln_dim, adaln_dtype=None, dtype=None, device=None, operations=None):
        super().__init__()
        self.linear = operations.Linear(adaln_dim, 2 * dim, dtype=adaln_dtype, device=device)
        self.norm = operations.LayerNorm(dim, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)

    def forward(self, x, vec):
        scale, shift = self.linear(vec).to(x.dtype).unsqueeze(1).chunk(2, dim=-1)
        return modulated_norm(x, self.norm, scale, shift)


class PAN2(nn.Module):
    def __init__(self, in_channels=97, out_channels=48, dim=4096, num_heads=32, num_layers=44, mlp_ratio=8.0, context_mlp_ratio=4.0,
                 text_dim=4096, num_refiner_layers=4, patch_size=(1, 2, 2), rope_axes_dim=(16, 56, 56), rope_theta=256.0, rope_temporal_scale=16 / 24,
                 adaln_curve_grid=None, adaln_curve_dim=None, image_model=None, dtype=None, device=None, operations=None):
        super().__init__()
        self.dtype = dtype
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.rope_axes_dim = rope_axes_dim
        self.rope_theta = rope_theta
        # frame positions in 16 fps units for the 24 fps data
        self.rope_temporal_scale = rope_temporal_scale
        self.use_adaln_curves = adaln_curve_grid is not None
        # curve-form checkpoints replace the time embedder and the full-width adaln weights with a small basis of the time-embedding curve silu(temb(t))
        adaln = {"adaln_dim": adaln_curve_dim, "adaln_dtype": torch.float32} if self.use_adaln_curves else {"adaln_dim": dim, "adaln_dtype": dtype}
        self.x_embedder = operations.Conv3d(in_channels, dim, kernel_size=patch_size, stride=patch_size, dtype=dtype, device=device)
        if self.use_adaln_curves:
            self.register_buffer("adaln_t_table", torch.empty(adaln_curve_grid, adaln_curve_dim, dtype=torch.float32, device=device))
        else:
            self.time_embedder = TimestepEmbedding(dim, dtype=dtype, device=device, operations=operations)
        self.context_embedder = nn.Sequential(operations.RMSNorm(text_dim, eps=1e-5, dtype=dtype, device=device), operations.Linear(text_dim, dim, dtype=dtype, device=device))
        self.context_refiner = nn.ModuleList([RefinerBlock(dim, num_heads, dtype=dtype, device=device, operations=operations) for _ in range(num_refiner_layers)])
        self.context_type_embedding = nn.Parameter(torch.empty(dim, dtype=dtype, device=device))
        self.transformer_blocks = nn.ModuleList([JointBlock(dim, num_heads, mlp_ratio, context_mlp_ratio, **adaln, dtype=dtype, device=device, operations=operations) for _ in range(num_layers)])
        self.norm_out = OutputNorm(dim, **adaln, dtype=dtype, device=device, operations=operations)
        self.proj_out = operations.Linear(dim, out_channels * patch_size[0] * patch_size[1] * patch_size[2], dtype=dtype, device=device)

    def rope(self, t, h, w, device):
        ids = torch.stack(torch.meshgrid(torch.arange(t, device=device, dtype=torch.float32) * self.rope_temporal_scale,
                                         torch.arange(h, device=device, dtype=torch.float32),
                                         torch.arange(w, device=device, dtype=torch.float32), indexing="ij"), dim=-1).reshape(1, -1, 3)
        pe = torch.cat([comfy.ldm.flux.math.rope(ids[..., i], d, self.rope_theta) for i, d in enumerate(self.rope_axes_dim)], dim=-3)
        # (1, N, 1, ...) for (B, N, H, D) queries and keys
        return pe.unsqueeze(2)

    def time_embedding(self, timestep, dtype):
        if self.use_adaln_curves:
            # curve coordinates lerped from the table, t in [0, 1] -> fractional grid row
            table = comfy.model_management.cast_to(self.adaln_t_table, device=timestep.device)
            pos = timestep.float().clamp(0.0, 1.0) * (table.shape[0] - 1)
            i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
            return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
        return F.silu(self.time_embedder(timestep, dtype))

    def forward(self, x, timestep, context, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, timestep, context, transformer_options, **kwargs)

    def _forward(self, x, timestep, context, transformer_options={}, **kwargs):
        B, _, T, H, W = x.shape
        dtype, device = x.dtype, x.device
        transformer_options = transformer_options.copy()
        x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size)
        t, h, w = (s // p for s, p in zip(x.shape[2:], self.patch_size))
        # allocated outside the memory compiler region, the blocks update both streams in place
        img = self.x_embedder(x).flatten(2).transpose(1, 2).contiguous()
        del x
        vec = self.time_embedding(timestep, dtype)
        pe = self.rope(t, h, w, device)

        txt = self.context_embedder(context)
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.context_refiner), device, transformer_options)
        for block in self.context_refiner:
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, dtype)
            txt = block(txt, transformer_options)
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)
        txt = txt + comfy.ops.cast_to_input(self.context_type_embedding, txt)

        blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
        transformer_options["total_blocks"] = len(self.transformer_blocks)
        transformer_options["block_type"] = "double"
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.transformer_blocks), device, transformer_options)
        comfy.model_prefetch.malloc_graph_begin(device)
        for i, block in enumerate(self.transformer_blocks):
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, dtype, malloc_scope="block")
            transformer_options["block_index"] = i
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(args["img"], args["txt"], args["vec"], args["pe"], args["transformer_options"])
                    return out
                out = blocks_replace[("double_block", i)]({"img": img, "txt": txt, "vec": vec, "pe": pe, "transformer_options": transformer_options}, {"original_block": block_wrap})
                img, txt = out["img"], out["txt"]
            else:
                img, txt = block(img, txt, vec, pe, transformer_options)
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None, malloc_scope="block")
        comfy.model_prefetch.malloc_graph_end()

        img = self.proj_out(self.norm_out(img, vec))
        pt, ph, pw = self.patch_size
        img = img.view(B, t, h, w, self.out_channels, pt, ph, pw).permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(B, self.out_channels, t * pt, h * ph, w * pw)
        return img[:, :, :T, :H, :W]
