import torch
from torch import nn

import comfy.ldm.flux.math
import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy.patcher_extension
import comfy.quant_ops
import comfy.utils
from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention


def situ(hidden, gate):
    # soft-capped gated activation
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def modulated_rms_norm(x, norm, scale, shift):
    if comfy.model_management.in_training:
        return norm(x) * (1 + scale) + shift
    # the norm weight is folded into the scale of the fused rmsnorm(x) * (1 + scale) + shift
    weight, _, stream = comfy.ops.cast_bias_weight(norm, x, offloadable=True)
    out = comfy.quant_ops.ck.rms_adaln(x, weight * (1 + scale) - 1, shift, norm.eps)
    comfy.ops.uncast_bias_weight(norm, weight, None, stream)
    return out


def gated_residual(x, y, gate):
    # in place inside the memory compiler region; out of place keeps x for backward when training
    if comfy.model_management.in_training:
        return torch.addcmul(x, y, gate)
    return x.addcmul_(y, gate)


def modulate(x, norm, mod, i, ref):
    # mod[i] is the shift and mod[i + 1] the scale; reference tokens from ref[0] on take the t=0 modulation ref[1]
    if ref is None:
        return modulated_rms_norm(x, norm, mod[i + 1], mod[i])
    s, ref_mod = ref[0], ref[1]
    return torch.cat((modulated_rms_norm(x[:, :s], norm, mod[i + 1], mod[i]), modulated_rms_norm(x[:, s:], norm, ref_mod[i + 1], ref_mod[i])), dim=1)


def residual(x, y, mod, i, ref):
    if ref is None:
        return gated_residual(x, y, mod[i])
    s, ref_mod = ref[0], ref[1]
    if comfy.model_management.in_training:
        return torch.cat((gated_residual(x[:, :s], y[:, :s], mod[i]), gated_residual(x[:, s:], y[:, s:], ref_mod[i])), dim=1)
    gated_residual(x[:, :s], y[:, :s], mod[i])
    gated_residual(x[:, s:], y[:, s:], ref_mod[i])
    return x


def rope_split_half(x, pe):
    # differentiable split-half RoPE for training, the kitchen ops have no autograd
    x_ = x.to(pe.dtype).unflatten(-1, (2, -1)).movedim(-2, -1).unsqueeze(-2)
    return (pe[..., 0] * x_[..., 0] + pe[..., 1] * x_[..., 1]).movedim(-1, -2).flatten(-2).type_as(x)


def attention(q, k, v, q_norm, k_norm, pe, heads, comfy_attention, transformer_options, ref_kv=None, kv_out=None):
    B, N, _ = q.shape
    q = q.view(B, N, heads, -1)
    k = k.view(B, N, heads, -1)
    if comfy.model_management.in_training:
        q, k = rope_split_half(q_norm(q), pe), rope_split_half(k_norm(k), pe)
    else:
        q_scale, _, q_stream = comfy.ops.cast_bias_weight(q_norm, q, offloadable=True)
        k_scale, _, k_stream = comfy.ops.cast_bias_weight(k_norm, k, offloadable=True)
        q, k = comfy.quant_ops.ck.rms_rope_split_half(q, k, pe, q_scale, k_scale, q_norm.eps)
        comfy.ops.uncast_bias_weight(q_norm, q_scale, None, q_stream)
        comfy.ops.uncast_bias_weight(k_norm, k_scale, None, k_stream)
    q, k = q.flatten(2), k.flatten(2)
    if kv_out is not None:
        kv_out[:, 0].copy_(k)
        kv_out[:, 1].copy_(v)
    if ref_kv is not None:
        # the references' keys and values, computed once
        k, v = torch.cat((k, ref_kv[:, 0]), dim=1), torch.cat((v, ref_kv[:, 1]), dim=1)
    q, k, v = AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v)
    return optimized_attention(q, k, v, heads, transformer_options=transformer_options, preferred_attention=comfy_attention)


class TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, dtype=None, device=None, operations=None):
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.num_heads = num_heads
        self.norm1 = operations.RMSNorm(dim, eps=1e-6, dtype=dtype, device=device)
        self.norm2 = operations.RMSNorm(dim, eps=1e-6, dtype=dtype, device=device)
        self.q_norm = operations.RMSNorm(dim // num_heads, eps=1e-6, dtype=dtype, device=device)
        self.k_norm = operations.RMSNorm(dim // num_heads, eps=1e-6, dtype=dtype, device=device)
        self.qkv = operations.Linear(dim, 3 * dim, dtype=dtype, device=device)
        self.proj = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.ffn_gate = operations.Linear(dim, ffn_dim, dtype=dtype, device=device)
        self.ffn_up = operations.Linear(dim, ffn_dim, dtype=dtype, device=device)
        self.ffn_down = operations.Linear(ffn_dim, dim, dtype=dtype, device=device)

    def forward(self, x, pe, transformer_options={}):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        x = x + self.proj(attention(q, k, v, self.q_norm, self.k_norm, pe, self.num_heads, self.comfy_attention, transformer_options))
        y = self.norm2(x)
        return x + self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y)))


class TextEmbedder(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, num_heads, dtype=None, device=None, operations=None):
        super().__init__()
        self.proj = operations.Linear(text_dim, dim, dtype=dtype, device=device)
        self.blocks = nn.ModuleList([TextBlock(dim, ffn_dim, num_heads, dtype=dtype, device=device, operations=operations) for _ in range(2)])

    def forward(self, x, pe, transformer_options={}):
        x = self.proj(x)
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.blocks), x.device, transformer_options)
        for block in self.blocks:
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, x.device, block, x.dtype)
            x = block(x, pe, transformer_options)
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, x.device, None)
        return x


class Block(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, fused_qkv=False, dtype=None, device=None, operations=None):
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.num_heads = num_heads
        self.fused_qkv = fused_qkv
        self.norm1 = operations.RMSNorm(dim, eps=1e-6, dtype=dtype, device=device)
        self.norm2 = operations.RMSNorm(dim, eps=1e-6, dtype=dtype, device=device)
        self.q_norm = operations.RMSNorm(dim // num_heads, eps=1e-6, dtype=dtype, device=device)
        self.k_norm = operations.RMSNorm(dim // num_heads, eps=1e-6, dtype=dtype, device=device)
        # repacked files fuse q/k/v: one GEMM and one activation quantization instead of three
        names = ("qkv", "o") if fused_qkv else ("q", "k", "v", "o")
        self.proj = nn.ModuleDict({name: operations.Linear(dim, 3 * dim if name == "qkv" else dim, bias=False, dtype=dtype, device=device) for name in names})
        self.proj["ffn_up_gate"] = operations.Linear(dim, 2 * ffn_dim, bias=False, dtype=dtype, device=device)
        self.proj["ffn_down"] = operations.Linear(ffn_dim, dim, bias=False, dtype=dtype, device=device)

    def forward(self, x, mod, pe, transformer_options={}, ref=None, ref_kv=None, kv_out=None):
        # ref: (first reference token, their t=0 modulation) when the references are in the sequence
        y = modulate(x, self.norm1, mod, 0, ref)
        if self.fused_qkv:
            q, k, v = self.proj["qkv"](y).chunk(3, dim=-1)
        else:
            q, k, v = self.proj["q"](y), self.proj["k"](y), self.proj["v"](y)
        out = attention(q, k, v, self.q_norm, self.k_norm, pe, self.num_heads, self.comfy_attention, transformer_options, ref_kv, kv_out)
        x = residual(x, self.proj["o"](out), mod, 2, ref)
        y = modulate(x, self.norm2, mod, 3, ref)
        # situ on the [hidden | gate] halves; int8 weights fold it into the activation quantization
        return residual(x, comfy.ops.linear_input_act(self.proj["ffn_down"], self.proj["ffn_up_gate"](y), "situ"), mod, 5, ref)


class MOSSImage(nn.Module):
    def __init__(self, dim=3072, ffn_dim=8192, num_heads=24, num_layers=42, text_dim=4096, latent_channels=128, fused_qkv=False, image_model=None, dtype=None, device=None, operations=None):
        super().__init__()
        self.dtype = dtype
        self.latent_channels = latent_channels
        head_dim = dim // num_heads
        self.rope_axes_dim = [head_dim // 4] * 4
        self.video_embedder = operations.Linear(latent_channels, dim, dtype=dtype, device=device)
        self.text_embedder = TextEmbedder(text_dim, dim, ffn_dim, num_heads, dtype=dtype, device=device, operations=operations)
        self.time_embedding = nn.Sequential(operations.Linear(256, dim, dtype=dtype, device=device), nn.SiLU(), operations.Linear(dim, dim, dtype=dtype, device=device))
        self.shared_block_modulation = operations.Linear(dim, 6 * dim, dtype=dtype, device=device)
        self.layers = nn.ModuleList([Block(dim, ffn_dim, num_heads, fused_qkv=fused_qkv, dtype=dtype, device=device, operations=operations) for _ in range(num_layers)])
        self.final_norm_video = operations.RMSNorm(dim, eps=1e-6, dtype=dtype, device=device)
        self.final_linear_video = operations.Linear(dim, latent_channels, bias=False, dtype=dtype, device=device)
        self.video_head_modulation = operations.Linear(dim, 2 * dim, dtype=dtype, device=device)

    def rope(self, h, w, text_len, device, transformer_options={}, ref_shapes=(), f_offset=0):
        # (r, f, h, w): the image sits after the text on f, its rows and columns centered; reference i on r = i + 1, f = 0
        rope_options = transformer_options.get("rope_options", {})
        # NTK-aware axis scales, zero collapses the axis
        scales = (1.0, 1.0, rope_options.get("scale_y", 1.0), rope_options.get("scale_x", 1.0))
        n = h * w + text_len
        ids = torch.zeros(1, n + sum(rh * rw for rh, rw in ref_shapes), 4, device=device, dtype=torch.float32)
        ids[0, :h * w, 1] = text_len + f_offset + rope_options.get("shift_t", 0.0)
        ids[0, :h * w, 2] = ((torch.arange(h, device=device, dtype=torch.float32) - (h - 1) / 2) * float(scales[2] != 0) + rope_options.get("shift_y", 0.0)).repeat_interleave(w)
        ids[0, :h * w, 3] = ((torch.arange(w, device=device, dtype=torch.float32) - (w - 1) / 2) * float(scales[3] != 0) + rope_options.get("shift_x", 0.0)).repeat(h)
        ids[0, h * w:n, 1] = torch.arange(text_len, device=device, dtype=torch.float32) + f_offset
        start = n
        for i, (rh, rw) in enumerate(ref_shapes):
            ref = ids[0, n:n + rh * rw]
            ref[:, 0] = i + 1
            ref[:, 2] = (torch.arange(rh, device=device, dtype=torch.float32) - (rh - 1) / 2).repeat_interleave(rw)
            ref[:, 3] = (torch.arange(rw, device=device, dtype=torch.float32) - (rw - 1) / 2).repeat(rh)
            n += rh * rw
        pe = torch.cat([comfy.ldm.flux.math.rope(ids[..., i], d, 10000 * (s or 1.0) ** (-d / (d - 2))) for i, (d, s) in enumerate(zip(self.rope_axes_dim, scales))], dim=-3)
        if len(ref_shapes) > 0:
            # the references keep the trained frequencies, Scale ROPE stretches only the generated image
            pe[:, start:] = torch.cat([comfy.ldm.flux.math.rope(ids[:, start:, i], d, 10000) for i, d in enumerate(self.rope_axes_dim)], dim=-3)
        return pe.unsqueeze(1).transpose(1, 2).contiguous()

    def preprocess_refs(self, refs, transformer_options={}):
        # references that attend only to each other at t = 0 depend on themselves alone: every block's keys and values for them
        batch = max(ref.shape[0] for ref in refs)
        x = torch.cat([self.video_embedder(comfy.utils.repeat_to_batch_size(ref, batch).flatten(2).transpose(1, 2)) for ref in refs], dim=1)
        pe = self.rope(0, 0, 0, x.device, transformer_options, [ref.shape[-2:] for ref in refs])
        vec = self.time_embedding(timestep_embedding(torch.zeros(1, device=x.device), 256).to(x.dtype))
        mod = self.shared_block_modulation(vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
        kv = x.new_empty((x.shape[0], len(self.layers), 2) + x.shape[1:])
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.layers), x.device, transformer_options)
        for i, block in enumerate(self.layers):
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, x.device, block, x.dtype)
            x = block(x, mod, pe, transformer_options, kv_out=kv[:, i])
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, x.device, None)
        return kv

    def forward(self, x, timestep, context, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, timestep, context, transformer_options, **kwargs)

    def _forward(self, x, timestep, context, transformer_options={}, ref_latents=[], ref_latents_method=None, ref_kv=None, **kwargs):
        B, _, H, W = x.shape
        dtype, device = x.dtype, x.device
        transformer_options = transformer_options.copy()
        text_len = context.shape[1]
        if len(ref_latents) > 0 and ref_latents_method not in ("index", "index_timestep_zero"):
            # every step when extra_conds could not precompute them (training, hooks)
            ref_kv, ref_latents = self.preprocess_refs(ref_latents, transformer_options), []
        if ref_kv is not None:
            # one set of references serves the whole batch
            ref_kv = ref_kv.expand(B, -1, -1, -1, -1) if ref_kv.shape[0] == 1 else comfy.utils.repeat_to_batch_size(ref_kv, B)
        refs = [comfy.utils.repeat_to_batch_size(ref, B) for ref in ref_latents]
        # ai-toolkit puts the references on the image's f, Moss-Image-Edit at the text start; shifting the rest keeps them at f = 0, where their keys were computed
        f_offset = -text_len if (len(refs) > 0 or ref_kv is not None) and ref_latents_method != "index_timestep_zero_text_start" else 0
        pe = self.rope(H, W, text_len, device, transformer_options, [ref.shape[-2:] for ref in refs], f_offset)

        vec = self.time_embedding(timestep_embedding(timestep, 256).to(dtype))
        mod = self.shared_block_modulation(vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
        head_shift, head_scale = self.video_head_modulation(vec).unflatten(-1, (2, -1)).unsqueeze(2).unbind(1)
        ref = None
        if len(refs) > 0 and ref_latents_method != "index":
            # ai-toolkit without kv_cache: the references ride along at t = 0 and attend to everything ("index": at the image's timestep)
            ref_vec = self.time_embedding(timestep_embedding(torch.zeros_like(timestep), 256).to(dtype))
            ref = (H * W + text_len, self.shared_block_modulation(ref_vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1))

        txt = self.text_embedder(context, pe[:, H * W:H * W + text_len], transformer_options)
        # allocated outside the memory compiler region, the blocks update it in place
        x = torch.cat([self.video_embedder(x.flatten(2).transpose(1, 2)), txt] + [self.video_embedder(r.flatten(2).transpose(1, 2)) for r in refs], dim=1)
        del txt, refs

        blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
        transformer_options["total_blocks"] = len(self.layers)
        transformer_options["block_type"] = "single"
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.layers), device, transformer_options)
        comfy.model_prefetch.malloc_graph_begin(device)
        for i, block in enumerate(self.layers):
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, dtype, malloc_scope="block")
            transformer_options["block_index"] = i
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    return {"img": block(args["img"], args["mod"], args["pe"], args["transformer_options"], ref, None if ref_kv is None else ref_kv[:, i])}
                x = blocks_replace[("single_block", i)]({"img": x, "vec": vec, "mod": mod, "pe": pe, "transformer_options": transformer_options}, {"original_block": block_wrap})["img"]
            else:
                x = block(x, mod, pe, transformer_options, ref, None if ref_kv is None else ref_kv[:, i])
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None, malloc_scope="block")
        comfy.model_prefetch.malloc_graph_end()

        out = self.final_linear_video(modulated_rms_norm(x[:, :H * W], self.final_norm_video, head_scale, head_shift))
        return out.transpose(1, 2).reshape(B, self.latent_channels, H, W)
