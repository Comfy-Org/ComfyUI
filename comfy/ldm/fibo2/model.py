import math

import torch
import torch.nn.functional as F
from torch import nn

import comfy.ldm.flux.math
import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy.patcher_extension
import comfy.utils
from comfy.ldm.lumina.model import FinalLayer, JointAttention, apply_gate, clamp_fp16, modulate, pos_ids_x
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention
from comfy.ldm.modules.diffusionmodules.mmdit import TimestepEmbedder


def residual(x, y, gate=None):
    # in place inside the memory compiler region; out of place keeps x for backward when training
    if comfy.model_management.in_training:
        return x + y if gate is None else torch.addcmul(x, y, gate)
    return x.add_(y) if gate is None else x.addcmul_(y, gate)


def gist_count(text_len, max_gist):
    # 256 gist tokens up to 256 text tokens, rising linearly to max_gist at 2048, in steps of 64
    ratio = min(1.0, max(0.0, (text_len - 256) / (2048 - 256)))
    return max(256, min(max_gist, math.ceil((256 + ratio * (max_gist - 256)) / 64) * 64))


class Attention(nn.Module):
    # no RoPE: the cross-attention to the text, and the Perceiver's self-attention
    def __init__(self, dim, n_heads, kv_dim, norm_eps, dtype=None, device=None, operations=None):
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.n_heads = n_heads
        self.to_q = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.to_kv = operations.Linear(kv_dim, 2 * dim, bias=False, dtype=dtype, device=device)
        self.to_out = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.norm_q = operations.RMSNorm(dim // n_heads, eps=norm_eps, dtype=dtype, device=device)
        self.norm_k = operations.RMSNorm(dim // n_heads, eps=norm_eps, dtype=dtype, device=device)

    def forward(self, x, context, transformer_options={}):
        q = self.norm_q(self.to_q(x).unflatten(-1, (self.n_heads, -1)))
        k, v = self.to_kv(context).unflatten(-1, (2, self.n_heads, -1)).unbind(2)
        k = self.norm_k(k)
        q, k, v = (AttentionTensorContainer(t.movedim(1, 2)) for t in (q, k, v))
        return self.to_out(optimized_attention(q, k, v, self.n_heads, skip_reshape=True, transformer_options=transformer_options, preferred_attention=self.comfy_attention))


class FeedForward(nn.Module):
    # SwiGLU; fused files carry w1 and w3 as one w13, the hidden width zero padded to a multiple of 256 so int8 weights quantize swiglu(x) inside w2's GEMM
    def __init__(self, dim, hidden_dim, fused, dtype=None, device=None, operations=None):
        super().__init__()
        self.fused = fused
        if fused:
            self.w13 = operations.Linear(dim, 2 * hidden_dim, bias=False, dtype=dtype, device=device)
        else:
            self.w1 = operations.Linear(dim, hidden_dim, bias=False, dtype=dtype, device=device)
            self.w3 = operations.Linear(dim, hidden_dim, bias=False, dtype=dtype, device=device)
        self.w2 = operations.Linear(hidden_dim, dim, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        if self.fused:
            return comfy.ops.linear_input_act(self.w2, self.w13(x), "swiglu")
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    # Z-Image's block with in place residuals; an injection block also cross-attends to its own text bundle
    def __init__(self, dim, n_heads, ffn_dim, fused_ffn, norm_eps, cap_feat_dim=None, operation_settings={}):
        super().__init__()
        operations, device, dtype = operation_settings["operations"], operation_settings["device"], operation_settings["dtype"]
        self.attention = JointAttention(dim, n_heads, None, True, operation_settings=operation_settings)
        self.attention.q_norm.eps = self.attention.k_norm.eps = norm_eps
        self.feed_forward = FeedForward(dim, ffn_dim, fused_ffn, dtype=dtype, device=device, operations=operations)
        self.attention_norm1 = operations.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
        self.ffn_norm1 = operations.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
        self.attention_norm2 = operations.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
        self.ffn_norm2 = operations.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
        self.adaLN_modulation = nn.Sequential(operations.Linear(256, 4 * dim, bias=True, dtype=dtype, device=device))
        self.cross_attention = None
        if cap_feat_dim is not None:
            self.cross_attn_norm1 = operations.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
            self.cross_attention = Attention(dim, n_heads, cap_feat_dim, norm_eps, dtype=dtype, device=device, operations=operations)
            self.cross_attn_norm2 = operations.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
            self.cross_attn_gate = nn.Parameter(torch.empty(1, dtype=dtype, device=device))

    def forward(self, x, freqs_cis, adaln_input, timestep_zero_index=None, context=None, transformer_options={}):
        scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(4, dim=1)
        y = self.attention(modulate(self.attention_norm1(x), scale_msa, timestep_zero_index=timestep_zero_index), None, freqs_cis, transformer_options=transformer_options)
        x = residual(x, apply_gate(gate_msa.unsqueeze(1).tanh(), self.attention_norm2(clamp_fp16(y)), timestep_zero_index=timestep_zero_index))
        if self.cross_attention is not None:
            y = self.cross_attention(self.cross_attn_norm1(x), context, transformer_options)
            x = residual(x, self.cross_attn_norm2(y), comfy.ops.cast_to_input(self.cross_attn_gate, y).tanh())
        y = self.feed_forward(modulate(self.ffn_norm1(x), scale_mlp, timestep_zero_index=timestep_zero_index))
        return residual(x, apply_gate(gate_mlp.unsqueeze(1).tanh(), self.ffn_norm2(clamp_fp16(y)), timestep_zero_index=timestep_zero_index))


class PerceiverLayer(nn.Module):
    def __init__(self, dim, n_heads, ffn_dim, fused_ffn, cap_feat_dim, norm_eps, operation_settings={}):
        super().__init__()
        operations, device, dtype = operation_settings["operations"], operation_settings["device"], operation_settings["dtype"]
        self.cross_attn_norm = operations.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
        self.cross_attention = Attention(dim, n_heads, cap_feat_dim, norm_eps, dtype=dtype, device=device, operations=operations)
        self.self_attn_norm = operations.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
        self.self_attention = Attention(dim, n_heads, dim, norm_eps, dtype=dtype, device=device, operations=operations)
        self.ffn_norm = operations.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
        self.ffn = FeedForward(dim, ffn_dim, fused_ffn, dtype=dtype, device=device, operations=operations)

    def forward(self, x, context, transformer_options={}):
        x = x + self.cross_attention(self.cross_attn_norm(x), context, transformer_options)
        y = self.self_attn_norm(x)
        x = x + self.self_attention(y, y, transformer_options)
        return x + self.ffn(self.ffn_norm(x))


class Perceiver(nn.Module):
    # squeezes the text encoder's hidden states into gist tokens
    def __init__(self, dim, n_heads, ffn_dim, fused_ffn, cap_feat_dim, num_queries, num_layers, norm_eps, operation_settings={}):
        super().__init__()
        operations, device, dtype = operation_settings["operations"], operation_settings["device"], operation_settings["dtype"]
        self.latent_queries = nn.Parameter(torch.empty(num_queries, dim, dtype=dtype, device=device))
        self.layers = nn.ModuleList([PerceiverLayer(dim, n_heads, ffn_dim, fused_ffn, cap_feat_dim, norm_eps, operation_settings=operation_settings) for _ in range(num_layers)])
        self.norm_out = operations.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)

    def forward(self, context, num_gist, transformer_options={}):
        x = comfy.ops.cast_to_input(self.latent_queries[:num_gist], context).unsqueeze(0).expand(context.shape[0], -1, -1)
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.layers), context.device, transformer_options)
        for layer in self.layers:
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, context.device, layer, context.dtype)
            x = layer(x, context, transformer_options)
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, context.device, None)
        return self.norm_out(x)


class Fibo2(nn.Module):
    def __init__(self, in_channels=128, dim=3584, n_heads=28, n_layers=44, n_refiner_layers=2, injection_layers=(6, 14, 22, 30, 38), cap_feat_dim=7680,
                 perceiver_layers=2, max_gist_tokens=512, ffn_dim=9728, fused_ffn=True, norm_eps=1e-5, image_model=None, dtype=None, device=None, operations=None):
        super().__init__()
        self.dtype = dtype
        operation_settings = {"operations": operations, "device": device, "dtype": dtype}
        self.max_gist_tokens = max_gist_tokens
        self.injection_layers = injection_layers

        self.x_embedder = operations.Linear(in_channels, dim, bias=True, dtype=dtype, device=device)
        self.t_embedder = TimestepEmbedder(1024, output_size=256, **operation_settings)
        self.noise_refiner = nn.ModuleList([Block(dim, n_heads, ffn_dim, fused_ffn, norm_eps, operation_settings=operation_settings) for _ in range(n_refiner_layers)])
        self.perceiver = Perceiver(dim, n_heads, ffn_dim, fused_ffn, cap_feat_dim, max_gist_tokens, perceiver_layers, norm_eps, operation_settings=operation_settings)
        self.layers = nn.ModuleList([Block(dim, n_heads, ffn_dim, fused_ffn, norm_eps, cap_feat_dim if i in injection_layers else None, operation_settings=operation_settings) for i in range(n_layers)])
        self.final_layer = FinalLayer(dim, 1, in_channels, z_image_modulation=True, operation_settings=operation_settings)
        self.rope_axes_dim = [32, 48, 48]

    def forward(self, x, timestep, context, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, timestep, context, transformer_options, **kwargs)

    def _forward(self, x, timestep, context, transformer_options={}, ref_latents=[], **kwargs):
        B, C, H, W = x.shape
        dtype, device = x.dtype, x.device
        transformer_options = transformer_options.copy()

        # the image and the reference tokens are noisy and clean: the second half of the adaln batch modulates the image
        adaln_mixed = self.t_embedder(torch.cat((torch.zeros_like(timestep), timestep * 1000)), dtype)
        adaln_noisy = adaln_mixed[B:]

        # gist tokens on rope planes 1..num_gist, the image on plane max_gist_tokens + 1 whatever the gist count, reference k on the k-th plane after it
        num_gist = gist_count(context.shape[1], self.max_gist_tokens)
        plane = self.max_gist_tokens + 1
        # NTK-aware axis scales: the positions keep their spacing, zero collapses the axis
        rope_options = transformer_options.get("rope_options", {})
        scales = (1.0, rope_options.get("scale_y", 1.0), rope_options.get("scale_x", 1.0))
        pos_options = {"rope_options": {**rope_options, "scale_y": float(scales[1] != 0), "scale_x": float(scales[2] != 0)}}
        tokens = [x.flatten(2).transpose(1, 2)]
        ids = [torch.zeros(1, num_gist, 3, dtype=torch.float32, device=device), pos_ids_x(plane, H, W, 1, device, transformer_options=pos_options)]
        ids[0][0, :, 0] = torch.arange(1, num_gist + 1, dtype=torch.float32, device=device)
        for k, ref in enumerate(ref_latents):
            tokens.append(comfy.utils.repeat_to_batch_size(ref, B).flatten(2).transpose(1, 2))
            ids.append(pos_ids_x(plane + 1 + k, ref.shape[-2], ref.shape[-1], 1, device, transformer_options=pos_options))
        img = self.x_embedder(torch.cat(tokens, dim=1))
        ids = torch.cat(ids, dim=1)
        freqs_cis = torch.cat([comfy.ldm.flux.math.rope(ids[..., i], d, 256.0 * (s or 1.0) ** (-d / (d - 2))) for i, (d, s) in enumerate(zip(self.rope_axes_dim, scales))], dim=-3).unsqueeze(2)
        del tokens

        n_img = H * W
        refiner_adaln, refiner_index = (adaln_mixed, [(0, n_img)]) if len(ref_latents) > 0 else (adaln_noisy, None)
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.noise_refiner), device, transformer_options)
        for block in self.noise_refiner:
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, dtype)
            img = block(img, freqs_cis[:, num_gist:], refiner_adaln, refiner_index, transformer_options=transformer_options)
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)

        # the Perceiver reads the first text bundle, each injection block its own
        bundles = context.chunk(1 + len(self.injection_layers), dim=-1)
        gist = self.perceiver(bundles[0], num_gist, transformer_options)
        text = dict(zip(self.injection_layers, bundles[1:]))

        # allocated outside the memory compiler region, the blocks update it in place
        x = torch.cat((gist, img), dim=1)
        del gist, img
        image_index = [(num_gist, num_gist + n_img)]

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
                    return {"img": block(args["img"], args["pe"], args["vec"], image_index, text.get(i), args["transformer_options"])}
                x = blocks_replace[("single_block", i)]({"img": x, "vec": adaln_mixed, "pe": freqs_cis, "transformer_options": transformer_options}, {"original_block": block_wrap})["img"]
            else:
                x = block(x, freqs_cis, adaln_mixed, image_index, text.get(i), transformer_options)
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None, malloc_scope="block")
        comfy.model_prefetch.malloc_graph_end()

        out = self.final_layer(x[:, num_gist:num_gist + n_img], adaln_noisy)
        return out.transpose(1, 2).reshape(B, C, H, W)
