import torch
from torch import nn
import math

import comfy.ldm.common_dit
from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.flux.math import apply_rope1
from comfy.ldm.flux.layers import EmbedND
from comfy.ldm.kandinsky5.utils_nabla import (
    fractal_flatten,
    fractal_unflatten,
    fast_sta_nabla,
    nabla,
)

def attention(q, k, v, heads, transformer_options={}):
    return optimized_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        heads=heads,
        skip_reshape=True,
        transformer_options=transformer_options
    )

def apply_scale_shift_norm(norm, x, scale, shift):
    return torch.addcmul(shift, norm(x), scale + 1.0)

def apply_gate_sum(x, out, gate):
    return torch.addcmul(x, gate, out)

def get_shift_scale_gate(params):
    shift, scale, gate = torch.chunk(params, 3, dim=-1)
    return tuple(x.unsqueeze(1) for x in (shift, scale, gate))

def get_freqs(dim, max_period=10000.0):
    return torch.exp(-math.log(max_period) * torch.arange(start=0, end=dim, dtype=torch.float32) / dim)


class TimeEmbeddings(nn.Module):
    def __init__(self, model_dim, time_dim, max_period=10000.0, operation_settings=None):
        super().__init__()
        assert model_dim % 2 == 0
        self.model_dim = model_dim
        self.max_period = max_period
        self.register_buffer("freqs", get_freqs(model_dim // 2, max_period), persistent=False)
        operations = operation_settings.get("operations")
        self.in_layer = operations.Linear(model_dim, time_dim, bias=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.activation = nn.SiLU()
        self.out_layer = operations.Linear(time_dim, time_dim, bias=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))

    def forward(self, timestep, dtype):
        args = torch.outer(timestep, self.freqs.to(device=timestep.device))
        time_embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(dtype)
        time_embed = self.out_layer(self.activation(self.in_layer(time_embed)))
        return time_embed


class TextEmbeddings(nn.Module):
    def __init__(self, text_dim, model_dim, operation_settings=None):
        super().__init__()
        operations = operation_settings.get("operations")
        self.in_layer = operations.Linear(text_dim, model_dim, bias=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.norm = operations.LayerNorm(model_dim, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))

    def forward(self, text_embed):
        text_embed = self.in_layer(text_embed)
        return self.norm(text_embed).type_as(text_embed)


class VisualEmbeddings(nn.Module):
    def __init__(self, visual_dim, model_dim, patch_size, operation_settings=None):
        super().__init__()
        self.patch_size = patch_size
        operations = operation_settings.get("operations")
        self.in_layer = operations.Linear(visual_dim, model_dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))

    def forward(self, x):
        x = x.movedim(1, -1)  # B C T H W -> B T H W C
        B, T, H, W, dim = x.shape
        pt, ph, pw = self.patch_size

        x = x.view(
            B,
            T // pt, pt,
            H // ph, ph,
            W // pw, pw,
            dim,
        ).permute(0, 1, 3, 5, 2, 4, 6, 7).flatten(4, 7)

        return self.in_layer(x)


class Modulation(nn.Module):
    def __init__(self, time_dim, model_dim, num_params, operation_settings=None):
        super().__init__()
        self.activation = nn.SiLU()
        self.out_layer = operation_settings.get("operations").Linear(time_dim, num_params * model_dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))

    def forward(self, x):
        return self.out_layer(self.activation(x))


class SelfAttention(nn.Module):
    def __init__(self, num_channels, head_dim, operation_settings=None):
        super().__init__()
        assert num_channels % head_dim == 0
        self.num_heads = num_channels // head_dim
        self.head_dim = head_dim

        operations = operation_settings.get("operations")
        self.to_query = operations.Linear(num_channels, num_channels, bias=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.to_key = operations.Linear(num_channels, num_channels, bias=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.to_value = operations.Linear(num_channels, num_channels, bias=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.query_norm = operations.RMSNorm(head_dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.key_norm = operations.RMSNorm(head_dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))

        self.out_layer = operations.Linear(num_channels, num_channels, bias=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.num_chunks = 2

    def _compute_qk(self, x, freqs, proj_fn, norm_fn):
        result = proj_fn(x).view(*x.shape[:-1], self.num_heads, -1)
        return apply_rope1(norm_fn(result), freqs)

    def _forward(self, x, freqs, sparse_params=None, transformer_options={}):
        q = self._compute_qk(x, freqs, self.to_query, self.query_norm)
        k = self._compute_qk(x, freqs, self.to_key, self.key_norm)
        v = self.to_value(x).view(*x.shape[:-1], self.num_heads, -1)
        if sparse_params is None:
            out = attention(q, k, v, self.num_heads, transformer_options=transformer_options)
        else:
            out = nabla(q, k, v, sparse_params)
        return self.out_layer(out)

    def _forward_chunked(self, x, freqs, sparse_params=None, transformer_options={}):
        def process_chunks(proj_fn, norm_fn):
            x_chunks = torch.chunk(x, self.num_chunks, dim=1)
            freqs_chunks = torch.chunk(freqs, self.num_chunks, dim=1)
            chunks = []
            for x_chunk, freqs_chunk in zip(x_chunks, freqs_chunks):
                chunks.append(self._compute_qk(x_chunk, freqs_chunk, proj_fn, norm_fn))
            return torch.cat(chunks, dim=1)

        q = process_chunks(self.to_query, self.query_norm)
        k = process_chunks(self.to_key, self.key_norm)
        v = self.to_value(x).view(*x.shape[:-1], self.num_heads, -1)
        if sparse_params is None:
            out = attention(q, k, v, self.num_heads, transformer_options=transformer_options)
        else:
            out = nabla(q, k, v, sparse_params)
        return self.out_layer(out)

    def forward(self, x, freqs, sparse_params=None, transformer_options={}):
        if x.shape[1] > 8192:
            return self._forward_chunked(x, freqs, sparse_params=sparse_params, transformer_options=transformer_options)
        else:
            return self._forward(x, freqs, sparse_params=sparse_params, transformer_options=transformer_options)


class CrossAttention(SelfAttention):
    def get_qkv(self, x, context):
        q = self.to_query(x).view(*x.shape[:-1], self.num_heads, -1)
        k = self.to_key(context).view(*context.shape[:-1], self.num_heads, -1)
        v = self.to_value(context).view(*context.shape[:-1], self.num_heads, -1)
        return q, k, v

    def forward(self, x, context, transformer_options={}):
        q, k, v = self.get_qkv(x, context)
        out = attention(self.query_norm(q), self.key_norm(k), v, self.num_heads, transformer_options=transformer_options)
        return self.out_layer(out)


class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim, operation_settings=None):
        super().__init__()
        operations = operation_settings.get("operations")
        self.in_layer = operations.Linear(dim, ff_dim, bias=False, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.activation = nn.GELU()
        self.out_layer = operations.Linear(ff_dim, dim, bias=False, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.num_chunks = 4

    def _forward(self, x):
        return self.out_layer(self.activation(self.in_layer(x)))

    def _forward_chunked(self, x):
        chunks = torch.chunk(x, self.num_chunks, dim=1)
        output_chunks = []
        for chunk in chunks:
            output_chunks.append(self._forward(chunk))
        return torch.cat(output_chunks, dim=1)

    def forward(self, x):
        if x.shape[1] > 8192:
            return self._forward_chunked(x)
        else:
            return self._forward(x)


class OutLayer(nn.Module):
    def __init__(self, model_dim, time_dim, visual_dim, patch_size, operation_settings=None):
        super().__init__()
        self.patch_size = patch_size
        self.modulation = Modulation(time_dim, model_dim, 2, operation_settings=operation_settings)
        operations = operation_settings.get("operations")
        self.norm = operations.LayerNorm(model_dim, elementwise_affine=False, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.out_layer = operations.Linear(model_dim, math.prod(patch_size) * visual_dim, bias=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))

    def forward(self, visual_embed, time_embed):
        B, T, H, W, _ = visual_embed.shape
        shift, scale = torch.chunk(self.modulation(time_embed), 2, dim=-1)
        scale = scale[:, None, None, None, :]
        shift = shift[:, None, None, None, :]
        visual_embed = apply_scale_shift_norm(self.norm, visual_embed, scale, shift)
        x = self.out_layer(visual_embed)

        out_dim = x.shape[-1] // (self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
        x = x.view(
            B, T, H, W,
            out_dim,
            self.patch_size[0], self.patch_size[1], self.patch_size[2]
        )
        return x.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(2, 3).flatten(3, 4).flatten(4, 5)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, model_dim, time_dim, ff_dim, head_dim, operation_settings=None):
        super().__init__()
        self.text_modulation = Modulation(time_dim, model_dim, 6, operation_settings=operation_settings)
        operations = operation_settings.get("operations")

        self.self_attention_norm = operations.LayerNorm(model_dim, elementwise_affine=False, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.self_attention = SelfAttention(model_dim, head_dim, operation_settings=operation_settings)

        self.feed_forward_norm = operations.LayerNorm(model_dim, elementwise_affine=False, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.feed_forward = FeedForward(model_dim, ff_dim, operation_settings=operation_settings)

    def forward(self, x, time_embed, freqs, transformer_options={}):
        self_attn_params, ff_params = torch.chunk(self.text_modulation(time_embed), 2, dim=-1)
        shift, scale, gate = get_shift_scale_gate(self_attn_params)
        out = apply_scale_shift_norm(self.self_attention_norm, x, scale, shift)
        out = self.self_attention(out, freqs, transformer_options=transformer_options)
        x = apply_gate_sum(x, out, gate)

        shift, scale, gate = get_shift_scale_gate(ff_params)
        out = apply_scale_shift_norm(self.feed_forward_norm, x, scale, shift)
        out = self.feed_forward(out)
        x = apply_gate_sum(x, out, gate)
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, model_dim, time_dim, ff_dim, head_dim, operation_settings=None):
        super().__init__()
        self.visual_modulation = Modulation(time_dim, model_dim, 9, operation_settings=operation_settings)

        operations = operation_settings.get("operations")
        self.self_attention_norm = operations.LayerNorm(model_dim, elementwise_affine=False, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.self_attention = SelfAttention(model_dim, head_dim, operation_settings=operation_settings)

        self.cross_attention_norm = operations.LayerNorm(model_dim, elementwise_affine=False, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.cross_attention = CrossAttention(model_dim, head_dim, operation_settings=operation_settings)

        self.feed_forward_norm = operations.LayerNorm(model_dim, elementwise_affine=False, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.feed_forward = FeedForward(model_dim, ff_dim, operation_settings=operation_settings)

    def forward(self, visual_embed, text_embed, time_embed, freqs, sparse_params=None, transformer_options={}):
        self_attn_params, cross_attn_params, ff_params = torch.chunk(self.visual_modulation(time_embed), 3, dim=-1)
        # self attention
        shift, scale, gate = get_shift_scale_gate(self_attn_params)
        visual_out = apply_scale_shift_norm(self.self_attention_norm, visual_embed, scale, shift)
        visual_out = self.self_attention(visual_out, freqs, sparse_params=sparse_params, transformer_options=transformer_options)
        visual_embed = apply_gate_sum(visual_embed, visual_out, gate)
        # cross attention
        shift, scale, gate = get_shift_scale_gate(cross_attn_params)
        visual_out = apply_scale_shift_norm(self.cross_attention_norm, visual_embed, scale, shift)
        visual_out = self.cross_attention(visual_out, text_embed, transformer_options=transformer_options)
        visual_embed = apply_gate_sum(visual_embed, visual_out, gate)
        # feed forward
        shift, scale, gate = get_shift_scale_gate(ff_params)
        visual_out = apply_scale_shift_norm(self.feed_forward_norm, visual_embed, scale, shift)
        visual_out = self.feed_forward(visual_out)
        visual_embed = apply_gate_sum(visual_embed, visual_out, gate)
        return visual_embed


class Kandinsky5(nn.Module):
    def __init__(
        self,
        in_visual_dim=16, out_visual_dim=16, in_text_dim=3584, in_text_dim2=768, time_dim=512,
        model_dim=1792, ff_dim=7168, visual_embed_dim=132, patch_size=(1, 2, 2), num_text_blocks=2, num_visual_blocks=32,
        axes_dims=(16, 24, 24), rope_scale_factor=(1.0, 2.0, 2.0),
        dtype=None, device=None, operations=None, **kwargs
    ):
        super().__init__()
        head_dim = sum(axes_dims)
        self.rope_scale_factor = rope_scale_factor
        self.in_visual_dim = in_visual_dim
        self.model_dim = model_dim
        self.patch_size = patch_size
        self.visual_embed_dim = visual_embed_dim
        self.dtype = dtype
        self.device = device
        operation_settings = {"operations": operations, "device": device, "dtype": dtype}

        self.time_embeddings = TimeEmbeddings(model_dim, time_dim, operation_settings=operation_settings)
        self.text_embeddings = TextEmbeddings(in_text_dim, model_dim, operation_settings=operation_settings)
        self.pooled_text_embeddings = TextEmbeddings(in_text_dim2, time_dim, operation_settings=operation_settings)
        self.visual_embeddings = VisualEmbeddings(visual_embed_dim, model_dim, patch_size, operation_settings=operation_settings)

        self.text_transformer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(model_dim, time_dim, ff_dim, head_dim, operation_settings=operation_settings) for _ in range(num_text_blocks)]
        )

        self.visual_transformer_blocks = nn.ModuleList(
            [TransformerDecoderBlock(model_dim, time_dim, ff_dim, head_dim, operation_settings=operation_settings) for _ in range(num_visual_blocks)]
        )

        self.out_layer = OutLayer(model_dim, time_dim, out_visual_dim, patch_size, operation_settings=operation_settings)

        self.rope_embedder_3d = EmbedND(dim=head_dim, theta=10000.0, axes_dim=axes_dims)
        self.rope_embedder_1d = EmbedND(dim=head_dim, theta=10000.0, axes_dim=[head_dim])

    def rope_encode_1d(self, seq_len, seq_start=0, steps=None, device=None, dtype=None, transformer_options={}):
        steps = seq_len if steps is None else steps
        seq_ids = torch.linspace(seq_start, seq_start + (seq_len - 1), steps=steps, device=device, dtype=dtype)
        seq_ids = seq_ids.reshape(-1, 1).unsqueeze(0)  # Shape: (1, steps, 1)
        freqs = self.rope_embedder_1d(seq_ids).movedim(1, 2)
        return freqs

    def rope_encode_3d(self, t, h, w, t_start=0, steps_t=None, steps_h=None, steps_w=None, device=None, dtype=None, transformer_options={}):

        patch_size = self.patch_size
        t_len = ((t + (patch_size[0] // 2)) // patch_size[0])
        h_len = ((h + (patch_size[1] // 2)) // patch_size[1])
        w_len = ((w + (patch_size[2] // 2)) // patch_size[2])

        if steps_t is None:
            steps_t = t_len
        if steps_h is None:
            steps_h = h_len
        if steps_w is None:
            steps_w = w_len

        h_start = 0
        w_start = 0
        rope_options = transformer_options.get("rope_options", None)
        if rope_options is not None:
            t_len = (t_len - 1.0) * rope_options.get("scale_t", 1.0) + 1.0
            h_len = (h_len - 1.0) * rope_options.get("scale_y", 1.0) + 1.0
            w_len = (w_len - 1.0) * rope_options.get("scale_x", 1.0) + 1.0

            t_start += rope_options.get("shift_t", 0.0)
            h_start += rope_options.get("shift_y", 0.0)
            w_start += rope_options.get("shift_x", 0.0)
        else:
            rope_scale_factor = self.rope_scale_factor
            if self.model_dim == 4096: # pro video model uses different rope scaling at higher resolutions
                if h * w >= 14080:
                    rope_scale_factor = (1.0, 3.16, 3.16)

            t_len = (t_len - 1.0) / rope_scale_factor[0] + 1.0
            h_len = (h_len - 1.0) / rope_scale_factor[1] + 1.0
            w_len = (w_len - 1.0) / rope_scale_factor[2] + 1.0

        img_ids = torch.zeros((steps_t, steps_h, steps_w, 3), device=device, dtype=dtype)
        img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(t_start, t_start + (t_len - 1), steps=steps_t, device=device, dtype=dtype).reshape(-1, 1, 1)
        img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(h_start, h_start + (h_len - 1), steps=steps_h, device=device, dtype=dtype).reshape(1, -1, 1)
        img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(w_start, w_start + (w_len - 1), steps=steps_w, device=device, dtype=dtype).reshape(1, 1, -1)
        img_ids = img_ids.reshape(1, -1, img_ids.shape[-1])

        freqs = self.rope_embedder_3d(img_ids).movedim(1, 2)
        return freqs

    def forward_orig(self, x, timestep, context, y, freqs, freqs_text, transformer_options={}, **kwargs):
        patches_replace = transformer_options.get("patches_replace", {})
        context = self.text_embeddings(context)
        time_embed = self.time_embeddings(timestep, x.dtype) + self.pooled_text_embeddings(y)

        for block in self.text_transformer_blocks:
            context = block(context, time_embed, freqs_text, transformer_options=transformer_options)

        visual_embed = self.visual_embeddings(x)
        visual_shape = visual_embed.shape[:-1]

        blocks_replace = patches_replace.get("dit", {})
        transformer_options["total_blocks"] = len(self.visual_transformer_blocks)
        transformer_options["block_type"] = "double"

        B, _, T, H, W = x.shape
        NABLA_THR = 31 # long (10 sec) generation
        if T > NABLA_THR:
            assert self.patch_size[0] == 1

            # pro video model uses lower P at higher resolutions
            P = 0.7 if self.model_dim == 4096 and H * W >= 14080 else 0.9

            freqs = freqs.view(freqs.shape[0], *visual_shape[1:], *freqs.shape[2:])
            visual_embed, freqs = fractal_flatten(visual_embed, freqs, visual_shape[1:])
            pt, ph, pw = self.patch_size
            T, H, W = T // pt, H // ph, W // pw

            wT, wW, wH = 11, 3, 3
            sta_mask = fast_sta_nabla(T, H // 8, W // 8, wT, wH, wW, device=x.device)

            sparse_params = dict(
                sta_mask=sta_mask.unsqueeze_(0).unsqueeze_(0),
                attention_type="nabla",
                to_fractal=True,
                P=P,
                wT=wT, wW=wW, wH=wH,
                add_sta=True,
                visual_shape=(T, H, W),
                method="topcdf",
            )
        else:
            sparse_params = None
            visual_embed = visual_embed.flatten(1, -2)

        for i, block in enumerate(self.visual_transformer_blocks):
            transformer_options["block_index"] = i
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    return block(
                        x=args["x"],
                        context=args["context"],
                        time_embed=args["time_embed"],
                        freqs=args["freqs"],
                        sparse_params=args.get("sparse_params"),
                        transformer_options=args.get("transformer_options"),
                    )
                visual_embed = blocks_replace[("double_block", i)](
                    {
                        "x": visual_embed,
                        "context": context,
                        "time_embed": time_embed,
                        "freqs": freqs,
                        "sparse_params": sparse_params,
                        "transformer_options": transformer_options,
                    },
                    {"original_block": block_wrap},
                )["x"]
            else:
                visual_embed = block(
                    visual_embed,
                    context,
                    time_embed,
                    freqs=freqs,
                    sparse_params=sparse_params,
                    transformer_options=transformer_options,
                )

        if T > NABLA_THR:
            visual_embed = fractal_unflatten(
                visual_embed,
                visual_shape[1:],
            )
        else:
            visual_embed = visual_embed.reshape(*visual_shape, -1)

        return self.out_layer(visual_embed, time_embed)

    def _forward(self, x, timestep, context, y, time_dim_replace=None, transformer_options={}, **kwargs):
        original_dims = x.ndim
        if original_dims == 4:
            x = x.unsqueeze(2)
        bs, c, t_len, h, w = x.shape
        x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size)

        if time_dim_replace is not None:
            time_dim_replace = comfy.ldm.common_dit.pad_to_patch_size(time_dim_replace, self.patch_size)
            x[:, :time_dim_replace.shape[1], :time_dim_replace.shape[2]] = time_dim_replace

        freqs = self.rope_encode_3d(t_len, h, w, device=x.device, dtype=x.dtype, transformer_options=transformer_options)
        freqs_text = self.rope_encode_1d(context.shape[1], device=x.device, dtype=x.dtype, transformer_options=transformer_options)

        out = self.forward_orig(x, timestep, context, y, freqs, freqs_text, transformer_options=transformer_options, **kwargs)
        if original_dims == 4:
            out = out.squeeze(2)
        return out

    def forward(self, x, timestep, context, y, time_dim_replace=None, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, timestep, context, y, time_dim_replace=time_dim_replace, transformer_options=transformer_options, **kwargs)
