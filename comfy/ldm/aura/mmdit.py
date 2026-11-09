#AuraFlow MMDiT
#Originally written by the AuraFlow Authors

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention
import comfy.ops
import comfy.patcher_extension
import comfy.ldm.common_dit

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim=None, dtype=None, device=None, operations=None) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim

        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = find_multiple(n_hidden, 256)

        self.c_fc1 = operations.Linear(dim, n_hidden, bias=False, dtype=dtype, device=device)
        self.c_fc2 = operations.Linear(dim, n_hidden, bias=False, dtype=dtype, device=device)
        self.c_proj = operations.Linear(n_hidden, dim, bias=False, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


class MultiHeadLayerNorm(nn.Module):
    def __init__(self, hidden_size=None, eps=1e-5, dtype=None, device=None):
        # Copy pasta from https://github.com/huggingface/transformers/blob/e5f71ecaae50ea476d1e12351003790273c4b2ed/src/transformers/models/cohere/modeling_cohere.py#L78

        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size, dtype=dtype, device=device))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        mean = hidden_states.mean(-1, keepdim=True)
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        hidden_states = (hidden_states - mean) * torch.rsqrt(
            variance + self.variance_epsilon
        )
        hidden_states = self.weight.to(torch.float32) * hidden_states
        return hidden_states.to(input_dtype)

class SingleAttention(nn.Module):
    def __init__(self, dim, n_heads, mh_qknorm=False, dtype=None, device=None, operations=None):
        super().__init__()
        self.comfy_attention = ComfyAttention()

        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # this is for cond
        self.w1q = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w1k = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w1v = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w1o = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)

        self.q_norm1 = (
            MultiHeadLayerNorm((self.n_heads, self.head_dim), dtype=dtype, device=device)
            if mh_qknorm
            else operations.LayerNorm(self.head_dim, elementwise_affine=False, dtype=dtype, device=device)
        )
        self.k_norm1 = (
            MultiHeadLayerNorm((self.n_heads, self.head_dim), dtype=dtype, device=device)
            if mh_qknorm
            else operations.LayerNorm(self.head_dim, elementwise_affine=False, dtype=dtype, device=device)
        )

    #@torch.compile()
    def forward(self, c, transformer_options={}):

        bsz, seqlen1, _ = c.shape

        q, k, v = self.w1q(c), self.w1k(c), self.w1v(c)
        q = q.view(bsz, seqlen1, self.n_heads, self.head_dim)
        k = k.view(bsz, seqlen1, self.n_heads, self.head_dim)
        v = v.view(bsz, seqlen1, self.n_heads, self.head_dim)
        q, k = self.q_norm1(q), self.k_norm1(k)

        q, k, v = AttentionTensorContainer(q.permute(0, 2, 1, 3)), AttentionTensorContainer(k.permute(0, 2, 1, 3)), AttentionTensorContainer(v.permute(0, 2, 1, 3))
        output = optimized_attention(q, k, v, self.n_heads, skip_reshape=True, preferred_attention=self.comfy_attention, transformer_options=transformer_options)
        c = self.w1o(output)
        return c



class DoubleAttention(nn.Module):
    def __init__(self, dim, n_heads, mh_qknorm=False, dtype=None, device=None, operations=None):
        super().__init__()
        self.comfy_attention = ComfyAttention()

        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # this is for cond
        self.w1q = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w1k = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w1v = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w1o = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)

        # this is for x
        self.w2q = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w2k = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w2v = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.w2o = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)

        self.q_norm1 = (
            MultiHeadLayerNorm((self.n_heads, self.head_dim), dtype=dtype, device=device)
            if mh_qknorm
            else operations.LayerNorm(self.head_dim, elementwise_affine=False, dtype=dtype, device=device)
        )
        self.k_norm1 = (
            MultiHeadLayerNorm((self.n_heads, self.head_dim), dtype=dtype, device=device)
            if mh_qknorm
            else operations.LayerNorm(self.head_dim, elementwise_affine=False, dtype=dtype, device=device)
        )

        self.q_norm2 = (
            MultiHeadLayerNorm((self.n_heads, self.head_dim), dtype=dtype, device=device)
            if mh_qknorm
            else operations.LayerNorm(self.head_dim, elementwise_affine=False, dtype=dtype, device=device)
        )
        self.k_norm2 = (
            MultiHeadLayerNorm((self.n_heads, self.head_dim), dtype=dtype, device=device)
            if mh_qknorm
            else operations.LayerNorm(self.head_dim, elementwise_affine=False, dtype=dtype, device=device)
        )


    #@torch.compile()
    def forward(self, c, x, transformer_options={}):

        bsz, seqlen1, _ = c.shape
        bsz, seqlen2, _ = x.shape

        cq, ck, cv = self.w1q(c), self.w1k(c), self.w1v(c)
        cq = cq.view(bsz, seqlen1, self.n_heads, self.head_dim)
        ck = ck.view(bsz, seqlen1, self.n_heads, self.head_dim)
        cv = cv.view(bsz, seqlen1, self.n_heads, self.head_dim)
        cq, ck = self.q_norm1(cq), self.k_norm1(ck)

        xq, xk, xv = self.w2q(x), self.w2k(x), self.w2v(x)
        xq = xq.view(bsz, seqlen2, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen2, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen2, self.n_heads, self.head_dim)
        xq, xk = self.q_norm2(xq), self.k_norm2(xk)

        # concat all
        q, k, v = (
            torch.cat([cq, xq], dim=1),
            torch.cat([ck, xk], dim=1),
            torch.cat([cv, xv], dim=1),
        )

        del cq, ck, cv, xq, xk, xv
        q, k, v = AttentionTensorContainer(q.permute(0, 2, 1, 3)), AttentionTensorContainer(k.permute(0, 2, 1, 3)), AttentionTensorContainer(v.permute(0, 2, 1, 3))
        output = optimized_attention(q, k, v, self.n_heads, skip_reshape=True, preferred_attention=self.comfy_attention, transformer_options=transformer_options)

        c, x = output.split([seqlen1, seqlen2], dim=1)
        c = self.w1o(c)
        x = self.w2o(x)

        return c, x


class MMDiTBlock(nn.Module):
    def __init__(self, dim, heads=8, global_conddim=1024, is_last=False, dtype=None, device=None, operations=None):
        super().__init__()

        self.normC1 = operations.LayerNorm(dim, elementwise_affine=False, dtype=dtype, device=device)
        self.normC2 = operations.LayerNorm(dim, elementwise_affine=False, dtype=dtype, device=device)
        if not is_last:
            self.mlpC = MLP(dim, hidden_dim=dim * 4, dtype=dtype, device=device, operations=operations)
            self.modC = nn.Sequential(
                nn.SiLU(),
                operations.Linear(global_conddim, 6 * dim, bias=False, dtype=dtype, device=device),
            )
        else:
            self.modC = nn.Sequential(
                nn.SiLU(),
                operations.Linear(global_conddim, 2 * dim, bias=False, dtype=dtype, device=device),
            )

        self.normX1 = operations.LayerNorm(dim, elementwise_affine=False, dtype=dtype, device=device)
        self.normX2 = operations.LayerNorm(dim, elementwise_affine=False, dtype=dtype, device=device)
        self.mlpX = MLP(dim, hidden_dim=dim * 4, dtype=dtype, device=device, operations=operations)
        self.modX = nn.Sequential(
            nn.SiLU(),
            operations.Linear(global_conddim, 6 * dim, bias=False, dtype=dtype, device=device),
        )

        self.attn = DoubleAttention(dim, heads, dtype=dtype, device=device, operations=operations)
        self.is_last = is_last

    #@torch.compile()
    def forward(self, c, x, global_cond, transformer_options={}, **kwargs):

        cres, xres = c, x

        cshift_msa, cscale_msa, cgate_msa, cshift_mlp, cscale_mlp, cgate_mlp = (
            self.modC(global_cond).chunk(6, dim=1)
        )

        c = modulate(self.normC1(c), cshift_msa, cscale_msa)

        # xpath
        xshift_msa, xscale_msa, xgate_msa, xshift_mlp, xscale_mlp, xgate_mlp = (
            self.modX(global_cond).chunk(6, dim=1)
        )

        x = modulate(self.normX1(x), xshift_msa, xscale_msa)

        # attention
        c, x = self.attn(c, x, transformer_options=transformer_options)


        c = self.normC2(cres + cgate_msa.unsqueeze(1) * c)
        c = cgate_mlp.unsqueeze(1) * self.mlpC(modulate(c, cshift_mlp, cscale_mlp))
        c = cres + c

        x = self.normX2(xres + xgate_msa.unsqueeze(1) * x)
        x = xgate_mlp.unsqueeze(1) * self.mlpX(modulate(x, xshift_mlp, xscale_mlp))
        x = xres + x

        return c, x

class DiTBlock(nn.Module):
    # like MMDiTBlock, but it only has X
    def __init__(self, dim, heads=8, global_conddim=1024, dtype=None, device=None, operations=None):
        super().__init__()

        self.norm1 = operations.LayerNorm(dim, elementwise_affine=False, dtype=dtype, device=device)
        self.norm2 = operations.LayerNorm(dim, elementwise_affine=False, dtype=dtype, device=device)

        self.modCX = nn.Sequential(
            nn.SiLU(),
            operations.Linear(global_conddim, 6 * dim, bias=False, dtype=dtype, device=device),
        )

        self.attn = SingleAttention(dim, heads, dtype=dtype, device=device, operations=operations)
        self.mlp = MLP(dim, hidden_dim=dim * 4, dtype=dtype, device=device, operations=operations)

    #@torch.compile()
    def forward(self, cx, global_cond, transformer_options={}, **kwargs):
        cxres = cx
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.modCX(
            global_cond
        ).chunk(6, dim=1)
        cx = modulate(self.norm1(cx), shift_msa, scale_msa)
        cx = self.attn(cx, transformer_options=transformer_options)
        cx = self.norm2(cxres + gate_msa.unsqueeze(1) * cx)
        mlpout = self.mlp(modulate(cx, shift_mlp, scale_mlp))
        cx = gate_mlp.unsqueeze(1) * mlpout

        cx = cxres + cx

        return cx



class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=None, device=None, operations=None):
        super().__init__()
        self.mlp = nn.Sequential(
            operations.Linear(frequency_embedding_size, hidden_size, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Linear(hidden_size, hidden_size, dtype=dtype, device=device),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = 1000 * torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        ).to(t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    #@torch.compile()
    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class MMDiT(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        patch_size=2,
        dim=3072,
        n_layers=36,
        n_double_layers=4,
        n_heads=12,
        global_conddim=3072,
        cond_seq_dim=2048,
        max_seq=32 * 32,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        self.dtype = dtype

        self.t_embedder = TimestepEmbedder(global_conddim, dtype=dtype, device=device, operations=operations)

        self.cond_seq_linear = operations.Linear(
            cond_seq_dim, dim, bias=False, dtype=dtype, device=device
        )  # linear for something like text sequence.
        self.init_x_linear = operations.Linear(
            patch_size * patch_size * in_channels, dim, dtype=dtype, device=device
        )  # init linear for patchified image.

        self.positional_encoding = nn.Parameter(torch.empty(1, max_seq, dim, dtype=dtype, device=device))
        self.register_tokens = nn.Parameter(torch.empty(1, 8, dim, dtype=dtype, device=device))

        self.double_layers = nn.ModuleList([])
        self.single_layers = nn.ModuleList([])


        for idx in range(n_double_layers):
            self.double_layers.append(
                MMDiTBlock(dim, n_heads, global_conddim, is_last=(idx == n_layers - 1), dtype=dtype, device=device, operations=operations)
            )

        for idx in range(n_double_layers, n_layers):
            self.single_layers.append(
                DiTBlock(dim, n_heads, global_conddim, dtype=dtype, device=device, operations=operations)
            )


        self.final_linear = operations.Linear(
            dim, patch_size * patch_size * out_channels, bias=False, dtype=dtype, device=device
        )

        self.modF = nn.Sequential(
            nn.SiLU(),
            operations.Linear(global_conddim, 2 * dim, bias=False, dtype=dtype, device=device),
        )

        self.out_channels = out_channels
        self.patch_size = patch_size
        self.n_double_layers = n_double_layers
        self.n_layers = n_layers

        self.h_max = round(max_seq**0.5)
        self.w_max = round(max_seq**0.5)

    @torch.no_grad()
    def extend_pe(self, init_dim=(16, 16), target_dim=(64, 64)):
        # extend pe
        pe_data = self.positional_encoding.data.squeeze(0)[: init_dim[0] * init_dim[1]]

        pe_as_2d = pe_data.view(init_dim[0], init_dim[1], -1).permute(2, 0, 1)

        # now we need to extend this to target_dim. for this we will use interpolation.
        # we will use torch.nn.functional.interpolate
        pe_as_2d = F.interpolate(
            pe_as_2d.unsqueeze(0), size=target_dim, mode="bilinear"
        )
        pe_new = pe_as_2d.squeeze(0).permute(1, 2, 0).flatten(0, 1)
        self.positional_encoding.data = pe_new.unsqueeze(0).contiguous()
        self.h_max, self.w_max = target_dim

    def pe_selection_index_based_on_dim(self, h, w):
        h_p, w_p = h // self.patch_size, w // self.patch_size
        original_pe_indexes = torch.arange(self.positional_encoding.shape[1])
        original_pe_indexes = original_pe_indexes.view(self.h_max, self.w_max)
        starth =  self.h_max // 2 - h_p // 2
        endh =starth + h_p
        startw = self.w_max // 2 - w_p // 2
        endw = startw + w_p
        original_pe_indexes = original_pe_indexes[
            starth:endh, startw:endw
        ]
        return original_pe_indexes.flatten()

    def unpatchify(self, x, h, w):
        c = self.out_channels
        p = self.patch_size

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def patchify(self, x):
        B, C, H, W = x.size()
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))
        x = x.view(
            B,
            C,
            (H + 1) // self.patch_size,
            self.patch_size,
            (W + 1) // self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)
        return x

    def apply_pos_embeds(self, x, h, w):
        h = (h + 1) // self.patch_size
        w = (w + 1) // self.patch_size
        max_dim = max(h, w)

        cur_dim = self.h_max
        pos_encoding = comfy.ops.cast_to_input(self.positional_encoding.reshape(1, cur_dim, cur_dim, -1), x)

        if max_dim > cur_dim:
            pos_encoding = F.interpolate(pos_encoding.movedim(-1, 1), (max_dim, max_dim), mode="bilinear").movedim(1, -1)
            cur_dim = max_dim

        from_h = (cur_dim - h) // 2
        from_w = (cur_dim - w) // 2
        pos_encoding = pos_encoding[:,from_h:from_h+h,from_w:from_w+w]
        return x + pos_encoding.reshape(1, -1, self.positional_encoding.shape[-1])

    def forward(self, x, timestep, context, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, timestep, context, transformer_options, **kwargs)

    def _forward(self, x, timestep, context, transformer_options={}, **kwargs):
        patches_replace = transformer_options.get("patches_replace", {})
        # patchify x, add PE
        b, c, h, w = x.shape

        # pe_indexes = self.pe_selection_index_based_on_dim(h, w)
        # print(pe_indexes, pe_indexes.shape)

        x = self.init_x_linear(self.patchify(x))  # B, T_x, D
        x = self.apply_pos_embeds(x, h, w)
        # x = x + self.positional_encoding[:, : x.size(1)].to(device=x.device, dtype=x.dtype)
        # x = x + self.positional_encoding[:, pe_indexes].to(device=x.device, dtype=x.dtype)

        # process conditions for MMDiT Blocks
        c_seq = context  # B, T_c, D_c
        t = timestep

        c = self.cond_seq_linear(c_seq)  # B, T_c, D
        c = torch.cat([comfy.ops.cast_to_input(self.register_tokens, c).repeat(c.size(0), 1, 1), c], dim=1)

        global_cond = self.t_embedder(t, x.dtype)  # B, D

        blocks_replace = patches_replace.get("dit", {})
        if len(self.double_layers) > 0:
            for i, layer in enumerate(self.double_layers):
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["txt"], out["img"] = layer(args["txt"],
                                                       args["img"],
                                                       args["vec"],
                                                       transformer_options=args["transformer_options"])
                        return out
                    out = blocks_replace[("double_block", i)]({"img": x, "txt": c, "vec": global_cond, "transformer_options": transformer_options}, {"original_block": block_wrap})
                    c = out["txt"]
                    x = out["img"]
                else:
                    c, x = layer(c, x, global_cond, transformer_options=transformer_options, **kwargs)

        if len(self.single_layers) > 0:
            c_len = c.size(1)
            cx = torch.cat([c, x], dim=1)
            for i, layer in enumerate(self.single_layers):
                if ("single_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"] = layer(args["img"], args["vec"], transformer_options=args["transformer_options"])
                        return out

                    out = blocks_replace[("single_block", i)]({"img": cx, "vec": global_cond, "transformer_options": transformer_options}, {"original_block": block_wrap})
                    cx = out["img"]
                else:
                    cx = layer(cx, global_cond, transformer_options=transformer_options, **kwargs)

            x = cx[:, c_len:]

        fshift, fscale = self.modF(global_cond).chunk(2, dim=1)

        x = modulate(x, fshift, fscale)
        x = self.final_linear(x)
        x = self.unpatchify(x, (h + 1) // self.patch_size, (w + 1) // self.patch_size)[:,:,:h,:w]
        return x
