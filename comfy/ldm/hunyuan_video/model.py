#Based on Flux code because of weird hunyuan video code license.

import torch
import comfy.patcher_extension
import comfy.ldm.flux.layers
import comfy.ldm.modules.diffusionmodules.mmdit
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention

from dataclasses import dataclass
from einops import repeat

from torch import Tensor, nn

from comfy.ldm.flux.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding
)

import comfy.ldm.common_dit


@dataclass
class HunyuanVideoParams:
    in_channels: int
    out_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list
    theta: int
    patch_size: list
    qkv_bias: bool
    guidance_embed: bool
    byt5: bool
    meanflow: bool
    use_cond_type_embedding: bool
    vision_in_dim: int
    meanflow_sum: bool


class SelfAttentionRef(nn.Module):
    def __init__(self, dim: int, qkv_bias: bool = False, dtype=None, device=None, operations=None):
        super().__init__()
        self.qkv = operations.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        self.proj = operations.Linear(dim, dim, dtype=dtype, device=device)


class TokenRefinerBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        heads,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.heads = heads
        mlp_hidden_dim = hidden_size * 4

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            operations.Linear(hidden_size, 2 * hidden_size, bias=True, dtype=dtype, device=device),
        )

        self.norm1 = operations.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
        self.self_attn = SelfAttentionRef(hidden_size, True, dtype=dtype, device=device, operations=operations)

        self.norm2 = operations.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)

        self.mlp = nn.Sequential(
            operations.Linear(hidden_size, mlp_hidden_dim, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Linear(mlp_hidden_dim, hidden_size, bias=True, dtype=dtype, device=device),
        )

    def forward(self, x, c, mask, transformer_options={}):
        mod1, mod2 = self.adaLN_modulation(c).chunk(2, dim=1)

        norm_x = self.norm1(x)
        qkv = self.self_attn.qkv(norm_x)
        q, k, v = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v)
        del qkv
        attn = optimized_attention(q, k, v, self.heads, mask=mask, skip_reshape=True, preferred_attention=self.comfy_attention, transformer_options=transformer_options)

        x = x + self.self_attn.proj(attn) * mod1.unsqueeze(1)
        x = x + self.mlp(self.norm2(x)) * mod2.unsqueeze(1)
        return x


class IndividualTokenRefiner(nn.Module):
    def __init__(
        self,
        hidden_size,
        heads,
        num_blocks,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TokenRefinerBlock(
                    hidden_size=hidden_size,
                    heads=heads,
                    dtype=dtype,
                    device=device,
                    operations=operations
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x, c, mask, transformer_options={}):
        m = None
        if mask is not None:
            m = mask.view(mask.shape[0], 1, 1, mask.shape[1]).repeat(1, 1, mask.shape[1], 1)
            m = m + m.transpose(2, 3)

        for block in self.blocks:
            x = block(x, c, m, transformer_options=transformer_options)
        return x



class TokenRefiner(nn.Module):
    def __init__(
        self,
        text_dim,
        hidden_size,
        heads,
        num_blocks,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()

        self.input_embedder = operations.Linear(text_dim, hidden_size, bias=True, dtype=dtype, device=device)
        self.t_embedder = MLPEmbedder(256, hidden_size, dtype=dtype, device=device, operations=operations)
        self.c_embedder = MLPEmbedder(text_dim, hidden_size, dtype=dtype, device=device, operations=operations)
        self.individual_token_refiner = IndividualTokenRefiner(hidden_size, heads, num_blocks, dtype=dtype, device=device, operations=operations)

    def forward(
        self,
        x,
        timesteps,
        mask,
        transformer_options={},
    ):
        t = self.t_embedder(timestep_embedding(timesteps, 256, time_factor=1.0).to(x.dtype))
        # m = mask.float().unsqueeze(-1)
        # c = (x.float() * m).sum(dim=1) / m.sum(dim=1) #TODO: the following works when the x.shape is the same length as the tokens but might break otherwise
        if x.dtype == torch.float16:
            c = x.float().sum(dim=1) / x.shape[1]
        else:
            c = x.sum(dim=1) / x.shape[1]

        c = t + self.c_embedder(c.to(x.dtype))
        x = self.input_embedder(x)
        x = self.individual_token_refiner(x, c, mask, transformer_options=transformer_options)
        return x


class ByT5Mapper(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, out_dim1, use_res=False, dtype=None, device=None, operations=None):
        super().__init__()
        self.layernorm = operations.LayerNorm(in_dim, dtype=dtype, device=device)
        self.fc1 = operations.Linear(in_dim, hidden_dim, dtype=dtype, device=device)
        self.fc2 = operations.Linear(hidden_dim, out_dim, dtype=dtype, device=device)
        self.fc3 = operations.Linear(out_dim, out_dim1, dtype=dtype, device=device)
        self.use_res = use_res
        self.act_fn = nn.GELU()

    def forward(self, x):
        if self.use_res:
            res = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        x2 = self.act_fn(x)
        x2 = self.fc3(x2)
        if self.use_res:
            x2 = x2 + res
        return x2

class HunyuanVideo(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, image_model=None, final_layer=True, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        operation_settings = {"operations": operations, "device": device, "dtype": dtype}

        params = HunyuanVideoParams(**kwargs)
        self.params = params
        self.patch_size = params.patch_size
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        self.use_cond_type_embedding = params.use_cond_type_embedding
        self.vision_in_dim = params.vision_in_dim
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)

        self.img_in = comfy.ldm.modules.diffusionmodules.mmdit.PatchEmbed(None, self.patch_size, self.in_channels, self.hidden_size, conv3d=len(self.patch_size) == 3, dtype=dtype, device=device, operations=operations)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, dtype=dtype, device=device, operations=operations)
        if params.vec_in_dim is not None:
            self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size, dtype=dtype, device=device, operations=operations)
        else:
            self.vector_in = None

        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, dtype=dtype, device=device, operations=operations) if params.guidance_embed else nn.Identity()
        )

        self.txt_in = TokenRefiner(params.context_in_dim, self.hidden_size, self.num_heads, 2, dtype=dtype, device=device, operations=operations)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                    dtype=dtype, device=device, operations=operations
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio, dtype=dtype, device=device, operations=operations)
                for _ in range(params.depth_single_blocks)
            ]
        )

        if params.byt5:
            self.byt5_in = ByT5Mapper(
                in_dim=1472,
                out_dim=2048,
                hidden_dim=2048,
                out_dim1=self.hidden_size,
                use_res=False,
                dtype=dtype, device=device, operations=operations
            )
        else:
            self.byt5_in = None

        if params.meanflow:
            self.time_r_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, dtype=dtype, device=device, operations=operations)
        else:
            self.time_r_in = None

        if final_layer:
            self.final_layer = LastLayer(self.hidden_size, self.patch_size[-1], self.out_channels, dtype=dtype, device=device, operations=operations)

        # HunyuanVideo 1.5 specific modules
        if self.vision_in_dim is not None:
            from comfy.ldm.wan.model import MLPProj
            self.vision_in = MLPProj(in_dim=self.vision_in_dim, out_dim=self.hidden_size, operation_settings=operation_settings)
        else:
            self.vision_in = None
        if self.use_cond_type_embedding:
            # 0: text_encoder feature 1: byt5 feature 2: vision_encoder feature
            self.cond_type_embedding = nn.Embedding(3, self.hidden_size)
        else:
            self.cond_type_embedding = None

    def forward_orig(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        txt_mask: Tensor,
        timesteps: Tensor,
        y: Tensor = None,
        txt_byt5=None,
        clip_fea=None,
        guidance: Tensor = None,
        guiding_frame_index=None,
        ref_latent=None,
        disable_time_r=False,
        control=None,
        transformer_options={},
    ) -> Tensor:
        transformer_options = transformer_options.copy()
        patches_replace = transformer_options.get("patches_replace", {})

        initial_shape = list(img.shape)
        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256, time_factor=1.0).to(img.dtype))

        if (self.time_r_in is not None) and (not disable_time_r):
            w = torch.where(transformer_options['sigmas'][0] == transformer_options['sample_sigmas'])[0]  # This most likely could be improved
            if len(w) > 0:
                timesteps_r = transformer_options['sample_sigmas'][w[0] + 1]
                timesteps_r = timesteps_r.unsqueeze(0).to(device=timesteps.device, dtype=timesteps.dtype)
                vec_r = self.time_r_in(timestep_embedding(timesteps_r, 256, time_factor=1000.0).to(img.dtype))
                vec = (vec + vec_r) if self.params.meanflow_sum else (vec + vec_r) / 2

        if ref_latent is not None:
            ref_latent_ids = self.img_ids(ref_latent)
            ref_latent = self.img_in(ref_latent)
            img = torch.cat([ref_latent, img], dim=-2)
            ref_latent_ids[..., 0] = -1
            ref_latent_ids[..., 2] += (initial_shape[-1] // self.patch_size[-1])
            img_ids = torch.cat([ref_latent_ids, img_ids], dim=-2)

        if guiding_frame_index is not None:
            token_replace_vec = self.time_in(timestep_embedding(guiding_frame_index, 256, time_factor=1.0))
            if self.vector_in is not None:
                vec_ = self.vector_in(y[:, :self.params.vec_in_dim])
                vec = torch.cat([(vec_ + token_replace_vec).unsqueeze(1), (vec_ + vec).unsqueeze(1)], dim=1)
            else:
                vec = torch.cat([(token_replace_vec).unsqueeze(1), (vec).unsqueeze(1)], dim=1)
            frame_tokens = (initial_shape[-1] // self.patch_size[-1]) * (initial_shape[-2] // self.patch_size[-2])
            modulation_dims = [(0, frame_tokens, 0), (frame_tokens, None, 1)]
            modulation_dims_txt = [(0, None, 1)]
        else:
            if self.vector_in is not None:
                vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])
            modulation_dims = None
            modulation_dims_txt = None

        if self.params.guidance_embed:
            if guidance is not None:
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        if txt_mask is not None and not torch.is_floating_point(txt_mask):
            txt_mask = (txt_mask - 1).to(img.dtype) * torch.finfo(img.dtype).max

        txt = self.txt_in(txt, timesteps, txt_mask, transformer_options=transformer_options)

        if self.cond_type_embedding is not None:
            self.cond_type_embedding.to(txt.device)
            cond_emb = self.cond_type_embedding(torch.zeros_like(txt[:, :, 0], device=txt.device, dtype=torch.long))
            txt = txt + cond_emb.to(txt.dtype)

        if self.byt5_in is not None and txt_byt5 is not None:
            txt_byt5 = self.byt5_in(txt_byt5)
            if self.cond_type_embedding is not None:
                cond_emb = self.cond_type_embedding(torch.ones_like(txt_byt5[:, :, 0], device=txt_byt5.device, dtype=torch.long))
                txt_byt5 = txt_byt5 + cond_emb.to(txt_byt5.dtype)
                txt = torch.cat((txt_byt5, txt), dim=1) # byt5 first for HunyuanVideo1.5
            else:
                txt = torch.cat((txt, txt_byt5), dim=1)
            txt_byt5_ids = torch.zeros((txt_ids.shape[0], txt_byt5.shape[1], txt_ids.shape[-1]), device=txt_ids.device, dtype=txt_ids.dtype)
            txt_ids = torch.cat((txt_ids, txt_byt5_ids), dim=1)

        if clip_fea is not None:
            txt_vision_states = self.vision_in(clip_fea)
            if self.cond_type_embedding is not None:
                cond_emb = self.cond_type_embedding(2 * torch.ones_like(txt_vision_states[:, :, 0], dtype=torch.long, device=txt_vision_states.device))
                txt_vision_states = txt_vision_states + cond_emb
            txt = torch.cat((txt_vision_states.to(txt.dtype), txt), dim=1)
            extra_txt_ids = torch.zeros((txt_ids.shape[0], txt_vision_states.shape[1], txt_ids.shape[-1]), device=txt_ids.device, dtype=txt_ids.dtype)
            txt_ids = torch.cat((txt_ids, extra_txt_ids), dim=1)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        img_len = img.shape[1]
        if txt_mask is not None:
            attn_mask_len = img_len + txt.shape[1]
            attn_mask = torch.zeros((1, 1, attn_mask_len), dtype=img.dtype, device=img.device)
            attn_mask[:, 0, :txt.shape[1]] = txt_mask
        else:
            attn_mask = None

        blocks_replace = patches_replace.get("dit", {})
        transformer_options["total_blocks"] = len(self.double_blocks)
        transformer_options["block_type"] = "double"
        for i, block in enumerate(self.double_blocks):
            transformer_options["block_index"] = i
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"], attn_mask=args["attention_mask"], modulation_dims_img=args["modulation_dims_img"], modulation_dims_txt=args["modulation_dims_txt"], transformer_options=args["transformer_options"])
                    return out

                out = blocks_replace[("double_block", i)]({"img": img, "txt": txt, "vec": vec, "pe": pe, "attention_mask": attn_mask, 'modulation_dims_img': modulation_dims, 'modulation_dims_txt': modulation_dims_txt, 'transformer_options': transformer_options}, {"original_block": block_wrap})
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, attn_mask=attn_mask, modulation_dims_img=modulation_dims, modulation_dims_txt=modulation_dims_txt, transformer_options=transformer_options)

            if control is not None: # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img += add

        img = torch.cat((txt, img), 1)

        transformer_options["total_blocks"] = len(self.single_blocks)
        transformer_options["block_type"] = "single"
        transformer_options["img_slice"] = [txt.shape[1], img.shape[1]]
        for i, block in enumerate(self.single_blocks):
            transformer_options["block_index"] = i
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"], attn_mask=args["attention_mask"], modulation_dims=args["modulation_dims"], transformer_options=args["transformer_options"])
                    return out

                out = blocks_replace[("single_block", i)]({"img": img, "vec": vec, "pe": pe, "attention_mask": attn_mask, 'modulation_dims': modulation_dims, 'transformer_options': transformer_options}, {"original_block": block_wrap})
                img = out["img"]
            else:
                img = block(img, vec=vec, pe=pe, attn_mask=attn_mask, modulation_dims=modulation_dims, transformer_options=transformer_options)

            if control is not None: # Controlnet
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, txt.shape[1]: img_len + txt.shape[1]] += add

        img = img[:, txt.shape[1]: img_len + txt.shape[1]]
        if ref_latent is not None:
            img = img[:, ref_latent.shape[1]:]

        img = self.final_layer(img, vec, modulation_dims=modulation_dims)  # (N, T, patch_size ** 2 * out_channels)

        shape = initial_shape[-len(self.patch_size):]
        for i in range(len(shape)):
            shape[i] = shape[i] // self.patch_size[i]
        img = img.reshape([img.shape[0]] + shape + [self.out_channels] + self.patch_size)
        if img.ndim == 8:
            img = img.permute(0, 4, 1, 5, 2, 6, 3, 7)
            img = img.reshape(initial_shape[0], self.out_channels, initial_shape[2], initial_shape[3], initial_shape[4])
        else:
            img = img.permute(0, 3, 1, 4, 2, 5)
            img = img.reshape(initial_shape[0], self.out_channels, initial_shape[2], initial_shape[3])
        return img

    def img_ids(self, x):
        bs, c, t, h, w = x.shape
        patch_size = self.patch_size
        t_len = ((t + (patch_size[0] // 2)) // patch_size[0])
        h_len = ((h + (patch_size[1] // 2)) // patch_size[1])
        w_len = ((w + (patch_size[2] // 2)) // patch_size[2])
        img_ids = torch.zeros((t_len, h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(0, t_len - 1, steps=t_len, device=x.device, dtype=x.dtype).reshape(-1, 1, 1)
        img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).reshape(1, -1, 1)
        img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).reshape(1, 1, -1)
        return repeat(img_ids, "t h w c -> b (t h w) c", b=bs)

    def img_ids_2d(self, x):
        bs, c, h, w = x.shape
        patch_size = self.patch_size
        h_len = ((h + (patch_size[0] // 2)) // patch_size[0])
        w_len = ((w + (patch_size[1] // 2)) // patch_size[1])
        img_ids = torch.zeros((h_len, w_len, 2), device=x.device, dtype=x.dtype)
        img_ids[:, :, 0] = img_ids[:, :, 0] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).unsqueeze(1)
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).unsqueeze(0)
        return repeat(img_ids, "h w c -> b (h w) c", b=bs)

    def forward(self, x, timestep, context, y=None, txt_byt5=None, clip_fea=None, guidance=None, attention_mask=None, guiding_frame_index=None, ref_latent=None, disable_time_r=False, control=None, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, timestep, context, y, txt_byt5, clip_fea, guidance, attention_mask, guiding_frame_index, ref_latent, disable_time_r, control, transformer_options, **kwargs)

    def _forward(self, x, timestep, context, y=None, txt_byt5=None, clip_fea=None, guidance=None, attention_mask=None, guiding_frame_index=None, ref_latent=None, disable_time_r=False, control=None, transformer_options={}, **kwargs):
        bs = x.shape[0]
        if len(self.patch_size) == 3:
            img_ids = self.img_ids(x)
            txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        else:
            img_ids = self.img_ids_2d(x)
            txt_ids = torch.zeros((bs, context.shape[1], 2), device=x.device, dtype=x.dtype)
        out = self.forward_orig(x, img_ids, context, txt_ids, attention_mask, timestep, y, txt_byt5, clip_fea, guidance, guiding_frame_index, ref_latent, disable_time_r=disable_time_r, control=control, transformer_options=transformer_options)
        return out
