# original version: https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
from einops import rearrange

from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.flux.layers import EmbedND
from comfy.ldm.flux.math import apply_rope1, rope
import comfy.ldm.common_dit
import comfy.model_management
import comfy.patcher_extension


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float32)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 kv_dim=None,
                 operation_settings={}):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        if kv_dim is None:
            kv_dim = dim

        # layers
        self.q = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.k = operation_settings.get("operations").Linear(kv_dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.v = operation_settings.get("operations").Linear(kv_dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.o = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.norm_q = operation_settings.get("operations").RMSNorm(dim, eps=eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")) if qk_norm else nn.Identity()
        self.norm_k = operation_settings.get("operations").RMSNorm(dim, eps=eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")) if qk_norm else nn.Identity()

    def forward(self, x, freqs, transformer_options={}):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        patches = transformer_options.get("patches", {})

        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        def qkv_fn_q(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            return apply_rope1(q, freqs)

        def qkv_fn_k(x):
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            return apply_rope1(k, freqs)

        #These two are VRAM hogs, so we want to do all of q computation and
        #have pytorch garbage collect the intermediates on the sub function
        #return before we touch k
        q = qkv_fn_q(x)
        k = qkv_fn_k(x)

        x = optimized_attention(
            q.view(b, s, n * d),
            k.view(b, s, n * d),
            self.v(x).view(b, s, n * d),
            heads=self.num_heads,
            transformer_options=transformer_options,
        )

        if "attn1_patch" in patches:
            for p in patches["attn1_patch"]:
                x = p({"x": x, "q": q, "k": k, "transformer_options": transformer_options})

        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, transformer_options={}, **kwargs):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
        """
        # compute query, key, value
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        v = self.v(context)

        # compute attention
        x = optimized_attention(q, k, v, heads=self.num_heads, transformer_options=transformer_options)

        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6, operation_settings={}):
        super().__init__(dim, num_heads, window_size, qk_norm, eps, operation_settings=operation_settings)

        self.k_img = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.v_img = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = operation_settings.get("operations").RMSNorm(dim, eps=eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")) if qk_norm else nn.Identity()

    def forward(self, x, context, context_img_len, transformer_options={}):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
        """
        context_img = context[:, :context_img_len]
        context = context[:, context_img_len:]

        # compute query, key, value
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        v = self.v(context)
        k_img = self.norm_k_img(self.k_img(context_img))
        v_img = self.v_img(context_img)
        img_x = optimized_attention(q, k_img, v_img, heads=self.num_heads, transformer_options=transformer_options)
        # compute attention
        x = optimized_attention(q, k, v, heads=self.num_heads, transformer_options=transformer_options)

        # output
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


def repeat_e(e, x):
    repeats = 1
    if e.size(1) > 1:
        repeats = x.size(1) // e.size(1)
    if repeats == 1:
        return e
    if repeats * e.size(1) == x.size(1):
        return torch.repeat_interleave(e, repeats, dim=1)
    else:
        return torch.repeat_interleave(e, repeats + 1, dim=1)[:, :x.size(1)]


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6, operation_settings={}):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = operation_settings.get("operations").LayerNorm(dim, eps, elementwise_affine=False, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps, operation_settings=operation_settings)
        self.norm3 = operation_settings.get("operations").LayerNorm(
            dim, eps,
            elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps, operation_settings=operation_settings)
        self.norm2 = operation_settings.get("operations").LayerNorm(dim, eps, elementwise_affine=False, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.ffn = nn.Sequential(
            operation_settings.get("operations").Linear(dim, ffn_dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")), nn.GELU(approximate='tanh'),
            operation_settings.get("operations").Linear(ffn_dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))

        # modulation
        self.modulation = nn.Parameter(torch.empty(1, 6, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))

    def forward(
        self,
        x,
        e,
        freqs,
        context,
        context_img_len=257,
        transformer_options={},
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # assert e.dtype == torch.float32

        patches = transformer_options.get("patches", {})

        if e.ndim < 4:
            e = (comfy.model_management.cast_to(self.modulation, dtype=x.dtype, device=x.device) + e).chunk(6, dim=1)
        else:
            e = (comfy.model_management.cast_to(self.modulation, dtype=x.dtype, device=x.device).unsqueeze(0) + e).unbind(2)
        # assert e[0].dtype == torch.float32

        # self-attention
        x = x.contiguous() # otherwise implicit in LayerNorm
        y = self.self_attn(
            torch.addcmul(repeat_e(e[0], x), self.norm1(x), 1 + repeat_e(e[1], x)),
            freqs, transformer_options=transformer_options)

        x = torch.addcmul(x, y, repeat_e(e[2], x))
        del y

        # cross-attention & ffn
        x = x + self.cross_attn(self.norm3(x), context, context_img_len=context_img_len, transformer_options=transformer_options)

        if "attn2_patch" in patches:
            for p in patches["attn2_patch"]:
                x = p({"x": x, "transformer_options": transformer_options})

        y = self.ffn(torch.addcmul(repeat_e(e[3], x), self.norm2(x), 1 + repeat_e(e[4], x)))
        x = torch.addcmul(x, y, repeat_e(e[5], x))
        return x


class VaceWanAttentionBlock(WanAttentionBlock):
    def __init__(
            self,
            cross_attn_type,
            dim,
            ffn_dim,
            num_heads,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=False,
            eps=1e-6,
            block_id=0,
            operation_settings={}
    ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps, operation_settings=operation_settings)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = operation_settings.get("operations").Linear(self.dim, self.dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.after_proj = operation_settings.get("operations").Linear(self.dim, self.dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))

    def forward(self, c, x, **kwargs):
        if self.block_id == 0:
            c = self.before_proj(c) + x
        c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)
        return c_skip, c


class WanCamAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, num_residual_blocks=1, operation_settings={}):
        super(WanCamAdapter, self).__init__()

        # Pixel Unshuffle: reduce spatial dimensions by a factor of 8
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=8)

        # Convolution: reduce spatial dimensions by a factor
        #  of 2 (without overlap)
        self.conv = operation_settings.get("operations").Conv2d(in_dim * 64, out_dim, kernel_size=kernel_size, stride=stride, padding=0, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))

        # Residual blocks for feature extraction
        self.residual_blocks = nn.Sequential(
            *[WanCamResidualBlock(out_dim, operation_settings = operation_settings) for _ in range(num_residual_blocks)]
        )

    def forward(self, x):
        # Reshape to merge the frame dimension into batch
        bs, c, f, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(bs * f, c, h, w)

        # Pixel Unshuffle operation
        x_unshuffled = self.pixel_unshuffle(x)

        # Convolution operation
        x_conv = self.conv(x_unshuffled)

        # Feature extraction with residual blocks
        out = self.residual_blocks(x_conv)

        # Reshape to restore original bf dimension
        out = out.view(bs, f, out.size(1), out.size(2), out.size(3))

        # Permute dimensions to reorder (if needed), e.g., swap channels and feature frames
        out = out.permute(0, 2, 1, 3, 4)

        return out


class WanCamResidualBlock(nn.Module):
    def __init__(self, dim, operation_settings={}):
        super(WanCamResidualBlock, self).__init__()
        self.conv1 = operation_settings.get("operations").Conv2d(dim, dim, kernel_size=3, padding=1, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = operation_settings.get("operations").Conv2d(dim, dim, kernel_size=3, padding=1, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6, operation_settings={}):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = operation_settings.get("operations").LayerNorm(dim, eps, elementwise_affine=False, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.head = operation_settings.get("operations").Linear(dim, out_dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))

        # modulation
        self.modulation = nn.Parameter(torch.empty(1, 2, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        # assert e.dtype == torch.float32
        if e.ndim < 3:
            e = (comfy.model_management.cast_to(self.modulation, dtype=x.dtype, device=x.device) + e.unsqueeze(1)).chunk(2, dim=1)
        else:
            e = (comfy.model_management.cast_to(self.modulation, dtype=x.dtype, device=x.device).unsqueeze(0) + e.unsqueeze(2)).unbind(2)

        x = (self.head(torch.addcmul(repeat_e(e[0], x), self.norm(x), 1 + repeat_e(e[1], x))))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim, flf_pos_embed_token_number=None, operation_settings={}):
        super().__init__()

        self.proj = torch.nn.Sequential(
            operation_settings.get("operations").LayerNorm(in_dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")), operation_settings.get("operations").Linear(in_dim, in_dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")),
            torch.nn.GELU(), operation_settings.get("operations").Linear(in_dim, out_dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")),
            operation_settings.get("operations").LayerNorm(out_dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))

        if flf_pos_embed_token_number is not None:
            self.emb_pos = nn.Parameter(torch.empty((1, flf_pos_embed_token_number, in_dim), device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))
        else:
            self.emb_pos = None

    def forward(self, image_embeds):
        if self.emb_pos is not None:
            image_embeds = image_embeds[:, :self.emb_pos.shape[1]] + comfy.model_management.cast_to(self.emb_pos[:, :image_embeds.shape[1]], dtype=image_embeds.dtype, device=image_embeds.device)

        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(torch.nn.Module):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 flf_pos_embed_token_number=None,
                 in_dim_ref_conv=None,
                 wan_attn_block_class=WanAttentionBlock,
                 image_model=None,
                 device=None,
                 dtype=None,
                 operations=None,
                 ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()
        self.dtype = dtype
        operation_settings = {"operations": operations, "device": device, "dtype": dtype}

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = operations.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size, device=operation_settings.get("device"), dtype=torch.float32)
        self.text_embedding = nn.Sequential(
            operations.Linear(text_dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")), nn.GELU(approximate='tanh'),
            operations.Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))

        self.time_embedding = nn.Sequential(
            operations.Linear(freq_dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")), nn.SiLU(), operations.Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))
        self.time_projection = nn.Sequential(nn.SiLU(), operations.Linear(dim, dim * 6, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            wan_attn_block_class(cross_attn_type, dim, ffn_dim, num_heads,
                                 window_size, qk_norm, cross_attn_norm, eps, operation_settings=operation_settings)
            for i in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps, operation_settings=operation_settings)

        d = dim // num_heads
        self.rope_embedder = EmbedND(dim=d, theta=10000.0, axes_dim=[d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)])

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim, flf_pos_embed_token_number=flf_pos_embed_token_number, operation_settings=operation_settings)
        else:
            self.img_emb = None

        if in_dim_ref_conv is not None:
            self.ref_conv = operations.Conv2d(in_dim_ref_conv, dim, kernel_size=patch_size[1:], stride=patch_size[1:], device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        else:
            self.ref_conv = None

    def forward_orig(
        self,
        x,
        t,
        context,
        clip_fea=None,
        freqs=None,
        transformer_options={},
        **kwargs,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (Tensor):
                List of input video tensors with shape [B, C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [B, L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # embeddings
        x = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        transformer_options["grid_sizes"] = grid_sizes
        x = x.flatten(2).transpose(1, 2)

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x[0].dtype))
        e = e.reshape(t.shape[0], -1, e.shape[-1])
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))

        full_ref = None
        if self.ref_conv is not None:
            full_ref = kwargs.get("reference_latent", None)
            if full_ref is not None:
                full_ref = self.ref_conv(full_ref).flatten(2).transpose(1, 2)
                x = torch.concat((full_ref, x), dim=1)

        # In-context reference (Bernini)
        context_latents = kwargs.get("context_latents", None)
        main_len = x.shape[1]
        if context_latents is not None:
            for lat in context_latents:
                cl = self.patch_embedding(lat.float().to(x.device)).to(x.dtype).flatten(2).transpose(1, 2)
                x = torch.cat([x, cl], dim=1)

        # context
        context = self.text_embedding(context)

        context_img_len = None
        if clip_fea is not None:
            if self.img_emb is not None:
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context = torch.concat([context_clip, context], dim=1)
            context_img_len = clip_fea.shape[-2]

        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        transformer_options["total_blocks"] = len(self.blocks)
        transformer_options["block_type"] = "double"
        for i, block in enumerate(self.blocks):
            transformer_options["block_index"] = i
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len, transformer_options=args["transformer_options"])
                    return out
                out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs, "transformer_options": transformer_options}, {"original_block": block_wrap})
                x = out["img"]
            else:
                x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len, transformer_options=transformer_options)

        # head
        x = self.head(x, e)

        if context_latents is not None:
            x = x[:, :main_len]

        if full_ref is not None:
            x = x[:, full_ref.shape[1]:]

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x

    def rope_encode(self, t, h, w, t_start=0, steps_t=None, steps_h=None, steps_w=None, device=None, dtype=None, transformer_options={}, source_id=0):
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

        img_ids = torch.zeros((steps_t, steps_h, steps_w, 3), device=device, dtype=dtype)
        img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(t_start, t_start + (t_len - 1), steps=steps_t, device=device, dtype=dtype).reshape(-1, 1, 1)
        img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(h_start, h_start + (h_len - 1), steps=steps_h, device=device, dtype=dtype).reshape(1, -1, 1)
        img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(w_start, w_start + (w_len - 1), steps=steps_w, device=device, dtype=dtype).reshape(1, 1, -1)
        img_ids = img_ids.reshape(1, -1, img_ids.shape[-1])

        freqs = self.rope_embedder(img_ids).movedim(1, 2)

        # In-context reference: a non-zero source_id composes an extra rotation into the spatial rope
        if source_id:
            d = self.dim // self.num_heads
            pos = torch.tensor([[float(source_id)]], device=freqs.device, dtype=torch.float32)
            id_rot = rope(pos, d, self.rope_embedder.theta).reshape(1, 1, 1, d // 2, 2, 2).to(freqs.dtype)
            freqs = torch.einsum('...ij,...jk->...ik', freqs, id_rot)
        return freqs

    def forward(self, x, timestep, context, clip_fea=None, time_dim_concat=None, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, timestep, context, clip_fea, time_dim_concat, transformer_options, **kwargs)

    def _forward(self, x, timestep, context, clip_fea=None, time_dim_concat=None, transformer_options={}, **kwargs):
        bs, c, t, h, w = x.shape
        x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size)

        t_len = t
        if time_dim_concat is not None:
            time_dim_concat = comfy.ldm.common_dit.pad_to_patch_size(time_dim_concat, self.patch_size)
            x = torch.cat([x, time_dim_concat], dim=2)
            t_len = x.shape[2]

        if self.ref_conv is not None and "reference_latent" in kwargs:
            t_len += 1

        freqs = self.rope_encode(t_len, h, w, device=x.device, dtype=x.dtype, transformer_options=transformer_options)

        # In-context reference: one rope block per stream, each with it's own source_id (1, 2, ...) to distinguish from the target (id 0).
        context_latents = kwargs.get("context_latents", None)
        if context_latents is not None:
            context_latents = [comfy.ldm.common_dit.pad_to_patch_size(lat, self.patch_size) for lat in context_latents]
            for i, lat in enumerate(context_latents):
                freqs = torch.cat([freqs, self.rope_encode(lat.shape[-3], lat.shape[-2], lat.shape[-1], device=x.device, dtype=x.dtype, transformer_options=transformer_options, source_id=i + 1)], dim=1)
            kwargs = {**kwargs, "context_latents": context_latents}

        return self.forward_orig(x, timestep, context, clip_fea=clip_fea, freqs=freqs, transformer_options=transformer_options, **kwargs)[:, :, :t, :h, :w]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [L, C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        u = x
        b = u.shape[0]
        u = u[:, :math.prod(grid_sizes)].view(b, *grid_sizes, *self.patch_size, c)
        u = torch.einsum('bfhwpqrc->bcfphqwr', u)
        u = u.reshape(b, c, *[i * j for i, j in zip(grid_sizes, self.patch_size)])
        return u


class VaceWanModel(WanModel):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    def __init__(self,
                 model_type='vace',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 flf_pos_embed_token_number=None,
                 image_model=None,
                 vace_layers=None,
                 vace_in_dim=None,
                 device=None,
                 dtype=None,
                 operations=None,
                 ):

        super().__init__(model_type='t2v', patch_size=patch_size, text_len=text_len, in_dim=in_dim, dim=dim, ffn_dim=ffn_dim, freq_dim=freq_dim, text_dim=text_dim, out_dim=out_dim, num_heads=num_heads, num_layers=num_layers, window_size=window_size, qk_norm=qk_norm, cross_attn_norm=cross_attn_norm, eps=eps, flf_pos_embed_token_number=flf_pos_embed_token_number, image_model=image_model, device=device, dtype=dtype, operations=operations)
        operation_settings = {"operations": operations, "device": device, "dtype": dtype}

        # Vace
        if vace_layers is not None:
            self.vace_layers = vace_layers
            self.vace_in_dim = vace_in_dim
            # vace blocks
            self.vace_blocks = nn.ModuleList([
                VaceWanAttentionBlock('t2v_cross_attn', self.dim, self.ffn_dim, self.num_heads, self.window_size, self.qk_norm, self.cross_attn_norm, self.eps, block_id=i, operation_settings=operation_settings)
                for i in range(self.vace_layers)
            ])

            self.vace_layers_mapping = {i: n for n, i in enumerate(range(0, self.num_layers, self.num_layers // self.vace_layers))}
            # vace patch embeddings
            self.vace_patch_embedding = operations.Conv3d(
                self.vace_in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size, device=device, dtype=torch.float32
            )

    def forward_orig(
        self,
        x,
        t,
        context,
        vace_context,
        vace_strength,
        clip_fea=None,
        freqs=None,
        transformer_options={},
        **kwargs,
    ):
        # embeddings
        x = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        transformer_options["grid_sizes"] = grid_sizes
        x = x.flatten(2).transpose(1, 2)

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # context
        context = self.text_embedding(context)

        context_img_len = None
        if clip_fea is not None:
            if self.img_emb is not None:
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context = torch.concat([context_clip, context], dim=1)
            context_img_len = clip_fea.shape[-2]

        orig_shape = list(vace_context.shape)
        vace_context = vace_context.movedim(0, 1).reshape([-1] + orig_shape[2:])
        c = self.vace_patch_embedding(vace_context.float()).to(vace_context.dtype)
        c = c.flatten(2).transpose(1, 2)
        c = list(c.split(orig_shape[0], dim=0))

        # arguments
        x_orig = x

        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        transformer_options["total_blocks"] = len(self.blocks)
        transformer_options["block_type"] = "double"
        for i, block in enumerate(self.blocks):
            transformer_options["block_index"] = i
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len, transformer_options=args["transformer_options"])
                    return out
                out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs, "transformer_options": transformer_options}, {"original_block": block_wrap})
                x = out["img"]
            else:
                x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len, transformer_options=transformer_options)

            ii = self.vace_layers_mapping.get(i, None)
            if ii is not None:
                for iii in range(len(c)):
                    c_skip, c[iii] = self.vace_blocks[ii](c[iii], x=x_orig, e=e0, freqs=freqs, context=context, context_img_len=context_img_len, transformer_options=transformer_options)
                    x += c_skip * vace_strength[iii]
                del c_skip
        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x

class CameraWanModel(WanModel):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    def __init__(self,
                 model_type='camera',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 flf_pos_embed_token_number=None,
                 image_model=None,
                 in_dim_control_adapter=24,
                 device=None,
                 dtype=None,
                 operations=None,
                 ):

        if model_type == 'camera':
            model_type = 'i2v'
        else:
            model_type = 't2v'

        super().__init__(model_type=model_type, patch_size=patch_size, text_len=text_len, in_dim=in_dim, dim=dim, ffn_dim=ffn_dim, freq_dim=freq_dim, text_dim=text_dim, out_dim=out_dim, num_heads=num_heads, num_layers=num_layers, window_size=window_size, qk_norm=qk_norm, cross_attn_norm=cross_attn_norm, eps=eps, flf_pos_embed_token_number=flf_pos_embed_token_number, image_model=image_model, device=device, dtype=dtype, operations=operations)
        operation_settings = {"operations": operations, "device": device, "dtype": dtype}

        self.control_adapter = WanCamAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:], operation_settings=operation_settings)


    def forward_orig(
        self,
        x,
        t,
        context,
        clip_fea=None,
        freqs=None,
        camera_conditions = None,
        transformer_options={},
        **kwargs,
    ):
        # embeddings
        x = self.patch_embedding(x.float()).to(x.dtype)
        if self.control_adapter is not None and camera_conditions is not None:
            x = x + self.control_adapter(camera_conditions).to(x.dtype)
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # context
        context = self.text_embedding(context)

        context_img_len = None
        if clip_fea is not None:
            if self.img_emb is not None:
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context = torch.concat([context_clip, context], dim=1)
            context_img_len = clip_fea.shape[-2]

        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        transformer_options["total_blocks"] = len(self.blocks)
        transformer_options["block_type"] = "double"
        for i, block in enumerate(self.blocks):
            transformer_options["block_index"] = i
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len, transformer_options=args["transformer_options"])
                    return out
                out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs, "transformer_options": transformer_options}, {"original_block": block_wrap})
                x = out["img"]
            else:
                x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len, transformer_options=transformer_options)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x


class CausalConv1d(nn.Module):

    def __init__(self,
                 chan_in,
                 chan_out,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 pad_mode='replicate',
                 operations=None,
                 **kwargs):
        super().__init__()

        self.pad_mode = pad_mode
        padding = (kernel_size - 1, 0)  # T
        self.time_causal_padding = padding

        self.conv = operations.Conv1d(
            chan_in,
            chan_out,
            kernel_size,
            stride=stride,
            dilation=dilation,
            **kwargs)

    def forward(self, x):
        x = torch.nn.functional.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)


class MotionEncoder_tc(nn.Module):

    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 num_heads: int,
                 need_global=True,
                 dtype=None,
                 device=None,
                 operations=None,):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()

        self.num_heads = num_heads
        self.need_global = need_global
        self.conv1_local = CausalConv1d(in_dim, hidden_dim // 4 * num_heads, 3, stride=1, operations=operations, **factory_kwargs)
        if need_global:
            self.conv1_global = CausalConv1d(
                in_dim, hidden_dim // 4, 3, stride=1, operations=operations, **factory_kwargs)
        self.norm1 = operations.LayerNorm(
            hidden_dim // 4,
            elementwise_affine=False,
            eps=1e-6,
            **factory_kwargs)
        self.act = nn.SiLU()
        self.conv2 = CausalConv1d(hidden_dim // 4, hidden_dim // 2, 3, stride=2, operations=operations, **factory_kwargs)
        self.conv3 = CausalConv1d(hidden_dim // 2, hidden_dim, 3, stride=2, operations=operations, **factory_kwargs)

        if need_global:
            self.final_linear = operations.Linear(hidden_dim, hidden_dim, **factory_kwargs)

        self.norm1 = operations.LayerNorm(
            hidden_dim // 4,
            elementwise_affine=False,
            eps=1e-6,
            **factory_kwargs)

        self.norm2 = operations.LayerNorm(
            hidden_dim // 2,
            elementwise_affine=False,
            eps=1e-6,
            **factory_kwargs)

        self.norm3 = operations.LayerNorm(
            hidden_dim, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.padding_tokens = nn.Parameter(torch.empty(1, 1, 1, hidden_dim, **factory_kwargs))

    def forward(self, x):
        x = rearrange(x, 'b t c -> b c t')
        x_ori = x.clone()
        b, c, t = x.shape
        x = self.conv1_local(x)
        x = rearrange(x, 'b (n c) t -> (b n) t c', n=self.num_heads)
        x = self.norm1(x)
        x = self.act(x)
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv2(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm2(x)
        x = self.act(x)
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv3(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm3(x)
        x = self.act(x)
        x = rearrange(x, '(b n) t c -> b t n c', b=b)
        padding = comfy.model_management.cast_to(self.padding_tokens, dtype=x.dtype, device=x.device).repeat(b, x.shape[1], 1, 1)
        x = torch.cat([x, padding], dim=-2)
        x_local = x.clone()

        if not self.need_global:
            return x_local

        x = self.conv1_global(x_ori)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm1(x)
        x = self.act(x)
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv2(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm2(x)
        x = self.act(x)
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv3(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm3(x)
        x = self.act(x)
        x = self.final_linear(x)
        x = rearrange(x, '(b n) t c -> b t n c', b=b)

        return x, x_local


class CausalAudioEncoder(nn.Module):

    def __init__(self,
                 dim=5120,
                 num_layers=25,
                 out_dim=2048,
                 video_rate=8,
                 num_token=4,
                 need_global=False,
                 dtype=None,
                 device=None,
                 operations=None):
        super().__init__()
        self.encoder = MotionEncoder_tc(
            in_dim=dim,
            hidden_dim=out_dim,
            num_heads=num_token,
            need_global=need_global, dtype=dtype, device=device, operations=operations)
        weight = torch.empty((1, num_layers, 1, 1), dtype=dtype, device=device)

        self.weights = torch.nn.Parameter(weight)
        self.act = torch.nn.SiLU()

    def forward(self, features):
        # features B * num_layers * dim * video_length
        weights = self.act(comfy.model_management.cast_to(self.weights, dtype=features.dtype, device=features.device))
        weights_sum = weights.sum(dim=1, keepdims=True)
        weighted_feat = ((features * weights) / weights_sum).sum(
            dim=1)  # b dim f
        weighted_feat = weighted_feat.permute(0, 2, 1)  # b f dim
        res = self.encoder(weighted_feat)  # b f n dim
        return res  # b f n dim


class AdaLayerNorm(nn.Module):
    def __init__(self, embedding_dim, output_dim=None, norm_elementwise_affine=False, norm_eps=1e-5, dtype=None, device=None, operations=None):
        super().__init__()

        output_dim = output_dim or embedding_dim * 2

        self.silu = nn.SiLU()
        self.linear = operations.Linear(embedding_dim, output_dim, dtype=dtype, device=device)
        self.norm = operations.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine, dtype=dtype, device=device)

    def forward(self, x, temb):
        temb = self.linear(self.silu(temb))
        shift, scale = temb.chunk(2, dim=1)
        shift = shift[:, None, :]
        scale = scale[:, None, :]
        x = self.norm(x) * (1 + scale) + shift
        return x


class AudioInjector_WAN(nn.Module):

    def __init__(self,
                 dim=2048,
                 num_heads=32,
                 inject_layer=[0, 27],
                 root_net=None,
                 enable_adain=False,
                 adain_dim=2048,
                 adain_mode=None,
                 dtype=None,
                 device=None,
                 operations=None):
        super().__init__()
        self.enable_adain = enable_adain
        self.adain_mode = adain_mode
        self.injected_block_id = {}
        audio_injector_id = 0
        for inject_id in inject_layer:
            self.injected_block_id[inject_id] = audio_injector_id
            audio_injector_id += 1

        self.injector = nn.ModuleList([
            WanT2VCrossAttention(
                dim=dim,
                num_heads=num_heads,
                qk_norm=True, operation_settings={"operations": operations, "device": device, "dtype": dtype}
            ) for _ in range(audio_injector_id)
        ])
        self.injector_pre_norm_feat = nn.ModuleList([
            operations.LayerNorm(
                dim,
                elementwise_affine=False,
                eps=1e-6, dtype=dtype, device=device
            ) for _ in range(audio_injector_id)
        ])
        self.injector_pre_norm_vec = nn.ModuleList([
            operations.LayerNorm(
                dim,
                elementwise_affine=False,
                eps=1e-6, dtype=dtype, device=device
            ) for _ in range(audio_injector_id)
        ])
        if enable_adain:
            self.injector_adain_layers = nn.ModuleList([
                AdaLayerNorm(
                    output_dim=dim * 2, embedding_dim=adain_dim, dtype=dtype, device=device, operations=operations)
                for _ in range(audio_injector_id)
            ])
            if adain_mode != "attn_norm":
                self.injector_adain_output_layers = nn.ModuleList(
                    [operations.Linear(dim, dim, dtype=dtype, device=device) for _ in range(audio_injector_id)])

    def forward(self, x, block_id, audio_emb, audio_emb_global, seq_len, scale=1.0):
        audio_attn_id = self.injected_block_id.get(block_id, None)
        if audio_attn_id is None:
            return x

        num_frames = audio_emb.shape[1]
        input_hidden_states = rearrange(x[:, :seq_len], "b (t n) c -> (b t) n c", t=num_frames)
        if self.enable_adain and self.adain_mode == "attn_norm":
            audio_emb_global = rearrange(audio_emb_global, "b t n c -> (b t) n c")
            adain_hidden_states = self.injector_adain_layers[audio_attn_id](input_hidden_states, temb=audio_emb_global[:, 0])
            attn_hidden_states = adain_hidden_states
        else:
            attn_hidden_states = self.injector_pre_norm_feat[audio_attn_id](input_hidden_states)

        if audio_emb.dim() == 3: # WanDancer case
            attn_audio_emb = rearrange(audio_emb, "b t c -> (b t) 1 c", t=num_frames)
        else: # S2V case
            attn_audio_emb = rearrange(audio_emb, "b t n c -> (b t) n c", t=num_frames)

        residual_out = self.injector[audio_attn_id](x=attn_hidden_states, context=attn_audio_emb)
        residual_out = rearrange(residual_out, "(b t) n c -> b (t n) c", t=num_frames)
        x[:, :seq_len] = x[:, :seq_len] + residual_out * scale
        return x


class FramePackMotioner(nn.Module):
    def __init__(
            self,
            inner_dim=1024,
            num_heads=16,  # Used to indicate the number of heads in the backbone network; unrelated to this module's design
            zip_frame_buckets=[
                1, 2, 16
            ],  # Three numbers representing the number of frames sampled for patch operations from the nearest to the farthest frames
            drop_mode="drop",  # If not "drop", it will use "padd", meaning padding instead of deletion
            dtype=None,
            device=None,
            operations=None):
        super().__init__()
        self.proj = operations.Conv3d(16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2), dtype=dtype, device=device)
        self.proj_2x = operations.Conv3d(16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4), dtype=dtype, device=device)
        self.proj_4x = operations.Conv3d(16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8), dtype=dtype, device=device)
        self.zip_frame_buckets = zip_frame_buckets

        self.inner_dim = inner_dim
        self.num_heads = num_heads

        self.drop_mode = drop_mode

    def forward(self, motion_latents, rope_embedder, add_last_motion=2):
        lat_height, lat_width = motion_latents.shape[3], motion_latents.shape[4]
        padd_lat = torch.zeros(motion_latents.shape[0], 16, sum(self.zip_frame_buckets), lat_height, lat_width).to(device=motion_latents.device, dtype=motion_latents.dtype)
        overlap_frame = min(padd_lat.shape[2], motion_latents.shape[2])
        if overlap_frame > 0:
            padd_lat[:, :, -overlap_frame:] = motion_latents[:, :, -overlap_frame:]

        if add_last_motion < 2 and self.drop_mode != "drop":
            zero_end_frame = sum(self.zip_frame_buckets[:len(self.zip_frame_buckets) - add_last_motion - 1])
            padd_lat[:, :, -zero_end_frame:] = 0

        clean_latents_4x, clean_latents_2x, clean_latents_post = padd_lat[:, :, -sum(self.zip_frame_buckets):, :, :].split(self.zip_frame_buckets[::-1], dim=2)  # 16, 2 ,1

        # patchfy
        clean_latents_post = self.proj(clean_latents_post).flatten(2).transpose(1, 2)
        clean_latents_2x = self.proj_2x(clean_latents_2x)
        l_2x_shape = clean_latents_2x.shape
        clean_latents_2x = clean_latents_2x.flatten(2).transpose(1, 2)
        clean_latents_4x = self.proj_4x(clean_latents_4x)
        l_4x_shape = clean_latents_4x.shape
        clean_latents_4x = clean_latents_4x.flatten(2).transpose(1, 2)

        if add_last_motion < 2 and self.drop_mode == "drop":
            clean_latents_post = clean_latents_post[:, :
                                                    0] if add_last_motion < 2 else clean_latents_post
            clean_latents_2x = clean_latents_2x[:, :
                                                0] if add_last_motion < 1 else clean_latents_2x

        motion_lat = torch.cat([clean_latents_post, clean_latents_2x, clean_latents_4x], dim=1)

        rope_post = rope_embedder.rope_encode(1, lat_height, lat_width, t_start=-1, device=motion_latents.device, dtype=motion_latents.dtype)
        rope_2x = rope_embedder.rope_encode(1, lat_height, lat_width, t_start=-3, steps_h=l_2x_shape[-2], steps_w=l_2x_shape[-1], device=motion_latents.device, dtype=motion_latents.dtype)
        rope_4x = rope_embedder.rope_encode(4, lat_height, lat_width, t_start=-19, steps_h=l_4x_shape[-2], steps_w=l_4x_shape[-1], device=motion_latents.device, dtype=motion_latents.dtype)

        rope = torch.cat([rope_post, rope_2x, rope_4x], dim=1)
        return motion_lat, rope


class WanModel_S2V(WanModel):
    def __init__(self,
                 model_type='s2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 audio_dim=1024,
                 num_audio_token=4,
                 enable_adain=True,
                 cond_dim=16,
                 audio_inject_layers=[0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39],
                 adain_mode="attn_norm",
                 framepack_drop_mode="padd",
                 image_model=None,
                 device=None,
                 dtype=None,
                 operations=None,
                 ):

        super().__init__(model_type='t2v', patch_size=patch_size, text_len=text_len, in_dim=in_dim, dim=dim, ffn_dim=ffn_dim, freq_dim=freq_dim, text_dim=text_dim, out_dim=out_dim, num_heads=num_heads, num_layers=num_layers, window_size=window_size, qk_norm=qk_norm, cross_attn_norm=cross_attn_norm, eps=eps, image_model=image_model, device=device, dtype=dtype, operations=operations)

        self.trainable_cond_mask = operations.Embedding(3, self.dim, device=device, dtype=dtype)

        self.casual_audio_encoder = CausalAudioEncoder(
            dim=audio_dim,
            out_dim=self.dim,
            num_token=num_audio_token,
            need_global=enable_adain, dtype=dtype, device=device, operations=operations)

        if cond_dim > 0:
            self.cond_encoder = operations.Conv3d(
                cond_dim,
                self.dim,
                kernel_size=self.patch_size,
                stride=self.patch_size, device=device, dtype=dtype)

        self.audio_injector = AudioInjector_WAN(
            dim=self.dim,
            num_heads=self.num_heads,
            inject_layer=audio_inject_layers,
            root_net=self,
            enable_adain=enable_adain,
            adain_dim=self.dim,
            adain_mode=adain_mode,
            dtype=dtype, device=device, operations=operations
        )

        self.frame_packer = FramePackMotioner(
            inner_dim=self.dim,
            num_heads=self.num_heads,
            zip_frame_buckets=[1, 2, 16],
            drop_mode=framepack_drop_mode,
            dtype=dtype, device=device, operations=operations)

    def forward_orig(
        self,
        x,
        t,
        context,
        audio_embed=None,
        reference_latent=None,
        control_video=None,
        reference_motion=None,
        clip_fea=None,
        freqs=None,
        transformer_options={},
        **kwargs,
    ):
        if audio_embed is not None:
            num_embeds = x.shape[-3] * 4
            audio_emb_global, audio_emb = self.casual_audio_encoder(audio_embed[:, :, :, :num_embeds])
        else:
            audio_emb = None

        # embeddings
        bs, _, time, height, width = x.shape
        x = self.patch_embedding(x.float()).to(x.dtype)
        if control_video is not None:
            x = x + self.cond_encoder(control_video)

        if t.ndim == 1:
            t = t.unsqueeze(1).repeat(1, x.shape[2])

        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        seq_len = x.size(1)

        cond_mask_weight = comfy.model_management.cast_to(self.trainable_cond_mask.weight, dtype=x.dtype, device=x.device).unsqueeze(1).unsqueeze(1)
        x = x + cond_mask_weight[0]

        if reference_latent is not None:
            ref = self.patch_embedding(reference_latent.float()).to(x.dtype)
            ref = ref.flatten(2).transpose(1, 2)
            freqs_ref = self.rope_encode(reference_latent.shape[-3], reference_latent.shape[-2], reference_latent.shape[-1], t_start=max(30, time + 9), device=x.device, dtype=x.dtype)
            ref = ref + cond_mask_weight[1]
            x = torch.cat([x, ref], dim=1)
            freqs = torch.cat([freqs, freqs_ref], dim=1)
            t = torch.cat([t, torch.zeros((t.shape[0], reference_latent.shape[-3]), device=t.device, dtype=t.dtype)], dim=1)
            del ref, freqs_ref

        if reference_motion is not None:
            motion_encoded, freqs_motion = self.frame_packer(reference_motion, self)
            motion_encoded = motion_encoded + cond_mask_weight[2]
            x = torch.cat([x, motion_encoded], dim=1)
            freqs = torch.cat([freqs, freqs_motion], dim=1)

            t = torch.repeat_interleave(t, 2, dim=1)
            t = torch.cat([t, torch.zeros((t.shape[0], 3), device=t.device, dtype=t.dtype)], dim=1)
            del motion_encoded, freqs_motion

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x[0].dtype))
        e = e.reshape(t.shape[0], -1, e.shape[-1])
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))

        # context
        context = self.text_embedding(context)

        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        transformer_options["total_blocks"] = len(self.blocks)
        transformer_options["block_type"] = "double"
        for i, block in enumerate(self.blocks):
            transformer_options["block_index"] = i
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], transformer_options=args["transformer_options"])
                    return out
                out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs, "transformer_options": transformer_options}, {"original_block": block_wrap})
                x = out["img"]
            else:
                x = block(x, e=e0, freqs=freqs, context=context, transformer_options=transformer_options)
            if audio_emb is not None:
                x = self.audio_injector(x, i, audio_emb, audio_emb_global, seq_len)
        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x


class WanT2VCrossAttentionGather(WanSelfAttention):

    def forward(self, x, context, transformer_options={}, **kwargs):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C] - video tokens
            context(Tensor): Shape [B, L2, C] - audio tokens with shape [B, frames*16, 1536]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        v = self.v(context)

        # Handle audio temporal structure (16 tokens per frame)
        k = k.reshape(-1, 16, n, d).transpose(1, 2)
        v = v.reshape(-1, 16, n, d).transpose(1, 2)

        # Handle video spatial structure
        q = q.reshape(k.shape[0], -1, n, d).transpose(1, 2)

        x = optimized_attention(q, k, v, heads=self.num_heads, skip_reshape=True, skip_output_reshape=True, transformer_options=transformer_options)

        x = x.transpose(1, 2).reshape(b, -1, n * d)
        x = self.o(x)
        return x


class AudioCrossAttentionWrapper(nn.Module):
    def __init__(self, dim, kv_dim, num_heads, qk_norm=True, eps=1e-6, operation_settings={}):
        super().__init__()

        self.audio_cross_attn = WanT2VCrossAttentionGather(dim, num_heads, qk_norm=qk_norm, kv_dim=kv_dim, eps=eps, operation_settings=operation_settings)
        self.norm1_audio = operation_settings.get("operations").LayerNorm(dim, eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))

    def forward(self, x, audio, transformer_options={}):
        x = x + self.audio_cross_attn(self.norm1_audio(x), audio, transformer_options=transformer_options)
        return x


class WanAttentionBlockAudio(WanAttentionBlock):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6, operation_settings={}):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps, operation_settings)
        self.audio_cross_attn_wrapper = AudioCrossAttentionWrapper(dim, 1536, num_heads, qk_norm, eps, operation_settings=operation_settings)

    def forward(
        self,
        x,
        e,
        freqs,
        context,
        context_img_len=257,
        audio=None,
        transformer_options={},
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # assert e.dtype == torch.float32

        if e.ndim < 4:
            e = (comfy.model_management.cast_to(self.modulation, dtype=x.dtype, device=x.device) + e).chunk(6, dim=1)
        else:
            e = (comfy.model_management.cast_to(self.modulation, dtype=x.dtype, device=x.device).unsqueeze(0) + e).unbind(2)
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            torch.addcmul(repeat_e(e[0], x), self.norm1(x), 1 + repeat_e(e[1], x)),
            freqs, transformer_options=transformer_options)

        x = torch.addcmul(x, y, repeat_e(e[2], x))

        # cross-attention & ffn
        x = x + self.cross_attn(self.norm3(x), context, context_img_len=context_img_len, transformer_options=transformer_options)
        if audio is not None:
            x = self.audio_cross_attn_wrapper(x, audio, transformer_options=transformer_options)
        y = self.ffn(torch.addcmul(repeat_e(e[3], x), self.norm2(x), 1 + repeat_e(e[4], x)))
        x = torch.addcmul(x, y, repeat_e(e[5], x))
        return x

class DummyAdapterLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, *args, **kwargs):
        return self.layer(*args, **kwargs)


class AudioProjModel(nn.Module):
    def __init__(
        self,
        seq_len=5,
        blocks=13,  # add a new parameter blocks
        channels=768,  # add a new parameter channels
        intermediate_dim=512,
        output_dim=1536,
        context_tokens=16,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = seq_len * blocks * channels  # update input_dim to be the product of blocks and channels.
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        # define multiple linear layers
        self.audio_proj_glob_1 = DummyAdapterLayer(operations.Linear(self.input_dim, intermediate_dim, dtype=dtype, device=device))
        self.audio_proj_glob_2 = DummyAdapterLayer(operations.Linear(intermediate_dim, intermediate_dim, dtype=dtype, device=device))
        self.audio_proj_glob_3 = DummyAdapterLayer(operations.Linear(intermediate_dim, context_tokens * output_dim, dtype=dtype, device=device))

        self.audio_proj_glob_norm = DummyAdapterLayer(operations.LayerNorm(output_dim, dtype=dtype, device=device))

    def forward(self, audio_embeds):
        video_length = audio_embeds.shape[1]
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)

        audio_embeds = torch.relu(self.audio_proj_glob_1(audio_embeds))
        audio_embeds = torch.relu(self.audio_proj_glob_2(audio_embeds))

        context_tokens = self.audio_proj_glob_3(audio_embeds).reshape(batch_size, self.context_tokens, self.output_dim)

        context_tokens = self.audio_proj_glob_norm(context_tokens)
        context_tokens = rearrange(context_tokens, "(bz f) m c -> bz f m c", f=video_length)

        return context_tokens


class HumoWanModel(WanModel):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    def __init__(self,
                 model_type='humo',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 flf_pos_embed_token_number=None,
                 image_model=None,
                 audio_token_num=16,
                 device=None,
                 dtype=None,
                 operations=None,
                 ):

        super().__init__(model_type='t2v', patch_size=patch_size, text_len=text_len, in_dim=in_dim, dim=dim, ffn_dim=ffn_dim, freq_dim=freq_dim, text_dim=text_dim, out_dim=out_dim, num_heads=num_heads, num_layers=num_layers, window_size=window_size, qk_norm=qk_norm, cross_attn_norm=cross_attn_norm, eps=eps, flf_pos_embed_token_number=flf_pos_embed_token_number, wan_attn_block_class=WanAttentionBlockAudio, image_model=image_model, device=device, dtype=dtype, operations=operations)

        self.audio_proj = AudioProjModel(seq_len=8, blocks=5, channels=1280, intermediate_dim=512, output_dim=1536, context_tokens=audio_token_num, dtype=dtype, device=device, operations=operations)

    def forward_orig(
        self,
        x,
        t,
        context,
        freqs=None,
        audio_embed=None,
        reference_latent=None,
        transformer_options={},
        **kwargs,
    ):
        bs, _, time, height, width = x.shape

        # embeddings
        x = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x[0].dtype))
        e = e.reshape(t.shape[0], -1, e.shape[-1])
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))

        if reference_latent is not None:
            ref = self.patch_embedding(reference_latent.float()).to(x.dtype)
            ref = ref.flatten(2).transpose(1, 2)
            freqs_ref = self.rope_encode(reference_latent.shape[-3], reference_latent.shape[-2], reference_latent.shape[-1], t_start=time, device=x.device, dtype=x.dtype)
            x = torch.cat([x, ref], dim=1)
            freqs = torch.cat([freqs, freqs_ref], dim=1)
            del ref, freqs_ref

        # context
        context = self.text_embedding(context)
        context_img_len = None

        if audio_embed is not None:
            if reference_latent is not None:
                zero_audio_pad = torch.zeros(audio_embed.shape[0], reference_latent.shape[-3], *audio_embed.shape[2:], device=audio_embed.device, dtype=audio_embed.dtype)
                audio_embed = torch.cat([audio_embed, zero_audio_pad], dim=1)
            audio = self.audio_proj(audio_embed).permute(0, 3, 1, 2).flatten(2).transpose(1, 2)
        else:
            audio = None

        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        transformer_options["total_blocks"] = len(self.blocks)
        transformer_options["block_type"] = "double"
        for i, block in enumerate(self.blocks):
            transformer_options["block_index"] = i
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len, audio=audio, transformer_options=args["transformer_options"])
                    return out
                out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs, "transformer_options": transformer_options}, {"original_block": block_wrap})
                x = out["img"]
            else:
                x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len, audio=audio, transformer_options=transformer_options)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x

class SCAILWanModel(WanModel):
    def __init__(self, model_type="scail", patch_size=(1, 2, 2), in_dim=20, dim=5120, operations=None, device=None, dtype=None, **kwargs):
        super().__init__(model_type='i2v', patch_size=patch_size, in_dim=in_dim, dim=dim, operations=operations, device=device, dtype=dtype, **kwargs)

        self.patch_embedding_pose = operations.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size, device=device, dtype=torch.float32)

    def forward_orig(self, x, t, context, clip_fea=None, freqs=None, transformer_options={}, pose_latents=None, reference_latent=None, ref_mask_latents=None, sam_latents=None, **kwargs):

        if reference_latent is not None:
            x = torch.cat((reference_latent, x), dim=2)

        # embeddings
        x = self.patch_embedding(x.float()).to(x.dtype)
        if ref_mask_latents is not None:  # SCAIL-2 additive mask stream (one identity mask frame per reference, then video)
            x = x + self.patch_embedding_mask(ref_mask_latents.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        transformer_options["grid_sizes"] = grid_sizes
        x = x.flatten(2).transpose(1, 2)

        scail_pose_seq_len = 0
        if pose_latents is not None:
            scail_x = self.patch_embedding_pose(pose_latents.float()).to(x.dtype)
            if sam_latents is not None:  # SCAIL-2 additive mask stream
                scail_x = scail_x + self.patch_embedding_mask(sam_latents.float()).to(x.dtype)
            scail_x = scail_x.flatten(2).transpose(1, 2)
            scail_pose_seq_len = scail_x.shape[1]
            x = torch.cat([x, scail_x], dim=1)
            del scail_x

        # time embeddings
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x[0].dtype))
        e = e.reshape(t.shape[0], -1, e.shape[-1])
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))

        # context
        context = self.text_embedding(context)

        context_img_len = None
        if clip_fea is not None:
            if self.img_emb is not None:
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context = torch.cat([context_clip, context], dim=1)
            context_img_len = clip_fea.shape[-2]

        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        transformer_options["total_blocks"] = len(self.blocks)
        transformer_options["block_type"] = "double"
        for i, block in enumerate(self.blocks):
            transformer_options["block_index"] = i
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len, transformer_options=args["transformer_options"])
                    return out
                out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs, "transformer_options": transformer_options}, {"original_block": block_wrap})
                x = out["img"]
            else:
                x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len, transformer_options=transformer_options)

        # head
        x = self.head(x, e)

        if scail_pose_seq_len > 0:
            x = x[:, :-scail_pose_seq_len]

        # unpatchify
        x = self.unpatchify(x, grid_sizes)

        if reference_latent is not None:
            x = x[:, :, reference_latent.shape[2]:]

        return x

    # ref_mask_flag is a scalar bool (CONDConstant, SCAIL-2 only). False => replacement mode,
    # which places ref/pose via H/W rope shifts instead of the animation-mode temporal offset.
    # reference_latent may stack several frames: the last is the primary reference adjacent to the video, the earlier frames are additional references.
    def rope_encode(self, t, h, w, t_start=0, steps_t=None, steps_h=None, steps_w=None, device=None, dtype=None, pose_latents=None, reference_latent=None, ref_mask_flag=None, transformer_options={}):
        ref_t_patches = 0
        if reference_latent is not None:
            ref_t_patches = (reference_latent.shape[2] + (self.patch_size[0] // 2)) // self.patch_size[0]

        if ref_mask_flag is not None and not bool(ref_mask_flag):
            REF_ROPE_H = 120.0
            POSE_ROPE_W = 120.0

            main_t_patches = t - ref_t_patches
            video_t_start = max(ref_t_patches - 1, 0)

            parts = []
            if ref_t_patches > 0:
                ref_tf = {"rope_options": {"shift_y": REF_ROPE_H, "shift_x": 0.0, "scale_y": 1.0, "scale_x": 1.0}}
                parts.append(super().rope_encode(ref_t_patches, h, w, t_start=0, device=device, dtype=dtype, transformer_options=ref_tf))
            if main_t_patches > 0:
                parts.append(super().rope_encode(main_t_patches, h, w, t_start=video_t_start, device=device, dtype=dtype, transformer_options=transformer_options))

            if pose_latents is not None:
                F_pose, H_pose, W_pose = pose_latents.shape[-3], pose_latents.shape[-2], pose_latents.shape[-1]
                h_scale = h / H_pose
                w_scale = w / W_pose
                h_shift = (h_scale - 1) / 2
                w_shift = (w_scale - 1) / 2
                pose_tf = {"rope_options": {"shift_y": h_shift, "shift_x": POSE_ROPE_W + w_shift, "scale_y": h_scale, "scale_x": w_scale}}
                parts.append(super().rope_encode(F_pose, H_pose, W_pose, t_start=video_t_start, device=device, dtype=dtype, transformer_options=pose_tf))

            return torch.cat(parts, dim=1)

        main_freqs = super().rope_encode(t, h, w, t_start=t_start, steps_t=steps_t, steps_h=steps_h, steps_w=steps_w, device=device, dtype=dtype, transformer_options=transformer_options)

        if pose_latents is None:
            return main_freqs

        F_pose, H_pose, W_pose = pose_latents.shape[-3], pose_latents.shape[-2], pose_latents.shape[-1]

        # if pose is at half resolution, scale_y/scale_x=2 stretches the position range to cover the same RoPE extent as the main frames
        h_scale = h / H_pose
        w_scale = w / W_pose

        # 120 w-offset and shift 0.5 to place positions at midpoints (0.5, 2.5, ...) to match the original code
        h_shift = (h_scale - 1) / 2
        w_shift = (w_scale - 1) / 2
        pose_transformer_options = {"rope_options": {"shift_y": h_shift, "shift_x": 120.0 + w_shift, "scale_y": h_scale, "scale_x": w_scale}}
        pose_freqs = super().rope_encode(F_pose, H_pose, W_pose, t_start=t_start+ref_t_patches, device=device, dtype=dtype, transformer_options=pose_transformer_options)

        return torch.cat([main_freqs, pose_freqs], dim=1)

    def _forward(self, x, timestep, context, clip_fea=None, time_dim_concat=None, transformer_options={}, pose_latents=None, ref_mask_latents=None, sam_latents=None, **kwargs):
        bs, c, t, h, w = x.shape
        x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size)

        if pose_latents is not None:
            pose_latents = comfy.ldm.common_dit.pad_to_patch_size(pose_latents, self.patch_size)
        if ref_mask_latents is not None:  # SCAIL-2
            ref_mask_latents = comfy.ldm.common_dit.pad_to_patch_size(ref_mask_latents, self.patch_size)
        if sam_latents is not None:  # SCAIL-2
            sam_latents = comfy.ldm.common_dit.pad_to_patch_size(sam_latents, self.patch_size)

        t_len = t
        if time_dim_concat is not None:
            time_dim_concat = comfy.ldm.common_dit.pad_to_patch_size(time_dim_concat, self.patch_size)
            x = torch.cat([x, time_dim_concat], dim=2)
            t_len = x.shape[2]

        reference_latent = None
        if "reference_latent" in kwargs:
            reference_latent = comfy.ldm.common_dit.pad_to_patch_size(kwargs.pop("reference_latent"), self.patch_size)
            t_len += reference_latent.shape[2]

        ref_mask_flag = kwargs.pop("ref_mask_flag", None)  # SCAIL-2

        freqs = self.rope_encode(t_len, h, w, device=x.device, dtype=x.dtype, transformer_options=transformer_options, pose_latents=pose_latents, reference_latent=reference_latent, ref_mask_flag=ref_mask_flag)
        return self.forward_orig(x, timestep, context, clip_fea=clip_fea, freqs=freqs, transformer_options=transformer_options, pose_latents=pose_latents, reference_latent=reference_latent, ref_mask_latents=ref_mask_latents, sam_latents=sam_latents, **kwargs)[:, :, :t, :h, :w]


class SCAIL2WanModel(SCAILWanModel):
    """SCAIL-2: SCAIL-Preview + an additive binary multi-identity mask stream."""

    def __init__(self, model_type="scail2", patch_size=(1, 2, 2), in_dim=20, mask_in_dim=28, dim=5120, operations=None, device=None, dtype=None, **kwargs):
        super().__init__(model_type=model_type, patch_size=patch_size, in_dim=in_dim, dim=dim, operations=operations, device=device, dtype=dtype, **kwargs)
        self.patch_embedding_mask = operations.Conv3d(mask_in_dim, dim, kernel_size=patch_size, stride=patch_size, device=device, dtype=torch.float32)
