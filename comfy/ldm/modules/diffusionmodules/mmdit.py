from functools import partial
from typing import Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
from ..attention import optimized_attention
from einops import rearrange, repeat
from .util import timestep_embedding
import comfy.ops
import comfy.ldm.common_dit

def default(x, y):
    if x is not None:
        return x
    return y

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
            dtype=None,
            device=None,
            operations=None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = drop
        linear_layer = partial(operations.Conv2d, kernel_size=1) if use_conv else operations.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias, dtype=dtype, device=device)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias, dtype=dtype, device=device)
        self.drop2 = nn.Dropout(drop_probs)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
            self,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer = None,
            flatten: bool = True,
            bias: bool = True,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = True,
            padding_mode='circular',
            conv3d=False,
            dtype=None,
            device=None,
            operations=None,
    ):
        super().__init__()
        try:
            len(patch_size)
            self.patch_size = patch_size
        except:
            if conv3d:
                self.patch_size = (patch_size, patch_size, patch_size)
            else:
                self.patch_size = (patch_size, patch_size)
        self.padding_mode = padding_mode

        # flatten spatial dim and transpose to channels last, kept for bwd compat
        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad
        if conv3d:
            self.proj = operations.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias, dtype=dtype, device=device)
        else:
            self.proj = operations.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias, dtype=dtype, device=device)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        if self.dynamic_img_pad:
            x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size, padding_mode=self.padding_mode)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        x = self.norm(x)
        return x

def modulate(x, shift, scale):
    if shift is None:
        shift = torch.zeros_like(scale)
    return torch.addcmul(shift.unsqueeze(1), x, 1+ scale.unsqueeze(1))


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################


def get_2d_sincos_pos_embed(
    embed_dim,
    grid_size,
    cls_token=False,
    extra_tokens=0,
    scaling_factor=None,
    offset=None,
):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    if scaling_factor is not None:
        grid = grid / scaling_factor
    if offset is not None:
        grid = grid - offset

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos, device=None, dtype=torch.float32):
    omega = torch.arange(embed_dim // 2, device=device, dtype=dtype)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product
    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)
    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed_torch(embed_dim, w, h, val_center=7.5, val_magnitude=7.5, device=None, dtype=torch.float32):
    small = min(h, w)
    val_h = (h / small) * val_magnitude
    val_w = (w / small) * val_magnitude
    grid_h, grid_w = torch.meshgrid(torch.linspace(-val_h + val_center, val_h + val_center, h, device=device, dtype=dtype), torch.linspace(-val_w + val_center, val_w + val_center, w, device=device, dtype=dtype), indexing='ij')
    emb_h = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid_h, device=device, dtype=dtype)
    emb_w = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid_w, device=device, dtype=dtype)
    emb = torch.cat([emb_w, emb_h], dim=1)  # (H*W, D)
    return emb


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, output_size=None, dtype=None, device=None, operations=None, max_period=10000):
        super().__init__()
        if output_size is None:
            output_size = hidden_size
        self.mlp = nn.Sequential(
            operations.Linear(frequency_embedding_size, hidden_size, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Linear(hidden_size, output_size, bias=True, dtype=dtype, device=device),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

    def forward(self, t, dtype, **kwargs):
        t_freq = timestep_embedding(t, self.frequency_embedding_size, max_period=self.max_period).to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class VectorEmbedder(nn.Module):
    """
    Embeds a flat vector of dimension input_dim
    """

    def __init__(self, input_dim: int, hidden_size: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.mlp = nn.Sequential(
            operations.Linear(input_dim, hidden_size, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.mlp(x)
        return emb


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


def split_qkv(qkv, head_dim):
    qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, -1, head_dim).movedim(2, 0)
    return qkv[0], qkv[1], qkv[2]


class SelfAttention(nn.Module):
    ATTENTION_MODES = ("xformers", "torch", "torch-hb", "math", "debug")

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        proj_drop: float = 0.0,
        attn_mode: str = "xformers",
        pre_only: bool = False,
        qk_norm: Optional[str] = None,
        rmsnorm: bool = False,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = operations.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        if not pre_only:
            self.proj = operations.Linear(dim, dim, dtype=dtype, device=device)
            self.proj_drop = nn.Dropout(proj_drop)
        assert attn_mode in self.ATTENTION_MODES
        self.attn_mode = attn_mode
        self.pre_only = pre_only

        if qk_norm == "rms":
            self.ln_q = RMSNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6, dtype=dtype, device=device)
            self.ln_k = RMSNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6, dtype=dtype, device=device)
        elif qk_norm == "ln":
            self.ln_q = operations.LayerNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6, dtype=dtype, device=device)
            self.ln_k = operations.LayerNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6, dtype=dtype, device=device)
        elif qk_norm is None:
            self.ln_q = nn.Identity()
            self.ln_k = nn.Identity()
        else:
            raise ValueError(qk_norm)

    def pre_attention(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        qkv = self.qkv(x)
        q, k, v = split_qkv(qkv, self.head_dim)
        q = self.ln_q(q).reshape(q.shape[0], q.shape[1], -1)
        k = self.ln_k(k).reshape(q.shape[0], q.shape[1], -1)
        return (q, k, v)

    def post_attention(self, x: torch.Tensor) -> torch.Tensor:
        assert not self.pre_only
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.pre_attention(x)
        x = optimized_attention(
            q, k, v, heads=self.num_heads
        )
        x = self.post_attention(x)
        return x


class RMSNorm(torch.nn.Module):
    def __init__(
        self, dim: int, elementwise_affine: bool = False, eps: float = 1e-6, device=None, dtype=None, **kwargs
    ):
        """
        Initialize the RMSNorm normalization layer.
        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.
        """
        super().__init__()
        self.eps = eps
        self.learnable_scale = elementwise_affine
        if self.learnable_scale:
            self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        return comfy.ldm.common_dit.rms_norm(x, self.weight, self.eps)



class SwiGLUFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class DismantledBlock(nn.Module):
    """
    A DiT block with gated adaptive layer norm (adaLN) conditioning.
    """

    ATTENTION_MODES = ("xformers", "torch", "torch-hb", "math", "debug")

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: str = "xformers",
        qkv_bias: bool = False,
        pre_only: bool = False,
        rmsnorm: bool = False,
        scale_mod_only: bool = False,
        swiglu: bool = False,
        qk_norm: Optional[str] = None,
        x_block_self_attn: bool = False,
        dtype=None,
        device=None,
        operations=None,
        **block_kwargs,
    ):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        if not rmsnorm:
            self.norm1 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        else:
            self.norm1 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_mode=attn_mode,
            pre_only=pre_only,
            qk_norm=qk_norm,
            rmsnorm=rmsnorm,
            dtype=dtype,
            device=device,
            operations=operations
        )
        if x_block_self_attn:
            assert not pre_only
            assert not scale_mod_only
            self.x_block_self_attn = True
            self.attn2 = SelfAttention(
                dim=hidden_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_mode=attn_mode,
                pre_only=False,
                qk_norm=qk_norm,
                rmsnorm=rmsnorm,
                dtype=dtype,
                device=device,
                operations=operations
            )
        else:
            self.x_block_self_attn = False
        if not pre_only:
            if not rmsnorm:
                self.norm2 = operations.LayerNorm(
                    hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
                )
            else:
                self.norm2 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if not pre_only:
            if not swiglu:
                self.mlp = Mlp(
                    in_features=hidden_size,
                    hidden_features=mlp_hidden_dim,
                    act_layer=lambda: nn.GELU(approximate="tanh"),
                    drop=0,
                    dtype=dtype,
                    device=device,
                    operations=operations
                )
            else:
                self.mlp = SwiGLUFeedForward(
                    dim=hidden_size,
                    hidden_dim=mlp_hidden_dim,
                    multiple_of=256,
                )
        self.scale_mod_only = scale_mod_only
        if x_block_self_attn:
            assert not pre_only
            assert not scale_mod_only
            n_mods = 9
        elif not scale_mod_only:
            n_mods = 6 if not pre_only else 2
        else:
            n_mods = 4 if not pre_only else 1
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), operations.Linear(hidden_size, n_mods * hidden_size, bias=True, dtype=dtype, device=device)
        )
        self.pre_only = pre_only

    def pre_attention(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if not self.pre_only:
            if not self.scale_mod_only:
                (
                    shift_msa,
                    scale_msa,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                ) = self.adaLN_modulation(c).chunk(6, dim=1)
            else:
                shift_msa = None
                shift_mlp = None
                (
                    scale_msa,
                    gate_msa,
                    scale_mlp,
                    gate_mlp,
                ) = self.adaLN_modulation(
                    c
                ).chunk(4, dim=1)
            qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
            return qkv, (
                x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            )
        else:
            if not self.scale_mod_only:
                (
                    shift_msa,
                    scale_msa,
                ) = self.adaLN_modulation(
                    c
                ).chunk(2, dim=1)
            else:
                shift_msa = None
                scale_msa = self.adaLN_modulation(c)
            qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
            return qkv, None

    def post_attention(self, attn, x, gate_msa, shift_mlp, scale_mlp, gate_mlp):
        assert not self.pre_only
        x = x + gate_msa.unsqueeze(1) * self.attn.post_attention(attn)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x

    def pre_attention_x(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        assert self.x_block_self_attn
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            shift_msa2,
            scale_msa2,
            gate_msa2,
        ) = self.adaLN_modulation(c).chunk(9, dim=1)
        x_norm = self.norm1(x)
        qkv = self.attn.pre_attention(modulate(x_norm, shift_msa, scale_msa))
        qkv2 = self.attn2.pre_attention(modulate(x_norm, shift_msa2, scale_msa2))
        return qkv, qkv2, (
            x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            gate_msa2,
        )

    def post_attention_x(self, attn, attn2, x, gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_msa2):
        assert not self.pre_only
        attn1 = self.attn.post_attention(attn)
        attn2 = self.attn2.post_attention(attn2)
        x = gate_cat(x, gate_msa, gate_msa2, attn1, attn2)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        assert not self.pre_only
        if self.x_block_self_attn:
            qkv, qkv2, intermediates = self.pre_attention_x(x, c)
            attn, _ = optimized_attention(
                qkv[0], qkv[1], qkv[2],
                num_heads=self.attn.num_heads,
            )
            attn2, _ = optimized_attention(
                qkv2[0], qkv2[1], qkv2[2],
                num_heads=self.attn2.num_heads,
            )
            return self.post_attention_x(attn, attn2, *intermediates)
        else:
            qkv, intermediates = self.pre_attention(x, c)
            attn = optimized_attention(
                qkv[0], qkv[1], qkv[2],
                heads=self.attn.num_heads,
            )
            return self.post_attention(attn, *intermediates)

def gate_cat(x, gate_msa, gate_msa2, attn1, attn2):
    out1 = gate_msa.unsqueeze(1) * attn1
    out2 = gate_msa2.unsqueeze(1) * attn2
    x = torch.stack([x, out1, out2], dim=0).sum(dim=0)
    return x

def block_mixing(*args, use_checkpoint=True, **kwargs):
    if use_checkpoint:
        return torch.utils.checkpoint.checkpoint(
            _block_mixing, *args, use_reentrant=False, **kwargs
        )
    else:
        return _block_mixing(*args, **kwargs)


def _block_mixing(context, x, context_block, x_block, c, transformer_options={}):
    context_qkv, context_intermediates = context_block.pre_attention(context, c)

    if x_block.x_block_self_attn:
        x_qkv, x_qkv2, x_intermediates = x_block.pre_attention_x(x, c)
    else:
        x_qkv, x_intermediates = x_block.pre_attention(x, c)

    o = []
    for t in range(3):
        o.append(torch.cat((context_qkv[t], x_qkv[t]), dim=1))
    qkv = tuple(o)

    attn = optimized_attention(
        qkv[0], qkv[1], qkv[2],
        heads=x_block.attn.num_heads,
        transformer_options=transformer_options,
    )
    context_attn, x_attn = (
        attn[:, : context_qkv[0].shape[1]],
        attn[:, context_qkv[0].shape[1] :],
    )

    if not context_block.pre_only:
        context = context_block.post_attention(context_attn, *context_intermediates)

    else:
        context = None
    if x_block.x_block_self_attn:
        attn2 = optimized_attention(
                x_qkv2[0], x_qkv2[1], x_qkv2[2],
                heads=x_block.attn2.num_heads,
                transformer_options=transformer_options,
            )
        x = x_block.post_attention_x(x_attn, attn2, *x_intermediates)
    else:
        x = x_block.post_attention(x_attn, *x_intermediates)
    return context, x


class JointBlock(nn.Module):
    """just a small wrapper to serve as a fsdp unit"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()
        pre_only = kwargs.pop("pre_only")
        qk_norm = kwargs.pop("qk_norm", None)
        x_block_self_attn = kwargs.pop("x_block_self_attn", False)
        self.context_block = DismantledBlock(*args, pre_only=pre_only, qk_norm=qk_norm, **kwargs)
        self.x_block = DismantledBlock(*args,
                                       pre_only=False,
                                       qk_norm=qk_norm,
                                       x_block_self_attn=x_block_self_attn,
                                       **kwargs)

    def forward(self, *args, **kwargs):
        return block_mixing(
            *args, context_block=self.context_block, x_block=self.x_block, **kwargs
        )


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        total_out_channels: Optional[int] = None,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.norm_final = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.linear = (
            operations.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True, dtype=dtype, device=device)
            if (total_out_channels is None)
            else operations.Linear(hidden_size, total_out_channels, bias=True, dtype=dtype, device=device)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), operations.Linear(hidden_size, 2 * hidden_size, bias=True, dtype=dtype, device=device)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class SelfAttentionContext(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dtype=None, device=None, operations=None):
        super().__init__()
        dim_head = dim // heads
        inner_dim = dim

        self.heads = heads
        self.dim_head = dim_head

        self.qkv = operations.Linear(dim, dim * 3, bias=True, dtype=dtype, device=device)

        self.proj = operations.Linear(inner_dim, dim, dtype=dtype, device=device)

    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = split_qkv(qkv, self.dim_head)
        x = optimized_attention(q.reshape(q.shape[0], q.shape[1], -1), k, v, heads=self.heads)
        return self.proj(x)

class ContextProcessorBlock(nn.Module):
    def __init__(self, context_size, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm1 = operations.LayerNorm(context_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.attn = SelfAttentionContext(context_size, dtype=dtype, device=device, operations=operations)
        self.norm2 = operations.LayerNorm(context_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.mlp = Mlp(in_features=context_size, hidden_features=(context_size * 4), act_layer=lambda: nn.GELU(approximate="tanh"), drop=0, dtype=dtype, device=device, operations=operations)

    def forward(self, x):
        x += self.attn(self.norm1(x))
        x += self.mlp(self.norm2(x))
        return x

class ContextProcessor(nn.Module):
    def __init__(self, context_size, num_layers, dtype=None, device=None, operations=None):
        super().__init__()
        self.layers = torch.nn.ModuleList([ContextProcessorBlock(context_size, dtype=dtype, device=device, operations=operations) for i in range(num_layers)])
        self.norm = operations.LayerNorm(context_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)
        return self.norm(x)

class MMDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        depth: int = 28,
        # hidden_size: Optional[int] = None,
        # num_heads: Optional[int] = None,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = False,
        adm_in_channels: Optional[int] = None,
        context_embedder_config: Optional[Dict] = None,
        compile_core: bool = False,
        use_checkpoint: bool = False,
        register_length: int = 0,
        attn_mode: str = "torch",
        rmsnorm: bool = False,
        scale_mod_only: bool = False,
        swiglu: bool = False,
        out_channels: Optional[int] = None,
        pos_embed_scaling_factor: Optional[float] = None,
        pos_embed_offset: Optional[float] = None,
        pos_embed_max_size: Optional[int] = None,
        num_patches = None,
        qk_norm: Optional[str] = None,
        qkv_bias: bool = True,
        context_processor_layers = None,
        x_block_self_attn: bool = False,
        x_block_self_attn_layers: Optional[List[int]] = [],
        context_size = 4096,
        num_blocks = None,
        final_layer = True,
        skip_blocks = False,
        dtype = None, #TODO
        device = None,
        operations = None,
    ):
        super().__init__()
        self.dtype = dtype
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        default_out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = default(out_channels, default_out_channels)
        self.patch_size = patch_size
        self.pos_embed_scaling_factor = pos_embed_scaling_factor
        self.pos_embed_offset = pos_embed_offset
        self.pos_embed_max_size = pos_embed_max_size
        self.x_block_self_attn_layers = x_block_self_attn_layers

        # hidden_size = default(hidden_size, 64 * depth)
        # num_heads = default(num_heads, hidden_size // 64)

        # apply magic --> this defines a head_size of 64
        self.hidden_size = 64 * depth
        num_heads = depth
        if num_blocks is None:
            num_blocks = depth

        self.depth = depth
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(
            input_size,
            patch_size,
            in_channels,
            self.hidden_size,
            bias=True,
            strict_img_size=self.pos_embed_max_size is None,
            dtype=dtype,
            device=device,
            operations=operations
        )
        self.t_embedder = TimestepEmbedder(self.hidden_size, dtype=dtype, device=device, operations=operations)

        self.y_embedder = None
        if adm_in_channels is not None:
            assert isinstance(adm_in_channels, int)
            self.y_embedder = VectorEmbedder(adm_in_channels, self.hidden_size, dtype=dtype, device=device, operations=operations)

        if context_processor_layers is not None:
            self.context_processor = ContextProcessor(context_size, context_processor_layers, dtype=dtype, device=device, operations=operations)
        else:
            self.context_processor = None

        self.context_embedder = nn.Identity()
        if context_embedder_config is not None:
            if context_embedder_config["target"] == "torch.nn.Linear":
                self.context_embedder = operations.Linear(**context_embedder_config["params"], dtype=dtype, device=device)

        self.register_length = register_length
        if self.register_length > 0:
            self.register = nn.Parameter(torch.randn(1, register_length, self.hidden_size, dtype=dtype, device=device))

        # num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        # just use a buffer already
        if num_patches is not None:
            self.register_buffer(
                "pos_embed",
                torch.empty(1, num_patches, self.hidden_size, dtype=dtype, device=device),
            )
        else:
            self.pos_embed = None

        self.use_checkpoint = use_checkpoint
        if not skip_blocks:
            self.joint_blocks = nn.ModuleList(
                [
                    JointBlock(
                        self.hidden_size,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        attn_mode=attn_mode,
                        pre_only=(i == num_blocks - 1) and final_layer,
                        rmsnorm=rmsnorm,
                        scale_mod_only=scale_mod_only,
                        swiglu=swiglu,
                        qk_norm=qk_norm,
                        x_block_self_attn=(i in self.x_block_self_attn_layers) or x_block_self_attn,
                        dtype=dtype,
                        device=device,
                        operations=operations,
                    )
                    for i in range(num_blocks)
                ]
            )

        if final_layer:
            self.final_layer = FinalLayer(self.hidden_size, patch_size, self.out_channels, dtype=dtype, device=device, operations=operations)

        if compile_core:
            assert False
            self.forward_core_with_concat = torch.compile(self.forward_core_with_concat)

    def cropped_pos_embed(self, hw, device=None):
        p = self.x_embedder.patch_size[0]
        h, w = hw
        # patched size
        h = (h + 1) // p
        w = (w + 1) // p
        if self.pos_embed is None:
            return get_2d_sincos_pos_embed_torch(self.hidden_size, w, h, device=device)
        assert self.pos_embed_max_size is not None
        assert h <= self.pos_embed_max_size, (h, self.pos_embed_max_size)
        assert w <= self.pos_embed_max_size, (w, self.pos_embed_max_size)
        top = (self.pos_embed_max_size - h) // 2
        left = (self.pos_embed_max_size - w) // 2
        spatial_pos_embed = rearrange(
            self.pos_embed,
            "1 (h w) c -> 1 h w c",
            h=self.pos_embed_max_size,
            w=self.pos_embed_max_size,
        )
        spatial_pos_embed = spatial_pos_embed[:, top : top + h, left : left + w, :]
        spatial_pos_embed = rearrange(spatial_pos_embed, "1 h w c -> 1 (h w) c")
        # print(spatial_pos_embed, top, left, h, w)
        # # t = get_2d_sincos_pos_embed_torch(self.hidden_size, w, h, 7.875, 7.875, device=device) #matches exactly for 1024 res
        # t = get_2d_sincos_pos_embed_torch(self.hidden_size, w, h, 7.5, 7.5, device=device) #scales better
        # # print(t)
        # return t
        return spatial_pos_embed

    def unpatchify(self, x, hw=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        if hw is None:
            h = w = int(x.shape[1] ** 0.5)
        else:
            h, w = hw
            h = (h + 1) // p
            w = (w + 1) // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward_core_with_concat(
        self,
        x: torch.Tensor,
        c_mod: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        control = None,
        transformer_options = {},
    ) -> torch.Tensor:
        patches_replace = transformer_options.get("patches_replace", {})
        if self.register_length > 0:
            context = torch.cat(
                (
                    repeat(self.register, "1 ... -> b ...", b=x.shape[0]),
                    default(context, torch.Tensor([]).type_as(x)),
                ),
                1,
            )

        # context is B, L', D
        # x is B, L, D
        blocks_replace = patches_replace.get("dit", {})
        blocks = len(self.joint_blocks)
        for i in range(blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["txt"], out["img"] = self.joint_blocks[i](args["txt"], args["img"], c=args["vec"], transformer_options=args["transformer_options"])
                    return out

                out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": c_mod, "transformer_options": transformer_options}, {"original_block": block_wrap})
                context = out["txt"]
                x = out["img"]
            else:
                context, x = self.joint_blocks[i](
                    context,
                    x,
                    c=c_mod,
                    use_checkpoint=self.use_checkpoint,
                    transformer_options=transformer_options,
                )
            if control is not None:
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        x += add

        x = self.final_layer(x, c_mod)  # (N, T, patch_size ** 2 * out_channels)
        return x

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        control = None,
        transformer_options = {},
    ) -> torch.Tensor:
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """

        if self.context_processor is not None:
            context = self.context_processor(context)

        hw = x.shape[-2:]
        x = self.x_embedder(x) + comfy.ops.cast_to_input(self.cropped_pos_embed(hw, device=x.device), x)
        c = self.t_embedder(t, dtype=x.dtype)  # (N, D)
        if y is not None and self.y_embedder is not None:
            y = self.y_embedder(y)  # (N, D)
            c = c + y  # (N, D)

        if context is not None:
            context = self.context_embedder(context)

        x = self.forward_core_with_concat(x, c, context, control, transformer_options)

        x = self.unpatchify(x, hw=hw)  # (N, out_channels, H, W)
        return x[:,:,:hw[-2],:hw[-1]]


class OpenAISignatureMMDITWrapper(MMDiT):
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        control = None,
        transformer_options = {},
        **kwargs,
    ) -> torch.Tensor:
        return super().forward(x, timesteps, context=context, y=y, control=control, transformer_options=transformer_options)

