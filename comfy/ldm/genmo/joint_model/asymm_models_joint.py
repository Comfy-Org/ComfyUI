#original code from https://github.com/genmoai/models under apache 2.0 license
#adapted to ComfyUI

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# from flash_attn import flash_attn_varlen_qkvpacked_func
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention

from .layers import (
    FeedForward,
    PatchEmbed,
    TimestepEmbedder,
)

from .rope_mixed import (
    compute_mixed_rotation,
    create_position_matrix,
)
from .temporal_rope import apply_rotary_emb_qk_real
from .utils import (
    AttentionPool,
    modulate,
)

import comfy.ldm.common_dit
import comfy.ops


def modulated_rmsnorm(x, scale, eps=1e-6):
    # Normalize and modulate
    x_normed = comfy.ldm.common_dit.rms_norm(x, eps=eps)
    x_modulated = x_normed * (1 + scale.unsqueeze(1))

    return x_modulated


def residual_tanh_gated_rmsnorm(x, x_res, gate, eps=1e-6):
    # Apply tanh to gate
    tanh_gate = torch.tanh(gate).unsqueeze(1)

    # Normalize and apply gated scaling
    x_normed = comfy.ldm.common_dit.rms_norm(x_res, eps=eps) * tanh_gate

    # Apply residual connection
    output = x + x_normed

    return output

class AsymmetricAttention(nn.Module):
    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        update_y: bool = True,
        out_bias: bool = True,
        attend_to_padding: bool = False,
        softmax_scale: Optional[float] = None,
        device: Optional[torch.device] = None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.num_heads = num_heads
        self.head_dim = dim_x // num_heads
        self.attn_drop = attn_drop
        self.update_y = update_y
        self.attend_to_padding = attend_to_padding
        self.softmax_scale = softmax_scale
        if dim_x % num_heads != 0:
            raise ValueError(
                f"dim_x={dim_x} should be divisible by num_heads={num_heads}"
            )

        # Input layers.
        self.qkv_bias = qkv_bias
        self.qkv_x = operations.Linear(dim_x, 3 * dim_x, bias=qkv_bias, device=device, dtype=dtype)
        # Project text features to match visual features (dim_y -> dim_x)
        self.qkv_y = operations.Linear(dim_y, 3 * dim_x, bias=qkv_bias, device=device, dtype=dtype)

        # Query and key normalization for stability.
        assert qk_norm
        self.q_norm_x = operations.RMSNorm(self.head_dim, eps=1e-5, device=device, dtype=dtype)
        self.k_norm_x = operations.RMSNorm(self.head_dim, eps=1e-5, device=device, dtype=dtype)
        self.q_norm_y = operations.RMSNorm(self.head_dim, eps=1e-5, device=device, dtype=dtype)
        self.k_norm_y = operations.RMSNorm(self.head_dim, eps=1e-5, device=device, dtype=dtype)

        # Output layers. y features go back down from dim_x -> dim_y.
        self.proj_x = operations.Linear(dim_x, dim_x, bias=out_bias, device=device, dtype=dtype)
        self.proj_y = (
            operations.Linear(dim_x, dim_y, bias=out_bias, device=device, dtype=dtype)
            if update_y
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,  # (B, N, dim_x)
        y: torch.Tensor,  # (B, L, dim_y)
        scale_x: torch.Tensor,  # (B, dim_x), modulation for pre-RMSNorm.
        scale_y: torch.Tensor,  # (B, dim_y), modulation for pre-RMSNorm.
        crop_y,
        transformer_options={},
        **rope_rotation,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rope_cos = rope_rotation.get("rope_cos")
        rope_sin = rope_rotation.get("rope_sin")
        # Pre-norm for visual features
        x = modulated_rmsnorm(x, scale_x)  # (B, M, dim_x) where M = N / cp_group_size

        # Process visual features
        # qkv_x = self.qkv_x(x)  # (B, M, 3 * dim_x)
        # assert qkv_x.dtype == torch.bfloat16
        # qkv_x = all_to_all_collect_tokens(
        #     qkv_x, self.num_heads
        # )  # (3, B, N, local_h, head_dim)

        # Process text features
        y = modulated_rmsnorm(y, scale_y)  # (B, L, dim_y)
        q_y, k_y, v_y = self.qkv_y(y).view(y.shape[0], y.shape[1], 3, self.num_heads, -1).unbind(2)  # (B, N, local_h, head_dim)

        q_y = self.q_norm_y(q_y)
        k_y = self.k_norm_y(k_y)

        # Split qkv_x into q, k, v
        q_x, k_x, v_x = self.qkv_x(x).view(x.shape[0], x.shape[1], 3, self.num_heads, -1).unbind(2)  # (B, N, local_h, head_dim)
        q_x = self.q_norm_x(q_x)
        q_x = apply_rotary_emb_qk_real(q_x, rope_cos, rope_sin)
        k_x = self.k_norm_x(k_x)
        k_x = apply_rotary_emb_qk_real(k_x, rope_cos, rope_sin)

        q = torch.cat([q_x, q_y[:, :crop_y]], dim=1).transpose(1, 2)
        k = torch.cat([k_x, k_y[:, :crop_y]], dim=1).transpose(1, 2)
        v = torch.cat([v_x, v_y[:, :crop_y]], dim=1).transpose(1, 2)

        seq_x, seq_y = q_x.shape[1], q_y.shape[1]
        del q_x, k_x, v_x, q_y, k_y, v_y
        q, k, v = AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v)
        xy = optimized_attention(q,
                                 k,
                                 v, self.num_heads, skip_reshape=True, preferred_attention=self.comfy_attention, transformer_options=transformer_options)

        x, y = torch.tensor_split(xy, (seq_x,), dim=1)
        x = self.proj_x(x)
        o = torch.zeros(y.shape[0], seq_y, y.shape[-1], device=y.device, dtype=y.dtype)
        o[:, :y.shape[1]] = y

        y = self.proj_y(o)
        # print("ox", x)
        # print("oy", y)
        return x, y


class AsymmetricJointBlock(nn.Module):
    def __init__(
        self,
        hidden_size_x: int,
        hidden_size_y: int,
        num_heads: int,
        *,
        mlp_ratio_x: float = 8.0,  # Ratio of hidden size to d_model for MLP for visual tokens.
        mlp_ratio_y: float = 4.0,  # Ratio of hidden size to d_model for MLP for text tokens.
        update_y: bool = True,  # Whether to update text tokens in this block.
        device: Optional[torch.device] = None,
        dtype=None,
        operations=None,
        **block_kwargs,
    ):
        super().__init__()
        self.update_y = update_y
        self.hidden_size_x = hidden_size_x
        self.hidden_size_y = hidden_size_y
        self.mod_x = operations.Linear(hidden_size_x, 4 * hidden_size_x, device=device, dtype=dtype)
        if self.update_y:
            self.mod_y = operations.Linear(hidden_size_x, 4 * hidden_size_y, device=device, dtype=dtype)
        else:
            self.mod_y = operations.Linear(hidden_size_x, hidden_size_y, device=device, dtype=dtype)

        # Self-attention:
        self.attn = AsymmetricAttention(
            hidden_size_x,
            hidden_size_y,
            num_heads=num_heads,
            update_y=update_y,
            device=device,
            dtype=dtype,
            operations=operations,
            **block_kwargs,
        )

        # MLP.
        mlp_hidden_dim_x = int(hidden_size_x * mlp_ratio_x)
        assert mlp_hidden_dim_x == int(1536 * 8)
        self.mlp_x = FeedForward(
            in_features=hidden_size_x,
            hidden_size=mlp_hidden_dim_x,
            multiple_of=256,
            ffn_dim_multiplier=None,
            device=device,
            dtype=dtype,
            operations=operations,
        )

        # MLP for text not needed in last block.
        if self.update_y:
            mlp_hidden_dim_y = int(hidden_size_y * mlp_ratio_y)
            self.mlp_y = FeedForward(
                in_features=hidden_size_y,
                hidden_size=mlp_hidden_dim_y,
                multiple_of=256,
                ffn_dim_multiplier=None,
                device=device,
                dtype=dtype,
                operations=operations,
            )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        y: torch.Tensor,
        transformer_options={},
        **attn_kwargs,
    ):
        """Forward pass of a block.

        Args:
            x: (B, N, dim) tensor of visual tokens
            c: (B, dim) tensor of conditioned features
            y: (B, L, dim) tensor of text tokens
            num_frames: Number of frames in the video. N = num_frames * num_spatial_tokens

        Returns:
            x: (B, N, dim) tensor of visual tokens after block
            y: (B, L, dim) tensor of text tokens after block
        """
        N = x.size(1)

        c = F.silu(c)
        mod_x = self.mod_x(c)
        scale_msa_x, gate_msa_x, scale_mlp_x, gate_mlp_x = mod_x.chunk(4, dim=1)

        mod_y = self.mod_y(c)
        if self.update_y:
            scale_msa_y, gate_msa_y, scale_mlp_y, gate_mlp_y = mod_y.chunk(4, dim=1)
        else:
            scale_msa_y = mod_y

        # Self-attention block.
        x_attn, y_attn = self.attn(
            x,
            y,
            scale_x=scale_msa_x,
            scale_y=scale_msa_y,
            transformer_options=transformer_options,
            **attn_kwargs,
        )

        assert x_attn.size(1) == N
        x = residual_tanh_gated_rmsnorm(x, x_attn, gate_msa_x)
        if self.update_y:
            y = residual_tanh_gated_rmsnorm(y, y_attn, gate_msa_y)

        # MLP block.
        x = self.ff_block_x(x, scale_mlp_x, gate_mlp_x)
        if self.update_y:
            y = self.ff_block_y(y, scale_mlp_y, gate_mlp_y)

        return x, y

    def ff_block_x(self, x, scale_x, gate_x):
        x_mod = modulated_rmsnorm(x, scale_x)
        x_res = self.mlp_x(x_mod)
        x = residual_tanh_gated_rmsnorm(x, x_res, gate_x)  # Sandwich norm
        return x

    def ff_block_y(self, y, scale_y, gate_y):
        y_mod = modulated_rmsnorm(y, scale_y)
        y_res = self.mlp_y(y_mod)
        y = residual_tanh_gated_rmsnorm(y, y_res, gate_y)  # Sandwich norm
        return y


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(
        self,
        hidden_size,
        patch_size,
        out_channels,
        device: Optional[torch.device] = None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        self.norm_final = operations.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype
        )
        self.mod = operations.Linear(hidden_size, 2 * hidden_size, device=device, dtype=dtype)
        self.linear = operations.Linear(
            hidden_size, patch_size * patch_size * out_channels, device=device, dtype=dtype
        )

    def forward(self, x, c):
        c = F.silu(c)
        shift, scale = self.mod(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class AsymmDiTJoint(nn.Module):
    """
    Diffusion model with a Transformer backbone.

    Ingests text embeddings instead of a label.
    """

    def __init__(
        self,
        *,
        patch_size=2,
        in_channels=4,
        hidden_size_x=1152,
        hidden_size_y=1152,
        depth=48,
        num_heads=16,
        mlp_ratio_x=8.0,
        mlp_ratio_y=4.0,
        use_t5: bool = False,
        t5_feat_dim: int = 4096,
        t5_token_length: int = 256,
        learn_sigma=True,
        patch_embed_bias: bool = True,
        timestep_mlp_bias: bool = True,
        attend_to_padding: bool = False,
        timestep_scale: Optional[float] = None,
        use_extended_posenc: bool = False,
        posenc_preserve_area: bool = False,
        rope_theta: float = 10000.0,
        image_model=None,
        device: Optional[torch.device] = None,
        dtype=None,
        operations=None,
        **block_kwargs,
    ):
        super().__init__()

        self.dtype = dtype
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size_x = hidden_size_x
        self.hidden_size_y = hidden_size_y
        self.head_dim = (
            hidden_size_x // num_heads
        )  # Head dimension and count is determined by visual.
        self.attend_to_padding = attend_to_padding
        self.use_extended_posenc = use_extended_posenc
        self.posenc_preserve_area = posenc_preserve_area
        self.use_t5 = use_t5
        self.t5_token_length = t5_token_length
        self.t5_feat_dim = t5_feat_dim
        self.rope_theta = (
            rope_theta  # Scaling factor for frequency computation for temporal RoPE.
        )

        self.x_embedder = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=hidden_size_x,
            bias=patch_embed_bias,
            dtype=dtype,
            device=device,
            operations=operations
        )
        # Conditionings
        # Timestep
        self.t_embedder = TimestepEmbedder(
            hidden_size_x, bias=timestep_mlp_bias, timestep_scale=timestep_scale, dtype=dtype, device=device, operations=operations
        )

        if self.use_t5:
            # Caption Pooling (T5)
            self.t5_y_embedder = AttentionPool(
                t5_feat_dim, num_heads=8, output_dim=hidden_size_x, dtype=dtype, device=device, operations=operations
            )

            # Dense Embedding Projection (T5)
            self.t5_yproj = operations.Linear(
                t5_feat_dim, hidden_size_y, bias=True, dtype=dtype, device=device
            )

        # Initialize pos_frequencies as an empty parameter.
        self.pos_frequencies = nn.Parameter(
            torch.empty(3, self.num_heads, self.head_dim // 2, dtype=dtype, device=device)
        )

        assert not self.attend_to_padding

        # for depth 48:
        #  b =  0: AsymmetricJointBlock, update_y=True
        #  b =  1: AsymmetricJointBlock, update_y=True
        #  ...
        #  b = 46: AsymmetricJointBlock, update_y=True
        #  b = 47: AsymmetricJointBlock, update_y=False. No need to update text features.
        blocks = []
        for b in range(depth):
            # Joint multi-modal block
            update_y = b < depth - 1
            block = AsymmetricJointBlock(
                hidden_size_x,
                hidden_size_y,
                num_heads,
                mlp_ratio_x=mlp_ratio_x,
                mlp_ratio_y=mlp_ratio_y,
                update_y=update_y,
                attend_to_padding=attend_to_padding,
                device=device,
                dtype=dtype,
                operations=operations,
                **block_kwargs,
            )

            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        self.final_layer = FinalLayer(
            hidden_size_x, patch_size, self.out_channels, dtype=dtype, device=device, operations=operations
        )

    def embed_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C=12, T, H, W) tensor of visual tokens

        Returns:
            x: (B, C=3072, N) tensor of visual tokens with positional embedding.
        """
        return self.x_embedder(x)  # Convert BcTHW to BCN

    def prepare(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        t5_feat: torch.Tensor,
        t5_mask: torch.Tensor,
    ):
        """Prepare input and conditioning embeddings."""
        # Visual patch embeddings with positional encoding.
        T, H, W = x.shape[-3:]
        pH, pW = H // self.patch_size, W // self.patch_size
        x = self.embed_x(x)  # (B, N, D), where N = T * H * W / patch_size ** 2
        assert x.ndim == 3

        pH, pW = H // self.patch_size, W // self.patch_size
        N = T * pH * pW
        assert x.size(1) == N
        pos = create_position_matrix(
            T, pH=pH, pW=pW, device=x.device, dtype=torch.float32
        )  # (N, 3)
        rope_cos, rope_sin = compute_mixed_rotation(
            freqs=comfy.ops.cast_to(self.pos_frequencies, dtype=x.dtype, device=x.device), pos=pos
        )  # Each are (N, num_heads, dim // 2)

        c_t = self.t_embedder(1 - sigma, out_dtype=x.dtype)  # (B, D)

        t5_y_pool = self.t5_y_embedder(t5_feat, t5_mask)  # (B, D)

        c = c_t + t5_y_pool

        y_feat = self.t5_yproj(t5_feat)  # (B, L, t5_feat_dim) --> (B, L, D)

        return x, c, y_feat, rope_cos, rope_sin

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: List[torch.Tensor],
        attention_mask: List[torch.Tensor],
        num_tokens=256,
        packed_indices: Dict[str, torch.Tensor] = None,
        rope_cos: torch.Tensor = None,
        rope_sin: torch.Tensor = None,
        control=None, transformer_options={}, **kwargs
    ):
        patches_replace = transformer_options.get("patches_replace", {})
        y_feat = context
        y_mask = attention_mask
        sigma = timestep
        """Forward pass of DiT.

        Args:
            x: (B, C, T, H, W) tensor of spatial inputs (images or latent representations of images)
            sigma: (B,) tensor of noise standard deviations
            y_feat: List((B, L, y_feat_dim) tensor of caption token features. For SDXL text encoders: L=77, y_feat_dim=2048)
            y_mask: List((B, L) boolean tensor indicating which tokens are not padding)
            packed_indices: Dict with keys for Flash Attention. Result of compute_packed_indices.
        """
        B, _, T, H, W = x.shape

        x, c, y_feat, rope_cos, rope_sin = self.prepare(
            x, sigma, y_feat, y_mask
        )
        del y_mask

        blocks_replace = patches_replace.get("dit", {})
        for i, block in enumerate(self.blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(
                                                    args["img"],
                                                    args["vec"],
                                                    args["txt"],
                                                    rope_cos=args["rope_cos"],
                                                    rope_sin=args["rope_sin"],
                                                    crop_y=args["num_tokens"],
                                                    transformer_options=args["transformer_options"]
                                                    )
                    return out
                out = blocks_replace[("double_block", i)]({"img": x, "txt": y_feat, "vec": c, "rope_cos": rope_cos, "rope_sin": rope_sin, "num_tokens": num_tokens, "transformer_options": transformer_options}, {"original_block": block_wrap})
                y_feat = out["txt"]
                x = out["img"]
            else:
                x, y_feat = block(
                    x,
                    c,
                    y_feat,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    crop_y=num_tokens,
                    transformer_options=transformer_options,
                )  # (B, M, D), (B, L, D)
        del y_feat  # Final layers don't use dense text features.

        x = self.final_layer(x, c)  # (B, M, patch_size ** 2 * out_channels)
        x = rearrange(
            x,
            "B (T hp wp) (p1 p2 c) -> B c T (hp p1) (wp p2)",
            T=T,
            hp=H // self.patch_size,
            wp=W // self.patch_size,
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.out_channels,
        )

        return -x
