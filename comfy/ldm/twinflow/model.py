"""
TwinFlow-Z-Image custom model architecture for ComfyUI.
Based on the Lumina-Image 2.0 / Z-Image architecture.
Supports the unique dual timestep embedding architecture of TwinFlow.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import comfy.ldm.common_dit

from comfy.ldm.modules.diffusionmodules.mmdit import TimestepEmbedder
from comfy.ldm.modules.attention import optimized_attention_masked
from comfy.ldm.flux.layers import EmbedND
from comfy.ldm.flux.math import apply_rope
import comfy.patcher_extension


def clamp_fp16(x):
    if x.dtype == torch.float16:
        return torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
    return x


def modulate(x, scale):
    return x * (1 + scale.unsqueeze(1))


class JointAttention(nn.Module):
    """Multi-head attention module with combined QKV weights."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int],
        qk_norm: bool,
        out_bias: bool = False,
        operation_settings={},
    ):
        super().__init__()
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_local_heads = n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads

        self.qkv = operation_settings.get("operations").Linear(
            dim,
            (n_heads + self.n_kv_heads + self.n_kv_heads) * self.head_dim,
            bias=False,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )
        self.out = operation_settings.get("operations").Linear(
            n_heads * self.head_dim,
            dim,
            bias=out_bias,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )

        if qk_norm:
            self.q_norm = operation_settings.get("operations").RMSNorm(
                self.head_dim,
                elementwise_affine=True,
                device=operation_settings.get("device"),
                dtype=operation_settings.get("dtype"),
            )
            self.k_norm = operation_settings.get("operations").RMSNorm(
                self.head_dim,
                elementwise_affine=True,
                device=operation_settings.get("device"),
                dtype=operation_settings.get("dtype"),
            )
        else:
            self.q_norm = self.k_norm = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        transformer_options={},
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        xq, xk, xv = torch.split(
            self.qkv(x),
            [
                self.n_local_heads * self.head_dim,
                self.n_local_kv_heads * self.head_dim,
                self.n_local_kv_heads * self.head_dim,
            ],
            dim=-1,
        )
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq, xk = apply_rope(xq, xk, freqs_cis)

        n_rep = self.n_local_heads // self.n_local_kv_heads
        if n_rep > 1:
            xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

        output = optimized_attention_masked(
            xq.movedim(1, 2),
            xk.movedim(1, 2),
            xv.movedim(1, 2),
            self.n_local_heads,
            x_mask,
            skip_reshape=True,
            transformer_options=transformer_options,
        )

        return self.out(output)


class FeedForward(nn.Module):
    """Feed-forward module with SiLU gating."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        operation_settings={},
    ):
        super().__init__()
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = operation_settings.get("operations").Linear(
            dim,
            hidden_dim,
            bias=False,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )
        self.w2 = operation_settings.get("operations").Linear(
            hidden_dim,
            dim,
            bias=False,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )
        self.w3 = operation_settings.get("operations").Linear(
            dim,
            hidden_dim,
            bias=False,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )

    def _forward_silu_gating(self, x1, x3):
        return clamp_fp16(F.silu(x1) * x3)

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


class TwinFlowTransformerBlock(nn.Module):
    """Transformer block with adaLN modulation for TwinFlow."""

    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
        norm_eps: float,
        qk_norm: bool,
        modulation=False,
        z_image_modulation=False,
        attn_out_bias=False,
        operation_settings={},
    ) -> None:
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = JointAttention(
            dim,
            n_heads,
            n_kv_heads,
            qk_norm,
            out_bias=attn_out_bias,
            operation_settings=operation_settings,
        )
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            operation_settings=operation_settings,
        )
        self.layer_id = layer_id
        self.attention_norm1 = operation_settings.get("operations").RMSNorm(
            dim,
            eps=norm_eps,
            elementwise_affine=True,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )
        self.ffn_norm1 = operation_settings.get("operations").RMSNorm(
            dim,
            eps=norm_eps,
            elementwise_affine=True,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )
        self.attention_norm2 = operation_settings.get("operations").RMSNorm(
            dim,
            eps=norm_eps,
            elementwise_affine=True,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )
        self.ffn_norm2 = operation_settings.get("operations").RMSNorm(
            dim,
            eps=norm_eps,
            elementwise_affine=True,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )

        if z_image_modulation:
            self.adaLN_modulation = nn.Sequential(
                operation_settings.get("operations").Linear(
                    min(dim, 256),
                    4 * dim,
                    bias=True,
                    device=operation_settings.get("device"),
                    dtype=operation_settings.get("dtype"),
                ),
            )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
        transformer_options={},
    ):
        if adaln_input is not None:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(4, dim=1)

            x = x + gate_msa.unsqueeze(1).tanh() * self.attention_norm2(
                clamp_fp16(
                    self.attention(
                        modulate(self.attention_norm1(x), scale_msa),
                        x_mask,
                        freqs_cis,
                        transformer_options=transformer_options,
                    )
                )
            )
            x = x + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(
                clamp_fp16(
                    self.feed_forward(
                        modulate(self.ffn_norm1(x), scale_mlp),
                    )
                )
            )
        else:
            x = x + self.attention_norm2(
                clamp_fp16(
                    self.attention(
                        self.attention_norm1(x),
                        x_mask,
                        freqs_cis,
                        transformer_options=transformer_options,
                    )
                )
            )
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        return x


class FinalLayer(nn.Module):
    """Final layer with LayerNorm and output projection."""

    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        z_image_modulation=False,
        operation_settings={},
    ):
        super().__init__()
        self.norm_final = operation_settings.get("operations").LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )
        self.linear = operation_settings.get("operations").Linear(
            hidden_size,
            patch_size * patch_size * out_channels,
            bias=True,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )

        min_mod = 256 if z_image_modulation else 1024

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            operation_settings.get("operations").Linear(
                min(hidden_size, min_mod),
                hidden_size,
                bias=True,
                device=operation_settings.get("device"),
                dtype=operation_settings.get("dtype"),
            ),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        scale = self.adaLN_modulation(c)
        x = modulate(self.norm_final(x), scale)
        x = self.linear(x)
        return x


class TwinFlowZImageTransformer(nn.Module):
    """
    TwinFlow-Z-Image transformer model.

    This custom architecture handles dual timestep embeddings
    (t_embedder and t_embedder_2), the primary TwinFlow distinction.
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        dim: int = 3840,
        n_layers: int = 30,
        n_refiner_layers: int = 2,
        n_heads: int = 30,
        n_kv_heads: Optional[int] = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: float = 2.6666666666666665,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
        cap_feat_dim: int = 2560,
        axes_dims: List[int] = (32, 48, 48),
        axes_lens: List[int] = (1, 1536, 512, 512),
        rope_theta: float = 256.0,
        z_image_modulation: bool = True,
        time_scale: float = 1000.0,
        pad_tokens_multiple=None,
        clip_text_dim=None,
        image_model=None,
        device=None,
        dtype=None,
        operations=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        operation_settings = {
            "operations": operations,
            "device": device,
            "dtype": dtype,
        }
        self.time_embed_dim = 256 if z_image_modulation else min(dim, 1024)
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.time_scale = time_scale
        self.pad_tokens_multiple = pad_tokens_multiple

        self.x_embedder = operation_settings.get("operations").Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=dim,
            bias=True,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )

        self.t_embedder = TimestepEmbedder(
            min(dim, 1024),
            output_size=self.time_embed_dim if z_image_modulation else None,
            **operation_settings,
        )
        self.t_embedder_2 = TimestepEmbedder(
            min(dim, 1024),
            output_size=self.time_embed_dim if z_image_modulation else None,
            **operation_settings,
        )

        self.noise_refiner = nn.ModuleList(
            [
                TwinFlowTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                    z_image_modulation=z_image_modulation,
                    operation_settings=operation_settings,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )

        self.context_refiner = nn.ModuleList(
            [
                TwinFlowTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                    operation_settings=operation_settings,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )

        self.cap_embedder = nn.Sequential(
            operation_settings.get("operations").RMSNorm(
                cap_feat_dim,
                eps=norm_eps,
                elementwise_affine=True,
                device=operation_settings.get("device"),
                dtype=operation_settings.get("dtype"),
            ),
            operation_settings.get("operations").Linear(
                cap_feat_dim,
                dim,
                bias=True,
                device=operation_settings.get("device"),
                dtype=operation_settings.get("dtype"),
            ),
        )

        self.clip_text_pooled_proj = None
        if clip_text_dim is not None:
            self.clip_text_dim = clip_text_dim
            self.clip_text_pooled_proj = nn.Sequential(
                operation_settings.get("operations").RMSNorm(
                    clip_text_dim,
                    eps=norm_eps,
                    elementwise_affine=True,
                    device=operation_settings.get("device"),
                    dtype=operation_settings.get("dtype"),
                ),
                operation_settings.get("operations").Linear(
                    clip_text_dim,
                    clip_text_dim,
                    bias=True,
                    device=operation_settings.get("device"),
                    dtype=operation_settings.get("dtype"),
                ),
            )
            self.clip_text_concat_proj = nn.Sequential(
                operation_settings.get("operations").RMSNorm(
                    clip_text_dim + self.time_embed_dim,
                    eps=norm_eps,
                    elementwise_affine=True,
                    device=operation_settings.get("device"),
                    dtype=operation_settings.get("dtype"),
                ),
                operation_settings.get("operations").Linear(
                    clip_text_dim + self.time_embed_dim,
                    self.time_embed_dim,
                    bias=True,
                    device=operation_settings.get("device"),
                    dtype=operation_settings.get("dtype"),
                ),
            )

        self.layers = nn.ModuleList(
            [
                TwinFlowTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    z_image_modulation=z_image_modulation,
                    attn_out_bias=False,
                    operation_settings=operation_settings,
                )
                for layer_id in range(n_layers)
            ]
        )

        self.final_layer = FinalLayer(
            dim,
            patch_size,
            self.out_channels,
            z_image_modulation=z_image_modulation,
            operation_settings=operation_settings,
        )

        if self.pad_tokens_multiple is not None:
            self.x_pad_token = nn.Parameter(torch.zeros((1, dim), device=device, dtype=dtype))
            self.cap_pad_token = nn.Parameter(torch.zeros((1, dim), device=device, dtype=dtype))

        assert (dim // n_heads) == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.rope_embedder = EmbedND(dim=dim // n_heads, theta=rope_theta, axes_dim=axes_dims)
        self.dim = dim
        self.n_heads = n_heads

    def _compute_twinflow_adaln(self, t: torch.Tensor, x_dtype: torch.dtype, transformer_options={}):
        """
        Compute TwinFlow adaLN input.

        If `target_timestep` is provided in transformer options, apply the
        TwinFlow delta-time conditioning:
          t_emb + t_embedder_2((target - t) * time_scale) * abs(target - t)
        otherwise fallback to the baseline additive embedding.
        """
        t_emb = self.t_embedder(t * self.time_scale, dtype=x_dtype)
        target_timestep = transformer_options.get("target_timestep", None)
        if target_timestep is None:
            t_emb_2 = self.t_embedder_2(t * self.time_scale, dtype=x_dtype)
            return t_emb + t_emb_2

        target_t = torch.as_tensor(target_timestep, device=t.device, dtype=t.dtype)
        if target_t.ndim == 0:
            target_t = target_t.expand_as(t)

        # If values look scaled (roughly sigma/timestep in [0..1000]), normalize.
        t_abs_max = float(t.detach().abs().max().item()) if t.numel() else 0.0
        tt_abs_max = float(target_t.detach().abs().max().item()) if target_t.numel() else 0.0
        scaled_domain = (max(t_abs_max, tt_abs_max) > 2.0) and (self.time_scale > 2.0)
        if scaled_domain:
            t_norm = t / self.time_scale
            tt_norm = target_t / self.time_scale
        else:
            t_norm = t
            tt_norm = target_t

        delta_abs = (t_norm - tt_norm).abs().unsqueeze(1).to(t_emb.dtype)
        diff_in = (tt_norm - t_norm) * self.time_scale
        t_emb_2 = self.t_embedder_2(diff_in, dtype=x_dtype)
        return t_emb + t_emb_2 * delta_abs

    def unpatchify(
        self,
        x: torch.Tensor,
        img_size: List[Tuple[int, int]],
        cap_size: List[int],
        return_tensor=False,
    ) -> List[torch.Tensor]:
        pH = pW = self.patch_size
        imgs = []
        for i in range(x.size(0)):
            H, W = img_size[i]
            begin = cap_size[i]
            end = begin + (H // pH) * (W // pW)
            imgs.append(
                x[i][begin:end]
                .view(H // pH, W // pW, pH, pW, self.out_channels)
                .permute(4, 0, 2, 1, 3)
                .flatten(3, 4)
                .flatten(1, 2)
            )

        if return_tensor:
            imgs = torch.stack(imgs, dim=0)
        return imgs

    def patchify_and_embed(
        self,
        x: List[torch.Tensor] | torch.Tensor,
        cap_feats: torch.Tensor,
        cap_mask: torch.Tensor,
        t: torch.Tensor,
        num_tokens,
        transformer_options={},
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]], List[int], torch.Tensor]:
        bsz = len(x)
        pH = pW = self.patch_size
        device = x[0].device

        cap_pos_ids = torch.zeros(bsz, cap_feats.shape[1], 3, dtype=torch.float32, device=device)
        cap_pos_ids[:, :, 0] = torch.arange(cap_feats.shape[1], dtype=torch.float32, device=device) + 1.0

        B, C, H, W = x.shape
        x = self.x_embedder(
            x.view(B, C, H // pH, pH, W // pW, pW).permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2)
        )

        rope_options = transformer_options.get("rope_options", {})
        h_scale = rope_options.get("scale_y", 1.0)
        w_scale = rope_options.get("scale_x", 1.0)
        h_start = rope_options.get("shift_y", 0.0)
        w_start = rope_options.get("shift_x", 0.0)

        H_tokens, W_tokens = H // pH, W // pW
        x_pos_ids = torch.zeros((bsz, x.shape[1], 3), dtype=torch.float32, device=device)
        x_pos_ids[:, :, 0] = cap_feats.shape[1] + 1
        x_pos_ids[:, :, 1] = (
            torch.arange(H_tokens, dtype=torch.float32, device=device) * h_scale + h_start
        ).view(-1, 1).repeat(1, W_tokens).flatten()
        x_pos_ids[:, :, 2] = (
            torch.arange(W_tokens, dtype=torch.float32, device=device) * w_scale + w_start
        ).view(1, -1).repeat(H_tokens, 1).flatten()

        x_pad_extra = 0
        if self.pad_tokens_multiple is not None:
            x_pad_extra = (-x.shape[1]) % self.pad_tokens_multiple
            x = torch.cat(
                (
                    x,
                    self.x_pad_token.to(device=x.device, dtype=x.dtype, copy=True).unsqueeze(0).repeat(x.shape[0], x_pad_extra, 1),
                ),
                dim=1,
            )
            x_pos_ids = torch.nn.functional.pad(x_pos_ids, (0, 0, 0, x_pad_extra))

        cap_pad_extra = 0
        if self.pad_tokens_multiple is not None:
            cap_pad_extra = (-cap_feats.shape[1]) % self.pad_tokens_multiple
            cap_feats = torch.cat(
                (
                    cap_feats,
                    self.cap_pad_token.to(device=cap_feats.device, dtype=cap_feats.dtype, copy=True)
                    .unsqueeze(0)
                    .repeat(cap_feats.shape[0], cap_pad_extra, 1),
                ),
                dim=1,
            )
            cap_pos_ids = torch.nn.functional.pad(cap_pos_ids, (0, 0, 0, cap_pad_extra), value=0)
            if cap_mask is not None and cap_pad_extra > 0:
                cap_mask = torch.nn.functional.pad(cap_mask, (0, cap_pad_extra), value=0)

        freqs_cis = self.rope_embedder(torch.cat((cap_pos_ids, x_pos_ids), dim=1)).movedim(1, 2)

        for layer in self.context_refiner:
            cap_feats = layer(
                cap_feats,
                cap_mask,
                freqs_cis[:, : cap_pos_ids.shape[1]],
                transformer_options=transformer_options,
            )

        padded_img_mask = None
        for _, layer in enumerate(self.noise_refiner):
            x = layer(
                x,
                padded_img_mask,
                freqs_cis[:, cap_pos_ids.shape[1] :],
                t,
                transformer_options=transformer_options,
            )

        padded_full_embed = torch.cat((cap_feats, x), dim=1)
        if cap_mask is not None:
            cap_mask_bool = cap_mask if cap_mask.dtype == torch.bool else cap_mask > 0
            img_mask = torch.ones((bsz, x.shape[1]), device=cap_mask.device, dtype=torch.bool)
            if x_pad_extra > 0:
                img_mask[:, -x_pad_extra:] = False
            mask = torch.cat((cap_mask_bool, img_mask), dim=1)
        else:
            mask = None
        img_sizes = [(H, W)] * bsz
        l_effective_cap_len = [cap_feats.shape[1]] * bsz

        return padded_full_embed, mask, img_sizes, l_effective_cap_len, freqs_cis

    def forward(self, x, timesteps, context, num_tokens, attention_mask=None, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                kwargs.get("transformer_options", {}),
            ),
        ).execute(x, timesteps, context, num_tokens, attention_mask, **kwargs)

    def _forward(
        self,
        x,
        timesteps,
        context,
        num_tokens,
        attention_mask=None,
        transformer_options=None,
        **kwargs,
    ):
        if transformer_options is None:
            transformer_options = {}

        t = 1.0 - timesteps

        adaln_input = self._compute_twinflow_adaln(t, x.dtype, transformer_options=transformer_options)

        cap_feats = context
        cap_mask = attention_mask
        bs, c, h, w = x.shape
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))

        cap_feats = self.cap_embedder(cap_feats)

        if self.clip_text_pooled_proj is not None:
            pooled = kwargs.get("clip_text_pooled", None)
            if pooled is not None:
                pooled = self.clip_text_pooled_proj(pooled)
            else:
                pooled = torch.zeros((x.shape[0], self.clip_text_dim), device=x.device, dtype=x.dtype)
            adaln_input = torch.cat((adaln_input, pooled), dim=-1)
            adaln_input = self.clip_text_concat_proj(adaln_input)

        img, mask, img_size, cap_size, freqs_cis = self.patchify_and_embed(
            x,
            cap_feats,
            cap_mask,
            adaln_input,
            num_tokens,
            transformer_options=transformer_options,
        )
        freqs_cis = freqs_cis.to(img.device)

        transformer_options["total_blocks"] = len(self.layers)
        transformer_options["block_type"] = "double"

        for i, layer in enumerate(self.layers):
            transformer_options["block_index"] = i
            img = layer(img, mask, freqs_cis, adaln_input, transformer_options=transformer_options)

        img = self.final_layer(img, adaln_input)
        img = self.unpatchify(
            img,
            img_size,
            cap_size,
            return_tensor=isinstance(x, torch.Tensor),
        )[:, :, :h, :w]

        return -img
