# https://github.com/jdopensource/JoyAI-Image-Edit (Apache 2.0)
import math
from typing import Optional, Tuple

import comfy_kitchen
import torch
import torch.nn as nn

import comfy.ldm.common_dit
import comfy.ops
import comfy.patcher_extension
from comfy.ldm.lightricks.model import GELU_approx, PixArtAlphaTextProjection, TimestepEmbedding, Timesteps
from comfy.ldm.modules.attention import optimized_attention


class JoyImageModulate(nn.Module):
    def __init__(self, hidden_size: int, factor: int, dtype=None, device=None):
        super().__init__()
        self.factor = factor
        self.modulate_table = nn.Parameter(
            torch.empty(1, factor, hidden_size, dtype=dtype, device=device)
        )

    def forward(self, x: torch.Tensor) -> list:
        if x.ndim != 3:
            x = x.unsqueeze(1)
        table = comfy.ops.cast_to_input(self.modulate_table, x)
        return [o.squeeze(1) for o in (table + x).chunk(self.factor, dim=1)]


class JoyImageFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        inner_dim: int,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.net = nn.ModuleList([
            GELU_approx(dim, inner_dim, dtype=dtype, device=device, operations=operations),
            nn.Identity(),
            operations.Linear(inner_dim, dim, bias=True, dtype=dtype, device=device),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            x = module(x)
        return x


class JoyImageAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-6,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        inner_dim = num_attention_heads * attention_head_dim

        self.img_attn_qkv = operations.Linear(dim, inner_dim * 3, bias=True, dtype=dtype, device=device)
        self.img_attn_q_norm = operations.RMSNorm(attention_head_dim, eps=eps, dtype=dtype, device=device)
        self.img_attn_k_norm = operations.RMSNorm(attention_head_dim, eps=eps, dtype=dtype, device=device)
        self.img_attn_proj = operations.Linear(inner_dim, dim, bias=True, dtype=dtype, device=device)

        self.txt_attn_qkv = operations.Linear(dim, inner_dim * 3, bias=True, dtype=dtype, device=device)
        self.txt_attn_q_norm = operations.RMSNorm(attention_head_dim, eps=eps, dtype=dtype, device=device)
        self.txt_attn_k_norm = operations.RMSNorm(attention_head_dim, eps=eps, dtype=dtype, device=device)
        self.txt_attn_proj = operations.Linear(inner_dim, dim, bias=True, dtype=dtype, device=device)

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        transformer_options=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        heads = self.num_attention_heads

        img_q, img_k, img_v = self.img_attn_qkv(img).chunk(3, dim=-1)
        txt_q, txt_k, txt_v = self.txt_attn_qkv(txt).chunk(3, dim=-1)

        img_q = img_q.unflatten(-1, (heads, -1))
        img_k = img_k.unflatten(-1, (heads, -1))
        img_v = img_v.unflatten(-1, (heads, -1))
        txt_q = txt_q.unflatten(-1, (heads, -1))
        txt_k = txt_k.unflatten(-1, (heads, -1))
        txt_v = txt_v.unflatten(-1, (heads, -1))

        txt_q = self.txt_attn_q_norm(txt_q)
        txt_k = self.txt_attn_k_norm(txt_k)

        img_q_scale, _, img_q_offload_stream = comfy.ops.cast_bias_weight(self.img_attn_q_norm, img_q, offloadable=True)
        img_k_scale, _, img_k_offload_stream = comfy.ops.cast_bias_weight(self.img_attn_k_norm, img_k, offloadable=True)
        img_q, img_k = comfy_kitchen.rms_rope(
            img_q,
            img_k,
            image_rotary_emb,
            img_q_scale,
            img_k_scale,
            self.img_attn_q_norm.eps,
        )
        comfy.ops.uncast_bias_weight(self.img_attn_q_norm, img_q_scale, None, img_q_offload_stream)
        comfy.ops.uncast_bias_weight(self.img_attn_k_norm, img_k_scale, None, img_k_offload_stream)

        joint_q = torch.cat([img_q, txt_q], dim=1)
        joint_k = torch.cat([img_k, txt_k], dim=1)
        joint_v = torch.cat([img_v, txt_v], dim=1)

        joint_q = joint_q.flatten(2, 3)
        joint_k = joint_k.flatten(2, 3)
        joint_v = joint_v.flatten(2, 3)

        joint_out = optimized_attention(joint_q, joint_k, joint_v, heads=heads, transformer_options=transformer_options)

        seq_img = img.shape[1]
        img_out = joint_out[:, :seq_img, :]
        txt_out = joint_out[:, seq_img:, :]

        img_out = self.img_attn_proj(img_out)
        txt_out = self.txt_attn_proj(txt_out)
        return img_out, txt_out


class JoyImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_width_ratio: float = 4.0,
        eps: float = 1e-6,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_width_ratio)

        self.img_mod = JoyImageModulate(dim, factor=6, dtype=dtype, device=device)
        self.img_norm1 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.img_norm2 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.img_mlp = JoyImageFeedForward(dim, inner_dim=mlp_hidden_dim, dtype=dtype, device=device, operations=operations)

        self.txt_mod = JoyImageModulate(dim, factor=6, dtype=dtype, device=device)
        self.txt_norm1 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.txt_norm2 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.txt_mlp = JoyImageFeedForward(dim, inner_dim=mlp_hidden_dim, dtype=dtype, device=device, operations=operations)

        self.attn = JoyImageAttention(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            eps=eps,
            dtype=dtype,
            device=device,
            operations=operations,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        transformer_options=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(temb)
        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(temb)

        img_normed = self.img_norm1(hidden_states)
        txt_normed = self.txt_norm1(encoder_hidden_states)
        img_modulated = img_normed * (1 + img_mod1_scale.unsqueeze(1)) + img_mod1_shift.unsqueeze(1)
        txt_modulated = txt_normed * (1 + txt_mod1_scale.unsqueeze(1)) + txt_mod1_shift.unsqueeze(1)

        img_attn, txt_attn = self.attn(img_modulated, txt_modulated, image_rotary_emb, transformer_options=transformer_options)

        hidden_states = hidden_states + img_attn * img_mod1_gate.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + txt_attn * txt_mod1_gate.unsqueeze(1)

        img_ffn_normed = self.img_norm2(hidden_states)
        txt_ffn_normed = self.txt_norm2(encoder_hidden_states)
        img_ffn_input = img_ffn_normed * (1 + img_mod2_scale.unsqueeze(1)) + img_mod2_shift.unsqueeze(1)
        txt_ffn_input = txt_ffn_normed * (1 + txt_mod2_scale.unsqueeze(1)) + txt_mod2_shift.unsqueeze(1)
        hidden_states = hidden_states + self.img_mlp(img_ffn_input) * img_mod2_gate.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + self.txt_mlp(txt_ffn_input) * txt_mod2_gate.unsqueeze(1)

        return hidden_states, encoder_hidden_states


class JoyImageTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim,
            time_embed_dim=dim,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.act_fn = nn.SiLU()
        self.time_proj = operations.Linear(dim, time_proj_dim, bias=True, dtype=dtype, device=device)
        self.text_embedder = PixArtAlphaTextProjection(
            text_embed_dim, dim, act_fn="gelu_tanh", dtype=dtype, device=device, operations=operations,
        )

    def forward(self, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor):
        timestep = self.timesteps_proj(timestep)
        temb = self.time_embedder(timestep.to(dtype=encoder_hidden_states.dtype)).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))
        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        return temb, timestep_proj, encoder_hidden_states


class JoyImageTransformer3DModel(nn.Module):
    def __init__(
        self,
        patch_size: list = [1, 2, 2],
        in_channels: int = 16,
        out_channels: Optional[int] = None,
        hidden_size: int = 3072,
        num_attention_heads: int = 24,
        text_dim: int = 4096,
        mlp_width_ratio: float = 4.0,
        num_layers: int = 20,
        rope_dim_list: list = [16, 56, 56],
        theta: int = 256,
        image_model=None,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.dtype = dtype
        self.out_channels = out_channels or in_channels
        self.patch_size = list(patch_size)
        self.rope_dim_list = list(rope_dim_list)
        self.theta = theta

        attention_head_dim = hidden_size // num_attention_heads

        self.img_in = operations.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=tuple(self.patch_size),
            stride=tuple(self.patch_size),
            dtype=dtype,
            device=device,
        )

        self.condition_embedder = JoyImageTimeTextImageEmbedding(
            dim=hidden_size,
            time_freq_dim=256,
            time_proj_dim=hidden_size * 6,
            text_embed_dim=text_dim,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        self.double_blocks = nn.ModuleList([
            JoyImageTransformerBlock(
                dim=hidden_size,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                mlp_width_ratio=mlp_width_ratio,
                dtype=dtype,
                device=device,
                operations=operations,
            )
            for _ in range(num_layers)
        ])

        self.norm_out = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.proj_out = operations.Linear(
            hidden_size,
            self.out_channels * math.prod(self.patch_size),
            bias=True,
            dtype=dtype,
            device=device,
        )

    def _get_rotary_pos_embed_for_range(
        self,
        start: Tuple[int, int, int],
        stop: Tuple[int, int, int],
        device=None,
    ) -> torch.Tensor:
        # 3D RoPE for the patch grid range [start, stop) over (t, h, w). Token order after
        # reshape(-1) is (t, h, w), matching the img_in Conv3d flatten.
        rope_dim_list = self.rope_dim_list

        grids = [torch.arange(start[i], stop[i], dtype=torch.float32, device=device) for i in range(3)]
        mesh = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=0)

        angles_parts = []
        for i, dim in enumerate(rope_dim_list):
            pos = mesh[i].reshape(-1)
            freqs = 1.0 / (self.theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device)[: (dim // 2)] / dim))
            angles_parts.append(torch.outer(pos, freqs))

        angles = torch.cat(angles_parts, dim=1)
        cos = angles.cos()
        sin = angles.sin()
        return torch.stack((cos, -sin, sin, cos), dim=-1).unflatten(-1, (2, 2))

    def get_rotary_pos_embed_for_components(
        self,
        component_sizes,
        device=None,
    ) -> torch.Tensor:
        # Per-component 3D RoPE. component_sizes is a list of (t, h, w) patch grid sizes in
        # sequence order [target, ref0, ref1, ...]; h/w restart at 0 for each component while t
        # continues from the running offset, giving every image its own temporal position band.
        freqs_parts = []
        t_offset = 0
        for (t, h, w) in component_sizes:
            freqs = self._get_rotary_pos_embed_for_range(
                start=(t_offset, 0, 0),
                stop=(t_offset + t, h, w),
                device=device,
            )
            freqs_parts.append(freqs)
            t_offset += t
        return torch.cat(freqs_parts, dim=0).unsqueeze(0).unsqueeze(2)

    def unpatchify(self, x: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
        c = self.out_channels
        pt, ph, pw = self.patch_size
        x = x.reshape(x.shape[0], t, h, w, pt, ph, pw, c)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return x.reshape(x.shape[0], c, t * pt, h * ph, w * pw)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor = None,
        ref_latents=None,
        control=None,
        transformer_options=None,
        **kwargs,
    ) -> torch.Tensor:
        transformer_options = {} if transformer_options is None else transformer_options.copy()
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(hidden_states, timestep, context, ref_latents, transformer_options, **kwargs)

    def _forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        ref_latents=None,
        transformer_options=None,
        **kwargs,
    ) -> torch.Tensor:
        pt, ph, pw = self.patch_size
        _, _, ot, oh, ow = hidden_states.shape

        components = [hidden_states, *(ref_latents or [])]
        component_sizes = []
        img_tokens = []
        for comp in components:
            comp = comfy.ldm.common_dit.pad_to_patch_size(comp, self.patch_size)
            _, _, ct, ch, cw = comp.shape
            component_sizes.append((ct // pt, ch // ph, cw // pw))
            tokens = self.img_in(comp).flatten(2).transpose(1, 2)  # (B, n_i, D)
            img_tokens.append(tokens)

        img = torch.cat(img_tokens, dim=1)

        _, vec, txt = self.condition_embedder(timestep, context)
        vec = vec.unflatten(1, (6, -1))

        image_rotary_emb = self.get_rotary_pos_embed_for_components(
            component_sizes,
            device=hidden_states.device,
        )

        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        transformer_options["total_blocks"] = len(self.double_blocks)
        transformer_options["block_type"] = "double"
        for i, block in enumerate(self.double_blocks):
            transformer_options["block_index"] = i
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(
                        hidden_states=args["img"],
                        encoder_hidden_states=args["txt"],
                        temb=args["vec"],
                        image_rotary_emb=args["pe"],
                        transformer_options=args.get("transformer_options"),
                    )
                    return out

                out = blocks_replace[("double_block", i)]({"img": img,
                                                           "txt": txt,
                                                           "vec": vec,
                                                           "pe": image_rotary_emb,
                                                           "transformer_options": transformer_options},
                                                          {"original_block": block_wrap})
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(
                    hidden_states=img,
                    encoder_hidden_states=txt,
                    temb=vec,
                    image_rotary_emb=image_rotary_emb,
                    transformer_options=transformer_options,
                )

        tt, th, tw = component_sizes[0]
        target_tokens = tt * th * tw
        img = img[:, :target_tokens, :]
        img = self.proj_out(self.norm_out(img))
        img = self.unpatchify(img, tt, th, tw)
        return img[:, :, :ot, :oh, :ow]
