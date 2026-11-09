# Mage-Flow (https://github.com/microsoft/Mage) native-resolution MMDiT (MIT)
# Architecture is a 12-layer variant of the Qwen-Image double-stream block with
# patch_size=1 (no 2x2 packing), unrotated text tokens and a bf16-rounded
# timestep frequency table.
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

from comfy.ldm.lightricks.model import TimestepEmbedding
from comfy.ldm.flux.layers import EmbedND
from comfy.ldm.qwen_image.model import QwenImageTransformerBlock, LastLayer
import comfy.patcher_extension


class MageTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim,
            dtype=dtype, device=device, operations=operations
        )

    def forward(self, timestep, hidden_states):
        half_dim = 128
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timestep.device) / half_dim
        emb = torch.exp(exponent).to(timestep.dtype)
        emb = timestep[:, None].float() * emb[None, :]
        emb = 1000.0 * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)  # flip_sin_to_cos
        return self.timestep_embedder(emb.to(dtype=hidden_states.dtype))


class MageFlowTransformer2DModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: Optional[int] = 128,
        num_layers: int = 12,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 2560,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        image_model=None,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.dtype = dtype
        self.patch_size = 1
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pe_embedder = EmbedND(dim=attention_head_dim, theta=10000, axes_dim=list(axes_dims_rope))

        self.time_text_embed = MageTimestepProjEmbeddings(embedding_dim=self.inner_dim, dtype=dtype, device=device, operations=operations)

        self.txt_norm = operations.RMSNorm(joint_attention_dim, eps=1e-6, dtype=dtype, device=device)
        self.img_in = operations.Linear(in_channels, self.inner_dim, dtype=dtype, device=device)
        self.txt_in = operations.Linear(joint_attention_dim, self.inner_dim, dtype=dtype, device=device)

        self.transformer_blocks = nn.ModuleList([
            QwenImageTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                dtype=dtype,
                device=device,
                operations=operations
            )
            for _ in range(num_layers)
        ])

        self.norm_out = LastLayer(self.inner_dim, self.inner_dim, dtype=dtype, device=device, operations=operations)
        self.proj_out = operations.Linear(self.inner_dim, self.out_channels, bias=True, dtype=dtype, device=device)

    def process_img(self, x, index=0):
        # patch_size=1: tokens are raw latent pixels, no 2x2 packing.
        bs, c, h, w = x.shape
        hidden_states = x.movedim(1, -1).reshape(bs, h * w, c)

        img_ids = torch.zeros((h, w, 3), device=x.device)
        # Frame axis: positive image index (0 = target, 1..N = reference images).
        img_ids[:, :, 0] = index
        # Mage scale_rope centering: positions [-ceil(n/2), floor(n/2)), i.e.
        # offset by (n - n//2). Differs from Qwen-Image's -(n//2) for odd sizes.
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.arange(h, device=x.device)[:, None] - (h - h // 2)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.arange(w, device=x.device)[None, :] - (w - w // 2)
        return hidden_states, img_ids.reshape(h * w, 3).unsqueeze(0).expand(bs, -1, -1), (h, w)

    def forward(self, x, timestep, context, attention_mask=None, ref_latents=None, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, timestep, context, attention_mask, ref_latents, transformer_options, **kwargs)

    def _forward(self, x, timestep, context, attention_mask=None, ref_latents=None, transformer_options={}, control=None, **kwargs):
        if attention_mask is not None and not torch.is_floating_point(attention_mask):
            attention_mask = (attention_mask - 1).to(x.dtype) * torch.finfo(x.dtype).max

        hidden_states, img_ids, orig_shape = self.process_img(x)
        num_embeds = hidden_states.shape[1]

        if ref_latents is not None:
            ref_num_tokens = []
            index = 0
            for ref in ref_latents:
                index += 1
                kontext, kontext_ids, _ = self.process_img(ref, index=index)
                hidden_states = torch.cat([hidden_states, kontext], dim=1)
                img_ids = torch.cat([img_ids, kontext_ids], dim=1)
                ref_num_tokens.append(kontext.shape[1])
            transformer_options = transformer_options.copy()
            transformer_options["reference_image_num_tokens"] = ref_num_tokens

        # Text tokens are not rotated in Mage-Flow: RoPE at position 0 is the
        # identity rotation.
        txt_ids = torch.zeros((x.shape[0], context.shape[1], 3), device=x.device)

        hidden_states = self.img_in(hidden_states)
        context = self.txt_norm(context)
        context = self.txt_in(context)

        temb = self.time_text_embed(timestep, hidden_states)

        patches_replace = transformer_options.get("patches_replace", {})
        patches = transformer_options.get("patches", {})
        blocks_replace = patches_replace.get("dit", {})

        if "post_input" in patches:
            for p in patches["post_input"]:
                out = p({"img": hidden_states, "txt": context, "img_ids": img_ids, "txt_ids": txt_ids, "transformer_options": transformer_options})
                hidden_states = out["img"]
                context = out["txt"]
                img_ids = out["img_ids"]
                txt_ids = out["txt_ids"]

        ids = torch.cat((txt_ids, img_ids), dim=1)
        image_rotary_emb = self.pe_embedder(ids).contiguous()
        del ids, txt_ids, img_ids

        transformer_options["total_blocks"] = len(self.transformer_blocks)
        transformer_options["block_type"] = "double"
        for i, block in enumerate(self.transformer_blocks):
            transformer_options["block_index"] = i
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["txt"], out["img"] = block(hidden_states=args["img"], encoder_hidden_states=args["txt"], encoder_hidden_states_mask=attention_mask, temb=args["vec"], image_rotary_emb=args["pe"], transformer_options=args["transformer_options"])
                    return out
                out = blocks_replace[("double_block", i)]({"img": hidden_states, "txt": context, "vec": temb, "pe": image_rotary_emb, "transformer_options": transformer_options}, {"original_block": block_wrap})
                hidden_states = out["img"]
                context = out["txt"]
            else:
                context, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=context,
                    encoder_hidden_states_mask=attention_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    transformer_options=transformer_options,
                )

            if "double_block" in patches:
                for p in patches["double_block"]:
                    out = p({"img": hidden_states, "txt": context, "x": x, "block_index": i, "transformer_options": transformer_options})
                    hidden_states = out["img"]
                    context = out["txt"]

            if control is not None:  # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        hidden_states[:, :add.shape[1]] += add

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states[:, :num_embeds]
        h, w = orig_shape
        return hidden_states.reshape(x.shape[0], h, w, self.out_channels).movedim(-1, 1)
