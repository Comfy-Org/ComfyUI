"""HiDream-O1-Image transformer.

Pixel-space DiT built on Qwen3-VL: the vision tower (Qwen35VisionModel)
encodes ref images, the Qwen3-VL-8B decoder (Llama2_ with interleaved MRoPE)
processes a unified text+image sequence, and 32x32 patch embed/unembed
shims map raw RGB in and out of LLM hidden space. The Qwen3-VL deepstack
mergers go unused — their weights are dropped at load.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import einops
import torch
import torch.nn as nn

import comfy.patcher_extension
from comfy.ldm.modules.diffusionmodules.mmdit import TimestepEmbedder
from comfy.text_encoders.llama import Llama2_
from comfy.text_encoders.qwen35 import Qwen35VisionModel

from .attention import make_two_pass_attention


IMAGE_TOKEN_ID = 151655   # Qwen3-VL <|image_pad|>
TMS_TOKEN_ID = 151673     # HiDream-O1 <|tms_token|>
PATCH_SIZE = 32


@dataclass
class HiDreamO1TextConfig:
    """Qwen3-VL-8B text-decoder dims (matches public Qwen3-VL-8B-Instruct)."""
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 128000
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    rope_scale: Optional[float] = None
    rope_dims: List[int] = field(default_factory=lambda: [24, 20, 20])
    interleaved_mrope: bool = True
    transformer_type: str = "llama"
    rms_norm_add: bool = False
    mlp_activation: str = "silu"
    qkv_bias: bool = False
    q_norm: str = "gemma3"
    k_norm: str = "gemma3"
    final_norm: bool = True
    lm_head: bool = False
    stop_tokens: List[int] = field(default_factory=lambda: [151643, 151645])


QWEN3VL_VISION_DEFAULTS = dict(
    hidden_size=1152,
    num_heads=16,
    intermediate_size=4304,
    depth=27,
    patch_size=16,
    temporal_patch_size=2,
    in_channels=3,
    spatial_merge_size=2,
    num_position_embeddings=2304,
    deepstack_visual_indexes=(8, 16, 24),
    out_hidden_size=4096,  # final merger projects directly into LLM hidden
)


class BottleneckPatchEmbed(nn.Module):
    # 3072 -> 1024 -> 4096 (raw 32x32 RGB patch -> bottleneck -> LLM hidden).
    def __init__(self, patch_size=32, in_chans=3, pca_dim=1024, embed_dim=4096, bias=True, device=None, dtype=None, ops=None):
        super().__init__()
        self.proj1 = ops.Linear(patch_size * patch_size * in_chans, pca_dim, bias=False, device=device, dtype=dtype)
        self.proj2 = ops.Linear(pca_dim, embed_dim, bias=bias, device=device, dtype=dtype)

    def forward(self, x):
        return self.proj2(self.proj1(x))


class FinalLayer(nn.Module):
    # 4096 -> 3072 (LLM hidden -> flat pixel patch).
    def __init__(self, hidden_size, patch_size=32, out_channels=3, device=None, dtype=None, ops=None):
        super().__init__()
        self.linear = ops.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True, device=device, dtype=dtype)

    def forward(self, x):
        return self.linear(x)


class HiDreamO1Transformer(nn.Module):
    """HiDream-O1 unified pixel-level transformer."""

    def __init__(self, image_model=None, dtype=None, device=None, operations=None,
                 text_config_overrides=None, vision_config_overrides=None, **kwargs):
        super().__init__()
        self.dtype = dtype

        text_cfg = HiDreamO1TextConfig(**(text_config_overrides or {}))
        vision_cfg = dict(QWEN3VL_VISION_DEFAULTS)
        if vision_config_overrides:
            vision_cfg.update(vision_config_overrides)
        vision_cfg["out_hidden_size"] = text_cfg.hidden_size

        self.text_config = text_cfg
        self.vision_config = vision_cfg
        self.hidden_size = text_cfg.hidden_size
        self.patch_size = PATCH_SIZE
        self.in_channels = 3
        self.tms_token_id = TMS_TOKEN_ID

        self.visual = Qwen35VisionModel(vision_cfg, device=device, dtype=dtype, ops=operations)
        self.language_model = Llama2_(text_cfg, device=device, dtype=dtype, ops=operations)
        self.t_embedder1 = TimestepEmbedder(
            text_cfg.hidden_size, device=device, dtype=dtype, operations=operations,
        )
        self.x_embedder = BottleneckPatchEmbed(
            patch_size=self.patch_size, in_chans=self.in_channels,
            pca_dim=text_cfg.hidden_size // 4, embed_dim=text_cfg.hidden_size,
            bias=True, device=device, dtype=dtype, ops=operations,
        )
        self.final_layer2 = FinalLayer(
            text_cfg.hidden_size, patch_size=self.patch_size,
            out_channels=self.in_channels, device=device, dtype=dtype, ops=operations,
        )

        self._visual_cache = None
        self._kv_cache_entries = []

    def clear_kv_cache(self):
        self._kv_cache_entries = []
        self._visual_cache = None

    def forward(self, x, timesteps, context=None, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, timesteps, context, transformer_options, **kwargs)

    def _forward(self, x, timesteps, context=None, transformer_options={}, input_ids=None, attention_mask=None, position_ids=None,
                 vinput_mask=None, ar_len=None, ref_pixel_values=None, ref_image_grid_thw=None, ref_patches=None, **kwargs):
        """Returns flow-match velocity (x - x_pred) / sigma"""

        if input_ids is None or position_ids is None:
            raise ValueError("HiDreamO1Transformer requires input_ids and position_ids in conditioning")

        B, _, H, W = x.shape
        h_p, w_p = H // self.patch_size, W // self.patch_size
        tgt_image_len = h_p * w_p

        z = einops.rearrange(
            x, 'B C (H p1) (W p2) -> B (H W) (C p1 p2)',
            p1=self.patch_size, p2=self.patch_size,
        )
        vinputs = torch.cat([z, ref_patches.to(z.dtype)], dim=1) if ref_patches is not None else z

        inputs_embeds = self.language_model.embed_tokens(input_ids).to(x.dtype)

        if ref_pixel_values is not None and ref_image_grid_thw is not None:
            # ViT output is constant across sampling steps within a generation
            # identity-key by the input tensor so refs don't recompute every step.
            cached = self._visual_cache
            if cached is not None and cached[0] is ref_pixel_values:
                image_embeds = cached[1]
            else:
                ref_pv = ref_pixel_values.to(inputs_embeds.device)
                ref_grid = ref_image_grid_thw.to(inputs_embeds.device).long()
                # extra_conds wraps with a leading batch dim; refs are model-level so [0] always recovers them.
                if ref_pv.dim() == 3:
                    ref_pv = ref_pv[0]
                if ref_grid.dim() == 3:
                    ref_grid = ref_grid[0]
                image_embeds = self.visual(ref_pv, ref_grid).to(inputs_embeds.dtype)
                self._visual_cache = (ref_pixel_values, image_embeds)
            # image_pad positions identical across batch (input_ids shared cond/uncond).
            image_idx = (input_ids[0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
            if image_idx.shape[0] != image_embeds.shape[0]:
                raise ValueError(
                    f"Image-token count {image_idx.shape[0]} != ViT output count "
                    f"{image_embeds.shape[0]}; check tokenizer/processor alignment."
                )
            inputs_embeds[:, image_idx] = image_embeds.unsqueeze(0).expand(B, -1, -1)

        sigma = timesteps.float() / 1000.0
        t_pixeldit = 1.0 - sigma
        t_emb = self.t_embedder1(t_pixeldit * 1000, inputs_embeds.dtype)
        tms_mask_3d = (input_ids == self.tms_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = torch.where(tms_mask_3d, t_emb.unsqueeze(1).expand_as(inputs_embeds), inputs_embeds)

        vinputs_embedded = self.x_embedder(vinputs.to(inputs_embeds.dtype))
        inputs_embeds = torch.cat([inputs_embeds, vinputs_embedded], dim=1)

        # extra_conds stores position_ids as (1, 3, T); process_cond repeats dim 0 to B. Take row 0.
        freqs_cis = self.language_model.compute_freqs_cis(position_ids[0].to(x.device), x.device)
        freqs_cis = tuple(t.to(x.dtype) for t in freqs_cis)

        two_pass_attn = make_two_pass_attention(ar_len, transformer_options=transformer_options)
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        transformer_options["total_blocks"] = len(self.language_model.layers)
        transformer_options["block_type"] = "double"

        # Cache prefix K/V across steps. Key includes input_ids (prompt), ref_id
        # (refs scatter into inputs_embeds), and position_ids (RoPE baked into cached K).
        can_cache = not blocks_replace and ar_len > 0
        cache_len = ar_len if can_cache else 0
        ref_id = id(ref_pixel_values) if ref_pixel_values is not None else None
        pos_ids_key = position_ids[..., :cache_len] if can_cache else position_ids
        cache_entries = self._kv_cache_entries
        # Drop stale entries from a previous device (model was unloaded and reloaded).
        if cache_entries and cache_entries[0]["input_ids"].device != input_ids.device:
            cache_entries = []
            self._kv_cache_entries = []
        kv_cache = None
        if can_cache:
            for entry in cache_entries:
                ck = entry["input_ids"]
                ep = entry["position_ids"]
                if (entry["cache_len"] == cache_len
                        and ck.shape == input_ids.shape and torch.equal(ck, input_ids)
                        and entry["ref_id"] == ref_id
                        and ep.shape == pos_ids_key.shape and torch.equal(ep, pos_ids_key)):
                    kv_cache = entry
                    break

        if kv_cache is not None:
            # Hot path: project Q/K/V only for fresh positions; past_key_value prepends cached AR K/V.
            hidden_states = inputs_embeds[:, cache_len:]
            sliced_freqs = tuple(t[..., cache_len:, :] for t in freqs_cis)
            for i, layer in enumerate(self.language_model.layers):
                transformer_options["block_index"] = i
                K_i, V_i = kv_cache["kv"][i]
                hidden_states, _ = layer(
                    x=hidden_states, attention_mask=None, freqs_cis=sliced_freqs, optimized_attention=two_pass_attn,
                    past_key_value=(K_i, V_i, cache_len),
                )
        else:
            # Cold path: run full sequence; if cacheable, snapshot K/V at AR positions.
            snapshots = [] if can_cache else None
            past_kv_cold = () if can_cache else None
            hidden_states = inputs_embeds
            for i, layer in enumerate(self.language_model.layers):
                transformer_options["block_index"] = i
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args, _layer=layer):
                        out = {}
                        out["x"], _ = _layer(
                            x=args["x"], attention_mask=args.get("attention_mask"),
                            freqs_cis=args["freqs_cis"], optimized_attention=args["optimized_attention"],
                            past_key_value=None,
                        )
                        return out
                    out = blocks_replace[("double_block", i)](
                        {"x": hidden_states, "attention_mask": None,
                         "freqs_cis": freqs_cis, "optimized_attention": two_pass_attn,
                         "transformer_options": transformer_options},
                        {"original_block": block_wrap},
                    )
                    hidden_states = out["x"]
                else:
                    hidden_states, present_kv = layer(
                        x=hidden_states, attention_mask=None,
                        freqs_cis=freqs_cis, optimized_attention=two_pass_attn,
                        past_key_value=past_kv_cold,
                    )
                    if snapshots is not None:
                        K, V, _ = present_kv
                        snapshots.append((K[:, :, :cache_len].contiguous(),
                                          V[:, :, :cache_len].contiguous()))
            if snapshots is not None:
                # Cap at 2 entries (cond + uncond). Multi-cond workflows LRU-evict.
                new_entry = {
                    "input_ids": input_ids.clone(),
                    "cache_len": cache_len,
                    "kv": snapshots,
                    "ref_id": ref_id,
                    "position_ids": pos_ids_key.clone(),
                }
                self._kv_cache_entries = (cache_entries + [new_entry])[-2:]

        if self.language_model.norm is not None:
            hidden_states = self.language_model.norm(hidden_states)

        # Slice target-image positions before the final projection so the Linear only runs on tgt_image_len tokens.
        # In the hot path hidden_states starts at original position cache_len, so masks/indices shift by cache_len.
        sliced_offset = cache_len if kv_cache is not None else 0
        if vinput_mask is not None:
            vmask = vinput_mask.to(x.device).bool()
            if sliced_offset > 0:
                vmask = vmask[:, sliced_offset:]
            target_hidden = hidden_states[vmask].view(B, -1, hidden_states.shape[-1])[:, :tgt_image_len]
        else:
            txt_seq_len = input_ids.shape[1]
            start = txt_seq_len - sliced_offset
            target_hidden = hidden_states[:, start:start + tgt_image_len]
        x_pred_tgt = self.final_layer2(target_hidden)

        # fp32 final subtraction, bf16 here noticeably degrades samples.
        x_pred_img = einops.rearrange(
            x_pred_tgt, 'B (H W) (C p1 p2) -> B C (H p1) (W p2)',
            H=h_p, W=w_p, p1=self.patch_size, p2=self.patch_size,
        )
        return (x.float() - x_pred_img.float()) / sigma.view(B, 1, 1, 1).clamp_min(1e-3)
