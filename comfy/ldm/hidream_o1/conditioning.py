"""HiDream-O1 conditioning prep — ref-image dual path + extra_conds assembly.

Each ref image goes through two paths: a 32x32 patchified stream concatenated
to the noised target, and a Qwen3-VL ViT path producing tokens that scatter
into input_ids at <|image_pad|> positions.
"""

from typing import List

import torch

import comfy.utils
from comfy.text_encoders.qwen_vl import process_qwen2vl_images

from .utils import (PATCH_SIZE, calculate_dimensions, cond_image_size, ref_max_size, resize_tensor)

# Qwen3-VL ViT preprocessing constants (preprocessor_config.json).
VIT_PATCH = 16
VIT_MERGE = 2
VIT_IMAGE_MEAN = [0.5, 0.5, 0.5]
VIT_IMAGE_STD = [0.5, 0.5, 0.5]


def prepare_ref_images(
    ref_images: List[torch.Tensor],
    target_h: int,
    target_w: int,
    device: torch.device,
    dtype: torch.dtype,
):
    """Build the dual-path tensors for K reference images at (target_h, target_w).

    Returns None for K=0, else a dict with ref_patches, ref_pixel_values,
    ref_image_grid_thw, per_ref_vit_tokens, per_ref_patch_grids.
    """
    K = len(ref_images)
    if K == 0:
        return None
    max_size = ref_max_size(max(target_h, target_w), K)
    cis = cond_image_size(K)

    refs_t = [img[0].clamp(0, 1).permute(2, 0, 1).unsqueeze(0).contiguous().float() for img in ref_images]
    refs_t = [resize_tensor(t, max_size, PATCH_SIZE) for t in refs_t]

    # 32-patch path.
    ref_patches_per = []
    per_ref_patch_grids = []
    for t in refs_t:
        t_norm = (t.squeeze(0) - 0.5) / 0.5  # (3, H, W) in [-1, 1]
        h_p, w_p = t_norm.shape[-2] // PATCH_SIZE, t_norm.shape[-1] // PATCH_SIZE
        per_ref_patch_grids.append((h_p, w_p))
        patches = (
            t_norm.reshape(3, h_p, PATCH_SIZE, w_p, PATCH_SIZE)
            .permute(1, 3, 0, 2, 4)
            .reshape(h_p * w_p, 3 * PATCH_SIZE * PATCH_SIZE)
        )
        ref_patches_per.append(patches)
    ref_patches = torch.cat(ref_patches_per, dim=0).unsqueeze(0).to(device=device, dtype=dtype)

    # ViT path.
    refs_vlm_t = []
    for t in refs_t:
        _, _, h, w = t.shape
        cond_w, cond_h = calculate_dimensions(cis, w / h)
        cond_w = max(cond_w, VIT_PATCH * VIT_MERGE)
        cond_h = max(cond_h, VIT_PATCH * VIT_MERGE)
        refs_vlm_t.append(comfy.utils.common_upscale(t, cond_w, cond_h, "lanczos", "disabled"))

    pv_list, grid_list, per_ref_vit_tokens = [], [], []
    for t_v in refs_vlm_t:
        pv, grid_thw = process_qwen2vl_images(
            t_v.permute(0, 2, 3, 1),
            min_pixels=0, max_pixels=10**12,
            patch_size=VIT_PATCH, merge_size=VIT_MERGE,
            image_mean=VIT_IMAGE_MEAN, image_std=VIT_IMAGE_STD,
        )
        grid_thw = grid_thw[0]
        pv_list.append(pv.to(device=device, dtype=dtype))
        grid_list.append(grid_thw.to(device=device))
        # Post-merge token count = number of <|image_pad|> tokens this image expands to in input_ids.
        gh, gw = int(grid_thw[1].item()), int(grid_thw[2].item())
        per_ref_vit_tokens.append((gh // VIT_MERGE) * (gw // VIT_MERGE))

    return {
        "ref_patches": ref_patches,
        "ref_pixel_values": torch.cat(pv_list, dim=0),
        "ref_image_grid_thw": torch.stack(grid_list, dim=0),
        "per_ref_vit_tokens": per_ref_vit_tokens,
        "per_ref_patch_grids": per_ref_patch_grids,
    }


def build_ref_input_ids(
    text_input_ids: torch.Tensor,
    per_ref_vit_tokens: List[int],
    image_token_id: int,
    vision_start_id: int,
    vision_end_id: int,
):
    """Splice [vision_start, image_pad*N, vision_end] blocks into input_ids
    after the [im_start, user, \\n] prefix (matches original chat template).
    """
    ids = text_input_ids[0].tolist()
    inserted = []
    for n_pad in per_ref_vit_tokens:
        inserted.extend([vision_start_id] + [image_token_id] * n_pad + [vision_end_id])
    new_ids = ids[:3] + inserted + ids[3:]  # 3 = len([im_start, user, \n])
    return torch.tensor([new_ids], dtype=text_input_ids.dtype, device=text_input_ids.device)


def build_extra_conds(
    text_input_ids: torch.Tensor,
    noise: torch.Tensor,
    ref_images: List[torch.Tensor] = None,
    target_patch_size: int = 32,
):
    """Assemble all conditioning tensors for HiDreamO1Transformer.forward:
    input_ids (with ref-vision tokens spliced in for the edit/IP path),
    position_ids (MRoPE), token_types, vinput_mask, plus the ref
    dual-path tensors when refs are provided.
    """
    from .utils import get_rope_index_fix_point
    from comfy.text_encoders.hidream_o1 import (
        IMAGE_TOKEN_ID, VISION_START_ID, VISION_END_ID,
    )

    if text_input_ids.dim() == 1:
        text_input_ids = text_input_ids.unsqueeze(0)
    text_input_ids = text_input_ids.long().to(noise.device)
    B = noise.shape[0]
    if text_input_ids.shape[0] == 1 and B > 1:
        text_input_ids = text_input_ids.expand(B, -1)

    H, W = noise.shape[-2], noise.shape[-1]
    h_p, w_p = H // target_patch_size, W // target_patch_size
    image_len = h_p * w_p
    image_grid_thw_tgt = torch.tensor(
        [[1, h_p, w_p]], dtype=torch.long, device=text_input_ids.device,
    )

    out = {}
    if ref_images:
        ref = prepare_ref_images(ref_images, H, W, device=noise.device, dtype=noise.dtype)
        text_input_ids = build_ref_input_ids(
            text_input_ids, ref["per_ref_vit_tokens"],
            IMAGE_TOKEN_ID, VISION_START_ID, VISION_END_ID,
        )
        new_txt_len = text_input_ids.shape[1]

        # Each ref's patchified stream gets a [vision_start, image_pad*N-1]
        # block in the position-id stream after the noised target.
        ref_grid_lengths = [hp * wp for (hp, wp) in ref["per_ref_patch_grids"]]
        tgt_vision = torch.full((1, image_len), IMAGE_TOKEN_ID,
                                dtype=text_input_ids.dtype, device=text_input_ids.device)
        tgt_vision[:, 0] = VISION_START_ID
        ref_vision_blocks = []
        for rl in ref_grid_lengths:
            blk = torch.full((1, rl), IMAGE_TOKEN_ID,
                             dtype=text_input_ids.dtype, device=text_input_ids.device)
            blk[:, 0] = VISION_START_ID
            ref_vision_blocks.append(blk)
        ref_vision_cat = torch.cat([tgt_vision] + ref_vision_blocks, dim=1)
        input_ids_pad = torch.cat([text_input_ids, ref_vision_cat], dim=-1)
        total_ref_patches_len = sum(ref_grid_lengths)
        total_len = new_txt_len + image_len + total_ref_patches_len

        # K (ViT, post-merge) + 1 (target) + K (ref-patches) image grids.
        K = len(ref_images)
        igthw_cond = ref["ref_image_grid_thw"].clone()
        igthw_cond[:, 1] //= 2
        igthw_cond[:, 2] //= 2
        image_grid_thw_ref = torch.tensor(
            [[1, hp, wp] for (hp, wp) in ref["per_ref_patch_grids"]],
            dtype=torch.long, device=text_input_ids.device,
        )
        igthw_all = torch.cat([
            igthw_cond.to(text_input_ids.device),
            image_grid_thw_tgt,
            image_grid_thw_ref,
        ], dim=0)
        position_ids, _ = get_rope_index_fix_point(
            spatial_merge_size=1,
            image_token_id=IMAGE_TOKEN_ID,
            vision_start_token_id=VISION_START_ID,
            input_ids=input_ids_pad, image_grid_thw=igthw_all,
            attention_mask=None,
            skip_vision_start_token=[0] * K + [1] + [1] * K,
            fix_point=4096,
        )

        # tms + target_image + ref_patches are all gen.
        tms_pos = new_txt_len - 1
        ar_len = tms_pos
        token_types = torch.zeros(B, total_len, dtype=torch.long, device=noise.device)
        token_types[:, tms_pos:] = 1
        vinput_mask = torch.zeros(B, total_len, dtype=torch.bool, device=noise.device)
        vinput_mask[:, new_txt_len:] = True

        # Leading batch dim sidesteps CONDRegular.process_cond's repeat_to_batch_size truncation
        out["ref_pixel_values"] = ref["ref_pixel_values"].unsqueeze(0)
        out["ref_image_grid_thw"] = ref["ref_image_grid_thw"].unsqueeze(0)
        out["ref_patches"] = ref["ref_patches"]
    else:
        # T2I: text + noised target only, vision_start replaces the first image token
        txt_len = text_input_ids.shape[1]
        total_len = txt_len + image_len
        vision_tokens = torch.full((B, image_len), IMAGE_TOKEN_ID,
                                   dtype=text_input_ids.dtype, device=text_input_ids.device)
        vision_tokens[:, 0] = VISION_START_ID
        input_ids_pad = torch.cat([text_input_ids, vision_tokens], dim=-1)
        position_ids, _ = get_rope_index_fix_point(
            spatial_merge_size=1,
            image_token_id=IMAGE_TOKEN_ID,
            vision_start_token_id=VISION_START_ID,
            input_ids=input_ids_pad, image_grid_thw=image_grid_thw_tgt,
            attention_mask=None,
            skip_vision_start_token=[1],
        )
        ar_len = txt_len - 1
        token_types = torch.zeros(B, total_len, dtype=torch.long, device=noise.device)
        token_types[:, ar_len:] = 1
        vinput_mask = torch.zeros(B, total_len, dtype=torch.bool, device=noise.device)
        vinput_mask[:, txt_len:] = True

    out["input_ids"] = text_input_ids
    out["position_ids"] = position_ids[:, 0].unsqueeze(0) # Collapse position_ids batch and add a leading dim so CONDRegular's batch-resize doesn't truncate the 3-axis MRoPE dim
    out["token_types"] = token_types
    out["vinput_mask"] = vinput_mask
    out["ar_len"] = ar_len
    return out
