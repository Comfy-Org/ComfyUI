"""HiDream-O1 input-prep helpers: image/resolution math and unified-sequence
RoPE position-id assembly. The fix_point offset in get_rope_index_fix_point
lets the target image and patchified ref images share spatial RoPE positions
despite living at different sequence indices — same 2D image plane.
"""

import math
from typing import Optional

import torch


PATCH_SIZE = 32
CONDITION_IMAGE_SIZE = 384  # ViT-side base size for ref images


def resize_tensor(img_t, image_size, patch_size=16):
    """img_t: (1, 3, H, W) float [0, 1]. Fit to image_size**2 area, patch-aligned, center-cropped."""

    while min(img_t.shape[-2], img_t.shape[-1]) >= 2 * image_size: # Pre-halves with 2x2 box averaging while the image is still very large
        img_t = torch.nn.functional.avg_pool2d(img_t, kernel_size=2, stride=2)

    _, _, height, width = img_t.shape
    m = patch_size
    s_max = image_size * image_size
    scale = math.sqrt(s_max / (width * height))

    candidates = [
        (round(width * scale) // m * m, round(height * scale) // m * m),
        (round(width * scale) // m * m, math.floor(height * scale) // m * m),
        (math.floor(width * scale) // m * m, round(height * scale) // m * m),
        (math.floor(width * scale) // m * m, math.floor(height * scale) // m * m),
    ]
    candidates = sorted(candidates, key=lambda x: x[0] * x[1], reverse=True)
    new_size = candidates[-1]
    for c in candidates:
        if c[0] * c[1] <= s_max:
            new_size = c
            break

    new_w, new_h = new_size
    s1 = width / new_w
    s2 = height / new_h
    if s1 < s2:
        resize_w, resize_h = new_w, round(height / s1)
    else:
        resize_w, resize_h = round(width / s2), new_h
    img_t = torch.nn.functional.interpolate(img_t, size=(resize_h, resize_w), mode="bicubic")
    top = (resize_h - new_h) // 2
    left = (resize_w - new_w) // 2
    return img_t[..., top:top + new_h, left:left + new_w]


def calculate_dimensions(max_size, ratio):
    """(W, H) for an aspect ratio fitting in max_size**2 area, 32-aligned."""
    width = math.sqrt(max_size * max_size * ratio)
    height = width / ratio
    width = int(width / 32) * 32
    height = int(height / 32) * 32
    return width, height


def ref_max_size(target_max_dim, k):
    """K-dependent ref-image max dim before patchifying."""
    if k == 1:
        return target_max_dim
    if k == 2:
        return target_max_dim * 48 // 64
    if k <= 4:
        return target_max_dim // 2
    if k <= 8:
        return target_max_dim * 24 // 64
    return target_max_dim // 4


def cond_image_size(k):
    """K-dependent ViT-side image size."""
    if k <= 4:
        return CONDITION_IMAGE_SIZE
    if k <= 8:
        return CONDITION_IMAGE_SIZE * 48 // 64
    return CONDITION_IMAGE_SIZE // 2


def get_rope_index_fix_point(
    spatial_merge_size: int,
    image_token_id: int,
    vision_start_token_id: int,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    skip_vision_start_token=None,
    fix_point: int = 4096,
):
    mrope_position_deltas = []
    if input_ids is not None and image_grid_thw is not None:
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3, input_ids.shape[0], input_ids.shape[1],
            dtype=input_ids.dtype, device=input_ids.device,
        )
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids_b in enumerate(total_input_ids):
            fp = fix_point
            image_index = 0
            input_ids_b = input_ids_b[attention_mask[i] == 1]
            vision_start_indices = torch.argwhere(input_ids_b == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids_b[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            input_tokens = input_ids_b.tolist()
            llm_pos_ids_list = []
            st = 0
            remain_images = image_nums
            for _ in range(image_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed = input_tokens.index(image_token_id, st)
                else:
                    ed = len(input_tokens) + 1
                t = image_grid_thw[image_index][0]
                h = image_grid_thw[image_index][1]
                w = image_grid_thw[image_index][2]
                image_index += 1
                remain_images -= 1
                llm_grid_t = t.item()
                llm_grid_h = h.item() // spatial_merge_size
                llm_grid_w = w.item() // spatial_merge_size
                text_len = ed - st
                text_len -= skip_vision_start_token[image_index - 1]
                text_len = max(0, text_len)
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()

                if skip_vision_start_token[image_index - 1]:
                    if fp > 0:
                        fp = fp - st_idx
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + fp + st_idx)
                    fp = 0
                else:
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
    else:
        position_ids = (
            torch.arange(input_ids.shape[1], device=input_ids.device)
            .view(1, 1, -1).expand(3, input_ids.shape[0], -1)
        )
        mrope_position_deltas = torch.zeros(
            [input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype,
        )
    return position_ids, mrope_position_deltas
