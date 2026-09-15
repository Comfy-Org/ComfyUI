# Copyright (c) 2026 ByteDance Ltd. and/or its affiliate
# SPDX-License-Identifier: Apache-2.0
"""Qwen2.5-VL MRoPE position indices used by Bernini v2."""

from __future__ import annotations

import torch


def get_rope_index(
    input_ids: torch.Tensor,
    *,
    image_grid_thw: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    image_token_id: int = 151655,
    video_token_id: int = 151656,
    vision_start_token_id: int = 151652,
    spatial_merge_size: int = 2,
    tokens_per_second: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return official token-based MRoPE positions and deltas."""

    if image_grid_thw is None and video_grid_thw is None:
        if attention_mask is None:
            position_ids = torch.arange(
                input_ids.shape[1], device=input_ids.device
            ).view(1, 1, -1)
            position_ids = position_ids.expand(3, input_ids.shape[0], -1)
            deltas = torch.zeros(
                input_ids.shape[0], 1, device=input_ids.device, dtype=input_ids.dtype
            )
            return position_ids, deltas
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        maximum = position_ids.max(0).values.max(-1, keepdim=True).values
        return position_ids, maximum + 1 - attention_mask.shape[-1]

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    positions = torch.ones(
        3, *input_ids.shape, dtype=input_ids.dtype, device=input_ids.device
    )
    deltas = []
    image_index = 0
    video_index = 0
    for batch_index, batch_ids in enumerate(input_ids):
        active_ids = batch_ids[attention_mask[batch_index] == 1]
        starts = torch.argwhere(active_ids == vision_start_token_id).squeeze(1)
        visual_tokens = active_ids[starts + 1]
        image_count = int((visual_tokens == image_token_id).sum())
        video_count = int((visual_tokens == video_token_id).sum())
        token_list = active_ids.tolist()
        chunks = []
        start = 0
        remaining_images = image_count
        remaining_videos = video_count
        for _ in range(image_count + video_count):
            image_end = (
                token_list.index(image_token_id, start)
                if remaining_images
                else len(token_list) + 1
            )
            video_end = (
                token_list.index(video_token_id, start)
                if remaining_videos
                else len(token_list) + 1
            )
            if image_end < video_end:
                if image_grid_thw is None:
                    raise ValueError("image grid is missing")
                grid_t, grid_h, grid_w = image_grid_thw[image_index]
                seconds_per_grid = 0.0
                image_index += 1
                remaining_images -= 1
                end = image_end
            else:
                if video_grid_thw is None:
                    raise ValueError("video grid is missing")
                grid_t, grid_h, grid_w = video_grid_thw[video_index]
                seconds_per_grid = 1.0
                video_index += 1
                remaining_videos -= 1
                end = video_end
            grid_t = int(grid_t)
            grid_h = int(grid_h) // spatial_merge_size
            grid_w = int(grid_w) // spatial_merge_size
            text_length = end - start
            chunk_start = int(chunks[-1].max()) + 1 if chunks else 0
            chunks.append(
                torch.arange(text_length, device=input_ids.device)
                .view(1, -1)
                .expand(3, -1)
                + chunk_start
            )
            time = (
                torch.arange(grid_t, device=input_ids.device)
                .view(-1, 1)
                .expand(-1, grid_h * grid_w)
            )
            time = (time * seconds_per_grid * tokens_per_second).long().flatten()
            height = (
                torch.arange(grid_h, device=input_ids.device)
                .view(1, -1, 1)
                .expand(grid_t, -1, grid_w)
                .flatten()
            )
            width = (
                torch.arange(grid_w, device=input_ids.device)
                .view(1, 1, -1)
                .expand(grid_t, grid_h, -1)
                .flatten()
            )
            chunks.append(
                torch.stack((time, height, width)) + text_length + chunk_start
            )
            start = end + grid_t * grid_h * grid_w
        if start < len(token_list):
            chunk_start = int(chunks[-1].max()) + 1 if chunks else 0
            text_length = len(token_list) - start
            chunks.append(
                torch.arange(text_length, device=input_ids.device)
                .view(1, -1)
                .expand(3, -1)
                + chunk_start
            )
        batch_positions = torch.cat(chunks, dim=1).reshape(3, -1)
        positions[:, batch_index, attention_mask[batch_index] == 1] = batch_positions
        deltas.append(batch_positions.max() + 1 - len(batch_ids))
    return positions, torch.tensor(deltas, device=input_ids.device).unsqueeze(1)
