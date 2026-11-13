# Copyright (c) 2026 ByteDance Ltd. and/or its affiliate
# SPDX-License-Identifier: Apache-2.0
"""Bernini v2 inference chat template and block attention mask."""

from __future__ import annotations

from collections.abc import Sequence

import torch

SYSTEM_PROMPTS = {
    "default": "You are a helpful assistant.",
    "t2i": "You are a helpful assistant specialized in text-to-image generation.",
    "t2v": "You are a helpful assistant specialized in text-to-video generation.",
    "i2i": "You are a helpful assistant specialized in image editing.",
    "v2v": "You are a helpful assistant specialized in video editing.",
    "r2v": "You are a helpful assistant specialized in subject-to-video generation.",
    "rv2v": "You are a helpful assistant specialized in video editing with reference.",
}


def build_custom_attention_mask(
    token_type: torch.Tensor,
    token_segment_ids: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build Bernini's ``[B,L,L]`` additive text/source/target mask."""

    _, length = token_type.shape
    query_type = token_type.unsqueeze(2)
    key_type = token_type.unsqueeze(1)
    query_id = token_segment_ids.unsqueeze(2)
    key_id = token_segment_ids.unsqueeze(1)
    causal = torch.tril(
        torch.ones(length, length, device=token_type.device, dtype=torch.bool)
    ).unsqueeze(0)
    key_is_text_or_input = (key_type == 0) | (key_type == 2)
    input_visible = causal & key_is_text_or_input
    planned_visible = (key_type == 1) & (query_id == key_id)
    output_visible = (key_type == 3) & (query_id == key_id)
    query_is_text_or_input = (query_type == 0) | (query_type == 2)
    visible = query_is_text_or_input & input_visible
    visible |= (query_type == 1) & (input_visible | planned_visible)
    visible |= (query_type == 3) & (input_visible | output_visible)
    mask = torch.zeros(visible.shape, device=token_type.device, dtype=dtype)
    return mask.masked_fill(~visible, float("-inf"))


class BerniniTemplate:
    """Inference-only form of the official Bernini multimodal template."""

    def __init__(self, tokenizer, max_visual_items: int = 64):
        self.tokenizer = tokenizer
        self.image_pad_id = 151655
        self.video_pad_id = 151656
        self.vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.visual_input_pads = [
            f"<|visual_input_token_pad_{index}|>" for index in range(max_visual_items)
        ]
        self.visual_output_pads = [
            f"<|visual_output_token_pad_{index}|>" for index in range(max_visual_items)
        ]
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": self.visual_input_pads
                + self.visual_output_pads
            }
        )
        self.visual_input_ids = tokenizer.convert_tokens_to_ids(self.visual_input_pads)
        self.visual_output_ids = tokenizer.convert_tokens_to_ids(
            self.visual_output_pads
        )

    def _visual_pattern(self, count: int, visual_id: int, *, output: bool) -> str:
        pads = self.visual_output_pads if output else self.visual_input_pads
        if not 0 <= visual_id < len(pads):
            raise ValueError(
                f"Bernini v2 supports at most {len(pads)} visual items, got {visual_id + 1}"
            )
        return "<|vision_start|>" + pads[visual_id] * count + "<|vision_end|>"

    def encode(
        self,
        conversations: Sequence[dict[str, object]],
        *,
        num_tokens: dict[str, Sequence[int]],
        task: str,
        drop_text: bool = False,
        drop_images: bool = False,
        drop_videos: bool = False,
        negative_prompt: str = "",
        mask_dtype: torch.dtype = torch.float32,
    ) -> dict[str, object]:
        image_counts = iter(num_tokens.get("image", []))
        video_counts = iter(num_tokens.get("video", []))
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPTS.get(task, SYSTEM_PROMPTS["default"]),
            }
        ]
        content = ""
        previous_has_loss = False
        visual_id_to_type: dict[int, int] = {}
        visual_types: list[int] = []
        visual_source_indices: list[int] = []
        image_id = 0
        video_id = 0
        visual_id = 0

        def flush(has_loss: bool) -> None:
            nonlocal content
            if content.strip():
                messages.append(
                    {"role": "assistant" if has_loss else "user", "content": content}
                )
            content = ""

        for raw_message in conversations:
            message = dict(raw_message)
            message_type = str(message["type"])
            if message_type == "special_token":
                continue
            has_loss = bool(
                message.get(
                    "has_loss", message_type in ("image_gen", "video_gen", "frame_gen")
                )
            )
            if has_loss != previous_has_loss:
                flush(previous_has_loss)
                previous_has_loss = has_loss

            if message_type in ("text", "cot_text"):
                if negative_prompt.strip():
                    text = negative_prompt
                else:
                    text = "" if drop_text else str(message.get("text", ""))
                content += text
                continue

            if message_type in ("image", "image_gen"):
                token_count = int(next(image_counts))
                include = has_loss or not drop_images
                if include:
                    content += self._visual_pattern(
                        token_count, visual_id, output=has_loss
                    )
                    visual_types.append(0)
                    visual_source_indices.append(image_id)
                visual_id_to_type[visual_id] = 0
                image_id += 1
            elif message_type in ("video", "frame_gen", "video_gen"):
                token_count = int(next(video_counts))
                include = has_loss or not drop_videos
                if include:
                    content += self._visual_pattern(
                        token_count, visual_id, output=has_loss
                    )
                    visual_types.append(1)
                    visual_source_indices.append(video_id)
                visual_id_to_type[visual_id] = 1
                video_id += 1
            else:
                raise ValueError(f"unsupported Bernini message type: {message_type}")
            visual_id += 1
        flush(previous_has_loss)

        input_ids: list[int] = []
        attention_mask: list[int] = []
        for message in messages:
            role_ids = self.tokenizer.encode(
                f"<|im_start|>{message['role']}\n",
                add_special_tokens=False,
            )
            content_ids = self.tokenizer.encode(
                str(message["content"]).strip(),
                add_special_tokens=False,
            )
            message_ids = role_ids + content_ids
            input_ids.extend(message_ids)
            attention_mask.extend([1] * len(message_ids))

        ids = torch.tensor(input_ids, dtype=torch.long)
        token_types = torch.zeros_like(ids, dtype=torch.int)
        token_segment_ids = torch.arange(len(ids), dtype=torch.long)
        visual_input_mask = torch.zeros_like(ids, dtype=torch.bool)
        visual_output_mask = torch.zeros_like(ids, dtype=torch.bool)

        for current_id, token_id in enumerate(self.visual_input_ids):
            selected = ids == token_id
            if not selected.any():
                continue
            token_types[selected] = 2
            visual_input_mask[selected] = True
            token_segment_ids[selected] = current_id + 1
            ids[selected] = (
                self.image_pad_id
                if visual_id_to_type[current_id] == 0
                else self.video_pad_id
            )

        for current_id, token_id in enumerate(self.visual_output_ids):
            selected = ids == token_id
            if not selected.any():
                continue
            token_types[selected] = 3
            visual_output_mask[selected] = True
            token_segment_ids[selected] = current_id + 1
            ids[selected] = (
                self.image_pad_id
                if visual_id_to_type[current_id] == 0
                else self.video_pad_id
            )

        attention_4d = build_custom_attention_mask(
            token_types.unsqueeze(0),
            token_segment_ids.unsqueeze(0),
            dtype=mask_dtype,
        )
        return {
            "input_ids": ids,
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "attention_mask_4d": attention_4d,
            "visual_input_token_mask": visual_input_mask,
            "visual_output_token_mask": visual_output_mask,
            "vit_type_list": torch.tensor(visual_types, dtype=torch.long),
            "vit_source_indices": torch.tensor(visual_source_indices, dtype=torch.long),
        }


def build_conversation(
    prompt: str,
    *,
    source_videos: int,
    source_images: int,
    output_is_image: bool,
) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = [
        {"type": "special_token", "text": "[CLS]", "has_loss": 0}
    ]
    messages.extend({"type": "video", "has_loss": 0} for _ in range(source_videos))
    messages.extend({"type": "image", "has_loss": 0} for _ in range(source_images))
    messages.append({"type": "text", "text": prompt, "has_loss": 0})
    messages.append(
        {
            "type": "image_gen" if output_is_image else "video_gen",
            "has_loss": 1,
        }
    )
    return messages
