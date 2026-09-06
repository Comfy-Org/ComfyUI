from dataclasses import dataclass
from typing import Callable

import torch

from .conditioning import (
    block_causal_mask,
    condition_input_ids,
    preprocess_references,
    thw_indexes,
)
from .model import MERGED_PATCH_SIZE


IMAGE_START_TOKEN_ID = 151670
EOS_TOKEN_ID = 151645


@dataclass
class InterleavePrefix:
    """Autoregressive KV state for one conditioning branch."""

    keys: list
    values: list
    time: torch.Tensor


@dataclass
class InterleaveResult:
    """Text, images, and token metadata produced by an interleave session."""

    text: str
    images: list[torch.Tensor]
    token_ids: list[int]
    stop_reason: str


class SenseNovaInterleaveSession:
    """Generate SenseNova text and image events while retaining both KV prefixes."""

    def __init__(
        self,
        model,
        positive_prefix: tuple,
        negative_prefix: tuple,
        decode_tokens: Callable[[list[int]], str],
        transformer_options=None,
    ):
        self.model = model
        self.decode_tokens = decode_tokens
        self.transformer_options = transformer_options
        hidden, keys, values, time = model._preprocess_prefix_state(
            *positive_prefix, transformer_options
        )
        self.positive = InterleavePrefix(keys, values, time)
        self.next_token = model._next_text_token(hidden)
        _, keys, values, time = model._preprocess_prefix_state(
            *negative_prefix, transformer_options
        )
        self.negative = InterleavePrefix(keys, values, time)

    def _append_token(self, prefix, token):
        hidden, prefix.keys, prefix.values, prefix.time = self.model._decode_text_token(
            token,
            prefix.keys,
            prefix.values,
            prefix.time,
            self.transformer_options,
        )
        return hidden

    def _append_image(self, prefix, image):
        hidden, prefix.keys, prefix.values, prefix.time = (
            self.model.append_interleave_image(
                image,
                prefix.keys,
                prefix.values,
                prefix.time,
                self.transformer_options,
            )
        )
        return hidden

    def generate(
        self,
        sample_image,
        max_text_tokens,
        max_images,
        progress=None,
        interrupt=None,
    ):
        """Run autoregressive decoding and invoke ``sample_image`` on image events."""

        text = ""
        token_ids = []
        chunk_tokens = []
        images = []
        stop_reason = "eos"
        text_token_count = 0

        def flush_text():
            nonlocal text
            if chunk_tokens:
                text += self.decode_tokens(chunk_tokens)
                chunk_tokens.clear()

        while True:
            if interrupt is not None:
                interrupt()
            token_id = int(self.next_token.item())
            token_ids.append(token_id)
            if token_id == EOS_TOKEN_ID:
                flush_text()
                break
            if token_id == IMAGE_START_TOKEN_ID:
                flush_text()
                if len(images) >= max_images:
                    stop_reason = "max_images"
                    break
                self._append_token(self.positive, self.next_token)
                self._append_token(self.negative, self.next_token)
                image = sample_image(self.positive, self.negative)
                images.append(image)
                text += "<image>"
                hidden = self._append_image(self.positive, image)
                self._append_image(self.negative, image)
                self.next_token = self.model._next_text_token(hidden)
                continue

            text_token_count += 1
            chunk_tokens.append(token_id)
            hidden = self._append_token(self.positive, self.next_token)
            if progress is not None:
                progress(text_token_count)
            if text_token_count >= max_text_tokens:
                flush_text()
                stop_reason = "max_text_tokens"
                break
            self.next_token = self.model._next_text_token(hidden)

        return InterleaveResult(text, images, token_ids, stop_reason)


def live_conditioning(prefix):
    """Convert a live interleave KV prefix into ComfyUI conditioning."""

    return [
        [
            None,
            {
                "prefix_keys": prefix.keys,
                "prefix_values": prefix.values,
                "prefix_time": prefix.time,
            },
        ]
    ]


def prefix_arguments(metadata, device, dtype, image_only):
    """Prepare text and reference-image inputs for prefix preprocessing."""

    input_ids = metadata["text_input_ids"]
    references = metadata.get("reference_latents")
    if references:
        references = preprocess_references(references)
        reference_grids = [
            (
                max(
                    1,
                    (image.shape[-2] + MERGED_PATCH_SIZE - 1)
                    // MERGED_PATCH_SIZE,
                ),
                max(
                    1,
                    (image.shape[-1] + MERGED_PATCH_SIZE - 1)
                    // MERGED_PATCH_SIZE,
                ),
            )
            for image in references
        ]
        input_ids = condition_input_ids(
            input_ids,
            reference_grids,
            image_only=image_only,
            append_image_start=not image_only,
        )
        indexes = thw_indexes(input_ids, reference_grids)
        prefix_mask = block_causal_mask(indexes, dtype=dtype)
        references = [
            image.to(device=device, dtype=dtype) for image in references
        ]
        indexes = indexes.to(device=device)
        prefix_mask = prefix_mask.to(device=device)
    else:
        references = None
        indexes = None
        prefix_mask = None
    return input_ids.to(device=device), references, indexes, prefix_mask


def _parse_interleave_parts(text, num_images):
    parts = []
    image_index = 0
    in_think = False
    cursor = 0
    tags = ("<think>", "</think>", "<image>")

    while cursor < len(text):
        matches = [
            (index, tag)
            for tag in tags
            if (index := text.find(tag, cursor)) >= 0
        ]
        if not matches:
            value = text[cursor:].strip()
            if value:
                parts.append(
                    {"type": "think" if in_think else "text", "text": value}
                )
            break
        index, tag = min(matches, key=lambda value: value[0])
        value = text[cursor:index].strip()
        if value:
            parts.append(
                {"type": "think" if in_think else "text", "text": value}
            )
        cursor = index + len(tag)
        if tag == "<think>":
            in_think = True
        elif tag == "</think>":
            in_think = False
        else:
            image_part = {"type": "image", "index": image_index}
            if image_index >= num_images:
                image_part["missing"] = True
            parts.append(image_part)
            image_index += 1

    while image_index < num_images:
        parts.append({"type": "image", "index": image_index})
        image_index += 1
    return parts


def build_interleave_result(result):
    """Serialize an interleave result into ordered frontend-friendly parts."""

    parts = _parse_interleave_parts(result.text, len(result.images))
    think_text = "\n\n".join(
        part["text"] for part in parts if part["type"] == "think"
    )
    return {
        "version": 1,
        "parts": parts,
        "text": result.text,
        "think_text": think_text,
        "token_ids": result.token_ids,
        "num_images": len(result.images),
        "stop_reason": result.stop_reason,
    }


def interleave_result_to_markdown(result, include_think=True):
    """Render serialized interleave parts as markdown with optional thinking."""

    blocks = []
    for part in result.get("parts", []):
        part_type = part.get("type")
        if part_type == "think" and include_think:
            blocks.append(
                "<details><summary>think</summary>\n\n"
                f"{part.get('text', '')}\n\n</details>"
            )
        elif part_type == "text":
            text = str(part.get("text", "")).strip()
            if text:
                blocks.append(text)
        elif part_type == "image":
            blocks.append(f"[image:{int(part.get('index', 0))}]")
    return "\n\n".join(blocks)
