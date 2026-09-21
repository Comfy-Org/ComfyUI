from dataclasses import dataclass

import torch

from .conditioning import (
    block_causal_mask,
    condition_input_ids,
    preprocess_references,
    thw_indexes,
)
from .model import MERGED_PATCH_SIZE


IMAGE_CONTEXT_TOKEN_ID = 151669
IMAGE_START_TOKEN_ID = 151670
IMAGE_END_TOKEN_ID = 151671
EOS_TOKEN_ID = 151645


@dataclass
class InterleaveTextEvent:
    """One text-generation stage ending at an image request or stop condition."""

    token_ids: list[int]
    stop_reason: str

    @property
    def image_requested(self):
        return self.stop_reason == "image"


def generate_text_event(
    model,
    prefix_args,
    max_text_tokens,
    transformer_options=None,
    progress=None,
    interrupt=None,
):
    """Decode one interleave text stage without carrying KV state across nodes."""

    if not model.has_lm_head:
        raise RuntimeError(
            "This SenseNova checkpoint does not contain language_model.lm_head.weight, "
            "which is required for interleaved generation."
        )
    if prefix_args[0].shape[0] != 1:
        raise ValueError("SenseNova interleave currently requires a single prompt batch.")

    hidden, prefix_keys, prefix_values, prefix_time = model._preprocess_prefix_state(
        *prefix_args, transformer_options
    )
    next_token = model._next_text_token(hidden)
    token_ids = []
    for step in range(max_text_tokens):
        if interrupt is not None:
            interrupt()
        token_id = int(next_token.item())
        if token_id == EOS_TOKEN_ID:
            return InterleaveTextEvent(token_ids, "eos")
        token_ids.append(token_id)
        if token_id == IMAGE_START_TOKEN_ID:
            return InterleaveTextEvent(token_ids, "image")
        hidden, prefix_keys, prefix_values, prefix_time = model._decode_text_token(
            next_token,
            prefix_keys,
            prefix_values,
            prefix_time,
            transformer_options,
        )
        if progress is not None:
            progress(step + 1)
        next_token = model._next_text_token(hidden)
    return InterleaveTextEvent(token_ids, "max_text_tokens")


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
        if not metadata.get("sensenova_interleave_expanded"):
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


def expand_interleave_metadata(metadata, image_only):
    """Create a reconstructable token/image history for interleave stages."""

    expanded = dict(metadata)
    if expanded.get("sensenova_interleave_expanded"):
        return expanded

    input_ids = expanded["text_input_ids"]
    references = expanded.get("reference_latents") or []
    references = [
        image
        for value in references
        for image in (
            value.unsqueeze(0) if value.ndim == 3 else value
        ).split(1)
    ]
    if references:
        reference_grids = [
            (
                max(1, (image.shape[-3] + MERGED_PATCH_SIZE - 1) // MERGED_PATCH_SIZE),
                max(1, (image.shape[-2] + MERGED_PATCH_SIZE - 1) // MERGED_PATCH_SIZE),
            )
            for image in references
        ]
        input_ids = condition_input_ids(
            input_ids,
            reference_grids,
            image_only=image_only,
            append_image_start=False,
        )

    expanded.update(
        {
            "text_input_ids": input_ids,
            "reference_latents": references,
            "sensenova_interleave_expanded": True,
            "sensenova_interleave_pending_image": False,
        }
    )
    return expanded


def append_interleave_tokens(metadata, token_ids, image_requested):
    """Append generated text tokens to an expanded interleave history."""

    updated = dict(metadata)
    if token_ids:
        suffix = updated["text_input_ids"].new_tensor(token_ids).unsqueeze(0)
        updated["text_input_ids"] = torch.cat(
            (updated["text_input_ids"], suffix), dim=1
        )
    updated["sensenova_interleave_pending_image"] = image_requested
    return updated


def append_interleave_image(metadata, latent_samples):
    """Append one sampled image to an expanded interleave history."""

    if not metadata.get("sensenova_interleave_pending_image"):
        raise ValueError("SenseNova interleave received an image without an image request.")
    if latent_samples.ndim != 4 or latent_samples.shape[0] != 1:
        raise ValueError("SenseNova interleave requires one BCHW latent image per stage.")

    image = (latent_samples * 0.5 + 0.5).movedim(1, -1)
    token_height = max(
        1, (latent_samples.shape[-2] + MERGED_PATCH_SIZE - 1) // MERGED_PATCH_SIZE
    )
    token_width = max(
        1, (latent_samples.shape[-1] + MERGED_PATCH_SIZE - 1) // MERGED_PATCH_SIZE
    )
    suffix = metadata["text_input_ids"].new_tensor(
        [IMAGE_CONTEXT_TOKEN_ID] * (token_height * token_width)
        + [IMAGE_END_TOKEN_ID]
    ).unsqueeze(0)
    updated = dict(metadata)
    updated["text_input_ids"] = torch.cat(
        (metadata["text_input_ids"], suffix), dim=1
    )
    updated["reference_latents"] = [
        *(metadata.get("reference_latents") or []),
        image,
    ]
    updated["sensenova_interleave_pending_image"] = False
    return updated


def _parse_interleave_parts(text, num_images, include_context=False):
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
            if include_context:
                image_part["in_think"] = in_think
            if image_index >= num_images:
                image_part["missing"] = True
            parts.append(image_part)
            image_index += 1

    while image_index < num_images:
        parts.append({"type": "image", "index": image_index})
        image_index += 1
    return parts


def split_interleave_text(text):
    """Split serialized interleave output into visible text and thinking."""

    text = str(text or "")
    if "<think>" not in text and "</think>" not in text:
        return text, ""
    if "<think>" not in text and "</think>" in text:
        text = f"<think>{text}"

    visible_parts = []
    thinking_parts = []
    for part in _parse_interleave_parts(
        text, text.count("<image>"), include_context=True
    ):
        part_type = part["type"]
        if part_type == "think":
            thinking_parts.append(part["text"])
        elif part_type == "text":
            visible_parts.append(part["text"])
        elif part_type == "image" and not part["in_think"]:
            visible_parts.append("<image>")
    return "\n\n".join(visible_parts), "\n\n".join(thinking_parts)
