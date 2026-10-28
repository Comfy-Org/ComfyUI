import torch

from .conditioning import (
    IMAGE_CONTEXT_ID,
    IMAGE_END_ID,
    condition_input_ids,
    prepare_prefix_inputs,
    split_reference_batches,
)
from .model import MERGED_PATCH_SIZE


def prefix_arguments(metadata, device, dtype, image_only):
    """Prepare text and reference-image inputs for prefix preprocessing."""

    input_ids, references, indexes, prefix_mask = prepare_prefix_inputs(
        metadata["text_input_ids"], metadata.get("reference_latents"),
        MERGED_PATCH_SIZE, dtype,
        image_only=image_only,
        expand_images=not metadata.get("sensenova_interleave_expanded"),
        append_image_start=not image_only,
    )
    if references:
        references = [
            image.to(device=device, dtype=dtype) for image in references
        ]
        indexes = indexes.to(device=device)
        prefix_mask = prefix_mask.to(device=device)
    return input_ids.to(device=device), references, indexes, prefix_mask


def expand_interleave_inputs(input_ids, reference_images, image_only):
    """Expand reference-image tokens while retaining unnormalized images."""

    references = split_reference_batches(reference_images or [])
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

    return input_ids, references


def append_image_to_prefix(input_ids, latent_samples):
    """Append sampled image tokens and return the unnormalized reference image."""

    if latent_samples.ndim != 4 or latent_samples.shape[0] != 1:
        raise ValueError("SenseNova interleave requires one BCHW latent image per stage.")

    image = (latent_samples * 0.5 + 0.5).movedim(1, -1)
    token_height = max(
        1, (latent_samples.shape[-2] + MERGED_PATCH_SIZE - 1) // MERGED_PATCH_SIZE
    )
    token_width = max(
        1, (latent_samples.shape[-1] + MERGED_PATCH_SIZE - 1) // MERGED_PATCH_SIZE
    )
    suffix = input_ids.new_tensor(
        [IMAGE_CONTEXT_ID] * (token_height * token_width)
        + [IMAGE_END_ID]
    ).unsqueeze(0)
    return torch.cat((input_ids, suffix), dim=1), image
