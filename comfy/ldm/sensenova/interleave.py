import torch

from .conditioning import (
    IMAGE_CONTEXT_ID,
    IMAGE_END_ID,
    condition_input_ids,
    prepare_prefix_inputs,
    reference_grid,
    split_reference_batches,
)
from .model import MERGED_PATCH_SIZE


def prefix_arguments(metadata, device, dtype):
    """Prepare text and reference-image inputs for prefix preprocessing."""

    input_ids, references, indexes, prefix_mask = prepare_prefix_inputs(
        metadata["text_input_ids"], metadata.get("reference_latents"),
        MERGED_PATCH_SIZE, dtype,
        expand_images=not metadata.get("sensenova_interleave_expanded"),
        generated_start=metadata.get("sensenova_interleave_start"),
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
            reference_grid(image.shape[-3], image.shape[-2], MERGED_PATCH_SIZE)
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
    if latent_samples.shape[1] != 3:
        raise ValueError("SenseNova interleave requires a three-channel RGB pixel latent from KSampler.")

    image = (latent_samples * 0.5 + 0.5).movedim(1, -1)
    token_height, token_width = reference_grid(
        latent_samples.shape[-2], latent_samples.shape[-1], MERGED_PATCH_SIZE
    )
    suffix = input_ids.new_tensor(
        [IMAGE_CONTEXT_ID] * (token_height * token_width)
        + [IMAGE_END_ID]
    ).unsqueeze(0)
    return torch.cat((input_ids, suffix), dim=1), image
