# Copyright (c) 2026 ByteDance Ltd. and/or its affiliate
# SPDX-License-Identifier: Apache-2.0
"""Framework-independent Bernini v2 renderer guidance math."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def guidance_chunks(names: list[str], batch_size: str | int) -> list[list[str]]:
    """Split condition arms without changing their evaluation or composition order."""

    if batch_size == "all":
        size = len(names)
    elif batch_size == "auto":
        # Video arms have very large activations. One arm at a time is the safe
        # default; users can explicitly trade memory for throughput.
        size = 1
    else:
        try:
            size = int(batch_size)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid guidance batch size: {batch_size!r}") from error
    if size < 1:
        raise ValueError("guidance batch size must be at least 1")
    return [names[index : index + size] for index in range(0, len(names), size)]


def unipc_flow_sigmas(steps: int, shift: float) -> torch.Tensor:
    """Return the released Diffusers UniPC flow schedule without importing Diffusers."""
    raw = torch.linspace(0.999, 0.0, steps + 1, dtype=torch.float32)[:-1]
    shifted = shift * raw / (1.0 + (shift - 1.0) * raw)
    return torch.cat([shifted, shifted.new_zeros(1)])


def append_dims(value: torch.Tensor, ndim: int) -> torch.Tensor:
    """Append singleton dimensions until ``value`` broadcasts over a latent."""
    if value.ndim > ndim:
        raise ValueError(
            f"cannot broadcast a {value.ndim}D sigma over a {ndim}D latent"
        )
    return value.reshape(value.shape + (1,) * (ndim - value.ndim))


def apg_delta(
    delta: torch.Tensor,
    reference: torch.Tensor,
    *,
    parallel_scale: float = 0.2,
    orthogonal_scale: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Project a guidance delta exactly as Bernini's renderer does."""
    batch = delta.shape[0]
    delta_flat = delta.reshape(batch, -1)
    reference_flat = reference.reshape(batch, -1)
    norm_squared = (
        (reference_flat * reference_flat).sum(dim=1, keepdim=True).clamp_min(eps)
    )
    coefficient = (delta_flat * reference_flat).sum(dim=1, keepdim=True) / norm_squared
    parallel = coefficient * reference_flat
    orthogonal = delta_flat - parallel
    return parallel_scale * parallel.reshape_as(
        delta
    ) + orthogonal_scale * orthogonal.reshape_as(delta)


def denoised_to_velocity(
    denoised: torch.Tensor,
    sample: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    sigma = append_dims(sigma, sample.ndim).clamp_min(torch.finfo(torch.float32).eps)
    return (sample - denoised) / sigma


def velocity_to_denoised(
    velocity: torch.Tensor,
    sample: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    return sample - append_dims(sigma, sample.ndim) * velocity


def compose_velocity_guidance(
    predictions: Mapping[str, torch.Tensor],
    *,
    omega_video: float,
    omega_image: float,
    omega_text: float,
    omega_target: float,
    rv2v: bool,
) -> torch.Tensor:
    """Compose official Bernini renderer arms in flow-velocity space.

    Required arms are ``base``, ``text`` and ``target``. ``source`` is the
    combined source-media arm used by all tasks except rv2v. The rv2v chain
    instead uses separate optional ``video`` and ``image`` arms.
    """
    base = predictions["base"]
    if rv2v:
        video = predictions.get("video", base)
        image = predictions.get("image", video)
        text = predictions["text"]
        target = predictions["target"]
        # Despite the upstream mode name rv2v_wapg, the released inference
        # code uses direct chained deltas for this branch.
        return (
            base
            + omega_video * (video - base)
            + omega_image * (image - video)
            + omega_text * (text - image)
            + omega_target * (target - text)
        )

    source = predictions.get("source", base)
    text = predictions["text"]
    target = predictions["target"]
    return (
        base
        + omega_image * apg_delta(source - base, source)
        + omega_text * apg_delta(text - source, text)
        + omega_target * apg_delta(target - text, target)
    )


def compose_denoised_guidance(
    predictions: Mapping[str, torch.Tensor],
    sample: torch.Tensor,
    sigma: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    velocities = {
        name: denoised_to_velocity(prediction, sample, sigma)
        for name, prediction in predictions.items()
    }
    guided = compose_velocity_guidance(velocities, **kwargs)
    return velocity_to_denoised(guided, sample, sigma)
