"""Shared source-media geometry matching Bernini's VAE transform."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypeVar

SourceT = TypeVar("SourceT")


def fit_media_size(
    height: int,
    width: int,
    *,
    max_size: int,
    min_size: int = 240,
    stride: int = 16,
) -> tuple[int, int]:
    """Preserve aspect, cap the long edge, floor the short edge, and snap."""

    def divisible(value: float) -> int:
        return max(stride, int(round(value / stride) * stride))

    scale = min(max_size / max(width, height), 1.0)
    scale = max(scale, min_size / min(width, height))
    resized_width = divisible(round(width * scale))
    resized_height = divisible(round(height * scale))
    if max(resized_width, resized_height) > max_size:
        scale = max_size / max(resized_width, resized_height)
        resized_width = max(stride, math.floor(resized_width * scale / stride) * stride)
        resized_height = max(
            stride, math.floor(resized_height * scale / stride) * stride
        )
    return resized_height, resized_width


def ordered_renderer_sources(
    *,
    image_sources: Sequence[SourceT],
    video_sources: Sequence[SourceT],
) -> list[SourceT]:
    """Pack renderer contexts in the release's image-then-video order."""
    return [*image_sources, *video_sources]
