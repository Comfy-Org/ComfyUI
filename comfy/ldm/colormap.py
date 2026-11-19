"""Colormap utilities for depth and geometry visualisation."""

from __future__ import annotations

import torch


def turbo(x: torch.Tensor) -> torch.Tensor:
    """Anton Mikhailov polynomial approximation of the Turbo colormap.

    Args:
        x: Float tensor with values in [0, 1].

    Returns:
        RGB tensor of the same shape as ``x`` with a trailing size-3 dimension.
    """
    x = x.clamp(0.0, 1.0)
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    r = 0.13572138 + 4.61539260*x - 42.66032258*x2 + 132.13108234*x3 - 152.94239396*x4 + 59.28637943*x5
    g = 0.09140261 + 2.19418839*x + 4.84296658*x2 - 14.18503333*x3 +   4.27729857*x4 +  2.82956604*x5
    b = 0.10667330 + 12.64194608*x - 60.58204836*x2 + 110.36276771*x3 - 89.90310912*x4 + 27.34824973*x5
    return torch.stack([r, g, b], dim=-1).clamp(0.0, 1.0)
