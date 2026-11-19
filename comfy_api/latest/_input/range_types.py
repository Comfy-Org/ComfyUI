from __future__ import annotations

import logging
import math
import numpy as np

logger = logging.getLogger(__name__)


class RangeInput:
    """Represents a levels/range adjustment: input range [min, max] with
    optional midpoint (gamma control).

    Generates a 1D LUT identical to GIMP's levels mapping:
        1. Normalize input to [0, 1] using [min, max]
        2. Apply gamma correction: pow(value, 1/gamma)
        3. Clamp to [0, 1]

    The midpoint field is a position in [0, 1] representing where the
    midtone falls within [min, max]. It maps to gamma via:
        gamma = -log2(midpoint)
    So midpoint=0.5 → gamma=1.0 (linear).
    """

    def __init__(self, min_val: float, max_val: float, midpoint: float | None = None):
        self.min_val = min_val
        self.max_val = max_val
        self.midpoint = midpoint

    @staticmethod
    def from_raw(data) -> RangeInput:
        if isinstance(data, RangeInput):
            return data
        if isinstance(data, dict):
            return RangeInput(
                min_val=float(data.get("min", 0.0)),
                max_val=float(data.get("max", 1.0)),
                midpoint=float(data["midpoint"]) if data.get("midpoint") is not None else None,
            )
        raise TypeError(f"Cannot convert {type(data)} to RangeInput")

    def to_lut(self, size: int = 256) -> np.ndarray:
        """Generate a float64 lookup table mapping [0, 1] input through this
        levels adjustment.

        The LUT maps normalized input values (0..1) to output values (0..1),
        matching the GIMP levels formula.
        """
        xs = np.linspace(0.0, 1.0, size, dtype=np.float64)

        in_range = self.max_val - self.min_val
        if abs(in_range) < 1e-10:
            return np.where(xs >= self.min_val, 1.0, 0.0).astype(np.float64)

        # Normalize: map [min, max] → [0, 1]
        result = (xs - self.min_val) / in_range
        result = np.clip(result, 0.0, 1.0)

        # Gamma correction from midpoint
        if self.midpoint is not None and self.midpoint > 0 and self.midpoint != 0.5:
            gamma = max(-math.log2(self.midpoint), 0.001)
            inv_gamma = 1.0 / gamma
            mask = result > 0
            result[mask] = np.power(result[mask], inv_gamma)

        return result

    def __repr__(self) -> str:
        mid = f", midpoint={self.midpoint}" if self.midpoint is not None else ""
        return f"RangeInput(min={self.min_val}, max={self.max_val}{mid})"
