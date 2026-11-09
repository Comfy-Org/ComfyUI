from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger(__name__)


CurvePoint = tuple[float, float]


class CurveInput(ABC):
    """Abstract base class for curve inputs.

    Subclasses represent different curve representations (control-point
    interpolation, analytical functions, LUT-based, etc.) while exposing a
    uniform evaluation interface to downstream nodes.
    """

    @property
    @abstractmethod
    def points(self) -> list[CurvePoint]:
        """The control points that define this curve."""

    @abstractmethod
    def interp(self, x: float) -> float:
        """Evaluate the curve at a single *x* value in [0, 1]."""

    def interp_array(self, xs: np.ndarray) -> np.ndarray:
        """Vectorised evaluation over a numpy array of x values.

        Subclasses should override this for better performance. The default
        falls back to scalar ``interp`` calls.
        """
        return np.fromiter((self.interp(float(x)) for x in xs), dtype=np.float64, count=len(xs))

    def to_lut(self, size: int = 256) -> np.ndarray:
        """Generate a float64 lookup table of *size* evenly-spaced samples in [0, 1]."""
        return self.interp_array(np.linspace(0.0, 1.0, size))

    @staticmethod
    def from_raw(data) -> CurveInput:
        """Convert raw curve data (dict or point list) to a CurveInput instance.

        Accepts:
        - A ``CurveInput`` instance (returned as-is).
        - A dict with ``"points"`` and optional ``"interpolation"`` keys.
        - A bare list/sequence of ``(x, y)`` pairs (defaults to monotone cubic).
        """
        if isinstance(data, CurveInput):
            return data
        if isinstance(data, dict):
            raw_points = data["points"]
            interpolation = data.get("interpolation", "monotone_cubic")
        else:
            raw_points = data
            interpolation = "monotone_cubic"
        points = [(float(x), float(y)) for x, y in raw_points]
        if interpolation == "linear":
            return LinearCurve(points)
        if interpolation != "monotone_cubic":
            logger.warning("Unknown curve interpolation %r, falling back to monotone_cubic", interpolation)
        return MonotoneCubicCurve(points)


class MonotoneCubicCurve(CurveInput):
    """Monotone cubic Hermite interpolation over control points.

    Mirrors the frontend ``createMonotoneInterpolator`` in
    ``ComfyUI_frontend/src/components/curve/curveUtils.ts`` so that
    backend evaluation matches the editor preview exactly.

    All heavy work (sorting, slope computation) happens once at construction.
    ``interp_array`` is fully vectorised with numpy.
    """

    def __init__(self, control_points: list[CurvePoint]):
        sorted_pts = sorted(control_points, key=lambda p: p[0])
        self._points = [(float(x), float(y)) for x, y in sorted_pts]
        self._xs = np.array([p[0] for p in self._points], dtype=np.float64)
        self._ys = np.array([p[1] for p in self._points], dtype=np.float64)
        self._slopes = self._compute_slopes()

    @property
    def points(self) -> list[CurvePoint]:
        return list(self._points)

    def _compute_slopes(self) -> np.ndarray:
        xs, ys = self._xs, self._ys
        n = len(xs)
        if n < 2:
            return np.zeros(n, dtype=np.float64)

        dx = np.diff(xs)
        dy = np.diff(ys)
        dx_safe = np.where(dx == 0, 1.0, dx)
        deltas = np.where(dx == 0, 0.0, dy / dx_safe)

        slopes = np.empty(n, dtype=np.float64)
        slopes[0] = deltas[0]
        slopes[-1] = deltas[-1]
        for i in range(1, n - 1):
            if deltas[i - 1] * deltas[i] <= 0:
                slopes[i] = 0.0
            else:
                slopes[i] = (deltas[i - 1] + deltas[i]) / 2

        for i in range(n - 1):
            if deltas[i] == 0:
                slopes[i] = 0.0
                slopes[i + 1] = 0.0
            else:
                alpha = slopes[i] / deltas[i]
                beta = slopes[i + 1] / deltas[i]
                s = alpha * alpha + beta * beta
                if s > 9:
                    t = 3 / math.sqrt(s)
                    slopes[i] = t * alpha * deltas[i]
                    slopes[i + 1] = t * beta * deltas[i]
        return slopes

    def interp(self, x: float) -> float:
        xs, ys, slopes = self._xs, self._ys, self._slopes
        n = len(xs)
        if n == 0:
            return 0.0
        if n == 1:
            return float(ys[0])
        if x <= xs[0]:
            return float(ys[0])
        if x >= xs[-1]:
            return float(ys[-1])

        hi = int(np.searchsorted(xs, x, side='right'))
        hi = min(hi, n - 1)
        lo = hi - 1

        dx = xs[hi] - xs[lo]
        if dx == 0:
            return float(ys[lo])

        t = (x - xs[lo]) / dx
        t2 = t * t
        t3 = t2 * t
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        return float(h00 * ys[lo] + h10 * dx * slopes[lo] + h01 * ys[hi] + h11 * dx * slopes[hi])

    def interp_array(self, xs_in: np.ndarray) -> np.ndarray:
        """Fully vectorised evaluation using numpy."""
        xs, ys, slopes = self._xs, self._ys, self._slopes
        n = len(xs)
        if n == 0:
            return np.zeros_like(xs_in, dtype=np.float64)
        if n == 1:
            return np.full_like(xs_in, ys[0], dtype=np.float64)

        hi = np.searchsorted(xs, xs_in, side='right').clip(1, n - 1)
        lo = hi - 1

        dx = xs[hi] - xs[lo]
        dx_safe = np.where(dx == 0, 1.0, dx)
        t = np.where(dx == 0, 0.0, (xs_in - xs[lo]) / dx_safe)
        t2 = t * t
        t3 = t2 * t

        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2

        result = h00 * ys[lo] + h10 * dx * slopes[lo] + h01 * ys[hi] + h11 * dx * slopes[hi]
        result = np.where(xs_in <= xs[0], ys[0], result)
        result = np.where(xs_in >= xs[-1], ys[-1], result)
        return result

    def __repr__(self) -> str:
        return f"MonotoneCubicCurve(points={self._points})"


class LinearCurve(CurveInput):
    """Piecewise linear interpolation over control points.

    Mirrors the frontend ``createLinearInterpolator`` in
    ``ComfyUI_frontend/src/components/curve/curveUtils.ts``.
    """

    def __init__(self, control_points: list[CurvePoint]):
        sorted_pts = sorted(control_points, key=lambda p: p[0])
        self._points = [(float(x), float(y)) for x, y in sorted_pts]
        self._xs = np.array([p[0] for p in self._points], dtype=np.float64)
        self._ys = np.array([p[1] for p in self._points], dtype=np.float64)

    @property
    def points(self) -> list[CurvePoint]:
        return list(self._points)

    def interp(self, x: float) -> float:
        xs, ys = self._xs, self._ys
        n = len(xs)
        if n == 0:
            return 0.0
        if n == 1:
            return float(ys[0])
        return float(np.interp(x, xs, ys))

    def interp_array(self, xs_in: np.ndarray) -> np.ndarray:
        if len(self._xs) == 0:
            return np.zeros_like(xs_in, dtype=np.float64)
        if len(self._xs) == 1:
            return np.full_like(xs_in, self._ys[0], dtype=np.float64)
        return np.interp(xs_in, self._xs, self._ys)

    def __repr__(self) -> str:
        return f"LinearCurve(points={self._points})"
