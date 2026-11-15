import math
from typing import NamedTuple, Optional, Union

import numpy as np

EPSILON = 1e-6

LUM_R = 0.2224884
LUM_G = 0.71690369
LUM_B = 0.06060791

ArrayLike = Union[np.ndarray, float]


def srgb_to_linear(c: ArrayLike) -> np.ndarray:
    c = np.asarray(c, dtype=np.float32)
    high = ((np.maximum(c, 0.0) + 0.055) / 1.055) ** 2.4
    return np.where(c <= 0.04045, c / 12.92, high).astype(np.float32)


def linear_to_srgb(c: ArrayLike) -> np.ndarray:
    c = np.asarray(c, dtype=np.float32)
    high = 1.055 * np.maximum(c, 0.0) ** (1.0 / 2.4) - 0.055
    return np.where(c <= 0.0031308, 12.92 * c, high).astype(np.float32)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * LUM_R + rgb[..., 1] * LUM_G + rgb[..., 2] * LUM_B


def safe_div(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    a, b = np.broadcast_arrays(
        np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    )
    out = np.zeros(b.shape, dtype=np.float32)
    np.divide(a, b, out=out, where=np.abs(b) >= EPSILON)
    return out


CHANNEL_BLEND = {
    "normal": lambda i, l: l,
    "multiply": lambda i, l: i * l,
    "screen": lambda i, l: 1 - (1 - i) * (1 - l),
    "overlay": lambda i, l: np.where(i < 0.5, 2 * i * l, 1 - 2 * (1 - l) * (1 - i)),
    "darken": lambda i, l: np.minimum(i, l),
    "lighten": lambda i, l: np.maximum(i, l),
    "color-dodge": lambda i, l: np.where(
        i <= 0,
        0.0,
        np.where(1 - l <= EPSILON, 1.0, np.minimum(safe_div(i, 1 - l), 1.0)),
    ),
    "color-burn": lambda i, l: np.where(
        i >= 1,
        1.0,
        np.where(l <= EPSILON, 0.0, 1 - np.minimum(safe_div(1 - i, l), 1.0)),
    ),
    "hard-light": lambda i, l: np.where(
        l > 0.5,
        np.minimum(1 - (1 - i) * (1 - (l - 0.5) * 2), 1),
        np.minimum(i * (l * 2), 1),
    ),
    "soft-light": lambda i, l: (1 - i) * (i * l) + i * (1 - (1 - i) * (1 - l)),
    "difference": lambda i, l: np.abs(i - l),
    "exclusion": lambda i, l: 0.5 - 2 * (i - 0.5) * (l - 0.5),
    "linear-dodge": lambda i, l: i + l,
    "linear-burn": lambda i, l: i + l - 1,
    "vivid-light": lambda i, l: np.where(
        l <= 0.5,
        np.where(
            i >= 1,
            1.0,
            np.where(
                2 * l <= EPSILON,
                0.0,
                np.maximum(1 - safe_div(1 - i, 2 * l), 0.0),
            ),
        ),
        np.where(
            i <= 0,
            0.0,
            np.where(
                2 * (1 - l) <= EPSILON,
                1.0,
                np.minimum(safe_div(i, 2 * (1 - l)), 1.0),
            ),
        ),
    ),
    "pin-light": lambda i, l: np.where(
        l > 0.5, np.maximum(i, 2 * (l - 0.5)), np.minimum(i, 2 * l)
    ),
    "linear-light": lambda i, l: i + 2 * l - 1,
    "hard-mix": lambda i, l: np.where(i + l < 1, 0.0, 1.0),
    "subtract": lambda i, l: np.maximum(i - l, 0),
    "divide": lambda i, l: np.clip(i / np.maximum(l, EPSILON), 0, 1),
    "grain-extract": lambda i, l: i - l + 0.5,
    "grain-merge": lambda i, l: i + l - 0.5,
}


def _blend_hue(i: np.ndarray, l: np.ndarray) -> np.ndarray:
    src_min = l.min(axis=-1)
    src_max = l.max(axis=-1)
    src_delta = src_max - src_min
    achromatic = src_delta <= EPSILON
    dest_max = i.max(axis=-1)
    dest_delta = dest_max - i.min(axis=-1)
    dest_s = np.where(dest_max != 0, dest_delta / np.where(dest_max != 0, dest_max, 1), 0)
    ratio = np.where(
        achromatic, 0, dest_s * dest_max / np.where(achromatic, 1, src_delta)
    )
    offset = dest_max - src_max * ratio
    return np.where(achromatic[..., None], i, l * ratio[..., None] + offset[..., None])


def _blend_saturation(i: np.ndarray, l: np.ndarray) -> np.ndarray:
    dest_max = i.max(axis=-1)
    dest_delta = dest_max - i.min(axis=-1)
    flat = dest_delta <= EPSILON
    src_max = l.max(axis=-1)
    src_delta = src_max - l.min(axis=-1)
    src_s = np.where(src_max != 0, src_delta / np.where(src_max != 0, src_max, 1), 0)
    ratio = np.where(flat, 0, src_s * dest_max / np.where(flat, 1, dest_delta))
    offset = (1 - ratio) * dest_max
    return np.where(
        flat[..., None],
        np.broadcast_to(dest_max[..., None], i.shape),
        i * ratio[..., None] + offset[..., None],
    )


def _blend_color(i: np.ndarray, l: np.ndarray) -> np.ndarray:
    dest_l = (i.min(axis=-1) + i.max(axis=-1)) / 2
    src_l = (l.min(axis=-1) + l.max(axis=-1)) / 2
    gray = (np.abs(src_l) <= EPSILON) | (np.abs(1 - src_l) <= EPSILON)
    dest_high = dest_l > 0.5
    src_high = src_l > 0.5
    dl = np.minimum(dest_l, 1 - dest_l)
    sl = np.minimum(src_l, 1 - src_l)
    ratio = dl / np.where(gray, 1, sl)
    offset = np.where(dest_high, 1 - 2 * dl, 0) + np.where(src_high, 2 * dl - ratio, 0)
    return np.where(
        gray[..., None],
        np.broadcast_to(dest_l[..., None], i.shape),
        l * ratio[..., None] + offset[..., None],
    )


def _blend_luminosity(i: np.ndarray, l: np.ndarray) -> np.ndarray:
    # Scale the backdrop so it carries the layer's luminance. Where the backdrop
    # has no luminance to scale there is no hue or saturation to preserve either,
    # so the result is a neutral grey at the layer's luminance - which is also the
    # analytic limit of i * lum(l)/lum(i) as a grey backdrop approaches black.
    # Guarding the numerator here instead (returning black) makes a luminosity
    # layer disappear over dark backdrops; see tests-unit/comfy_extras_test/
    # compositor_blend_golden.json.
    lum_i = luminance(i)
    lum_l = luminance(l)
    degenerate = lum_i <= EPSILON
    ratio = np.where(degenerate, 0.0, lum_l / np.where(degenerate, 1.0, lum_i))
    return np.where(
        degenerate[..., None],
        np.broadcast_to(lum_l[..., None], i.shape),
        i * ratio[..., None],
    )


HSL_BLEND = {
    "hue": _blend_hue,
    "saturation": _blend_saturation,
    "color": _blend_color,
    "luminosity": _blend_luminosity,
}


def blend_pixel(blend: str, in_rgb: np.ndarray, layer_rgb: np.ndarray) -> np.ndarray:
    in_rgb = np.asarray(in_rgb, dtype=np.float32)
    layer_rgb = np.asarray(layer_rgb, dtype=np.float32)
    hsl = HSL_BLEND.get(blend)
    if hsl is not None:
        return np.asarray(hsl(in_rgb, layer_rgb), dtype=np.float32)
    fn = CHANNEL_BLEND.get(blend, CHANNEL_BLEND["normal"])
    return np.asarray(fn(in_rgb, layer_rgb), dtype=np.float32)


def _composite_union(in_c, layer, comp, cov):
    in_a = in_c[..., 3]
    layer_a = layer[..., 3] * cov
    new_a = layer_a + (1 - layer_a) * in_a
    ratio = np.where(new_a != 0, layer_a / np.where(new_a != 0, new_a, 1), 0)
    blended = (
        ratio[..., None]
        * (in_a[..., None] * (comp - layer[..., :3]) + layer[..., :3] - in_c[..., :3])
        + in_c[..., :3]
    )
    keep = (layer_a == 0) | (new_a == 0)
    rgb = np.where(
        keep[..., None],
        in_c[..., :3],
        np.where((in_a == 0)[..., None], layer[..., :3], blended),
    )
    return np.concatenate([rgb, new_a[..., None]], axis=-1)


def _composite_clip_to_backdrop(in_c, layer, comp, cov):
    in_a = in_c[..., 3]
    layer_a = layer[..., 3] * cov
    mixed = comp * layer_a[..., None] + in_c[..., :3] * (1 - layer_a[..., None])
    keep = (in_a == 0) | (layer_a == 0)
    rgb = np.where(keep[..., None], in_c[..., :3], mixed)
    return np.concatenate([rgb, in_a[..., None]], axis=-1)


def _composite_clip_to_layer(in_c, layer, comp, cov):
    in_a = in_c[..., 3]
    layer_a = layer[..., 3] * cov
    mixed = comp * in_a[..., None] + layer[..., :3] * (1 - in_a[..., None])
    rgb = np.where(
        (layer_a == 0)[..., None],
        in_c[..., :3],
        np.where((in_a == 0)[..., None], layer[..., :3], mixed),
    )
    return np.concatenate([rgb, layer_a[..., None]], axis=-1)


def _composite_intersection(in_c, layer, comp, cov):
    new_a = in_c[..., 3] * layer[..., 3] * cov
    rgb = np.where((new_a == 0)[..., None], in_c[..., :3], comp)
    return np.concatenate([rgb, new_a[..., None]], axis=-1)


_COMPOSITE = {
    "union": _composite_union,
    "clip-to-backdrop": _composite_clip_to_backdrop,
    "clip-to-layer": _composite_clip_to_layer,
    "intersection": _composite_intersection,
}


def run_composite(mode: str, in_c, layer, comp, cov) -> np.ndarray:
    fn = _COMPOSITE.get(mode, _composite_union)
    return fn(in_c, layer, comp, cov)


def _to_space(rgb: np.ndarray, space: str) -> np.ndarray:
    return rgb if space == "linear" else linear_to_srgb(rgb)


def _from_space(rgb: np.ndarray, space: str) -> np.ndarray:
    return rgb if space == "linear" else srgb_to_linear(rgb)


class EffectiveMode(NamedTuple):
    blend: str
    blend_space: str
    composite: str


_LAYER_MODES = {
    "normal": ("linear", "union"),
    "multiply": ("linear", "clip-to-backdrop"),
    "screen": ("perceptual", "clip-to-backdrop"),
    "overlay": ("perceptual", "clip-to-backdrop"),
    "darken": ("linear", "clip-to-backdrop"),
    "lighten": ("linear", "clip-to-backdrop"),
    "color-dodge": ("perceptual", "clip-to-backdrop"),
    "color-burn": ("perceptual", "clip-to-backdrop"),
    "hard-light": ("perceptual", "clip-to-backdrop"),
    "soft-light": ("perceptual", "clip-to-backdrop"),
    "difference": ("perceptual", "clip-to-backdrop"),
    "exclusion": ("perceptual", "clip-to-backdrop"),
    "linear-dodge": ("linear", "clip-to-backdrop"),
    "linear-burn": ("perceptual", "clip-to-backdrop"),
    "vivid-light": ("perceptual", "clip-to-backdrop"),
    "pin-light": ("perceptual", "clip-to-backdrop"),
    "linear-light": ("perceptual", "clip-to-backdrop"),
    "hard-mix": ("perceptual", "clip-to-backdrop"),
    "subtract": ("linear", "clip-to-backdrop"),
    "divide": ("linear", "clip-to-backdrop"),
    "grain-extract": ("perceptual", "clip-to-backdrop"),
    "grain-merge": ("perceptual", "clip-to-backdrop"),
    "hue": ("perceptual", "clip-to-backdrop"),
    "saturation": ("perceptual", "clip-to-backdrop"),
    "color": ("perceptual", "clip-to-backdrop"),
    "luminosity": ("linear", "clip-to-backdrop"),
}


def resolve_mode(blend: str = "normal") -> EffectiveMode:
    blend_space, composite = _LAYER_MODES.get(blend, _LAYER_MODES["normal"])
    return EffectiveMode(
        blend=blend,
        blend_space=blend_space,
        composite=composite,
    )


def blend_composite(
    mode: EffectiveMode,
    backdrop: np.ndarray,
    layer: np.ndarray,
    opacity: float,
    mask: Optional[ArrayLike] = None,
) -> np.ndarray:
    backdrop = np.asarray(backdrop, dtype=np.float32)
    layer = np.asarray(layer, dtype=np.float32)
    cov = opacity * (1.0 if mask is None else mask)

    in_b = _to_space(backdrop[..., :3], mode.blend_space)
    layer_b = _to_space(layer[..., :3], mode.blend_space)
    comp = _from_space(blend_pixel(mode.blend, in_b, layer_b), mode.blend_space)

    return run_composite(mode.composite, backdrop, layer, comp, cov)


def placed_bounds(
    x: float, y: float, w: float, h: float, rotation: float
) -> tuple[int, int, int, int]:
    cx = x + w / 2
    cy = y + h / 2
    cos = math.cos(rotation)
    sin = math.sin(rotation)
    hw = w / 2
    hh = h / 2
    corners = ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))
    xs = [cx + dx * cos - dy * sin for dx, dy in corners]
    ys = [cy + dx * sin + dy * cos for dx, dy in corners]
    bx = math.floor(min(xs))
    by = math.floor(min(ys))
    bw = max(1, math.ceil(max(xs)) - bx)
    bh = max(1, math.ceil(max(ys)) - by)
    return bx, by, bw, bh
