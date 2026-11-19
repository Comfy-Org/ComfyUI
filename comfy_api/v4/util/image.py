"""Images and latents: resizing, running a function over tiles, and opening image files."""

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Protocol

import torch

import comfy.utils as _comfy_utils
import node_helpers as _node_helpers

# Arguments go to ComfyUI by name, so a parameter renamed there fails here instead of shifting.

MAX_RESOLUTION: int
"""The largest width or height most of ComfyUI's own nodes accept; a few accept more."""


# Read where ComfyUI defines it, on access: importing nodes loads the engine.
def __getattr__(name: str):
    if name == "MAX_RESOLUTION":
        import nodes

        return nodes.MAX_RESOLUTION
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class ProgressCounter(Protocol):
    """What a `pbar` argument provides, as ComfyUI's progress bar does."""

    def update(self, value: float) -> None: ...


def common_upscale(
    samples: torch.Tensor, width: int, height: int, upscale_method: str, crop: str
) -> torch.Tensor:
    """Resize `samples`, channels first (batch, channels, ..., height, width), to `width` x `height`.

    `upscale_method` is `"bislerp"`, `"lanczos"`, or a 2-d `torch.nn.functional.interpolate`
    mode such as `"nearest-exact"`, `"bilinear"`, `"bicubic"` or `"area"`. `"bislerp"`
    interpolates each pixel's channels spherically. `"lanczos"` is Pillow's filter: the images
    go through 8 bits, so values come back clipped to 0..1 in steps of 1/255, and a 4-d input
    of one channel comes back as (batch, height, width). `crop="center"` first crops to the
    new aspect ratio; any other value does not crop.
    """
    return _comfy_utils.common_upscale(
        samples=samples,
        width=width,
        height=height,
        upscale_method=upscale_method,
        crop=crop,
    )


def _refuse_sizes(tiles: Iterable[int | None], overlaps: Iterable[int | None]) -> None:
    """Raise `ValueError` for a tile size that is not positive or an overlap that is negative,
    which no input makes valid."""
    tiles, overlaps = tuple(tiles), tuple(overlaps)
    if any(tile is not None and tile <= 0 for tile in tiles) or any(
        overlap is not None and overlap < 0 for overlap in overlaps
    ):
        raise ValueError(
            f"tile sizes must be positive and overlaps not negative, not {tiles} and {overlaps}"
        )


def _refuse_overlaps(
    sizes: Sequence[int], tiles: Sequence[int], overlap: int | Sequence[int]
) -> None:
    """Raise `ValueError` for an overlap as large as its tile along a dimension longer than
    the tile: ComfyUI then steps no tile along it, and gives a result that is not a number."""
    overlaps = overlap if isinstance(overlap, Sequence) else (overlap,) * len(tiles)
    if any(
        size > tile and lapped >= tile
        for size, tile, lapped in zip(sizes, tiles, overlaps)
    ):
        raise ValueError(
            f"an overlap must be below its tile size, not {tuple(overlaps)} for {tuple(tiles)}"
        )


def tiled_scale(
    samples: torch.Tensor,
    function: Callable[[torch.Tensor], torch.Tensor],
    tile: Sequence[int] = (64, 64),
    overlap: int | Sequence[int] = 8,
    upscale_amount: float | Sequence[float] = 4,
    out_channels: int = 3,
    output_device: str | torch.device = "cpu",
    downscale: bool = False,
    index_formulas: float | Sequence[float] | None = None,
    pbar: ProgressCounter | None = None,
) -> torch.Tensor:
    """Run `function` on overlapping tiles of `samples` and blend the results.

    `samples` is (batch, channels, *dims), an image being (batch, channels, height, width), with
    one `tile` size per dim; `overlap`, `upscale_amount` and `index_formulas` are one value or
    one per dim. `function` gets one tile of one sample, (1, channels, *tile), and returns it
    `upscale_amount` times the size, or divided by it with `downscale`, with `out_channels`
    channels. A tile's start is multiplied (or divided, with `downscale`) by `index_formulas`,
    by default `upscale_amount`, to place its result. The result is float32 on `output_device`,
    made under `torch.inference_mode()`: clone it before changing it in place or using it with
    autograd. `pbar` advances once per tile. Raises `ValueError` for a tile size that is not
    positive, a negative overlap, or an overlap as large as its tile along a dimension that
    is longer than the tile.
    """
    _refuse_sizes(tile, overlap if isinstance(overlap, Sequence) else (overlap,))
    _refuse_overlaps(samples.shape[2:], tile, overlap)
    # The native helper only recognizes list/tuple as a per-dim vector; any other Sequence (such
    # as a range) would be treated as a scalar instead and fail during its own arithmetic.
    overlap = tuple(overlap) if isinstance(overlap, Sequence) else overlap
    upscale_amount = (
        tuple(upscale_amount) if isinstance(upscale_amount, Sequence) else upscale_amount
    )
    index_formulas = (
        tuple(index_formulas) if isinstance(index_formulas, Sequence) else index_formulas
    )
    return _comfy_utils.tiled_scale_multidim(
        samples=samples,
        function=function,
        tile=tuple(tile),
        overlap=overlap,
        upscale_amount=upscale_amount,
        out_channels=out_channels,
        output_device=output_device,
        downscale=downscale,
        index_formulas=index_formulas,
        pbar=pbar,
    )


def get_tiled_scale_steps(size: Sequence[int], tile: Sequence[int], overlap: int) -> int:
    """Return how many tiles `tiled_scale` runs for one image of `size`, (height, width), with
    `tile` in the same order; sizes are refused as `tiled_scale` refuses them."""
    (height, width), (tile_y, tile_x) = size, tile
    _refuse_sizes(tile, (overlap,))
    _refuse_overlaps(size, tile, overlap)
    return _comfy_utils.get_tiled_scale_steps(
        width=width, height=height, tile_x=tile_x, tile_y=tile_y, overlap=overlap
    )


def pillow(function: Callable[[Any], Any], arg: Any) -> Any:
    """Return `function(arg)`; on `OSError` or `ValueError`, call it again with truncated images allowed.

    Any error of these types, such as `FileNotFoundError`, causes the second call, whose error is
    raised. ComfyUI turns Pillow's truncated-images setting on for the second call and puts back
    what it was; the setting is the whole process's, so calls from two threads at once can leave
    it on.
    """
    return _node_helpers.pillow(fn=function, arg=arg)


__all__ = [
    "MAX_RESOLUTION",
    "ProgressCounter",
    "common_upscale",
    "get_tiled_scale_steps",
    "pillow",
    "tiled_scale",
]
