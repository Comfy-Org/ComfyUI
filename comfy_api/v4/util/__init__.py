"""Helpers for node code: image and latent resizing, noise, conditioning, weight files, folders."""

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Protocol

import torch

import comfy.utils as _comfy_utils
import node_helpers as _node_helpers
from comfy_api.latest import io

from . import folders

# Arguments go to ComfyUI by name, so a parameter renamed there fails here instead of shifting.

MAX_RESOLUTION: int
"""The largest width or height most of ComfyUI's own nodes accept; a few accept more."""
SAMPLER_NAMES: list[str]
"""The list of samplers ComfyUI's KSampler offers: ComfyUI's own list, not a copy."""
SCHEDULER_NAMES: list[str]
"""The list of schedulers ComfyUI's KSampler offers: ComfyUI's own list, not a copy."""


# Read where ComfyUI defines them, on access: importing nodes or comfy.samplers loads the engine.
def __getattr__(name: str):
    match name:
        case "MAX_RESOLUTION":
            import nodes

            return nodes.MAX_RESOLUTION
        case "SAMPLER_NAMES":
            from comfy.samplers import KSampler

            return KSampler.SAMPLERS
        case "SCHEDULER_NAMES":
            from comfy.samplers import KSampler

            return KSampler.SCHEDULERS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class ProgressCounter(Protocol):
    """What a `pbar` argument provides, as ComfyUI's progress bar does."""

    def update(self, value: float) -> None: ...


def common_upscale(
    samples: torch.Tensor, width: int, height: int, upscale_method: str, crop: str
) -> torch.Tensor:
    """Resize `samples`, channels first (batch, channels, ..., height, width), to `width` x `height`.

    `upscale_method` is `"bislerp"`, `"lanczos"`, or a 2-d `torch.nn.functional.interpolate`
    mode such as `"nearest-exact"`, `"bilinear"`, `"bicubic"` or `"area"`. With `"lanczos"`, one
    channel comes back as (batch, height, width), as `lanczos` gives it. `crop="center"` first crops
    to the new aspect ratio; any other value does not crop.
    """
    return _comfy_utils.common_upscale(
        samples=samples,
        width=width,
        height=height,
        upscale_method=upscale_method,
        crop=crop,
    )


def bislerp(samples: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Resize (batch, channels, height, width) by spherical interpolation of each pixel's channels."""
    return _comfy_utils.bislerp(samples=samples, width=width, height=height)


def lanczos(samples: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Resize (batch, channels, height, width) images with Pillow's Lanczos filter.

    The images go through 8 bits: values come back clipped to 0..1 in steps of 1/255. One to four
    channels; one channel comes back as (batch, height, width). With two or four channels the last is
    alpha: when the size changes, the others come back 0 where the resized alpha is 0. Raises
    `TypeError` for `bfloat16`.
    """
    return _comfy_utils.lanczos(samples=samples, width=width, height=height)


def _refuse_sizes(tiles: Iterable[int | None], overlaps: Iterable[int | None]) -> None:
    """Raise `ValueError` for a tile size that is not positive or an overlap that is negative:
    ComfyUI would divide by zero or blend the tiles into NaN."""
    tiles, overlaps = tuple(tiles), tuple(overlaps)
    if any(tile is not None and tile <= 0 for tile in tiles) or any(
        overlap is not None and overlap < 0 for overlap in overlaps
    ):
        raise ValueError(
            f"tile sizes must be positive and overlaps not negative, not {tiles} and {overlaps}"
        )


def tiled_scale(
    samples: torch.Tensor,
    function: Callable[[torch.Tensor], torch.Tensor],
    tile_x: int = 64,
    tile_y: int = 64,
    overlap: int = 8,
    upscale_amount: float = 4,
    out_channels: int = 3,
    output_device: str | torch.device = "cpu",
    pbar: ProgressCounter | None = None,
) -> torch.Tensor:
    """Run `function` on overlapping tiles of (batch, channels, height, width) and blend the results.

    `function` gets one tile of one image, (1, channels, height, width), and returns it
    `upscale_amount` times the size with `out_channels` channels. The result is float32 on
    `output_device`, made under `torch.inference_mode()`: clone it before changing it in place or
    using it with autograd. `pbar` advances once per tile. Raises `ValueError` for a tile size
    that is not positive or a negative `overlap`; keep `overlap` below the tile size.
    """
    _refuse_sizes((tile_x, tile_y), (overlap,))
    return _comfy_utils.tiled_scale(
        samples=samples,
        function=function,
        tile_x=tile_x,
        tile_y=tile_y,
        overlap=overlap,
        upscale_amount=upscale_amount,
        out_channels=out_channels,
        output_device=output_device,
        pbar=pbar,
    )


def tiled_scale_multidim(
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
    """Tile and blend as `tiled_scale` does, over every dimension after channels.

    `samples` is (batch, channels, *dims) with one `tile` size per dim; `overlap`, `upscale_amount`
    and `index_formulas` are one value or one per dim. `downscale` divides by `upscale_amount` where
    `tiled_scale` multiplies. A tile's start is multiplied (or divided, with `downscale`) by
    `index_formulas`, by default `upscale_amount`, to place its result. Sizes are refused as
    `tiled_scale` refuses them.
    """
    _refuse_sizes(tile, overlap if isinstance(overlap, Sequence) else (overlap,))
    return _comfy_utils.tiled_scale_multidim(
        samples=samples,
        function=function,
        tile=tile,
        overlap=overlap,
        upscale_amount=upscale_amount,
        out_channels=out_channels,
        output_device=output_device,
        downscale=downscale,
        index_formulas=index_formulas,
        pbar=pbar,
    )


def get_tiled_scale_steps(
    width: int, height: int, tile_x: int, tile_y: int, overlap: int
) -> int:
    """Return how many tiles `tiled_scale` runs for one image of this size; sizes are refused as
    `tiled_scale` refuses them."""
    _refuse_sizes((tile_x, tile_y), (overlap,))
    return _comfy_utils.get_tiled_scale_steps(
        width=width, height=height, tile_x=tile_x, tile_y=tile_y, overlap=overlap
    )


def prepare_noise(
    latent_image: torch.Tensor, seed: int, noise_inds: list[int] | None = None
) -> torch.Tensor:
    """Return the noise ComfyUI's KSampler adds for `seed`: made on the CPU, in the latent's dtype.

    With `noise_inds` (a latent's `"batch_index"`), entry i gets the noise of batch position
    `noise_inds[i]`. Seeds torch's global CPU random generator with `seed`, so later unseeded torch
    random calls in the process follow from it.
    """
    import comfy.sample

    return comfy.sample.prepare_noise(
        latent_image=latent_image, seed=seed, noise_inds=noise_inds
    )


def conditioning_set_values(
    conditioning: io.Conditioning.Type, values: dict[str, Any], append: bool = False
) -> io.Conditioning.Type:
    """Return a copy of `conditioning` with `values` set in every entry's options.

    With `append`, a value is added (`+`) to the one already there. Only the list, the entries and
    the options dicts are new: tensors and every other option value, such as a list or dict, are
    shared with `conditioning`.
    """
    return _node_helpers.conditioning_set_values(
        conditioning=conditioning, values=values, append=append
    )


def conditioning_set_values_with_timestep_range(
    conditioning: io.Conditioning.Type,
    values: dict[str, Any],
    start_percent: float = 0.0,
    end_percent: float = 1.0,
) -> io.Conditioning.Type:
    """Set `values` as `conditioning_set_values` does, from `start_percent` to `end_percent` only.

    An entry's own `"start_percent"`/`"end_percent"` is respected: the part of its range outside
    the requested one keeps its options, as extra entries, and an entry whose range does not
    overlap is put in the result as it is, not copied. If `start_percent` is above `end_percent`,
    `conditioning` itself is returned and a warning is logged.
    """
    return _node_helpers.conditioning_set_values_with_timestep_range(
        conditioning=conditioning,
        values=values,
        start_percent=start_percent,
        end_percent=end_percent,
    )


def pillow(function: Callable[[Any], Any], arg: Any) -> Any:
    """Return `function(arg)`; on `OSError` or `ValueError`, call it again with truncated images allowed.

    Any error of these types, such as `FileNotFoundError`, causes the second call, whose error is
    raised. Pillow's truncated-images setting is process-wide during that call and restored after.
    """
    return _node_helpers.pillow(fn=function, arg=arg)


def load_torch_file(
    ckpt: str,
    safe_load: bool = False,
    device: torch.device | None = None,
    return_metadata: bool = False,
) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, str] | None]:
    """Read a `.safetensors` or `.sft` file, or a torch file, into `{name: tensor}`.

    Torch files are always read with `weights_only=True`, so `safe_load` changes nothing. A
    torch file's `"state_dict"` entry, or its only entry if that is a dict, is what is
    returned. With `return_metadata`, returns `(weights, metadata)`; metadata is None for torch
    files, and None or `{}` for a safetensors file without metadata.
    """
    return _comfy_utils.load_torch_file(
        ckpt=ckpt, safe_load=True, device=device, return_metadata=return_metadata
    )


def save_torch_file(
    sd: dict[str, torch.Tensor], ckpt: str, metadata: dict[str, str] | None = None
) -> None:
    """Write `{name: tensor}` to `ckpt` as safetensors, whatever its extension, with `metadata`.

    Raises `RuntimeError` if tensors share memory, such as tied weights, `ValueError` for a tensor
    that is not contiguous, and `TypeError` for a `metadata` value that is not a string.
    `load_torch_file` reads the file back only under a `.safetensors` or `.sft` name.
    """
    _comfy_utils.save_torch_file(sd=sd, ckpt=ckpt, metadata=metadata)


__all__ = [
    "MAX_RESOLUTION",
    "ProgressCounter",
    "SAMPLER_NAMES",
    "SCHEDULER_NAMES",
    "bislerp",
    "common_upscale",
    "conditioning_set_values",
    "conditioning_set_values_with_timestep_range",
    "folders",
    "get_tiled_scale_steps",
    "lanczos",
    "load_torch_file",
    "pillow",
    "prepare_noise",
    "save_torch_file",
    "tiled_scale",
    "tiled_scale_multidim",
]
