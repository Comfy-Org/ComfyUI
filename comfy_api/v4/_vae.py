from __future__ import annotations

from copy import copy
from typing import TypedDict

import torch

from comfy_api.internal.singleton import ProxiedSingleton
from comfy_api.latest import io

from .util import _refuse_sizes


class VaeInfo(TypedDict):
    latent_channels: int
    dtype: str | None
    spacial_compression_decode: int | float
    spacial_compression_encode: int | float
    temporal_compression_decode: int | None
    downscale_ratio: int | float | None
    upscale_ratio: int | float | None


def _number(ratio: object) -> int | float | None:
    return ratio if isinstance(ratio, (int, float)) else None


class Vae(ProxiedSingleton):
    async def from_state_dict(
        self,
        sd: dict[str, torch.Tensor],
        *,
        metadata: dict[str, object] | None = None,
    ) -> io.Vae.Type:
        """Build a VAE from weights, with the checkpoint's metadata if its format needs it.

        `metadata` is a safetensors header, as `eng.models.read` returns it. Raises `RuntimeError`
        if ComfyUI does not recognize the weights as a VAE.
        """
        from comfy.sd import VAE

        # Loaders may rename keys. copy() keeps a state dict's _metadata; .copy() drops it.
        vae = VAE(sd=copy(sd), metadata=metadata)
        vae.throw_exception_if_invalid()
        return vae

    async def info(self, vae: io.Vae.Type) -> VaeInfo:
        """Describe a VAE without running it.

        `temporal_compression_decode` is None for a VAE without a time axis. `downscale_ratio`
        and `upscale_ratio` are None for a VAE that scales each axis differently; the `spacial_*`
        fields are always numbers. `dtype` is None for a VAE without weights.
        """
        dtype = getattr(vae, "vae_dtype", None)
        return {
            "latent_channels": vae.latent_channels,
            "dtype": None if dtype is None else str(dtype).removeprefix("torch."),
            "spacial_compression_decode": vae.spacial_compression_decode(),
            "spacial_compression_encode": vae.spacial_compression_encode(),
            "temporal_compression_decode": vae.temporal_compression_decode(),
            "downscale_ratio": _number(vae.downscale_ratio),
            "upscale_ratio": _number(vae.upscale_ratio),
        }

    async def encode(self, vae: io.Vae.Type, pixels: torch.Tensor) -> torch.Tensor:
        """Encode pixels to a latent.

        `pixels` is channels-last with values from 0 to 1, as an IMAGE is; channels beyond the
        VAE's are dropped. Most VAEs center-crop height and width to a multiple of
        `spacial_compression_encode` from `info`. A video VAE usually takes the batch as the frames
        of one video.
        """
        return vae.encode(pixels)

    async def decode(self, vae: io.Vae.Type, samples: torch.Tensor) -> torch.Tensor:
        """Decode a latent to pixels.

        Returns channels-last pixels, as an IMAGE is. A video VAE returns
        `(batch, frames, height, width, channels)`; the VAEDecode node flattens that to one image
        per frame, this call does not.
        """
        return vae.decode(samples)

    async def encode_tiled(
        self,
        vae: io.Vae.Type,
        pixels: torch.Tensor,
        *,
        tile_x: int | None = None,
        tile_y: int | None = None,
        overlap: int | None = None,
        tile_t: int | None = None,
        overlap_t: int | None = None,
    ) -> torch.Tensor:
        """Encode in overlapping tiles to bound memory.

        Sizes are in pixels, and frames for `tile_t` and `overlap_t`; for an audio VAE whose latent
        has one axis besides channels, such as Stable Audio's, `tile_x` and `overlap` are in
        samples. An audio VAE whose latent has two, such as ACE-Step's, cannot encode in tiles:
        ComfyUI raises `IndexError`. None lets the VAE choose. A VAE that tiles itself may ignore
        some sizes. Raises `ValueError` for a tile size that is not positive or a negative
        overlap; keep each overlap below half its tile size.
        """
        _refuse_sizes((tile_x, tile_y, tile_t), (overlap, overlap_t))
        if tile_y is None and vae.latent_dim == 1:
            tile_y = 0  # ComfyUI's 1D path requires tile_y, then drops it.
        return vae.encode_tiled(
            pixels,
            tile_x=tile_x,
            tile_y=tile_y,
            overlap=overlap,
            tile_t=tile_t,
            overlap_t=overlap_t,
        )

    async def decode_tiled(
        self,
        vae: io.Vae.Type,
        samples: torch.Tensor,
        *,
        tile_x: int | None = None,
        tile_y: int | None = None,
        overlap: int | None = None,
        tile_t: int | None = None,
        overlap_t: int | None = None,
    ) -> torch.Tensor:
        """Decode in overlapping tiles to bound memory.

        Sizes are in latent units. None lets the VAE choose, except `overlap` for a video VAE
        ComfyUI tiles: that is 64 pixels, and at most a quarter of a tile. Sizes are refused as
        `encode_tiled` refuses them.
        """
        _refuse_sizes((tile_x, tile_y, tile_t), (overlap, overlap_t))
        if tile_y is None and (samples.ndim == 3 or vae.extra_1d_channel is not None):
            tile_y = 0  # ComfyUI's 1D path requires tile_y, then drops it.
        if overlap is None and samples.ndim == 5 and not vae.handles_tiling:
            # ComfyUI's video path has no default overlap.
            tiles = [tile // 4 for tile in (tile_x, tile_y) if tile is not None]
            overlap = min([64 // vae.spacial_compression_decode(), *tiles])
        return vae.decode_tiled(
            samples,
            tile_x=tile_x,
            tile_y=tile_y,
            overlap=overlap,
            tile_t=tile_t,
            overlap_t=overlap_t,
        )
