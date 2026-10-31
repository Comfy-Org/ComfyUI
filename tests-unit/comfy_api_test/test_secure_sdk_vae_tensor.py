import asyncio

import pytest
import torch

from comfy_api.latest._sdk import (
    ImageRef,
    InProcessOps,
    InProcessRefResolver,
    LatentRef,
    TensorRef,
    VaeRef,
    bind_runtime,
)


class _LegacyPackedVae:
    """Small stand-in for an old external channel-packed VAE loader."""

    def __init__(self):
        self.tiled_kwargs = None

    def _assert_current_defaults(self):
        assert self.handles_tiling is False
        assert self.format_encoded is None

    def encode(self, pixels):
        self._assert_current_defaults()
        return torch.zeros(
            (pixels.shape[0], 4, pixels.shape[1], pixels.shape[2]),
            dtype=pixels.dtype,
        )

    def decode(self, samples):
        self._assert_current_defaults()
        batch, _, height, width = samples.shape
        return torch.arange(
            batch * height * width * 12, dtype=torch.float32,
        ).reshape(batch, height, width, 12)

    def decode_tiled(self, samples, **kwargs):
        self._assert_current_defaults()
        self.tiled_kwargs = kwargs
        return self.decode(samples) + 1

    def temporal_compression_decode(self):
        return None

    def spacial_compression_decode(self):
        return 8


def test_vae_tensor_decode_keeps_channel_postprocessing_pack_side():
    async def run():
        refs = InProcessRefResolver()
        ops = InProcessOps()
        value = _LegacyPackedVae()
        vae = VaeRef._wrap(await refs.create("VAE", value))
        latent = LatentRef._wrap(await refs.create(
            "LATENT", {"samples": torch.zeros((1, 4, 2, 3))},
        ))
        image = ImageRef._wrap(await refs.create(
            "IMAGE", torch.zeros((1, 2, 3, 3)),
        ))
        with bind_runtime(refs, None, ops):
            encoded = await vae.encode(image)
            decoded = await vae.decode_tensor(latent)
            tiled = await vae.decode_tensor_tiled(
                latent,
                tile_size=64,
                overlap=16,
                temporal_size=64,
                temporal_overlap=8,
            )
        return (
            value,
            await refs.resolve(encoded),
            decoded,
            await refs.resolve(decoded),
            tiled,
            await refs.resolve(tiled),
        )

    value, encoded, decoded_ref, decoded, tiled_ref, tiled = asyncio.run(run())

    assert isinstance(decoded_ref, TensorRef) and decoded_ref.kind == "TENSOR"
    assert isinstance(tiled_ref, TensorRef) and tiled_ref.kind == "TENSOR"
    assert encoded["samples"].shape == (1, 4, 2, 3)
    assert decoded.shape == (1, 2, 3, 12)
    assert torch.equal(tiled, decoded + 1)
    assert value.handles_tiling is False
    assert value.format_encoded is None
    assert value.tiled_kwargs == {
        "tile_x": 8,
        "tile_y": 8,
        "overlap": 2,
        "tile_t": None,
        "overlap_t": None,
    }


def test_vae_tensor_decode_uses_canonical_tile_bounds():
    async def run():
        refs = InProcessRefResolver()
        vae = VaeRef._wrap(await refs.create("VAE", _LegacyPackedVae()))
        latent = LatentRef._wrap(await refs.create(
            "LATENT", {"samples": torch.zeros((1, 4, 2, 3))},
        ))
        with bind_runtime(refs, None, InProcessOps()):
            await vae.decode_tensor_tiled(latent, tile_size=32)

    with pytest.raises(ValueError, match="tile_size"):
        asyncio.run(run())
