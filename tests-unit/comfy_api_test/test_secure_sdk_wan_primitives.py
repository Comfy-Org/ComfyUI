import asyncio

import pytest
import torch

import comfy.clip_vision
from comfy_api.latest._sdk import (
    ClipVisionOutputRef,
    CondRef,
    ImageRef,
    InProcessOps,
    InProcessRefResolver,
    VaeRef,
    bind_runtime,
)


class _LayoutVae:
    latent_channels = 16

    @staticmethod
    def spacial_compression_encode():
        return 8

    @staticmethod
    def temporal_compression_encode():
        return 4


def test_wan_layout_vision_conditioning_and_batch_selection_stay_opaque():
    async def run():
        refs = InProcessRefResolver()
        vae = VaeRef._wrap(await refs.create("VAE", _LayoutVae()))
        images_value = torch.arange(5 * 2 * 3 * 3).reshape(5, 2, 3, 3)
        images = ImageRef._wrap(await refs.create("IMAGE", images_value))

        first = comfy.clip_vision.Output()
        first.penultimate_hidden_states = torch.arange(
            8, dtype=torch.float32).reshape(1, 2, 4)
        second = comfy.clip_vision.Output()
        second.penultimate_hidden_states = torch.arange(
            12, dtype=torch.float32).reshape(1, 3, 4) + 100
        first_ref = ClipVisionOutputRef._wrap(
            await refs.create("CLIP_VISION_OUTPUT", first))
        second_ref = ClipVisionOutputRef._wrap(
            await refs.create("CLIP_VISION_OUTPUT", second))
        conditioning_value = [[
            torch.zeros((1, 1, 4)), {"stable": "metadata"},
        ]]
        conditioning = CondRef._wrap(await refs.create(
            "CONDITIONING", conditioning_value))

        with bind_runtime(refs, None, InProcessOps()):
            layout = await vae.latent_layout()
            selected = await images.select_batch([4, 1])
            combined = await first_ref.concat(second_ref)
            attached = await conditioning.with_clip_vision_output(combined)
            with pytest.raises(ValueError, match="unique integers"):
                await images.select_batch([1, 1])
            with pytest.raises(IndexError, match="out of range"):
                await images.select_batch([5])

        return (
            refs,
            layout,
            await refs.resolve(selected),
            await refs.resolve(combined),
            await refs.resolve(attached),
            conditioning_value,
        )

    refs, layout, selected, combined, attached, source = asyncio.run(run())
    assert refs is not None
    assert layout == {
        "channels": 16,
        "spatial_compression": 8,
        "temporal_compression": 4,
    }
    assert torch.equal(selected, torch.stack((
        torch.arange(5 * 2 * 3 * 3).reshape(5, 2, 3, 3)[4],
        torch.arange(5 * 2 * 3 * 3).reshape(5, 2, 3, 3)[1],
    )))
    assert combined.penultimate_hidden_states.shape == (1, 5, 4)
    assert attached[0][1]["stable"] == "metadata"
    assert attached[0][1]["clip_vision_output"] is combined
    assert "clip_vision_output" not in source[0][1]


def test_wan_layout_and_vision_concat_fail_closed():
    class _BadLayout:
        latent_channels = 0

        @staticmethod
        def spacial_compression_encode():
            return 8

    async def run():
        refs = InProcessRefResolver()
        vae = VaeRef._wrap(await refs.create("VAE", _BadLayout()))
        first = comfy.clip_vision.Output()
        first.penultimate_hidden_states = torch.zeros((1, 2, 4))
        second = comfy.clip_vision.Output()
        second.penultimate_hidden_states = torch.zeros((2, 2, 4))
        first_ref = ClipVisionOutputRef._wrap(
            await refs.create("CLIP_VISION_OUTPUT", first))
        second_ref = ClipVisionOutputRef._wrap(
            await refs.create("CLIP_VISION_OUTPUT", second))
        with bind_runtime(refs, None, InProcessOps()):
            with pytest.raises(ValueError, match="latent layout"):
                await vae.latent_layout()
            with pytest.raises(ValueError, match="compatible hidden states"):
                await first_ref.concat(second_ref)

    asyncio.run(run())
