import asyncio
import types

import pytest
import torch

from comfy_api.latest._prompt_relay_transform import apply_prompt_relay
from comfy_api.latest._sdk import (
    AudioRef,
    InProcessOps,
    InProcessRefResolver,
    VaeRef,
    bind_runtime,
)


class _AudioLayout:
    latent_frequency_bins = 6

    @staticmethod
    def num_of_latents_from_frames(video_frames, frame_rate):
        assert video_frames == 25
        assert frame_rate == 24.0
        return 4


class _AudioVae:
    latent_channels = 8

    def __init__(self):
        self.first_stage_model = _AudioLayout()
        self.seen = None

    def encode(self, waveform):
        self.seen = waveform.clone()
        return torch.ones((waveform.shape[0], 8, 3, 6))


def test_audio_vae_encode_and_empty_layout_are_typed_and_bounded():
    async def run():
        refs = InProcessRefResolver()
        value = _AudioVae()
        vae = VaeRef._wrap(await refs.create("VAE", value))
        audio = AudioRef._wrap(await refs.create("AUDIO", {
            "waveform": torch.arange(20, dtype=torch.float32).reshape(1, 2, 10),
            "sample_rate": 48_000,
        }))
        with bind_runtime(refs, None, InProcessOps()):
            encoded_ref = await vae.encode_audio(audio)
            empty_ref = await vae.empty_audio_latent(25, 24.0)
        return value, await refs.resolve(encoded_ref), await refs.resolve(empty_ref)

    value, encoded, empty = asyncio.run(run())
    assert value.seen.shape == (1, 10, 2)
    assert encoded["type"] == "audio"
    assert encoded["samples"].shape == (1, 8, 3, 6)
    assert empty["type"] == "audio"
    assert empty["samples"].shape == (1, 8, 4, 6)
    assert torch.count_nonzero(empty["samples"]) == 0


def test_audio_vae_rejects_wrong_waveform_and_unpublished_layout():
    async def wrong_waveform():
        refs = InProcessRefResolver()
        vae = VaeRef._wrap(await refs.create("VAE", _AudioVae()))
        audio = AudioRef._wrap(await refs.create("AUDIO", {
            "waveform": torch.zeros((1, 1, 1, 1)), "sample_rate": 44_100,
        }))
        with bind_runtime(refs, None, InProcessOps()):
            await vae.encode_audio(audio)

    with pytest.raises(ValueError, match="2D or 3D"):
        asyncio.run(wrong_waveform())

    async def wrong_layout():
        refs = InProcessRefResolver()
        vae = VaeRef._wrap(await refs.create("VAE", object()))
        with bind_runtime(refs, None, InProcessOps()):
            await vae.empty_audio_latent(25, 24.0)

    with pytest.raises(ValueError, match="audio latent layout"):
        asyncio.run(wrong_layout())


class _Attention:
    def __init__(self):
        self.calls = []

    def forward(
        self, x, context=None, mask=None, pe=None, k_pe=None,
        transformer_options=None,
    ):
        self.calls.append({
            "mask": mask,
            "options": transformer_options,
            "context": context,
        })
        return x


class _LtxPatcher:
    def __init__(self):
        self.attn2 = _Attention()
        self.audio_attn2 = _Attention()
        diffusion = types.SimpleNamespace(
            patchifier=object(),
            vae_scale_factors=(8, 32, 32),
            transformer_blocks=[types.SimpleNamespace(
                attn2=self.attn2, audio_attn2=self.audio_attn2,
            )],
        )
        self.model = types.SimpleNamespace(diffusion_model=diffusion)
        self.model_options = {"transformer_options": {}}
        self.object_patches = {}

    def clone(self):
        clone = _LtxPatcher.__new__(_LtxPatcher)
        clone.attn2 = self.attn2
        clone.audio_attn2 = self.audio_attn2
        clone.model = self.model
        clone.model_options = {"transformer_options": {}}
        clone.object_patches = {}
        return clone

    def get_model_object(self, key):
        attribute = key.split(".")[-2]
        return getattr(self, attribute).forward

    def add_object_patch(self, key, value):
        self.object_patches[key] = value


def test_prompt_relay_builds_expected_temporal_penalty_and_wraps_ltx_attention():
    source = _LtxPatcher()
    latent = {"samples": torch.zeros((1, 128, 4, 1, 1))}
    patched = apply_prompt_relay(
        source, latent,
        token_starts=[1, 3], token_ends=[3, 5],
        pixel_lengths=[17, 8], epsilon=0.001,
    )
    assert patched is not source
    assert set(patched.object_patches) == {
        "diffusion_model.transformer_blocks.0.attn2.forward",
        "diffusion_model.transformer_blocks.0.audio_attn2.forward",
    }

    mask_fn = patched.model_options["transformer_options"][
        "promptrelay_mask_fn"]
    conditional = mask_fn(4, 6, torch.float32, torch.device("cpu"), {
        "cond_or_uncond": [0], "grid_sizes": (4, 1, 1),
    })
    assert conditional.shape == (4, 6)
    assert torch.equal(conditional[:, 0], torch.zeros(4))
    assert torch.equal(conditional[:, 5], torch.zeros(4))
    assert torch.equal(conditional[1, 1:3], torch.zeros(2))
    assert torch.all(conditional[3, 1:3] < 0)
    assert torch.all(conditional[0, 3:5] < 0)
    assert torch.equal(conditional[3, 3:5], torch.zeros(2))
    assert mask_fn(4, 4, torch.float32, torch.device("cpu"), {}) is None
    assert mask_fn(4, 6, torch.float32, torch.device("cpu"), {
        "cond_or_uncond": [1],
    }) is None

    wrapper = patched.object_patches[
        "diffusion_model.transformer_blocks.0.attn2.forward"]
    x = torch.zeros((1, 4, 2))
    context = torch.zeros((1, 6, 2))
    assert wrapper(x, context=context, transformer_options={}) is x
    call = source.attn2.calls[-1]
    assert call["mask"].shape == (4, 6)
    assert callable(call["options"]["optimized_attention_override"])


@pytest.mark.parametrize("starts,ends,lengths", [
    ([], [], []),
    ([2], [2], [8]),
    ([0], [3], [0]),
    ([0, 2], [2], [8]),
])
def test_prompt_relay_rejects_invalid_closed_transform_arguments(
    starts, ends, lengths,
):
    with pytest.raises(ValueError):
        apply_prompt_relay(
            _LtxPatcher(), {"samples": torch.zeros((1, 4, 2, 1, 1))},
            starts, ends, lengths, 0.001,
        )
