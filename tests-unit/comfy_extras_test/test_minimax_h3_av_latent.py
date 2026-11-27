import asyncio

import pytest
import torch

import comfy.nested_tensor
from comfy_extras.nodes_minimax_h3 import MiniMaxH3AVLatent, MiniMaxH3Extension


def make_latents(video_shape=(1, 24, 107, 2, 3), audio_shape=(1, 32, 2, 603)):
    return {"samples": torch.zeros(video_shape)}, {"samples": torch.zeros(audio_shape)}


def test_builds_minimax_h3_nested_av_latent():
    video_latent, audio_latent = make_latents()

    output = MiniMaxH3AVLatent.execute(video_latent, audio_latent)[0]

    assert isinstance(output["samples"], comfy.nested_tensor.NestedTensor)
    video, audio = output["samples"].unbind()
    assert video is video_latent["samples"]
    assert audio is audio_latent["samples"]


@pytest.mark.parametrize(
    ("video_shape", "audio_shape", "message"),
    [
        ((1, 24, 107, 2), (1, 32, 2, 603), "video latent must have shape"),
        ((1, 16, 107, 2, 3), (1, 32, 2, 603), "video latent must have shape"),
        ((1, 24, 107, 2, 3), (1, 32, 603), "audio latent must have shape"),
        ((1, 24, 107, 2, 3), (1, 16, 2, 603), "audio latent must have shape"),
        ((1, 24, 107, 2, 3), (1, 32, 1, 603), "audio latent must have shape"),
    ],
)
def test_rejects_invalid_stream_shapes(video_shape, audio_shape, message):
    video_latent, audio_latent = make_latents(video_shape, audio_shape)

    with pytest.raises(ValueError, match=message):
        MiniMaxH3AVLatent.execute(video_latent, audio_latent)


def test_rejects_mismatched_batch_sizes():
    video_latent, audio_latent = make_latents(audio_shape=(2, 32, 2, 603))

    with pytest.raises(ValueError, match="batch sizes must match"):
        MiniMaxH3AVLatent.execute(video_latent, audio_latent)


def test_accepts_one_tick_rounding_difference_and_rejects_obvious_timeline_mismatch():
    video_latent, audio_latent = make_latents(audio_shape=(1, 32, 2, 604))
    assert MiniMaxH3AVLatent.execute(video_latent, audio_latent)[0]["samples"].unbind()[1] is audio_latent["samples"]

    video_latent, audio_latent = make_latents(audio_shape=(1, 32, 2, 600))
    with pytest.raises(ValueError, match="timelines do not match"):
        MiniMaxH3AVLatent.execute(video_latent, audio_latent)


def test_node_is_registered_next_to_empty_av_latent():
    node_list = asyncio.run(MiniMaxH3Extension().get_node_list())

    index = node_list.index(MiniMaxH3AVLatent)
    assert node_list[index - 1].__name__ == "EmptyMiniMaxH3LatentAV"
