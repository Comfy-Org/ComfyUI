from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ldm.seedvr.vae as vae_mod  # noqa: E402
from comfy_extras import nodes_seedvr  # noqa: E402


_LATENT_CHANNELS = vae_mod.SEEDVR2_LATENT_CHANNELS


def _make_wrapper() -> vae_mod.VideoAutoencoderKLWrapper:
    wrapper = vae_mod.VideoAutoencoderKLWrapper.__new__(
        vae_mod.VideoAutoencoderKLWrapper
    )
    nn.Module.__init__(wrapper)
    return wrapper


def _fingerprint_decode_(self, z, return_dict=True):
    b = int(z.shape[0])
    t = int(z.shape[2])
    h = int(z.shape[3])
    w = int(z.shape[4])
    out = torch.empty(b, 3, t, h * 8, w * 8)
    for batch_idx in range(b):
        out[batch_idx].fill_(float(batch_idx + 1))
    return out


def _decode_with_patches(wrapper, z):
    with patch.object(vae_mod.VideoAutoencoderKL, "decode_", _fingerprint_decode_):
        return wrapper.decode(z)


def test_decode_b2_t3_multi_frame_batch_unchanged():
    wrapper = _make_wrapper()

    out = _decode_with_patches(wrapper, torch.zeros(2, _LATENT_CHANNELS * 3, 2, 2))

    assert tuple(out.shape) == (2, 3, 3, 16, 16)


class _Wrapper(vae_mod.VideoAutoencoderKLWrapper):
    def __init__(self):
        nn.Module.__init__(self)
        self.calls = []

    def parameters(self):
        return iter([torch.nn.Parameter(torch.zeros(()))])

def _decode_stub(self, latent):
    self.calls.append(tuple(latent.shape))
    return torch.zeros(latent.shape[0], 3, latent.shape[2], latent.shape[3] * 8, latent.shape[4] * 8)


def test_seedvr2_wrapper_decode_accepts_5d_channel_first_latents_without_preprocessor_state():
    wrapper = _Wrapper()

    with patch.object(vae_mod.VideoAutoencoderKL, "decode_", _decode_stub):
        out = wrapper.decode(torch.zeros(1, _LATENT_CHANNELS, 2, 4, 5))

    assert tuple(out.shape) == (1, 3, 2, 32, 40)
    assert wrapper.calls == [(1, _LATENT_CHANNELS, 2, 4, 5)]


def test_seedvr2_wrapper_decode_rejects_wrong_rank_latents():
    wrapper = _Wrapper()

    with pytest.raises(RuntimeError, match=r"latent input must be 4-D collapsed .* or 5-D"):
        wrapper.decode(torch.zeros(1, _LATENT_CHANNELS, 4))


def _t_padded(t_in: int) -> int:
    if t_in == 1:
        return 1
    if t_in <= 4:
        return 5
    if (t_in - 1) % 4 == 0:
        return t_in
    return t_in + (4 - ((t_in - 1) % 4))


@pytest.mark.parametrize("t_in", [1, 5, 9])
def test_t_padded_matches_cut_videos(t_in):
    dummy = torch.zeros(1, t_in, 1, 1, 1)
    assert nodes_seedvr.cut_videos(dummy).shape[1] == _t_padded(t_in)
