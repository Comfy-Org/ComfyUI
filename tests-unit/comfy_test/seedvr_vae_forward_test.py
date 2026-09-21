"""Regression tests for the SeedVR2 VAE forward return contract."""

import pytest
import torch
import torch.nn as nn

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy.ldm.seedvr.vae import SEEDVR2_LATENT_CHANNELS, VideoAutoencoderKL  # noqa: E402


_LATENT_SHAPE = (1, SEEDVR2_LATENT_CHANNELS, 2, 2, 2)
_DECODED_SHAPE = (1, 3, 5, 16, 16)
_INPUT_ENCODE_SHAPE = (1, 3, 5, 16, 16)
_INPUT_DECODE_SHAPE = _LATENT_SHAPE


class _StubVAE(VideoAutoencoderKL):
    def __init__(self):
        nn.Module.__init__(self)
        self._encode_out = torch.zeros(*_LATENT_SHAPE)
        self._decode_out = torch.zeros(*_DECODED_SHAPE)

    def encode(self, x, return_dict=True):
        return self._encode_out

    def decode_(self, z, return_dict=True):
        return self._decode_out


def test_forward_encode_returns_tensor():
    vae = _StubVAE()
    x = torch.zeros(*_INPUT_ENCODE_SHAPE)
    result = vae.forward(x, mode="encode")
    assert type(result) is torch.Tensor
    assert result.shape == torch.Size(_LATENT_SHAPE)


def test_forward_decode_returns_tensor():
    vae = _StubVAE()
    z = torch.zeros(*_INPUT_DECODE_SHAPE)
    result = vae.forward(z, mode="decode")
    assert type(result) is torch.Tensor
    assert result.shape == torch.Size(_DECODED_SHAPE)


class _TupleReturningStubVAE(VideoAutoencoderKL):
    def __init__(self):
        nn.Module.__init__(self)
        self._encode_tensor = torch.zeros(*_LATENT_SHAPE)
        self._decode_tensor = torch.zeros(*_DECODED_SHAPE)

    def encode(self, x, return_dict=True):
        return (self._encode_tensor,)

    def decode_(self, z, return_dict=True):
        return (self._decode_tensor,)


def test_forward_all_unwraps_one_tuple_at_each_step():
    vae = _TupleReturningStubVAE()
    x = torch.zeros(*_INPUT_ENCODE_SHAPE)
    result = vae.forward(x, mode="all")
    assert type(result) is torch.Tensor
    assert result.shape == torch.Size(_DECODED_SHAPE)


def test_forward_rejects_unknown_mode():
    vae = _StubVAE()
    with pytest.raises(ValueError, match="Unknown SeedVR2 VAE forward mode"):
        vae.forward(torch.zeros(*_INPUT_ENCODE_SHAPE), mode="bogus")
