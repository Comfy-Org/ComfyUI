from unittest.mock import MagicMock, call

import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy.ldm.trellis2.vae import ShapeVae  # noqa: E402


class _FakeDeviceTensor(torch.Tensor):
    """A tensor that reports a chosen device without needing the real
    backend, and only reports a new device once `.to()` is actually called
    with it, so tests can tell a real device move from a value that already
    happened to report the target device."""

    @staticmethod
    def __new__(cls, data, device):
        t = torch.Tensor._make_subclass(cls, data)
        t._fake_device = torch.device(device)
        return t

    @property
    def device(self):
        return self._fake_device

    def cpu(self):
        return _FakeDeviceTensor(torch.Tensor(self), "cpu")

    def to(self, device, *args, **kwargs):
        if isinstance(device, (torch.device, str)):
            return _FakeDeviceTensor(torch.Tensor(self), device)
        return torch.Tensor(self).to(device, *args, **kwargs)


def test_decode_structure_runs_struct_dec_on_cpu_for_mps_input():
    # nn.Conv3d gives incorrect results on MPS (pytorch/pytorch#197114), so
    # decode_structure must run struct_dec on CPU for an MPS input. The
    # decoder output starts out reporting CPU and only becomes MPS via the
    # `out.to(device)` call, and struct_dec itself must be moved to CPU and
    # back to MPS, in that order.
    vae = ShapeVae.__new__(ShapeVae)
    decoded = _FakeDeviceTensor(torch.ones(1, 1, 4, 4, 4), "cpu")
    vae.struct_dec = MagicMock(return_value=decoded)

    x = _FakeDeviceTensor(torch.zeros(1, 8, 16, 16, 16), "mps")
    out = vae.decode_structure(x)

    called_with = vae.struct_dec.call_args[0][0]
    assert called_with.device.type == "cpu"
    assert out.device.type == "mps"
    assert vae.struct_dec.to.call_args_list == [call("cpu"), call(torch.device("mps"))]


def test_decode_structure_restores_struct_dec_device_on_failure():
    # If struct_dec raises while decoding on CPU, it must still be moved
    # back to its original device rather than left stranded on CPU.
    vae = ShapeVae.__new__(ShapeVae)
    vae.struct_dec = MagicMock(side_effect=RuntimeError("boom"))

    x = _FakeDeviceTensor(torch.zeros(1, 8, 16, 16, 16), "mps")
    with pytest.raises(RuntimeError):
        vae.decode_structure(x)

    assert vae.struct_dec.to.call_args_list == [call("cpu"), call(torch.device("mps"))]


def test_decode_structure_stays_on_device_for_non_mps_input():
    vae = ShapeVae.__new__(ShapeVae)
    vae.struct_dec = MagicMock(side_effect=lambda t: torch.ones(t.shape[0], 1, 4, 4, 4))

    x = torch.zeros(1, 8, 16, 16, 16)
    out = vae.decode_structure(x)

    called_with = vae.struct_dec.call_args[0][0]
    assert called_with.device.type == "cpu"
    assert out.device.type == "cpu"
    vae.struct_dec.to.assert_not_called()
