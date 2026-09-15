from unittest.mock import MagicMock

import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy.ldm.trellis2.vae import ShapeVae  # noqa: E402


class _FakeMpsTensor(torch.Tensor):
    """A tensor that reports itself as being on MPS without needing the real
    backend, so the CPU-fallback routing in ShapeVae.decode_structure can be
    tested on a machine without Apple Silicon."""

    @staticmethod
    def __new__(cls, data):
        return torch.Tensor._make_subclass(cls, data)

    @property
    def device(self):
        return torch.device("mps")

    def cpu(self):
        return torch.Tensor(self).cpu()

    def to(self, device, *args, **kwargs):
        if isinstance(device, torch.device) and device.type == "mps":
            return self
        return torch.Tensor(self).to(device, *args, **kwargs)


def test_decode_structure_runs_struct_dec_on_cpu_for_mps_input():
    # nn.Conv3d gives incorrect results on MPS (pytorch/pytorch#197114), so
    # decode_structure must run struct_dec on CPU for an MPS input, even
    # though its output still ends up on the original (MPS) device.
    vae = ShapeVae.__new__(ShapeVae)
    vae.struct_dec = MagicMock(
        side_effect=lambda t: _FakeMpsTensor(torch.ones(t.shape[0], 1, 4, 4, 4))
    )

    x = _FakeMpsTensor(torch.zeros(1, 8, 16, 16, 16))
    out = vae.decode_structure(x)

    called_with = vae.struct_dec.call_args[0][0]
    assert called_with.device.type == "cpu"
    assert out.device.type == "mps"


def test_decode_structure_stays_on_device_for_non_mps_input():
    vae = ShapeVae.__new__(ShapeVae)
    vae.struct_dec = MagicMock(side_effect=lambda t: torch.ones(t.shape[0], 1, 4, 4, 4))

    x = torch.zeros(1, 8, 16, 16, 16)
    out = vae.decode_structure(x)

    called_with = vae.struct_dec.call_args[0][0]
    assert called_with.device.type == "cpu"
    assert out.device.type == "cpu"
    vae.struct_dec.to.assert_not_called()
