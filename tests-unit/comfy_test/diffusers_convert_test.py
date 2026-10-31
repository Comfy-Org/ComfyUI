import collections

import torch

from comfy.diffusers_convert import cat_tensors

FakeDevice = collections.namedtuple("FakeDevice", ["type", "index"])("comfy-lazy-caster", 0)


class LazyCastingLikeTensor(torch.Tensor):
    """Mimics comfy.model_patcher.LazyCastingParam: its .device property lies
    (returns a non-torch.device object) but .to() materializes the real tensor."""

    @staticmethod
    def __new__(cls, tensor):
        return super().__new__(cls, tensor)

    @property
    def device(self):
        return FakeDevice

    def to(self, *args, **kwargs):
        return torch.Tensor(self).to("cpu")


def test_cat_tensors_with_lazy_casting_param():
    real = torch.randn(3, 4)
    fake = LazyCastingLikeTensor(real)

    out = cat_tensors([fake, fake])
    regular_out = cat_tensors([real, real])

    assert out.device.type == "cpu"
    assert torch.equal(out, torch.cat([real, real], dim=0))
    assert torch.equal(regular_out, torch.cat([real, real], dim=0))
