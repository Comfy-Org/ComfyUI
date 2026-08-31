import torch
from torch import nn

from comfy.ldm.lama import FourierUnit


def test_fourier_unit_preserves_non_square_spatial_shape():
    unit = FourierUnit(4, nn).eval()
    source = torch.randn((2, 4, 6, 10), dtype=torch.float32)

    result = unit(source)

    assert result.shape == source.shape
    assert result.dtype == source.dtype
    assert torch.isfinite(result).all()
