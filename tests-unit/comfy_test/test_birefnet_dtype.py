from functools import partial

import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ops
from comfy.background_removal.birefnet import BasicLayer


def _basic_layer(dtype, window_size=4):
    # Mirrors how SwinTransformer builds its stages: a depth of 2 gives one
    # regular block and one shifted block, and the shifted one is the only
    # consumer of the attention mask built in BasicLayer.forward.
    return BasicLayer(
        dim=32,
        depth=2,
        num_heads=4,
        window_size=window_size,
        norm_layer=partial(comfy.ops.manual_cast.LayerNorm, device="cpu", dtype=dtype),
        device="cpu",
        dtype=dtype,
        operations=comfy.ops.manual_cast,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_birefnet_shifted_window_attention_keeps_compute_dtype(dtype):
    # A resolution that is not a multiple of the window size forces the padded,
    # shifted-window path where the attention mask is applied.
    layer = _basic_layer(dtype)
    height = width = 6
    x = torch.randn(1, height * width, 32, dtype=dtype)

    out = layer(x, height, width)[0]

    assert out.dtype == dtype
