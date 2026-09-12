from unittest.mock import patch

import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy.ldm.modules.attention import (  # noqa: E402
    attention_basic,
    attention_sub_quad,
    optimized_attention_for_device,
)


def test_small_input_without_pytorch_attention_uses_chunked_fallback():
    with patch("comfy.ldm.modules.attention.model_management.pytorch_attention_enabled", return_value=False):
        func = optimized_attention_for_device(torch.device("cpu"), small_input=True)

    assert func is attention_sub_quad
    assert func is not attention_basic
