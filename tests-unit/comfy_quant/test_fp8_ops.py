import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy import ops


class TestFP8Ops(unittest.TestCase):
    def test_vbar_requantizes_weight_but_not_bias(self):
        weight = torch.zeros((16, 16), dtype=torch.float8_e4m3fn)
        bias = torch.zeros(16, dtype=torch.float8_e4m3fn)
        patch = lambda tensor: tensor
        layer = SimpleNamespace(
            weight=weight,
            bias=bias,
            weight_function=[],
            bias_function=[],
            weight_lowvram_function=patch,
            bias_lowvram_function=patch,
            seed_key="layer",
            _prefetch={
                "resident": False,
                "xfer_dest": torch.empty(1, dtype=torch.uint8),
                "needs_cast": False,
                "cast_geometry": None,
                "signature": (1,),
            },
        )

        with (
            mock.patch.object(ops.comfy.memory_management, "interpret_gathered_like", return_value=[weight, bias]),
            mock.patch.object(ops.comfy.float, "stochastic_rounding", side_effect=lambda tensor, dtype, seed: tensor.to(dtype)) as rounding,
        ):
            cast_weight, cast_bias = ops.resolve_cast_module_with_vbar(
                layer,
                torch.float8_e4m3fn,
                torch.device("cpu"),
                torch.float16,
                torch.float16,
                True,
            )

        self.assertEqual(cast_weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(cast_bias.dtype, torch.float16)
        self.assertEqual(rounding.call_count, 2)

    def test_fp8_linear_unpins_vbar_when_linear_fails(self):
        weight = torch.zeros((16, 16), dtype=torch.float8_e4m3fn)
        bias = torch.zeros(16, dtype=torch.float8_e4m3fn)
        vbar = object()
        layer = SimpleNamespace(weight=weight, _v=vbar)
        input_tensor = torch.zeros((2, 16), dtype=torch.float16)
        offload_info = (None, torch.device("cpu"), None)

        with (
            mock.patch.object(ops, "cast_bias_weight", return_value=(weight, bias, offload_info)),
            mock.patch.object(ops.comfy_aimdo.model_vbar, "vbar_unpin") as unpin,
            mock.patch.object(torch.nn.functional, "linear", side_effect=RuntimeError("FP8 linear failed")),
        ):
            with self.assertRaisesRegex(RuntimeError, "FP8 linear failed"):
                ops.fp8_linear(layer, input_tensor)

        unpin.assert_called_once_with(vbar)
