import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from comfy.cli_args import args
if not torch.cuda.is_available():
    args.cpu = True

from comfy import ops
import comfy.memory_management
from comfy.model_base import BaseModel


class SimpleModel(torch.nn.Module):
    def __init__(self, operations=ops.disable_weight_init):
        super().__init__()
        self.layer1 = operations.Linear(10, 20, device="cpu", dtype=torch.bfloat16)

    def forward(self, x):
        return self.layer1(x)


class FakeModelConfig:
    def process_unet_state_dict(self, sd):
        return sd


class TestUnrecognizedQuantMetadata(unittest.TestCase):
    def test_stray_quant_scale_tensors_raise(self):
        """A checkpoint whose per-layer quantization scales are present but not
        recognized (no _quantization_metadata / comfy_quant marker) must fail
        loudly instead of silently loading the packed weight as unquantized.

        The lazy-load path (used with dynamic VRAM loading, aimdo_enabled) assigns
        the incoming tensor directly without a shape check, which is how the packed
        NVFP4 weight ends up wired into a mismatched matmul at forward time."""
        old_aimdo_enabled = comfy.memory_management.aimdo_enabled
        comfy.memory_management.aimdo_enabled = True
        try:
            model = BaseModel.__new__(BaseModel)
            torch.nn.Module.__init__(model)
            model.model_config = FakeModelConfig()
            model.diffusion_model = SimpleModel()

            state_dict = {
                "layer1.weight": torch.zeros(20, 5, dtype=torch.uint8),
                "layer1.weight_scale": torch.zeros(20, dtype=torch.float8_e4m3fn),
                "layer1.weight_scale_2": torch.tensor(1.0),
                "layer1.input_scale": torch.tensor(1.0),
            }

            with self.assertRaisesRegex(
                RuntimeError,
                r"unrecognized quantization scale tensors.*layer1\.weight_scale",
            ):
                model.load_model_weights(state_dict, assign=True)
        finally:
            comfy.memory_management.aimdo_enabled = old_aimdo_enabled


if __name__ == "__main__":
    unittest.main()
