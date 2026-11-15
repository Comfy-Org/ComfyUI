import unittest
from unittest.mock import MagicMock, patch
import json
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from comfy.cli_args import args
original_cpu = args.cpu
try:
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        args.cpu = True
    from comfy import ops
    import comfy.memory_management
    import comfy.utils
    import comfy.sd
    from comfy.model_base import BaseModel
finally:
    args.cpu = original_cpu


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

    def test_stray_quant_scale_tensors_raise_in_convert_old_quants(self):
        """Text encoders load through CLIP.load_sd(), which never reaches
        BaseModel.load_model_weights, but does go through convert_old_quants()
        (comfy/sd.py load_clip). A checkpoint with per-layer scales but no
        scaled_fp8/comfy_quant/_quantization_metadata marker must fail there too."""
        state_dict = {
            "encoder.block.0.weight": torch.zeros(20, 20, dtype=torch.float8_e4m3fn),
            "encoder.block.0.weight_scale": torch.tensor(700.0),
        }

        with self.assertRaisesRegex(
            RuntimeError,
            r"unrecognized quantization scale tensors.*encoder\.block\.0\.weight_scale",
        ):
            comfy.utils.convert_old_quants(state_dict, model_prefix="")

    def test_existing_quant_marker_is_accepted(self):
        marker = torch.tensor(list(b'{"format":"float8_e4m3fn"}'), dtype=torch.uint8)
        state_dict = {"encoder.weight_scale": torch.tensor(1.0), "encoder.comfy_quant": marker}
        converted, _ = comfy.utils.convert_old_quants(state_dict)
        self.assertIs(converted["encoder.comfy_quant"], marker)
        state_dict["other.weight_scale"] = torch.tensor(1.0)
        with self.assertRaisesRegex(RuntimeError, "other.weight_scale"):
            comfy.utils.convert_old_quants(state_dict)

    def test_combined_checkpoint_clip_quantization(self):
        for kind in ("unquantized", "native", "metadata", "legacy", "unrecognized", "custom"):
            with self.subTest(kind=kind):
                layer = "text_encoders.encoder.block.0"
                state_dict = {layer + ".weight": torch.ones(2, 2)}
                metadata = {}
                model_options = {}
                if kind != "unquantized":
                    state_dict[layer + ".weight_scale"] = torch.tensor(1.0)
                if kind == "native":
                    state_dict[layer + ".comfy_quant"] = torch.tensor(
                        list(b'{"format":"float8_e4m3fn"}'), dtype=torch.uint8,
                    )
                elif kind == "metadata":
                    metadata["_quantization_metadata"] = json.dumps({
                        "layers": {layer: {"format": "float8_e4m3fn"}},
                    })
                elif kind == "legacy":
                    state_dict["text_encoders.scaled_fp8"] = torch.tensor(1.0)
                    state_dict[layer + ".scale_weight"] = state_dict.pop(layer + ".weight_scale")
                elif kind == "custom":
                    model_options["custom_operations"] = object()

                config = MagicMock()
                config.supported_inference_dtypes = [torch.float32]
                config.quant_config = None
                config.clip_vision_prefix = None
                config.process_clip_state_dict.side_effect = lambda sd: comfy.utils.state_dict_prefix_replace(
                    sd, {"text_encoders.": ""}, filter_keys=True,
                )
                with patch.object(comfy.sd.model_detection, "unet_prefix_from_state_dict", return_value="model.diffusion_model."), \
                     patch.object(comfy.sd.model_detection, "model_config_from_unet", return_value=config), \
                     patch.object(comfy.sd.model_management, "get_torch_device", return_value=torch.device("cpu")), \
                     patch.object(comfy.sd.model_management, "unet_dtype", return_value=torch.float32), \
                     patch.object(comfy.sd.model_management, "unet_manual_cast", return_value=None), \
                     patch.object(comfy.sd, "CLIP") as clip:
                    if kind == "unrecognized":
                        with self.assertRaisesRegex(RuntimeError, r"unrecognized quantization scale tensors.*encoder.block.0.weight_scale"):
                            comfy.sd.load_state_dict_guess_config(
                                state_dict, output_model=False, output_vae=False, metadata=metadata,
                            )
                        clip.assert_not_called()
                    else:
                        result = comfy.sd.load_state_dict_guess_config(
                            state_dict, output_model=False, output_vae=False,
                            metadata=metadata, te_model_options=model_options,
                        )
                        self.assertIs(result[1], clip.return_value)
                        loaded = clip.call_args.kwargs["state_dict"]
                        self.assertIn("encoder.block.0.weight", loaded)
                        if kind in ("native", "metadata", "legacy"):
                            self.assertIn("encoder.block.0.comfy_quant", loaded)


if __name__ == "__main__":
    unittest.main()
