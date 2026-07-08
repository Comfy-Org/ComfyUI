import unittest
import torch
import sys
import os
import json

# Add comfy to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

def has_gpu():
    return torch.cuda.is_available()

from comfy.cli_args import args
if not has_gpu():
    args.cpu = True

from comfy import ops
from comfy.quant_ops import QuantizedTensor
import comfy.utils


class SimpleModel(torch.nn.Module):
    def __init__(self, operations=ops.disable_weight_init):
        super().__init__()
        self.layer1 = operations.Linear(10, 20, device="cpu", dtype=torch.bfloat16)
        self.layer2 = operations.Linear(20, 30, device="cpu", dtype=torch.bfloat16)
        self.layer3 = operations.Linear(30, 40, device="cpu", dtype=torch.bfloat16)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.functional.relu(x)
        x = self.layer2(x)
        x = torch.nn.functional.relu(x)
        x = self.layer3(x)
        return x


class TestMixedPrecisionOps(unittest.TestCase):

    def test_all_layers_standard(self):
        """Test that model with no quantization works normally"""
        # Create model
        model = SimpleModel(operations=ops.mixed_precision_ops({}))

        # Initialize weights manually
        model.layer1.weight = torch.nn.Parameter(torch.randn(20, 10, dtype=torch.bfloat16))
        model.layer1.bias = torch.nn.Parameter(torch.randn(20, dtype=torch.bfloat16))
        model.layer2.weight = torch.nn.Parameter(torch.randn(30, 20, dtype=torch.bfloat16))
        model.layer2.bias = torch.nn.Parameter(torch.randn(30, dtype=torch.bfloat16))
        model.layer3.weight = torch.nn.Parameter(torch.randn(40, 30, dtype=torch.bfloat16))
        model.layer3.bias = torch.nn.Parameter(torch.randn(40, dtype=torch.bfloat16))

        # Initialize weight_function and bias_function
        for layer in [model.layer1, model.layer2, model.layer3]:
            layer.weight_function = []
            layer.bias_function = []

        # Forward pass
        input_tensor = torch.randn(5, 10, dtype=torch.bfloat16)
        output = model(input_tensor)

        self.assertEqual(output.shape, (5, 40))
        self.assertEqual(output.dtype, torch.bfloat16)

    def test_mixed_precision_load(self):
        """Test loading a mixed precision model from state dict"""
        # Configure mixed precision: layer1 is FP8, layer2 and layer3 are standard
        layer_quant_config = {
            "layer1": {
                "format": "float8_e4m3fn",
                "params": {}
            },
            "layer3": {
                "format": "float8_e4m3fn",
                "params": {}
            }
        }

        # Create state dict with mixed precision
        fp8_weight1 = torch.randn(20, 10, dtype=torch.float32).to(torch.float8_e4m3fn)
        fp8_weight3 = torch.randn(40, 30, dtype=torch.float32).to(torch.float8_e4m3fn)

        state_dict = {
            # Layer 1: FP8 E4M3FN
            "layer1.weight": fp8_weight1,
            "layer1.bias": torch.randn(20, dtype=torch.bfloat16),
            "layer1.weight_scale": torch.tensor(2.0, dtype=torch.float32),

            # Layer 2: Standard BF16
            "layer2.weight": torch.randn(30, 20, dtype=torch.bfloat16),
            "layer2.bias": torch.randn(30, dtype=torch.bfloat16),

            # Layer 3: FP8 E4M3FN
            "layer3.weight": fp8_weight3,
            "layer3.bias": torch.randn(40, dtype=torch.bfloat16),
            "layer3.weight_scale": torch.tensor(1.5, dtype=torch.float32),
        }

        state_dict, _ = comfy.utils.convert_old_quants(state_dict, metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})})
        # Create model and load state dict (strict=False because custom loading pops keys)
        model = SimpleModel(operations=ops.mixed_precision_ops({}))
        model.load_state_dict(state_dict, strict=False)

        # Verify weights are wrapped in QuantizedTensor
        self.assertIsInstance(model.layer1.weight, QuantizedTensor)
        self.assertEqual(model.layer1.weight._layout_cls, "TensorCoreFP8E4M3Layout")

        # Layer 2 should NOT be quantized
        self.assertNotIsInstance(model.layer2.weight, QuantizedTensor)

        # Layer 3 should be quantized
        self.assertIsInstance(model.layer3.weight, QuantizedTensor)
        self.assertEqual(model.layer3.weight._layout_cls, "TensorCoreFP8E4M3Layout")

        # Verify scales were loaded
        self.assertEqual(model.layer1.weight._params.scale.item(), 2.0)
        self.assertEqual(model.layer3.weight._params.scale.item(), 1.5)

        # Forward pass
        input_tensor = torch.randn(5, 10, dtype=torch.bfloat16)
        with torch.inference_mode():
            output = model(input_tensor)

        self.assertEqual(output.shape, (5, 40))

    def test_state_dict_quantized_preserved(self):
        """Test that quantized weights are preserved in state_dict()"""
        # Configure mixed precision
        layer_quant_config = {
            "layer1": {
                "format": "float8_e4m3fn",
                "params": {}
            }
        }

        # Create and load model
        fp8_weight = torch.randn(20, 10, dtype=torch.float32).to(torch.float8_e4m3fn)
        state_dict1 = {
            "layer1.weight": fp8_weight,
            "layer1.bias": torch.randn(20, dtype=torch.bfloat16),
            "layer1.weight_scale": torch.tensor(3.0, dtype=torch.float32),
            "layer2.weight": torch.randn(30, 20, dtype=torch.bfloat16),
            "layer2.bias": torch.randn(30, dtype=torch.bfloat16),
            "layer3.weight": torch.randn(40, 30, dtype=torch.bfloat16),
            "layer3.bias": torch.randn(40, dtype=torch.bfloat16),
        }

        state_dict1, _ = comfy.utils.convert_old_quants(state_dict1, metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})})
        model = SimpleModel(operations=ops.mixed_precision_ops({}))
        model.load_state_dict(state_dict1, strict=False)

        # Save state dict
        state_dict2 = model.state_dict()

        # Verify layer1.weight is a QuantizedTensor with scale preserved
        self.assertTrue(torch.equal(state_dict2["layer1.weight"].view(torch.uint8), fp8_weight.view(torch.uint8)))
        self.assertEqual(state_dict2["layer1.weight_scale"].item(), 3.0)
        self.assertEqual(model.layer1.weight._layout_cls, "TensorCoreFP8E4M3Layout")

        # Verify non-quantized layers are standard tensors
        self.assertNotIsInstance(state_dict2["layer2.weight"], QuantizedTensor)
        self.assertNotIsInstance(state_dict2["layer3.weight"], QuantizedTensor)

    def test_weight_function_compatibility(self):
        """Test that weight_function (LoRA) works with quantized layers"""
        # Configure FP8 quantization
        layer_quant_config = {
            "layer1": {
                "format": "float8_e4m3fn",
                "params": {}
            }
        }

        # Create and load model
        fp8_weight = torch.randn(20, 10, dtype=torch.float32).to(torch.float8_e4m3fn)
        state_dict = {
            "layer1.weight": fp8_weight,
            "layer1.bias": torch.randn(20, dtype=torch.bfloat16),
            "layer1.weight_scale": torch.tensor(2.0, dtype=torch.float32),
            "layer2.weight": torch.randn(30, 20, dtype=torch.bfloat16),
            "layer2.bias": torch.randn(30, dtype=torch.bfloat16),
            "layer3.weight": torch.randn(40, 30, dtype=torch.bfloat16),
            "layer3.bias": torch.randn(40, dtype=torch.bfloat16),
        }

        state_dict, _ = comfy.utils.convert_old_quants(state_dict, metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})})
        model = SimpleModel(operations=ops.mixed_precision_ops({}))
        model.load_state_dict(state_dict, strict=False)

        # Add a weight function (simulating LoRA)
        # This should trigger dequantization during forward pass
        def apply_lora(weight):
            lora_delta = torch.randn_like(weight) * 0.01
            return weight + lora_delta

        model.layer1.weight_function.append(apply_lora)

        # Forward pass should work with LoRA (triggers weight_function path)
        input_tensor = torch.randn(5, 10, dtype=torch.bfloat16)
        output = model(input_tensor)

        self.assertEqual(output.shape, (5, 40))

    def test_error_handling_unknown_format(self):
        """Test that unknown formats raise error"""
        # Configure with unknown format
        layer_quant_config = {
            "layer1": {
                "format": "unknown_format_xyz",
                "params": {}
            }
        }

        # Create state dict
        state_dict = {
            "layer1.weight": torch.randn(20, 10, dtype=torch.bfloat16),
            "layer1.bias": torch.randn(20, dtype=torch.bfloat16),
            "layer2.weight": torch.randn(30, 20, dtype=torch.bfloat16),
            "layer2.bias": torch.randn(30, dtype=torch.bfloat16),
            "layer3.weight": torch.randn(40, 30, dtype=torch.bfloat16),
            "layer3.bias": torch.randn(40, dtype=torch.bfloat16),
        }

        state_dict, _ = comfy.utils.convert_old_quants(state_dict, metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})})

        # Load should raise KeyError for unknown format in QUANT_FORMAT_MIXINS
        model = SimpleModel(operations=ops.mixed_precision_ops({}))
        with self.assertRaises(KeyError):
            model.load_state_dict(state_dict, strict=False)

    def test_int8_convrot_metadata_loads_into_params(self):
        """ConvRot metadata must reach TensorWiseINT8Layout params."""
        torch.manual_seed(123)
        layer_quant_config = {
            "layer": {
                "format": "int8_tensorwise",
                "convrot": True,
                "convrot_groupsize": 256,
            }
        }
        weight = torch.randn(16, 256, dtype=torch.bfloat16)
        bias = torch.randn(16, dtype=torch.bfloat16)
        q_weight = QuantizedTensor.from_float(
            weight,
            "TensorWiseINT8Layout",
            per_channel=True,
            convrot=True,
            convrot_groupsize=256,
        )
        state_dict = {
            "layer.weight": q_weight._qdata,
            "layer.bias": bias,
            "layer.weight_scale": q_weight._params.scale,
        }

        state_dict, _ = comfy.utils.convert_old_quants(
            state_dict,
            metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})},
        )
        model = torch.nn.Module()
        model.layer = ops.mixed_precision_ops({}).Linear(256, 16, device="cpu", dtype=torch.bfloat16)
        model.load_state_dict(state_dict, strict=False)

        self.assertIsInstance(model.layer.weight, QuantizedTensor)
        self.assertEqual(model.layer.weight._layout_cls, "TensorWiseINT8Layout")
        self.assertTrue(model.layer.weight._params.convrot)
        self.assertEqual(model.layer.weight._params.convrot_groupsize, 256)

        input_tensor = torch.randn(4, 256, dtype=torch.bfloat16)
        loaded_out = model.layer(input_tensor)
        ref_out = torch.nn.functional.linear(input_tensor, q_weight, bias)
        self.assertTrue(torch.equal(loaded_out, ref_out))

        fp16_input = input_tensor.to(torch.float16)
        loaded_fp16_out = model.layer(fp16_input)
        ref_fp16_out = torch.nn.functional.linear(
            fp16_input,
            q_weight.to(dtype=torch.float16),
            bias.to(dtype=torch.float16),
        )
        self.assertTrue(torch.equal(loaded_fp16_out, ref_fp16_out))

        saved = model.state_dict()
        saved_conf = json.loads(saved["layer.comfy_quant"].numpy().tobytes())
        self.assertTrue(saved_conf["convrot"])
        self.assertEqual(saved_conf["convrot_groupsize"], 256)

if __name__ == "__main__":
    unittest.main()
