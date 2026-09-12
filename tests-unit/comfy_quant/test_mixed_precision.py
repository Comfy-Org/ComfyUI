import unittest
from unittest import mock
import torch
import sys
import os
import json
from types import SimpleNamespace

# Add comfy to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

def has_gpu():
    return torch.cuda.is_available()

from comfy.cli_args import args
if not has_gpu():
    args.cpu = True

import comfy.model_management as model_management
from comfy import hooks, ops
from comfy.model_patcher import ModelPatcher, ModelPatcherDynamic
from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor, TensorCoreFP8E4M3Layout
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

    def test_int8_disabled_on_unsupported_device_falls_back_to_full_precision(self):
        """On a device that can't run comfy_kitchen's fast int8 matmul (e.g. MPS,
        which lacks aten::_int_mm), pick_operations must mark int8 formats as
        disabled so layers dequantize instead of taking the fast quantized path."""
        import comfy.model_management as mm

        orig_supports_int8 = mm.supports_int8_compute
        mm.supports_int8_compute = lambda device=None: False
        try:
            model_config = SimpleNamespace(quant_config={"layer": {"format": "int8_tensorwise"}})
            operations = ops.pick_operations(torch.bfloat16, torch.bfloat16, model_config=model_config)

            torch.manual_seed(789)
            weight = torch.randn(16, 256, dtype=torch.bfloat16)
            bias = torch.randn(16, dtype=torch.bfloat16)
            q_weight = QuantizedTensor.from_float(weight, "TensorWiseINT8Layout", per_channel=True)
            state_dict = {
                "layer.weight": q_weight._qdata,
                "layer.bias": bias,
                "layer.weight_scale": q_weight._params.scale,
            }
            layer_quant_config = {"layer": {"format": "int8_tensorwise"}}
            state_dict, _ = comfy.utils.convert_old_quants(
                state_dict,
                metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})},
            )

            model = torch.nn.Module()
            model.layer = operations.Linear(256, 16, device="cpu", dtype=torch.bfloat16)
            model.load_state_dict(state_dict, strict=False)

            self.assertIsInstance(model.layer.weight, QuantizedTensor)
            # The layer must be forced onto the full-precision (dequantized)
            # path since the fast int8 path isn't usable on this device.
            self.assertTrue(model.layer._full_precision_mm)

            # The weight's orig_dtype matches the compute dtype here (both bfloat16),
            # so cast_bias_weight's dtype-change check alone won't dequantize it. Confirm
            # the module still hands a real Tensor (not a QuantizedTensor) to the plain
            # linear() call, since dispatching a QuantizedTensor there would route back
            # into the disabled fast int8 matmul instead of the full-precision fallback.
            seen_weight_types = []
            orig_module_forward = model.layer._forward
            def _capturing_forward(input, weight, bias, _orig=orig_module_forward):
                seen_weight_types.append(type(weight))
                return _orig(input, weight, bias)
            model.layer._forward = _capturing_forward

            input_tensor = torch.randn(4, 256, dtype=torch.bfloat16)
            output = model.layer(input_tensor)
            self.assertEqual(output.shape, (4, 16))
            self.assertEqual(seen_weight_types, [torch.Tensor])
        finally:
            mm.supports_int8_compute = orig_supports_int8

    def test_supports_int8_compute_treats_mps_mode_as_unsupported_when_device_is_none(self):
        """Call sites (like pick_operations' default) may omit load_device. On an
        MPS machine that must still report int8 as unsupported instead of
        silently defaulting to True, matching supports_fp64's handling of the
        same device=None case (see Comfy-Org/ComfyUI#16136)."""
        import comfy.model_management as mm

        orig_cpu_state = mm.cpu_state
        mm.cpu_state = mm.CPUState.MPS
        try:
            self.assertFalse(mm.supports_int8_compute(None))
        finally:
            mm.cpu_state = orig_cpu_state

    def test_convrot_w4a4_loads_into_params(self):
        """ConvRot W4A4 checkpoints must load as the dedicated kitchen layout."""
        if "convrot_w4a4" not in QUANT_ALGOS:
            self.skipTest("comfy_kitchen does not provide ConvRot W4A4")

        torch.manual_seed(456)
        layer_quant_config = {
            "layer": {
                "format": "convrot_w4a4",
                "convrot_groupsize": 256,
                "linear_dtype": "int8",
            }
        }
        weight = torch.randn(16, 256, dtype=torch.bfloat16)
        bias = torch.randn(16, dtype=torch.bfloat16)
        q_weight = QuantizedTensor.from_float(
            weight,
            "TensorCoreConvRotW4A4Layout",
            convrot_groupsize=256,
            quant_group_size=64,
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
        self.assertEqual(model.layer.weight._layout_cls, "TensorCoreConvRotW4A4Layout")
        self.assertEqual(model.layer.weight._params.convrot_groupsize, 256)
        self.assertEqual(model.layer.weight._params.quant_group_size, 64)
        self.assertEqual(model.layer.weight._params.linear_dtype, "int8")

        input_tensor = torch.randn(4, 256, dtype=torch.bfloat16)
        loaded_out = model.layer(input_tensor)
        ref_out = torch.nn.functional.linear(input_tensor, q_weight, bias)
        self.assertTrue(torch.equal(loaded_out, ref_out))

        saved = model.state_dict()
        saved_conf = json.loads(saved["layer.comfy_quant"].numpy().tobytes())
        self.assertEqual(saved_conf["format"], "convrot_w4a4")
        self.assertEqual(saved_conf["convrot_groupsize"], 256)
        self.assertEqual(saved_conf["linear_dtype"], "int8")
        self.assertNotIn("quant_group_size", saved_conf)

    def test_hook_patches_skip_only_quantized_weight_pieces(self):
        operations = ops.mixed_precision_ops(compute_dtype=torch.float32)
        model = torch.nn.Module()
        model.linear = operations.Linear(4, 4, bias=False, device="cpu")
        qdata, params = TensorCoreFP8E4M3Layout.quantize(
            torch.ones(4, 4), scale="recalculate"
        )
        model.linear.quant_format = "float8_e4m3fn"
        model.linear.layout_type = "TensorCoreFP8E4M3Layout"
        model.linear.weight = torch.nn.Parameter(
            QuantizedTensor(qdata, model.linear.layout_type, params),
            requires_grad=False,
        )
        model.linear.input_scale = torch.nn.Parameter(
            torch.tensor(0.125), requires_grad=False
        )
        model.linear_alias = model.linear
        model.patch_target = torch.nn.Linear(4, 4, bias=False)
        torch.nn.init.zeros_(model.patch_target.weight)

        patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
        weight = model.linear.weight
        input_scale = model.linear.input_scale
        hook = hooks.WeightHook()
        hook.need_weight_init = False
        hook.weights = {
            "patch_target.weight": (torch.ones_like(model.patch_target.weight),)
        }
        hook_group = hooks.HookGroup()
        hook_group.add(hook)
        patcher.register_all_hook_patches(
            hook_group, hooks.create_target_dict(hooks.EnumWeightTarget.Model)
        )
        patcher.patch_hooks(hook_group)

        self.assertIs(model.linear.weight, weight)
        self.assertIs(model.linear.input_scale, input_scale)
        self.assertTrue(
            torch.equal(
                model.patch_target.weight,
                torch.ones_like(model.patch_target.weight),
            )
        )

        self.assertEqual(
            set(patcher.get_key_patches()),
            {
                "linear.weight",
                "linear.input_scale",
                "linear_alias.weight",
                "linear_alias.input_scale",
                "patch_target.weight",
            },
        )

    def test_hook_targets_disable_async_offload(self):
        model = torch.nn.Module()
        model.linear = ops.disable_weight_init.Linear(
            4, 4, bias=False, device="cpu", dtype=torch.bfloat16
        )
        model.other = ops.disable_weight_init.Linear(
            4, 4, bias=False, device="cpu", dtype=torch.bfloat16
        )
        model.linear.weight = torch.nn.Parameter(
            model.linear.weight.float(), requires_grad=False
        )
        model.other.weight = torch.nn.Parameter(
            model.other.weight.float(), requires_grad=False
        )
        model.linear.weight_comfy_model_dtype = torch.bfloat16
        model.other.weight_comfy_model_dtype = torch.bfloat16
        model.control = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.zeros_(model.linear.weight)
        patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))

        hook = hooks.WeightHook()
        hook.need_weight_init = False
        hook.weights = {
            "linear.weight": (
                torch.zeros_like(model.linear.weight),
            )
        }
        group = hooks.HookGroup()
        group.add(hook)
        patcher.register_all_hook_patches(
            group, hooks.create_target_dict(hooks.EnumWeightTarget.Model)
        )

        patcher.patch_hooks(group)
        self.assertTrue(model.linear.comfy_disable_async_offload)
        self.assertTrue(model.other.comfy_disable_async_offload)
        self.assertTrue(model.linear.comfy_cast_weights)
        self.assertTrue(model.other.comfy_cast_weights)
        output = model.linear(torch.zeros(1, 4, dtype=torch.bfloat16))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(output).all())
        patcher.patch_hooks(None)

        model.linear.comfy_disable_async_offload = False
        model.other.comfy_disable_async_offload = False
        patcher._hook_async_offload_disabled = False
        patcher.patch_hooks(group)
        self.assertTrue(model.linear.comfy_disable_async_offload)
        self.assertTrue(model.other.comfy_disable_async_offload)
        patcher.patch_hooks(None)

    def test_cast_weight_honors_async_offload_disable(self):
        linear = ops.disable_weight_init.Linear(
            4, 4, bias=False, device="cpu", dtype=torch.float32
        )
        target = torch.device("cpu", 1)

        linear.comfy_disable_async_offload = True
        with mock.patch(
            "comfy.model_management.get_offload_stream", return_value=None
        ) as get_offload_stream:
            ops.cast_bias_weight(
                linear,
                dtype=torch.float32,
                device=target,
                offloadable=True,
            )
        get_offload_stream.assert_not_called()

        linear.comfy_disable_async_offload = False
        with mock.patch(
            "comfy.model_management.get_offload_stream", return_value=None
        ) as get_offload_stream:
            ops.cast_bias_weight(
                linear,
                dtype=torch.float32,
                device=target,
                offloadable=True,
            )
        get_offload_stream.assert_called_once_with(target)

    def test_dynamic_delegate_is_released_only_after_model_unload(self):
        patcher = object.__new__(ModelPatcherDynamic)
        patcher.model = torch.nn.Module()
        delegate = object()
        patcher.non_dynamic_delegate_model = delegate
        patcher.partially_unload_ram = mock.Mock()
        patcher.partially_unload = mock.Mock()

        with mock.patch.object(ModelPatcher, "unpatch_model") as base_unpatch:
            patcher.unpatch_model(unpatch_weights=False)
            base_unpatch.assert_called_once_with(
                device_to=None, unpatch_weights=False
            )
        self.assertIs(patcher.non_dynamic_delegate_model, delegate)
        patcher.partially_unload_ram.assert_not_called()
        patcher.partially_unload.assert_not_called()

        patcher.partially_unload_ram.reset_mock()
        patcher.partially_unload.reset_mock()
        with mock.patch.object(ModelPatcher, "unpatch_model") as base_unpatch:
            patcher.unpatch_model(
                device_to=torch.device("cpu"), unpatch_weights=True
            )
            base_unpatch.assert_called_once_with(
                device_to=None, unpatch_weights=False
            )
        self.assertIs(patcher.non_dynamic_delegate_model, delegate)
        patcher.partially_unload_ram.assert_called_once_with(1e32)
        patcher.partially_unload.assert_called_once_with(None, 1e32)

        patcher.finalize_model_unload()
        self.assertIsNone(patcher.non_dynamic_delegate_model)

    def test_dynamic_delegate_is_recreated_after_model_unload(self):
        patcher = object.__new__(ModelPatcherDynamic)
        initial_override = object()
        first_override = object()
        second_override = object()
        first_delegate = mock.Mock()
        second_delegate = mock.Mock()
        first_delegate.get_clone_model_override.return_value = first_override
        second_delegate.get_clone_model_override.return_value = second_override
        patcher.non_dynamic_delegate_model = initial_override
        patcher.clone = mock.Mock(
            side_effect=[first_delegate, second_delegate]
        )

        self.assertIs(patcher.get_non_dynamic_delegate(), first_delegate)
        patcher.clone.assert_called_with(
            disable_dynamic=True, model_override=initial_override
        )
        self.assertIs(patcher.non_dynamic_delegate_model, first_override)

        patcher.finalize_model_unload()
        self.assertIsNone(patcher.non_dynamic_delegate_model)

        self.assertIs(patcher.get_non_dynamic_delegate(), second_delegate)
        patcher.clone.assert_called_with(
            disable_dynamic=True, model_override=None
        )
        self.assertIs(patcher.non_dynamic_delegate_model, second_override)

    def test_free_memory_finalizes_only_removed_models(self):
        device = torch.device("cpu")
        kept_patcher = mock.Mock()
        kept_patcher.model = torch.nn.Module()
        kept = mock.Mock()
        kept.device = device
        kept.model = kept_patcher
        kept.is_dead.return_value = False

        unloaded_patcher = mock.Mock()
        unloaded_patcher.model = torch.nn.Module()
        unload = mock.Mock()
        unload.device = device
        unload.model = unloaded_patcher
        unload.is_dead.return_value = False
        unload.model_offloaded_memory.return_value = 0
        unload.model_memory.return_value = 1
        unload.model_unload.return_value = True

        def assert_only_kept_before_finalize():
            self.assertEqual(
                model_management.current_loaded_models, [kept]
            )

        unloaded_patcher.finalize_model_unload.side_effect = (
            assert_only_kept_before_finalize
        )
        with (
            mock.patch.object(
                model_management, "current_loaded_models", [kept, unload]
            ),
            mock.patch.object(model_management, "cleanup_models_gc"),
            mock.patch.object(
                model_management, "get_free_memory", return_value=0
            ),
            mock.patch.object(model_management, "soft_empty_cache"),
        ):
            unloaded = model_management.free_memory(
                1, device, keep_loaded=[kept]
            )
            self.assertEqual(unloaded, [unload])
            self.assertEqual(
                model_management.current_loaded_models, [kept]
            )

        kept.model_unload.assert_not_called()
        kept_patcher.finalize_model_unload.assert_not_called()
        unloaded_patcher.finalize_model_unload.assert_called_once_with()

if __name__ == "__main__":
    unittest.main()
